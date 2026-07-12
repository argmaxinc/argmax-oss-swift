//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
import Foundation

// MARK: - KV Cache

/// KV cache for autoregressive decoder models.
///
/// Manages external cache arrays (`keyCache`/`valueCache`) for non-stateful models,
/// or position tracking and attention masks only for stateful models whose weights
/// are managed internally by CoreML via `MLState`.
///
/// Components that consume more than one new position per call (e.g. the
/// throughput-optimized SpeechDecoder, which processes 4 RVQ frames at once)
/// pass `codesPerStep > 1`. The `kvCacheUpdateMask` shape and update logic
/// adapt accordingly.
///
/// Thread safety: each `Qwen3GenerateTask` creates its own cache instance.
/// Caches are never shared across concurrent tasks.
public class KVCache: @unchecked Sendable {
    public var cacheLength: Int32 = 0
    public let maxSeqLength: Int
    public let cacheDim: Int
    public let isStateful: Bool
    /// Number of new positions written per `update()` call.
    public let codesPerStep: Int
    /// When `true`, `kvCacheUpdateMask` is `[1, codesPerStep, maxSeqLength]` (rank 3, one row
    /// per new position). When `false`, `[1, maxSeqLength]` (rank 2 — single-position path).
    public let useRank3UpdateMask: Bool

    /// External key cache -- nil for stateful models
    public let keyCache: MLMultiArray?
    /// External value cache -- nil for stateful models
    public let valueCache: MLMultiArray?
    public let kvCacheUpdateMask: MLMultiArray
    public let keyPaddingMask: MLMultiArray

    /// Additive attention-mask bias applied to masked-out positions (padded key
    /// slots and future positions in the intra-step causal triangle). A large
    /// negative value so that, after softmax, attention to those positions is ~0.
    static let maskedAttentionBias = FloatType(-10000)

    public init(
        cacheDim: Int,
        maxSeqLength: Int,
        codesPerStep: Int = 1,
        useRank3UpdateMask: Bool = false,
        isStateful: Bool = false
    ) throws {
        // Validate the configuration and fail gracefully (the init already throws) so a
        // misconfigured cache surfaces an error to the caller instead of crashing the app.
        guard codesPerStep >= 1 else {
            throw TTSError.invalidConfiguration("KVCache: codesPerStep must be >= 1, got \(codesPerStep)")
        }
        // Multi-position updates require the rank-3 update mask: the rank-2 mask is
        // `[1, maxSeqLength]` and can only encode a single active update column, so it
        // cannot describe `codesPerStep > 1` new positions per call.
        guard codesPerStep == 1 || useRank3UpdateMask else {
            throw TTSError.invalidConfiguration(
                "KVCache: codesPerStep > 1 (\(codesPerStep)) requires useRank3UpdateMask " +
                "(the rank-2 update mask only encodes one position)"
            )
        }
        self.cacheDim = cacheDim
        self.maxSeqLength = maxSeqLength
        self.codesPerStep = codesPerStep
        self.useRank3UpdateMask = useRank3UpdateMask
        self.isStateful = isStateful

        if isStateful {
            // Stateful models manage KV cache internally via MLState
            keyCache = nil
            valueCache = nil
        } else {
            keyCache = try MLMultiArray(
                shape: [1, NSNumber(value: cacheDim), 1, NSNumber(value: maxSeqLength)],
                dataType: .float16
            )
            valueCache = try MLMultiArray(
                shape: [1, NSNumber(value: cacheDim), 1, NSNumber(value: maxSeqLength)],
                dataType: .float16
            )
        }

        let updateShape: [NSNumber] = useRank3UpdateMask
            ? [1, NSNumber(value: codesPerStep), NSNumber(value: maxSeqLength)]
            : [1, NSNumber(value: maxSeqLength)]
        kvCacheUpdateMask = try MLMultiArray(shape: updateShape, dataType: .float16)
        keyPaddingMask = try MLMultiArray(
            shape: [1, NSNumber(value: maxSeqLength)],
            dataType: .float16
        )

        reset()
    }

    public func reset() {
        cacheLength = 0
        let seqLen = maxSeqLength

        // Zero-fill external KV caches (stateful models don't have them)
        if let keyCache, let valueCache {
            memset(keyCache.dataPointer, 0, cacheDim * seqLen * MemoryLayout<FloatType>.size)
            memset(valueCache.dataPointer, 0, cacheDim * seqLen * MemoryLayout<FloatType>.size)
        }

        // Update mask: zero everything then set initial active entries.
        memset(kvCacheUpdateMask.dataPointer, 0, kvCacheUpdateMask.count * MemoryLayout<FloatType>.size)
        let updatePtr = kvCacheUpdateMask.dataPointer.bindMemory(
            to: FloatType.self, capacity: kvCacheUpdateMask.count
        )
        if useRank3UpdateMask {
            // [1, codesPerStep, maxSeqLength]: mask[0, i, i] = 1 for i in 0..<codesPerStep
            for i in 0..<codesPerStep where i < seqLen {
                updatePtr[i * seqLen + i] = FloatType(1)
            }
        } else {
            // [1, maxSeqLength]: mask[0, 0] = 1
            updatePtr[0] = FloatType(1)
        }

        // Padding mask: unmasked region is the first codesPerStep positions at startup.
        let paddingPtr = keyPaddingMask.dataPointer.bindMemory(to: FloatType.self, capacity: seqLen)
        for j in 0..<seqLen {
            paddingPtr[j] = (j < codesPerStep) ? FloatType(0) : Self.maskedAttentionBias
        }
    }

    /// Write cache updates at the current position and advance by `codesPerStep`.
    /// For stateful models, only advances position and updates masks (KV cache is internal).
    public func update(keyCacheUpdates: MLMultiArray? = nil, valueCacheUpdates: MLMultiArray? = nil) {
        let writePos = Int(cacheLength)
        let seqLen = maxSeqLength

        // Defensive bounds check. The per-element writes below clamp at `seqLen`, so a
        // multi-position update that runs past the end of the cache would silently drop
        // the overflow. In normal operation generation halts on `isFull` before this can
        // trigger; surface it as a warning rather than dropping K/V (and mask) values
        // without any signal.
        if writePos + codesPerStep > seqLen {
            Logging.error(
                "KVCache.update: write at position \(writePos) + codesPerStep \(codesPerStep) exceeds " +
                "capacity \(seqLen); \(writePos + codesPerStep - seqLen) trailing position(s) dropped."
            )
        }

        // Scatter `codesPerStep` new K/V positions into external KV cache (skip for stateful).
        if !isStateful, let keyCache, let valueCache, let keyCacheUpdates, let valueCacheUpdates {
            let embedDim = cacheDim
            let keyCachePtr = keyCache.dataPointer.bindMemory(to: FloatType.self, capacity: embedDim * seqLen)
            let valueCachePtr = valueCache.dataPointer.bindMemory(to: FloatType.self, capacity: embedDim * seqLen)

            let keyUpdatePtr = keyCacheUpdates.dataPointer.bindMemory(to: FloatType.self, capacity: keyCacheUpdates.count)
            let valueUpdatePtr = valueCacheUpdates.dataPointer.bindMemory(to: FloatType.self, capacity: valueCacheUpdates.count)
            // Strides along the embed-dim axis (1) and frame axis (3) of the
            // `[1, embedDim, 1, codesPerStep]` updates tensor.
            let keyEmbedStride = keyCacheUpdates.strides[1].intValue
            let valueEmbedStride = valueCacheUpdates.strides[1].intValue
            let keyFrameStride = keyCacheUpdates.strides[3].intValue
            let valueFrameStride = valueCacheUpdates.strides[3].intValue

            for dim in 0..<embedDim {
                let cacheBase = dim * seqLen
                let keyBase = dim * keyEmbedStride
                let valueBase = dim * valueEmbedStride
                for i in 0..<codesPerStep {
                    let dst = writePos + i
                    if dst < seqLen {
                        keyCachePtr[cacheBase + dst] = keyUpdatePtr[keyBase + i * keyFrameStride]
                        valueCachePtr[cacheBase + dst] = valueUpdatePtr[valueBase + i * valueFrameStride]
                    }
                }
            }
        }

        // Incrementally update masks for the *next* iteration.
        let oldCacheLen = writePos
        let newCacheLen = writePos + codesPerStep
        let updatePtr = kvCacheUpdateMask.dataPointer.bindMemory(
            to: FloatType.self, capacity: kvCacheUpdateMask.count
        )
        if useRank3UpdateMask {
            // For each row i: clear (oldCacheLen + i), set (newCacheLen + i).
            for i in 0..<codesPerStep {
                let oldCol = oldCacheLen + i
                let newCol = newCacheLen + i
                if oldCol < seqLen { updatePtr[i * seqLen + oldCol] = FloatType(0) }
                if newCol < seqLen { updatePtr[i * seqLen + newCol] = FloatType(1) }
            }
        } else {
            // [1, maxSeqLength]: clear writePos, set writePos+codesPerStep.
            if oldCacheLen < seqLen { updatePtr[oldCacheLen] = FloatType(0) }
            if newCacheLen < seqLen { updatePtr[newCacheLen] = FloatType(1) }
        }

        // Extend the unmasked region by `codesPerStep` (positions
        // [newCacheLen, newCacheLen + codesPerStep)).
        let paddingPtr = keyPaddingMask.dataPointer.bindMemory(to: FloatType.self, capacity: seqLen)
        for j in newCacheLen..<min(newCacheLen + codesPerStep, seqLen) {
            paddingPtr[j] = FloatType(0)
        }

        cacheLength = Int32(newCacheLen)
    }

    /// `true` once there is no longer room for a full `codesPerStep`-wide write.
    /// Reserves `codesPerStep` positions (one write) so `update()` never overruns;
    /// for `codesPerStep == 1` this is the original `maxSeqLength - 1` threshold.
    public var isFull: Bool { Int(cacheLength) >= maxSeqLength - codesPerStep }

    /// How many free positions remain before the cache is full.
    public var freePositions: Int { maxSeqLength - codesPerStep - Int(cacheLength) }

    public func makeCacheLengthArray() throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1], dataType: .int32)
        arr[0] = NSNumber(value: cacheLength)
        return arr
    }
}

// MARK: - MLTensor Access

@available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
public extension KVCache {
    /// Cache position as a `[1]` Int32 tensor.
    var cacheLengthTensor: MLTensor { MLTensor(shape: [1], scalars: [Int32(cacheLength)]) }
    /// Update-mask tensor. Shape is `[1, maxSeqLength]` (rank-2 path) or
    /// `[1, codesPerStep, maxSeqLength]` (rank-3 path) — driven by the cache's mode.
    var kvCacheUpdateMaskTensor: MLTensor { MLTensor(MLShapedArray<FloatType>(kvCacheUpdateMask)) }
    /// Padding-mask as a `[1, maxSeqLength]` Float16 tensor.
    var keyPaddingMaskTensor: MLTensor { MLTensor(MLShapedArray<FloatType>(keyPaddingMask)) }
    /// External key-cache tensor - `nil` for stateful models.
    var keyCacheTensor: MLTensor? { keyCache.map { MLTensor(MLShapedArray<FloatType>($0)) } }
    /// External value-cache tensor - `nil` for stateful models.
    var valueCacheTensor: MLTensor? { valueCache.map { MLTensor(MLShapedArray<FloatType>($0)) } }

    /// Async update from MLTensor outputs - materializes without blocking the cooperative pool.
    func update(keyTensor: MLTensor, valueTensor: MLTensor) async {
        async let keyArr = keyTensor.toMLMultiArray()
        async let valArr = valueTensor.toMLMultiArray()
        let (keyMaterialized, valueMaterialized) = await (keyArr, valArr)
        update(keyCacheUpdates: keyMaterialized, valueCacheUpdates: valueMaterialized)
    }
}

// MARK: - Speech Decoder Cache

/// Extended KV cache for the SpeechDecoder.
///
/// Adds:
/// - A rolling hidden-context buffer `[1, hiddenDim, 1, hiddenContextLen]` whose
///   length is read from the loaded model (4 for the latency function, 1 for
///   throughput).
/// - An optional `qkMask` of shape `[1, codesPerStep, maxSeqLength]` that encodes
///   the intra-step causal triangle; only allocated when `codesPerStep > 1` (the
///   throughput function). The latency function does not consume `qk_mask`.
///
/// Always uses the rank-3 `kvCacheUpdateMask` shape since the new multifunction
/// SpeechDecoder asset requires it for both functions (`[1, 1, maxSeqLength]` for
/// the latency function, `[1, 4, maxSeqLength]` for throughput).
public class SpeechDecoderCache: KVCache, @unchecked Sendable {
    public let hiddenContext: MLMultiArray // [1, hiddenDim, 1, contextLen]
    public let hiddenDim: Int
    public let hiddenContextLen: Int
    /// Intra-step causal mask `[1, codesPerStep, maxSeqLength]`. `nil` when
    /// `codesPerStep == 1` (latency function does not consume a `qk_mask` input).
    public let qkMask: MLMultiArray?

    public init(
        cacheDim: Int = Qwen3TTSConstants.sdCacheDim,
        maxSeqLength: Int = Qwen3TTSConstants.sdMaxSeq,
        hiddenDim: Int = Qwen3TTSConstants.sdHiddenDim,
        hiddenContextLen: Int = Qwen3TTSConstants.sdHiddenContextLen,
        codesPerStep: Int = 1
    ) throws {
        self.hiddenDim = hiddenDim
        self.hiddenContextLen = hiddenContextLen
        hiddenContext = try MLMultiArray(
            shape: [1, NSNumber(value: hiddenDim), 1, NSNumber(value: hiddenContextLen)],
            dataType: .float16
        )
        memset(hiddenContext.dataPointer, 0, hiddenDim * hiddenContextLen * MemoryLayout<FloatType>.size)

        if codesPerStep > 1 {
            qkMask = try MLMultiArray(
                shape: [1, NSNumber(value: codesPerStep), NSNumber(value: maxSeqLength)],
                dataType: .float16
            )
        } else {
            qkMask = nil
        }

        // Speech decoder is non-stateful; always uses the rank-3 update mask
        // (latency = [1, 1, maxSeqLength], throughput = [1, codesPerStep, maxSeqLength]).
        try super.init(
            cacheDim: cacheDim,
            maxSeqLength: maxSeqLength,
            codesPerStep: codesPerStep,
            useRank3UpdateMask: true,
            isStateful: false
        )

    }

    /// Reset all state (KV cache, masks, hidden context buffer, qk mask).
    /// Swift's two-phase initialization guarantees the subclass's stored properties
    /// are set before `super.init` runs `reset()`, so it is safe to touch them here.
    public override func reset() {
        super.reset()
        memset(hiddenContext.dataPointer, 0, hiddenDim * hiddenContextLen * MemoryLayout<FloatType>.size)
        resetQkMask()
    }

    /// Zero the qk mask, then write the initial intra-step causal triangle at
    /// `cacheLength = 0` (one row per code in the current step).
    private func resetQkMask() {
        guard let qkMask else { return }
        let seqLen = maxSeqLength
        let totalCount = qkMask.count
        memset(qkMask.dataPointer, 0, totalCount * MemoryLayout<FloatType>.size)
        let ptr = qkMask.dataPointer.bindMemory(to: FloatType.self, capacity: totalCount)
        // qkMask[0, i, j] = -1e4 for i < j < codesPerStep
        for i in 0..<codesPerStep {
            for j in (i + 1)..<min(codesPerStep, seqLen) {
                ptr[i * seqLen + j] = FloatType(-10000)
            }
        }
    }

    /// Advance the cache and incrementally update the qk mask in addition to the base
    /// kv update / padding masks.
    public override func update(keyCacheUpdates: MLMultiArray? = nil, valueCacheUpdates: MLMultiArray? = nil) {
        let writePos = Int(cacheLength)
        let seqLen = maxSeqLength
        super.update(keyCacheUpdates: keyCacheUpdates, valueCacheUpdates: valueCacheUpdates)

        guard let qkMask else { return }
        let ptr = qkMask.dataPointer.bindMemory(to: FloatType.self, capacity: qkMask.count)
        let oldCacheLen = writePos
        let newCacheLen = writePos + codesPerStep
        for i in 0..<codesPerStep {
            // Clear old triangle for row i: columns
            //     (oldCacheLen + i + 1 ..< oldCacheLen + codesPerStep)
            let oldStart = oldCacheLen + i + 1
            let oldEnd = min(oldCacheLen + codesPerStep, seqLen)
            if oldStart < oldEnd {
                for j in oldStart..<oldEnd {
                    ptr[i * seqLen + j] = FloatType(0)
                }
            }
            // Set new triangle for row i: columns
            //     (newCacheLen + i + 1 ..< newCacheLen + codesPerStep)
            let newStart = newCacheLen + i + 1
            let newEnd = min(newCacheLen + codesPerStep, seqLen)
            if newStart < newEnd {
                for j in newStart..<newEnd {
                    ptr[i * seqLen + j] = Self.maskedAttentionBias
                }
            }
        }
    }

    /// Pull cache updates + hidden context update from a CoreML output feature provider,
    /// scatter the new KV slots, and roll the hidden context buffer left by `codesPerStep`.
    public func updateWithHiddenContext(output: MLFeatureProvider) {
        guard let keyCU = output.featureValue(for: "key_cache_updates")?.multiArrayValue,
            let valCU = output.featureValue(for: "value_cache_updates")?.multiArrayValue
        else {
            return
        }
        update(keyCacheUpdates: keyCU, valueCacheUpdates: valCU)

        guard let updateArr = output.featureValue(for: "hidden_context_update")?.multiArrayValue else { return }
        rollHiddenContext(from: updateArr)
    }

    /// Roll the rolling hidden-context buffer:
    /// `combined = concat(currentHC, newHC)` (length `hiddenContextLen + codesPerStep`);
    /// keep the last `hiddenContextLen` columns.
    ///
    /// - When `codesPerStep >= hiddenContextLen`: result is the trailing
    ///   `hiddenContextLen` columns of newHC (full overwrite).
    /// - When `codesPerStep <  hiddenContextLen`: shift current left by
    ///   `codesPerStep`, then append the new `codesPerStep` columns.
    private func rollHiddenContext(from updateArr: MLMultiArray) {
        let hcPtr = hiddenContext.dataPointer.bindMemory(
            to: FloatType.self, capacity: hiddenDim * hiddenContextLen
        )
        let updatePtr = updateArr.dataPointer.bindMemory(to: FloatType.self, capacity: updateArr.count)
        // updateArr shape: [1, hiddenDim, 1, codesPerStep] — strides along the
        // hidden-dim axis (1) and the frame axis (3).
        let updateHiddenStride = updateArr.strides[1].intValue
        let updateFrameStride = updateArr.strides[3].intValue

        if codesPerStep >= hiddenContextLen {
            // Result is the last `hiddenContextLen` columns of newHC.
            for dim in 0..<hiddenDim {
                let dst = dim * hiddenContextLen
                let src = dim * updateHiddenStride
                let offset = codesPerStep - hiddenContextLen
                for i in 0..<hiddenContextLen {
                    hcPtr[dst + i] = updatePtr[src + (offset + i) * updateFrameStride]
                }
            }
        } else {
            // Shift left by `codesPerStep`, then append `codesPerStep` new columns
            // at the end.
            for dim in 0..<hiddenDim {
                let dst = dim * hiddenContextLen
                let src = dim * updateHiddenStride
                // Shift: current[codesPerStep..<hiddenContextLen]
                //     -> dst[0..<hiddenContextLen - codesPerStep]
                for t in 0..<(hiddenContextLen - codesPerStep) {
                    hcPtr[dst + t] = hcPtr[dst + t + codesPerStep]
                }
                // Append the new `codesPerStep` columns
                for i in 0..<codesPerStep {
                    hcPtr[dst + (hiddenContextLen - codesPerStep + i)] = updatePtr[src + i * updateFrameStride]
                }
            }
        }
    }
}

// MARK: - Speech Decoder Cache MLTensor Access

@available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
public extension SpeechDecoderCache {
    var hiddenContextTensor: MLTensor { MLTensor(MLShapedArray<FloatType>(hiddenContext)) }
    var qkMaskTensor: MLTensor? { qkMask.map { MLTensor(MLShapedArray<FloatType>($0)) } }

    /// Update KV cache, qk mask, and rolling hidden context from `[String: MLTensor]`
    /// prediction outputs. Materializes tensors asynchronously to avoid blocking the
    /// cooperative thread pool.
    func updateWithHiddenContext(tensorOutputs: [String: MLTensor]) async {
        guard let keyUpdateTensor = tensorOutputs["key_cache_updates"],
            let valueUpdateTensor = tensorOutputs["value_cache_updates"]
        else {
            return
        }
        // Key and value outputs are independent. Materialize them concurrently
        // to avoid serializing two blocking conversions on the cooperative pool.
        async let keyArr = keyUpdateTensor.toMLMultiArray()
        async let valArr = valueUpdateTensor.toMLMultiArray()
        let (keyMaterialized, valueMaterialized) = await (keyArr, valArr)
        update(keyCacheUpdates: keyMaterialized, valueCacheUpdates: valueMaterialized)

        guard let hiddenUpdateTensor = tensorOutputs["hidden_context_update"] else { return }
        let updateArr = await hiddenUpdateTensor.toMLMultiArray()
        rollHiddenContext(from: updateArr)
    }
}

// MARK: - Stateful Model Cache Update

@available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
public extension KVCache {
    /// Write key/value cache updates into an MLState's internal buffers.
    ///
    /// Stateful CoreML models read their KV cache from MLState via `readState` ops,
    /// but do not write updates back automatically. The host must manually copy the
    /// model's output cache updates into the state at the correct position.
    ///
    /// Async variant that materializes `MLTensor` outputs before writing to `MLState`.
    ///
    /// - Parameters:
    ///   - state: The `MLState` object associated with the model.
    ///   - keyTensor: Key cache update tensor output from the model, shape [1, cacheDim, 1, 1].
    ///   - valueTensor: Value cache update tensor output from the model, shape [1, cacheDim, 1, 1].
    ///   - position: The cache position to write at (current `cacheLength` before increment).
    static func updateStateCache(
        state: MLState,
        keyTensor: MLTensor,
        valueTensor: MLTensor,
        position: Int
    ) async {
        // Key and value outputs are independent. Materialize them concurrently;
        // the state writes below remain ordered and use the same position.
        async let keyArr = keyTensor.toMLMultiArray()
        async let valArr = valueTensor.toMLMultiArray()
        let (keyMaterialized, valueMaterialized) = await (keyArr, valArr)
        updateStateCache(
            state: state,
            keyCacheUpdates: keyMaterialized,
            valueCacheUpdates: valueMaterialized,
            position: position
        )
    }

    static func updateStateCache(
        state: MLState,
        keyCacheUpdates: MLMultiArray,
        valueCacheUpdates: MLMultiArray,
        position: Int
    ) {
        let bytesPerSample = MemoryLayout<FloatType>.size

        let keyUpdatePtr = keyCacheUpdates.dataPointer.bindMemory(to: FloatType.self, capacity: keyCacheUpdates.count)
        let keyUpdateStride = keyCacheUpdates.strides[1].intValue
        let valueUpdatePtr = valueCacheUpdates.dataPointer.bindMemory(to: FloatType.self, capacity: valueCacheUpdates.count)
        let valueUpdateStride = valueCacheUpdates.strides[1].intValue

        state.withMultiArray(for: "self_attn_key_cache") { keyStateCache in
            let embedDim = keyStateCache.shape[1].intValue
            keyStateCache.withUnsafeMutableBytes { cachePtr, cacheStrides in
                guard let baseAddress = cachePtr.baseAddress else { return }
                for dim in 0..<embedDim {
                    let cacheByteOffset = (dim * cacheStrides[1] + position * cacheStrides[3]) * bytesPerSample
                    let dst = (baseAddress + cacheByteOffset).assumingMemoryBound(to: FloatType.self)
                    dst.pointee = keyUpdatePtr[dim * keyUpdateStride]
                }
            }
        }

        state.withMultiArray(for: "self_attn_value_cache") { valueStateCache in
            let embedDim = valueStateCache.shape[1].intValue
            valueStateCache.withUnsafeMutableBytes { cachePtr, cacheStrides in
                guard let baseAddress = cachePtr.baseAddress else { return }
                for dim in 0..<embedDim {
                    let cacheByteOffset = (dim * cacheStrides[1] + position * cacheStrides[3]) * bytesPerSample
                    let dst = (baseAddress + cacheByteOffset).assumingMemoryBound(to: FloatType.self)
                    dst.pointee = valueUpdatePtr[dim * valueUpdateStride]
                }
            }
        }
    }
}
