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
/// Thread safety: each `Qwen3GenerateTask` creates its own cache instance.
/// Caches are never shared across concurrent tasks.
public class KVCache: @unchecked Sendable {
    public var cacheLength: Int32 = 0
    public let maxSeqLength: Int
    public let cacheDim: Int
    public let isStateful: Bool

    /// External key cache -- nil for stateful models
    public let keyCache: MLMultiArray?
    /// External value cache -- nil for stateful models
    public let valueCache: MLMultiArray?
    public let kvCacheUpdateMask: MLMultiArray
    public let keyPaddingMask: MLMultiArray

    public init(cacheDim: Int, maxSeqLength: Int, isStateful: Bool = false) throws {
        self.cacheDim = cacheDim
        self.maxSeqLength = maxSeqLength
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

        kvCacheUpdateMask = try MLMultiArray(
            shape: [1, NSNumber(value: maxSeqLength)],
            dataType: .float16
        )
        keyPaddingMask = try MLMultiArray(
            shape: [1, NSNumber(value: maxSeqLength)],
            dataType: .float16
        )

        reset()
    }

    public func reset() {
        cacheLength = 0

        // Zero-fill external KV caches (stateful models don't have them)
        if let keyCache, let valueCache {
            memset(keyCache.dataPointer, 0, cacheDim * maxSeqLength * MemoryLayout<FloatType>.size)
            memset(valueCache.dataPointer, 0, cacheDim * maxSeqLength * MemoryLayout<FloatType>.size)
        }

        // Initialize masks: first position active, rest masked
        let updatePtr = kvCacheUpdateMask.dataPointer.bindMemory(to: FloatType.self, capacity: maxSeqLength)
        let paddingPtr = keyPaddingMask.dataPointer.bindMemory(to: FloatType.self, capacity: maxSeqLength)
        for i in 0..<maxSeqLength {
            updatePtr[i] = (i == 0) ? FloatType(1.0) : FloatType(0.0)
            paddingPtr[i] = (i == 0) ? FloatType(0.0) : FloatType(-10000.0)
        }
    }

    /// Write cache updates at current position and advance.
    /// For stateful models, only advances position and updates masks (KV cache is internal).
    public func update(keyCacheUpdates: MLMultiArray? = nil, valueCacheUpdates: MLMultiArray? = nil) {
        let writePos = Int(cacheLength)
        let seqLen = maxSeqLength

        // Update external KV cache if present (non-stateful model)
        if !isStateful, let keyCache, let valueCache, let keyCacheUpdates, let valueCacheUpdates {
            let embedDim = cacheDim
            let keyCachePtr = keyCache.dataPointer.bindMemory(to: FloatType.self, capacity: embedDim * seqLen)
            let valueCachePtr = valueCache.dataPointer.bindMemory(to: FloatType.self, capacity: embedDim * seqLen)

            let keyUpdatePtr = keyCacheUpdates.dataPointer.bindMemory(to: FloatType.self, capacity: keyCacheUpdates.count)
            let valueUpdatePtr = valueCacheUpdates.dataPointer.bindMemory(to: FloatType.self, capacity: valueCacheUpdates.count)
            let keyUpdateStride = keyCacheUpdates.strides[1].intValue
            let valueUpdateStride = valueCacheUpdates.strides[1].intValue

            for dim in 0..<embedDim {
                keyCachePtr[dim * seqLen + writePos] = keyUpdatePtr[dim * keyUpdateStride]
                valueCachePtr[dim * seqLen + writePos] = valueUpdatePtr[dim * valueUpdateStride]
            }
        }

        // Advance position and update masks
        cacheLength += 1
        let nextPos = Int(cacheLength)

        let updateMaskPtr = kvCacheUpdateMask.dataPointer.bindMemory(to: FloatType.self, capacity: seqLen)
        let paddingMaskPtr = keyPaddingMask.dataPointer.bindMemory(to: FloatType.self, capacity: seqLen)
        updateMaskPtr[writePos] = FloatType(0.0)
        if nextPos < seqLen {
            updateMaskPtr[nextPos] = FloatType(1.0)
            paddingMaskPtr[nextPos] = FloatType(0.0)
        }
    }

    public var isFull: Bool { Int(cacheLength) >= maxSeqLength - 1 }

    /// How many free positions remain before the cache is full
    public var freePositions: Int { maxSeqLength - 1 - Int(cacheLength) }

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
    /// Update-mask as a `[1, maxSeqLength]` Float16 tensor.
    var kvCacheUpdateMaskTensor: MLTensor { MLTensor(MLShapedArray<FloatType>(kvCacheUpdateMask)) }
    /// Padding-mask as a `[1, maxSeqLength]` Float16 tensor.
    var keyPaddingMaskTensor: MLTensor { MLTensor(MLShapedArray<FloatType>(keyPaddingMask)) }
    /// External key-cache tensor - `nil` for stateful models.
    var keyCacheTensor: MLTensor? { keyCache.map { MLTensor(MLShapedArray<FloatType>($0)) } }
    /// External value-cache tensor - `nil` for stateful models.
    var valueCacheTensor: MLTensor? { valueCache.map { MLTensor(MLShapedArray<FloatType>($0)) } }

    /// Async update from MLTensor outputs - materializes without blocking the cooperative pool.
    func update(keyTensor: MLTensor, valueTensor: MLTensor) async {
        let keyArr = await keyTensor.toMLMultiArray()
        let valArr = await valueTensor.toMLMultiArray()
        update(keyCacheUpdates: keyArr, valueCacheUpdates: valArr)
    }
}

// MARK: - Speech Decoder Cache

/// Extended KV cache for SpeechDecoder with a rolling hidden context buffer.
///
/// The hidden context window length varies by quantization variant
/// (e.g. 4 for W8A16, 16 for W16A16) and is read from the model at load time.
/// Task-local, never shared across concurrent tasks.
public class SpeechDecoderCache: KVCache, @unchecked Sendable {
    public let hiddenContext: MLMultiArray // [1, hiddenDim, 1, contextLen]
    public let hiddenDim: Int
    public let hiddenContextLen: Int

    public init(
        cacheDim: Int = Qwen3TTSConstants.sdCacheDim,
        maxSeqLength: Int = Qwen3TTSConstants.sdMaxSeq,
        hiddenDim: Int = Qwen3TTSConstants.sdHiddenDim,
        hiddenContextLen: Int = Qwen3TTSConstants.sdHiddenContextLen
    ) throws {
        self.hiddenDim = hiddenDim
        self.hiddenContextLen = hiddenContextLen
        hiddenContext = try MLMultiArray(
            shape: [1, NSNumber(value: hiddenDim), 1, NSNumber(value: hiddenContextLen)],
            dataType: .float16
        )
        memset(hiddenContext.dataPointer, 0, hiddenDim * hiddenContextLen * MemoryLayout<FloatType>.size)
        // Speech decoder is currently non-stateful
        try super.init(cacheDim: cacheDim, maxSeqLength: maxSeqLength, isStateful: false)
    }

    /// Update KV cache and roll hidden context left, appending new hidden state
    public func updateWithHiddenContext(output: MLFeatureProvider) {
        guard let keyCU = output.featureValue(for: "key_cache_updates")?.multiArrayValue,
            let valCU = output.featureValue(for: "value_cache_updates")?.multiArrayValue
        else {
            return
        }
        super.update(keyCacheUpdates: keyCU, valueCacheUpdates: valCU)

        // Roll hidden context left and append new state
        let hidDim = hiddenDim
        let contextLen = hiddenContextLen
        let hiddenContextPtr = hiddenContext.dataPointer.bindMemory(to: FloatType.self, capacity: hidDim * contextLen)
        guard let updateArr = output.featureValue(for: "hidden_context_update")?.multiArrayValue else { return }
        let hiddenUpdatePtr = updateArr.dataPointer.bindMemory(to: FloatType.self, capacity: updateArr.count)
        let hiddenUpdateStride = updateArr.strides[1].intValue

        for dim in 0..<hidDim {
            // Shift left by one position
            for timeStep in 0..<(contextLen - 1) {
                hiddenContextPtr[dim * contextLen + timeStep] = hiddenContextPtr[dim * contextLen + timeStep + 1]
            }
            // Append new value at the end (stride-safe)
            hiddenContextPtr[dim * contextLen + (contextLen - 1)] = hiddenUpdatePtr[dim * hiddenUpdateStride]
        }
    }
}

// MARK: - Speech Decoder Cache MLTensor Access

@available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
public extension SpeechDecoderCache {
    var hiddenContextTensor: MLTensor { MLTensor(MLShapedArray<FloatType>(hiddenContext)) }

    /// Update KV cache and rolling hidden context from `[String: MLTensor]` prediction outputs.
    /// Materializes tensors asynchronously to avoid blocking the cooperative thread pool.
    func updateWithHiddenContext(tensorOutputs: [String: MLTensor]) async {
        guard let keyUpdateTensor = tensorOutputs["key_cache_updates"],
            let valueUpdateTensor = tensorOutputs["value_cache_updates"]
        else {
            return
        }
        await super.update(keyTensor: keyUpdateTensor, valueTensor: valueUpdateTensor)

        let hidDim = hiddenDim
        let contextLen = hiddenContextLen
        let ctxPtr = hiddenContext.dataPointer.bindMemory(to: FloatType.self, capacity: hidDim * contextLen)
        guard let hiddenUpdateTensor = tensorOutputs["hidden_context_update"] else { return }
        let updateArr = await hiddenUpdateTensor.toMLMultiArray()
        let updatePtr = updateArr.dataPointer.bindMemory(to: FloatType.self, capacity: updateArr.count)
        let updateStride = updateArr.strides[1].intValue
        for dim in 0..<hidDim {
            for t in 0..<(contextLen - 1) {
                ctxPtr[dim * contextLen + t] = ctxPtr[dim * contextLen + t + 1]
            }
            ctxPtr[dim * contextLen + (contextLen - 1)] = updatePtr[dim * updateStride]
        }
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
        let keyArr = await keyTensor.toMLMultiArray()
        let valArr = await valueTensor.toMLMultiArray()
        updateStateCache(state: state, keyCacheUpdates: keyArr, valueCacheUpdates: valArr, position: position)
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
