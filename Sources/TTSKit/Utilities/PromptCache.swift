//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
import Foundation

// MARK: - Prompt Cache

/// Cached CodeDecoder KV state from prefilling the invariant prompt prefix.
///
/// The invariant prefix includes the role prefix tokens, speaker/language control tokens,
/// and optional instruction tokens - everything that stays the same across different text
/// inputs for the same voice configuration.
///
/// Create via `TTSKit.buildPromptCache(voice:language:instruction:)` or let
/// `generate` build one automatically on the first call.
///
/// **Savings**: For a typical 10-token prefix, caching saves ~9 CodeDecoder forward
/// passes (~290ms on M4), reducing prefill from ~320ms to ~32ms per generation.
///
/// Thread safety: immutable after creation. Multiple concurrent generation tasks can
/// safely restore from the same cache - each creates its own KV cache copy.
public final class TTSPromptCache: @unchecked Sendable {
    /// Voice identifier this cache was built for.
    public let voice: String
    /// Language identifier this cache was built for.
    public let language: String
    /// Instruction string this cache was built for (nil for no instruction).
    public let instruction: String?
    /// Number of invariant prefix tokens in this cache.
    public let prefixLength: Int

    let kvSnapshot: KVCacheSnapshot
    let stateData: KVStateData?

    init(
        voice: String,
        language: String,
        instruction: String?,
        prefixLength: Int,
        kvSnapshot: KVCacheSnapshot,
        stateData: KVStateData?
    ) {
        self.voice = voice
        self.language = language
        self.instruction = instruction
        self.prefixLength = prefixLength
        self.kvSnapshot = kvSnapshot
        self.stateData = stateData
    }

    /// Whether this cache matches the given generation parameters.
    public func matches(voice: String, language: String, instruction: String?) -> Bool {
        self.voice == voice && self.language == language && self.instruction == instruction
    }
}

// MARK: - KV Cache Snapshot

/// Serializable snapshot of a `KVCache` state (masks, position counter, and
/// optionally key/value cache data for non-stateful models).
public struct KVCacheSnapshot: Sendable {
    public let isStateful: Bool
    public let cacheDim: Int
    public let maxSeqLength: Int
    public let cacheLength: Int32

    /// Raw bytes from keyCache/valueCache (non-stateful models).
    /// Empty `Data()` for stateful models where KV data lives in MLState.
    public let keyCacheData: Data
    public let valueCacheData: Data

    public let updateMaskData: Data
    public let paddingMaskData: Data
}

/// Serializable snapshot of MLState KV buffers (stateful models only).
public struct KVStateData: Sendable {
    public let keyData: Data
    public let valueData: Data
}

// MARK: - KVCache Snapshot/Restore

public extension KVCache {
    /// Create a serializable snapshot of the current cache state.
    func snapshot() -> KVCacheSnapshot {
        let maskBytes = maxSeqLength * MemoryLayout<FloatType>.size

        return KVCacheSnapshot(
            isStateful: isStateful,
            cacheDim: cacheDim,
            maxSeqLength: maxSeqLength,
            cacheLength: cacheLength,
            keyCacheData: keyCache.map { Data(bytes: $0.dataPointer, count: cacheDim * maxSeqLength * MemoryLayout<FloatType>.size) } ?? Data(),
            valueCacheData: valueCache.map { Data(bytes: $0.dataPointer, count: cacheDim * maxSeqLength * MemoryLayout<FloatType>.size) } ?? Data(),
            updateMaskData: Data(bytes: kvCacheUpdateMask.dataPointer, count: maskBytes),
            paddingMaskData: Data(bytes: keyPaddingMask.dataPointer, count: maskBytes)
        )
    }

    /// Restore cache state from a snapshot. The snapshot must have matching geometry.
    func restore(from snapshot: KVCacheSnapshot) {
        precondition(
            snapshot.cacheDim == cacheDim && snapshot.maxSeqLength == maxSeqLength,
            "Cache geometry mismatch: expected (\(cacheDim), \(maxSeqLength)), got (\(snapshot.cacheDim), \(snapshot.maxSeqLength))"
        )

        cacheLength = snapshot.cacheLength

        let maskBytes = maxSeqLength * MemoryLayout<FloatType>.size

        if let keyCache, let valueCache, !snapshot.keyCacheData.isEmpty {
            let kvBytes = cacheDim * maxSeqLength * MemoryLayout<FloatType>.size
            snapshot.keyCacheData.copyBytes(to: keyCache.dataPointer.assumingMemoryBound(to: UInt8.self), count: min(snapshot.keyCacheData.count, kvBytes))
            snapshot.valueCacheData.copyBytes(to: valueCache.dataPointer.assumingMemoryBound(to: UInt8.self), count: min(snapshot.valueCacheData.count, kvBytes))
        }

        snapshot.updateMaskData.copyBytes(to: kvCacheUpdateMask.dataPointer.assumingMemoryBound(to: UInt8.self), count: min(snapshot.updateMaskData.count, maskBytes))
        snapshot.paddingMaskData.copyBytes(to: keyPaddingMask.dataPointer.assumingMemoryBound(to: UInt8.self), count: min(snapshot.paddingMaskData.count, maskBytes))
    }
}

// MARK: - MLState Snapshot/Restore

@available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
public extension MLState {
    /// Capture the current `self_attn_key_cache` and `self_attn_value_cache` buffers as raw `Data`.
    func snapshot() -> KVStateData {
        var keyData = Data()
        withMultiArray(for: "self_attn_key_cache") { arr in
            arr.withUnsafeMutableBytes { buf, _ in
                guard let baseAddress = buf.baseAddress else { return }
                keyData = Data(bytes: baseAddress, count: buf.count)
            }
        }
        var valueData = Data()
        withMultiArray(for: "self_attn_value_cache") { arr in
            arr.withUnsafeMutableBytes { buf, _ in
                guard let baseAddress = buf.baseAddress else { return }
                valueData = Data(bytes: baseAddress, count: buf.count)
            }
        }
        return KVStateData(keyData: keyData, valueData: valueData)
    }

    /// Overwrite the `self_attn_key_cache` and `self_attn_value_cache` buffers from a snapshot.
    func restore(from data: KVStateData) {
        withMultiArray(for: "self_attn_key_cache") { arr in
            arr.withUnsafeMutableBytes { buf, _ in
                guard let bufBase = buf.baseAddress else { return }
                data.keyData.withUnsafeBytes { src in
                    guard let srcBase = src.baseAddress else { return }
                    bufBase.copyMemory(from: srcBase, byteCount: min(src.count, buf.count))
                }
            }
        }
        withMultiArray(for: "self_attn_value_cache") { arr in
            arr.withUnsafeMutableBytes { buf, _ in
                guard let bufBase = buf.baseAddress else { return }
                data.valueData.withUnsafeBytes { src in
                    guard let srcBase = src.baseAddress else { return }
                    bufBase.copyMemory(from: srcBase, byteCount: min(src.count, buf.count))
                }
            }
        }
    }
}

// MARK: - Disk Persistence

public extension TTSPromptCache {
    /// Save this prompt cache to disk as a property list.
    ///
    /// The file can be reloaded with `TTSPromptCache.load(from:)` as long as
    /// the model variant and cache geometry haven't changed.
    func save(to url: URL) throws {
        let container = CacheContainer(
            voice: voice,
            language: language,
            instruction: instruction,
            prefixLength: prefixLength,
            isStateful: kvSnapshot.isStateful,
            cacheDim: kvSnapshot.cacheDim,
            maxSeqLength: kvSnapshot.maxSeqLength,
            cacheLength: kvSnapshot.cacheLength,
            keyCacheData: kvSnapshot.keyCacheData,
            valueCacheData: kvSnapshot.valueCacheData,
            updateMaskData: kvSnapshot.updateMaskData,
            paddingMaskData: kvSnapshot.paddingMaskData,
            stateKeyData: stateData?.keyData,
            stateValueData: stateData?.valueData
        )
        let data = try PropertyListEncoder().encode(container)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
    }

    /// Load a previously saved prompt cache from disk.
    static func load(from url: URL) throws -> TTSPromptCache {
        let data = try Data(contentsOf: url)
        let container = try PropertyListDecoder().decode(CacheContainer.self, from: data)
        let snapshot = KVCacheSnapshot(
            isStateful: container.isStateful,
            cacheDim: container.cacheDim,
            maxSeqLength: container.maxSeqLength,
            cacheLength: container.cacheLength,
            keyCacheData: container.keyCacheData,
            valueCacheData: container.valueCacheData,
            updateMaskData: container.updateMaskData,
            paddingMaskData: container.paddingMaskData
        )
        let stateData: KVStateData?
        if let keyData = container.stateKeyData, let valueData = container.stateValueData {
            stateData = KVStateData(keyData: keyData, valueData: valueData)
        } else {
            stateData = nil
        }
        return TTSPromptCache(
            voice: container.voice,
            language: container.language,
            instruction: container.instruction,
            prefixLength: container.prefixLength,
            kvSnapshot: snapshot,
            stateData: stateData
        )
    }

    /// File name for this cache based on voice/language/instruction.
    var cacheFileName: String {
        var name = "\(voice)_\(language)"
        if let instruction, !instruction.isEmpty {
            let hash = instruction.utf8.reduce(into: UInt64(5381)) { $0 = $0 &* 33 &+ UInt64($1) }
            name += "_\(String(hash, radix: 16))"
        }
        return name + ".promptcache"
    }
}

/// Codable container for plist serialization (Data fields are stored as binary, not base64).
private struct CacheContainer: Codable {
    let voice: String
    let language: String
    let instruction: String?
    let prefixLength: Int
    let isStateful: Bool
    let cacheDim: Int
    let maxSeqLength: Int
    let cacheLength: Int32
    let keyCacheData: Data
    let valueCacheData: Data
    let updateMaskData: Data
    let paddingMaskData: Data
    let stateKeyData: Data?
    let stateValueData: Data?
}
