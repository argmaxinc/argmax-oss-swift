//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Accelerate
import ArgmaxCore
import CoreML

// MARK: - TTS Embed Type Protocols

// These protocols allow embed tensors to be represented as either MLMultiArray (all platforms)
// or MLTensor (macOS 15+ / iOS 18+) without changing the calling convention.

/// Marker protocol for a raw TTS embedding tensor emitted by a CoreML model.
public protocol EmbedTensorType {}

/// Marker protocol for a TTS embedding value that can be used as CoreML model input.
public protocol EmbedInputType {}

extension MLMultiArray: EmbedTensorType {}
extension MLMultiArray: EmbedInputType {}

@available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
extension MLTensor: EmbedTensorType {}

@available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
extension MLTensor: EmbedInputType {}

extension Array: EmbedTensorType where Element == FloatType {}

// MARK: - Embedding Utilities

/// Static helpers for creating, combining, and extracting TTS embedding vectors.
///
/// Mirrors the `ModelUtilities` pattern in ArgmaxCore: all operations are pure
/// static functions with no shared state.
public enum EmbedUtilities {
    // MARK: - Float16 / FloatType helpers

    /// Element-wise addition of two equal-length embedding vectors via vDSP.
    public static func addEmbeddings(_ a: [FloatType], _ b: [FloatType]) -> [FloatType] {
        precondition(a.count == b.count)
        var result = [Float](repeating: 0, count: a.count)
        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                var aFloat = [Float](repeating: 0, count: a.count)
                var bFloat = [Float](repeating: 0, count: a.count)
                convertToFloat(aPtr, to: &aFloat)
                convertToFloat(bPtr, to: &bFloat)
                vDSP_vadd(aFloat, 1, bFloat, 1, &result, 1, vDSP_Length(a.count))
            }
        }
        return result.map { FloatType($0) }
    }

    /// Accumulates embeddings into a Float32 buffer to avoid repeated type conversions,
    /// reusing a single intermediate buffer rather than allocating one per embed.
    public static func sumEmbeddings(_ embeddings: [[FloatType]]) -> [FloatType] {
        guard let first = embeddings.first else { return [] }
        let count = first.count
        var accum = [Float](repeating: 0, count: count)
        var floatBuf = [Float](repeating: 0, count: count)
        for embed in embeddings {
            embed.withUnsafeBufferPointer { ptr in
                convertToFloat(ptr, to: &floatBuf)
                vDSP_vadd(accum, 1, floatBuf, 1, &accum, 1, vDSP_Length(count))
            }
        }
        var result = [FloatType](repeating: 0, count: count)
        result.withUnsafeMutableBufferPointer { dst in
            accum.withUnsafeBufferPointer { src in
                // vDSP Float16<->Float conversion is available on iOS 14+, macOS 11+, visionOS 1+
                // but not on watchOS. Fall back to scalar on watchOS and x86_64.
                #if arch(arm64) && !os(watchOS)
                vDSP.convertElements(of: src, to: &dst)
                #else
                for i in 0..<count {
                    dst[i] = FloatType(src[i])
                }
                #endif
            }
        }
        return result
    }

    /// Extract an embedding vector from a CoreML `MLMultiArray` output.
    ///
    /// Handles both flat `[D]` and shaped `[1, D, 1, 1]` outputs.
    public static func extractEmbed(from arr: MLMultiArray) -> [FloatType] {
        let dim: Int
        if arr.shape.count == 4 {
            dim = arr.shape[1].intValue
        } else {
            dim = arr.count
        }
        let ptr = arr.dataPointer.bindMemory(to: FloatType.self, capacity: arr.count)
        var result = [FloatType](repeating: 0, count: dim)

        if arr.shape.count == 4 {
            let stride1 = arr.strides[1].intValue
            if stride1 == 1 {
                // Contiguous layout ([1, D, 1, 1] with unit stride) - direct buffer copy
                result.withUnsafeMutableBufferPointer { dst in
                    guard let dstBase = dst.baseAddress else { return }
                    dstBase.update(from: ptr, count: dim)
                }
            } else {
                for d in 0..<dim {
                    result[d] = ptr[d * stride1]
                }
            }
        } else {
            for i in 0..<min(dim, arr.count) {
                result[i] = ptr[i]
            }
        }
        return result
    }

    /// Pack a `[FloatType]` embedding into a CoreML `MLMultiArray` with shape `[1, D, 1, 1]`.
    public static func createEmbedMLArray(_ values: [FloatType]) throws -> MLMultiArray {
        let dim = values.count
        let arr = try MLMultiArray(shape: [1, NSNumber(value: dim), 1, 1], dataType: .float16)
        let ptr = arr.dataPointer.bindMemory(to: FloatType.self, capacity: dim)
        values.withUnsafeBufferPointer { buf in
            guard let bufBase = buf.baseAddress else { return }
            ptr.update(from: bufBase, count: dim)
        }
        return arr
    }

    /// Create a zero-filled embedding vector.
    /// - Parameter dim: Embedding dimension (match the actual model's embed size).
    /// - Returns: A `[FloatType]` array of length `dim` filled with zeros.
    public static func zeroEmbed(dim: Int = 1024) -> [FloatType] {
        [FloatType](repeating: FloatType(0), count: dim)
    }

    /// Pack an `[Int32]` token-id array into a flat CoreML `MLMultiArray`.
    public static func makeInt32Array(_ values: [Int32]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [NSNumber(value: values.count)], dataType: .int32)
        let ptr = arr.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        for (index, value) in values.enumerated() {
            ptr[index] = value
        }
        return arr
    }

    // MARK: - MLTensor helpers

    /// Element-wise addition of two MLTensor embeddings. No data copy - deferred until materialised.
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    @inline(__always)
    public static func addEmbeddings(_ a: MLTensor, _ b: MLTensor) -> MLTensor { a + b }

    /// Sum a list of MLTensor embeddings element-wise.
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public static func sumEmbeddings(_ tensors: [MLTensor]) -> MLTensor {
        tensors.dropFirst().reduce(tensors[0], +)
    }

    // MARK: - Internal conversion helper

    /// Platform-safe conversion from FloatType buffer to Float array.
    /// On arm64 iOS/macOS/visionOS (Float16), uses vDSP for vectorized conversion.
    @inline(__always)
    static func convertToFloat(_ source: UnsafeBufferPointer<FloatType>, to dest: inout [Float]) {
        #if arch(arm64) && !os(watchOS)
        vDSP.convertElements(of: source, to: &dest)
        #else
        for i in 0..<min(source.count, dest.count) {
            dest[i] = Float(source[i])
        }
        #endif
    }
}

// MARK: - Array<FloatType> convenience

@available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
public extension Array where Element == FloatType {
    /// Wrap this embedding vector as a `[1, count, 1, 1]` Float16 MLTensor (one memcpy).
    func asMLTensor() -> MLTensor {
        MLTensor(MLShapedArray<FloatType>(scalars: self, shape: [1, count, 1, 1]))
    }
}
