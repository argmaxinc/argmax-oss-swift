//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Accelerate
import ArgmaxCore
import CoreML
import Foundation

// MARK: - Token Sampling

/// Protocol for TTS token sampling strategies.
public protocol TokenSampling {
    /// Sample a single token from codec-0 logits.
    /// `logits` is an MLTensor (macOS 15+ async path) or MLMultiArray (sync path).
    func sampleCodec0(
        logits: any EmbedTensorType,
        temperature: Float,
        topK: Int,
        generatedTokens: [Int32],
        repetitionPenalty: Float,
        suppressTokenIds: Set<Int>
    ) async -> Int32

    /// Sample a single token from one head of multi-code logits.
    /// `allLogits` is an MLTensor (macOS 15+ async path) or MLMultiArray (sync path).
    func sampleMultiHead(
        allLogits: any EmbedTensorType,
        headIndex: Int,
        temperature: Float,
        topK: Int
    ) async -> Int32
}

// MARK: - Greedy / Top-k Sampler

/// Greedy / top-k / temperature token sampler with a seedable RNG.
///
/// Thread safety: each `Qwen3GenerateTask` owns its own `GreedyTokenSampler`
/// instance (created per-task in `TTSKit.createTask()` with a derived seed).
/// The `var rng` is never accessed concurrently because it's single-owner.
public class GreedyTokenSampler: TokenSampling, @unchecked Sendable {
    private var rng: any RandomNumberGenerator

    /// Create a sampler with an optional seed for reproducibility.
    /// - Parameter seed: If provided, uses a deterministic RNG. If nil, uses system RNG.
    public init(seed: UInt64? = nil) {
        if let seed {
            self.rng = SeededRandomNumberGenerator(seed: seed)
        } else {
            self.rng = SystemRandomNumberGenerator()
        }
    }

    public func sampleCodec0(
        logits: any EmbedTensorType,
        temperature: Float,
        topK: Int,
        generatedTokens: [Int32],
        repetitionPenalty: Float,
        suppressTokenIds: Set<Int>
    ) async -> Int32 {
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            let tensor: MLTensor
            let vocabSize: Int
            if let logitsTensor = logits as? MLTensor {
                vocabSize = logitsTensor.shape.last ?? 0
                tensor = logitsTensor
            } else {
                guard let logitsArray = logits as? MLMultiArray else {
                    return Int32(Qwen3TTSConstants.codecBOS)
                }
                vocabSize = logitsArray.shape.last?.intValue ?? 0
                tensor = MLTensor(MLShapedArray<FloatType>(logitsArray))
            }
            return await sampleCodec0WithMLTensor(
                logitsTensor: tensor,
                vocabSize: vocabSize,
                temperature: temperature,
                topK: topK,
                generatedTokens: generatedTokens,
                repetitionPenalty: repetitionPenalty,
                suppressTokenIds: suppressTokenIds
            )
        }
        guard let logitsArray = logits as? MLMultiArray else {
            return Int32(Qwen3TTSConstants.codecBOS)
        }
        return sampleCodec0WithVDSP(
            logits: logitsArray,
            temperature: temperature,
            topK: topK,
            generatedTokens: generatedTokens,
            repetitionPenalty: repetitionPenalty,
            suppressTokenIds: suppressTokenIds
        )
    }

    public func sampleMultiHead(
        allLogits: any EmbedTensorType,
        headIndex: Int,
        temperature: Float,
        topK: Int
    ) async -> Int32 {
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *),
            let tensor = allLogits as? MLTensor
        {
            // Extract a single head lazily via gathering - the full [1, 15, vocabSize] tensor
            // stays on device; only the top-k results (~400 B) are downloaded to CPU.
            return await sampleMultiHeadFromTensor(tensor, headIndex: headIndex, temperature: temperature, topK: topK)
        }

        // MLMultiArray path: pointer arithmetic for zero-copy head extraction
        guard let allLogitsArray = allLogits as? MLMultiArray else {
            return Int32(Qwen3TTSConstants.codecBOS)
        }
        let vocabSize = allLogitsArray.shape[2].intValue
        let ptr = allLogitsArray.dataPointer.bindMemory(to: FloatType.self, capacity: allLogitsArray.count)
        let stride1 = allLogitsArray.strides[1].intValue
        let stride2 = allLogitsArray.strides[2].intValue
        var logitsF = [Float](repeating: 0, count: vocabSize)
        let base = headIndex * stride1

        if stride2 == 1 {
            let src = UnsafeBufferPointer(start: ptr.advanced(by: base), count: vocabSize)
            EmbedUtilities.convertToFloat(src, to: &logitsF)
        } else {
            for i in 0..<vocabSize {
                logitsF[i] = Float(ptr[base + i * stride2])
            }
        }

        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            return await sampleFromLogitsWithMLTensor(logitsF, temperature: temperature, topK: topK)
        }
        return sampleFromLogits(logitsF, temperature: temperature, topK: topK)
    }

    // MARK: - Sampling implementations

    private func sampleCodec0WithVDSP(
        logits: MLMultiArray,
        temperature: Float,
        topK: Int,
        generatedTokens: [Int32],
        repetitionPenalty: Float,
        suppressTokenIds: Set<Int>
    ) -> Int32 {
        let vocabSize = logits.shape.last?.intValue ?? 0
        var logitsF = extractFloat32Logits(logits, count: vocabSize)
        for id in suppressTokenIds where id < vocabSize {
            logitsF[id] = -.infinity
        }
        if repetitionPenalty != 1.0 && !generatedTokens.isEmpty {
            for tokenId in Set(generatedTokens) {
                let tokenIndex = Int(tokenId)
                guard tokenIndex < vocabSize else { continue }
                logitsF[tokenIndex] = logitsF[tokenIndex] > 0 ? logitsF[tokenIndex] / repetitionPenalty : logitsF[tokenIndex] * repetitionPenalty
            }
        }
        return sampleFromLogits(logitsF, temperature: temperature, topK: topK)
    }

    /// MLTensor-based codec-0 sampler (macOS 15+).
    /// `logitsTensor` arrives directly from the model output - no MLMultiArray conversion needed.
    /// Uses `MLTensor.topK()` - O(n) partial selection - instead of vDSP_vsort's O(n log n) full sort.
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    private func sampleCodec0WithMLTensor(
        logitsTensor: MLTensor,
        vocabSize: Int,
        temperature: Float,
        topK: Int,
        generatedTokens: [Int32],
        repetitionPenalty: Float,
        suppressTokenIds: Set<Int>
    ) async -> Int32 {
        let needsCPUPass = !suppressTokenIds.isEmpty || (repetitionPenalty != 1.0 && !generatedTokens.isEmpty)

        let processedTensor: MLTensor
        if needsCPUPass {
            // Materialise to Float32, apply scalar modifications, re-wrap as [1, vocabSize] tensor.
            var logitsF = await logitsTensor.reshaped(to: [1, vocabSize]).cast(to: Float.self).toFloatArray()
            for id in suppressTokenIds where id < vocabSize {
                logitsF[id] = -.infinity
            }
            if repetitionPenalty != 1.0 && !generatedTokens.isEmpty {
                for tokenId in Set(generatedTokens) {
                    let tokenIndex = Int(tokenId)
                    guard tokenIndex < vocabSize else { continue }
                    logitsF[tokenIndex] = logitsF[tokenIndex] > 0 ? logitsF[tokenIndex] / repetitionPenalty : logitsF[tokenIndex] * repetitionPenalty
                }
            }
            processedTensor = MLTensor(shape: [1, vocabSize], scalars: logitsF, scalarType: Float.self)
        } else {
            // Fully lazy path: cast + reshape stay on device until argmax/topK materializes them.
            // [1, vocabSize] shape is required - argmax on a 1D tensor yields a 0D scalar.
            processedTensor = logitsTensor.reshaped(to: [1, vocabSize]).cast(to: Float.self)
        }

        if temperature == 0 {
            return Int32(await processedTensor.argmax(alongAxis: -1).toIntArray()[0])
        }

        let probs = (processedTensor / temperature).softmax(alongAxis: -1)
        return await sampleFromProbs(probs, vocabSize: vocabSize, topK: topK)
    }

    /// Extract a single head from the multi-code logit tensor and sample from it (macOS 15+).
    /// The full [1, numHeads, vocabSize] tensor stays on device; `gathering` selects the head
    /// lazily so that only the top-k results (~400 B) need to be downloaded to CPU.
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    private func sampleMultiHeadFromTensor(
        _ allLogits: MLTensor,
        headIndex: Int,
        temperature: Float,
        topK: Int
    ) async -> Int32 {
        let vocabSize = allLogits.shape[2]
        // gathering along axis 1: [1, numHeads, vocabSize] -> [1, 1, vocabSize]
        let headLogits = allLogits.gathering(
            atIndices: MLTensor(shape: [1], scalars: [Int32(headIndex)], scalarType: Int32.self),
            alongAxis: 1
        )
        if temperature == 0 {
            return Int32(await headLogits.cast(to: Float.self).argmax(alongAxis: -1).toIntArray()[0])
        }
        let probs = (headLogits.cast(to: Float.self) / temperature).softmax(alongAxis: -1)
        return await sampleFromProbs(probs, vocabSize: vocabSize, topK: topK)
    }

    /// Shared topK multinomial sampler over an already-softmaxed probability MLTensor.
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    private func sampleFromProbs(_ probs: MLTensor, vocabSize: Int, topK: Int) async -> Int32 {
        if topK > 0 && topK < vocabSize {
            // Partial selection: O(n) vs O(n log n) full sort
            let (topKProbs, topKIndices) = probs.topK(topK)
            let probsArray = await topKProbs.toFloatArray()
            let idxArray = await topKIndices.toIntArray()
            let probSum = probsArray.reduce(0, +)
            // Numerical underflow at low temperature and small topK (e.g. 0.10
            // + 15 over long-form generation) can round every top-k probability
            // to zero. Float.random(in: 0..<0) crashes; fall back to greedy
            // (the highest-probability token, which topK returns first).
            // ref: https://github.com/argmaxinc/argmax-oss-swift/issues/450
            guard probSum > 0 else {
                return idxArray.first.map(Int32.init) ?? Int32(vocabSize - 1)
            }
            let randomValue = Float.random(in: 0..<probSum, using: &rng)
            var cumulativeSum: Float = 0
            for (i, probability) in probsArray.enumerated() {
                cumulativeSum += probability
                if cumulativeSum >= randomValue { return Int32(idxArray[i]) }
            }
            return idxArray.last.map(Int32.init) ?? Int32(vocabSize - 1)
        } else {
            let probsArray = await probs.toFloatArray()
            let probSum = probsArray.reduce(0, +)
            guard probSum > 0 else {
                return Int32(vocabSize - 1)
            }
            let randomValue = Float.random(in: 0..<1, using: &rng)
            var cumulativeSum: Float = 0
            for (i, probability) in probsArray.enumerated() {
                cumulativeSum += probability
                if cumulativeSum >= randomValue { return Int32(i) }
            }
            return Int32(vocabSize - 1)
        }
    }

    /// MLTensor sampling from a pre-extracted Float32 logits array (macOS 15+).
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    private func sampleFromLogitsWithMLTensor(_ logits: [Float], temperature: Float, topK: Int) async -> Int32 {
        let vocabSize = logits.count

        if temperature == 0 {
            var maxValue: Float = 0
            var maxIndex: vDSP_Length = 0
            vDSP_maxvi(logits, 1, &maxValue, &maxIndex, vDSP_Length(vocabSize))
            return Int32(maxIndex)
        }

        // [1, vocabSize] keeps topK/softmax results at least 2D so toFloatArray()/toIntArray() are safe
        let logitsTensor = MLTensor(shape: [1, vocabSize], scalars: logits, scalarType: Float.self)
        let probs = (logitsTensor / temperature).softmax(alongAxis: -1)
        return await sampleFromProbs(probs, vocabSize: vocabSize, topK: topK)
    }

    // MARK: - Private helpers

    private func extractFloat32Logits(_ arr: MLMultiArray, count: Int) -> [Float] {
        let ptr = arr.dataPointer.bindMemory(to: FloatType.self, capacity: arr.count)
        let lastStride = arr.strides.last?.intValue ?? 1
        var result = [Float](repeating: 0, count: count)
        if lastStride == 1 {
            let src = UnsafeBufferPointer(start: ptr, count: count)
            EmbedUtilities.convertToFloat(src, to: &result)
        } else {
            for i in 0..<count {
                result[i] = Float(ptr[i * lastStride])
            }
        }
        return result
    }

    private func sampleFromLogits(_ logits: [Float], temperature: Float, topK: Int) -> Int32 {
        var mutableLogits = logits
        let vocabSize = mutableLogits.count

        if temperature == 0 {
            var maxValue: Float = 0
            var maxIndex: vDSP_Length = 0
            vDSP_maxvi(mutableLogits, 1, &maxValue, &maxIndex, vDSP_Length(vocabSize))
            return Int32(maxIndex)
        }

        var temperatureScalar = temperature
        vDSP_vsdiv(mutableLogits, 1, &temperatureScalar, &mutableLogits, 1, vDSP_Length(vocabSize))

        if topK > 0 && topK < vocabSize {
            var sorted = mutableLogits
            vDSP_vsort(&sorted, vDSP_Length(vocabSize), -1)
            let threshold = sorted[topK - 1]
            for i in 0..<vocabSize where mutableLogits[i] < threshold {
                mutableLogits[i] = -.infinity
            }
        }

        var maxValue: Float = 0
        vDSP_maxv(mutableLogits, 1, &maxValue, vDSP_Length(vocabSize))
        var negMax = -maxValue
        vDSP_vsadd(mutableLogits, 1, &negMax, &mutableLogits, 1, vDSP_Length(vocabSize))
        var elementCount = Int32(vocabSize)
        vvexpf(&mutableLogits, mutableLogits, &elementCount)
        var sum: Float = 0
        vDSP_sve(mutableLogits, 1, &sum, vDSP_Length(vocabSize))
        if sum > 0 {
            vDSP_vsdiv(mutableLogits, 1, &sum, &mutableLogits, 1, vDSP_Length(vocabSize))
        }

        let randomValue = Float.random(in: 0..<1, using: &rng)
        var cumulativeSum: Float = 0
        for i in 0..<vocabSize {
            cumulativeSum += mutableLogits[i]
            if cumulativeSum >= randomValue { return Int32(i) }
        }
        return Int32(vocabSize - 1)
    }
}

// MARK: - Seedable RNG

/// A seedable random number generator using xoshiro256 algorithm.
/// Produces deterministic sequences for a given seed.
public struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: (UInt64, UInt64, UInt64, UInt64)

    public init(seed: UInt64) {
        var z = seed &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        let s0 = z ^ (z >> 31)

        z = (seed &+ 2 &* 0x9E37_79B9_7F4A_7C15)
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        let s1 = z ^ (z >> 31)

        z = (seed &+ 3 &* 0x9E37_79B9_7F4A_7C15)
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        let s2 = z ^ (z >> 31)

        z = (seed &+ 4 &* 0x9E37_79B9_7F4A_7C15)
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        let s3 = z ^ (z >> 31)

        state = (s0, s1, s2, s3)
    }

    public mutating func next() -> UInt64 {
        let result = rotl(state.1 &* 5, 7) &* 9
        let shifted = state.1 << 17
        state.2 ^= state.0
        state.3 ^= state.1
        state.1 ^= state.2
        state.0 ^= state.3
        state.2 ^= shifted
        state.3 = rotl(state.3, 45)
        return result
    }

    private func rotl(_ x: UInt64, _ k: Int) -> UInt64 {
        (x << k) | (x >> (64 - k))
    }
}
