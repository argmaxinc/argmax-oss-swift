//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Accelerate
import ArgmaxCore
import CoreML
import Foundation

// MARK: - Implementation

/// RVQ-to-audio waveform decoder backed by a CoreML model.
///
/// Thread safety: mutable state (`model`, dimension properties) is set once during
/// `loadModel()` and read-only thereafter. `MLModel.prediction()` is thread-safe.
public class Qwen3SpeechDecoder: SpeechDecoding, @unchecked Sendable {
    public var model: MLModel?

    // MARK: - Audio format

    public let sampleRate: Int = Qwen3TTSConstants.sampleRate
    public let samplesPerFrame: Int = Qwen3TTSConstants.samplesPerFrame
    /// Minimum pre-buffer: 80ms ≈ 2 audio frames at 24 kHz / 1920 spf.
    public let minimumBufferDuration: TimeInterval = 0.08

    /// Detected from model metadata at load time
    public private(set) var hiddenContextLen: Int = Qwen3TTSConstants.sdHiddenContextLen
    /// KV cache embedding dimension
    public private(set) var kvCacheEmbedDim: Int = Qwen3TTSConstants.sdCacheDim
    /// KV cache max sequence length
    public private(set) var kvCacheMaxSequenceLength: Int = Qwen3TTSConstants.sdMaxSeq
    /// Hidden state dimension
    public private(set) var hiddenDim: Int = Qwen3TTSConstants.sdHiddenDim
    public init() {}

    public func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool = false) async throws {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits
        let loaded = try await MLModel.load(contentsOf: url, configuration: modelConfig)

        // In prewarm mode, compilation is complete - discard to free memory before next model compiles
        guard !prewarmMode else { return }

        self.model = loaded

        // Detect dimensions from model description
        // hidden_context input shape: [1, hiddenDim, 1, contextLen]
        if let dim = ModelUtilities.getModelInputDimension(model, named: "hidden_context", position: 1) {
            self.hiddenDim = dim
        }
        if let ctxLen = ModelUtilities.getModelInputDimension(model, named: "hidden_context", position: 3) {
            self.hiddenContextLen = ctxLen
        }
        // key_cache input shape: [1, cacheDim, 1, maxSeqLen]
        if let dim = ModelUtilities.getModelInputDimension(model, named: "key_cache", position: 1) {
            self.kvCacheEmbedDim = dim
        }
        if let seq = ModelUtilities.getModelInputDimension(model, named: "key_cache", position: 3) {
            self.kvCacheMaxSequenceLength = seq
        }
    }

    public func decodeFrame(
        codes: [Int32],
        cache: SpeechDecoderCache
    ) async throws -> [Float] {
        guard let model else {
            throw TTSError.generationFailed("SpeechDecoder model not loaded")
        }

        let codesArr = try MLMultiArray(shape: [1, 16, 1], dataType: .int32)
        let codesPtr = codesArr.dataPointer.bindMemory(to: Int32.self, capacity: 16)
        for i in 0..<16 {
            codesPtr[i] = codes[i]
        }

        guard let keyCache = cache.keyCache, let valueCache = cache.valueCache else {
            throw TTSError.generationFailed("SpeechDecoder: KV cache not initialized")
        }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "audio_codes": MLFeatureValue(multiArray: codesArr),
            "cache_length": MLFeatureValue(multiArray: cache.makeCacheLengthArray()),
            "key_cache": MLFeatureValue(multiArray: keyCache),
            "value_cache": MLFeatureValue(multiArray: valueCache),
            "kv_cache_update_mask": MLFeatureValue(multiArray: cache.kvCacheUpdateMask),
            "key_padding_mask": MLFeatureValue(multiArray: cache.keyPaddingMask),
            "hidden_context": MLFeatureValue(multiArray: cache.hiddenContext)
        ])

        let output = try await model.asyncPrediction(from: input)
        cache.updateWithHiddenContext(output: output)

        guard let audioArr = output.featureValue(for: "audio")?.multiArrayValue else {
            throw TTSError.generationFailed("SpeechDecoder: missing audio output array")
        }
        let sampleCount = audioArr.count
        let audioPtr = audioArr.dataPointer.bindMemory(to: FloatType.self, capacity: sampleCount)
        var samples = [Float](repeating: 0, count: sampleCount)
        for i in 0..<sampleCount {
            samples[i] = Float(audioPtr[i])
        }

        return samples
    }

    /// Async variant for concurrent execution via `async let`.
    /// On macOS 15+ / iOS 18+ uses `[String: MLTensor]` directly to avoid FeatureProvider boxing.
    /// Falls back to `asyncPrediction(from:)` on older OS.
    /// Returns samples and a SpeechTimings with speechDecoder fields populated.
    public func decodeFrameAsync(
        codes: [Int32],
        cache: SpeechDecoderCache
    ) async throws -> SpeechDecoderTimedResult {
        guard let model else {
            throw TTSError.generationFailed("SpeechDecoder model not loaded")
        }

        var timings = SpeechTimings()

        // TODO: Remove forking logic with package with min os version upgrade
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            guard let keyCacheTensor = cache.keyCacheTensor,
                let valueCacheTensor = cache.valueCacheTensor
            else {
                throw TTSError.generationFailed("SpeechDecoder: KV cache tensors not initialized")
            }
            let inputs: [String: MLTensor] = [
                "audio_codes": MLTensor(shape: [1, codes.count, 1], scalars: codes),
                "cache_length": cache.cacheLengthTensor,
                "key_cache": keyCacheTensor,
                "value_cache": valueCacheTensor,
                "kv_cache_update_mask": cache.kvCacheUpdateMaskTensor,
                "key_padding_mask": cache.keyPaddingMaskTensor,
                "hidden_context": cache.hiddenContextTensor
            ]

            let predictionStart = CFAbsoluteTimeGetCurrent()
            let outputs = try await model.prediction(from: inputs)
            timings.speechDecoderPredictions += CFAbsoluteTimeGetCurrent() - predictionStart

            await cache.updateWithHiddenContext(tensorOutputs: outputs)

            guard let audioTensor = outputs["audio"] else {
                throw TTSError.generationFailed("SpeechDecoder: missing audio tensor output")
            }
            let samples = await audioTensor.toFloatArray()
            return SpeechDecoderTimedResult(samples: samples, timings: timings)
        } else {
            let codesArr = try MLMultiArray(shape: [1, 16, 1], dataType: .int32)
            let codesPtr = codesArr.dataPointer.bindMemory(to: Int32.self, capacity: 16)
            for i in 0..<16 {
                codesPtr[i] = codes[i]
            }
            guard let keyCacheFallback = cache.keyCache, let valueCacheFallback = cache.valueCache else {
                throw TTSError.generationFailed("SpeechDecoder: KV cache not initialized (legacy path)")
            }
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "audio_codes": MLFeatureValue(multiArray: codesArr),
                "cache_length": MLFeatureValue(multiArray: cache.makeCacheLengthArray()),
                "key_cache": MLFeatureValue(multiArray: keyCacheFallback),
                "value_cache": MLFeatureValue(multiArray: valueCacheFallback),
                "kv_cache_update_mask": MLFeatureValue(multiArray: cache.kvCacheUpdateMask),
                "key_padding_mask": MLFeatureValue(multiArray: cache.keyPaddingMask),
                "hidden_context": MLFeatureValue(multiArray: cache.hiddenContext)
            ])

            let predictionStart = CFAbsoluteTimeGetCurrent()
            let output = try await model.asyncPrediction(from: input, options: MLPredictionOptions())
            timings.speechDecoderPredictions += CFAbsoluteTimeGetCurrent() - predictionStart

            cache.updateWithHiddenContext(output: output)
            guard let audioArr = output.featureValue(for: "audio")?.multiArrayValue else {
                throw TTSError.generationFailed("SpeechDecoder: missing audio output array (legacy path)")
            }
            let sampleCount = audioArr.count
            let audioPtr = audioArr.dataPointer.bindMemory(to: FloatType.self, capacity: sampleCount)
            var samples = [Float](repeating: 0, count: sampleCount)
            for i in 0..<sampleCount {
                samples[i] = Float(audioPtr[i])
            }
            return SpeechDecoderTimedResult(samples: samples, timings: timings)
        }
    }

    public func unloadModel() {
        model = nil
    }
}
