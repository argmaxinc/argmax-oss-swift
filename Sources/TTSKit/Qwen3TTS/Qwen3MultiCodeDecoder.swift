//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
@preconcurrency import CoreML
import Foundation

// MARK: - Supporting Types

/// Update and padding masks for a single MLTensor decode step.
@available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
struct MLTensorMasks {
    let updateMask: MLTensor
    let paddingMask: MLTensor
}

/// Result of a single MLTensor prediction step, including the updated KV cache.
@available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
struct MLTensorStepResult {
    let outputs: [String: MLTensor]
    let keyCache: MLTensor
    let valueCache: MLTensor
    let cachePosition: Int32
    let predictionTime: TimeInterval
    let cacheUpdateTime: TimeInterval
}

// MARK: - Implementation

/// Multi-code RVQ decoder backed by a CoreML model.
///
/// Thread safety: mutable state (`model`, dimension properties) is set once during
/// `loadModel()` and read-only thereafter. `MLModel.prediction()` is thread-safe.
/// Per-call state is created locally within `generateMultiCodes()` and never stored
/// on this shared instance.
public class Qwen3MultiCodeDecoder: MultiCodeDecoding, @unchecked Sendable {
    @Protected public var model: MLModel?

    /// KV cache embedding dimension, detected from model at load time
    public private(set) var kvCacheEmbedDim: Int = Qwen3TTSConstants.mcdCacheDim
    /// KV cache max sequence length, detected from model at load time
    public private(set) var kvCacheMaxSequenceLength: Int = Qwen3TTSConstants.mcdMaxSeq
    /// Codec vocabulary size per head (codes 1-15), detected from model output
    public private(set) var codecVocabSize: Int = Qwen3TTSConstants.codecVocabSize

    public init() {}

    public func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool = false) async throws {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits
        let loaded = try await MLModel.load(contentsOf: url, configuration: modelConfig)

        // In prewarm mode, compilation is complete - discard to free memory before next model compiles
        guard !prewarmMode else { return }

        self.model = loaded

        // Fused variant: the entire 15-code frame is ONE CoreML program with
        // in-graph Gumbel-max top-k sampling and embedding lookups -
        // identified by its `gumbel` noise input (stepped models have none).
        self.isFused = loaded.modelDescription.inputDescriptionsByName["gumbel"] != nil

        // Detect dimensions from model description
        if let dim = ModelUtilities.getModelOutputDimension(model, named: "key_cache_updates", position: 1) {
            self.kvCacheEmbedDim = dim
        }
        if let seq = ModelUtilities.getModelInputDimension(model, named: "key_padding_mask", position: 1) {
            self.kvCacheMaxSequenceLength = seq
        }
        // all_logits output shape: [1, 15, codecVocabSize]
        if let vocab = ModelUtilities.getModelOutputDimension(model, named: "all_logits", position: 2) {
            self.codecVocabSize = vocab
        }
        // input_embeds shape: [1, embedDim, 1, 1]
        if let embedDim = ModelUtilities.getModelInputDimension(model, named: "input_embeds", position: 1) {
            self.inputEmbedDim = embedDim
        }
    }

    /// Embedding dimension for `input_embeds`, detected from the model at load time.
    public private(set) var inputEmbedDim: Int = Qwen3TTSConstants.embedDim

    /// True when the loaded model is the fused one-call-per-frame variant.
    public private(set) var isFused = false

    /// One CoreML call for the whole 15-code frame. Sampling runs in-graph via
    /// the Gumbel-max trick: `argmax(logits/T + G)` with `G = -log(-log(U))`
    /// draws from `softmax(logits/T)` - the same distribution as the stepped
    /// host sampler (top-k is baked into the graph). The host supplies the
    /// noise, so the call is deterministic given it. `embed_sum` is the sum of
    /// the sampled codes' MultiCodeEmbedder rows - the caller adds
    /// CodeEmbedder(code0) and the text embed to form the next talker input.
    ///
    /// Note: noise comes from the system RNG, so `GenerationOptions.seed` does
    /// not reproduce fused-path audio (the stepped path keeps that property).
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public func generateMultiCodesFused(
        hiddenStatesTensor: MLTensor,
        code0EmbedTensor: MLTensor,
        options: GenerationOptions
    ) async throws -> (codes: [Int32], embedSum: MLTensor, predictionTime: TimeInterval) {
        guard let model else { throw TTSError.generationFailed("MultiCodeDecoder model not loaded") }
        // The graph divides logits by temperature; keep it strictly positive.
        let temperature = max(options.temperature, 0.05)
        var noise = [FloatType](repeating: 0, count: Qwen3TTSConstants.mcdHeads * codecVocabSize)
        for index in noise.indices {
            let uniform = Float.random(in: Float.leastNormalMagnitude..<1)
            noise[index] = FloatType(-log(-log(uniform)))
        }
        let inputs: [String: MLTensor] = [
            "talker_hidden": hiddenStatesTensor,
            "code0_embed": code0EmbedTensor,
            "gumbel": MLTensor(shape: [Qwen3TTSConstants.mcdHeads, codecVocabSize], scalars: noise),
            "temperature": MLTensor(shape: [1], scalars: [FloatType(temperature)]),
        ]
        let predictionStart = CFAbsoluteTimeGetCurrent()
        let outputs = try await model.prediction(from: inputs)
        guard let codesTensor = outputs["codes"], let embedSum = outputs["embed_sum"] else {
            throw TTSError.generationFailed("Fused MultiCodeDecoder: missing codes/embed_sum outputs")
        }
        let codes = await codesTensor.toIntArray().map(Int32.init)
        guard codes.count == Qwen3TTSConstants.mcdHeads else {
            throw TTSError.generationFailed("Fused MultiCodeDecoder: expected 15 codes, got \(codes.count)")
        }
        return (codes, embedSum, CFAbsoluteTimeGetCurrent() - predictionStart)
    }

    public var isStateful: Bool {
        guard let model else { return false }
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            return !model.modelDescription.stateDescriptionsByName.isEmpty
        }
        return false
    }

    /// Create a fresh MLState for a new RVQ frame (stateful models only).
    /// Returns nil for non-stateful models. The caller owns the returned state.
    public func makeState() -> Any? {
        guard isStateful, let model else { return nil }
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            return model.makeState()
        }
        return nil
    }

    /// Pre-initialize the ANE pipeline by running dummy predictions before the first
    /// real generation step. Without this, the first `generateMultiCodes` call per
    /// generation is ~7x slower than steady state due to lazy ANE pipeline setup.
    ///
    /// Run concurrently with CodeDecoder prefill so there is no net TTFB cost.
    /// Replicates the exact loop pattern used in `generateMultiCodes` (4 passes
    /// × 16 predictions each) to match what the ANE needs to pipeline.
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public func prewarmInference() async throws {
        guard let model else { return }
        if isFused {
            // The fused model has different inputs (no external caches/masks);
            // a few dummy full-frame calls pipeline the ANE the same way.
            let dummyHidden = MLTensor(zeros: [1, inputEmbedDim, 1, 1], scalarType: FloatType.self)
            let dummyGumbel = MLTensor(zeros: [Qwen3TTSConstants.mcdHeads, codecVocabSize], scalarType: FloatType.self)
            let dummyTemp = MLTensor(shape: [1], scalars: [FloatType(1)])
            for _ in 0..<4 {
                _ = try await model.prediction(from: [
                    "talker_hidden": dummyHidden, "code0_embed": dummyHidden,
                    "gumbel": dummyGumbel, "temperature": dummyTemp,
                ])
            }
            return
        }
        let sequenceLength = kvCacheMaxSequenceLength
        let dummyInput = MLTensor(zeros: [1, inputEmbedDim, 1, 1], scalarType: FloatType.self)
        for _ in 0..<4 {
            var keyCache = MLTensor(zeros: [1, kvCacheEmbedDim, 1, sequenceLength], scalarType: FloatType.self)
            var valueCache = MLTensor(zeros: [1, kvCacheEmbedDim, 1, sequenceLength], scalarType: FloatType.self)
            var cachePosition: Int32 = 0
            for _ in 0..<16 {
                let stepResult = try await predictMLTensorStep(
                    inputEmbeds: dummyInput, model: model,
                    keyCache: keyCache, valueCache: valueCache,
                    cachePosition: cachePosition, sequenceLength: sequenceLength
                )
                keyCache = stepResult.keyCache
                valueCache = stepResult.valueCache
                cachePosition = stepResult.cachePosition
            }
        }
    }

    public func decode(inputEmbeds: any EmbedInputType, cache: KVCache, state: Any? = nil) async throws -> MultiCodeDecoderOutput {
        guard let model else {
            throw TTSError.generationFailed("MultiCodeDecoder model not loaded")
        }
        guard let array = inputEmbeds as? MLMultiArray else {
            throw TTSError.generationFailed("MultiCodeDecoder: unsupported embed input type \(type(of: inputEmbeds))")
        }

        var dict: [String: MLFeatureValue] = try [
            "input_embeds": MLFeatureValue(multiArray: array),
            "cache_length": MLFeatureValue(multiArray: cache.makeCacheLengthArray()),
            "kv_cache_update_mask": MLFeatureValue(multiArray: cache.kvCacheUpdateMask),
            "key_padding_mask": MLFeatureValue(multiArray: cache.keyPaddingMask)
        ]

        // Only pass external KV cache for non-stateful models
        if !isStateful, let keyCache = cache.keyCache, let valueCache = cache.valueCache {
            dict["key_cache"] = MLFeatureValue(multiArray: keyCache)
            dict["value_cache"] = MLFeatureValue(multiArray: valueCache)
        }

        let input = try MLDictionaryFeatureProvider(dictionary: dict)

        let output: MLFeatureProvider
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), let mlState = state as? MLState {
            output = try await model.asyncPrediction(from: input, using: mlState)
        } else {
            output = try await model.asyncPrediction(from: input)
        }

        guard let keyCacheUpdates = output.featureValue(for: "key_cache_updates")?.multiArrayValue,
            let valueCacheUpdates = output.featureValue(for: "value_cache_updates")?.multiArrayValue
        else {
            throw TTSError.generationFailed("MultiCodeDecoder: missing key/value cache update arrays")
        }

        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), let mlState = state as? MLState, isStateful {
            KVCache.updateStateCache(
                state: mlState,
                keyCacheUpdates: keyCacheUpdates,
                valueCacheUpdates: valueCacheUpdates,
                position: Int(cache.cacheLength)
            )
        }

        guard let allLogitsArray = output.featureValue(for: "all_logits")?.multiArrayValue else {
            throw TTSError.generationFailed("MultiCodeDecoder: missing all_logits array")
        }
        return MultiCodeDecoderOutput(
            allLogits: allLogitsArray,
            keyCacheUpdates: keyCacheUpdates,
            valueCacheUpdates: valueCacheUpdates
        )
    }

    /// Pure MLTensor path - cache lives as tensors, updated via element-wise masking.
    /// No MLMultiArray round-trip: prediction takes/returns MLTensor, cache updates
    /// are lazy tensor ops, and only the logits are materialized (by the sampler).
    /// Build the update mask and padding mask tensors for a given cache position.
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func buildMasks(position: Int, sequenceLength: Int) -> MLTensorMasks {
        var updateData = [FloatType](repeating: 0, count: sequenceLength)
        var paddingData = [FloatType](repeating: -10000, count: sequenceLength)
        if position < sequenceLength { updateData[position] = 1 }
        for index in 0...min(position, sequenceLength - 1) {
            paddingData[index] = 0
        }
        return MLTensorMasks(
            updateMask: MLTensor(shape: [1, sequenceLength], scalars: updateData),
            paddingMask: MLTensor(shape: [1, sequenceLength], scalars: paddingData)
        )
    }

    /// Run one MLTensor prediction step and return the outputs with an updated KV cache.
    ///
    /// Cache updates are performed in tensor space via element-wise masking -
    /// no MLMultiArray round-trip occurs. The model must be compiled for
    /// single-token `[1, embedDim, 1, 1]` input.
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func predictMLTensorStep(
        inputEmbeds: MLTensor,
        model: MLModel,
        keyCache: MLTensor,
        valueCache: MLTensor,
        cachePosition: Int32,
        sequenceLength: Int
    ) async throws -> MLTensorStepResult {
        let masks = buildMasks(position: Int(cachePosition), sequenceLength: sequenceLength)
        let predictionStart = CFAbsoluteTimeGetCurrent()
        let outputs = try await model.prediction(from: [
            "input_embeds": inputEmbeds,
            "cache_length": MLTensor(shape: [1], scalars: [cachePosition]),
            "kv_cache_update_mask": masks.updateMask,
            "key_padding_mask": masks.paddingMask,
            "key_cache": keyCache,
            "value_cache": valueCache
        ])
        let predictionTime = CFAbsoluteTimeGetCurrent() - predictionStart

        let cacheUpdateStart = CFAbsoluteTimeGetCurrent()
        var positionMaskData = [FloatType](repeating: 0, count: sequenceLength)
        positionMaskData[Int(cachePosition)] = 1
        let positionMask = MLTensor(shape: [1, 1, 1, sequenceLength], scalars: positionMaskData)
        let invertedMask = MLTensor(repeating: FloatType(1), shape: [1, 1, 1, sequenceLength]) - positionMask
        guard let keyCacheOutput = outputs["key_cache_updates"],
            let valueCacheOutput = outputs["value_cache_updates"]
        else {
            throw TTSError.generationFailed("MultiCodeDecoder: missing key/value cache update tensors")
        }
        let updatedKeyCache = keyCache * invertedMask + keyCacheOutput * positionMask
        let updatedValueCache = valueCache * invertedMask + valueCacheOutput * positionMask
        let cacheUpdateTime = CFAbsoluteTimeGetCurrent() - cacheUpdateStart

        return MLTensorStepResult(
            outputs: outputs,
            keyCache: updatedKeyCache,
            valueCache: updatedValueCache,
            cachePosition: cachePosition + 1,
            predictionTime: predictionTime,
            cacheUpdateTime: cacheUpdateTime
        )
    }

    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public func generateMultiCodes(
        hiddenStatesTensor: MLTensor,
        code0EmbedTensor: MLTensor,
        multiCodeEmbedder: any MultiCodeEmbedding,
        sampler: any TokenSampling,
        options: GenerationOptions
    ) async throws -> MultiCodeGenerationResult {
        guard let model else { throw TTSError.generationFailed("MultiCodeDecoder model not loaded") }

        let sequenceLength = kvCacheMaxSequenceLength
        var keyCache = MLTensor(zeros: [1, kvCacheEmbedDim, 1, sequenceLength], scalarType: FloatType.self)
        var valueCache = MLTensor(zeros: [1, kvCacheEmbedDim, 1, sequenceLength], scalarType: FloatType.self)
        var cachePosition: Int32 = 0
        var timings = SpeechTimings()

        // Prefill: hidden_states, then code_0_embed. First call's logits are discarded.
        var stepResult = try await predictMLTensorStep(
            inputEmbeds: hiddenStatesTensor, model: model,
            keyCache: keyCache, valueCache: valueCache,
            cachePosition: cachePosition, sequenceLength: sequenceLength
        )
        keyCache = stepResult.keyCache
        valueCache = stepResult.valueCache
        cachePosition = stepResult.cachePosition
        timings.multiCodeDecoderPredictions += stepResult.predictionTime
        timings.totalMultiCodeDecoderPredictions += 1
        timings.decodingKvCaching += stepResult.cacheUpdateTime

        stepResult = try await predictMLTensorStep(
            inputEmbeds: code0EmbedTensor, model: model,
            keyCache: keyCache, valueCache: valueCache,
            cachePosition: cachePosition, sequenceLength: sequenceLength
        )
        keyCache = stepResult.keyCache
        valueCache = stepResult.valueCache
        cachePosition = stepResult.cachePosition
        timings.multiCodeDecoderPredictions += stepResult.predictionTime
        timings.totalMultiCodeDecoderPredictions += 1
        timings.decodingKvCaching += stepResult.cacheUpdateTime

        var stepIndex = 0
        let samplingStart = CFAbsoluteTimeGetCurrent()
        guard let firstStepLogits = stepResult.outputs["all_logits"] else {
            throw TTSError.generationFailed("MultiCodeDecoder: missing all_logits tensor on step 0")
        }
        let code1 = await sampler.sampleMultiHead(
            allLogits: firstStepLogits,
            headIndex: stepIndex,
            temperature: options.temperature,
            topK: options.topK
        )
        timings.multiCodeDecoderSampling += CFAbsoluteTimeGetCurrent() - samplingStart
        var codes: [Int32] = [code1]

        var offsetCodeEmbedTensors: [MLTensor] = []
        offsetCodeEmbedTensors.reserveCapacity(14)

        for _ in 0..<14 {
            let embeddingStart = CFAbsoluteTimeGetCurrent()
            guard let lastCode = codes.last else {
                throw TTSError.generationFailed("MultiCodeDecoder: codes array is empty in MLTensor loop")
            }
            let offsetId = lastCode + Int32(codecVocabSize * stepIndex)
            let embedTensor: MLTensor = try await multiCodeEmbedder.embed(tokenId: offsetId)
            offsetCodeEmbedTensors.append(embedTensor)
            timings.multiCodeDecoderEmbedding += CFAbsoluteTimeGetCurrent() - embeddingStart

            stepResult = try await predictMLTensorStep(
                inputEmbeds: embedTensor, model: model,
                keyCache: keyCache, valueCache: valueCache,
                cachePosition: cachePosition, sequenceLength: sequenceLength
            )
            keyCache = stepResult.keyCache
            valueCache = stepResult.valueCache
            cachePosition = stepResult.cachePosition
            timings.multiCodeDecoderPredictions += stepResult.predictionTime
            timings.totalMultiCodeDecoderPredictions += 1
            timings.decodingKvCaching += stepResult.cacheUpdateTime

            stepIndex += 1

            let nextSamplingStart = CFAbsoluteTimeGetCurrent()
            guard let nextStepLogits = stepResult.outputs["all_logits"] else {
                throw TTSError.generationFailed("MultiCodeDecoder: missing all_logits tensor on step \(stepIndex)")
            }
            let code = await sampler.sampleMultiHead(
                allLogits: nextStepLogits,
                headIndex: stepIndex,
                temperature: options.temperature,
                topK: options.topK
            )
            timings.multiCodeDecoderSampling += CFAbsoluteTimeGetCurrent() - nextSamplingStart
            codes.append(code)
        }

        return MultiCodeGenerationResult(codes: codes, timings: timings, offsetCodeEmbedTensors: offsetCodeEmbedTensors)
    }

    /// Legacy path - kept for OS compatibility (pre-macOS 15).
    // TODO: Remove forking logic with package with min os version upgrade
    public func generateMultiCodes(
        hiddenStates: [FloatType],
        code0Embed: [FloatType],
        multiCodeEmbedder: any MultiCodeEmbedding,
        sampler: any TokenSampling,
        options: GenerationOptions
    ) async throws -> MultiCodeGenerationResult {
        var timings = SpeechTimings()
        let mcdCache = try KVCache(
            cacheDim: kvCacheEmbedDim,
            maxSeqLength: kvCacheMaxSequenceLength,
            isStateful: isStateful
        )

        let frameState = makeState()
        let embedDim = hiddenStates.count
        let reuseArray = try MLMultiArray(shape: [1, NSNumber(value: embedDim), 1, 1], dataType: .float16)

        // Prefill step 1: hiddenStates from CodeDecoder
        let prefillStart = CFAbsoluteTimeGetCurrent()
        var mcdOutput = try await decodeEmbedBuffer(hiddenStates, reuseArray: reuseArray, cache: mcdCache, state: frameState)
        timings.multiCodeDecoderPredictions += CFAbsoluteTimeGetCurrent() - prefillStart
        timings.totalMultiCodeDecoderPredictions += 1

        let cacheUpdateStart = CFAbsoluteTimeGetCurrent()
        guard let prefillKeyUpdates = mcdOutput.keyCacheUpdates,
            let prefillValueUpdates = mcdOutput.valueCacheUpdates
        else {
            throw TTSError.generationFailed("MultiCodeDecoder: missing cache updates after prefill step 1")
        }
        mcdCache.update(keyCacheUpdates: prefillKeyUpdates, valueCacheUpdates: prefillValueUpdates)
        timings.decodingKvCaching += CFAbsoluteTimeGetCurrent() - cacheUpdateStart

        // Prefill step 2: code0 embedding
        let prefill2Start = CFAbsoluteTimeGetCurrent()
        mcdOutput = try await decodeEmbedBuffer(code0Embed, reuseArray: reuseArray, cache: mcdCache, state: frameState)
        timings.multiCodeDecoderPredictions += CFAbsoluteTimeGetCurrent() - prefill2Start
        timings.totalMultiCodeDecoderPredictions += 1

        var stepIndex = 0
        let samplingStart = CFAbsoluteTimeGetCurrent()
        let code1 = await sampler.sampleMultiHead(
            allLogits: mcdOutput.allLogits,
            headIndex: stepIndex,
            temperature: options.temperature,
            topK: options.topK
        )
        timings.multiCodeDecoderSampling += CFAbsoluteTimeGetCurrent() - samplingStart
        var codes: [Int32] = [code1]

        var offsetCodeEmbeds: [[FloatType]] = []
        offsetCodeEmbeds.reserveCapacity(14)

        for _ in 0..<14 {
            let cacheStep = CFAbsoluteTimeGetCurrent()
            guard let loopKeyUpdates = mcdOutput.keyCacheUpdates,
                let loopValueUpdates = mcdOutput.valueCacheUpdates
            else {
                throw TTSError.generationFailed("MultiCodeDecoder: missing cache updates in generation loop")
            }
            mcdCache.update(keyCacheUpdates: loopKeyUpdates, valueCacheUpdates: loopValueUpdates)
            timings.decodingKvCaching += CFAbsoluteTimeGetCurrent() - cacheStep

            let embeddingStart = CFAbsoluteTimeGetCurrent()
            guard let lastCode = codes.last else {
                throw TTSError.generationFailed("MultiCodeDecoder: codes array is empty in legacy loop")
            }
            let offsetId = lastCode + Int32(codecVocabSize * stepIndex)
            let codeEmbedBuf = try await multiCodeEmbedder.embed(tokenId: offsetId)
            offsetCodeEmbeds.append(codeEmbedBuf)
            timings.multiCodeDecoderEmbedding += CFAbsoluteTimeGetCurrent() - embeddingStart

            let decodingStart = CFAbsoluteTimeGetCurrent()
            mcdOutput = try await decodeEmbedBuffer(codeEmbedBuf, reuseArray: reuseArray, cache: mcdCache, state: frameState)
            timings.multiCodeDecoderPredictions += CFAbsoluteTimeGetCurrent() - decodingStart
            timings.totalMultiCodeDecoderPredictions += 1

            stepIndex += 1

            let nextSamplingStart = CFAbsoluteTimeGetCurrent()
            let code = await sampler.sampleMultiHead(
                allLogits: mcdOutput.allLogits,
                headIndex: stepIndex,
                temperature: options.temperature,
                topK: options.topK
            )
            timings.multiCodeDecoderSampling += CFAbsoluteTimeGetCurrent() - nextSamplingStart
            codes.append(code)
        }

        return MultiCodeGenerationResult(
            codes: codes,
            timings: timings,
            offsetCodeEmbeds: offsetCodeEmbeds
        )
    }

    /// Copy `embed` into a pre-allocated reuse array and run a single decode step.
    /// Reusing `reuseArray` avoids per-step MLMultiArray allocation.
    func decodeEmbedBuffer(
        _ embed: [FloatType],
        reuseArray: MLMultiArray,
        cache: KVCache,
        state: Any?
    ) async throws -> MultiCodeDecoderOutput {
        let reusePointer = reuseArray.dataPointer.bindMemory(to: FloatType.self, capacity: embed.count)
        embed.withUnsafeBufferPointer { src in
            guard let baseAddress = src.baseAddress else { return }
            reusePointer.update(from: baseAddress, count: embed.count)
        }
        return try await decode(inputEmbeds: reuseArray, cache: cache, state: state)
    }

    public func unloadModel() {
        model = nil
    }
}
