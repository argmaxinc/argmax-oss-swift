//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
@preconcurrency import CoreML
import Foundation

// MARK: - Implementation

/// Autoregressive codec-0 decoder backed by a CoreML model.
///
/// Thread safety: mutable state (`model`, dimension properties) is set once during
/// `loadModel()` and read-only thereafter. `MLModel.prediction()` is thread-safe.
/// Per-generation `MLState` is created via `makeState()` and owned by each task,
/// never stored on this shared instance.
public class Qwen3CodeDecoder: CodeDecoding, @unchecked Sendable {
    public var model: MLModel?

    /// KV cache embedding dimension, detected from model at load time
    public private(set) var kvCacheEmbedDim: Int = Qwen3TTSConstants.cdCacheDim
    /// KV cache max sequence length, detected from model at load time
    public private(set) var kvCacheMaxSequenceLength: Int = Qwen3TTSConstants.cdMaxSeq
    /// Input embedding dimension
    public private(set) var embedSize: Int = Qwen3TTSConstants.embedDim

    public init() {}

    public func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool = false) async throws {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits
        let loaded = try await MLModel.load(contentsOf: url, configuration: modelConfig)

        // In prewarm mode, compilation is complete - discard to free memory before next model compiles
        guard !prewarmMode else { return }

        self.model = loaded

        // Detect dimensions from model description
        // key_cache_updates output shape: [1, cacheDim, 1, 1]
        if let dim = ModelUtilities.getModelOutputDimension(model, named: "key_cache_updates", position: 1) {
            self.kvCacheEmbedDim = dim
        }
        // key_padding_mask input shape: [1, maxSeqLen]
        if let seq = ModelUtilities.getModelInputDimension(model, named: "key_padding_mask", position: 1) {
            self.kvCacheMaxSequenceLength = seq
        }
        // input_embeds input shape: [1, embedDim, 1, 1]
        if let dim = ModelUtilities.getModelInputDimension(model, named: "input_embeds", position: 1) {
            self.embedSize = dim
        }
    }

    public var isStateful: Bool {
        guard let model else { return false }
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            return !model.modelDescription.stateDescriptionsByName.isEmpty
        }
        return false
    }

    /// Create a fresh MLState for a new generation session (stateful models only).
    /// Returns nil for non-stateful models. The caller owns the returned state.
    public func makeState() -> Any? {
        guard isStateful, let model else { return nil }
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            return model.makeState()
        }
        return nil
    }

    public func decode(inputEmbeds: any EmbedInputType, cache: KVCache, state: Any? = nil) async throws -> CodeDecoderOutput {
        guard let model else { throw TTSError.generationFailed("CodeDecoder model not loaded") }

        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), let tensor = inputEmbeds as? MLTensor {
            return try await decodeWithTensor(tensor, model: model, cache: cache, state: state)
        }

        guard let array = inputEmbeds as? MLMultiArray else {
            throw TTSError.generationFailed("CodeDecoder: unsupported embed input type \(type(of: inputEmbeds))")
        }
        return try await decodeWithMultiArray(array, model: model, cache: cache, state: state)
    }

    /// MLTensor path: passes `[String: MLTensor]` directly - no FeatureProvider boxing.
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    private func decodeWithTensor(_ inputEmbeds: MLTensor, model: MLModel, cache: KVCache, state: Any?) async throws -> CodeDecoderOutput {
        var inputs: [String: MLTensor] = [
            "input_embeds": inputEmbeds,
            "cache_length": cache.cacheLengthTensor,
            "kv_cache_update_mask": cache.kvCacheUpdateMaskTensor,
            "key_padding_mask": cache.keyPaddingMaskTensor
        ]
        if !isStateful, let keyCacheTensor = cache.keyCacheTensor, let valueCacheTensor = cache.valueCacheTensor {
            inputs["key_cache"] = keyCacheTensor
            inputs["value_cache"] = valueCacheTensor
        }

        let outputs: [String: MLTensor]
        if let mlState = state as? MLState {
            outputs = try await model.prediction(from: inputs, using: mlState)
        } else {
            outputs = try await model.prediction(from: inputs)
        }

        guard let keyTensor = outputs["key_cache_updates"],
            let valueTensor = outputs["value_cache_updates"]
        else {
            throw TTSError.generationFailed("CodeDecoder: missing key/value cache update tensors")
        }

        if let mlState = state as? MLState, isStateful {
            await KVCache.updateStateCache(
                state: mlState,
                keyTensor: keyTensor,
                valueTensor: valueTensor,
                position: Int(cache.cacheLength)
            )
        }

        let cacheUpdateStart = CFAbsoluteTimeGetCurrent()
        await cache.update(keyTensor: keyTensor, valueTensor: valueTensor)
        let cacheTime = CFAbsoluteTimeGetCurrent() - cacheUpdateStart

        guard let logitsTensor = outputs["logits"],
            let hiddenStatesTensor = outputs["hidden_states"]
        else {
            throw TTSError.generationFailed("CodeDecoder: missing logits or hidden_states tensor")
        }
        return CodeDecoderOutput(
            logits: logitsTensor,
            hiddenStates: hiddenStatesTensor,
            keyCacheUpdates: nil,
            valueCacheUpdates: nil,
            internalCacheUpdateTime: cacheTime
        )
    }

    /// MLMultiArray path: FeatureProvider-based prediction for older OS versions.
    private func decodeWithMultiArray(_ inputEmbeds: MLMultiArray, model: MLModel, cache: KVCache, state: Any?) async throws -> CodeDecoderOutput {
        var dict: [String: MLFeatureValue] = try [
            "input_embeds": MLFeatureValue(multiArray: inputEmbeds),
            "cache_length": MLFeatureValue(multiArray: cache.makeCacheLengthArray()),
            "kv_cache_update_mask": MLFeatureValue(multiArray: cache.kvCacheUpdateMask),
            "key_padding_mask": MLFeatureValue(multiArray: cache.keyPaddingMask)
        ]
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
            throw TTSError.generationFailed("CodeDecoder: missing key/value cache update arrays")
        }

        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), let mlState = state as? MLState, isStateful {
            KVCache.updateStateCache(
                state: mlState,
                keyCacheUpdates: keyCacheUpdates,
                valueCacheUpdates: valueCacheUpdates,
                position: Int(cache.cacheLength)
            )
        }

        guard let logitsArray = output.featureValue(for: "logits")?.multiArrayValue,
            let hiddenStatesArray = output.featureValue(for: "hidden_states")?.multiArrayValue
        else {
            throw TTSError.generationFailed("CodeDecoder: missing logits or hidden_states array")
        }
        return CodeDecoderOutput(
            logits: logitsArray,
            hiddenStates: EmbedUtilities.extractEmbed(from: hiddenStatesArray),
            keyCacheUpdates: keyCacheUpdates,
            valueCacheUpdates: valueCacheUpdates
        )
    }

    public func unloadModel() {
        model = nil
    }
}
