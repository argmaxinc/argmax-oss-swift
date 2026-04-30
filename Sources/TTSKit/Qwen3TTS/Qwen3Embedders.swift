//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
import Foundation

// MARK: - Code Embedder Implementation

/// Codec-0 token embedder backed by a CoreML model.
///
/// Thread safety: per-call input tensors - safe for concurrent use from multiple tasks.
public class Qwen3CodeEmbedder: CodeEmbedding, @unchecked Sendable {
    public var model: MLModel?

    public init() {}

    public func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool = false) async throws {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits
        let loaded = try await MLModel.load(contentsOf: url, configuration: modelConfig)

        guard !prewarmMode else { return }

        self.model = loaded
    }

    public func embed(tokenId: Int32) async throws -> [FloatType] {
        guard let model else { throw TTSError.generationFailed("CodeEmbedder model not loaded") }
        let ids = try EmbedUtilities.makeInt32Array([tokenId])
        let provider = try MLDictionaryFeatureProvider(dictionary: ["input_ids": MLFeatureValue(multiArray: ids)])
        let output = try await model.asyncPrediction(from: provider)
        guard let embedArray = output.featureValue(for: "input_embeds")?.multiArrayValue else {
            throw TTSError.generationFailed("CodeEmbedder: missing input_embeds output")
        }
        return EmbedUtilities.extractEmbed(from: embedArray)
    }

    /// Optimised async path: passes `[String: MLTensor]` directly - no FeatureProvider boxing.
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public func embed(tokenId: Int32) async throws -> MLTensor {
        guard let model else { throw TTSError.generationFailed("CodeEmbedder model not loaded") }
        let outputs = try await model.prediction(from: [
            "input_ids": MLTensor(shape: [1], scalars: [tokenId])
        ])
        guard let embedTensor = outputs["input_embeds"] else {
            throw TTSError.generationFailed("CodeEmbedder: missing input_embeds tensor output")
        }
        return embedTensor
    }

    public func unloadModel() {
        model = nil
    }
}

// MARK: - Multi-Code Embedder Implementation

/// Codec-1..15 token embedder backed by a CoreML model.
///
/// Thread safety: same as Qwen3CodeEmbedder - per-call input tensors, safe for concurrent use.
public class Qwen3MultiCodeEmbedder: MultiCodeEmbedding, @unchecked Sendable {
    public var model: MLModel?

    public init() {}

    public func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool = false) async throws {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits
        let loaded = try await MLModel.load(contentsOf: url, configuration: modelConfig)

        guard !prewarmMode else { return }

        self.model = loaded
    }

    public func embed(tokenId: Int32) async throws -> [FloatType] {
        guard let model else { throw TTSError.generationFailed("MultiCodeEmbedder model not loaded") }
        let ids = try EmbedUtilities.makeInt32Array([tokenId])
        let provider = try MLDictionaryFeatureProvider(dictionary: ["input_ids": MLFeatureValue(multiArray: ids)])
        let output = try await model.asyncPrediction(from: provider)
        guard let embedArray = output.featureValue(for: "input_embeds")?.multiArrayValue else {
            throw TTSError.generationFailed("MultiCodeEmbedder: missing input_embeds output")
        }
        return EmbedUtilities.extractEmbed(from: embedArray)
    }

    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public func embed(tokenId: Int32) async throws -> MLTensor {
        guard let model else { throw TTSError.generationFailed("MultiCodeEmbedder model not loaded") }
        let outputs = try await model.prediction(from: [
            "input_ids": MLTensor(shape: [1], scalars: [tokenId])
        ])
        guard let embedTensor = outputs["input_embeds"] else {
            throw TTSError.generationFailed("MultiCodeEmbedder: missing input_embeds tensor output")
        }
        return embedTensor
    }

    public func unloadModel() {
        model = nil
    }
}
