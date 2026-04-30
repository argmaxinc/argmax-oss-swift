//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
import Foundation

// MARK: - Text Projector Protocol

/// Projects text token IDs into the shared embedding space.
public protocol TextProjecting: MLModelLoading {
    var model: MLModel? { get }
    func project(tokenId: Int32) async throws -> [FloatType]
}

// MARK: - Code Embedder Protocol

/// Embeds codec-0 tokens into the shared embedding space.
public protocol CodeEmbedding: MLModelLoading {
    var model: MLModel? { get }
    func embed(tokenId: Int32) async throws -> [FloatType]
}

// MARK: - Multi-Code Embedder Protocol

/// Embeds codec-1..15 tokens (with position offsets) into the shared embedding space.
public protocol MultiCodeEmbedding: MLModelLoading {
    var model: MLModel? { get }
    func embed(tokenId: Int32) async throws -> [FloatType]
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func embed(tokenId: Int32) async throws -> MLTensor
}

// MARK: - Code Decoder Protocol

/// Autoregressive decoder that generates codec-0 tokens from combined embeddings.
public protocol CodeDecoding: MLModelLoading {
    var model: MLModel? { get }
    var isStateful: Bool { get }
    var kvCacheEmbedDim: Int { get }
    var kvCacheMaxSequenceLength: Int { get }
    var embedSize: Int { get }
    func decode(inputEmbeds: any EmbedInputType, cache: KVCache, state: Any?) async throws -> CodeDecoderOutput
    func makeState() -> Any?
}

// MARK: - Code Decoder Output

public struct CodeDecoderOutput {
    /// Codec-0 logits - MLTensor (macOS 15+ async path) or MLMultiArray (sync path).
    public let logits: any EmbedTensorType // [1, 1, vocabSize]
    /// Hidden states - `[FloatType]` (legacy path) or MLTensor (async). Passed to MultiCodeDecoder.
    public let hiddenStates: any EmbedTensorType
    /// KV cache updates - populated by the sync path; nil when the async decoder updates the cache internally.
    public let keyCacheUpdates: MLMultiArray?
    public let valueCacheUpdates: MLMultiArray?
    /// Time spent on KV cache update inside the decoder (async path only). Lets callers
    /// subtract this from total decode time to isolate pure prediction cost.
    public var internalCacheUpdateTime: TimeInterval = 0
}

// MARK: - Multi-Code Decoder Output

public struct MultiCodeDecoderOutput {
    /// All-head logits - MLTensor (macOS 15+ async path) or MLMultiArray (sync path).
    public let allLogits: any EmbedTensorType // [1, 15, vocabSize]
    /// KV cache updates - populated by the sync path; nil when the async decoder updates the cache internally.
    public let keyCacheUpdates: MLMultiArray?
    public let valueCacheUpdates: MLMultiArray?
}

// MARK: - Multi-Code Decoder Protocol

/// Generates codec-1..15 tokens for a single RVQ frame given hidden states and codec-0 embedding.
public protocol MultiCodeDecoding: MLModelLoading {
    var model: MLModel? { get }
    var isStateful: Bool { get }

    // MARK: - Cache geometry (read after loadModel)

    var kvCacheEmbedDim: Int { get }
    var kvCacheMaxSequenceLength: Int { get }
    var codecVocabSize: Int { get }

    // MARK: - Decoding

    func decode(inputEmbeds: any EmbedInputType, cache: KVCache, state: Any?) async throws -> MultiCodeDecoderOutput
    func makeState() -> Any?

    /// Generate codes 1-15 for one RVQ frame.
    func generateMultiCodes(
        hiddenStates: [FloatType],
        code0Embed: [FloatType],
        multiCodeEmbedder: any MultiCodeEmbedding,
        sampler: any TokenSampling,
        options: GenerationOptions
    ) async throws -> MultiCodeGenerationResult
}

// MARK: - Speech Decoder Protocol

/// Decodes RVQ code frames into audio waveform samples.
public protocol SpeechDecoding: MLModelLoading {
    var model: MLModel? { get }

    // MARK: - Audio format (model-specific)

    /// Output sample rate in Hz (e.g. 24000 for Qwen3 TTS).
    var sampleRate: Int { get }
    /// Number of PCM samples produced per decoded RVQ frame (e.g. 1920 for Qwen3 TTS).
    var samplesPerFrame: Int { get }
    /// Minimum pre-buffer duration (seconds) in `.auto` playback mode.
    var minimumBufferDuration: TimeInterval { get }

    // MARK: - Cache geometry (read after loadModel)

    var kvCacheEmbedDim: Int { get }
    var kvCacheMaxSequenceLength: Int { get }
    var hiddenDim: Int { get }
    var hiddenContextLen: Int { get }

    // MARK: - Decoding

    /// Decode a single RVQ frame (16 codes) into audio samples.
    func decodeFrame(
        codes: [Int32],
        cache: SpeechDecoderCache
    ) async throws -> [Float]

    /// Async decode that returns audio samples with wall-clock timing.
    func decodeFrameAsync(
        codes: [Int32],
        cache: SpeechDecoderCache
    ) async throws -> SpeechDecoderTimedResult
}
