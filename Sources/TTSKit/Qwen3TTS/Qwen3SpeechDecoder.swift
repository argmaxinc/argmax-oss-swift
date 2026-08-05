//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Accelerate
import ArgmaxCore
import CoreML
import Foundation

// MARK: - Implementation

/// RVQ-to-audio waveform decoder backed by a CoreML model.
///
/// Two asset layouts are supported, distinguished at load time by inspecting the
/// asset rather than by any variant string:
///
/// - **Multifunction** (`W8A16-multifunction`): bundles two graphs (`latency` and
///   `throughput`) that differ in how many RVQ frames they consume per call. The
///   choice is made via `MLModelConfiguration.functionName` and surfaced as
///   `codesPerStep` (1 for latency, 4 for throughput).
/// - **Single-function** (legacy `W8A16`): one graph, one frame per call, and a
///   rank-2 `kv_cache_update_mask` instead of the multifunction rank-3 mask.
///
/// Everything the host has to vary between the two — `codesPerStep`, the update-mask
/// rank, whether a `qk_mask` input is consumed, and the cache geometry — is read from
/// the loaded model's description, so a new asset layout needs no code change here.
///
/// Thread safety: mutable state (`model`, dimension properties) is set once during
/// `loadModel()` and read-only thereafter. `MLModel.prediction()` is thread-safe.
public class Qwen3SpeechDecoder: SpeechDecoding, @unchecked Sendable {
    @Protected public var model: MLModel?

    // MARK: - Audio format

    public let sampleRate: Int = Qwen3TTSConstants.sampleRate
    public let samplesPerFrame: Int = Qwen3TTSConstants.samplesPerFrame
    /// Minimum pre-buffer in seconds. Defaults to 0.08 (RVQ frame = 80ms @ 24kHz); at
    /// load time it becomes `0.08 * codesPerStep` — ~80 ms for latency, ~320 ms for
    /// throughput.
    public private(set) var minimumBufferDuration: TimeInterval = 0.08

    // MARK: - Mode (set by TTSKit before loadModel)

    /// Which function of the multifunction `.mlmodelc` to load. Ignored — beyond a
    /// compatibility check — when the asset is single-function.
    public var mode: Qwen3SpeechDecoderMode = .latencyOptimized

    /// `true` when the loaded asset exposed CoreML functions and one was selected.
    public private(set) var isMultifunction: Bool = false

    // MARK: - Detected from the loaded model

    /// Detected from model metadata at load time
    public private(set) var hiddenContextLen: Int = Qwen3TTSConstants.sdHiddenContextLen
    /// KV cache embedding dimension
    public private(set) var kvCacheEmbedDim: Int = Qwen3TTSConstants.sdCacheDim
    /// KV cache max sequence length
    public private(set) var kvCacheMaxSequenceLength: Int = Qwen3TTSConstants.sdMaxSeq
    /// Hidden state dimension
    public private(set) var hiddenDim: Int = Qwen3TTSConstants.sdHiddenDim
    /// Number of RVQ frames consumed per `decodeFrame{Async}` call. 1 for latency,
    /// 4 for throughput. Read from the loaded model's `audio_codes` input shape.
    public private(set) var codesPerStep: Int = 1
    /// `true` when the model's `kv_cache_update_mask` input is rank 3
    /// (`[1, codesPerStep, maxSeq]`, multifunction asset), `false` for the rank-2
    /// `[1, maxSeq]` mask of the legacy single-function asset. Read from the model.
    public private(set) var usesRank3UpdateMask: Bool = true

    public init() {}

    public func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool = false) async throws {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits
        // The tensor prediction path below is iOS 18+ / macOS 15+, as is CoreML
        // function selection, so both asset layouts share that floor.
        guard #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) else {
            throw TTSError.modelLoadingFailed(
                "SpeechDecoder requires macOS 15 / iOS 18"
            )
        }

        // Probe the asset for CoreML functions instead of assuming a layout: a
        // multifunction asset names its graphs, a legacy single-function asset
        // reports none and must be loaded without a `functionName`.
        let functionNames = await Self.functionNames(at: url)
        let selectedFunction: String?
        if functionNames.isEmpty {
            // Single-function asset. It only implements the 1-frame-per-call graph, so
            // a throughput request cannot be honored — surface that instead of silently
            // producing latency-mode audio under a throughput-mode config.
            guard mode == .latencyOptimized else {
                throw TTSError.invalidConfiguration(
                    "SpeechDecoder: \(url.lastPathComponent) is a single-function asset and " +
                    "only supports \(Qwen3SpeechDecoderMode.latencyOptimized.rawValue). Use the " +
                    "multifunction variant ('\(Qwen3VariantDefaults.speechDecoder)') for " +
                    "\(mode.rawValue)."
                )
            }
            selectedFunction = nil
        } else {
            guard functionNames.contains(mode.functionName) else {
                throw TTSError.modelLoadingFailed(
                    "SpeechDecoder: \(url.lastPathComponent) has no function " +
                    "'\(mode.functionName)' (available: \(functionNames.joined(separator: ", ")))."
                )
            }
            selectedFunction = mode.functionName
            modelConfig.functionName = mode.functionName
        }

        let loaded: MLModel
        do {
            loaded = try await MLModel.load(contentsOf: url, configuration: modelConfig)
        } catch {
            let functionSuffix = selectedFunction.map { " (function '\($0)')" } ?? ""
            throw TTSError.modelLoadingFailed(
                "SpeechDecoder: failed to load \(url.lastPathComponent)\(functionSuffix) on " +
                "macOS 15 / iOS 18 or newer. (\(error.localizedDescription))"
            )
        }

        // In prewarm mode, compilation is complete - discard to free memory before next model compiles
        guard !prewarmMode else { return }

        self.model = loaded
        self.isMultifunction = selectedFunction != nil

        // Read required dimensions from the model description. Every input below has a
        // fixed shape in a well-formed SpeechDecoder asset of either layout, so a missing
        // dimension means the asset is malformed — fail loudly instead of silently
        // keeping the defaults. In particular, silently defaulting `codesPerStep` to 1
        // would mis-drive a throughput-optimized model (consuming 1 of its 4 frames per
        // call) and produce corrupt audio.
        let assetName = url.lastPathComponent
        // audio_codes input shape: [1, 16, codesPerStep]
        self.codesPerStep = try Self.requireInputDimension(named: "audio_codes", position: 2, in: loaded, assetName: assetName)
        // hidden_context input shape: [1, hiddenDim, 1, hiddenContextLen]
        self.hiddenDim = try Self.requireInputDimension(named: "hidden_context", position: 1, in: loaded, assetName: assetName)
        self.hiddenContextLen = try Self.requireInputDimension(named: "hidden_context", position: 3, in: loaded, assetName: assetName)
        // key_cache input shape: [1, kvCacheEmbedDim, 1, kvCacheMaxSequenceLength]
        self.kvCacheEmbedDim = try Self.requireInputDimension(named: "key_cache", position: 1, in: loaded, assetName: assetName)
        self.kvCacheMaxSequenceLength = try Self.requireInputDimension(named: "key_cache", position: 3, in: loaded, assetName: assetName)

        // Update-mask layout: rank 3 `[1, codesPerStep, maxSeq]` (multifunction) or
        // rank 2 `[1, maxSeq]` (legacy single-function). The host allocates the mask,
        // so getting this wrong is a hard CoreML input-shape error on first prediction.
        guard let updateMaskShape = ModelUtilities.getModelInputShape(loaded, named: "kv_cache_update_mask") else {
            throw TTSError.modelLoadingFailed(
                "SpeechDecoder: missing expected input 'kv_cache_update_mask' in \(assetName)."
            )
        }
        switch updateMaskShape.count {
            case 2: self.usesRank3UpdateMask = false
            case 3: self.usesRank3UpdateMask = true
            default:
                throw TTSError.modelLoadingFailed(
                    "SpeechDecoder: unsupported 'kv_cache_update_mask' rank " +
                    "\(updateMaskShape.count) (shape \(updateMaskShape)) in \(assetName); " +
                    "expected rank 2 or 3."
                )
        }
        // A multi-frame graph cannot be driven by a single-column rank-2 mask.
        guard codesPerStep == 1 || usesRank3UpdateMask else {
            throw TTSError.modelLoadingFailed(
                "SpeechDecoder: \(assetName) consumes \(codesPerStep) frames per call but " +
                "declares a rank-2 'kv_cache_update_mask', which can only encode one position."
            )
        }

        // The `qk_mask` (intra-step causal triangle) is meaningful only for graphs that
        // consume more than one frame per call, so the host derives it from `codesPerStep`.
        // Check the asset agrees rather than trusting the invariant silently.
        let declaresQkMask = ModelUtilities.getModelInputShape(loaded, named: "qk_mask") != nil
        guard declaresQkMask == (codesPerStep > 1) else {
            throw TTSError.modelLoadingFailed(
                "SpeechDecoder: \(assetName) consumes \(codesPerStep) frames per call but " +
                "\(declaresQkMask ? "declares" : "does not declare") a 'qk_mask' input."
            )
        }

        // Scale streaming buffer to the audio output per call.
        self.minimumBufferDuration = 0.08 * TimeInterval(codesPerStep)
    }

    /// CoreML function names exposed by the asset at `url`, or `[]` when it is a
    /// single-function asset (or cannot be inspected — the subsequent `MLModel.load`
    /// reports the real failure with better context).
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    private static func functionNames(at url: URL) async -> [String] {
        do {
            return try await MLModelAsset(url: url).functionNames
        } catch {
            Logging.debug("SpeechDecoder: could not read function names from \(url.lastPathComponent): \(error)")
            return []
        }
    }

    /// Read a required fixed-shape input dimension from a loaded model, throwing a
    /// descriptive error if the named input or its dimension at `position` is absent.
    private static func requireInputDimension(
        named name: String, position: Int, in model: MLModel, assetName: String
    ) throws -> Int {
        guard let value = ModelUtilities.getModelInputDimension(model, named: name, position: position) else {
            throw TTSError.modelLoadingFailed(
                "SpeechDecoder: missing expected input dimension '\(name)'[\(position)] in " +
                "\(assetName). Expected a well-formed multifunction SpeechDecoder asset."
            )
        }
        return value
    }

    public func decodeFrame(
        codes: [[Int32]],
        cache: SpeechDecoderCache
    ) async throws -> [Float] {
        let result = try await decodeFrameAsync(codes: codes, cache: cache)
        return result.samples
    }

    /// Async multi-frame decode. Returns audio samples plus per-call timings.
    /// On macOS 15+/iOS 18+ uses `[String: MLTensor]` directly to avoid FeatureProvider boxing.
    public func decodeFrameAsync(
        codes: [[Int32]],
        cache: SpeechDecoderCache
    ) async throws -> SpeechDecoderTimedResult {
        guard let model else {
            throw TTSError.generationFailed("SpeechDecoder model not loaded")
        }
        guard codes.count == codesPerStep else {
            throw TTSError.generationFailed(
                "SpeechDecoder: expected \(codesPerStep) RVQ frames per call, got \(codes.count)"
            )
        }
        for (i, frame) in codes.enumerated() {
            guard frame.count == 16 else {
                throw TTSError.generationFailed(
                    "SpeechDecoder: frame \(i) has \(frame.count) codes, expected 16"
                )
            }
        }

        var timings = SpeechTimings()

        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            // Flatten codes into [1, 16, codesPerStep] with codes[frameIndex][codebookIndex]
            // at position (codebookIndex * codesPerStep + frameIndex) because the natural
            // layout for that shape is (..., codebookIndex, frameIndex) — the last axis
            // (codesPerStep) is the fastest-varying.
            var flat = [Int32](repeating: 0, count: 16 * codesPerStep)
            for codebookIndex in 0..<16 {
                let base = codebookIndex * codesPerStep
                for frameIndex in 0..<codesPerStep {
                    flat[base + frameIndex] = codes[frameIndex][codebookIndex]
                }
            }

            guard let keyCacheTensor = cache.keyCacheTensor,
                let valueCacheTensor = cache.valueCacheTensor
            else {
                throw TTSError.generationFailed("SpeechDecoder: KV cache tensors not initialized")
            }
            var inputs: [String: MLTensor] = [
                "audio_codes": MLTensor(shape: [1, 16, codesPerStep], scalars: flat),
                "cache_length": cache.cacheLengthTensor,
                "key_cache": keyCacheTensor,
                "value_cache": valueCacheTensor,
                "kv_cache_update_mask": cache.kvCacheUpdateMaskTensor,
                "key_padding_mask": cache.keyPaddingMaskTensor,
                "hidden_context": cache.hiddenContextTensor
            ]
            if codesPerStep > 1, let qkMaskTensor = cache.qkMaskTensor {
                inputs["qk_mask"] = qkMaskTensor
            }

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
            throw TTSError.generationFailed(
                "SpeechDecoder requires macOS 15 / iOS 18 (multifunction CoreML model)"
            )
        }
    }

    public func unloadModel() {
        model = nil
    }
}
