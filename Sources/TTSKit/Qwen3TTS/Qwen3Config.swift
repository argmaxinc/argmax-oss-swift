//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
import Foundation

// MARK: - Model Family

/// Identifies the TTS model architecture. Used by `setupPipeline`, `loadModels`,
/// and `setupGenerateTask` to dispatch to the correct model-specific code paths.
@frozen
public enum TTSModelFamily: String, Sendable {
    case qwen3
    // case kokoro  // future
}

// MARK: - Model Variants

/// Pre-configured TTS model sizes with matching variant defaults.
///
/// Pass a variant to `TTSKitConfig(model:)` to select the desired model size.
/// Mirrors `ModelVariant` in WhisperKit: `@frozen`, `CustomStringConvertible`, `CaseIterable`.
@frozen
public enum TTSModelVariant: String, CustomStringConvertible, CaseIterable, Sendable {
    case qwen3TTS_0_6b = "0.6b"
    case qwen3TTS_1_7b = "1.7b"

    /// The model architecture family this variant belongs to.
    public var family: TTSModelFamily {
        switch self {
            case .qwen3TTS_0_6b, .qwen3TTS_1_7b: return .qwen3
        }
    }

    public var description: String {
        switch self {
            case .qwen3TTS_0_6b: return "Qwen3-TTS-0.6B"
            case .qwen3TTS_1_7b: return "Qwen3-TTS-1.7B"
        }
    }

    /// Display name suitable for UI presentation.
    public var displayName: String {
        switch self {
            case .qwen3TTS_0_6b: return "Qwen3 TTS 0.6B"
            case .qwen3TTS_1_7b: return "Qwen3 TTS 1.7B"
        }
    }

    /// Whether this model supports the voice instruction prompt.
    /// Only the 1.7B variant has the capacity to follow style instructions.
    public var supportsVoiceDirection: Bool {
        switch self {
            case .qwen3TTS_0_6b: return false
            case .qwen3TTS_1_7b: return true
        }
    }

    /// Returns `true` when this variant can run on the current platform.
    ///
    /// The 1.7B model requires more peak memory during CoreML compilation than
    /// iOS/iPadOS devices can reliably provide, so it is restricted to macOS.
    public var isAvailableOnCurrentPlatform: Bool {
        #if os(macOS)
        return true
        #else
        switch self {
            case .qwen3TTS_0_6b: return true
            case .qwen3TTS_1_7b: return false
        }
        #endif
    }

    /// The best default variant for the current platform.
    public static var defaultForCurrentPlatform: TTSModelVariant {
        return .qwen3TTS_0_6b
    }

    public var versionDir: String {
        switch self {
            case .qwen3TTS_0_6b: return "12hz-0.6b-customvoice"
            case .qwen3TTS_1_7b: return "12hz-1.7b-customvoice"
        }
    }

    /// Recommended code decoder variant for this model size.
    public var codeDecoderVariant: String { Qwen3VariantDefaults.codeDecoder }
    /// Recommended multi-code decoder variant for this model size.
    public var multiCodeDecoderVariant: String { Qwen3VariantDefaults.multiCodeDecoder }
    /// Code embedder variant (same across model sizes).
    public var codeEmbedderVariant: String { Qwen3VariantDefaults.codeEmbedder }
    /// Multi-code embedder variant (same across model sizes).
    public var multiCodeEmbedderVariant: String { Qwen3VariantDefaults.multiCodeEmbedder }
    /// Text projector variant (same across model sizes).
    public var textProjectorVariant: String { Qwen3VariantDefaults.textProjector }
    /// Recommended speech decoder variant for this model size.
    public var speechDecoderVariant: String { Qwen3VariantDefaults.speechDecoder }
    /// HuggingFace repo used to load the tokenizer for this model size.
    public var tokenizerRepo: String { Qwen3TTSConstants.defaultTokenizerRepo }
}

// MARK: - Variant Defaults

/// Default quantization variant strings matching the standard model repository layout.
public enum Qwen3VariantDefaults {
    public static let codeDecoder = "W8A16-stateful"
    public static let multiCodeDecoder = "W8A16"
    public static let codeEmbedder = "W16A16"
    public static let multiCodeEmbedder = "W16A16"
    public static let textProjector = "W8A16"
    public static let speechDecoder = "W8A16"
}

// MARK: - TTSKit Configuration

/// Configuration for initializing a `TTSKit` instance.
///
/// Mirrors `WhisperKitConfig`: an `open class` so you can subclass for custom
/// configurations without modifying `TTSKit` itself.
///
/// **Minimal usage** (auto-downloads models and tokenizer):
/// ```swift
/// let tts = try await TTSKit()
/// ```
///
/// **Local models** (skip download):
/// ```swift
/// let config = TTSKitConfig(modelFolder: URL(fileURLWithPath: "/path/to/models"))
/// let tts = try await TTSKit(config)
/// ```
///
/// **Expected local directory layout:**
/// ```
/// modelFolder/
/// └── qwen3_tts/
///     ├── code_decoder/<versionDir>/<variant>/*.mlmodelc
///     ├── multi_code_decoder/<versionDir>/<variant>/*.mlmodelc
///     ├── code_embedder/<versionDir>/<variant>/*.mlmodelc
///     ├── multi_code_embedder/<versionDir>/<variant>/*.mlmodelc
///     ├── text_projector/<versionDir>/<variant>/*.mlmodelc
///     └── speech_decoder/<versionDir>/<variant>/*.mlmodelc
/// ```
open class TTSKitConfig {
    // MARK: - Model selection

    /// Model variant that determines default versionDir, component variants, and tokenizer.
    public var model: TTSModelVariant

    // MARK: - Model location

    /// Local URL to the model repository directory.
    /// If `nil`, models are downloaded from `modelRepo` on HuggingFace Hub.
    public var modelFolder: URL?

    /// Base URL for downloading and caching models.
    /// `nil` uses the Hub library's default cache directory.
    public var downloadBase: URL?

    /// HuggingFace repo ID for auto-downloading models.
    public var modelRepo: String

    // MARK: - Tokenizer

    /// HuggingFace repo ID or local folder path for the tokenizer
    /// (resolved from `model` by default).
    public var tokenizerFolder: URL?

    // MARK: - Authentication

    /// HuggingFace API token for private repos (or set the `HF_TOKEN` env var).
    public var modelToken: String?

    /// HuggingFace Hub endpoint URL.
    ///
    /// Override to point at a regional mirror or an on-premise Hub instance.
    /// Mirrors `WhisperKitConfig` (via `Constants.defaultRemoteEndpoint`).
    public var modelEndpoint: String

    // MARK: - Component variants

    /// Version directory shared across all components (resolved from `model` by default).
    public var versionDir: String

    /// Per-component quantization variant (resolved from `model` by default).
    public var codeDecoderVariant: String
    public var multiCodeDecoderVariant: String
    public var codeEmbedderVariant: String
    public var multiCodeEmbedderVariant: String
    public var textProjectorVariant: String
    public var speechDecoderVariant: String

    // MARK: - Compute

    /// Compute unit configuration per model component.
    public var computeOptions: ComputeOptions

    // MARK: - Logging

    /// Whether to emit diagnostic logs during loading and generation.
    public var verbose: Bool

    /// Logging level when `verbose` is `true`. Defaults to `.debug`.
    public var logLevel: Logging.LogLevel

    // MARK: - Download options

    /// Specific git revision (commit SHA, tag, or branch) to download from the Hub.
    /// `nil` (default) resolves to the repo's default branch head.
    /// Mirrors `WhisperKit.download(variant:revision:)`.
    public var downloadRevision: String?

    /// Additional glob patterns to include during model download, appended to the
    /// patterns generated from the configured component variants.
    public var downloadAdditionalPatterns: [String]

    /// Use a background `URLSession` for model downloads.
    /// Mirrors `WhisperKitConfig.useBackgroundDownloadSession`.
    public var useBackgroundDownloadSession: Bool

    /// Download models if not already available locally.
    /// When `true` (default), `loadModels()` will trigger a download if `modelFolder` is nil.
    public var download: Bool

    // MARK: - Lifecycle flags

    /// Enable model prewarming.
    ///
    /// Prewarming compiles each CoreML model sequentially then discards weights,
    /// minimizing peak memory during compilation. Call before `loadModels()` on first
    /// launch or after a model update. Mirrors `WhisperKitConfig.prewarm`.
    public var prewarm: Bool?

    /// Load models immediately after init.
    /// `nil` loads when `modelFolder` is non-nil, matching WhisperKit's default.
    public var load: Bool?

    // MARK: - Generation

    /// Optional seed for reproducible generation.
    /// Each concurrent task receives a derived seed (`seed ^ taskIndex`).
    public var seed: UInt64?

    // MARK: - Component overrides

    /// Set any of these to substitute a custom implementation for that model component.
    /// `nil` means TTSKit will use the default Qwen3 TTS class for that component.
    ///
    /// Example:
    /// ```swift
    /// let config = TTSKitConfig()
    /// config.codeDecoder = MyCodeDecoder()
    /// let tts = try await TTSKit(config)
    /// ```
    public var textProjector: (any TextProjecting)?
    public var codeEmbedder: (any CodeEmbedding)?
    public var multiCodeEmbedder: (any MultiCodeEmbedding)?
    public var codeDecoder: (any CodeDecoding)?
    public var multiCodeDecoder: (any MultiCodeDecoding)?
    public var speechDecoder: (any SpeechDecoding)?

    public init(
        model: TTSModelVariant = .qwen3TTS_0_6b,
        modelFolder: URL? = nil,
        downloadBase: URL? = nil,
        modelRepo: String = Qwen3TTSConstants.defaultModelRepo,
        tokenizerFolder: URL? = nil,
        modelToken: String? = nil,
        modelEndpoint: String = Qwen3TTSConstants.defaultEndpoint,
        versionDir: String? = nil,
        codeDecoderVariant: String? = nil,
        multiCodeDecoderVariant: String? = nil,
        codeEmbedderVariant: String? = nil,
        multiCodeEmbedderVariant: String? = nil,
        textProjectorVariant: String? = nil,
        speechDecoderVariant: String? = nil,
        computeOptions: ComputeOptions = ComputeOptions(),
        verbose: Bool = true,
        logLevel: Logging.LogLevel = .info,
        downloadRevision: String? = nil,
        downloadAdditionalPatterns: [String] = [],
        useBackgroundDownloadSession: Bool = false,
        download: Bool = true,
        prewarm: Bool? = nil,
        load: Bool? = nil,
        seed: UInt64? = nil
    ) {
        self.model = model
        self.modelFolder = modelFolder
        self.downloadBase = downloadBase
        self.modelRepo = modelRepo
        self.tokenizerFolder = tokenizerFolder
        self.modelToken = modelToken
        self.modelEndpoint = modelEndpoint
        self.versionDir = versionDir ?? model.versionDir
        self.codeDecoderVariant = codeDecoderVariant ?? model.codeDecoderVariant
        self.multiCodeDecoderVariant = multiCodeDecoderVariant ?? model.multiCodeDecoderVariant
        self.codeEmbedderVariant = codeEmbedderVariant ?? model.codeEmbedderVariant
        self.multiCodeEmbedderVariant = multiCodeEmbedderVariant ?? model.multiCodeEmbedderVariant
        self.textProjectorVariant = textProjectorVariant ?? model.textProjectorVariant
        self.speechDecoderVariant = speechDecoderVariant ?? model.speechDecoderVariant
        self.computeOptions = computeOptions
        self.verbose = verbose
        self.logLevel = logLevel
        self.downloadRevision = downloadRevision
        self.downloadAdditionalPatterns = downloadAdditionalPatterns
        self.useBackgroundDownloadSession = useBackgroundDownloadSession
        self.download = download
        self.prewarm = prewarm
        self.load = load
        self.seed = seed
    }

    // MARK: - Path resolution

    /// Resolve the full path to a component's model bundle.
    ///
    /// Requires `modelFolder` to be set (either directly or via download).
    /// Delegates to `ModelUtilities.detectModelURL(inFolder:)`, which prefers
    /// a compiled `.mlmodelc` bundle and falls back to `.mlpackage`.
    public func modelURL(component: String, variant: String) -> URL? {
        guard let modelFolder else { return nil }

        let variantDir = modelFolder.appending(path: Qwen3TTSConstants.modelFamilyDir)
            .appending(path: component).appending(path: versionDir)
            .appending(path: variant)

        return ModelUtilities.detectModelURL(inFolder: variantDir)
    }

    /// The effective tokenizer source: local `tokenizerFolder` if set, otherwise the
    /// model's default HuggingFace repo ID.
    public var tokenizerSource: String {
        tokenizerFolder?.path ?? model.tokenizerRepo
    }

    /// Component names in the model layout.
    public static let componentNames = [
        "text_projector", "code_embedder", "multi_code_embedder",
        "code_decoder", "multi_code_decoder", "speech_decoder"
    ]

    /// Version-specific directory for each component inside `modelFolder`.
    ///
    /// e.g. `modelFolder/qwen3_tts/code_decoder/12hz-0.6b-customvoice`
    ///
    /// Useful for targeted deletion or disk-size calculation of a single variant.
    public func componentDirectories(in folder: URL? = nil) -> [URL] {
        guard let base = folder ?? modelFolder else { return [] }
        return Self.componentNames.map { component in
            base
                .appending(path: Qwen3TTSConstants.modelFamilyDir)
                .appending(path: component)
                .appending(path: versionDir)
        }
    }

    /// Glob patterns used to download only the files needed for the configured variants.
    public var downloadPatterns: [String] {
        let components: [(String, String)] = [
            ("text_projector", textProjectorVariant),
            ("code_embedder", codeEmbedderVariant),
            ("multi_code_embedder", multiCodeEmbedderVariant),
            ("code_decoder", codeDecoderVariant),
            ("multi_code_decoder", multiCodeDecoderVariant),
            ("speech_decoder", speechDecoderVariant)
        ]
        return components.map {
            "\(Qwen3TTSConstants.modelFamilyDir)/\($0.0)/\(versionDir)/\($0.1)/**"
        }
    }
}
