//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

@_exported import ArgmaxCore
import ArgmaxCore
import CoreML
import Foundation
import os

// MARK: - Callback Typealiases

// MARK: - TTSKit

/// Generic TTS orchestrator: text chunking, concurrent generation, crossfade, and audio playback.
///
/// Following the WhisperKit pattern, `TTSKit` exposes each model component as a
/// protocol-typed `public var`. Swap any component at runtime to change behaviour:
/// ```swift
/// let config = TTSKitConfig(load: false)
/// let tts = try await TTSKit(config)
/// tts.codeDecoder = MyOptimisedCodeDecoder()
/// try await tts.loadModels()
/// ```
///
/// The default implementation uses Qwen3 TTS components (`Sources/TTSKit/Qwen3TTS/`).
/// Components from entirely different model families can be plugged in by conforming
/// to the same component protocols, or by implementing `SpeechModel` directly.
///
/// `setupGenerateTask(...)` returns an `any SpeechGenerating` - override it to use a
/// completely different generation algorithm while keeping the chunking, concurrency,
/// crossfade, and playback orchestration provided by `generate` and `play`.
/// Mirrors `WhisperKit.setupTranscribeTask(...)`.
open class TTSKit: @unchecked Sendable {
    // MARK: - Model components (protocol-typed, swappable)

    /// Text token -> embedding. Conforms to `TextProjecting`.
    public var textProjector: any TextProjecting = Qwen3TextProjector()
    /// Codec-0 token -> embedding. Conforms to `CodeEmbedding`.
    public var codeEmbedder: any CodeEmbedding = Qwen3CodeEmbedder()
    /// Multi-code token -> embedding. Conforms to `MultiCodeEmbedding`.
    public var multiCodeEmbedder: any MultiCodeEmbedding = Qwen3MultiCodeEmbedder()
    /// Autoregressive code-0 decoder. Conforms to `CodeDecoding`.
    public var codeDecoder: any CodeDecoding = Qwen3CodeDecoder()
    /// Per-frame decoder. Conforms to `MultiCodeDecoding`.
    public var multiCodeDecoder: any MultiCodeDecoding = Qwen3MultiCodeDecoder()
    /// RVQ codes -> audio waveform. Conforms to `SpeechDecoding`.
    public var speechDecoder: any SpeechDecoding = Qwen3SpeechDecoder()
    /// Tokenizer. `nil` before the first `loadModels()` call or after `unloadModels()`.
    public var tokenizer: (any TTSTokenizer)?

    // MARK: - Model state

    /// Current lifecycle state of the loaded models.
    /// Mirrors `WhisperKit.modelState`. Transitions:
    /// `.unloaded` -> `.downloading` -> `.downloaded` -> `.loading` -> `.loaded`
    /// `.unloaded` -> `.prewarming` -> `.prewarmed`
    public private(set) var modelState: ModelState = .unloaded {
        didSet { modelStateCallback?(oldValue, modelState) }
    }

    // MARK: - Configuration & timing

    public var config: TTSKitConfig

    /// Direct accessor for the resolved local model folder.
    ///
    /// Mirrors `WhisperKit.modelFolder`. Backed by `config.modelFolder`; set by
    /// `setupModels()` and may also be assigned directly for offline usage.
    public var modelFolder: URL? {
        get { config.modelFolder }
        set { config.modelFolder = newValue }
    }

    /// Whether to use a background `URLSession` for model downloads.
    ///
    /// Mirrors `WhisperKit.useBackgroundDownloadSession`. Backed by
    /// `config.useBackgroundDownloadSession`.
    public var useBackgroundDownloadSession: Bool {
        get { config.useBackgroundDownloadSession }
        set { config.useBackgroundDownloadSession = newValue }
    }

    /// Cumulative timings for the most recent pipeline run.
    /// `modelLoading` and `tokenizerLoadTime` are populated after `loadModels()`.
    public private(set) var currentTimings = SpeechTimings()

    /// Wall-clock seconds for the most recent full model load.
    public var modelLoadTime: TimeInterval { currentTimings.modelLoading }
    /// Wall-clock seconds for the most recent tokenizer load.
    public var tokenizerLoadTime: TimeInterval { currentTimings.tokenizerLoadTime }

    // MARK: - Audio output

    /// Audio output used by `play`.
    /// `AudioOutput` is playback-only; WhisperKit's `AudioProcessor` is capture-only.
    /// They serve complementary roles and do not need to be merged.
    public let audioOutput = AudioOutput()

    // MARK: - Prompt cache

    /// Cached prefix state for the most recently used voice/language/instruction.
    /// Automatically built on the first `generate` call and reused for subsequent
    /// calls with the same parameters. Set to `nil` to force a full prefill.
    public var promptCache: TTSPromptCache?

    // MARK: - Callbacks

    /// Invoked whenever `modelState` changes.
    public var modelStateCallback: ModelStateCallback?

    // MARK: - Seed

    public let seed: UInt64?
    private var taskCounter: UInt64 = 0

    // MARK: - Initialization

    /// Create a `TTSKit` instance from a `TTSKitConfig`.
    ///
    /// Uses the component overrides in `config` if set; otherwise instantiates the default
    /// components for the selected model family. Components can also be replaced after init.
    ///
    /// - Parameter config: Pipeline configuration (model variant, paths, compute units,
    ///   component overrides, lifecycle flags).
    /// - Throws: `TTSError` if the model family is unsupported or component instantiation fails.
    public init(_ config: TTSKitConfig = TTSKitConfig()) async throws {
        self.config = config
        self.seed = config.seed

        Logging.updateLogLevel(config.verbose ? config.logLevel : .none)

        setupPipeline(for: config.model, config: config)

        // Resolve or download the model folder so that the load condition below
        // can use `config.modelFolder != nil` as its auto-load sentinel.
        try await setupModels(
            model: config.model,
            downloadBase: config.downloadBase,
            modelRepo: config.modelRepo,
            modelToken: config.modelToken,
            modelFolder: config.modelFolder,
            download: config.download,
            endpoint: config.modelEndpoint
        )

        if let prewarm = config.prewarm, prewarm {
            Logging.info("Prewarming models...")
            try await prewarmModels()
        }

        // Load if explicitly requested, or if a local folder is now available
        // (either provided directly or populated by setupModels above).
        if config.load ?? (config.modelFolder != nil) {
            Logging.info("Loading models...")
            try await loadModels()
        }
    }

    /// Convenience initializer that exposes all configuration fields as individual parameters.
    ///
    /// Mirrors `WhisperKit.init(model:modelFolder:...)`. Constructs a `TTSKitConfig` and
    /// delegates to `init(_ config:)`.
    public convenience init(
        model: TTSModelVariant = .qwen3TTS_0_6b,
        modelFolder: URL? = nil,
        downloadBase: URL? = nil,
        modelRepo: String = Qwen3TTSConstants.defaultModelRepo,
        tokenizerFolder: URL? = nil,
        modelToken: String? = nil,
        computeOptions: ComputeOptions? = nil,
        textProjector: (any TextProjecting)? = nil,
        codeEmbedder: (any CodeEmbedding)? = nil,
        multiCodeEmbedder: (any MultiCodeEmbedding)? = nil,
        codeDecoder: (any CodeDecoding)? = nil,
        multiCodeDecoder: (any MultiCodeDecoding)? = nil,
        speechDecoder: (any SpeechDecoding)? = nil,
        verbose: Bool = false,
        logLevel: Logging.LogLevel = .debug,
        prewarm: Bool? = nil,
        load: Bool? = nil,
        download: Bool = true,
        useBackgroundDownloadSession: Bool = false,
        seed: UInt64? = nil
    ) async throws {
        let config = TTSKitConfig(
            model: model,
            modelFolder: modelFolder,
            downloadBase: downloadBase,
            modelRepo: modelRepo,
            tokenizerFolder: tokenizerFolder,
            modelToken: modelToken,
            computeOptions: computeOptions ?? ComputeOptions(),
            verbose: verbose,
            logLevel: logLevel,
            useBackgroundDownloadSession: useBackgroundDownloadSession,
            download: download,
            prewarm: prewarm,
            load: load,
            seed: seed
        )
        config.textProjector = textProjector
        config.codeEmbedder = codeEmbedder
        config.multiCodeEmbedder = multiCodeEmbedder
        config.codeDecoder = codeDecoder
        config.multiCodeDecoder = multiCodeDecoder
        config.speechDecoder = speechDecoder
        try await self.init(config)
    }

    // MARK: - Pipeline setup

    /// Configure the model-specific component properties for the active model family.
    ///
    /// Uses the component overrides in `config` if set; otherwise instantiates the
    /// default components for the given variant's model family. Called from `init` and
    /// can be called again to reconfigure the pipeline for a different variant.
    ///
    /// Mirrors how WhisperKit configures its encoder/decoder components based on the
    /// selected model.
    open func setupPipeline(for variant: TTSModelVariant, config: TTSKitConfig) {
        switch variant.family {
            case .qwen3:
                self.textProjector = config.textProjector ?? Qwen3TextProjector()
                self.codeEmbedder = config.codeEmbedder ?? Qwen3CodeEmbedder()
                self.multiCodeEmbedder = config.multiCodeEmbedder ?? Qwen3MultiCodeEmbedder()
                self.codeDecoder = config.codeDecoder ?? Qwen3CodeDecoder()
                self.multiCodeDecoder = config.multiCodeDecoder ?? Qwen3MultiCodeDecoder()
                self.speechDecoder = config.speechDecoder ?? Qwen3SpeechDecoder()
        }
    }

    // MARK: - Model Discovery

    /// Returns the recommended model variant for the current platform.
    ///
    /// Mirrors `WhisperKit.recommendedModels()`.
    public static func recommendedModels() -> TTSModelVariant {
        return TTSModelVariant.defaultForCurrentPlatform
    }

    /// Fetch all available model variants from the HuggingFace Hub.
    ///
    /// Mirrors `WhisperKit.fetchAvailableModels(from:matching:downloadBase:token:)`.
    ///
    /// - Parameters:
    ///   - repo: HuggingFace repo ID to query. Defaults to the standard Qwen3 TTS repo.
    ///   - matching: Glob patterns to filter returned variant names. Defaults to `["*"]` (all variants).
    ///   - downloadBase: Optional base URL for the Hub cache when listing files; `nil` uses the Hub default.
    ///   - token: HuggingFace API token (or set `HF_TOKEN` env var).
    ///   - endpoint: HuggingFace Hub endpoint URL.
    /// - Returns: Display names of available model variants matching the given patterns.
    /// - Throws: `TTSError` if the Hub request fails.
    public static func fetchAvailableModels(
        from repo: String = Qwen3TTSConstants.defaultModelRepo,
        matching: [String] = ["*"],
        downloadBase: URL? = nil,
        token: String? = nil,
        endpoint: String = Qwen3TTSConstants.defaultEndpoint
    ) async throws -> [String] {
        let downloader = ModelDownloader(endpoint: endpoint, repoName: repo, modelToken: token)
        let files = try await downloader.fetchFilenames(matching: ["qwen3_tts/**"], downloadBase: downloadBase)
        var variants: [String] = []
        for variant in TTSModelVariant.allCases {
            let prefix = "qwen3_tts/code_decoder/\(variant.versionDir)/"
            if files.contains(where: { $0.hasPrefix(prefix) }) {
                variants.append(variant.displayName)
            }
        }
        let allVariants = variants.isEmpty ? TTSModelVariant.allCases.map(\.displayName) : variants
        var filteredVariants: Set<String> = []
        for glob in matching {
            filteredVariants = filteredVariants.union(allVariants.matching(glob: glob))
        }
        return Array(filteredVariants)
    }

    // MARK: - Download

    /// Download models for a specific variant from HuggingFace Hub.
    ///
    /// Mirrors `WhisperKit.download(variant:downloadBase:from:token:progressCallback:)`.
    ///
    /// - Parameters:
    ///   - variant: The model variant to download.
    ///   - downloadBase: Base URL for the local cache. Defaults to the Hub library default.
    ///   - useBackgroundSession: Use a background `URLSession` for the download.
    ///   - repo: HuggingFace repo ID. Defaults to the standard Qwen3 TTS repo.
    ///   - token: HuggingFace API token (or set `HF_TOKEN` env var).
    ///   - endpoint: HuggingFace Hub endpoint URL.
    ///   - revision: Specific git revision (commit SHA, tag, or branch) to download.
    ///   - additionalPatterns: Extra glob patterns to include alongside the default component patterns.
    ///   - progressCallback: Optional closure receiving `Progress` updates; `progress.fractionCompleted` is in [0, 1].
    /// - Returns: Local URL of the downloaded model folder.
    /// - Throws: `TTSError` if the Hub download fails.
    open class func download(
        variant: TTSModelVariant = .defaultForCurrentPlatform,
        downloadBase: URL? = nil,
        useBackgroundSession: Bool = false,
        from repo: String = Qwen3TTSConstants.defaultModelRepo,
        token: String? = nil,
        endpoint: String = Qwen3TTSConstants.defaultEndpoint,
        revision: String? = nil,
        additionalPatterns: [String] = [],
        progressCallback: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        let config = TTSKitConfig(
            model: variant,
            downloadBase: downloadBase,
            modelRepo: repo,
            modelToken: token,
            modelEndpoint: endpoint,
            downloadRevision: revision,
            downloadAdditionalPatterns: additionalPatterns,
            useBackgroundDownloadSession: useBackgroundSession
        )
        return try await download(config: config, progressCallback: progressCallback)
    }

    /// Download models using a full `TTSKitConfig`.
    ///
    /// Downloads only the files matching the configured component variants at the
    /// config’s `downloadRevision`. Files are cached locally by the Hub library.
    ///
    /// - Parameters:
    ///   - config: Pipeline configuration including `modelRepo`, `modelToken`, `downloadBase`,
    ///     `downloadRevision`, `downloadAdditionalPatterns`, `useBackgroundDownloadSession`, and variant settings.
    ///   - progressCallback: Optional closure receiving `Progress` updates; `progress.fractionCompleted` is in [0, 1].
    /// - Returns: Local URL of the downloaded model folder.
    /// - Throws: `TTSError` if the Hub download fails.
    open class func download(
        config: TTSKitConfig = TTSKitConfig(),
        progressCallback: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        let downloader = ModelDownloader(
            endpoint: config.modelEndpoint,
            repoName: config.modelRepo,
            modelToken: config.modelToken,
            revision: config.downloadRevision ?? "main",
            useBackgroundSession: config.useBackgroundDownloadSession
        )
        let patterns = config.downloadPatterns + config.downloadAdditionalPatterns

        do {
            return try await downloader.resolveRepo(
                patterns: patterns,
                downloadBase: config.downloadBase,
                download: true,
                progressCallback: progressCallback
            )
        } catch {
            throw TTSError.generationFailed(
                "Failed to download models from \(config.modelRepo). Check that the repo exists and you have access. Error: \(error.localizedDescription)"
            )
        }
    }

    // MARK: - Model lifecycle

    /// Resolve the local model folder, downloading from HuggingFace Hub if needed.
    ///
    /// Mirrors `WhisperKit.setupModels(model:downloadBase:modelRepo:...)`. Populates
    /// `config.modelFolder` so that `loadModels()` can be called immediately after.
    /// Separated from `loadModels()` so callers can call setup once and load separately,
    /// or override just the resolution logic without touching the load path.
    ///
    /// - Parameters:
    ///   - model: Model variant to download. `nil` uses `config.model`.
    ///   - downloadBase: Base URL for Hub cache. `nil` uses the Hub library default.
    ///   - modelRepo: HuggingFace repo ID. `nil` uses `config.modelRepo`.
    ///   - modelToken: HuggingFace API token. `nil` uses `config.modelToken`.
    ///   - modelFolder: Explicit local folder URL. When non-nil the download is skipped.
    ///   - download: When `true` and `modelFolder` is nil, download from the resolved repo.
    ///   - endpoint: HuggingFace Hub endpoint URL. Defaults to `Qwen3TTSConstants.defaultEndpoint`.
    /// - Throws: `TTSError` if the download fails or the model folder cannot be resolved.
    open func setupModels(
        model: TTSModelVariant? = nil,
        downloadBase: URL? = nil,
        modelRepo: String? = nil,
        modelToken: String? = nil,
        modelFolder: URL? = nil,
        download: Bool,
        endpoint: String = Qwen3TTSConstants.defaultEndpoint
    ) async throws {
        if let folder = modelFolder {
            config.modelFolder = folder
        } else if download {
            let resolvedModel = model ?? config.model
            let resolvedRepo = modelRepo ?? config.modelRepo
            let resolvedToken = modelToken ?? config.modelToken

            let downloadConfig = TTSKitConfig(
                model: resolvedModel,
                downloadBase: downloadBase ?? config.downloadBase,
                modelRepo: resolvedRepo,
                modelToken: resolvedToken,
                modelEndpoint: endpoint,
                downloadRevision: config.downloadRevision,
                downloadAdditionalPatterns: config.downloadAdditionalPatterns,
                useBackgroundDownloadSession: config.useBackgroundDownloadSession
            )

            modelState = .downloading
            Logging.info("Downloading models from \(resolvedRepo)...")
            do {
                let folder = try await TTSKit.download(config: downloadConfig) { progress in
                    let percent = Int(progress.fractionCompleted * 100)
                    Logging.debug("  Download: \(percent)%")
                }
                config.modelFolder = folder
                modelState = .downloaded
                Logging.info("Models cached at \(folder.path)")
            } catch {
                modelState = .unloaded
                throw TTSError.modelNotFound(
                    "Model download failed. Check the repo name and try again. Error: \(error)"
                )
            }
        }
    }

    /// Prewarm all CoreML models by compiling them sequentially, then discarding weights.
    ///
    /// Serializes CoreML compilation to cap peak memory. Call before `loadModels()` on
    /// first launch or after a model update. Mirrors `WhisperKit.prewarmModels()`.
    open func prewarmModels() async throws {
        try await loadModels(prewarmMode: true)
    }

    /// Load all models and the tokenizer.
    ///
    /// Expects `config.modelFolder` to be set (call `setupModels` first if needed).
    /// Mirrors `WhisperKit.loadModels(prewarmMode:)`.
    ///
    /// - Parameter prewarmMode: When `true`, compile models one at a time and discard weights
    ///   to limit peak memory (prewarm). When `false` (default), load all concurrently.
    /// - Throws: `TTSError` if model compilation or tokenizer loading fails.
    open func loadModels(prewarmMode: Bool = false) async throws {
        modelState = prewarmMode ? .prewarming : .loading

        let embedUnits = config.computeOptions.embedderComputeUnits
        let cdUnits = config.computeOptions.codeDecoderComputeUnits
        let mcdUnits = config.computeOptions.multiCodeDecoderComputeUnits
        let sdUnits = config.computeOptions.speechDecoderComputeUnits

        guard let modelFolder = config.modelFolder,
            FileManager.default.fileExists(atPath: modelFolder.path)
        else {
            modelState = .unloaded
            throw TTSError.modelNotFound(config.modelFolder?.path ?? "<nil>")
        }

        // Resolve all six component URLs. A nil result means the .mlmodelc bundle is
        // missing from disk - surface this immediately rather than crashing later.
        func requireURL(_ component: String, _ variant: String) throws -> URL {
            guard let url = config.modelURL(component: component, variant: variant) else {
                throw TTSError.invalidConfiguration(
                    "No .mlmodelc found at \(component)/\(config.versionDir)/\(variant) inside \(modelFolder.path)."
                )
            }
            return url
        }
        let tpURL = try requireURL("text_projector", config.textProjectorVariant)
        let ceURL = try requireURL("code_embedder", config.codeEmbedderVariant)
        let mceURL = try requireURL("multi_code_embedder", config.multiCodeEmbedderVariant)
        let cdURL = try requireURL("code_decoder", config.codeDecoderVariant)
        let mcdURL = try requireURL("multi_code_decoder", config.multiCodeDecoderVariant)
        let sdURL = try requireURL("speech_decoder", config.speechDecoderVariant)

        // Load tokenizer (skipped in prewarm - only CoreML compilation needed).
        if !prewarmMode {
            try await loadTokenizerIfNeeded()
        }

        // Load the six CoreML models.
        // Prewarm: sequential to serialize compilation -> lower peak memory.
        // Normal: concurrent since compiled artifacts are already cached.
        let modelLoadStart = CFAbsoluteTimeGetCurrent()

        if prewarmMode {
            Logging.info("Prewarming 6 CoreML models sequentially (serializing compilation)...")
            try await textProjector.loadModel(at: tpURL, computeUnits: embedUnits, prewarmMode: true)
            try await codeEmbedder.loadModel(at: ceURL, computeUnits: embedUnits, prewarmMode: true)
            try await multiCodeEmbedder.loadModel(at: mceURL, computeUnits: embedUnits, prewarmMode: true)
            try await codeDecoder.loadModel(at: cdURL, computeUnits: cdUnits, prewarmMode: true)
            try await multiCodeDecoder.loadModel(at: mcdURL, computeUnits: mcdUnits, prewarmMode: true)
            try await speechDecoder.loadModel(at: sdURL, computeUnits: sdUnits, prewarmMode: true)
            Logging.info(String(format: "Prewarm complete in %.2fs", CFAbsoluteTimeGetCurrent() - modelLoadStart))
            modelState = .prewarmed
        } else {
            Logging.info("Loading 6 CoreML models concurrently...")
            Logging.debug("  TextProjector:     \(tpURL.lastPathComponent)  compute: \(embedUnits.description)")
            Logging.debug("  CodeEmbedder:      \(ceURL.lastPathComponent)  compute: \(embedUnits.description)")
            Logging.debug("  MultiCodeEmbedder: \(mceURL.lastPathComponent) compute: \(embedUnits.description)")
            Logging.debug("  CodeDecoder:       \(cdURL.lastPathComponent)  (\(config.codeDecoderVariant), compute: \(cdUnits.description))")
            Logging.debug("  MultiCodeDecoder:  \(mcdURL.lastPathComponent) (\(config.multiCodeDecoderVariant), compute: \(mcdUnits.description))")
            Logging.debug("  SpeechDecoder:     \(sdURL.lastPathComponent)  (\(config.speechDecoderVariant), compute: \(sdUnits.description))")

            async let loadTP: Void = textProjector.loadModel(at: tpURL, computeUnits: embedUnits)
            async let loadCE: Void = codeEmbedder.loadModel(at: ceURL, computeUnits: embedUnits)
            async let loadMCE: Void = multiCodeEmbedder.loadModel(at: mceURL, computeUnits: embedUnits)
            async let loadCD: Void = codeDecoder.loadModel(at: cdURL, computeUnits: cdUnits)
            async let loadMCD: Void = multiCodeDecoder.loadModel(at: mcdURL, computeUnits: mcdUnits)
            async let loadSD: Void = speechDecoder.loadModel(at: sdURL, computeUnits: sdUnits)
            _ = try await (loadTP, loadCE, loadMCE, loadCD, loadMCD, loadSD)

            currentTimings.modelLoading = CFAbsoluteTimeGetCurrent() - modelLoadStart

            // Sync audio output sample rate to the loaded speech decoder.
            audioOutput.configure(sampleRate: speechDecoder.sampleRate)

            Logging.info(String(format: "Total model load: %.2fs", modelLoadTime))
            modelState = .loaded
        }
    }

    /// Load the tokenizer only if it has not been loaded yet.
    ///
    /// Mirrors `WhisperKit.loadTokenizerIfNeeded()`. Skips loading when `tokenizer` is
    /// already set, avoiding redundant network calls or file-system work on repeated
    /// `loadModels()` calls.
    open func loadTokenizerIfNeeded() async throws {
        guard tokenizer == nil else {
            Logging.debug("Tokenizer already loaded, skipping")
            return
        }
        self.tokenizer = try await loadTokenizer()
    }

    /// Load the tokenizer from `config.tokenizerSource`.
    ///
    /// Checks for a local `tokenizer.json` file first; falls back to downloading from
    /// the Hugging Face Hub if no local file is found. Updates `currentTimings.tokenizerLoadTime`.
    ///
    /// Override this method to plug in a custom tokenizer loading strategy (e.g. fully
    /// offline from a bundled path) without touching the rest of `loadModels()`.
    open func loadTokenizer() async throws -> any TTSTokenizer {
        let start = CFAbsoluteTimeGetCurrent()
        Logging.info("Loading tokenizer from \(config.tokenizerSource)...")
        let tokenizerURL = URL(fileURLWithPath: config.tokenizerSource)
        let wrapper: TokenizerWrapper
        if FileManager.default.fileExists(atPath: tokenizerURL.appending(path: "tokenizer.json").path) {
            wrapper = try await AutoTokenizerWrapper.from(modelFolder: tokenizerURL)
        } else {
            wrapper = try await AutoTokenizerWrapper.from(pretrained: config.tokenizerSource)
        }
        currentTimings.tokenizerLoadTime = CFAbsoluteTimeGetCurrent() - start
        Logging.info(String(format: "Tokenizer loaded in %.2fs", tokenizerLoadTime))
        return TTSTokenizerWrapper(wrapper)
    }

    /// Release all model weights and the tokenizer from memory.
    ///
    /// Mirrors `WhisperKit.unloadModels()`. Transitions through `.unloading` before
    /// reaching `.unloaded` so observers can distinguish the in-progress state.
    open func unloadModels() async {
        modelState = .unloading
        textProjector.unloadModel()
        codeEmbedder.unloadModel()
        multiCodeEmbedder.unloadModel()
        codeDecoder.unloadModel()
        multiCodeDecoder.unloadModel()
        speechDecoder.unloadModel()
        tokenizer = nil
        modelState = .unloaded
        Logging.info("Unloaded all models")
    }

    /// Reset all accumulated timing statistics.
    ///
    /// Mirrors `WhisperKit.clearState()`. Call between generation runs when you want
    /// fresh per-run timing data without triggering a full reload.
    open func clearState() {
        currentTimings = SpeechTimings()
    }

    deinit {
        Task { [audioOutput] in
            await audioOutput.stopPlayback(waitForCompletion: false)
        }
    }

    /// Register a custom log sink for all `Logging` output from TTSKit.
    ///
    /// Mirrors `WhisperKit.loggingCallback(_:)`. Pass `nil` to restore the default
    /// print-based logger.
    open func loggingCallback(_ callback: Logging.LoggingCallback?) {
        Logging.updateCallback(callback)
    }

    // MARK: - Prompt cache management

    /// Build a prompt cache for the given voice/language/instruction combination.
    ///
    /// Pre-computes the invariant prefix embeddings and prefills them through the
    /// CodeDecoder, returning a reusable cache that eliminates ~90% of prefill cost
    /// on subsequent `generate` calls.
    ///
    /// The cache is stored on `self.promptCache` for automatic reuse. Delegates to
    /// `Qwen3GenerateTask.buildPromptCache` on the task returned by
    /// `setupGenerateTask(...)`, so Qwen3 models get prompt caching automatically.
    ///
    /// - Parameters:
    ///   - voice: Voice/speaker identifier. `nil` uses the model's `defaultVoice`.
    ///   - language: Language identifier. `nil` uses the model's `defaultLanguage`.
    ///   - instruction: Optional style instruction prepended to the TTS prompt.
    /// - Returns: The built `TTSPromptCache` that can be passed to subsequent `generate` calls.
    /// - Throws: `TTSError` if the model is not loaded or prompt caching is unsupported.
    @discardableResult
    open func buildPromptCache(
        voice: String? = nil,
        language: String? = nil,
        instruction: String? = nil
    ) async throws -> TTSPromptCache {
        let task = try createTask()
        let resolvedVoice = voice ?? task.defaultVoice
        let resolvedLanguage = language ?? task.defaultLanguage
        guard let qwen3Task = task as? Qwen3GenerateTask else {
            throw TTSError.generationFailed("Prompt caching is not supported by this model family.")
        }
        let cache = try await qwen3Task.buildPromptCache(voice: resolvedVoice, language: resolvedLanguage, instruction: instruction)
        self.promptCache = cache
        return cache
    }

    /// Save the current prompt cache to disk under the model's embeddings directory.
    ///
    /// The file is saved at `<modelFolder>/embeddings/<voice>_<language>.promptcache`.
    public func savePromptCache() throws {
        guard let cache = promptCache else { return }
        guard let url = promptCacheURL(for: cache) else {
            throw TTSError.generationFailed("Cannot determine prompt cache path (modelFolder not set)")
        }
        try cache.save(to: url)
        Logging.info("Saved prompt cache to \(url.path)")
    }

    /// Load a prompt cache from disk if one exists for the given parameters.
    ///
    /// Returns `nil` if no cached file exists. Also stores the loaded cache
    /// on `self.promptCache` for automatic reuse.
    ///
    /// - Parameters:
    ///   - voice: Voice/speaker identifier.
    ///   - language: Language identifier.
    ///   - instruction: Optional style instruction.
    /// - Returns: The loaded cache, or `nil` if not found.
    @discardableResult
    public func loadPromptCache(
        voice: String,
        language: String,
        instruction: String? = nil
    ) -> TTSPromptCache? {
        let probe = TTSPromptCache(
            voice: voice, language: language, instruction: instruction,
            prefixLength: 0,
            kvSnapshot: KVCacheSnapshot(
                isStateful: false, cacheDim: 0, maxSeqLength: 0, cacheLength: 0,
                keyCacheData: Data(), valueCacheData: Data(),
                updateMaskData: Data(), paddingMaskData: Data()
            ),
            stateData: nil
        )
        guard let url = promptCacheURL(for: probe),
            FileManager.default.fileExists(atPath: url.path)
        else { return nil }
        do {
            let cache = try TTSPromptCache.load(from: url)
            self.promptCache = cache
            Logging.info("Loaded prompt cache from \(url.path)")
            return cache
        } catch {
            Logging.error("Failed to load prompt cache: \(error)")
            return nil
        }
    }

    private func promptCacheURL(for cache: TTSPromptCache) -> URL? {
        guard let modelFolder = config.modelFolder else { return nil }
        return
            modelFolder
            .appendingPathComponent("embeddings")
            .appendingPathComponent(cache.cacheFileName)
    }

    // MARK: - Task factory

    /// Setup the generate task used for speech synthesis.
    /// Subclasses may override to provide custom behavior.
    ///
    /// Mirrors `WhisperKit.setupTranscribeTask(...)`. Model-agnostic params are passed
    /// explicitly; model-specific components are accessed from `self` (configured by
    /// `setupPipeline`).
    open func setupGenerateTask(
        currentTimings: SpeechTimings,
        progress: Progress,
        tokenizer: any TTSTokenizer,
        sampler: any TokenSampling
    ) throws -> any SpeechGenerating {
        switch config.model.family {
            case .qwen3:
                guard let qwen3TextProjector = textProjector as? Qwen3TextProjector,
                    let qwen3CodeEmbedder = codeEmbedder as? Qwen3CodeEmbedder,
                    let qwen3MultiCodeEmbedder = multiCodeEmbedder as? Qwen3MultiCodeEmbedder,
                    let qwen3CodeDecoder = codeDecoder as? Qwen3CodeDecoder,
                    let qwen3MultiCodeDecoder = multiCodeDecoder as? Qwen3MultiCodeDecoder,
                    let qwen3SpeechDecoder = speechDecoder as? Qwen3SpeechDecoder
                else {
                    throw TTSError.generationFailed("Qwen3 model family requires Qwen3-specific model components")
                }
                return Qwen3GenerateTask(
                    textProjector: qwen3TextProjector,
                    codeEmbedder: qwen3CodeEmbedder,
                    multiCodeEmbedder: qwen3MultiCodeEmbedder,
                    codeDecoder: qwen3CodeDecoder,
                    multiCodeDecoder: qwen3MultiCodeDecoder,
                    speechDecoder: qwen3SpeechDecoder,
                    sampler: sampler,
                    tokenizer: tokenizer,
                    suppressTokenIds: Qwen3TTSConstants.suppressTokenIds,
                    loadTimings: currentTimings,
                    progress: progress
                )
        }
    }

    /// Create a fresh generation task with the guard/seed/counter boilerplate.
    ///
    /// Each call returns an independent task with its own sampler seed and per-task
    /// buffers. Delegates to `setupGenerateTask(...)` for the actual construction.
    open func createTask(progress: Progress? = nil) throws -> any SpeechGenerating {
        guard let tokenizer else {
            throw TTSError.tokenizerUnavailable("Tokenizer is not loaded. Call loadModels() before generating speech.")
        }
        let derivedSeed: UInt64? = seed.map { $0 ^ taskCounter }
        taskCounter += 1
        return try setupGenerateTask(
            currentTimings: currentTimings,
            progress: progress ?? Progress(),
            tokenizer: tokenizer,
            sampler: GreedyTokenSampler(seed: derivedSeed)
        )
    }

    // MARK: - Speech generation

    /// Synthesize speech from text and return the complete audio result.
    ///
    /// Mirrors `WhisperKit.transcribe(audioPath:decodeOptions:callback:)`.
    /// Handles text chunking, optional prompt caching, and concurrent multi-chunk generation.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voice: Voice/speaker identifier. Format is model-specific (e.g. `"ryan"` for Qwen3 TTS).
    ///   - language: Language identifier. Format is model-specific (e.g. `"english"` for Qwen3 TTS).
    ///   - options: Sampling and generation options.
    ///   - callback: Optional per-step callback receiving decoded audio chunks.
    ///               Return `false` to cancel; `nil` or `true` to continue.
    /// - Returns: A `SpeechResult` containing the raw audio samples and timing breakdown.
    /// - Throws: `TTSError` if text is empty, models are not loaded, or generation fails.
    open func generate(
        text: String,
        voice: String? = nil,
        language: String? = nil,
        options: GenerationOptions = GenerationOptions(),
        callback: SpeechCallback = nil
    ) async throws -> SpeechResult {
        // Auto-load models if they have not been loaded yet, mirroring WhisperKit's
        // runTranscribeTask which calls loadModels() when modelState != .loaded.
        if modelState != .loaded {
            try await loadModels()
        }

        try Task.checkCancellation()

        // Create the primary task to resolve model-specific defaults for voice/language.
        // This task is also reused for the single-chunk fast path to avoid a second allocation.
        let primaryTask = try createTask()
        let resolvedVoice = voice ?? primaryTask.defaultVoice
        let resolvedLanguage = language ?? primaryTask.defaultLanguage

        // Build prompt cache ahead of time if none exists or current doesn't match.
        let cache: TTSPromptCache?
        if let existing = promptCache, existing.matches(voice: resolvedVoice, language: resolvedLanguage, instruction: options.instruction) {
            cache = existing
        } else if tokenizer != nil {
            cache = try await buildPromptCache(voice: resolvedVoice, language: resolvedLanguage, instruction: options.instruction)
        } else {
            cache = nil
        }

        let effectiveStrategy = options.chunkingStrategy ?? .sentence
        let textChunks: [String]
        if effectiveStrategy == .none || tokenizer == nil {
            textChunks = [text]
        } else {
            guard let tokenizer else {
                throw TTSError.tokenizerUnavailable("Tokenizer is not loaded. Call loadModels() before generating speech.")
            }
            let chunker = TextChunker(
                targetChunkSize: options.targetChunkSize ?? TextChunker.defaultTargetChunkSize,
                minChunkSize: options.minChunkSize ?? TextChunker.defaultMinChunkSize,
                tokenizer: tokenizer
            )
            let chunks = chunker.chunk(text)
            textChunks = chunks.isEmpty ? [text] : chunks
        }

        // Single-chunk fast path: reuse primaryTask (already allocated above).
        if textChunks.count == 1 {
            return try await primaryTask.run(
                text: textChunks[0],
                voice: resolvedVoice,
                language: resolvedLanguage,
                options: options,
                callback: callback,
                prefixCache: cache
            )
        }

        let workerDesc = options.concurrentWorkerCount == 0 ? "max" : "\(options.concurrentWorkerCount)"
        Logging.info("Chunked TTS: \(textChunks.count) chunks, concurrency=\(workerDesc)")
        for (i, chunk) in textChunks.enumerated() {
            let truncated = chunk.count > 60 ? "\(chunk.prefix(60))..." : chunk
            Logging.debug("  Chunk \(i): \"\(truncated)\" (\(chunk.count) chars)")
        }

        let pipelineStart = CFAbsoluteTimeGetCurrent()
        var combinedTimings = SpeechTimings()
        combinedTimings.modelLoading = currentTimings.modelLoading
        combinedTimings.tokenizerLoadTime = currentTimings.tokenizerLoadTime

        let crossfadeSamples = primaryTask.sampleRate / 10 // 100ms crossfade
        var chunkAudioArrays = [[Float]](repeating: [], count: textChunks.count)

        let totalChunks = textChunks.count

        let maxSteps = totalChunks * options.maxNewTokens

        if options.concurrentWorkerCount == 1 {
            var stepsSoFar = 0
            for (i, chunkText) in textChunks.enumerated() {
                Logging.debug(String(format: "  Generating chunk %d/%d...", i + 1, totalChunks))
                let chunkStepBase = stepsSoFar
                let wrappedCallback: SpeechCallback = callback.map { cb in
                    { @Sendable progress in
                        var p = progress
                        p.chunkIndex = i
                        p.totalChunks = totalChunks
                        p.stepsCompleted = chunkStepBase + Int(progress.timings.totalDecodingLoops)
                        p.totalSteps = maxSteps
                        return cb(p)
                    }
                }
                let chunkResult = try await (createTask()).run(
                    text: chunkText, voice: resolvedVoice, language: resolvedLanguage,
                    options: options, callback: wrappedCallback, prefixCache: cache
                )
                stepsSoFar += options.maxNewTokens
                chunkAudioArrays[i] = chunkResult.audio
                combinedTimings.merge(chunkResult.timings)
                if i == 0 { combinedTimings.timeToFirstBuffer = chunkResult.timings.timeToFirstBuffer }
                Logging.debug(
                    String(
                        format: "  Chunk %d done: %.2fs audio (%d steps)",
                        i + 1, chunkResult.audioDuration, Int(chunkResult.timings.totalDecodingLoops))
                )
            }
        } else {
            let indexedChunks = textChunks.enumerated().map { (index: $0.offset, text: $0.element) }

            let effectiveWorkers = options.concurrentWorkerCount == 0 ? indexedChunks.count : options.concurrentWorkerCount

            let batchedChunks: [[(index: Int, text: String)]]
            batchedChunks = stride(from: 0, to: indexedChunks.count, by: effectiveWorkers).map {
                Array(indexedChunks[$0..<min($0 + effectiveWorkers, indexedChunks.count)])
            }

            let maxSteps = totalChunks * options.maxNewTokens
            let stepCounter = OSAllocatedUnfairLock(initialState: 0)

            for batch in batchedChunks {
                let chunkCount = textChunks.count
                var taskItems: [(index: Int, text: String, task: any SpeechGenerating)] = []
                for item in batch {
                    try taskItems.append((index: item.index, text: item.text, task: createTask()))
                }

                // Per-step callback for concurrent workers: reports step count with
                // empty audio (ordered audio is delivered after the batch).
                let workerCallback: SpeechCallback = callback.map { unwrappedCallback in
                    { @Sendable progress in
                        let steps = stepCounter.withLock { state -> Int in
                            state += 1
                            return state
                        }
                        let stepProgress = SpeechProgress(
                            audio: [], timings: progress.timings,
                            totalChunks: chunkCount,
                            stepsCompleted: steps, totalSteps: maxSteps
                        )
                        return unwrappedCallback(stepProgress)
                    }
                }

                let maxNewTokens = options.maxNewTokens
                let batchResults: [(index: Int, result: SpeechResult)] = try await withThrowingTaskGroup(
                    of: (index: Int, result: SpeechResult).self
                ) { group in
                    for item in taskItems {
                        group.addTask {
                            Logging.debug(String(format: "  Starting chunk %d/%d...", item.index + 1, chunkCount))
                            let chunkResult = try await item.task.run(
                                text: item.text, voice: resolvedVoice, language: resolvedLanguage,
                                options: options, callback: workerCallback, prefixCache: cache
                            )
                            // Snap progress forward to the full budget for this chunk
                            let actualSteps = Int(chunkResult.timings.totalDecodingLoops)
                            let remaining = maxNewTokens - actualSteps
                            if remaining > 0 {
                                stepCounter.withLock { $0 += remaining }
                            }
                            Logging.debug(
                                String(
                                    format: "  Chunk %d done: %.2fs audio (%d steps)",
                                    item.index + 1, chunkResult.audioDuration, actualSteps))
                            return (index: item.index, result: chunkResult)
                        }
                    }
                    var results = [(index: Int, result: SpeechResult)]()
                    for try await result in group {
                        results.append(result)
                    }
                    return results
                }

                for entry in batchResults {
                    chunkAudioArrays[entry.index] = entry.result.audio
                    combinedTimings.merge(entry.result.timings)
                    if entry.index == 0 { combinedTimings.timeToFirstBuffer = entry.result.timings.timeToFirstBuffer }
                }
            }

            // Deliver audio in order via callback after concurrent batch completes.
            if let callback {
                for (i, chunkAudio) in chunkAudioArrays.enumerated() {
                    let progress = SpeechProgress(
                        audio: chunkAudio, timings: combinedTimings,
                        stepTime: i == 0 ? 0 : nil,
                        chunkIndex: i, totalChunks: totalChunks
                    )
                    if callback(progress) == false { break }
                }
            }
        }

        // Crossfade consecutive chunks and assemble final audio.
        let allAudio = AudioOutput.crossfade(chunkAudioArrays, fadeLength: crossfadeSamples)

        combinedTimings.fullPipeline = CFAbsoluteTimeGetCurrent() - pipelineStart
        let sampleRate = primaryTask.sampleRate
        combinedTimings.inputAudioSeconds = Double(allAudio.count) / Double(sampleRate)

        let steps = Int(combinedTimings.totalDecodingLoops)
        let avgMs = steps > 0 ? combinedTimings.decodingLoop * 1000 / Double(steps) : 0
        Logging.info(
            String(
                format: "Chunked TTS: %d chunks, %d steps, %.1fms avg/step, %.2fs audio",
                textChunks.count, steps, avgMs, Double(allAudio.count) / Double(sampleRate)
            ))

        return SpeechResult(audio: allAudio, timings: combinedTimings, sampleRate: sampleRate)
    }

    // MARK: - Play Speech

    /// Generate speech and stream it through the audio output in real time.
    ///
    /// Generates speech and plays it back.
    ///
    /// For streaming strategies (auto, stream, buffered) chunking is forced to
    /// sequential (`concurrentWorkerCount = 1`) so frames can be enqueued in
    /// order. `generateFirst` respects the caller's concurrency setting so the
    /// full file can be generated with parallel workers before playback begins.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voice: Voice/speaker identifier.
    ///   - language: Language identifier.
    ///   - options: Sampling and generation options.
    ///   - playbackStrategy: Controls how audio is buffered before playback begins.
    ///   - callback: Optional per-step callback.
    /// - Returns: A `SpeechResult` with the complete audio and timing breakdown.
    /// - Throws: `TTSError` on generation failure or task cancellation.
    open func play(
        text: String,
        voice: String? = nil,
        language: String? = nil,
        options: GenerationOptions = GenerationOptions(),
        playbackStrategy: PlaybackStrategy = .auto,
        callback: SpeechCallback = nil
    ) async throws -> SpeechResult {
        try await play(
            text: text,
            voice: voice,
            language: language,
            options: options,
            playbackStrategy: playbackStrategy,
            preserveExistingAudioSession: false,
            callback: callback
        )
    }

    /// Generate speech and stream it through the audio output in real time.
    ///
    /// Generates speech and plays it back.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voice: Voice/speaker identifier.
    ///   - language: Language identifier.
    ///   - options: Sampling and generation options.
    ///   - playbackStrategy: Controls how audio is buffered before playback begins.
    ///   - preserveExistingAudioSession: When `true` on iOS, playback reuses the
    ///     host app's current audio session configuration instead of forcing
    ///     `.playback`.
    ///   - callback: Optional per-step callback.
    /// - Returns: A `SpeechResult` with the complete audio and timing breakdown.
    /// - Throws: `TTSError` on generation failure or task cancellation.
    open func play(
        text: String,
        voice: String? = nil,
        language: String? = nil,
        options: GenerationOptions = GenerationOptions(),
        playbackStrategy: PlaybackStrategy = .auto,
        preserveExistingAudioSession: Bool = false,
        callback: SpeechCallback = nil
    ) async throws -> SpeechResult {
        var playOptions = options

        let audioOut = audioOutput
        let maxTokens = playOptions.maxNewTokens

        // Pre-resolve audio format from the task so the playback closure doesn't
        // reach into model-specific components (keeps TTSKit model-agnostic).
        let formatTask = try createTask()
        let samplesPerFrame = formatTask.samplesPerFrame
        let sampleRate = formatTask.sampleRate
        let minBuffer = formatTask.minimumBufferDuration

        if case .generateFirst = playbackStrategy {
            let result = try await generate(
                text: text, voice: voice, language: language,
                options: playOptions, callback: callback
            )
            try audioOut.startPlayback(
                preserveExistingAudioSession: preserveExistingAudioSession
            )
            audioOut.setBufferDuration(0)
            audioOut.enqueueAudioChunk(result.audio)
            await audioOut.stopPlayback(waitForCompletion: true)
            return result
        }

        // Streaming requires sequential generation to preserve chunk order.
        playOptions.concurrentWorkerCount = 1

        try audioOut.startPlayback(
            deferEngineStart: true,
            preserveExistingAudioSession: preserveExistingAudioSession
        )
        switch playbackStrategy {
            case .stream: audioOut.setBufferDuration(0)
            case let .buffered(secs): audioOut.setBufferDuration(secs)
            case .auto: break
            case .generateFirst: break
        }

        let result = try await generate(
            text: text, voice: voice, language: language,
            options: playOptions,
            callback: { progress in
                if let stepTime = progress.stepTime, case .auto = playbackStrategy {
                    let buffer = PlaybackStrategy.requiredBuffer(
                        stepTime: stepTime,
                        maxNewTokens: maxTokens,
                        samplesPerFrame: samplesPerFrame,
                        sampleRate: sampleRate,
                        minimumBuffer: minBuffer
                    )
                    audioOut.setBufferDuration(buffer)
                    let speedRatio = PlaybackStrategy.audioPerStep(samplesPerFrame: samplesPerFrame, sampleRate: sampleRate) / stepTime
                    Logging.info(
                        String(
                            format: "Playback: step %.1fms (%.2fx real-time) -> buffer %.2fs",
                            stepTime * 1000, speedRatio, buffer))
                }
                audioOut.enqueueAudioChunk(progress.audio)
                return callback?(progress)
            }
        )

        await audioOut.stopPlayback(waitForCompletion: true)
        return result
    }

    // MARK: - Qwen3-typed convenience API

    /// Build a prompt cache using typed Qwen3 speaker and language enums.
    ///
    /// - Parameters:
    ///   - speaker: The `Qwen3Speaker` to pre-warm the cache for.
    ///   - language: The `Qwen3Language` to pre-warm the cache for.
    ///   - instruction: Optional style instruction (1.7B only).
    /// - Returns: A `TTSPromptCache` for the given parameters.
    /// - Throws: `TTSError` on generation failure.
    @discardableResult
    open func buildPromptCache(
        speaker: Qwen3Speaker,
        language: Qwen3Language,
        instruction: String? = nil
    ) async throws -> TTSPromptCache {
        try await buildPromptCache(
            voice: speaker.rawValue,
            language: language.rawValue,
            instruction: instruction
        )
    }

    /// Generate speech from text using typed Qwen3 speaker and language enums.
    ///
    /// - Parameters:
    ///   - text: Input text to synthesise.
    ///   - speaker: The `Qwen3Speaker` voice to use.
    ///   - language: The `Qwen3Language` to synthesise in.
    ///   - options: Generation options controlling sampling, chunking, and concurrency.
    ///   - callback: Per-step callback receiving decoded audio chunks. Return `false` to cancel.
    /// - Returns: The assembled `SpeechResult`.
    /// - Throws: `TTSError` on generation failure or task cancellation.
    open func generate(
        text: String,
        speaker: Qwen3Speaker,
        language: Qwen3Language = .english,
        options: GenerationOptions = GenerationOptions(),
        callback: SpeechCallback = nil
    ) async throws -> SpeechResult {
        try await generate(
            text: text,
            voice: speaker.rawValue,
            language: language.rawValue,
            options: options,
            callback: callback
        )
    }

    /// Generate speech and stream playback using typed Qwen3 speaker and language enums.
    ///
    /// - Parameters:
    ///   - text: Input text to synthesise.
    ///   - speaker: The `Qwen3Speaker` voice to use.
    ///   - language: The `Qwen3Language` to synthesise in.
    ///   - options: Generation options controlling sampling, chunking, and concurrency.
    ///   - playbackStrategy: Controls how much audio is buffered before playback begins.
    ///   - callback: Per-step callback receiving decoded audio chunks. Return `false` to cancel.
    /// - Returns: The assembled `SpeechResult`.
    /// - Throws: `TTSError` on generation failure or task cancellation.
    open func play(
        text: String,
        speaker: Qwen3Speaker,
        language: Qwen3Language = .english,
        options: GenerationOptions = GenerationOptions(),
        playbackStrategy: PlaybackStrategy = .auto,
        preserveExistingAudioSession: Bool = false,
        callback: SpeechCallback = nil
    ) async throws -> SpeechResult {
        try await play(
            text: text,
            voice: speaker.rawValue,
            language: language.rawValue,
            options: options,
            playbackStrategy: playbackStrategy,
            preserveExistingAudioSession: preserveExistingAudioSession,
            callback: callback
        )
    }
}

// MARK: - SpeechModel conformance

extension TTSKit: SpeechModel {
    /// The output sample rate of the currently loaded speech decoder.
    public var sampleRate: Int { speechDecoder.sampleRate }
}
