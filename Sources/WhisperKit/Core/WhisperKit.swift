//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

@_exported import ArgmaxCore
import Accelerate
import ArgmaxCore
import AVFoundation
import CoreML
import Foundation

open class WhisperKit {
    /// Models
    public private(set) var modelVariant: ModelVariant = .tiny
    public private(set) var modelState: ModelState = .unloaded {
        didSet {
            modelStateCallback?(oldValue, modelState)
        }
    }

    public var modelCompute: ModelComputeOptions
    public var audioInputConfig: AudioInputConfig
    public var tokenizer: WhisperTokenizer? {
        didSet {
            // Always sync the tokenizer to the text decoder when set
            textDecoder.tokenizer = tokenizer
        }
    }

    /// Protocols
    public var audioProcessor: any AudioProcessing
    public var featureExtractor: any FeatureExtracting
    public var audioEncoder: any AudioEncoding
    public var textDecoder: any TextDecoding
    public var segmentSeeker: any SegmentSeeking
    public var voiceActivityDetector: VoiceActivityDetector?

    /// Shapes
    public static let sampleRate: Int = 16000
    public static let hopLength: Int = 160
    public static let secondsPerTimeToken = Float(0.02)

    /// Progress
    public private(set) var currentTimings: TranscriptionTimings
    public private(set) var progress = Progress()

    /// Configuration
    public var modelFolder: URL?
    public var tokenizerFolder: URL?
    public private(set) var useBackgroundDownloadSession: Bool

    /// Callbacks
    public var segmentDiscoveryCallback: SegmentDiscoveryCallback?
    public var modelStateCallback: ModelStateCallback?
    public var transcriptionStateCallback: TranscriptionStateCallback?

    public init(_ config: WhisperKitConfig = WhisperKitConfig()) async throws {
        modelCompute = config.computeOptions ?? ModelComputeOptions()
        audioInputConfig = config.audioInputConfig ?? AudioInputConfig()
        audioProcessor = config.audioProcessor ?? AudioProcessor()
        featureExtractor = config.featureExtractor ?? FeatureExtractor()
        audioEncoder = config.audioEncoder ?? AudioEncoder()
        textDecoder = config.textDecoder ?? TextDecoder()
        if let logitsFilters = config.logitsFilters {
            textDecoder.logitsFilters = logitsFilters
        }
        
        segmentSeeker = config.segmentSeeker ?? SegmentSeeker()
        voiceActivityDetector = config.voiceActivityDetector
        tokenizerFolder = config.tokenizerFolder ?? config.downloadBase
        useBackgroundDownloadSession = config.useBackgroundDownloadSession
        currentTimings = TranscriptionTimings()
        Logging.updateLogLevel(config.verbose ? config.logLevel : .none)

        try await setupModels(
            model: config.model,
            downloadBase: config.downloadBase,
            modelRepo: config.modelRepo,
            modelToken: config.modelToken,
            modelFolder: config.modelFolder,
            download: config.download,
            endpoint: config.modelEndpoint ?? Constants.defaultRemoteEndpoint
        )

        if let prewarm = config.prewarm, prewarm {
            Logging.info("Prewarming models...")
            try await prewarmModels()
        }

        // If load is not passed in, load based on whether a modelFolder is passed
        if config.load ?? (config.modelFolder != nil) {
            Logging.info("Loading models...")
            try await loadModels()
        }
    }

    public convenience init(
        model: String? = nil,
        downloadBase: URL? = nil,
        modelRepo: String? = nil,
        modelFolder: String? = nil,
        tokenizerFolder: URL? = nil,
        computeOptions: ModelComputeOptions? = nil,
        audioProcessor: (any AudioProcessing)? = nil,
        featureExtractor: (any FeatureExtracting)? = nil,
        audioEncoder: (any AudioEncoding)? = nil,
        textDecoder: (any TextDecoding)? = nil,
        logitsFilters: [any LogitsFiltering]? = nil,
        segmentSeeker: (any SegmentSeeking)? = nil,
        verbose: Bool = true,
        logLevel: Logging.LogLevel = .info,
        prewarm: Bool? = nil,
        load: Bool? = nil,
        download: Bool = true,
        useBackgroundDownloadSession: Bool = false
    ) async throws {
        let config = WhisperKitConfig(
            model: model,
            downloadBase: downloadBase,
            modelRepo: modelRepo,
            modelFolder: modelFolder,
            tokenizerFolder: tokenizerFolder,
            computeOptions: computeOptions,
            audioProcessor: audioProcessor,
            featureExtractor: featureExtractor,
            audioEncoder: audioEncoder,
            textDecoder: textDecoder,
            logitsFilters: logitsFilters,
            segmentSeeker: segmentSeeker,
            verbose: verbose,
            logLevel: logLevel,
            prewarm: prewarm,
            load: load,
            download: download,
            useBackgroundDownloadSession: useBackgroundDownloadSession
        )
        try await self.init(config)
    }

    // MARK: - Model Loading

    public static func deviceName() -> String {
        #if !os(macOS) && !targetEnvironment(simulator)
        var utsname = utsname()
        uname(&utsname)
        let deviceName = withUnsafePointer(to: &utsname.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: Int(_SYS_NAMELEN)) {
                String(cString: $0)
            }
        }
        #else
        let deviceName = ProcessInfo.hwModel
        #endif
        return deviceName
    }

    public static func recommendedModels() -> ModelSupport {
        let deviceName = Self.deviceName()
        Logging.debug("Running on \(deviceName)")
        return ModelUtilities.modelSupport(for: deviceName)
    }

    public static func recommendedRemoteModels(
        from repo: String = "argmaxinc/whisperkit-coreml",
        downloadBase: URL? = nil,
        token: String? = nil,
        remoteConfigName: String = Constants.defaultRemoteConfigName,
        endpoint: String = Constants.defaultRemoteEndpoint
    ) async -> ModelSupport {
        let deviceName = Self.deviceName()
        let config = await Self.fetchModelSupportConfig(
            from: repo,
            downloadBase: downloadBase,
            token: token,
            remoteConfigName: remoteConfigName,
            endpoint: endpoint
        )
        
        return ModelUtilities.modelSupport(for: deviceName, from: config)
    }

    public static func fetchModelSupportConfig(
        from repo: String = "argmaxinc/whisperkit-coreml",
        downloadBase: URL? = nil,
        token: String? = nil,
        remoteConfigName: String = Constants.defaultRemoteConfigName,
        endpoint: String = Constants.defaultRemoteEndpoint
    ) async -> ModelSupportConfig {
        let hubApi = HubApiWrapper(downloadBase: downloadBase, hfToken: token, endpoint: endpoint)
        let repoRef = HubApiWrapper.Repo(id: repo)
        var modelSupportConfig = Constants.fallbackModelSupportConfig

        do {
            // Try to decode config file into ModelSupportConfig
            Logging.debug("Searching for config file matching \"\(remoteConfigName)\" in \(repo)")
            let files = try await hubApi.getFilenames(from: repoRef, matching: [remoteConfigName])
            if files.count > 1 {
                Logging.info("Multiple config files found (\(files.count)): \(files). Using first matching file: \(files.first ?? "none")")
            } else if files.isEmpty {
                // Early return for missing remote config
                Logging.info("No config files found matching \"\(remoteConfigName)\" in \(repo), using fallback")
                return modelSupportConfig
            }
            
            // Use the first file in the list, or default to Constants.defaultRemoteConfigName
            let configFileName = files.first ?? Constants.defaultRemoteConfigName
            
            let configUrl = try await hubApi.snapshot(from: repoRef, matching: [configFileName])
            let decoder = JSONDecoder()
            let jsonData = try Data(contentsOf: configUrl.appendingPathComponent(configFileName))
            modelSupportConfig = try decoder.decode(ModelSupportConfig.self, from: jsonData)
        } catch {
            // Allow this to fail gracefully as it uses fallback config by default
            Logging.error("Fetch model support config failed with error: \(error)")
        }

        return modelSupportConfig
    }

    public static func fetchAvailableModels(
        from repo: String = "argmaxinc/whisperkit-coreml",
        matching: [String] = ["*"],
        downloadBase: URL? = nil,
        token: String? = nil,
        remoteConfigName: String = Constants.defaultRemoteConfigName,
        endpoint: String = Constants.defaultRemoteEndpoint
    ) async throws -> [String] {
        let modelSupportConfig = await fetchModelSupportConfig(
            from: repo,
            downloadBase: downloadBase,
            token: token,
            remoteConfigName: remoteConfigName,
            endpoint: endpoint
        )
        let supportedModels = modelSupportConfig.modelSupport().supported
        var filteredSupportSet: Set<String> = []
        for glob in matching {
            filteredSupportSet = filteredSupportSet.union(supportedModels.matching(glob: glob))
        }
        let filteredSupport = Array(filteredSupportSet)

        return ModelUtilities.formatModelFiles(filteredSupport)
    }

    public static func download(
        variant: String,
        downloadBase: URL? = nil,
        useBackgroundSession: Bool = false,
        from repo: String = "argmaxinc/whisperkit-coreml",
        token: String? = nil,
        endpoint: String = Constants.defaultRemoteEndpoint,
        progressCallback: ProgressCallback? = nil
    ) async throws -> URL {
        let hubApi = HubApiWrapper(downloadBase: downloadBase, hfToken: token, endpoint: endpoint, useBackgroundSession: useBackgroundSession)
        let repo = HubApiWrapper.Repo(id: repo, type: .models)
        var modelSearchPath = "*\(variant.description)/*"
        do {
            Logging.debug("Searching for models matching \"\(modelSearchPath)\" in \(repo)")
            let modelFiles = try await hubApi.getFilenames(from: repo, matching: [modelSearchPath])
            var uniquePaths = Set(modelFiles.map { $0.components(separatedBy: "/").first! })

            var variantPath: String? = nil

            if uniquePaths.count == 1 {
                variantPath = uniquePaths.first
            } else {
                // If the model name search returns more than one unique model folder,
                // then prepend the default "openai" prefix from whisperkittools to disambiguate
                Logging.debug("Multiple models found matching \"\(modelSearchPath)\"")
                modelSearchPath = "*openai*\(variant.description)/*"
                Logging.debug("Searching for models matching \"\(modelSearchPath)\" in \(repo)")
                let adjustedModelFiles = try await hubApi.getFilenames(from: repo, matching: [modelSearchPath])
                uniquePaths = Set(adjustedModelFiles.map { $0.components(separatedBy: "/").first! })

                if uniquePaths.count == 1 {
                    variantPath = uniquePaths.first
                }
            }

            guard let variantPath else {
                // If there is still ambiguity, throw an error
                throw WhisperError.modelsUnavailable(
                    "\(uniquePaths.count > 1 ? "Multiple" : "No") models found matching \"\(modelSearchPath)\" in \(repo)"
                )
            }

            Logging.debug("Downloading model \(variantPath)...")
            let modelFolder = try await hubApi.snapshot(from: repo, matching: [modelSearchPath]) { progress in
                Logging.debug(progress.debugDescription)
                if let callback = progressCallback {
                    callback(progress)
                }
            }

            let modelFolderName = modelFolder.appending(path: variantPath)
            return modelFolderName
        } catch {
            Logging.debug(error.localizedDescription)
            throw error
        }
    }

    /// Sets up the model folder either from a local path or by downloading from a repository.
    open func setupModels(
        model: String?,
        downloadBase: URL? = nil,
        modelRepo: String?,
        modelToken: String? = nil,
        modelFolder: String?,
        download: Bool,
        remoteConfigName: String = Constants.defaultRemoteConfigName,
        endpoint: String = Constants.defaultRemoteEndpoint
    ) async throws {
        // If a local model folder is provided, use it; otherwise, download the model
        if let folder = modelFolder {
            self.modelFolder = URL(fileURLWithPath: folder)
        } else if download {
            // Determine the model variant to use
            let repo = modelRepo ?? "argmaxinc/whisperkit-coreml"
            let modelVariant: String
            if let model {
                // Model is specified, skip remote config check
                modelVariant = model
            } else {
                // Model not specified, fetch remote config to get the recommended default
                let modelSupport = await WhisperKit.recommendedRemoteModels(
                    from: repo,
                    downloadBase: downloadBase,
                    token: modelToken,
                    remoteConfigName: remoteConfigName,
                    endpoint: endpoint
                )
                modelVariant = modelSupport.default
            }

            do {
                self.modelFolder = try await Self.download(
                    variant: modelVariant,
                    downloadBase: downloadBase,
                    useBackgroundSession: useBackgroundDownloadSession,
                    from: repo,
                    token: modelToken,
                    endpoint: endpoint
                )
            } catch {
                // Handle errors related to model downloading
                throw WhisperError.modelsUnavailable("""
                Model not found. Please check the model or repo name and try again.
                Error: \(error)
                """)
            }
        }
    }

    open func prewarmModels() async throws {
        try await loadModels(prewarmMode: true)
    }

    open func loadModels(
        prewarmMode: Bool = false
    ) async throws {
        modelState = prewarmMode ? .prewarming : .loading

        let modelLoadStart = CFAbsoluteTimeGetCurrent()

        guard let path = modelFolder else {
            throw WhisperError.modelsUnavailable("Model folder is not set.")
        }

        Logging.debug("Loading models from \(path.path) with prewarmMode: \(prewarmMode)")

        // Find either mlmodelc or mlpackage models
        let logmelUrl = ModelUtilities.detectModelURL(inFolder: path, named: "MelSpectrogram")
        let encoderUrl = ModelUtilities.detectModelURL(inFolder: path, named: "AudioEncoder")
        let decoderUrl = ModelUtilities.detectModelURL(inFolder: path, named: "TextDecoder")

        for item in [logmelUrl, encoderUrl, decoderUrl] {
            if !FileManager.default.fileExists(atPath: item.path) {
                throw WhisperError.modelsUnavailable("Model file not found at \(item.path)")
            }
        }

        if let featureExtractor = featureExtractor as? WhisperMLModel {
            Logging.debug("Loading feature extractor")
            try await featureExtractor.loadModel(
                at: logmelUrl,
                computeUnits: modelCompute.melCompute, // hardcoded to use GPU
                prewarmMode: prewarmMode
            )
            Logging.debug("Loaded feature extractor")
        }

        if let textDecoder = textDecoder as? WhisperMLModel {
            Logging.debug("Loading text decoder")
            let decoderLoadStart = CFAbsoluteTimeGetCurrent()
            try await textDecoder.loadModel(
                at: decoderUrl,
                computeUnits: modelCompute.textDecoderCompute,
                prewarmMode: prewarmMode
            )

            if prewarmMode {
                currentTimings.decoderSpecializationTime = CFAbsoluteTimeGetCurrent() - decoderLoadStart
            } else {
                currentTimings.decoderLoadTime = CFAbsoluteTimeGetCurrent() - decoderLoadStart
            }

            Logging.debug("Loaded text decoder in \(String(format: "%.2f", currentTimings.decoderLoadTime))s")
        }

        if let audioEncoder = audioEncoder as? WhisperMLModel {
            Logging.debug("Loading audio encoder")
            let encoderLoadStart = CFAbsoluteTimeGetCurrent()

            try await audioEncoder.loadModel(
                at: encoderUrl,
                computeUnits: modelCompute.audioEncoderCompute,
                prewarmMode: prewarmMode
            )

            if prewarmMode {
                currentTimings.encoderSpecializationTime = CFAbsoluteTimeGetCurrent() - encoderLoadStart
            } else {
                currentTimings.encoderLoadTime = CFAbsoluteTimeGetCurrent() - encoderLoadStart
            }

            Logging.debug("Loaded audio encoder in \(String(format: "%.2f", currentTimings.encoderLoadTime))s")
        }

        if prewarmMode {
            modelState = .prewarmed
            currentTimings.prewarmLoadTime = CFAbsoluteTimeGetCurrent() - modelLoadStart
            return
        }

        try await loadTokenizerIfNeeded()

        modelState = .loaded

        currentTimings.modelLoading = CFAbsoluteTimeGetCurrent() - modelLoadStart + currentTimings.prewarmLoadTime

        Logging.info("Loaded models for whisper size: \(modelVariant) in \(String(format: "%.2f", currentTimings.modelLoading))s")
    }

    open func loadTokenizerIfNeeded() async throws {
        // Loading not needed if tokenizer is non-nil
        guard tokenizer == nil else {
            Logging.debug("Tokenizer already exists on WhisperKit, skipping load")
            return
        }

        // Load new tokenizer if able to detect variant, otherwise throw
        guard let logitsDim = textDecoder.logitsSize, let encoderDim = audioEncoder.embedSize else {
            throw WhisperError.tokenizerUnavailable()
        }

        textDecoder.isModelMultilingual = ModelUtilities.isModelMultilingual(logitsDim: logitsDim)
        modelVariant = ModelUtilities.detectVariant(logitsDim: logitsDim, encoderDim: encoderDim)

        Logging.debug("Loading tokenizer for \(modelVariant)")
        let tokenizerLoadStart = CFAbsoluteTimeGetCurrent()

        // Search model folder for tokenizer if it is bundled with the model
        let additionalSearchPaths: [URL]
        if let modelFolder {
            // TODO: remove hub path in future version, retained as additional search path for backward compatibility
            let hubTokenizerFolderFromModel = HubApiWrapper(downloadBase: modelFolder).localRepoLocation(
                HubApiWrapper.Repo(id: ModelUtilities.tokenizerNameForVariant(modelVariant))
            )

            additionalSearchPaths = [modelFolder] + [hubTokenizerFolderFromModel]
        } else {
            additionalSearchPaths = []
        }
        
        let tokenizer = try await ModelUtilities.loadTokenizer(
            for: modelVariant,
            tokenizerFolder: tokenizerFolder,
            additionalSearchPaths: additionalSearchPaths,
            useBackgroundSession: useBackgroundDownloadSession
        )
        currentTimings.tokenizerLoadTime = CFAbsoluteTimeGetCurrent() - tokenizerLoadStart

        self.tokenizer = tokenizer
        Logging.debug("Loaded tokenizer in \(String(format: "%.2f", currentTimings.tokenizerLoadTime))s")
    }

    open func unloadModels() async {
        modelState = .unloading

        for model in [featureExtractor, audioEncoder, textDecoder] {
            if let model = model as? WhisperMLModel {
                model.unloadModel()
            }
        }

        modelState = .unloaded

        Logging.info("Unloaded all models")
    }

    open func clearState() {
        audioProcessor.stopRecording()
        currentTimings = TranscriptionTimings()
    }

    deinit {
        audioProcessor.stopRecording()
    }

    /// Pass in your own logging callback here
    open func loggingCallback(_ callback: Logging.LoggingCallback?) {
        Logging.updateCallback(callback)
    }

    // MARK: - Detect language

    /// Detects the language of the audio file at the specified path.
    ///
    /// - Parameter audioPath: The file path of the audio file.
    /// - Returns: A tuple containing the detected language and the language log probabilities.
    open func detectLanguage(
        audioPath: String
    ) async throws -> (language: String, langProbs: [String: Float]) {
        // Only need the first 30s for language detection
        let audioBuffer = try AudioProcessor.loadAudio(fromPath: audioPath, endTime: 30.0)
        let audioArray = AudioProcessor.convertBufferToArray(buffer: audioBuffer)
        return try await detectLangauge(audioArray: audioArray)
    }

    /// Detects the language of the audio samples in the provided array.
    ///
    /// - Parameter audioArray: An array of audio samples.
    /// - Returns: A tuple containing the detected language and the language log probabilities.
    open func detectLangauge(
        audioArray: [Float]
    ) async throws -> (language: String, langProbs: [String: Float]) {
        if modelState != .loaded {
            try await loadModels()
        }

        // Ensure the model is multilingual, as language detection is only supported for these models
        guard textDecoder.isModelMultilingual else {
            throw WhisperError.decodingFailed("Language detection not supported for this model")
        }

        // Tokenizer required for decoding
        guard let tokenizer else {
            throw WhisperError.tokenizerUnavailable()
        }

        let options = DecodingOptions(verbose: Logging.isLoggingEnabled)
        let decoderInputs = try textDecoder.prepareDecoderInputs(withPrompt: [tokenizer.specialTokens.startOfTranscriptToken])

        // Detect language using up to the first 30 seconds
        guard let audioSamples = audioProcessor.padOrTrim(
            fromArray: audioArray,
            startAt: 0,
            toLength: featureExtractor.windowSamples ?? Constants.defaultWindowSamples
        ) else {
            throw WhisperError.transcriptionFailed("Audio samples are nil")
        }
        guard let melOutput = try await featureExtractor.logMelSpectrogram(fromAudio: audioSamples) else {
            throw WhisperError.transcriptionFailed("Mel output is nil")
        }
        guard let encoderOutput = try await audioEncoder.encodeFeatures(melOutput) else {
            throw WhisperError.transcriptionFailed("Encoder output is nil")
        }

        let tokenSampler = GreedyTokenSampler(temperature: 0, eotToken: tokenizer.specialTokens.endToken, decodingOptions: options)
        guard let languageDecodingResult: DecodingResult = try? await textDecoder.detectLanguage(
            from: encoderOutput,
            using: decoderInputs,
            sampler: tokenSampler,
            options: options,
            temperature: 0
        ) else {
            throw WhisperError.decodingFailed("Language detection failed")
        }

        return (language: languageDecodingResult.language, langProbs: languageDecodingResult.languageProbs)
    }

    // MARK: - Transcribe multiple audio files

    /// Convenience method to transcribe multiple audio files asynchronously and return the results as an array of optional arrays of `TranscriptionResult`.
    /// - Returns: An array of optional arrays containing `TranscriptionResult`.
    open func transcribe(
        audioPaths: [String],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback? = nil
    ) async -> [[TranscriptionResult]?] {
        let transcribeResults = await transcribeWithResults(
            audioPaths: audioPaths,
            decodeOptions: decodeOptions,
            callback: callback
        )
        let results = transcribeResults.toOptionalArrays()
        return results
    }

    /// Transcribes multiple audio files asynchronously and returns the results as an array of tuples containing the file path and the `Result` object.
    ///
    /// This method processes the provided audio file paths by loading the audio data and then transcribing the audio arrays.
    /// It handles any errors that occur during loading or transcription and ensures that the results are returned in the correct order.
    ///
    /// - Parameters:
    ///   - audioPaths: An array of file paths pointing to the audio files to be transcribed.
    ///   - decodeOptions: Optional decoding options to customize the transcription process.
    ///   - callback: Optional callback to receive updates during the transcription process.
    ///
    /// - Returns: An array of `Result` objects with either a successful transcription result or an error.
    open func transcribeWithResults(
        audioPaths: [String],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback? = nil
    ) async -> [Result<[TranscriptionResult], Swift.Error>] {
        transcriptionStateCallback?(.convertingAudio)

        // Start timing the audio loading and conversion process
        let loadAudioStart = Date()

        // Load and extract audio data from the provided file paths
        let loadedAudioResult = await AudioProcessor.loadAudio(at: audioPaths, channelMode: audioInputConfig.channelMode)
        let audioArrays = loadedAudioResult.compactMap { try? $0.get() }

        // Calculate the time taken to load and convert audio
        let loadAndConvertTime = Date().timeIntervalSince(loadAudioStart)
        currentTimings.audioLoading = loadAndConvertTime
        Logging.debug("Total Audio Loading and Converting Time: \(loadAndConvertTime)")

        transcriptionStateCallback?(.transcribing)
        defer {
            transcriptionStateCallback?(.finished)
        }

        // Transcribe the loaded audio arrays
        let transcribeResults = await transcribeWithResults(
            audioArrays: audioArrays,
            decodeOptions: decodeOptions,
            callback: callback
        )

        // Initialize the result array to hold final transcription results
        var result = [Result<[TranscriptionResult], Swift.Error>]()
        var transcribeResultIndex = 0

        // Iterate over loadedAudioResult and map each to the corresponding transcription result
        for audioResult in loadedAudioResult {
            switch audioResult {
                case .success:
                    // Append transcription result if audio loading was successful (may still contain failure)
                    result.append(transcribeResults[transcribeResultIndex])
                    transcribeResultIndex += 1
                case let .failure(error):
                    // Append failure result if audio loading failed
                    result.append(.failure(error))
            }
        }

        return result
    }

    // MARK: - Transcribe multiple audio arrays

    /// Convenience method to transcribe multiple audio arrays asynchronously and return the results as an array of optional arrays of `TranscriptionResult`.
    /// - Returns: An array of optional arrays containing `TranscriptionResult`.
    open func transcribe(
        audioArrays: [[Float]],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback? = nil
    ) async -> [[TranscriptionResult]?] {
        let transcribeResults = await transcribeWithResults(
            audioArrays: audioArrays,
            decodeOptions: decodeOptions,
            callback: callback
        )

        return transcribeResults.toOptionalArrays()
    }

    /// Transcribes multiple audio arrays asynchronously and returns the results as an array of `Result` objects.
    ///
    /// This method processes the provided audio arrays by dividing them into batches based on the concurrent worker count
    /// specified in `decodeOptions`, if any. The transcription is performed concurrently on these chunks, and the results
    /// are aggregated and returned in the original order.
    ///
    /// - Parameters:
    ///   - audioArrays: An array of arrays, each containing audio sample data to be transcribed.
    ///   - decodeOptions: Optional decoding options to customize the transcription process.
    ///   - callback: Optional callback to receive updates during the transcription process.
    ///
    /// - Returns: An array of `Result` objects, each containing either a successful transcription result or an error.
    open func transcribeWithResults(
        audioArrays: [[Float]],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback? = nil
    ) async -> [Result<[TranscriptionResult], Swift.Error>] {
        // Create an array of decoding options with the same value for each audio array
        let decodeOptionsArray = Array(repeating: decodeOptions, count: audioArrays.count)
        return await transcribeWithOptions(
            audioArrays: audioArrays,
            decodeOptionsArray: decodeOptionsArray,
            seekOffsets: nil,
            callback: callback
        )
    }

    /// Method to transcribe multiple audio arrays asynchronously with optional associated decoding options and seek offset indexes for position tracking.
    /// - Parameters:
    ///  - audioArrays: An array of arrays, each containing audio
    ///  - decodeOptionsArray: An array of optional decoding options corresponding to each audio array
    ///  - seekOffsets: Optional array of seek offset indexes for each audio array in the original audio
    ///  - callback: Optional callback to receive updates during the transcription process.
    ///
    /// - Returns: An array of `Result` objects, each containing either a successful transcription result or an error.
    open func transcribeWithOptions(
        audioArrays: [[Float]],
        decodeOptionsArray: [DecodingOptions?] = [nil],
        seekOffsets: [Int]? = nil,
        callback: TranscriptionCallback? = nil
    ) async -> [Result<[TranscriptionResult], Swift.Error>] {
        var result = [Result<[TranscriptionResult], Swift.Error>]()

        guard audioArrays.count == decodeOptionsArray.count else {
            return [.failure(WhisperError.transcriptionFailed("The number of audio arrays and decoding options must be balanced."))]
        }
        
        if let seekOffsets {
            guard audioArrays.count == seekOffsets.count else {
                return [.failure(WhisperError.transcriptionFailed("The number of audio arrays and seek offset indexes must be balanced."))]
            }
        }

        // Determine the number of concurrent workers from decodeOptions based on the maximum value or default to 0
        let concurrentWorkerCount = decodeOptionsArray.map { $0?.concurrentWorkerCount ?? 0 }.max() ?? 0

        // Chunk the audio arrays based on the number of concurrent workers
        // If concurrentWorkerCount is 0, all audio arrays are processed in one batch
        let batchedAudioArrays = concurrentWorkerCount == 0 ? [audioArrays] : audioArrays.batched(into: concurrentWorkerCount)

        for (batchIndex, audioArrayBatch) in batchedAudioArrays.enumerated() {
            // Use withTaskGroup to manage concurrent transcription tasks
            let partialResult = await withTaskGroup(of: [(index: Int, result: Result<[TranscriptionResult], Swift.Error>)].self) { taskGroup -> [Result<[TranscriptionResult], Swift.Error>] in
                for (audioIndex, audioArray) in audioArrayBatch.enumerated() {
                    let batchSize = audioArrayBatch.count

                    // Setup callback to keep track of batches and chunks
                    let batchedAudioCallback: TranscriptionCallback = { progress in
                        var batchedProgress = progress
                        batchedProgress.windowId = audioIndex + batchIndex * batchSize
                        return callback?(batchedProgress)
                    }

                    // Setup segment callback to track chunk seek positions for segment discovery
                    let batchedSegmentCallback: SegmentDiscoveryCallback? = if let seekOffsets {
                        { [segmentDiscoveryCallback] segments in
                            let windowId = audioIndex + batchIndex * batchSize
                            let seekOffset = seekOffsets[windowId]
                            var adjustedSegments = segments
                            for i in 0..<adjustedSegments.count {
                                adjustedSegments[i].seek += Int(seekOffset)
                            }
                            segmentDiscoveryCallback?(adjustedSegments)
                        }
                    } else {
                        self.segmentDiscoveryCallback
                    }

                    // Setup decoding options for the current audio array
                    let batchedDecodeOptions = decodeOptionsArray[audioIndex]

                    // Add a new task to the task group for each audio array
                    let weakSelf = WeakSendableWrapper(self)
                    taskGroup.addTask {
                        do {
                            guard let self = weakSelf.value else {
                                return [(index: audioIndex, result: .failure(WhisperError.transcriptionFailed("WhisperKit instance was deallocated")))]
                            }
                            let transcribeResult: [TranscriptionResult] = try await self.transcribe(
                                audioArray: audioArray,
                                decodeOptions: batchedDecodeOptions,
                                callback: batchedAudioCallback,
                                segmentCallback: batchedSegmentCallback
                            )
                            // Return the successful transcription result with its index
                            return [(index: audioIndex, result: .success(transcribeResult))]
                        } catch {
                            // Return the failure result with its index in case of an error
                            return [(index: audioIndex, result: .failure(error))]
                        }
                    }
                }

                // Collect results from all completed tasks in the task group
                var batchResult = [(index: Int, result: Result<[TranscriptionResult], Swift.Error>)]()
                for await result in taskGroup {
                    batchResult.append(contentsOf: result)
                }

                // Sort the results by index to maintain the original order (they may not be in order due to concurrency)
                batchResult.sort(by: { $0.index < $1.index })

                // Map the sorted batch results to a simple array of results
                return batchResult.map { $0.result }
            }

            // Append the results of each batch to the final result array
            result.append(contentsOf: partialResult)
        }

        return result
    }

    // MARK: - Transcribe single audio file

    /// Transcribes an audio file from the given path asynchronously.
    /// - Parameters:
    ///   - audioPath: The file path to the audio file to be transcribed.
    ///   - decodeOptions: Options for how to transcribe audio. Includes a chunking strategy and the number of concurrent workers to parallelize the task.
    ///   - callback: Optional callback to receive updates during the transcription process.
    /// - Returns: An array of `TranscriptionResult`.
    /// - Throws: An error if the transcription fails.
    open func transcribe(
        audioPath: String,
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback? = nil
    ) async throws -> [TranscriptionResult] {
        transcriptionStateCallback?(.convertingAudio)

        // Process input audio file into audio samples
        let audioArray = try await withThrowingTaskGroup(of: [Float].self) { group -> [Float] in
            let convertAudioStart = Date()
            defer {
                let convertTime = Date().timeIntervalSince(convertAudioStart)
                currentTimings.audioLoading = convertTime
                Logging.debug("Audio loading and convert time: \(convertTime)")
                Logging.logCurrentMemoryUsage("Audio Loading and Convert")
            }
            return try AudioProcessor.loadAudioAsFloatArray(fromPath: audioPath, channelMode: audioInputConfig.channelMode)
        }

        transcriptionStateCallback?(.transcribing)
        defer {
            transcriptionStateCallback?(.finished)
        }

        // Send converted samples to be transcribed
        let transcribeResults: [TranscriptionResult] = try await transcribe(
            audioArray: audioArray,
            decodeOptions: decodeOptions,
            callback: callback
        )

        return transcribeResults
    }

    // MARK: - Transcribe single audio sample array

    /// Main entry point for transcribing audio
    /// - Parameters:
    ///   - audioArray: Array of 16khz raw float audio samples
    ///   - decodeOptions: Options for how to transcribe audio. Including a chunking strategy and the number of concurrent workers will paralleize this task.
    ///   - callback: Optional callback to receive updates during the transcription process.
    ///   - segmentCallback: Optional callback to receive segment discovery updates during transcription.
    /// - Returns: An array of sorted `TranscriptionResult`.
    /// - Throws: An error if the transcription fails.
    open func transcribe(
        audioArray: [Float],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback? = nil,
        segmentCallback: SegmentDiscoveryCallback? = nil
    ) async throws -> [TranscriptionResult] {
        var transcribeResults = [TranscriptionResult]()

        // Determine if the audio array requires chunking
        let isChunkable = audioArray.count > featureExtractor.windowSamples ?? Constants.defaultWindowSamples
        switch (isChunkable, decodeOptions?.chunkingStrategy) {
            case (true, .vad):
                // We have some audio that will require multiple windows and a strategy to chunk them
                let vad = voiceActivityDetector ?? EnergyVAD()
                let chunker = VADAudioChunker(vad: vad)
                let audioChunks: [AudioChunk] = try await chunker.chunkAll(
                    audioArray: audioArray,
                    maxChunkLength: featureExtractor.windowSamples ?? Constants.defaultWindowSamples,
                    decodeOptions: decodeOptions
                )

                Logging.debug("Found \(audioChunks.count) VAD chunks")

                progress.totalUnitCount = max(progress.totalUnitCount, Int64(audioChunks.count))

                // Reset the seek times since we've already chunked the audio
                var chunkedOptions = decodeOptions
                chunkedOptions?.clipTimestamps = []
                let chunkedDecodeOptions = Array(repeating: chunkedOptions, count: audioChunks.count)

                // Send chunked samples to transcribe (note: this is recursive)
                let chunkedResults: [Result<[TranscriptionResult], Swift.Error>] = await transcribeWithOptions(
                    audioArrays: audioChunks.map { $0.audioSamples },
                    decodeOptionsArray: chunkedDecodeOptions,
                    seekOffsets: audioChunks.map { $0.seekOffsetIndex },
                    callback: callback
                )

                // Update the seek offsets based on the audio chunks
                let updatedTranscriptionResults = chunker.updateSeekOffsetsForResults(
                    chunkedResults: chunkedResults,
                    audioChunks: audioChunks
                )

                transcribeResults = updatedTranscriptionResults
            default:
                // Audio is short enough to transcribe in a single window and doesn't require chunking
                transcribeResults = try await runTranscribeTask(
                    audioArray: audioArray,
                    decodeOptions: decodeOptions,
                    callback: callback,
                    segmentCallback: segmentCallback ?? self.segmentDiscoveryCallback
                )
        }

        if let decodeOptions, decodeOptions.verbose {
            Logging.info("Total Transcription Results: \(transcribeResults.count)")
            for (i, transcribeTaskResult) in transcribeResults.enumerated() {
                Logging.debug("[Result \(i)]")
                transcribeTaskResult.logSegments()
            }
        }

        return transcribeResults
    }

    /// Setup the `TranscribeTask` used for decoding. Subclasses may override to provide custom behavior.
    open func setupTranscribeTask(
        currentTimings: TranscriptionTimings,
        progress: Progress,
        audioProcessor: any AudioProcessing,
        audioEncoder: any AudioEncoding,
        featureExtractor: any FeatureExtracting,
        segmentSeeker: any SegmentSeeking,
        textDecoder: any TextDecoding,
        tokenizer: any WhisperTokenizer
    ) -> TranscribeTask {
        TranscribeTask(
            currentTimings: currentTimings,
            progress: progress,
            audioProcessor: audioProcessor,
            audioEncoder: audioEncoder,
            featureExtractor: featureExtractor,
            segmentSeeker: segmentSeeker,
            textDecoder: textDecoder,
            tokenizer: tokenizer
        )
    }

    /// Runs the transcription task on a single audio sample array asynchronously with custom segment callback.
    /// - Returns: An array of `TranscriptionResult`.
    /// - Throws: An error if the transcription fails or if the tokenizer is unavailable.
    open func runTranscribeTask(
        audioArray: [Float],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback? = nil,
        segmentCallback: SegmentDiscoveryCallback? = nil
    ) async throws -> [TranscriptionResult] {
        if modelState != .loaded {
            try await loadModels()
        }

        guard let tokenizer else {
            // Tokenizer required for decoding
            throw WhisperError.tokenizerUnavailable()
        }

        do {
            try Task.checkCancellation()

            let childProgress = Progress()
            // Total can be set elsewhere, here ensures it is at least 1
            progress.totalUnitCount = max(1, progress.totalUnitCount)
            progress.addChild(childProgress, withPendingUnitCount: 1)

            let transcribeTask = setupTranscribeTask(
                currentTimings: currentTimings,
                progress: childProgress,
                audioProcessor: audioProcessor,
                audioEncoder: audioEncoder,
                featureExtractor: featureExtractor,
                segmentSeeker: segmentSeeker,
                textDecoder: textDecoder,
                tokenizer: tokenizer
            )

            transcribeTask.segmentDiscoveryCallback = segmentCallback

            let transcribeTaskResult = try await transcribeTask.run(
                audioArray: audioArray,
                decodeOptions: decodeOptions,
                callback: callback
            )

            if let decodeOptions, decodeOptions.verbose {
                transcribeTaskResult.logTimings()
            }

            if progress.isFinished {
                // Reset progress if it is completed
                progress = Progress()
            }

            return [transcribeTaskResult]
        } catch {
            // Handle cancellation
            if error is CancellationError {
                // Reset progress when cancelled
                progress = Progress()
            }
            throw error
        }
    }
}
