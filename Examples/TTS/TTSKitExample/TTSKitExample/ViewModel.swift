//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

@preconcurrency import AVFoundation
import CoreML
import Foundation
import Observation
import SwiftUI
import TTSKit
import WhisperKit
#if os(macOS)
import AppKit
#endif

// MARK: - Data Model

/// An in-memory representation of a generated audio clip.
///
/// All persistent fields live inside the M4A file's metadata - there is no
/// companion JSON or database. `Generation` is reconstructed on launch by
/// scanning the documents directory for `.m4a` files and calling
/// `AudioOutput.loadMetadata(from:)` on each one.
///
/// `isFavorite` is the only mutable bit of state not in the file itself; it
/// is stored as a `Set<UUID>` in `UserDefaults` so we never need to re-encode
/// the M4A.
@MainActor
struct Generation: Identifiable {
    let id: UUID
    let title: String
    let text: String
    let speaker: String
    let language: String
    let instruction: String
    let modelName: String
    let audioDuration: TimeInterval
    let realTimeFactor: Double
    let speedFactor: Double
    let stepsPerSecond: Double
    let timeToFirstBuffer: TimeInterval
    let date: Date
    var isFavorite: Bool
    let audioFileName: String
    var waveformSamples: [Float]?

    init(
        metadata: AudioMetadata,
        audioFileName: String,
        audioDuration: TimeInterval,
        isFavorite: Bool
    ) {
        self.id = metadata.id
        self.title = String(metadata.text.prefix(ViewModel.titleMaxLength))
        self.text = metadata.text
        self.speaker = metadata.speaker
        self.language = metadata.language
        self.instruction = metadata.instruction
        self.modelName = metadata.modelName
        self.audioDuration = audioDuration
        self.realTimeFactor = metadata.realTimeFactor
        self.speedFactor = metadata.speedFactor
        self.stepsPerSecond = metadata.stepsPerSecond
        self.timeToFirstBuffer = metadata.timeToFirstBuffer
        self.date = metadata.date
        self.isFavorite = isFavorite
        self.audioFileName = audioFileName
        self.waveformSamples = nil
    }
}

// MARK: - Model State

enum ModelState: Equatable {
    case unloaded
    case downloading
    case prewarming
    case loading
    case loaded
    case error(String)

    var label: String {
        switch self {
            case .unloaded: return "Unloaded"
            case .downloading: return "Downloading..."
            case .prewarming: return "Specializing..."
            case .loading: return "Loading..."
            case .loaded: return "Loaded"
            case let .error(msg): return "Error: \(msg)"
        }
    }

    var color: Color {
        switch self {
            case .unloaded: return .red
            case .downloading, .prewarming, .loading: return .yellow
            case .loaded: return .green
            case .error: return .red
        }
    }

    var isBusy: Bool {
        self == .downloading || self == .prewarming || self == .loading
    }
}

enum GenerationState: Equatable {
    case idle
    case generating
}

// MARK: - Settings Storage

/// Holds all persisted settings using @AppStorage.
/// Lives inside ViewModel and is read once at init time to seed the
/// observable stored properties. Each stored property's didSet writes
/// the new value back here so UserDefaults stays in sync.
@MainActor
private class Settings {
    /// Model selection
    @AppStorage("selectedPreset") var selectedPresetRaw: String = TTSModelVariant.defaultForCurrentPlatform.rawValue
    @AppStorage("selectedSpeaker") var selectedSpeakerRaw: String = Qwen3Speaker.ryan.rawValue
    @AppStorage("selectedLanguage") var selectedLanguageRaw: String = Qwen3Language.english.rawValue
    @AppStorage("playbackStrategyTag") var playbackStrategyTag: String = "auto"

    /// Generation options
    @AppStorage("genTemperature") var temperature: Double = .init(GenerationOptions.defaultTemperature)
    @AppStorage("genTopK") var topK: Double = .init(GenerationOptions.defaultTopK)
    @AppStorage("genRepetitionPenalty") var repetitionPenalty: Double = .init(GenerationOptions.defaultRepetitionPenalty)
    @AppStorage("genMaxNewTokens") var maxNewTokens: Double = .init(GenerationOptions.defaultMaxNewTokens)
    @AppStorage("genConcurrentWorkerCount") var concurrentWorkerCount: Double = 0
    @AppStorage("genChunkingStrategy") var chunkingStrategyTag: String = "sentence"
    @AppStorage("genTargetChunkSize") var targetChunkSize: Double = .init(TextChunker.defaultTargetChunkSize)
    @AppStorage("genMinChunkSize") var minChunkSize: Double = .init(TextChunker.defaultMinChunkSize)

    /// Compute units (stored as Int matching MLComputeUnits.rawValue)
    @AppStorage("embedderComputeUnits") var embedderComputeUnitsRaw: Int = MLComputeUnits.cpuOnly.rawValue
    @AppStorage("codeDecoderComputeUnits") var codeDecoderComputeUnitsRaw: Int = MLComputeUnits.cpuAndNeuralEngine.rawValue
    @AppStorage("multiCodeDecoderComputeUnits") var multiCodeDecoderComputeUnitsRaw: Int = MLComputeUnits.cpuAndNeuralEngine.rawValue
    @AppStorage("speechDecoderComputeUnits") var speechDecoderComputeUnitsRaw: Int = MLComputeUnits.cpuAndNeuralEngine.rawValue
}

// MARK: - View Model

@MainActor
@Observable
final class ViewModel: @unchecked Sendable {
    // MARK: - Constants

    fileprivate static let titleMaxLength = 40
    private static let historyLimit = 20
    /// Poll interval for playback position updates (~30 fps).
    private static let playbackPollIntervalMs = 33
    /// Debounce delay before running a token count after a keystroke.
    private static let tokenCountDebounceMs = 250

    // MARK: - Settings storage

    private let settings = Settings()

    // MARK: - Model management

    var modelState: ModelState = .unloaded
    var downloadProgress: Double = 0
    var localModelPaths: [TTSModelVariant: String] = [:]

    // MARK: - Persisted: model selection

    var selectedPreset: TTSModelVariant {
        didSet { settings.selectedPresetRaw = selectedPreset.rawValue }
    }

    // MARK: - Generation state

    var generationState: GenerationState = .idle
    var statusMessage = "Select a model to get started"

    // MARK: - Persisted: input defaults

    var selectedSpeaker: Qwen3Speaker {
        didSet { settings.selectedSpeakerRaw = selectedSpeaker.rawValue }
    }

    var selectedLanguage: Qwen3Language {
        didSet { settings.selectedLanguageRaw = selectedLanguage.rawValue }
    }

    var playbackStrategyTag: String {
        didSet { settings.playbackStrategyTag = playbackStrategyTag }
    }

    var selectedPlaybackStrategy: PlaybackStrategy {
        switch playbackStrategyTag {
            case "stream": return .stream
            case "generateFirst": return .generateFirst
            default: return .auto
        }
    }

    var instruction: String = ""

    // MARK: - Persisted: generation options

    var temperature: Double { didSet { settings.temperature = temperature } }
    var topK: Double { didSet { settings.topK = topK } }
    var repetitionPenalty: Double { didSet { settings.repetitionPenalty = repetitionPenalty } }
    var maxNewTokens: Double { didSet { settings.maxNewTokens = maxNewTokens } }
    var concurrentWorkerCount: Double { didSet { settings.concurrentWorkerCount = concurrentWorkerCount } }
    var chunkingStrategyTag: String { didSet { settings.chunkingStrategyTag = chunkingStrategyTag } }

    var chunkingStrategy: TextChunkingStrategy {
        TextChunkingStrategy(rawValue: chunkingStrategyTag) ?? .sentence
    }

    var targetChunkSize: Double { didSet { settings.targetChunkSize = targetChunkSize } }
    var minChunkSize: Double { didSet { settings.minChunkSize = minChunkSize } }

    // MARK: - Persisted: compute units

    var embedderComputeUnits: MLComputeUnits {
        didSet { settings.embedderComputeUnitsRaw = embedderComputeUnits.rawValue }
    }

    var codeDecoderComputeUnits: MLComputeUnits {
        didSet { settings.codeDecoderComputeUnitsRaw = codeDecoderComputeUnits.rawValue }
    }

    var multiCodeDecoderComputeUnits: MLComputeUnits {
        didSet { settings.multiCodeDecoderComputeUnitsRaw = multiCodeDecoderComputeUnits.rawValue }
    }

    var speechDecoderComputeUnits: MLComputeUnits {
        didSet { settings.speechDecoderComputeUnitsRaw = speechDecoderComputeUnits.rawValue }
    }

    var computeOptions: ComputeOptions {
        ComputeOptions(
            embedderComputeUnits: embedderComputeUnits,
            codeDecoderComputeUnits: codeDecoderComputeUnits,
            multiCodeDecoderComputeUnits: multiCodeDecoderComputeUnits,
            speechDecoderComputeUnits: speechDecoderComputeUnits
        )
    }

    // MARK: - Init

    init() {
        // Seed all persisted properties from @AppStorage backing store
        // Resolve the persisted preset, falling back to the platform default.
        // If a previously saved preset is no longer available on this platform (e.g. 0.6B
        // saved on macOS, then opened on iOS), quietly switch to the platform default.
        let savedPreset = TTSModelVariant(rawValue: settings.selectedPresetRaw)
        selectedPreset = (savedPreset?.isAvailableOnCurrentPlatform == true)
            ? savedPreset!
            : .defaultForCurrentPlatform
        selectedSpeaker = Qwen3Speaker(rawValue: settings.selectedSpeakerRaw) ?? .ryan
        selectedLanguage = Qwen3Language(rawValue: settings.selectedLanguageRaw) ?? .english
        playbackStrategyTag = settings.playbackStrategyTag
        temperature = settings.temperature
        topK = settings.topK
        repetitionPenalty = settings.repetitionPenalty
        maxNewTokens = settings.maxNewTokens
        concurrentWorkerCount = settings.concurrentWorkerCount
        chunkingStrategyTag = settings.chunkingStrategyTag
        targetChunkSize = settings.targetChunkSize
        minChunkSize = settings.minChunkSize
        embedderComputeUnits = MLComputeUnits(rawValue: settings.embedderComputeUnitsRaw) ?? .cpuOnly
        codeDecoderComputeUnits = MLComputeUnits(rawValue: settings.codeDecoderComputeUnitsRaw) ?? .cpuAndNeuralEngine
        multiCodeDecoderComputeUnits = MLComputeUnits(rawValue: settings.multiCodeDecoderComputeUnitsRaw) ?? .cpuAndNeuralEngine
        speechDecoderComputeUnits = MLComputeUnits(rawValue: settings.speechDecoderComputeUnitsRaw) ?? .cpuAndNeuralEngine
    }

    // MARK: - Generation output

    var currentWaveform: [Float] = []
    var currentAudioSamples: [Float] = []
    var currentDuration: TimeInterval = 0
    var currentRTF: Double = 0
    var currentSpeedFactor: Double = 0
    var currentStepsPerSecond: Double = 0
    var currentTimeToFirstBuffer: TimeInterval = 0

    // MARK: - Playback & streaming

    var isPlaying = false
    /// Current playback position in seconds (works for both streaming and replay)
    var playbackTime: TimeInterval = 0
    /// True while we're in a live generate-and-play session
    var isStreaming = false
    /// Reference to the active audio output during streaming, for querying playback position
    private var activeAudioOutput: AudioOutput?

    // MARK: - Generation progress (generateFirst mode)

    /// Running count of decoder steps completed across all chunks.
    var stepsCompleted: Int = 0
    /// Estimated total steps for the full request (maxNewTokens × totalChunks).
    var totalSteps: Int = 0
    /// Total number of text chunks in the current request.
    var chunksTotal: Int = 0

    /// Seconds of audio still accumulating in the pre-buffer before the next chunk
    /// flushes and playback resumes. Non-zero only while actively buffering mid-stream.
    var silentBufferRemaining: TimeInterval {
        guard isStreaming, let audioOut = activeAudioOutput else { return 0 }
        return audioOut.silentBufferRemaining
    }

    /// Total real audio scheduled to the player so far (excludes silent sentinel buffers).
    var scheduledAudioDuration: TimeInterval {
        guard isStreaming, let audioOut = activeAudioOutput else { return 0 }
        return audioOut.scheduledAudioDuration
    }

    private var playbackUpdateTask: Task<Void, Never>?
    private var generationTask: Task<Void, Never>?

    // MARK: - History

    /// A sentinel that drives the detail view into "new generation" mode without
    /// requiring a separate navigation Bool. Setting `selectedGenerationID` to this
    /// value shows the detail with empty inputs while no real generation is selected.
    static let newGenerationSentinel = UUID(uuidString: "00000000-0000-0000-0000-000000000000")!

    var generations: [Generation] = []
    var selectedGenerationID: UUID?
    var isLoadingHistory: Bool = true

    // MARK: - Search

    var searchText = ""

    // MARK: - Sheet presentation

    var showGenerationSettings = false

    // MARK: - Private

    private var ttsKit: TTSKit?
    private(set) var loadedPreset: TTSModelVariant?
    private var audioPlayer: AVAudioPlayer?
    private var tokenCountTask: Task<Void, Never>?

    // MARK: - Computed

    var selectedGeneration: Generation? {
        guard let id = selectedGenerationID else { return nil }
        return generations.first { $0.id == id }
    }

    var filteredGenerations: [Generation] {
        if searchText.isEmpty { return generations }
        let query = searchText.lowercased()
        return generations.filter {
            $0.title.lowercased().contains(query)
                || $0.text.lowercased().contains(query)
                || $0.speaker.lowercased().contains(query)
        }
    }

    var favoriteGenerations: [Generation] {
        filteredGenerations.filter { $0.isFavorite }
    }

    var recentGenerations: [Generation] {
        Array(filteredGenerations.prefix(Self.historyLimit))
    }

    var canGenerate: Bool {
        !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && generationState == .idle
            && !modelState.isBusy
    }

    var isModelDownloaded: Bool {
        localModelPaths[selectedPreset] != nil
    }

    var inputText = "" {
        didSet {
            guard inputText != oldValue else { return }
            scheduleTokenCount()
        }
    }

    /// Token count of `inputText` as measured by the loaded tokenizer.
    /// Updated with a short debounce as the user types; falls back to a character
    /// approximation in the UI when the tokenizer is not yet loaded.
    var inputTokenCount: Int?

    // MARK: - Lifecycle

    func onAppear() {
        Task { await loadGenerations() }
        scanLocalModels()
    }

    // MARK: - Model Management

    /// Hub storage path relative to Documents, matching the pattern used by WhisperAX / HubApi.
    /// The Hub library stores downloads at: Documents/huggingface/models/{repo}/
    private static let modelStorageBase = "huggingface/models"

    /// HuggingFace model repo for the selected preset
    var modelRepoURL: URL? {
        let config = TTSKitConfig(model: selectedPreset)
        return URL(string: "https://huggingface.co/\(config.modelRepo)")
    }

    /// Local path to the Hub-managed repo folder for a given config
    private func localRepoURL(for config: TTSKitConfig) -> URL {
        documentsDirectory
            .appendingPathComponent(Self.modelStorageBase)
            .appendingPathComponent(config.modelRepo)
    }

    /// Scan for previously downloaded models by checking the Hub cache in Documents
    func scanLocalModels() {
        for preset in TTSModelVariant.allCases {
            let config = TTSKitConfig(model: preset)
            let repoURL = localRepoURL(for: config)
            let repoPath = repoURL.path

            guard FileManager.default.fileExists(atPath: repoPath) else { continue }

            // Verify at least one component's version directory exists
            if hasModelFiles(at: repoPath, config: config) {
                localModelPaths[preset] = repoPath
            }
        }

        if isModelDownloaded, modelState == .unloaded {
            statusMessage = "Model downloaded"
        }
    }

    /// Check if any model component directories exist at a given repo path
    private func hasModelFiles(at basePath: String, config: TTSKitConfig) -> Bool {
        let baseURL = URL(fileURLWithPath: basePath)
        return config.componentDirectories(in: baseURL).contains {
            FileManager.default.fileExists(atPath: $0.path)
        }
    }

    /// Download the selected model preset without loading it
    func downloadModel() async {
        guard !modelState.isBusy else { return }
        modelState = .downloading
        downloadProgress = 0
        statusMessage = "Downloading \(selectedPreset.rawValue) model..."

        do {
            // Set a HuggingFace token via TTSKitConfig(token:) if the model repo is private.
            // For the public argmaxinc/ttskit-coreml repo no token is required.
            let config = TTSKitConfig(
                model: selectedPreset,
                verbose: true
            )
            let folder = try await TTSKit.download(config: config) { [weak self] progress in
                Task { @MainActor in
                    self?.downloadProgress = progress.fractionCompleted
                }
            }
            localModelPaths[selectedPreset] = folder.path
            modelState = .unloaded
            downloadProgress = 1.0
            statusMessage = "Downloaded"
        } catch {
            modelState = .error(error.localizedDescription)
            statusMessage = "Download failed: \(error.localizedDescription)"
        }
    }

    /// Load the selected model preset (downloads first if needed)
    func loadModel() async {
        guard !modelState.isBusy else { return }

        // If switching presets while loaded, unload first
        if loadedPreset != nil, loadedPreset != selectedPreset {
            unloadModel()
        }

        // Download if needed
        if localModelPaths[selectedPreset] == nil {
            await downloadModel()
            guard localModelPaths[selectedPreset] != nil else { return }
        }

        do {
            let ttsConfig = TTSKitConfig(
                model: selectedPreset,
                modelFolder: localModelPaths[selectedPreset].map { URL(fileURLWithPath: $0) },
                computeOptions: computeOptions,
                verbose: true
            )

            // Create TTSKit without loading - we drive load/prewarm explicitly
            ttsConfig.load = false
            let kit = try await TTSKit(ttsConfig)
            ttsKit = kit

            // Prewarm: compile each CoreML model sequentially, then discard.
            // This prevents memory exhaustion from concurrent compilation on first launch.
            // On subsequent launches the compiled cache is already on disk, so this is fast.
            modelState = .prewarming
            statusMessage = "Specializing \(selectedPreset.rawValue) for your device\nThis may take a few minutes on first load"
            do {
                try await kit.prewarmModels()
            } catch {
                // Prewarm failures are non-fatal (model may already be specialized).
                // Surface to the user so they know something unexpected happened,
                // but continue - the full load often succeeds regardless.
                let msg = error.localizedDescription
                Logging.debug("Prewarm warning: \(msg) - continuing to load")
                statusMessage = "Prewarm warning: \(msg)\nContinuing..."
            }

            // Load: compiled artifacts are cached, concurrent load is safe
            modelState = .loading
            statusMessage = "Loading \(selectedPreset.rawValue) model..."
            try await kit.loadModels()

            loadedPreset = selectedPreset
            modelState = .loaded
            statusMessage = "Ready - \(selectedPreset.rawValue) loaded"
            AccessibilityNotification.Announcement("\(selectedPreset.rawValue) model loaded and ready").post()
            // Refresh token count now that the tokenizer is available.
            scheduleTokenCount()
        } catch {
            modelState = .error(error.localizedDescription)
            statusMessage = "Load failed: \(error.localizedDescription)"
            AccessibilityNotification.Announcement("Model load failed: \(error.localizedDescription)").post()
        }
    }

    /// Reload the current model with updated compute options
    func reloadModelForComputeUnitChange() {
        guard modelState == .loaded, let preset = loadedPreset else { return }
        unloadModel()
        selectedPreset = preset
        Task { [weak self] in await self?.loadModel() }
    }

    /// Unload the current model from memory
    func unloadModel() {
        cancelAllTasks()
        let oldTTS = ttsKit
        ttsKit = nil
        Task { await oldTTS?.unloadModels() }
        loadedPreset = nil
        modelState = .unloaded
        statusMessage = isModelDownloaded ? "Model unloaded" : "Select a model to get started"
    }

    /// Reset all generation options to their factory defaults.
    func resetGenerationSettings() {
        temperature = Double(GenerationOptions.defaultTemperature)
        topK = Double(GenerationOptions.defaultTopK)
        repetitionPenalty = Double(GenerationOptions.defaultRepetitionPenalty)
        maxNewTokens = Double(GenerationOptions.defaultMaxNewTokens)
        concurrentWorkerCount = 0
        chunkingStrategyTag = "sentence"
        targetChunkSize = Double(TextChunker.defaultTargetChunkSize)
        minChunkSize = Double(TextChunker.defaultMinChunkSize)
    }

    /// Delete downloaded model files for a specific variant only,
    /// leaving other variants in the shared repo untouched.
    func deleteModel(preset: TTSModelVariant? = nil) {
        let target = preset ?? selectedPreset

        if loadedPreset == target {
            unloadModel()
        }

        let config = TTSKitConfig(model: target)
        let repoURL = localRepoURL(for: config)
        for dir in config.componentDirectories(in: repoURL) {
            guard FileManager.default.fileExists(atPath: dir.path) else { continue }
            try? FileManager.default.removeItem(at: dir)
        }
        localModelPaths.removeValue(forKey: target)

        if target == selectedPreset {
            modelState = .unloaded
            statusMessage = "Model deleted"
        }
    }

    /// Disk size of downloaded model files for a specific variant only.
    func modelDiskSize(for preset: TTSModelVariant) -> String? {
        guard localModelPaths[preset] != nil else { return nil }
        let config = TTSKitConfig(model: preset)
        let repoURL = localRepoURL(for: config)
        var total: UInt64 = 0
        for dir in config.componentDirectories(in: repoURL) {
            if let size = directorySize(at: dir) {
                total += size
            }
        }
        guard total > 0 else { return nil }
        return ByteCountFormatter.string(fromByteCount: Int64(total), countStyle: .file)
    }

    private func directorySize(at url: URL) -> UInt64? {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
            return nil
        }
        var total: UInt64 = 0
        for case let fileURL as URL in enumerator {
            if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                total += UInt64(size)
            }
        }
        return total
    }

    // MARK: - Generation

    /// Start generation in a tracked task that can be cancelled.
    /// If no model is loaded yet, automatically loads the selected (or default) model first.
    func startGeneration() {
        generationTask?.cancel()
        // Don't touch selectedGenerationID here. On iPhone, NavigationSplitView collapses
        // to a NavigationStack driven by List(selection:) - setting the selection to nil
        // immediately pops the detail view. The selection updates naturally when the new
        // generation is saved at the end of generate().
        generationTask = Task { [weak self] in
            guard let self else { return }
            if modelState != .loaded {
                await loadModel()
                guard !Task.isCancelled, modelState == .loaded else { return }
            }
            guard !Task.isCancelled else { return }
            await generate()
        }
    }

    /// Cancel all background tasks (generation, playback updates, token counting).
    /// Safe to call from any lifecycle event (view disappear, scene deactivation, etc.).
    func cancelAllTasks() {
        generationTask?.cancel()
        generationTask = nil
        tokenCountTask?.cancel()
        tokenCountTask = nil
        stopPlaybackUpdates()
        if activeAudioOutput != nil {
            let audioOut = activeAudioOutput
            activeAudioOutput = nil
            Task { await audioOut?.stopPlayback(waitForCompletion: false) }
        }
        audioPlayer?.stop()
        audioPlayer = nil
        isPlaying = false
        isStreaming = false
        if generationState == .generating {
            generationState = .idle
            statusMessage = "Cancelled"
        }
    }

    /// Cancel any in-progress generation and stop audio immediately.
    func cancelGeneration() {
        cancelAllTasks()
        generationState = .idle
        statusMessage = "Cancelled"
    }

    private func generate() async {
        guard canGenerate, let ttsKit else { return }

        generationState = .generating
        statusMessage = "Generating speech..."
        currentWaveform = []
        currentAudioSamples = []
        currentDuration = 0
        playbackTime = 0
        stepsCompleted = 0
        totalSteps = 0
        chunksTotal = 0

        let strategy = selectedPlaybackStrategy

        switch strategy {
        case .generateFirst:
            break
        case .auto, .stream, .buffered:
            isStreaming = true
            activeAudioOutput = ttsKit.audioOutput
            startPlaybackUpdates()
        }

        do {
            let result: SpeechResult

            switch strategy {
            case .generateFirst:
                result = try await generateFirstGeneration(tts: ttsKit)
                currentAudioSamples = result.audio
                currentDuration = result.audioDuration
                currentWaveform = peaksPerToken(from: result.audio)
                try await finalizeGeneration(result: result)
                if let gen = generations.first {
                    playGeneration(gen)
                }
            case .auto, .stream, .buffered:
                result = try await streamGeneration(tts: ttsKit)
                stopPlaybackUpdates()
                activeAudioOutput = nil
                isStreaming = false
                try await finalizeGeneration(result: result)
            }
        } catch {
            stopPlaybackUpdates()
            activeAudioOutput = nil
            isStreaming = false
            stepsCompleted = 0
            totalSteps = 0
            chunksTotal = 0
            statusMessage = "Error: \(error.localizedDescription)"
        }

        if generationState == .generating {
            generationState = .idle
            AccessibilityNotification.Announcement("Generation cancelled").post()
        }
    }

    /// Build the generation options from current UI state.
    private func buildOptions() -> GenerationOptions {
        let workerCount = Int(concurrentWorkerCount)
        var options = GenerationOptions(
            temperature: Float(temperature),
            topK: Int(topK),
            repetitionPenalty: Float(repetitionPenalty),
            maxNewTokens: Int(maxNewTokens),
            concurrentWorkerCount: workerCount,
            chunkingStrategy: chunkingStrategy,
            targetChunkSize: Int(targetChunkSize),
            minChunkSize: Int(minChunkSize)
        )
        if !instruction.isEmpty {
            options.instruction = instruction
        }
        return options
    }

    /// Generate all audio up front using `tts.generate()`, tracking step-level progress.
    private func generateFirstGeneration(tts: TTSKit) async throws -> SpeechResult {
        let options = buildOptions()

        let result = try await tts.generate(
            text: inputText,
            speaker: selectedSpeaker,
            language: selectedLanguage,
            options: options,
            callback: { [weak self] progress in
                let steps = progress.stepsCompleted ?? 0
                let maxSteps = progress.totalSteps ?? 1
                let chunks = progress.totalChunks ?? 1
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    stepsCompleted = steps
                    totalSteps = maxSteps
                    chunksTotal = chunks
                }
                return nil
            }
        )

        stepsCompleted = totalSteps
        currentRTF = result.timings.realTimeFactor
        currentSpeedFactor = result.timings.speedFactor
        currentStepsPerSecond = result.timings.tokensPerSecond
        currentTimeToFirstBuffer = result.timings.timeToFirstBuffer
        return result
    }

    /// Run `play`, streaming waveform peaks and audio samples back to the main actor.
    private func streamGeneration(tts: TTSKit) async throws -> SpeechResult {
        let options = buildOptions()
        let sampleRate = Double(Qwen3TTSConstants.sampleRate)

        let result = try await tts.play(
            text: inputText,
            speaker: selectedSpeaker,
            language: selectedLanguage,
            options: options,
            playbackStrategy: selectedPlaybackStrategy,
            callback: { [weak self] progress in
                let samples = progress.audio
                let peak = samples.reduce(Float(0)) { max($0, abs($1)) }
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    currentAudioSamples.append(contentsOf: samples)
                    currentDuration = Double(currentAudioSamples.count) / sampleRate
                    currentWaveform.append(peak)
                }
                return nil
            }
        )

        currentAudioSamples = result.audio
        currentDuration = result.audioDuration
        currentRTF = result.timings.realTimeFactor
        currentSpeedFactor = result.timings.speedFactor
        currentStepsPerSecond = result.timings.tokensPerSecond
        currentTimeToFirstBuffer = result.timings.timeToFirstBuffer
        playbackTime = 0
        return result
    }

    /// Persist the generation to disk as M4A, insert it into the history, and update UI state.
    private func finalizeGeneration(result: SpeechResult) async throws {
        let meta = AudioMetadata(
            text: inputText,
            speaker: selectedSpeaker.rawValue,
            language: selectedLanguage.rawValue,
            instruction: instruction,
            modelName: "\(Qwen3TTSConstants.modelFamilyDir)_\((loadedPreset ?? selectedPreset).versionDir)",
            realTimeFactor: result.timings.realTimeFactor,
            speedFactor: result.timings.speedFactor,
            stepsPerSecond: result.timings.tokensPerSecond,
            timeToFirstBuffer: result.timings.timeToFirstBuffer
        )
        let savedURL = try await AudioOutput.saveAudio(
            result.audio,
            toFolder: documentsDirectory,
            filename: meta.suggestedFileName,
            sampleRate: result.sampleRate,
            metadataProvider: meta.avMetadataItems
        )

        var gen = Generation(
            metadata: meta,
            audioFileName: savedURL.lastPathComponent,
            audioDuration: result.audioDuration,
            isFavorite: false
        )
        gen.waveformSamples = currentWaveform
        generations.insert(gen, at: 0)
        selectedGenerationID = gen.id

        generationState = .idle
        statusMessage = String(
            format: "Done generating %.1fs of audio, RTF %.2f",
            result.audioDuration,
            result.timings.realTimeFactor
        )
        AccessibilityNotification.Announcement(
            String(format: "Generation complete. %.1f seconds of audio.", result.audioDuration)
        ).post()
    }


    // MARK: - Playback position updates

    /// Starts a MainActor task that polls the audio engine's actual playback position at ~30fps.
    /// Using a Task instead of Timer ensures it always runs on the main thread.
    private func startPlaybackUpdates() {
        playbackUpdateTask?.cancel()
        playbackUpdateTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                guard let self else { break }

                if self.isStreaming, let audioOut = self.activeAudioOutput {
                    self.playbackTime = audioOut.currentPlaybackTime
                } else if self.isPlaying, let player = self.audioPlayer {
                    if player.isPlaying {
                        self.playbackTime = player.currentTime
                    } else {
                        // Replay finished
                        self.isPlaying = false
                        self.playbackTime = 0
                        break
                    }
                } else {
                    break
                }

                try? await Task.sleep(for: .milliseconds(Self.playbackPollIntervalMs))
            }
        }
    }

    private func stopPlaybackUpdates() {
        playbackUpdateTask?.cancel()
        playbackUpdateTask = nil
    }

    // MARK: - Playback (replay saved audio)

    /// Populate the input fields from a past generation so the user can edit and re-generate.
    /// Called automatically whenever a generation is selected from history.
    func loadInputs(from generation: Generation) {
        inputText = generation.text
        selectedSpeaker = Qwen3Speaker(rawValue: generation.speaker) ?? .ryan
        selectedLanguage = Qwen3Language(rawValue: generation.language) ?? .english
        instruction = generation.instruction
        currentRTF = generation.realTimeFactor
        currentSpeedFactor = generation.speedFactor
        currentStepsPerSecond = generation.stepsPerSecond
        currentTimeToFirstBuffer = generation.timeToFirstBuffer
    }

    func playGeneration(_ generation: Generation) {
        let url = documentsDirectory.appendingPathComponent(generation.audioFileName)
        guard FileManager.default.fileExists(atPath: url.path) else { return }

        do {
            #if os(iOS)
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default, options: [])
            try session.setActive(true)
            #endif

            audioPlayer?.stop()
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.play()
            isPlaying = true
            playbackTime = 0

            loadWaveform(for: generation)
            startPlaybackUpdates()
        } catch {
            statusMessage = "Playback error: \(error.localizedDescription)"
        }
    }

    func stopPlayback() {
        audioPlayer?.stop()
        isPlaying = false
        playbackTime = 0
        stopPlaybackUpdates()
    }

    /// URL for the audio file of a generation (for sharing/exporting)
    func audioFileURL(for generation: Generation) -> URL? {
        let url = documentsDirectory.appendingPathComponent(generation.audioFileName)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    // MARK: - History Management

    func toggleFavorite(_ id: UUID) {
        guard let idx = generations.firstIndex(where: { $0.id == id }) else { return }
        generations[idx].isFavorite.toggle()
        saveFavorites()
    }

    func deleteGeneration(_ id: UUID) {
        guard let idx = generations.firstIndex(where: { $0.id == id }) else { return }
        let url = documentsDirectory.appendingPathComponent(generations[idx].audioFileName)
        try? FileManager.default.removeItem(at: url)
        generations.remove(at: idx)
        if selectedGenerationID == id {
            selectedGenerationID = generations.first?.id
        }
    }

    func clearAllGenerations() {
        for gen in generations {
            let url = documentsDirectory.appendingPathComponent(gen.audioFileName)
            try? FileManager.default.removeItem(at: url)
        }
        generations.removeAll()
        selectedGenerationID = nil
        UserDefaults.standard.removeObject(forKey: Self.favoritesKey)
    }

    /// Debounced token counter: waits 250ms after the last keystroke before encoding.
    /// Uses the loaded tokenizer when available; silently skips if none is loaded yet.
    /// When the model loads for the first time, the next edit will trigger a real count.
    private func scheduleTokenCount() {
        tokenCountTask?.cancel()
        let text = inputText
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            inputTokenCount = nil
            return
        }
        tokenCountTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(Self.tokenCountDebounceMs))
            guard !Task.isCancelled, let self else { return }
            guard let count = self.ttsKit?.tokenizer?.encode(text: text).count else { return }
            await MainActor.run { self.inputTokenCount = count }
        }
    }

    func clearInput() {
        inputText = ""
        instruction = ""
        inputTokenCount = nil
        currentWaveform = []
        currentAudioSamples = []
        currentDuration = 0
        statusMessage = modelState == .loaded ? "Ready" : "Select a model to get started"
    }

    // MARK: - Persistence

    var documentsDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    // MARK: Generations - derived entirely from .m4a files on disk

    /// Scan the Documents directory for `.m4a` files, read embedded metadata from
    /// each one, and rebuild the in-memory `generations` array. No JSON sidecar needed.
    ///
    /// Files are loaded in parallel and streamed into `generations` as they complete,
    /// so the list populates progressively rather than appearing all at once after a
    /// full sequential scan. Waveform decoding runs on a detached background task to
    /// avoid blocking the main actor with synchronous PCM I/O.
    func loadGenerations() async {
        isLoadingHistory = true
        defer { isLoadingHistory = false }

        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(
            at: documentsDirectory,
            includingPropertiesForKeys: nil
        ) else { return }

        let m4aFiles = files
            .filter { $0.pathExtension == "m4a" }
            .sorted { $0.lastPathComponent > $1.lastPathComponent } // newest first by name

        let favorites = loadedFavoriteIDs()

        // Intermediate result: raw Sendable data that can cross the actor boundary.
        // Generation.init is @MainActor so it is constructed in the for-await loop below,
        // which runs back on the main actor.
        struct RawGenerationData: Sendable {
            let metadata: AudioMetadata
            let audioFileName: String
            let audioDuration: TimeInterval
            let isFavorite: Bool
            let waveformSamples: [Float]?
        }

        await withTaskGroup(of: RawGenerationData?.self) { group in
            for url in m4aFiles {
                group.addTask {
                    do {
                        guard let meta = try await AudioMetadata.load(from: url) else {
                            Logging.debug("Skipping \(url.lastPathComponent) - no TTSKit metadata found")
                            return nil
                        }
                        let dur = (try? await AudioOutput.duration(of: url)) ?? 0
                        // Decode PCM on a background thread to avoid blocking the main actor
                        let waveform = await Task.detached(priority: .utility) { [weak self] in
                            self?.waveformPeaks(from: url)
                        }.value
                        return RawGenerationData(
                            metadata: meta,
                            audioFileName: url.lastPathComponent,
                            audioDuration: dur,
                            isFavorite: favorites.contains(meta.id),
                            waveformSamples: waveform
                        )
                    } catch {
                        Logging.error("Failed to load \(url.lastPathComponent) - \(error.localizedDescription)")
                        return nil
                    }
                }
            }

            // for-await runs on the main actor; construct @MainActor Generation here
            var loaded: [Generation] = []
            for await rawData in group {
                guard let rawData else { continue }
                var gen = Generation(
                    metadata: rawData.metadata,
                    audioFileName: rawData.audioFileName,
                    audioDuration: rawData.audioDuration,
                    isFavorite: rawData.isFavorite
                )
                gen.waveformSamples = rawData.waveformSamples
                loaded.append(gen)
                generations = loaded.sorted { $0.audioFileName > $1.audioFileName }
            }
        }
    }

    // MARK: Favorites - stored in UserDefaults (only mutable state not in the file)

    private static let favoritesKey = "ttskit_favoriteGenerationIDs"

    private func loadedFavoriteIDs() -> Set<UUID> {
        let strings = UserDefaults.standard.stringArray(forKey: Self.favoritesKey) ?? []
        return Set(strings.compactMap { UUID(uuidString: $0) })
    }

    private func saveFavorites() {
        let ids = generations.filter(\.isFavorite).map(\.id.uuidString)
        UserDefaults.standard.set(ids, forKey: Self.favoritesKey)
    }

    // MARK: - Waveform

    /// Read an audio file from disk and return waveform peaks at token density.
    /// Returns `nil` if the file can't be read (missing, corrupt, etc.).
    nonisolated func waveformPeaks(from url: URL) -> [Float]? {
        guard FileManager.default.fileExists(atPath: url.path),
              let file = try? AVAudioFile(forReading: url) else { return nil }
        let frameCount = AVAudioFrameCount(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frameCount),
              (try? file.read(into: buffer)) != nil,
              let channelData = buffer.floatChannelData else { return nil }
        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: Int(buffer.frameLength)))
        return peaksPerToken(from: samples)
    }

    /// Resample raw audio into 1 peak per token (~80ms).
    /// Matches the fixed bar width used by WaveformView.
    nonisolated func peaksPerToken(from audioSamples: [Float]) -> [Float] {
        let samplesPerBar = Int(WaveformView.secondsPerBar * Double(Qwen3TTSConstants.sampleRate))
        guard samplesPerBar > 0, !audioSamples.isEmpty else { return [] }
        let barCount = (audioSamples.count + samplesPerBar - 1) / samplesPerBar
        return (0..<barCount).map { i in
            let start = i * samplesPerBar
            let end = min(start + samplesPerBar, audioSamples.count)
            return audioSamples[start..<end].reduce(Float(0)) { max($0, abs($1)) }
        }
    }

    /// Load waveform peaks for a generation, resampling from audio file if needed.
    func loadWaveform(for generation: Generation) {
        // If saved peaks match the expected token density, use them directly
        if let saved = generation.waveformSamples {
            let expectedBars = Int(generation.audioDuration / WaveformView.secondsPerBar)
            let tolerance = max(2, expectedBars / 10)
            if abs(saved.count - expectedBars) <= tolerance {
                currentWaveform = saved
                currentDuration = generation.audioDuration
                return
            }
        }

        // Otherwise regenerate from the audio file at the correct density
        let url = documentsDirectory.appendingPathComponent(generation.audioFileName)
        if let peaks = waveformPeaks(from: url) {
            currentWaveform = peaks
            currentDuration = generation.audioDuration
            return
        }

        // Fallback: use whatever we have
        currentWaveform = generation.waveformSamples ?? []
        currentDuration = generation.audioDuration
    }
}

// MARK: - TTSModelVariant Display Helpers

extension TTSModelVariant {
    var sizeEstimate: String {
        switch self {
            case .qwen3TTS_0_6b: return "~1 GB"
            case .qwen3TTS_1_7b: return "~2.2 GB"
        }
    }
}
