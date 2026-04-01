//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

/// Protocol for loading, unloading, and resolving ML model files.
///
/// Implement this protocol to provide backend-specific model handling.
/// Pass your implementation to ``ModelManager/init(loader:downloader:)``
/// instead of subclassing `ModelManager`.
///
/// Path values use plain `String` rather than `URL` for maximum portability.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public protocol ModelLoader: AnyObject, Sendable {
    /// Caller-provided local model directory path. When non-nil, ``ModelManager`` skips downloading.
    var modelFolder: String? { get }

    /// Resolve or download model files and return the local directory path.
    /// Called by ``ModelManager/downloadModels(progressCallback:)``.
    func resolveModels(downloader: ModelDownloader, progressCallback: ((Progress) -> Void)?) async throws -> String

    /// Load or prewarm models from a resolved path string.
    /// Called by ``ModelManager/loadModels()`` and ``ModelManager/prewarmModels()``.
    func load(from modelPath: String, prewarm: Bool) async throws

    /// Release loaded model weights from memory.
    /// Called by ``ModelManager/unloadModels()``.
    func unload() async
}

/// Manages the download -> load -> unload lifecycle of ML models.
///
/// Delegates backend-specific behaviour to a ``ModelLoader``. The lifecycle
/// state machine (``modelState`` transitions, error handling, concurrency
/// coalescing) lives here; the loader handles the actual I/O.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
open class ModelManager: @unchecked Sendable {
    /// Current lifecycle state of the managed models.
    /// Transitions:
    /// `.unloaded` -> `.downloading` -> `.downloaded` -> `.loading` -> `.loaded`
    /// `.unloaded` -> `.prewarming` -> `.prewarmed`
    public private(set) var modelState: ModelState = .unloaded {
        didSet {
            guard oldValue != modelState else { return }
            modelStateCallback?(oldValue, modelState)
        }
    }

    /// Called whenever `modelState` transitions to a new value.
    public var modelStateCallback: ((ModelState, ModelState) -> Void)?

    /// Local path where models were resolved (set after a successful `downloadModels()` call).
    public private(set) var modelPath: URL?

    /// Caller-provided local model directory, forwarded from the loader.
    public var modelFolder: URL? { loader.modelFolder.map { URL(fileURLWithPath: $0) } }

    /// Downloader used for model resolution.
    public let downloader: ModelDownloader

    /// The loader that handles backend-specific resolve / load / unload.
    public let loader: ModelLoader

    private lazy var loggerName: String = "\(type(of: self))"

    private let coordinator = LoadModelsCoordinator()

    public init(loader: ModelLoader, downloader: ModelDownloader) {
        self.loader = loader
        self.downloader = downloader
    }

    /// Downloads (if needed) and loads models. Concurrent callers coalesce onto a single task;
    /// on failure the task is cleared so the next caller can retry.
    public func ensureModelsLoaded() async throws {
        guard !isLoaded else { return }
        try await coordinator.run { [self] in
            guard !self.isLoaded else { return }
            try await self.downloadModels()
            try await self.loadModels()
        }
    }

    /// Downloads all required models into the local cache.
    ///
    /// Skips download if `modelState` is not `.unloaded`. After a successful call,
    /// `modelState` transitions to `.downloaded` and `modelPath` is set.
    public func downloadModels(progressCallback: ((Progress) -> Void)? = nil) async throws {
        guard modelState == .unloaded else {
            Logging.debug("[\(loggerName)] Models already downloaded (state: \(modelState)), skipping download")
            return
        }

        if let modelFolder {
            Logging.debug("[\(loggerName)] modelFolder is set, skipping download")
            modelPath = modelFolder
            modelState = .downloaded
            return
        }

        Logging.info("[\(loggerName)] Starting download process...")
        modelState = .downloading

        do {
            let pathString = try await loader.resolveModels(downloader: downloader, progressCallback: progressCallback)
            modelPath = URL(fileURLWithPath: pathString)
            modelState = .downloaded
            Logging.info("[\(loggerName)] Models downloaded to \(pathString)")
        } catch {
            modelState = .unloaded
            Logging.error("[\(loggerName)] Failed to download models: \(error)")
            throw error
        }
    }

    /// Compiles models into the OS cache without retaining weights in memory.
    ///
    /// Call before `loadModels()` on first launch or after a model update to minimize peak memory
    /// during compilation. Downloads automatically if models have not been downloaded yet.
    /// After a successful call, `modelState` is `.prewarmed`.
    public func prewarmModels() async throws {
        if modelState == .unloaded {
            try await downloadModels()
        }

        guard let pathString = modelPath?.path ?? loader.modelFolder else {
            throw ArgmaxCoreError.invalidConfiguration(
                "No model path available. Call downloadModels() first, or provide modelFolder."
            )
        }

        switch modelState {
        case .prewarmed:
            Logging.debug("[\(loggerName)] Models already prewarmed, skipping")
            return
        case .downloaded:
            break
        case .unloaded, .loaded, .downloading, .loading, .prewarming, .unloading:
            throw ArgmaxCoreError.invalidConfiguration("Cannot prewarm in state: \(modelState)")
        }

        modelState = .prewarming

        do {
            try await loader.load(from: pathString, prewarm: true)
            modelState = .prewarmed
            Logging.info("[\(loggerName)] Models prewarmed successfully")
        } catch {
            modelState = .unloaded
            Logging.error("[\(loggerName)] Failed to prewarm models: \(error)")
            throw error
        }
    }

    /// Loads models into memory for inference.
    ///
    /// Downloads automatically if models have not been downloaded yet.
    /// After a successful call, `modelState` is `.loaded`.
    public func loadModels() async throws {
        if modelState == .unloaded {
            try await downloadModels()
        }

        guard let pathString = modelPath?.path ?? loader.modelFolder else {
            throw ArgmaxCoreError.invalidConfiguration(
                "No model path available. Call downloadModels() first, or provide modelFolder."
            )
        }

        switch modelState {
        case .loaded:
            Logging.debug("[\(loggerName)] Models already loaded, skipping")
            return
        case .downloaded, .prewarmed:
            break
        case .unloaded, .downloading, .loading, .prewarming, .unloading:
            throw ArgmaxCoreError.invalidConfiguration("Cannot load in state: \(modelState)")
        }

        modelState = .loading

        do {
            try await loader.load(from: pathString, prewarm: false)
            modelState = .loaded
            Logging.info("[\(loggerName)] Models loaded successfully")
        } catch {
            modelState = .unloaded
            Logging.error("[\(loggerName)] Failed to load models: \(error)")
            throw error
        }
    }

    /// Releases loaded models from memory. No-op if models are not currently loaded or prewarmed.
    public func unloadModels() async {
        guard modelState == .loaded || modelState == .prewarmed else { return }
        modelState = .unloading
        await loader.unload()
        modelState = .unloaded
    }

    /// `true` when models have been downloaded or are in the process of loading/being loaded.
    public var isAvailable: Bool {
        modelState == .downloaded || modelState == .loading || modelState == .loaded
    }

    /// `true` only when models are fully loaded and ready for inference.
    public var isLoaded: Bool {
        modelState == .loaded
    }
}

// Coalesces concurrent ensureModelsLoaded() callers onto a single in-flight Task.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
private actor LoadModelsCoordinator {
    var inflightLoad: Task<Void, Error>?

    func run(_ work: @Sendable @escaping () async throws -> Void) async throws {
        if let existing = inflightLoad {
            try await existing.value
            return
        }
        let loadTask = Task { try await work() }
        inflightLoad = loadTask
        do {
            try await loadTask.value
            inflightLoad = nil
        } catch {
            inflightLoad = nil
            throw error
        }
    }
}
