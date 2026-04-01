//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
import Hub

// MARK: - ModelDownloader

public struct ModelDownloadConfig: Sendable {
    public let downloadBase: String?
    public let modelRepo: String
    public let modelToken: String?
    public let modelFolder: String?
    public let useBackgroundSession: Bool
    public let endpoint: String
    public let revision: String

    public init(
        downloadBase: String? = nil,
        modelRepo: String,
        modelToken: String? = nil,
        modelFolder: String? = nil,
        useBackgroundSession: Bool = false,
        endpoint: String = "https://huggingface.co",
        revision: String = "main"
    ) {
        self.downloadBase = downloadBase
        self.modelRepo = modelRepo
        self.modelToken = modelToken
        self.modelFolder = modelFolder
        self.useBackgroundSession = useBackgroundSession
        self.endpoint = endpoint
        self.revision = revision
    }
}

/// Downloads models from a HuggingFace repository with per-model resolution.
///
/// Primary entry point is ``init(config:)`` with a ``ModelDownloadConfig`` (endpoint, revision, repo, token, etc.).
/// The convenience ``init(endpoint:repoName:modelToken:revision:useBackgroundSession:)`` builds that config for simple repo-only use.
open class ModelDownloader {
    private let config: ModelDownloadConfig

    /// Creates a downloader using a fully configured ``ModelDownloadConfig``.
    public init(config: ModelDownloadConfig) {
        self.config = config
    }

    /// Creates a downloader for a single HuggingFace repo without constructing a ``ModelDownloadConfig`` manually.
    /// - Parameters:
    ///   - endpoint: HuggingFace endpoint URL. Defaults to `https://huggingface.co`.
    ///   - repoName: Fully-qualified repo identifier (e.g. `"argmaxinc/whisperkit-coreml"`).
    ///   - modelToken: Optional HuggingFace access token for private repos.
    ///   - revision: Branch, tag, or commit SHA to download from. Defaults to `"main"`.
    ///   - useBackgroundSession: When `true`, downloads run in a background `URLSession` so they can continue while the app is suspended.
    public convenience init(endpoint: String = "https://huggingface.co",
                repoName: String,
                modelToken: String? = nil,
                revision: String = "main",
                useBackgroundSession: Bool = false) {
        let config = ModelDownloadConfig(
            modelRepo: repoName,
            modelToken: modelToken,
            useBackgroundSession: useBackgroundSession,
            endpoint: endpoint,
            revision: revision
        )
        self.init(config: config)
    }

    /// Downloads the files for a single model from the configured HuggingFace repo.
    ///
    /// Uses a glob path built from `modelInfo.name`, `version`, and `variant` to select only the
    /// relevant files within the repo.
    ///
    /// - Parameters:
    ///   - modelInfo: Identifies the model name, version, and variant to download.
    ///   - downloadBase: Override for the Hub cache root directory. `nil` falls back to the config value, then the Hub default.
    ///   - useOfflineMode: When `true`, only the local Hub cache is searched and no network request is made.
    ///     Pass `nil` to use the default online behaviour.
    /// - Returns: The Hub snapshot root URL containing the downloaded model files.
    public func downloadModel(modelInfo: ModelInfo, downloadBase: URL? = nil, useOfflineMode: Bool? = nil) async throws -> URL {
        let searchPath = "\(modelInfo.name)/\(modelInfo.version ?? "*")/\(modelInfo.variant ?? "*")/*"

        let resolvedDownloadBase = downloadBase ?? config.downloadBase.map { URL(fileURLWithPath: $0) }
        let hubApi = HubApi(downloadBase: resolvedDownloadBase, hfToken: config.modelToken, endpoint: config.endpoint, useBackgroundSession: config.useBackgroundSession, useOfflineMode: useOfflineMode)
        let repo = Hub.Repo(id: config.modelRepo, type: .models)

        if useOfflineMode ?? false {
            Logging.debug("[ModelDownloader] Searching for models matching \"\(searchPath)\" in \(repo)")
            let modelFiles = try await hubApi.getFilenames(from: repo, revision: config.revision, matching: [searchPath])

            guard !modelFiles.isEmpty else {
                throw ModelDownloaderError.modelUnavailable("No models found matching \"\(searchPath)\" in \(config.modelRepo)")
            }
        }

        Logging.debug("[ModelDownloader] Downloading model \(searchPath) with offline mode: \(useOfflineMode ?? false)")
        let modelFolder = try await hubApi.snapshot(from: repo, revision: config.revision, matching: [searchPath])
        return modelFolder
    }

    /// Returns the local directory where the Hub library caches this repo's files.
    ///
    /// The returned URL may not exist on disk if the repo has never been downloaded.
    /// - Parameter downloadBase: Override for the Hub cache root. `nil` falls back to the config value, then the Hub default.
    public func localRepoLocation(downloadBase: URL? = nil) -> URL {
        let resolvedDownloadBase = downloadBase ?? config.downloadBase.map { URL(fileURLWithPath: $0) }
        let hubApi = HubApi(downloadBase: resolvedDownloadBase, hfToken: config.modelToken, endpoint: config.endpoint, useBackgroundSession: config.useBackgroundSession)
        let repo = Hub.Repo(id: config.modelRepo, type: .models)
        return hubApi.localRepoLocation(repo)
    }

    /// Resolves a specific model file using a three-step fallback strategy:
    /// 1. Model folder (if provided)
    /// 2. Local cache
    /// 3. Online download
    open func resolveModel(
        _ modelFileName: String,
        using info: ModelInfo,
        modelFolder: URL? = nil,
        downloadBase: URL? = nil,
        download: Bool = true,
        modelStateCallback: ((ModelState) -> Void)? = nil
    ) async throws -> URL {
        let resolvedModelFolder = modelFolder ?? config.modelFolder.map { URL(fileURLWithPath: $0) }
        if let resolvedModelFolder {
            let folderURL = info.modelURL(baseURL: resolvedModelFolder)
            if let fileURL = modelURLIfExists(inFolder: folderURL, named: modelFileName) {
                Logging.debug("[ModelDownloader] Found \(modelFileName) in model folder")
                return fileURL
            }
        }

        let resolvedDownloadBase = downloadBase ?? config.downloadBase.map { URL(fileURLWithPath: $0) }

        // Try the local HuggingFace cache first (offline mode) to avoid a network round-trip.
        // Falls through to a real download below only if the file is not found in the cache.
        if let downloadedBase = try? await downloadModel(modelInfo: info, downloadBase: resolvedDownloadBase, useOfflineMode: true) {
            let folderURL = info.modelURL(baseURL: downloadedBase)
            if let url = modelURLIfExists(inFolder: folderURL, named: modelFileName) {
                Logging.debug("[ModelDownloader] Found existing \(modelFileName) in cache")
                modelStateCallback?(.downloaded)
                return url
            }
        }

        guard download else {
            throw ModelDownloaderError.modelUnavailable("No model found for \(modelFileName) (download disabled)")
        }

        modelStateCallback?(.downloading)
        let downloadedBase = try await downloadModel(modelInfo: info, downloadBase: resolvedDownloadBase)
        let folderURL = info.modelURL(baseURL: downloadedBase)
        if let url = modelURLIfExists(inFolder: folderURL, named: modelFileName) {
            Logging.debug("[ModelDownloader] Downloaded \(modelFileName)")
            modelStateCallback?(.downloaded)
            return url
        }

        throw ModelDownloaderError.modelUnavailable("No model found for \(modelFileName)")
    }

    /// String-returning variant of ``resolveModel(_:using:modelFolder:downloadBase:download:modelStateCallback:)``.
    ///
    /// Returns the resolved file's absolute path as a plain `String`.
    open func resolveModelPath(
        _ modelFileName: String,
        using info: ModelInfo,
        modelFolder: URL? = nil,
        downloadBase: URL? = nil,
        download: Bool = true,
        modelStateCallback: ((ModelState) -> Void)? = nil
    ) async throws -> String {
        try await resolveModel(
            modelFileName,
            using: info,
            modelFolder: modelFolder,
            downloadBase: downloadBase,
            download: download,
            modelStateCallback: modelStateCallback
        ).path
    }

    /// Resolves an entire repository in a single snapshot call.
    ///
    /// Resolution order:
    /// 1. Local Hub cache — if all required pattern directories exist, return the cache root immediately.
    /// 2. Online download — fetches all files matching `patterns` at this downloader’s `revision` in one `HubApi.snapshot()` call.
    ///
    /// - Parameters:
    ///   - patterns: Glob patterns selecting the files to download (e.g. `modelInfo.downloadPattern`).
    ///   - downloadBase: Override for the Hub cache root. Defaults to the Hub default location.
    ///   - download: When `false`, throws if models are not already cached locally.
    ///   - progressCallback: Called with the Hub’s `Progress` each time it updates.
    /// - Returns: The Hub snapshot root URL containing the downloaded files.
    open func resolveRepo(
        patterns: [String],
        downloadBase: URL? = nil,
        download: Bool = true,
        progressCallback: ((Progress) -> Void)? = nil
    ) async throws -> URL {
        let resolvedDownloadBase = downloadBase ?? config.downloadBase.map { URL(fileURLWithPath: $0) }
        let localRoot = localRepoLocation(downloadBase: resolvedDownloadBase)
        if patternsExistLocally(patterns, in: localRoot) {
            Logging.debug("[ModelDownloader] All models found in local cache at \(localRoot.path)")
            return localRoot
        }

        guard download else {
            throw ModelDownloaderError.modelUnavailable(
                "No local models found for repo '\(config.modelRepo)' and download is disabled."
            )
        }

        let hubApi = HubApi(downloadBase: resolvedDownloadBase, hfToken: config.modelToken, endpoint: config.endpoint, useBackgroundSession: config.useBackgroundSession)
        let repo = Hub.Repo(id: config.modelRepo, type: .models)

        Logging.info("[ModelDownloader] Downloading \(patterns.count) model(s) from \(config.modelRepo)...")
        let snapshotRoot = try await hubApi.snapshot(from: repo, revision: config.revision, matching: patterns, progressHandler: progressCallback ?? { _ in })
        Logging.info("[ModelDownloader] Download complete: \(snapshotRoot.path)")
        return snapshotRoot
    }

    /// Returns the list of filenames in the repository matching the given glob patterns.
    /// Uses this downloader’s `revision` so the result matches the branch/tag/commit used for downloads.
    /// - Parameters:
    ///   - patterns: Glob patterns to filter files.
    ///   - downloadBase: Override for the Hub cache root. `nil` uses the Hub library default.
    /// - Returns: Matching filenames at the configured revision.
    public func fetchFilenames(matching patterns: [String], downloadBase: URL? = nil) async throws -> [String] {
        let resolvedDownloadBase = downloadBase ?? config.downloadBase.map { URL(fileURLWithPath: $0) }
        let hubApi = HubApi(downloadBase: resolvedDownloadBase, hfToken: config.modelToken, endpoint: config.endpoint, useBackgroundSession: config.useBackgroundSession)
        let repo = Hub.Repo(id: config.modelRepo, type: .models)
        return try await hubApi.getFilenames(from: repo, revision: config.revision, matching: patterns)
    }

    /// Returns `true` when the deepest concrete directory for every pattern exists in `root` and is non-empty.
    ///
    /// Walks each pattern up to its first wildcard segment (`*` or `**`) and checks that the resulting
    /// directory exists and contains at least one file. Checking only the top-level component is
    /// insufficient when multiple patterns share the same root (e.g. `qwen3_tts/code_decoder/**` and
    /// `qwen3_tts/speech_decoder/**` both start with `qwen3_tts/`): a single partially-downloaded
    /// component would pass the cache check for all patterns and skip re-downloading missing files.
    private func patternsExistLocally(_ patterns: [String], in root: URL) -> Bool {
        patterns.allSatisfy { pattern in
            let concreteComponents = pattern
                .split(separator: "/")
                .prefix(while: { !$0.contains("*") })
                .map(String.init)
            guard !concreteComponents.isEmpty else { return false }
            let dir = concreteComponents.reduce(root) { $0.appendingPathComponent($1) }
            guard FileManager.default.fileExists(atPath: dir.path) else { return false }
            let contents = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
            return !contents.isEmpty
        }
    }

    /// Returns the URL of a compiled model file inside `folder` if it exists on disk, or `nil` if it does not.
    ///
    /// Delegates path detection to ``ModelUtilities/detectModelURL(inFolder:named:)`` before
    /// confirming existence with `FileManager`.
    /// - Parameters:
    ///   - folder: Directory to search within.
    ///   - name: Model filename (with or without extension).
    public func modelURLIfExists(inFolder folder: URL, named name: String) -> URL? {
        let candidate = ModelUtilities.detectModelURL(inFolder: folder, named: name)
        guard FileManager.default.fileExists(atPath: candidate.path) else {
            return nil
        }
        return candidate
    }
}

@frozen
public enum ModelDownloaderError: Error, LocalizedError {
    case modelUnavailable(String)

    public var errorDescription: String? {
        switch self {
        case .modelUnavailable(let msg): return msg
        }
    }
}


// MARK: - ModelInfo

/// Metadata needed to identify and configure a model for download and execution.
public struct ModelInfo: CustomStringConvertible, CustomDebugStringConvertible, Sendable {
    public let version: String?
    public let variant: String?
    public let name: String
    public let computeUnits: MLComputeUnits

    public init(version: String? = nil, variant: String? = nil, name: String, computeUnits: MLComputeUnits) {
        self.version = version
        self.variant = variant
        self.name = name
        self.computeUnits = computeUnits
    }

    /// Compute model path based on model folder, name, version, and variant
    public func modelURL(baseURL: URL) -> URL {
        var result = baseURL.appendingPathComponent(name)
        if let version = version {
            result = result.appendingPathComponent(version)
        }
        if let variant = variant {
            result = result.appendingPathComponent(variant)
        }
        return result
    }

    /// Glob pattern for selecting all files belonging to this model within a HuggingFace repo.
    public var downloadPattern: String {
        "\(name)/\(version ?? "*")/\(variant ?? "*")/*"
    }

    public var description: String {
        [name, version, variant].compactMap { $0 }.joined(separator: "/")
    }

    public var debugDescription: String {
        "ModelInfo(name: \(name), version: \(version ?? "nil"), variant: \(variant ?? "nil"), computeUnit: \(computeUnits.description))"
    }

    /// Finds the base folder by traversing up the URL path until finding a directory with the model name
    public func findBaseFolder(in url: URL) -> URL? {
        var currentURL = url
        while currentURL.pathComponents.count > 1 {
            if currentURL.lastPathComponent == name {
                return currentURL.deletingLastPathComponent()
            }
            currentURL.deleteLastPathComponent()
        }
        return nil
    }
}
