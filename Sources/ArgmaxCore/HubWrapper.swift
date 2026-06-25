//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

// MARK: - Public HubApiWrapper

/// Client for downloading models and tokenizers from the Hugging Face Hub.
///
/// Use this to download WhisperKit models, fetch available model lists, and locate
/// cached model files on disk. It wraps the internal HubApi implementation so that
/// callers outside ArgmaxCore have no dependency on the internal Hub or HubApi types.
public struct HubApiWrapper: Sendable {
    private let impl: HubApi

    /// Provides access to the underlying HubApi for same-module callers (e.g., Hub.swift, Tokenizer.swift).
    var hubApi: HubApi { impl }

    /// Creates a Hub API client.
    ///
    /// - Parameters:
    ///   - downloadBase: Base directory for cached downloads (defaults to Documents/huggingface)
    ///   - hfToken: Hugging Face token for private or gated repos (defaults to HF_TOKEN env var)
    ///   - endpoint: Hub endpoint URL (defaults to https://huggingface.co)
    ///   - useBackgroundSession: Use a background URLSession for downloads
    public init(
        downloadBase: URL? = nil,
        hfToken: String? = nil,
        endpoint: String? = nil,
        useBackgroundSession: Bool = false
    ) {
        impl = HubApi(
            downloadBase: downloadBase,
            hfToken: hfToken,
            endpoint: endpoint,
            useBackgroundSession: useBackgroundSession
        )
    }

    /// Shared client with default configuration.
    public static let shared = HubApiWrapper()

    /// The type of Hugging Face repository.
    public enum RepoType: String, Codable, Sendable {
        /// Model repositories (e.g. argmaxinc/whisperkit-coreml).
        case models
        /// Dataset repositories used for evaluation and regression testing.
        case datasets
        case spaces
    }

    /// A reference to a repository on the Hugging Face Hub.
    public struct Repo: Codable, Sendable, Hashable {
        /// Repository identifier (e.g. "argmaxinc/whisperkit-coreml").
        public let id: String
        /// Repository type.
        public let type: RepoType

        /// - Parameters:
        ///   - id: Repository identifier
        ///   - type: Repository type (defaults to .models)
        public init(id: String, type: RepoType = .models) {
            self.id = id
            self.type = type
        }
    }

    /// Downloads matching files from a repository and returns the local snapshot URL.
    ///
    /// - Parameters:
    ///   - repo: The repository to download from
    ///   - revision: Git revision (defaults to "main")
    ///   - globs: File patterns to include (empty means all files)
    ///   - progressHandler: Called periodically with download progress
    /// - Returns: Local directory URL containing the downloaded files
    public func snapshot(
        from repo: Repo,
        revision: String = "main",
        matching globs: [String] = [],
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws -> URL {
        try await impl.snapshot(from: repo.asHubApiRepo, revision: revision, matching: globs, progressHandler: progressHandler)
    }

    /// Returns file names in a repository that match the given patterns.
    ///
    /// - Parameters:
    ///   - repo: The repository to query
    ///   - revision: Git revision (defaults to "main")
    ///   - globs: File patterns to match (empty means all files)
    /// - Returns: Matching file names relative to the repository root
    public func getFilenames(
        from repo: Repo,
        revision: String = "main",
        matching globs: [String] = []
    ) async throws -> [String] {
        try await impl.getFilenames(from: repo.asHubApiRepo, revision: revision, matching: globs)
    }

    /// Returns the local cache directory for a repository.
    ///
    /// - Parameter repo: The repository to locate
    /// - Returns: Local directory URL under the download base (may not exist yet)
    public func localRepoLocation(_ repo: Repo) -> URL {
        impl.localRepoLocation(repo.asHubApiRepo)
    }
}

// MARK: - Internal bridge

// HubApi.Repo and HubApi.RepoType are defined here as internal nested types so that
// HubApi.swift method signatures (snapshot, getFilenames, localRepoLocation, etc.) can
// reference `Repo` unqualified without change. The asHubApiRepo bridge below converts
// the public HubApiWrapper.Repo into HubApi.Repo when forwarding calls to the impl.
extension HubApi {
    enum RepoType: String, Codable, Sendable {
        case models
        case datasets
        case spaces
    }

    struct Repo: Codable, Sendable, Hashable {
        let id: String
        let type: RepoType

        init(id: String, type: RepoType = .models) {
            self.id = id
            self.type = type
        }
    }
}

extension HubApiWrapper.Repo {
    /// Converts to the internal HubApi.Repo type for passing to HubApi methods.
    var asHubApiRepo: HubApi.Repo {
        let repoType: HubApi.RepoType
        switch type {
            case .models: repoType = .models
            case .datasets: repoType = .datasets
            case .spaces: repoType = .spaces
        }
        return HubApi.Repo(id: id, type: repoType)
    }
}
