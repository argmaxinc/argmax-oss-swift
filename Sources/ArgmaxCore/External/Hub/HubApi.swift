// Originally from: https://github.com/huggingface/swift-transformers
// Version: 1.1.6 (commit: 573e5c9036c2f136b3a8a071da8e8907322403d0)
// License: Apache 2.0 (https://github.com/huggingface/swift-transformers/blob/main/LICENSE)
// Copyright 2022 Hugging Face SAS
// Modified by Argmax, Inc. See Argmax-modification: comments for changes.
//

//
//  HubApi.swift
//
//
//  Created by Pedro Cuenca on 20231230.
//

import CryptoKit
import Foundation
import Network
import os

/// https://datatracker.ietf.org/doc/html/rfc7540#section-8.1.2
/// `requests` in Python leaves headers as their original casing,
/// where as Swift strictly adheres to RFC 7540 and can force lower case.
/// This is relevant for Xet
enum HFHttpHeaders {
    static let location = "Location"
    static let etag = "Etag"
    static let contentLength = "Content-Length"
    static let repoCommit = "X-Repo-Commit"
    static let linkedEtag = "X-Linked-Etag"
    static let linkedSize = "X-Linked-Size"
    // Argmax-modification: removed xetHash, xetRefreshRoute, linkXetAuthKey — Xet protocol unused in this repo
}

// Argmax-modification: removed XetFileData struct — Xet protocol unused in this repo

/// `requests` automatically parses Link headers into `response.links`,
///  we implement similar functionality here.
extension HTTPURLResponse {
    func getLinkURL(for rel: String) -> String? {
        guard let linkHeader = allHeaderFields["Link"] as? String else {
            return nil
        }

        for link in linkHeader.split(separator: ",") {
            let trimmed = link.trimmingCharacters(in: .whitespaces)

            if trimmed.contains("rel=\"\(rel)\"") || trimmed.contains("rel=\(rel)") {
                if let start = trimmed.firstIndex(of: "<"),
                    let end = trimmed.firstIndex(of: ">"),
                    start < end
                {
                    let startIndex = trimmed.index(after: start)
                    return String(trimmed[startIndex..<end])
                }
            }
        }

        return nil
    }
}

/// Client for interacting with the Hugging Face Hub API.
///
/// HubApi provides methods for downloading files, retrieving metadata, managing repositories,
/// and handling authentication with the Hugging Face Hub. It supports offline mode,
/// background downloads, and automatic retry mechanisms for robust file transfers.
// Argmax-modification: HubApi made internal — public surface replaced by HubApiWrapper (see HubWrapper.swift)
struct HubApi: Sendable {
    var downloadBase: URL
    var hfToken: String?
    var endpoint: String
    var useBackgroundSession: Bool
    var useOfflineMode: Bool?

    private let networkMonitor = NetworkMonitor()
    // Argmax-modification: removed typealias Repo/RepoType — replaced by proper public types in HubWrapper.swift

    /// Session actor for metadata requests with relative redirect handling (used in HEAD requests).
    ///
    /// Static to share a single URLSession across all HubApi instances, preventing resource
    /// exhaustion when many instances are created. Persists for process lifetime.
    private static let redirectSession: RedirectSessionActor = .init()

    /// Initializes a new Hub API client.
    ///
    /// - Parameters:
    ///   - downloadBase: The base directory for downloads (defaults to Documents/huggingface)
    ///   - hfToken: The Hugging Face authentication token (defaults to environment variable)
    ///   - endpoint: The Hub endpoint URL (defaults to https://huggingface.co)
    ///   - useBackgroundSession: Whether to use background URL sessions for downloads
    ///   - useOfflineMode: Override for offline mode detection (defaults to automatic detection)
    init(
        downloadBase: URL? = nil,
        hfToken: String? = nil,
        endpoint: String? = nil,
        useBackgroundSession: Bool = false,
        useOfflineMode: Bool? = nil
    ) {
        self.hfToken = hfToken ?? Self.hfTokenFromEnv()

        let debugPrint = ProcessInfo.processInfo.environment["CI_DISABLE_NETWORK_MONITOR"] == "1"
        if debugPrint {
            print(self.hfToken == nil ? "🔴 NO TOKEN **" : "✅ got token")
        }
        if let downloadBase {
            self.downloadBase = downloadBase
        } else {
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            self.downloadBase = documents.appending(component: "huggingface")
        }
        self.endpoint = endpoint ?? Self.hfEndpointfromEnv()
        self.useBackgroundSession = useBackgroundSession
        self.useOfflineMode = useOfflineMode
        NetworkMonitor.shared.startMonitoring()
    }

    let sha256Pattern = "^[0-9a-f]{64}$"
    let commitHashPattern = "^[0-9a-f]{40}$"

    /// The shared Hub API instance with default configuration.
    static let shared = HubApi()

    private static let logger = Logger()
}

private extension HubApi {
    static func hfEndpointfromEnv() -> String {
        ProcessInfo.processInfo.environment["HF_ENDPOINT"] ?? "https://huggingface.co"
    }

    static func hfTokenFromEnv() -> String? {
        let possibleTokens = [
            { ProcessInfo.processInfo.environment["HF_TOKEN"] },
            { ProcessInfo.processInfo.environment["HUGGING_FACE_HUB_TOKEN"] },
            {
                ProcessInfo.processInfo.environment["HF_TOKEN_PATH"].flatMap {
                    try? String(
                        contentsOf: URL(filePath: NSString(string: $0).expandingTildeInPath),
                        encoding: .utf8
                    )
                }
            },
            {
                ProcessInfo.processInfo.environment["HF_HOME"].flatMap {
                    try? String(
                        contentsOf: URL(filePath: NSString(string: $0).expandingTildeInPath).appending(path: "token"),
                        encoding: .utf8
                    )
                }
            },
            { try? String(contentsOf: .homeDirectory.appending(path: ".cache/huggingface/token"), encoding: .utf8) },
            { try? String(contentsOf: .homeDirectory.appending(path: ".huggingface/token"), encoding: .utf8) },
        ]
        return possibleTokens
            .lazy
            .compactMap { $0() }
            .filter { !$0.isEmpty }
            .first
    }
}

/// File retrieval
extension HubApi {
    /// Represents a file in a repository.
    ///
    /// Contains metadata about files available in a Hub repository,
    /// used for file discovery and listing operations.
    struct Sibling: Codable {
        /// The relative filename within the repository.
        let rfilename: String
    }

    /// Response structure for repository file listings.
    ///
    /// Contains the list of files available in a repository,
    /// returned by the Hub API when querying repository contents.
    struct SiblingsResponse: Codable {
        /// Array of files in the repository.
        let siblings: [Sibling]
    }

    /// Performs an HTTP GET request with authentication and error handling.
    ///
    /// - Parameter url: The URL to request
    /// - Returns: A tuple containing the response data and HTTP response
    /// - Throws: HubClientError for authentication, network, or HTTP errors
    func httpGet(for url: URL) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url)
        if let hfToken, !hfToken.isEmpty {
            request.setValue("Bearer \(hfToken)", forHTTPHeaderField: "Authorization")
        }

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw Hub.HubClientError.unexpectedError
            }

            switch httpResponse.statusCode {
            case 200..<300:
                return (data, httpResponse)
            case 401, 403:
                throw Hub.HubClientError.authorizationRequired
            case 404:
                throw Hub.HubClientError.fileNotFound(url.lastPathComponent)
            default:
                throw Hub.HubClientError.httpStatusCode(httpResponse.statusCode)
            }
        } catch let error as Hub.HubClientError {
            throw error
        } catch {
            throw Hub.HubClientError.downloadError(error.localizedDescription)
        }
    }

    /// Performs an HTTP HEAD request to retrieve metadata without downloading content.
    ///
    /// Uses a shared URLSession with custom redirect handling that only allows relative redirects
    /// and blocks absolute redirects (important for LFS file security).
    ///
    /// - Parameter url: The URL to request
    /// - Returns: The HTTP response containing headers and status code
    /// - Throws: HubClientError if the page does not exist or is not accessible
    func httpHead(for url: URL) async throws -> HTTPURLResponse {
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        if let hfToken, !hfToken.isEmpty {
            request.setValue("Bearer \(hfToken)", forHTTPHeaderField: "Authorization")
        }
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")

        // Use shared session with redirect handling to avoid creating multiple URLSession instances
        let session = await Self.redirectSession.get()
        let (_, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse else { throw Hub.HubClientError.unexpectedError }

        switch response.statusCode {
        case 200..<400: break // Allow redirects to pass through to the redirect delegate
        case 401, 403: throw Hub.HubClientError.authorizationRequired
        case 404: throw Hub.HubClientError.fileNotFound(url.lastPathComponent)
        default: throw Hub.HubClientError.httpStatusCode(response.statusCode)
        }

        return response
    }

    /// Retrieves the list of filenames in a repository that match the specified glob patterns.
    ///
    /// - Parameters:
    ///   - repo: The repository to query
    ///   - revision: The git revision to use (defaults to "main")
    ///   - globs: Array of glob patterns to filter files (empty array returns all files)
    /// - Returns: Array of matching filenames
    /// - Throws: HubClientError if the repository cannot be accessed or parsed
    func getFilenames(from repo: Repo, revision: String = "main", matching globs: [String] = []) async throws -> [String] {
        // Read repo info and only parse "siblings"
        let url = URL(string: endpoint)!
            .appending(path: "api")
            .appending(path: repo.type.rawValue)
            .appending(path: repo.id)
            .appending(path: "revision")
            .appending(component: revision) // Encode slashes (e.g., "pr/1" -> "pr%2F1")
        let (data, _) = try await httpGet(for: url)
        let response = try JSONDecoder().decode(SiblingsResponse.self, from: data)
        let filenames = response.siblings.map { $0.rfilename }
        guard globs.count > 0 else { return filenames }

        var selected: Set<String> = []
        for glob in globs {
            selected = selected.union(filenames.matching(glob: glob))
        }
        return Array(selected)
    }

    func getFilenames(from repoId: String, matching globs: [String] = []) async throws -> [String] {
        try await getFilenames(from: Repo(id: repoId), matching: globs)
    }

    func getFilenames(from repo: Repo, matching glob: String) async throws -> [String] {
        try await getFilenames(from: repo, matching: [glob])
    }

    func getFilenames(from repoId: String, matching glob: String) async throws -> [String] {
        try await getFilenames(from: Repo(id: repoId), matching: [glob])
    }
}

/// Additional Errors
extension HubApi {
    enum EnvironmentError: LocalizedError {
        case invalidMetadataError(String)
        case offlineModeError(String)
        case fileIntegrityError(String)
        case fileWriteError(String)

        var errorDescription: String? {
            switch self {
            case let .invalidMetadataError(message):
                String(localized: "Invalid metadata: \(message)")
            case let .offlineModeError(message):
                String(localized: "Offline mode error: \(message)")
            case let .fileIntegrityError(message):
                String(localized: "File integrity check failed: \(message)")
            case let .fileWriteError(message):
                String(localized: "Failed to write file: \(message)")
            }
        }
    }
}

/// Configuration loading helpers
extension HubApi {
    /// Assumes the file has already been downloaded.
    /// `filename` is relative to the download base.
    func configuration(from filename: String, in repo: Repo) throws -> Config {
        let fileURL = localRepoLocation(repo).appending(path: filename)
        return try configuration(fileURL: fileURL)
    }

    /// Assumes the file is already present at local url.
    /// `fileURL` is a complete local file path for the given model
    func configuration(fileURL: URL) throws -> Config {
        let data = try Data(contentsOf: fileURL)
        guard let parsed = try? JSONSerialization.bomPreservingJsonObject(with: data) else {
            throw Hub.HubClientError.jsonSerialization(fileURL: fileURL, message: "JSON Serialization failed for \(fileURL). Please verify that you have set the HF_TOKEN environment variable.")
        }
        guard let dictionary = parsed as? [NSString: Any] else { throw Hub.HubClientError.parse }
        return Config(dictionary)
    }
}

/// Whoami
extension HubApi {
    func whoami() async throws -> Config {
        guard hfToken != nil else { throw Hub.HubClientError.authorizationRequired }

        let url = URL(string: endpoint)!
            .appending(path: "api")
            .appending(path: "whoami-v2")
        let (data, _) = try await httpGet(for: url)

        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = parsed as? [NSString: Any] else { throw Hub.HubClientError.parse }
        return Config(dictionary)
    }
}

/// Snaphsot download
extension HubApi {
    func localRepoLocation(_ repo: Repo) -> URL {
        downloadBase.appending(component: repo.type.rawValue).appending(component: repo.id)
    }

    /// Reads metadata about a file in the local directory related to a download process.
    ///
    /// Reference: https://github.com/huggingface/huggingface_hub/blob/b2c9a148d465b43ab90fab6e4ebcbbf5a9df27d4/src/huggingface_hub/_local_folder.py#L263
    ///
    /// - Parameter metadataPath: The path of the metadata file to read.
    /// - Throws: An `EnvironmentError.invalidMetadataError` if the metadata file is invalid and cannot be removed.
    /// - Returns: A `LocalDownloadFileMetadata` object if the metadata file exists and is valid, or `nil` if the file is missing or invalid.
    func readDownloadMetadata(metadataPath: URL) throws -> LocalDownloadFileMetadata? {
        if FileManager.default.fileExists(atPath: metadataPath.path) {
            do {
                let contents = try String(contentsOf: metadataPath, encoding: .utf8)
                let lines = contents.components(separatedBy: .newlines)

                guard lines.count >= 3 else {
                    throw EnvironmentError.invalidMetadataError(String(localized: "Metadata file is missing required fields"))
                }

                let commitHash = lines[0].trimmingCharacters(in: .whitespacesAndNewlines)
                let etag = lines[1].trimmingCharacters(in: .whitespacesAndNewlines)

                guard let timestamp = Double(lines[2].trimmingCharacters(in: .whitespacesAndNewlines)) else {
                    throw EnvironmentError.invalidMetadataError(String(localized: "Invalid timestamp format"))
                }

                let timestampDate = Date(timeIntervalSince1970: timestamp)
                let filename = metadataPath.lastPathComponent.replacingOccurrences(of: ".metadata", with: "")

                return LocalDownloadFileMetadata(commitHash: commitHash, etag: etag, filename: filename, timestamp: timestampDate)
            } catch let error as EnvironmentError {
                do {
                    HubApi.logger.warning("Invalid metadata file \(metadataPath): \(error.localizedDescription). Removing it from disk and continuing.")
                    try FileManager.default.removeItem(at: metadataPath)
                } catch {
                    throw EnvironmentError.invalidMetadataError(String(localized: "Could not remove corrupted metadata file: \(error.localizedDescription)"))
                }
                return nil
            } catch {
                do {
                    HubApi.logger.warning("Error reading metadata file \(metadataPath): \(error.localizedDescription). Removing it from disk and continuing.")
                    try FileManager.default.removeItem(at: metadataPath)
                } catch {
                    throw EnvironmentError.invalidMetadataError(String(localized: "Could not remove corrupted metadata file: \(error.localizedDescription)"))
                }
                return nil
            }
        }

        // metadata file does not exist
        return nil
    }

    func isValidHash(hash: String, pattern: String) -> Bool {
        let regex = try? NSRegularExpression(pattern: pattern)
        let range = NSRange(location: 0, length: hash.utf16.count)
        return regex?.firstMatch(in: hash, options: [], range: range) != nil
    }

    func computeFileHash(file url: URL) throws -> String {
        // Open file for reading
        guard let fileHandle = try? FileHandle(forReadingFrom: url) else {
            throw Hub.HubClientError.fileNotFound(url.lastPathComponent)
        }

        defer {
            try? fileHandle.close()
        }

        var hasher = SHA256()
        let chunkSize = 1024 * 1024 // 1MB chunks

        while autoreleasepool(invoking: {
            let nextChunk = try? fileHandle.read(upToCount: chunkSize)

            guard let nextChunk,
                !nextChunk.isEmpty
            else {
                return false
            }

            hasher.update(data: nextChunk)

            return true
        }) {}

        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Reference: https://github.com/huggingface/huggingface_hub/blob/b2c9a148d465b43ab90fab6e4ebcbbf5a9df27d4/src/huggingface_hub/_local_folder.py#L391
    func writeDownloadMetadata(commitHash: String, etag: String, metadataPath: URL) throws {
        let metadataContent = "\(commitHash)\n\(etag)\n\(Date().timeIntervalSince1970)\n"
        do {
            try FileManager.default.createDirectory(at: metadataPath.deletingLastPathComponent(), withIntermediateDirectories: true)
            try metadataContent.write(to: metadataPath, atomically: true, encoding: .utf8)
        } catch {
            throw EnvironmentError.fileWriteError(String(localized: "Failed to write metadata to \(metadataPath.path): \(error.localizedDescription)"))
        }
    }

    struct HubFileDownloader {
        let hub: HubApi
        let repo: Repo
        let revision: String
        let repoDestination: URL
        let repoMetadataDestination: URL
        let relativeFilename: String
        let hfToken: String?
        let endpoint: String?
        let backgroundSession: Bool

        var source: URL {
            // https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/tokenizer.json?download=true
            var url = URL(string: endpoint ?? "https://huggingface.co")!
            if repo.type != .models {
                url = url.appending(path: repo.type.rawValue)
            }
            url = url.appending(path: repo.id)
            url = url.appending(path: "resolve")
            url = url.appending(component: revision) // Encode slashes (e.g., "pr/1" -> "pr%2F1")
            url = url.appending(path: relativeFilename)
            return url
        }

        var destination: URL {
            repoDestination.appending(path: relativeFilename)
        }

        var metadataDestination: URL {
            repoMetadataDestination.appending(path: relativeFilename + ".metadata")
        }

        var downloaded: Bool {
            FileManager.default.fileExists(atPath: destination.path)
        }

        /// We're using incomplete destination to prepare cache destination because incomplete files include lfs + non-lfs files (vs only lfs for metadata files)
        func prepareCacheDestination(_ incompleteDestination: URL) throws {
            let directoryURL = incompleteDestination.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true, attributes: nil)
            if !FileManager.default.fileExists(atPath: incompleteDestination.path) {
                try "".write(to: incompleteDestination, atomically: true, encoding: .utf8)
            }
        }

        /// Downloads the file with progress tracking.
        /// - Parameter progressHandler: Called with download progress (0.0-1.0) and speed in bytes/sec, if available.
        /// - Returns: Local file URL (uses cached file if commit hash matches).
        /// - Throws: ``EnvironmentError`` errors for file and metadata validation failures, ``Downloader.DownloadError`` errors during transfer, or ``CancellationError`` if the task is cancelled.
        @discardableResult
        func download(progressHandler: @escaping (Double, Double?) -> Void) async throws -> URL {
            let localMetadata = try hub.readDownloadMetadata(metadataPath: metadataDestination)
            let remoteMetadata = try await hub.getFileMetadata(url: source)

            let localCommitHash = localMetadata?.commitHash ?? ""
            let remoteCommitHash = remoteMetadata.commitHash ?? ""

            // Local file exists + metadata exists + commit_hash matches => return file
            if hub.isValidHash(hash: remoteCommitHash, pattern: hub.commitHashPattern), downloaded, localMetadata != nil,
                localCommitHash == remoteCommitHash
            {
                return destination
            }

            // From now on, etag, commit_hash, url and size are not empty
            guard let remoteCommitHash = remoteMetadata.commitHash,
                let remoteEtag = remoteMetadata.etag,
                let remoteSize = remoteMetadata.size,
                remoteMetadata.location != ""
            else {
                throw EnvironmentError.invalidMetadataError("File metadata must have been retrieved from server")
            }

            // Local file exists => check if it's up-to-date
            if downloaded {
                // etag matches => update metadata and return file
                if localMetadata?.etag == remoteEtag {
                    try hub.writeDownloadMetadata(commitHash: remoteCommitHash, etag: remoteEtag, metadataPath: metadataDestination)
                    return destination
                }

                // etag is a sha256
                // => means it's an LFS file (large)
                // => let's compute local hash and compare
                // => if match, update metadata and return file
                if hub.isValidHash(hash: remoteEtag, pattern: hub.sha256Pattern) {
                    let fileHash = try hub.computeFileHash(file: destination)
                    if fileHash == remoteEtag {
                        try hub.writeDownloadMetadata(commitHash: remoteCommitHash, etag: remoteEtag, metadataPath: metadataDestination)
                        return destination
                    }
                }
            }

            // Otherwise, let's download the file!
            let incompleteDestination = repoMetadataDestination.appending(path: relativeFilename + ".\(remoteEtag).incomplete")
            try prepareCacheDestination(incompleteDestination)

            let downloader = Downloader(to: destination, incompleteDestination: incompleteDestination, inBackground: backgroundSession)

            try await withTaskCancellationHandler {
                let sub = await downloader.download(from: source, using: hfToken, expectedSize: remoteSize)
                listen: for await state in sub {
                    switch state {
                    case .notStarted:
                        continue
                    case let .downloading(progress, speed):
                        progressHandler(progress, speed)
                    case let .failed(error):
                        throw error
                    case .completed:
                        break listen
                    }
                }
            } onCancel: {
                Task {
                    await downloader.cancel()
                }
            }

            try hub.writeDownloadMetadata(commitHash: remoteCommitHash, etag: remoteEtag, metadataPath: metadataDestination)

            return destination
        }
    }

    @discardableResult
    func snapshot(from repo: Repo, revision: String = "main", matching globs: [String] = [], progressHandler: @escaping (Progress) -> Void = { _ in })
        async throws -> URL
    {
        let repoDestination = localRepoLocation(repo)
        let repoMetadataDestination =
            repoDestination
            .appending(path: ".cache")
            .appending(path: "huggingface")
            .appending(path: "download")

        let shouldUseOfflineMode = await NetworkMonitor.shared.state.shouldUseOfflineMode()

        if useOfflineMode ?? shouldUseOfflineMode {
            if !FileManager.default.fileExists(atPath: repoDestination.path) {
                throw EnvironmentError.offlineModeError(String(localized: "Repository not available locally"))
            }

            let fileUrls = try FileManager.default.getFileUrls(at: repoDestination)
            if fileUrls.isEmpty {
                throw EnvironmentError.offlineModeError(String(localized: "No files available locally for this repository"))
            }

            for fileUrl in fileUrls {
                let metadataPath = URL(
                    fileURLWithPath: fileUrl.path.replacingOccurrences(
                        of: repoDestination.path,
                        with: repoMetadataDestination.path
                    ) + ".metadata"
                )

                let localMetadata = try readDownloadMetadata(metadataPath: metadataPath)

                guard let localMetadata else {
                    throw EnvironmentError.offlineModeError(String(localized: "Metadata not available for \(fileUrl.lastPathComponent)"))
                }
                let localEtag = localMetadata.etag

                // LFS file so check file integrity
                if isValidHash(hash: localEtag, pattern: sha256Pattern) {
                    let fileHash = try computeFileHash(file: fileUrl)
                    if fileHash != localEtag {
                        throw EnvironmentError.fileIntegrityError(String(localized: "Hash mismatch for \(fileUrl.lastPathComponent)"))
                    }
                }
            }

            return repoDestination
        }

        let filenames = try await getFilenames(from: repo, revision: revision, matching: globs)
        let progress = Progress(totalUnitCount: Int64(filenames.count))
        for filename in filenames {
            let fileProgress = Progress(totalUnitCount: 100, parent: progress, pendingUnitCount: 1)
            let downloader = HubFileDownloader(
                hub: self,
                repo: repo,
                revision: revision,
                repoDestination: repoDestination,
                repoMetadataDestination: repoMetadataDestination,
                relativeFilename: filename,
                hfToken: hfToken,
                endpoint: endpoint,
                backgroundSession: useBackgroundSession
            )

            try await downloader.download { fractionDownloaded, speed in
                fileProgress.completedUnitCount = Int64(100 * fractionDownloaded)
                if let speed {
                    fileProgress.setUserInfoObject(speed, forKey: .throughputKey)
                    progress.setUserInfoObject(speed, forKey: .throughputKey)
                }
                progressHandler(progress)
            }
            if Task.isCancelled {
                return repoDestination
            }

            fileProgress.completedUnitCount = 100
        }

        progressHandler(progress)
        return repoDestination
    }

    /// New overloads exposing speed directly in the snapshot progress handler
    @discardableResult func snapshot(from repo: Repo, revision: String = "main", matching globs: [String] = [], progressHandler: @escaping (Progress, Double?) -> Void) async throws -> URL {
        try await snapshot(from: repo, revision: revision, matching: globs) { progress in
            let speed = progress.userInfo[.throughputKey] as? Double
            progressHandler(progress, speed)
        }
    }

    @discardableResult
    func snapshot(from repoId: String, revision: String = "main", matching globs: [String] = [], progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await snapshot(from: Repo(id: repoId), revision: revision, matching: globs, progressHandler: progressHandler)
    }

    @discardableResult
    func snapshot(from repo: Repo, revision: String = "main", matching glob: String, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await snapshot(from: repo, revision: revision, matching: [glob], progressHandler: progressHandler)
    }

    @discardableResult
    func snapshot(from repoId: String, revision: String = "main", matching glob: String, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await snapshot(from: Repo(id: repoId), revision: revision, matching: [glob], progressHandler: progressHandler)
    }

    /// Convenience overloads for other snapshot entry points with speed
    @discardableResult
    func snapshot(from repoId: String, revision: String = "main", matching globs: [String] = [], progressHandler: @escaping (Progress, Double?) -> Void) async throws -> URL {
        try await snapshot(from: Repo(id: repoId), revision: revision, matching: globs, progressHandler: progressHandler)
    }

    @discardableResult
    func snapshot(from repo: Repo, revision: String = "main", matching glob: String, progressHandler: @escaping (Progress, Double?) -> Void) async throws -> URL {
        try await snapshot(from: repo, revision: revision, matching: [glob], progressHandler: progressHandler)
    }

    @discardableResult
    func snapshot(from repoId: String, revision: String = "main", matching glob: String, progressHandler: @escaping (Progress, Double?) -> Void) async throws -> URL {
        try await snapshot(from: Repo(id: repoId), revision: revision, matching: [glob], progressHandler: progressHandler)
    }
}

/// Metadata
extension HubApi {
    /// Data structure containing information about a file versioned on the Hub
    struct FileMetadata {
        /// The commit hash related to the file
        let commitHash: String?

        /// Etag of the file on the server
        let etag: String?

        /// Location where to download the file. Can be a Hub url or not (CDN).
        let location: String

        /// Size of the file. In case of an LFS file, contains the size of the actual LFS file, not the pointer.
        let size: Int?
        // Argmax-modification: removed xetFileData field — Xet protocol unused in this repo
    }

    /// Metadata about a file in the local directory related to a download process
    struct LocalDownloadFileMetadata {
        /// Commit hash of the file in the repo
        let commitHash: String

        /// ETag of the file in the repo. Used to check if the file has changed.
        /// For LFS files, this is the sha256 of the file. For regular files, it corresponds to the git hash.
        let etag: String

        /// Path of the file in the repo
        let filename: String

        /// The timestamp of when the metadata was saved i.e. when the metadata was accurate
        let timestamp: Date
    }

    private func normalizeEtag(_ etag: String?) -> String? {
        guard let etag else { return nil }
        return etag.trimmingPrefix("W/").trimmingCharacters(in: CharacterSet(charactersIn: "\""))
    }

    func getFileMetadata(url: URL) async throws -> FileMetadata {
        let response = try await httpHead(for: url)
        let location = response.statusCode == 302 ? response.value(forHTTPHeaderField: "Location") : response.url?.absoluteString

        return FileMetadata(
            commitHash: response.value(forHTTPHeaderField: HFHttpHeaders.repoCommit),
            etag: normalizeEtag(
                (response.value(forHTTPHeaderField: HFHttpHeaders.linkedEtag)) ?? (response.value(forHTTPHeaderField: HFHttpHeaders.etag))
            ),
            location: location ?? url.absoluteString,
            size: Int(response.value(forHTTPHeaderField: HFHttpHeaders.linkedSize) ?? response.value(forHTTPHeaderField: HFHttpHeaders.contentLength) ?? "")
            // Argmax-modification: removed xetFileData: parseXetFileDataFromResponse(...) arg — Xet protocol unused in this repo
        )
    }

    // Argmax-modification: removed parseXetFileDataFromResponse method — Xet protocol unused in this repo

    func getFileMetadata(from repo: Repo, revision: String = "main", matching globs: [String] = []) async throws -> [FileMetadata] {
        let files = try await getFilenames(from: repo, matching: globs)
        let url = URL(string: endpoint)!
            .appending(path: repo.id)
            .appending(path: "resolve")
            .appending(component: revision) // Encode slashes (e.g., "pr/1" -> "pr%2F1")
        var selectedMetadata: [FileMetadata] = []
        for file in files {
            let fileURL = url.appending(path: file)
            try await selectedMetadata.append(getFileMetadata(url: fileURL))
        }
        return selectedMetadata
    }

    func getFileMetadata(from repoId: String, revision: String = "main", matching globs: [String] = []) async throws -> [FileMetadata] {
        try await getFileMetadata(from: Repo(id: repoId), revision: revision, matching: globs)
    }

    func getFileMetadata(from repo: Repo, revision: String = "main", matching glob: String) async throws -> [FileMetadata] {
        try await getFileMetadata(from: repo, revision: revision, matching: [glob])
    }

    func getFileMetadata(from repoId: String, revision: String = "main", matching glob: String) async throws -> [FileMetadata] {
        try await getFileMetadata(from: Repo(id: repoId), revision: revision, matching: [glob])
    }
}

/// Network monitor helper class to help decide whether to use offline mode
extension HubApi {
    private actor NetworkStateActor {
        /// Assume best case connection until updated by the monitor
        // Argmax-modification: removed public — HubApi is internal
        var isConnected: Bool = true
        var isExpensive: Bool = false
        var isConstrained: Bool = false

        func update(path: NWPath) {
            isConnected = path.status == .satisfied
            isExpensive = path.isExpensive
            isConstrained = path.isConstrained
        }

        func shouldUseOfflineMode() -> Bool {
            if ProcessInfo.processInfo.environment["CI_DISABLE_NETWORK_MONITOR"] == "1" {
                return false
            }
            return !isConnected
        }
    }

    private final class NetworkMonitor: Sendable {
        private let monitor: NWPathMonitor
        private let queue: DispatchQueue

        // Argmax-modification: removed public — HubApi is internal
        let state: NetworkStateActor = .init()

        static let shared = NetworkMonitor()

        init() {
            monitor = NWPathMonitor()
            queue = DispatchQueue(label: "HubApi.NetworkMonitor")
            startMonitoring()
        }

        func startMonitoring() {
            monitor.pathUpdateHandler = { [weak self] path in
                guard let self else { return }
                Task {
                    await self.state.update(path: path)
                }
            }

            monitor.start(queue: queue)
        }

        func stopMonitoring() {
            monitor.cancel()
        }

        deinit {
            stopMonitoring()
        }
    }
}

// Argmax-modification: removed `public extension Hub` — 17 static forwarders to HubApi.shared:
//   getFilenames(from:matching:) x4, snapshot(from:matching:progressHandler:) x8,
//   whoami(token:), getFileMetadata(fileURL:) + getFileMetadata(from:matching:) x4
//   All unused in this repo; callers hold a HubApi instance directly.

// Argmax-modification: matching(glob:) removed - already defined in FoundationExtensions.swift

private extension FileManager {
    func getFileUrls(at directoryUrl: URL) throws -> [URL] {
        var fileUrls = [URL]()

        // Get all contents including subdirectories
        guard
            let enumerator = FileManager.default.enumerator(
                at: directoryUrl,
                includingPropertiesForKeys: [.isRegularFileKey, .isHiddenKey],
                options: [.skipsHiddenFiles]
            )
        else {
            return fileUrls
        }

        for case let fileURL as URL in enumerator {
            do {
                let resourceValues = try fileURL.resourceValues(forKeys: [.isRegularFileKey, .isHiddenKey])
                if resourceValues.isRegularFile == true, resourceValues.isHidden != true {
                    fileUrls.append(fileURL)
                }
            } catch {
                throw error
            }
        }

        return fileUrls
    }
}

/// Only allow relative redirects and reject others
/// Reference: https://github.com/huggingface/huggingface_hub/blob/b2c9a148d465b43ab90fab6e4ebcbbf5a9df27d4/src/huggingface_hub/file_download.py#L258
private final class RedirectDelegate: NSObject, URLSessionTaskDelegate, Sendable {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        // Check if it's a redirect status code (300-399)
        if (300...399).contains(response.statusCode) {
            // Get the Location header
            if let locationString = response.value(forHTTPHeaderField: "Location"),
                let locationUrl = URL(string: locationString)
            {
                // Check if it's a relative redirect (no host component)
                if locationUrl.host == nil {
                    // For relative redirects, construct the new URL using the original request's base
                    if let originalUrl = task.originalRequest?.url,
                        var components = URLComponents(url: originalUrl, resolvingAgainstBaseURL: true)
                    {
                        // Update the path component with the relative path
                        components.path = locationUrl.path
                        components.query = locationUrl.query

                        // Create new request with the resolved URL
                        if let resolvedUrl = components.url {
                            var newRequest = URLRequest(url: resolvedUrl)
                            // Copy headers from original request
                            if let headers = task.originalRequest?.allHTTPHeaderFields {
                                for (key, value) in headers {
                                    newRequest.setValue(value, forHTTPHeaderField: key)
                                }
                            }
                            newRequest.setValue(resolvedUrl.absoluteString, forHTTPHeaderField: "Location")
                            completionHandler(newRequest)
                            return
                        }
                    }
                }
            }
        }

        // For all other cases (non-redirects or absolute redirects), prevent redirect
        completionHandler(nil)
    }
}

/// Actor to manage shared URLSession for redirect handling.
///
/// Lazily initializes and reuses a single URLSession across all HubApi instances
/// to avoid resource exhaustion when running multiple tests or creating many instances.
private actor RedirectSessionActor {
    private var urlSession: URLSession?

    func get() -> URLSession {
        if let urlSession = urlSession {
            return urlSession
        }

        // Create session once and reuse
        let redirectDelegate = RedirectDelegate()
        let session = URLSession(configuration: .default, delegate: redirectDelegate, delegateQueue: nil)
        self.urlSession = session
        return session
    }
}
