// Originally from: https://github.com/huggingface/swift-transformers
// Version: 1.1.6 (commit: 573e5c9036c2f136b3a8a071da8e8907322403d0)
// License: Apache 2.0 (https://github.com/huggingface/swift-transformers/blob/main/LICENSE)
// Copyright 2022 Hugging Face SAS
//

//
//  Downloader.swift
//
//  Adapted from https://github.com/huggingface/swift-coreml-diffusers/blob/d041577b9f5e201baa3465bc60bc5d0a1cf7ed7f/Diffusion/Common/Downloader.swift
//  Created by Pedro Cuenca on December 2022.
//  See LICENSE at https://github.com/huggingface/swift-coreml-diffusers/LICENSE
//

import Foundation

/// A robust file downloader with support for resumable downloads and progress reporting.
///
/// The Downloader class handles file downloads from remote URLs with features including
/// automatic resume capability, progress tracking, speed monitoring, and retry mechanisms.
/// It supports both foreground and background download sessions for different use cases.
final class Downloader: NSObject, Sendable {
    private let destination: URL
    private let incompleteDestination: URL
    private let downloadResumeState: DownloadResumeState = .init()
    private let chunkSize: Int

    /// Represents the current state of a download operation.
    enum DownloadState {
        /// Download has not yet started.
        case notStarted
        /// Download is in progress with completion percentage and optional speed in bytes/sec.
        case downloading(Double, Double?)
        /// Download completed successfully with the final file URL.
        case completed(URL)
        /// Download failed with an error.
        case failed(Error)
    }

    /// Errors specific to download operations.
    enum DownloadError: Error {
        /// The specified download location is invalid.
        case invalidDownloadLocation
        /// An unexpected error occurred during download.
        case unexpectedError
        /// The temporary file could not be found during resume.
        case tempFileNotFound
    }

    private let broadcaster: Broadcaster<DownloadState> = Broadcaster<DownloadState> {
        DownloadState.notStarted
    }

    private let sessionConfig: URLSessionConfiguration
    let session: SessionActor = .init()
    private let task: TaskActor = .init()

    /// Initializes a new downloader instance.
    ///
    /// - Parameters:
    ///   - destination: The final destination URL for the downloaded file
    ///   - incompleteDestination: The temporary location for incomplete downloads
    ///   - inBackground: Whether to use background URL session (defaults to false)
    ///   - chunkSize: Size of download chunks in bytes (defaults to 10MB)
    init(
        to destination: URL,
        incompleteDestination: URL,
        inBackground: Bool = false,
        chunkSize: Int = 10 * 1024 * 1024 // 10MB
    ) {
        self.destination = destination
        // Create incomplete file path based on destination
        self.incompleteDestination = incompleteDestination
        self.chunkSize = chunkSize

        let sessionIdentifier = "swift-transformers.hub.downloader"

        var config = URLSessionConfiguration.default
        if inBackground {
            config = URLSessionConfiguration.background(withIdentifier: sessionIdentifier)
            config.isDiscretionary = false
            config.sessionSendsLaunchEvents = true
        }
        sessionConfig = config
    }

    /// Starts a download operation and returns a stream of download states.
    ///
    /// - Parameters:
    ///   - url: The URL to download from
    ///   - authToken: Optional authentication token for the request
    ///   - headers: Additional HTTP headers to include
    ///   - expectedSize: The expected file size in bytes for progress calculation
    ///   - timeout: Request timeout interval in seconds (defaults to 10)
    ///   - numRetries: Maximum number of retry attempts (defaults to 5)
    /// - Returns: An async stream of download state updates
    func download(
        from url: URL,
        using authToken: String? = nil,
        headers: [String: String]? = nil,
        expectedSize: Int? = nil,
        timeout: TimeInterval = 10,
        numRetries: Int = 5
    ) async -> AsyncStream<DownloadState> {
        if let task = await task.get() {
            task.cancel()
        }
        await downloadResumeState.setExpectedSize(expectedSize)
        let resumeSize = Self.incompleteFileSize(at: incompleteDestination)
        await session.set(URLSession(configuration: sessionConfig, delegate: self, delegateQueue: nil))
        await setUpDownload(
            from: url,
            with: authToken,
            resumeSize: resumeSize,
            headers: headers,
            timeout: timeout,
            numRetries: numRetries
        )

        return await broadcaster.subscribe()
    }

    /// Sets up and initiates a file download operation
    ///
    /// - Parameters:
    ///   - url: Source URL to download from
    ///   - authToken: Bearer token for authentication with Hugging Face
    ///   - resumeSize: Number of bytes already downloaded for resuming interrupted downloads
    ///   - headers: Additional HTTP headers to include in the request
    ///   - timeout: Time interval before the request times out
    ///   - numRetries: Number of retry attempts for failed downloads
    private func setUpDownload(
        from url: URL,
        with authToken: String?,
        resumeSize: Int,
        headers: [String: String]?,
        timeout: TimeInterval,
        numRetries: Int
    ) async {
        let resumeSize = Self.incompleteFileSize(at: incompleteDestination)
        guard let tasks = await session.get()?.allTasks else {
            return
        }

        // If there's an existing pending background task with the same URL, let it proceed.
        if let existing = tasks.filter({ $0.originalRequest?.url == url }).first {
            switch existing.state {
            case .running:
                return
            case .suspended:
                existing.resume()
                return
            case .canceling, .completed:
                existing.cancel()
                break
            @unknown default:
                existing.cancel()
            }
        }

        await task.set(
            Task {
                do {
                    var request = URLRequest(url: url)

                    // Use headers from argument else create an empty header dictionary
                    var requestHeaders = headers ?? [:]

                    // Populate header auth and range fields
                    if let authToken {
                        requestHeaders["Authorization"] = "Bearer \(authToken)"
                    }

                    await self.downloadResumeState.setDownloadedSize(resumeSize)

                    if resumeSize > 0 {
                        requestHeaders["Range"] = "bytes=\(resumeSize)-"
                    }

                    // Set Range header if we're resuming
                    if resumeSize > 0 {
                        requestHeaders["Range"] = "bytes=\(resumeSize)-"

                        // Calculate and show initial progress
                        if let expectedSize = await self.downloadResumeState.expectedSize, expectedSize > 0 {
                            let initialProgress = Double(resumeSize) / Double(expectedSize)
                            await self.broadcaster.broadcast(state: .downloading(initialProgress, nil))
                        } else {
                            await self.broadcaster.broadcast(state: .downloading(0, nil))
                        }
                    } else {
                        await self.broadcaster.broadcast(state: .downloading(0, nil))
                    }

                    request.timeoutInterval = timeout
                    request.allHTTPHeaderFields = requestHeaders

                    // Open the incomplete file for writing
                    let tempFile = try FileHandle(forWritingTo: self.incompleteDestination)

                    // If resuming, seek to end of file
                    if resumeSize > 0 {
                        try tempFile.seekToEnd()
                    }

                    defer { tempFile.closeFile() }

                    try await self.httpGet(request: request, tempFile: tempFile, numRetries: numRetries)

                    try Task.checkCancellation()
                    try FileManager.default.moveDownloadedFile(from: self.incompleteDestination, to: self.destination)

                    //                    // Clean up and move the completed download to its final destination
                    //                    tempFile.closeFile()
                    //                    try FileManager.default.moveDownloadedFile(from: tempURL, to: self.destination)

                    await self.broadcaster.broadcast(state: .completed(self.destination))
                } catch {
                    await self.broadcaster.broadcast(state: .failed(error))
                }
            }
        )
    }

    /// Downloads a file from given URL using chunked transfer and handles retries.
    ///
    /// Reference: https://github.com/huggingface/huggingface_hub/blob/418a6ffce7881f5c571b2362ed1c23ef8e4d7d20/src/huggingface_hub/file_download.py#L306
    ///
    /// - Parameters:
    ///   - request: The URLRequest for the file to download
    ///   - tempFile: The file handle for writing downloaded data
    ///   - numRetries: The number of retry attempts remaining for failed downloads
    /// - Throws: `DownloadError.unexpectedError` if the response is invalid or file size mismatch occurs
    ///           `URLError` if the download fails after all retries are exhausted
    private func httpGet(
        request: URLRequest,
        tempFile: FileHandle,
        numRetries: Int
    ) async throws {
        guard let session = await session.get() else {
            throw DownloadError.unexpectedError
        }

        // Create a new request with Range header for resuming
        var newRequest = request
        if await downloadResumeState.downloadedSize > 0 {
            await newRequest.setValue("bytes=\(downloadResumeState.downloadedSize)-", forHTTPHeaderField: "Range")
        }

        // Start the download and get the byte stream
        let (asyncBytes, response) = try await session.bytes(for: newRequest)

        guard let response = response as? HTTPURLResponse else {
            throw DownloadError.unexpectedError
        }

        guard (200..<300).contains(response.statusCode) else {
            throw DownloadError.unexpectedError
        }

        // Create a buffer to collect bytes before writing to disk
        var buffer = Data(capacity: chunkSize)

        // Track speed (bytes per second) using sampling between broadcasts
        var lastSampleTime = Date()
        var totalDownloadedLocal = await downloadResumeState.downloadedSize
        var lastSampleBytes = totalDownloadedLocal

        var newNumRetries = numRetries
        do {
            // Batch collect bytes to reduce Data.append() overhead
            // Use ContiguousArray for better performance (no NSArray bridging overhead)
            let batchSize = 16384 // 16 kB
            var byteBatch = ContiguousArray<UInt8>()
            byteBatch.reserveCapacity(batchSize)

            for try await byte in asyncBytes {
                byteBatch.append(byte)

                // Append batch to main buffer
                if byteBatch.count >= batchSize {
                    buffer.append(contentsOf: byteBatch)
                    byteBatch.removeAll(keepingCapacity: true)
                }

                // When buffer is full, write to disk
                if buffer.count >= chunkSize {
                    if !buffer.isEmpty { // Filter out keep-alive chunks
                        try tempFile.write(contentsOf: buffer)
                        let bytesWritten = buffer.count
                        buffer.removeAll(keepingCapacity: true)

                        totalDownloadedLocal += bytesWritten
                        await downloadResumeState.incDownloadedSize(bytesWritten)
                        newNumRetries = 5
                        guard let expectedSize = await downloadResumeState.expectedSize else { continue }
                        let progress = expectedSize != 0 ? Double(totalDownloadedLocal) / Double(expectedSize) : 0

                        // Compute instantaneous speed based on bytes since last broadcast
                        let now = Date()
                        let elapsed = now.timeIntervalSince(lastSampleTime)
                        let deltaBytes = totalDownloadedLocal - lastSampleBytes
                        let speed = elapsed > 0 ? Double(deltaBytes) / elapsed : nil
                        lastSampleTime = now
                        lastSampleBytes = totalDownloadedLocal

                        await broadcaster.broadcast(state: .downloading(progress, speed))
                    }
                }
            }

            // Flush remaining bytes from batch
            if !byteBatch.isEmpty {
                buffer.append(contentsOf: byteBatch)
            }

            if !buffer.isEmpty {
                try tempFile.write(contentsOf: buffer)
                totalDownloadedLocal += buffer.count
                await downloadResumeState.incDownloadedSize(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                newNumRetries = 5
            }
        } catch let error as URLError {
            if newNumRetries <= 0 {
                throw error
            }
            try await Task.sleep(nanoseconds: 1_000_000_000)

            await self.session.set(URLSession(configuration: self.sessionConfig, delegate: self, delegateQueue: nil))

            try await httpGet(
                request: request,
                tempFile: tempFile,
                numRetries: newNumRetries - 1
            )
            return
        }

        // Verify the downloaded file size matches the expected size
        let actualSize = try tempFile.seekToEnd()
        if let expectedSize = await downloadResumeState.expectedSize, expectedSize != actualSize {
            throw DownloadError.unexpectedError
        }
    }

    func cancel() async {
        await session.get()?.invalidateAndCancel()
        await task.get()?.cancel()
        await broadcaster.broadcast(state: .failed(URLError(.cancelled)))
    }

    /// Check if an incomplete file exists for the destination and returns its size
    /// - Parameter incompletePath: The URL path for the incomplete file
    /// - Returns: Size of the incomplete file if it exists, otherwise 0
    static func incompleteFileSize(at incompletePath: URL) -> Int {
        if FileManager.default.fileExists(atPath: incompletePath.path) {
            if let attributes = try? FileManager.default.attributesOfItem(atPath: incompletePath.path), let fileSize = attributes[.size] as? Int {
                return fileSize
            }
        }

        return 0
    }
}

extension Downloader: URLSessionDownloadDelegate {
    func urlSession(_: URLSession, downloadTask: URLSessionDownloadTask, didWriteData _: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        Task {
            await self.broadcaster.broadcast(state: .downloading(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite), nil))
        }
    }

    func urlSession(_: URLSession, downloadTask _: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        do {
            // If the downloaded file already exists on the filesystem, overwrite it
            try FileManager.default.moveDownloadedFile(from: location, to: destination)
            Task {
                await self.broadcaster.broadcast(state: .completed(destination))
            }
        } catch {
            Task {
                await self.broadcaster.broadcast(state: .failed(error))
            }
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error {
            Task {
                await self.broadcaster.broadcast(state: .failed(error))
            }
            //        } else if let response = task.response as? HTTPURLResponse {
            //            print("HTTP response status code: \(response.statusCode)")
            //            let headers = response.allHeaderFields
            //            print("HTTP response headers: \(headers)")
        }
    }
}

extension FileManager {
    func moveDownloadedFile(from srcURL: URL, to dstURL: URL, percentEncoded: Bool = false) throws {
        if fileExists(atPath: dstURL.path(percentEncoded: percentEncoded)) {
            try removeItem(at: dstURL)
        }

        let directoryURL = dstURL.deletingLastPathComponent()
        try createDirectory(at: directoryURL, withIntermediateDirectories: true, attributes: nil)

        try moveItem(at: srcURL, to: dstURL)
    }
}

private actor DownloadResumeState {
    var expectedSize: Int?
    var downloadedSize: Int = 0

    func setExpectedSize(_ size: Int?) {
        expectedSize = size
    }

    func setDownloadedSize(_ size: Int) {
        downloadedSize = size
    }

    func incDownloadedSize(_ size: Int) {
        downloadedSize += size
    }
}

actor Broadcaster<E: Sendable> {
    private let initialState: @Sendable () async -> E?
    private var latestState: E?
    private var continuations: [UUID: AsyncStream<E>.Continuation] = [:]

    init(initialState: @Sendable @escaping () async -> E?) {
        self.initialState = initialState
    }

    deinit {
        self.continuations.removeAll()
    }

    func subscribe() -> AsyncStream<E> {
        AsyncStream { continuation in
            let id = UUID()
            self.continuations[id] = continuation

            continuation.onTermination = { @Sendable status in
                Task {
                    await self.unsubscribe(id)
                }
            }

            Task {
                if let state = self.latestState {
                    continuation.yield(state)
                    return
                }
                if let state = await self.initialState() {
                    continuation.yield(state)
                }
            }
        }
    }

    private func unsubscribe(_ id: UUID) {
        continuations.removeValue(forKey: id)
    }

    func broadcast(state: E) async {
        latestState = state
        for continuation in continuations.values {
            continuation.yield(state)
        }
    }
}

actor SessionActor {
    private var urlSession: URLSession?

    func set(_ urlSession: URLSession?) {
        self.urlSession = urlSession
    }

    func get() -> URLSession? {
        urlSession
    }
}

actor TaskActor {
    private var task: Task<Void, Error>?

    func set(_ task: Task<Void, Error>?) {
        self.task = task
    }

    func get() -> Task<Void, Error>? {
        task
    }
}
