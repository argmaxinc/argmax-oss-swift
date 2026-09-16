//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import Network
import XCTest

@testable import ArgmaxCore

final class HubDownloadProgressTests: XCTestCase {
    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempDir)
    }

    func testSnapshotProgressCountsBytesAcrossUnequalFiles() async throws {
        let (hub, repo, server) = try await makeHub(files: [
            .init(name: "small.bin", data: Data([1])),
            .init(name: "large.bin", data: Data(repeating: 2, count: 99)),
        ])
        var samples: [ProgressSample] = []

        let result = try await hub.snapshot(from: repo) { samples.append(ProgressSample($0)) }

        XCTAssertTrue(samples.contains { $0.completed == 1 && $0.total == 100 && $0.fraction == 0.01 },
                      "One byte out of 100 should report 1%, not one file out of two (50%): \(samples)")
        XCTAssertEqual(samples.last?.completed, 100)
        XCTAssertEqual(samples.last?.total, 100)
        XCTAssertEqual(samples.last?.fraction, 1)
        XCTAssertEqual(samples.last?.kind, .file)
        XCTAssertTrue(zip(samples, samples.dropFirst()).allSatisfy { $0.fraction <= $1.fraction })
        XCTAssertLessThanOrEqual(server.metadataRequestCount, 2, "Progress must not add duplicate metadata requests")
        XCTAssertEqual(try Data(contentsOf: result.appendingPathComponent("small.bin")), Data([1]))
        XCTAssertEqual(try Data(contentsOf: result.appendingPathComponent("large.bin")), Data(repeating: 2, count: 99))
    }

    func testCachedFilesContributeTheirByteSizes() async throws {
        let files: [ProgressHTTPServer.File] = [
            .init(name: "large.bin", data: Data(repeating: 2, count: 99)),
            .init(name: "small.bin", data: Data([1])),
        ]
        let (hub, repo, _) = try await makeHub(files: files)
        for file in files { try cache(file, hub: hub, repo: repo) }
        var samples: [ProgressSample] = []

        try await hub.snapshot(from: repo) { samples.append(ProgressSample($0)) }

        XCTAssertTrue(samples.contains { $0.completed == 99 && $0.total == 100 && $0.fraction == 0.99 })
        XCTAssertEqual(samples.last?.completed, 100)
        XCTAssertEqual(samples.last?.kind, .file)
    }

    func testResumedBytesAreIncludedInSnapshotProgress() async throws {
        let small = ProgressHTTPServer.File(name: "small.bin", data: Data([1]))
        let large = ProgressHTTPServer.File(name: "large.bin", data: Data(repeating: 2, count: 99), holdResponse: true)
        let (hub, repo, server) = try await makeHub(files: [small, large])
        try cache(small, hub: hub, repo: repo)
        let incomplete = hub.localRepoLocation(repo)
            .appendingPathComponent(".cache/huggingface/download/large.bin.large.bin-etag.incomplete")
        try large.data.prefix(49).write(to: incomplete)
        let resumed = XCTestExpectation(description: "Resumed progress reported before new bytes arrive")
        let task = Task {
            var samples: [ProgressSample] = []
            var observedResume = false
            let result = try await hub.snapshot(from: repo) { progress in
                samples.append(ProgressSample(progress))
                if progress.fractionCompleted > 0.01, progress.fractionCompleted < 1, !observedResume {
                    observedResume = true
                    resumed.fulfill()
                }
            }
            return (result, samples)
        }
        await fulfillment(of: [resumed], timeout: 5)
        server.releaseDownloads()
        let (result, samples) = try await task.value

        XCTAssertTrue(samples.contains { $0.completed == 50 && $0.total == 100 && $0.fraction == 0.5 },
                      "The cached byte plus 49 resumed bytes should report 50/100: \(samples)")
        XCTAssertEqual(samples.last?.completed, 100)
        XCTAssertEqual(try Data(contentsOf: result.appendingPathComponent("large.bin")), large.data)
    }

    func testMissingCachedFileSizePreservesFileWeightedProgress() async throws {
        let cached = ProgressHTTPServer.File(name: "cached.bin", data: Data([1]), reportsSize: false)
        let (hub, repo, _) = try await makeHub(files: [
            cached,
            .init(name: "large.bin", data: Data(repeating: 2, count: 99)),
        ])
        try cache(cached, hub: hub, repo: repo)
        var samples: [ProgressSample] = []

        try await hub.snapshot(from: repo) { samples.append(ProgressSample($0)) }

        XCTAssertTrue(samples.contains { $0.completed == 1 && $0.total == 2 && $0.fraction == 0.5 })
        XCTAssertTrue(samples.allSatisfy { $0.kind == nil }, "Unknown sizes must not be advertised as byte counts")
        XCTAssertEqual(samples.last?.completed, 2)
        XCTAssertEqual(samples.last?.fraction, 1)
    }

    func testZeroByteFileDoesNotAdvanceByteProgress() async throws {
        let (hub, repo, _) = try await makeHub(files: [
            .init(name: "empty.bin", data: Data()),
            .init(name: "large.bin", data: Data(repeating: 2, count: 100)),
        ])
        var samples: [ProgressSample] = []

        try await hub.snapshot(from: repo) { samples.append(ProgressSample($0)) }

        XCTAssertTrue(samples.contains { $0.kind == .file && $0.completed == 0 && $0.total == 100 })
        XCTAssertFalse(samples.contains { $0.fraction > 0 && $0.fraction < 1 })
        XCTAssertEqual(samples.last?.completed, 100)
        XCTAssertEqual(samples.last?.fraction, 1)
    }

    func testAllEmptyFilesStillReportCompletion() async throws {
        let (hub, repo, _) = try await makeHub(files: [
            .init(name: "first.bin", data: Data()),
            .init(name: "second.bin", data: Data()),
        ])
        var samples: [ProgressSample] = []

        try await hub.snapshot(from: repo) { samples.append(ProgressSample($0)) }

        XCTAssertEqual(samples.last?.fraction, 1)
        XCTAssertEqual(samples.last?.completed, 2)
        XCTAssertNil(samples.last?.kind)
    }

    private func cache(_ file: ProgressHTTPServer.File, hub: HubApi, repo: HubApi.Repo) throws {
        let root = hub.localRepoLocation(repo)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try file.data.write(to: root.appendingPathComponent(file.name))
        try hub.writeDownloadMetadata(
            commitHash: ProgressHTTPServer.commitHash,
            etag: "\(file.name)-etag",
            metadataPath: root.appendingPathComponent(".cache/huggingface/download/\(file.name).metadata")
        )
    }

    private func makeHub(files: [ProgressHTTPServer.File]) async throws -> (HubApi, HubApi.Repo, ProgressHTTPServer) {
        let server = try ProgressHTTPServer(files: files)
        addTeardownBlock { server.stop() }
        server.start()
        await fulfillment(of: [server.ready], timeout: 5)
        let port = try XCTUnwrap(server.port)
        return (
            HubApi(downloadBase: tempDir, hfToken: "", endpoint: "http://127.0.0.1:\(port)", useOfflineMode: false),
            HubApi.Repo(id: "test/progress"),
            server
        )
    }
}

private struct ProgressSample: CustomStringConvertible {
    let completed: Int64
    let total: Int64
    let fraction: Double
    let kind: ProgressKind?

    init(_ progress: Progress) {
        completed = progress.completedUnitCount
        total = progress.totalUnitCount
        fraction = progress.fractionCompleted
        kind = progress.kind
    }

    var description: String { "\(completed)/\(total) (\(fraction))" }
}

/// Uses the real download path with tiny files and no external network service.
/// Mutable connection state is confined to `queue`.
private final class ProgressHTTPServer: @unchecked Sendable {
    struct File: Sendable {
        let name: String
        let data: Data
        var reportsSize = true
        var holdResponse = false
    }

    static let commitHash = String(repeating: "1", count: 40)
    let ready = XCTestExpectation(description: "Local progress server is ready")
    private let files: [File]
    private let listener: NWListener
    private let queue = DispatchQueue(label: "HubDownloadProgressTests.HTTPServer")
    private var connections: [NWConnection] = []
    private var pending: [(NWConnection, Data)] = []
    private var downloadsReleased = false
    private var metadataRequests = 0

    var metadataRequestCount: Int { queue.sync { metadataRequests } }

    var port: UInt16? { listener.port?.rawValue }

    init(files: [File]) throws {
        self.files = files
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        listener = try NWListener(using: parameters)
    }

    func start() {
        listener.stateUpdateHandler = { [weak self] state in
            if case .ready = state { self?.ready.fulfill() }
        }
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            self.connections.append(connection)
            connection.start(queue: self.queue)
            self.receiveRequest(on: connection)
        }
        listener.start(queue: queue)
    }

    func releaseDownloads() {
        queue.async {
            self.downloadsReleased = true
            for (connection, response) in self.pending {
                self.send(response, on: connection)
            }
            self.pending.removeAll()
        }
    }

    func stop() {
        queue.sync {
            listener.cancel()
            connections.forEach { $0.cancel() }
            connections.removeAll()
            pending.removeAll()
        }
    }

    private func receiveRequest(on connection: NWConnection, buffer: Data = Data()) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, complete, error in
            guard let self, error == nil, let data else {
                connection.cancel()
                return
            }
            let received = buffer + data
            guard let request = String(data: received, encoding: .utf8), request.contains("\r\n\r\n") else {
                if complete { connection.cancel() }
                else { self.receiveRequest(on: connection, buffer: received) }
                return
            }
            self.respond(to: request, on: connection)
        }
    }

    private func respond(to request: String, on connection: NWConnection) {
        let parts = request.prefix(while: { $0 != "\r" }).split(separator: " ")
        guard parts.count >= 2 else { connection.cancel(); return }
        let method = parts[0]
        let path = String(parts[1])
        if method == "HEAD" { metadataRequests += 1 }
        let body: Data
        let file = files.first { path == "/test/progress/resolve/main/\($0.name)" }
        if path.hasPrefix("/api/models/") {
            body = try! JSONSerialization.data(withJSONObject: ["siblings": files.map { ["rfilename": $0.name] }])
        } else if let file {
            body = file.data
        } else {
            connection.cancel()
            return
        }
        let range = request.components(separatedBy: "\r\n")
            .first { $0.lowercased().hasPrefix("range: bytes=") }
        let offset = range.flatMap { Int($0.dropFirst("range: bytes=".count).prefix(while: { $0 != "-" })) } ?? 0
        guard offset >= 0, offset <= body.count else { connection.cancel(); return }
        let payload = body.dropFirst(offset)
        let status = offset > 0 ? "206 Partial Content" : "200 OK"
        let contentRange = offset > 0 ? "Content-Range: bytes \(offset)-\(body.count - 1)/\(body.count)\r\n" : ""
        let name = path.split(separator: "/").last ?? ""
        let contentLength = method == "HEAD" && file?.reportsSize == false ? "" : "Content-Length: \(payload.count)\r\n"
        let headers = "HTTP/1.1 \(status)\r\n\(contentLength)\(contentRange)X-Repo-Commit: \(Self.commitHash)\r\nETag: \(name)-etag\r\nConnection: close\r\n\r\n"
        var response = Data(headers.utf8)
        if method != "HEAD" { response.append(payload) }
        if method == "GET", file?.holdResponse == true, !downloadsReleased {
            pending.append((connection, response))
        } else {
            send(response, on: connection)
        }
    }

    private func send(_ response: Data, on connection: NWConnection) {
        connection.send(content: response, completion: .contentProcessed { _ in connection.cancel() })
    }
}
