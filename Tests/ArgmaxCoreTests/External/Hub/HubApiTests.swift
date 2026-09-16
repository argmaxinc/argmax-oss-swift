//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import Network
import XCTest

@testable import ArgmaxCore

final class HubApiTests: XCTestCase {
    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempDir)
    }

    func testSnapshotCancellationDoesNotReturnPartialRepository() async throws {
        let (hub, repo, repoRoot, pendingDownload) = try await makePartialSnapshot(name: "cancel")
        let task = Task {
            try await hub.snapshot(from: repo) { progress in
                XCTAssertEqual(progress.fractionCompleted, 0.5)
            }
        }

        // The first file is cached. Wait until the second request is in flight before cancelling.
        await fulfillment(of: [pendingDownload], timeout: 5)
        task.cancel()

        do {
            let result = try await task.value
            XCTFail("Cancelled snapshot returned a partial repository as success: \(result.path)")
        } catch is CancellationError {
            // Expected: cancellation must not be reported as a completed snapshot.
        }

        XCTAssertTrue(FileManager.default.fileExists(atPath: repoRoot.appendingPathComponent("config.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: repoRoot.appendingPathComponent("weights.bin").path))
    }

    func testSnapshotCompletesUncancelledDownload() async throws {
        let (hub, repo, repoRoot, _) = try await makePartialSnapshot(name: "complete")
        var finalProgress = 0.0

        let result = try await hub.snapshot(from: repo) { progress in
            finalProgress = progress.fractionCompleted
        }

        XCTAssertEqual(result, repoRoot)
        XCTAssertEqual(finalProgress, 1)
        XCTAssertEqual(try Data(contentsOf: repoRoot.appendingPathComponent("config.json")), SnapshotHTTPServer.fileData)
        XCTAssertEqual(try Data(contentsOf: repoRoot.appendingPathComponent("weights.bin")), SnapshotHTTPServer.fileData)
    }

    private func makePartialSnapshot(name: String) async throws -> (HubApi, HubApi.Repo, URL, XCTestExpectation) {
        let server = try SnapshotHTTPServer()
        addTeardownBlock { server.stop() }
        server.start()
        await fulfillment(of: [server.ready], timeout: 5)
        let port = try XCTUnwrap(server.port)
        let hub = HubApi(
            downloadBase: tempDir,
            hfToken: "",
            endpoint: "http://127.0.0.1:\(port)",
            useOfflineMode: false
        )
        let repo = HubApi.Repo(id: "test/\(name)")
        let repoRoot = hub.localRepoLocation(repo)
        try FileManager.default.createDirectory(at: repoRoot, withIntermediateDirectories: true)
        try SnapshotHTTPServer.fileData.write(to: repoRoot.appendingPathComponent("config.json"))
        try hub.writeDownloadMetadata(
            commitHash: SnapshotHTTPServer.commitHash,
            etag: "config-etag",
            metadataPath: repoRoot.appendingPathComponent(".cache/huggingface/download/config.json.metadata")
        )
        return (hub, repo, repoRoot, server.pendingDownload)
    }
}

/// Serves two tiny files over loopback, exercising the real URLSession download path.
/// Mutable connection state is confined to `queue`.
private final class SnapshotHTTPServer: @unchecked Sendable {
    static let fileData = Data("model-bytes".utf8)
    static let commitHash = String(repeating: "1", count: 40)

    let ready = XCTestExpectation(description: "Local snapshot server is ready")
    let pendingDownload = XCTestExpectation(description: "Missing file request is in flight")
    private let listener: NWListener
    private let queue = DispatchQueue(label: "HubApiTests.HTTPServer")
    private var connections: [NWConnection] = []

    var port: UInt16? { listener.port?.rawValue }

    init() throws {
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        listener = try NWListener(using: parameters)
    }

    func start() {
        listener.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            if case .ready = state {
                self.ready.fulfill()
            }
        }
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            self.connections.append(connection)
            connection.start(queue: self.queue)
            self.receiveRequest(on: connection)
        }
        listener.start(queue: queue)
    }

    func stop() {
        queue.sync {
            listener.cancel()
            connections.forEach { $0.cancel() }
            connections.removeAll()
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
                if complete {
                    connection.cancel()
                } else {
                    self.receiveRequest(on: connection, buffer: received)
                }
                return
            }
            self.respond(to: request, on: connection)
        }
    }

    private func respond(to request: String, on connection: NWConnection) {
        let parts = request.prefix(while: { $0 != "\r" }).split(separator: " ")
        guard parts.count >= 2 else {
            connection.cancel()
            return
        }
        let method = parts[0]
        let path = parts[1]
        // Hold the missing file until the test cancels; no sleeps or timing races.
        if method == "GET", path == "/test/cancel/resolve/main/weights.bin" {
            pendingDownload.fulfill()
            return
        }

        let body = path.hasPrefix("/api/models/")
            ? Data(#"{"siblings":[{"rfilename":"config.json"},{"rfilename":"weights.bin"}]}"#.utf8)
            : Self.fileData
        let etag = path.hasSuffix("config.json") ? "config-etag" : "weights-etag"
        let headers = "HTTP/1.1 200 OK\r\nContent-Length: \(body.count)\r\nX-Repo-Commit: \(Self.commitHash)\r\nETag: \(etag)\r\nConnection: close\r\n\r\n"
        var response = Data(headers.utf8)
        if method != "HEAD" {
            response.append(body)
        }
        connection.send(content: response, completion: .contentProcessed { _ in connection.cancel() })
    }
}
