//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import XCTest

@testable import ArgmaxCore

/// Tests for `ModelDownloader` and `ModelInfo`. The resolution tests run with `download: false` (or hit
/// only an explicit local folder), so they never touch the network.
final class ModelDownloaderTests: XCTestCase {
    private var tempDir: URL!

    /// A made-up repo id. These tests never go online, so this is only a local cache path component
    /// (`downloadBase/models/<id>`) and intentionally does not exist on HuggingFace.
    private let testRepo = "test-org/speakerkit-coreml"

    /// Three SpeakerKit-style patterns, each under its own prefix and ending in a wildcard.
    private let patterns = [
        "speaker_segmenter/pyannote-v3/W8A16/*",
        "speaker_embedder/pyannote-v3/W8A16/*",
        "speaker_clusterer/pyannote-v4/W32A32/*",
    ]

    /// One file per pattern, laid out the way a real `.mlmodelc` is on disk.
    private let componentFiles = [
        "speaker_segmenter/pyannote-v3/W8A16/SpeakerSegmenter.mlmodelc/coremldata.bin",
        "speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedder.mlmodelc/coremldata.bin",
        "speaker_clusterer/pyannote-v4/W32A32/PldaProjector.mlmodelc/coremldata.bin",
    ]

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDown() {
        if let tempDir, FileManager.default.fileExists(atPath: tempDir.path) {
            try? FileManager.default.removeItem(at: tempDir)
        }
        super.tearDown()
    }

    // MARK: - Helpers

    private func makeDownloader() -> (ModelDownloader, URL) {
        let base = tempDir.appendingPathComponent("hub")
        let config = ModelDownloadConfig(downloadBase: base.path, modelRepo: testRepo)
        let downloader = ModelDownloader(config: config)
        let repoRoot = downloader.localRepoLocation()
        return (downloader, repoRoot)
    }

    /// Writes a model file into the cache, optionally with the `.metadata` sidecar the Hub library
    /// writes after a real download. Offline validation needs that sidecar.
    ///
    /// - Parameter etag: Goes on the sidecar's etag line. The default isn't a SHA256, so the hash check
    ///   is skipped; pass a 64-hex-char value to force it (it's then compared against the file's bytes).
    private func writeCachedFile(_ relPath: String, in repoRoot: URL, withMetadata: Bool, etag: String = "etag-not-a-sha256") throws {
        let fileURL = repoRoot.appendingPathComponent(relPath)
        try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("model-bytes".utf8).write(to: fileURL)

        guard withMetadata else { return }
        // Sidecar layout: <repoRoot>/.cache/huggingface/download/<relPath>.metadata
        // Format (see HubApi.writeDownloadMetadata): commitHash\netag\ntimestamp\n
        let metaURL = repoRoot
            .appendingPathComponent(".cache/huggingface/download")
            .appendingPathComponent(relPath + ".metadata")
        try FileManager.default.createDirectory(at: metaURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let contents = "1111111111111111111111111111111111111111\n\(etag)\n1700000000.0\n"
        try contents.write(to: metaURL, atomically: true, encoding: .utf8)
    }

    private func populateCache(metadataFor includeMetadata: (Int) -> Bool, components: Range<Int>? = nil, in repoRoot: URL) throws {
        let indices = components ?? 0..<componentFiles.count
        for index in indices {
            try writeCachedFile(componentFiles[index], in: repoRoot, withMetadata: includeMetadata(index))
        }
    }

    // MARK: - resolveRepo

    /// A complete, valid cache resolves offline and returns the cache root — no network.
    func testResolveRepoReturnsValidatedCache() async throws {
        let (downloader, repoRoot) = makeDownloader()
        try populateCache(metadataFor: { _ in true }, in: repoRoot)

        let resolved = try await downloader.resolveRepo(patterns: patterns, download: false)

        XCTAssertEqual(resolved.standardizedFileURL, repoRoot.standardizedFileURL)
    }

    /// The reported bug: the segmenter dir exists and is non-empty (so the old directory-only check
    /// passed) but its file has no `.metadata` sidecar. That cache must be rejected, not served.
    func testResolveRepoRejectsPartialCacheMissingMetadata() async throws {
        let (downloader, repoRoot) = makeDownloader()
        // Segmenter has no sidecar (an interrupted download); the other two components are complete.
        try populateCache(metadataFor: { index in index != 0 }, in: repoRoot)

        // The segmenter dir is present and non-empty, so the old directory-only check would pass.
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: repoRoot.appendingPathComponent("speaker_segmenter/pyannote-v3/W8A16").path
        ))

        do {
            _ = try await downloader.resolveRepo(patterns: patterns, download: false)
            XCTFail("Expected resolveRepo to reject a partial cache that fails offline validation")
        } catch let error as ModelDownloaderError {
            XCTAssertTrue(
                error.localizedDescription.contains("download is disabled"),
                "Unexpected error: \(error.localizedDescription)"
            )
        }
    }

    /// A file whose sidecar claims a SHA256 that doesn't match its bytes (a corrupt download) must
    /// fail the hash check rather than be served from cache.
    func testResolveRepoRejectsCorruptLFSHash() async throws {
        let (downloader, repoRoot) = makeDownloader()
        // Segmenter's sidecar declares a SHA256 that won't match "model-bytes"; the others are intact.
        try writeCachedFile(componentFiles[0], in: repoRoot, withMetadata: true, etag: String(repeating: "0", count: 64))
        try writeCachedFile(componentFiles[1], in: repoRoot, withMetadata: true)
        try writeCachedFile(componentFiles[2], in: repoRoot, withMetadata: true)

        do {
            _ = try await downloader.resolveRepo(patterns: patterns, download: false)
            XCTFail("Expected resolveRepo to reject a cache whose LFS hash does not match")
        } catch let error as ModelDownloaderError {
            XCTAssertTrue(
                error.localizedDescription.contains("download is disabled"),
                "Unexpected error: \(error.localizedDescription)"
            )
        }
    }

    /// An unrelated file elsewhere in the repo (e.g. a different, partially-present variant with no
    /// metadata) must not fail validation when the requested patterns are complete — offline
    /// validation is scoped to `patterns`.
    func testResolveRepoIgnoresUnrelatedFilesInRepo() async throws {
        let (downloader, repoRoot) = makeDownloader()
        // The three requested components are complete and valid.
        try populateCache(metadataFor: { _ in true }, in: repoRoot)
        // A different segmenter variant (W4A16) that matches none of the requested patterns, with no
        // metadata sidecar — like a sideloaded or partially-downloaded asset.
        try writeCachedFile(
            "speaker_segmenter/pyannote-v3/W4A16/SpeakerSegmenter.mlmodelc/coremldata.bin",
            in: repoRoot, withMetadata: false
        )

        let resolved = try await downloader.resolveRepo(patterns: patterns, download: false)

        XCTAssertEqual(resolved.standardizedFileURL, repoRoot.standardizedFileURL)
    }

    /// A whole component is missing, so the presence check fails outright. The offline snapshot can't
    /// catch this on its own — it only validates files that exist — which is why we keep the check.
    func testResolveRepoThrowsWhenComponentMissing() async throws {
        let (downloader, repoRoot) = makeDownloader()
        // Only segmenter + embedder present; clusterer never written.
        try populateCache(metadataFor: { _ in true }, components: 0..<2, in: repoRoot)

        XCTAssertFalse(FileManager.default.fileExists(
            atPath: repoRoot.appendingPathComponent("speaker_clusterer/pyannote-v4/W32A32").path
        ))

        do {
            _ = try await downloader.resolveRepo(patterns: patterns, download: false)
            XCTFail("Expected resolveRepo to throw when a component is entirely missing")
        } catch is ModelDownloaderError {
            // expected
        }
    }

    // MARK: - resolveModel

    /// When the file is already in an explicit `modelFolder`, resolveModel returns it directly and
    /// never reaches the cache or the network.
    func testResolveModelReturnsFileFromModelFolder() async throws {
        let downloader = ModelDownloader(config: ModelDownloadConfig(modelRepo: testRepo))
        let info = ModelInfo(version: "pyannote-v3", variant: "W8A16", name: "speaker_segmenter", computeUnits: .cpuAndNeuralEngine)

        let modelFolder = tempDir.appendingPathComponent("models")
        let expected = info.modelURL(baseURL: modelFolder).appendingPathComponent("SpeakerSegmenter.mlmodelc")
        try FileManager.default.createDirectory(at: expected, withIntermediateDirectories: true)

        let resolved = try await downloader.resolveModel(
            "SpeakerSegmenter", using: info, modelFolder: modelFolder, download: false
        )

        XCTAssertEqual(resolved.standardizedFileURL, expected.standardizedFileURL)
    }

    // MARK: - ModelInfo

    /// `downloadPattern` is the glob resolveRepo / downloadModel build their requests from; missing
    /// fields become `*`.
    func testModelInfoDownloadPattern() {
        let full = ModelInfo(version: "pyannote-v3", variant: "W8A16", name: "speaker_segmenter", computeUnits: .cpuAndNeuralEngine)
        XCTAssertEqual(full.downloadPattern, "speaker_segmenter/pyannote-v3/W8A16/*")

        let minimal = ModelInfo(name: "speaker_segmenter", computeUnits: .cpuAndNeuralEngine)
        XCTAssertEqual(minimal.downloadPattern, "speaker_segmenter/*/*/*")
    }

    /// `modelURL` appends name/version/variant in order, skipping any that are nil.
    func testModelInfoModelURL() {
        let base = URL(fileURLWithPath: "/tmp/models")

        let full = ModelInfo(version: "pyannote-v3", variant: "W8A16", name: "speaker_segmenter", computeUnits: .cpuAndNeuralEngine)
        XCTAssertEqual(full.modelURL(baseURL: base).path, "/tmp/models/speaker_segmenter/pyannote-v3/W8A16")

        let nameOnly = ModelInfo(name: "speaker_segmenter", computeUnits: .cpuAndNeuralEngine)
        XCTAssertEqual(nameOnly.modelURL(baseURL: base).path, "/tmp/models/speaker_segmenter")
    }

    /// `findBaseFolder` walks up to the directory named after the model and returns its parent, or nil
    /// when the name isn't in the path.
    func testModelInfoFindBaseFolder() {
        let info = ModelInfo(version: "pyannote-v3", variant: "W8A16", name: "speaker_segmenter", computeUnits: .cpuAndNeuralEngine)

        let url = URL(fileURLWithPath: "/tmp/models/speaker_segmenter/pyannote-v3/W8A16")
        XCTAssertEqual(info.findBaseFolder(in: url)?.path, "/tmp/models")

        XCTAssertNil(info.findBaseFolder(in: URL(fileURLWithPath: "/tmp/somewhere/else")))
    }
}
