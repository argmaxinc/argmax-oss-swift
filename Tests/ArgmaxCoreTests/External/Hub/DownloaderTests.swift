// Originally from: https://github.com/huggingface/swift-transformers
// Version: 1.1.6 (commit: 573e5c9036c2f136b3a8a071da8e8907322403d0)
// License: Apache 2.0 (https://github.com/huggingface/swift-transformers/blob/main/LICENSE)
// Copyright 2022 Hugging Face SAS
// Modified by Argmax, Inc. See Argmax-modification: comments for changes.
//

//
//  DownloaderTests.swift
//  swift-transformers
//
//  Created by Arda Atahan Ibis on 1/28/25.
//

import XCTest

// Argmax-modification: consolidated the upstream swift-transformers module import(s) (Hub) into the single ArgmaxCore module that now vendors this code.
@testable import ArgmaxCore

// Argmax-modification: removed the network download tests (`testSuccessfulDownload`,
// `testDownloadFailsWithIncorrectSize`, `testSuccessfulInterruptedDownload`); only the local
// `testMoveDownloadedFilePercentEncodedFlag` is kept.
final class DownloaderTests: XCTestCase {
    var tempDir: URL!

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

    func testMoveDownloadedFilePercentEncodedFlag() throws {
        let appSupport = tempDir.appendingPathComponent("Application Support")
        let destination = appSupport.appendingPathComponent("config.json")
        let source1 = tempDir.appendingPathComponent("v1.incomplete")
        let source2 = tempDir.appendingPathComponent("v2.incomplete")

        try FileManager.default.createDirectory(at: appSupport, withIntermediateDirectories: true)
        try "existing".write(to: destination, atomically: true, encoding: .utf8)
        try "v1".write(to: source1, atomically: true, encoding: .utf8)
        try "v2".write(to: source2, atomically: true, encoding: .utf8)

        XCTAssertThrowsError(
            try FileManager.default.moveDownloadedFile(from: source1, to: destination, percentEncoded: true)
        ) { error in
            XCTAssertEqual((error as NSError).code, 516)
        }
        XCTAssertEqual(try String(contentsOf: destination, encoding: .utf8), "existing")

        XCTAssertNoThrow(
            try FileManager.default.moveDownloadedFile(from: source2, to: destination, percentEncoded: false)
        )
        XCTAssertEqual(try String(contentsOf: destination, encoding: .utf8), "v2")
    }
}
