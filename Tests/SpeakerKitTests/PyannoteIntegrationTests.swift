//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import XCTest
import WhisperKit
@testable import SpeakerKit

final class PyannoteIntegrationTests: XCTestCase {

    private func loadAudio(named name: String, extension ext: String = "wav") throws -> [Float] {
        guard let url = Bundle.module.url(forResource: name, withExtension: ext) else {
            throw XCTSkip("Audio file \(name).\(ext) not found in test bundle")
        }
        let audioBuffer = try AudioProcessor.loadAudio(fromPath: url.path)
        return AudioProcessor.convertBufferToArray(buffer: audioBuffer)
    }

    func testDiarizationWithCustomClusteringOptions() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()
        let defaultResult = try await speakerKit.diarize(audioArray: audioArray)
        XCTAssertGreaterThan(defaultResult.speakerCount, 0, "Should have at least one speaker")
        XCTAssertFalse(defaultResult.segments.isEmpty, "Should have at least one segment")

        let resultWithLowerThreshold = try await speakerKit.diarize(audioArray: audioArray, options: PyannoteDiarizationOptions(
            clusterDistanceThreshold: 0.2,
            minClusterSize: 1
        ))
        XCTAssertGreaterThanOrEqual(resultWithLowerThreshold.speakerCount, defaultResult.speakerCount,
                                    "Lower threshold should result in equal or more speakers")
    }

    func testDiarizationOptionsWithMinActiveOffset() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()
        let result = try await speakerKit.diarize(audioArray: audioArray, options: PyannoteDiarizationOptions(
            numberOfSpeakers: nil,
            minActiveOffset: 0.5,
            clusterDistanceThreshold: 0.3,
            minClusterSize: 1,
            useExclusiveReconciliation: false
        ))

        XCTAssertGreaterThan(result.speakerCount, 0, "Should have at least one speaker")
        XCTAssertFalse(result.segments.isEmpty, "Should have at least one segment")
    }

    func testDiarizationBasicSanity() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()
        let result = try await speakerKit.diarize(audioArray: audioArray)
        XCTAssertGreaterThan(result.speakerCount, 0, "Should have at least one speaker")
        XCTAssertFalse(result.segments.isEmpty, "Should have at least one segment")
    }
}
