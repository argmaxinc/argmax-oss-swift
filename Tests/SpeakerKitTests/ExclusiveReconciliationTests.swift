//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import XCTest
import WhisperKit
@testable import SpeakerKit

final class ExclusiveReconciliationTests: XCTestCase {

    private func loadAudio(named name: String, extension ext: String = "wav") throws -> [Float] {
        guard let url = Bundle.module.url(forResource: name, withExtension: ext) else {
            throw XCTSkip("Audio file \(name).\(ext) not found in test bundle")
        }
        let audioBuffer = try AudioProcessor.loadAudio(fromPath: url.path)
        return AudioProcessor.convertBufferToArray(buffer: audioBuffer)
    }

    func testExclusiveReconciliationWithPyannote4() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()

        let baseOptions = PyannoteDiarizationOptions(
            clusterDistanceThreshold: 0.3,
            minClusterSize: 1
        )

        let normalResult = try await speakerKit.diarize(audioArray: audioArray, options: PyannoteDiarizationOptions(
            numberOfSpeakers: baseOptions.numberOfSpeakers,
            minActiveOffset: baseOptions.minActiveOffset,
            clusterDistanceThreshold: baseOptions.clusterDistanceThreshold,
            minClusterSize: baseOptions.minClusterSize,
            useExclusiveReconciliation: false
        ))

        let exclusiveResult = try await speakerKit.diarize(audioArray: audioArray, options: PyannoteDiarizationOptions(
            numberOfSpeakers: baseOptions.numberOfSpeakers,
            minActiveOffset: baseOptions.minActiveOffset,
            clusterDistanceThreshold: baseOptions.clusterDistanceThreshold,
            minClusterSize: baseOptions.minClusterSize,
            useExclusiveReconciliation: true
        ))

        XCTAssertFalse(normalResult.segments.isEmpty, "Normal result should have segments")
        XCTAssertFalse(exclusiveResult.segments.isEmpty, "Exclusive result should have segments")
        XCTAssertEqual(normalResult.frameRate, exclusiveResult.frameRate)

        let normalOverlaps = countOverlappingFrames(in: normalResult)
        let exclusiveOverlaps = countOverlappingFrames(in: exclusiveResult)

        XCTAssertLessThanOrEqual(exclusiveOverlaps, normalOverlaps,
                                 "Exclusive reconciliation should reduce or maintain overlap count")
    }

    func testExclusiveReconciliationWithSpeakers() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()

        let aggressiveOptions = PyannoteDiarizationOptions(
            numberOfSpeakers: 3,
            clusterDistanceThreshold: 0.25,
            minClusterSize: 1,
            useExclusiveReconciliation: false
        )

        let normalResult = try await speakerKit.diarize(audioArray: audioArray, options: aggressiveOptions)

        let exclusiveResult = try await speakerKit.diarize(audioArray: audioArray, options: PyannoteDiarizationOptions(
            numberOfSpeakers: aggressiveOptions.numberOfSpeakers,
            minActiveOffset: aggressiveOptions.minActiveOffset,
            clusterDistanceThreshold: aggressiveOptions.clusterDistanceThreshold,
            minClusterSize: aggressiveOptions.minClusterSize,
            useExclusiveReconciliation: true
        ))

        XCTAssertFalse(normalResult.segments.isEmpty, "Normal result should have segments")
        XCTAssertFalse(exclusiveResult.segments.isEmpty, "Exclusive result should have segments")

        let normalOverlaps = countOverlappingFrames(in: normalResult)
        let exclusiveOverlaps = countOverlappingFrames(in: exclusiveResult)

        XCTAssertLessThanOrEqual(exclusiveOverlaps, normalOverlaps,
                                 "Exclusive reconciliation should reduce overlapping speakers")
    }

    func testExclusiveReconciliationWithForcedSpeakerCount() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()

        let forcedOptions = PyannoteDiarizationOptions(
            numberOfSpeakers: 3,
            clusterDistanceThreshold: 0.3,
            minClusterSize: 1
        )

        let normalResult = try await speakerKit.diarize(audioArray: audioArray, options: PyannoteDiarizationOptions(
            numberOfSpeakers: forcedOptions.numberOfSpeakers,
            minActiveOffset: forcedOptions.minActiveOffset,
            clusterDistanceThreshold: forcedOptions.clusterDistanceThreshold,
            minClusterSize: forcedOptions.minClusterSize,
            useExclusiveReconciliation: false
        ))

        let exclusiveResult = try await speakerKit.diarize(audioArray: audioArray, options: PyannoteDiarizationOptions(
            numberOfSpeakers: forcedOptions.numberOfSpeakers,
            minActiveOffset: forcedOptions.minActiveOffset,
            clusterDistanceThreshold: forcedOptions.clusterDistanceThreshold,
            minClusterSize: forcedOptions.minClusterSize,
            useExclusiveReconciliation: true
        ))

        let normalOverlaps = countOverlappingFrames(in: normalResult)
        let exclusiveOverlaps = countOverlappingFrames(in: exclusiveResult)

        XCTAssertLessThanOrEqual(exclusiveOverlaps, normalOverlaps,
                                 "Exclusive reconciliation should work with forced speaker count")
    }

    // MARK: - Helpers

    private func countOverlappingFrames(in result: DiarizationResult) -> Int {
        guard !result.segments.isEmpty else { return 0 }

        let totalFrames = result.totalFrames
        let speakerCount = result.speakerCount
        var binaryMatrix = Array(repeating: Array(repeating: 0, count: totalFrames), count: speakerCount)

        for segment in result.segments {
            for speakerId in segment.speaker.speakerIds {
                if speakerId < speakerCount {
                    for frame in segment.startFrame..<min(segment.endFrame, totalFrames) {
                        binaryMatrix[speakerId][frame] = 1
                    }
                }
            }
        }

        var overlappingFrames = 0
        for frameIndex in 0..<totalFrames {
            let activeSpeakers = binaryMatrix.filter { frameIndex < $0.count && $0[frameIndex] == 1 }.count
            if activeSpeakers > 1 {
                overlappingFrames += 1
            }
        }
        return overlappingFrames
    }
}
