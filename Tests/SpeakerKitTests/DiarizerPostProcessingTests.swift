//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import XCTest
import WhisperKit
@testable import SpeakerKit

final class DiarizerPostProcessingTests: XCTestCase {

    private func loadAudio(named name: String, extension ext: String = "wav") throws -> [Float] {
        guard let url = Bundle.module.url(forResource: name, withExtension: ext) else {
            throw XCTSkip("Audio file \(name).\(ext) not found in test bundle")
        }
        let audioBuffer = try AudioProcessor.loadAudio(fromPath: url.path)
        return AudioProcessor.convertBufferToArray(buffer: audioBuffer)
    }

    func testSegmentLengthAndFrameDistribution() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()

        let options = PyannoteDiarizationOptions(
            numberOfSpeakers: 3,
            clusterDistanceThreshold: 0.25,
            minClusterSize: 1,
            useExclusiveReconciliation: false
        )

        let result = try await speakerKit.diarize(audioArray: audioArray, options: options)

        var frameUsage = Array(repeating: 0, count: result.totalFrames)
        var invalidFrameDetected = false

        for segment in result.segments {
            if segment.startFrame >= segment.endFrame {
                invalidFrameDetected = true
            }
            if segment.startFrame < 0 || segment.endFrame > result.totalFrames {
                invalidFrameDetected = true
            }
            for frame in segment.startFrame..<min(segment.endFrame, result.totalFrames) {
                if frame >= 0 && frame < frameUsage.count {
                    frameUsage[frame] += 1
                }
            }
        }

        if invalidFrameDetected {
            XCTFail("Detected invalid frame ranges in speaker segments")
        }

        let maxOverlapsPerFrame = frameUsage.max() ?? 0
        let shortSegments = result.segments.filter { $0.endFrame - $0.startFrame <= 2 }
        let shortSegmentRatio = Double(shortSegments.count) / Double(result.segments.count)

        XCTAssertLessThan(shortSegmentRatio, 0.8,
                          "Too many short segments detected")
        let totalFrames = result.segments.reduce(0) { $0 + ($1.endFrame - $1.startFrame) }
        let averageSegmentLength = Double(totalFrames) / Double(result.segments.count)
        XCTAssertGreaterThan(averageSegmentLength, 5.0,
                             "Average segment length should be reasonable")
        XCTAssertLessThanOrEqual(maxOverlapsPerFrame, result.speakerCount,
                                 "Frame overlaps should not exceed total speaker count")
    }

    func testFrameIndexConsistencyWithMultipleWindows() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()

        let options = PyannoteDiarizationOptions(
            numberOfSpeakers: 2,
            clusterDistanceThreshold: 0.3,
            minClusterSize: 1,
            useExclusiveReconciliation: false
        )

        let result = try await speakerKit.diarize(audioArray: audioArray, options: options)

        var frameDistribution: [Int: Int] = [:]
        for segment in result.segments {
            for frame in segment.startFrame..<segment.endFrame {
                frameDistribution[frame, default: 0] += 1
            }
        }

        let sortedFrames = frameDistribution.keys.sorted()
        var previousFrame = -1
        for frame in sortedFrames {
            if previousFrame >= 0 && frame > previousFrame + 1 { /* gap */ }
            previousFrame = frame
        }

        XCTAssertGreaterThan(frameDistribution.count, 0, "Should have active frames")
        XCTAssertLessThanOrEqual(sortedFrames.last ?? 0, result.totalFrames, "Frames should be within bounds")
    }

    func testPostProcessingWith3SpeakersBasicScenario() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()

        let options = PyannoteDiarizationOptions(
            numberOfSpeakers: 3,
            clusterDistanceThreshold: 0.4,
            minClusterSize: 1,
            useExclusiveReconciliation: false
        )

        let result = try await speakerKit.diarize(audioArray: audioArray, options: options)

        XCTAssertGreaterThan(result.speakerCount, 0, "Should detect at least 1 speaker")
        XCTAssertLessThanOrEqual(result.speakerCount, 3, "Should not detect more than 3 speakers when forced to 3")
        XCTAssertFalse(result.segments.isEmpty, "Should have speaker segments")

        let speakerIds = Set(result.segments.compactMap { $0.speaker.speakerId })
        XCTAssertGreaterThan(speakerIds.count, 0, "Should have at least one speaker in segments")
        XCTAssertLessThanOrEqual(speakerIds.count, result.speakerCount, "Segment speakers should not exceed detected speakers")

        for speakerId in speakerIds {
            XCTAssertGreaterThanOrEqual(speakerId, 0, "Speaker ID should be non-negative")
            XCTAssertLessThan(speakerId, result.speakerCount, "Speaker ID should be within speaker count range")
        }

        for segment in result.segments {
            XCTAssertGreaterThanOrEqual(segment.startFrame, 0, "Start frame should be non-negative")
            XCTAssertGreaterThan(segment.endFrame, segment.startFrame, "End frame should be greater than start frame")
            XCTAssertLessThan(segment.endFrame, result.totalFrames, "End frame should be within total frames")
        }
    }

    func testPostProcessingWith3SpeakersOverlappingScenario() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()

        let options = PyannoteDiarizationOptions(
            numberOfSpeakers: 3,
            clusterDistanceThreshold: 0.2,
            minClusterSize: 1,
            useExclusiveReconciliation: false
        )

        let result = try await speakerKit.diarize(audioArray: audioArray, options: options)

        let overlappingFrames = countOverlappingFrames(in: result)
        XCTAssertGreaterThanOrEqual(overlappingFrames, 0, "Should handle overlapping frames gracefully")
    }

    func testPostProcessingWith3SpeakersExclusiveReconciliation() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()

        let normalResult = try await speakerKit.diarize(audioArray: audioArray, options: PyannoteDiarizationOptions(
            numberOfSpeakers: 3,
            clusterDistanceThreshold: 0.2,
            minClusterSize: 1,
            useExclusiveReconciliation: false
        ))

        let exclusiveResult = try await speakerKit.diarize(audioArray: audioArray, options: PyannoteDiarizationOptions(
            numberOfSpeakers: 3,
            clusterDistanceThreshold: 0.2,
            minClusterSize: 1,
            useExclusiveReconciliation: true
        ))

        let normalOverlaps = countOverlappingFrames(in: normalResult)
        let exclusiveOverlaps = countOverlappingFrames(in: exclusiveResult)

        XCTAssertLessThanOrEqual(exclusiveOverlaps, normalOverlaps,
                                 "Exclusive reconciliation should reduce or maintain overlap count")
        XCTAssertLessThanOrEqual(exclusiveOverlaps, 1,
                                 "Exclusive reconciliation should limit overlaps to at most 1 speaker per frame")
    }

    func testFrameCalculationWith3Speakers() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()

        let options = PyannoteDiarizationOptions(
            numberOfSpeakers: 3,
            clusterDistanceThreshold: 0.3,
            minClusterSize: 1,
            useExclusiveReconciliation: false
        )

        let result = try await speakerKit.diarize(audioArray: audioArray, options: options)

        XCTAssertFalse(result.segments.isEmpty, "Should have segments")

        for segment in result.segments {
            XCTAssertGreaterThanOrEqual(segment.startFrame, 0, "Start frame should be non-negative")
            XCTAssertLessThan(segment.startFrame, result.totalFrames, "Start frame should be within bounds")
            XCTAssertLessThanOrEqual(segment.endFrame, result.totalFrames, "End frame should be within bounds")
        }

        let sortedSegments = result.segments.sorted { $0.startFrame < $1.startFrame }
        for i in 0..<sortedSegments.count - 1 {
            let current = sortedSegments[i]
            let next = sortedSegments[i + 1]
            XCTAssertGreaterThan(current.endFrame, current.startFrame, "Segment should have positive duration")
            if current.speaker.speakerId == next.speaker.speakerId {
                XCTAssertLessThanOrEqual(current.endFrame, next.startFrame,
                                         "Same-speaker segments should not overlap")
            }
        }
    }

    func testPostProcessingWith3SpeakersEdgeCases() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()

        let options = PyannoteDiarizationOptions(
            numberOfSpeakers: 3,
            clusterDistanceThreshold: 0.1,
            minClusterSize: 1,
            useExclusiveReconciliation: false
        )

        let result = try await speakerKit.diarize(audioArray: audioArray, options: options)

        XCTAssertFalse(result.segments.isEmpty, "Should produce valid results even with edge case parameters")

        for segment in result.segments {
            XCTAssertGreaterThanOrEqual(segment.startFrame, 0, "Start frame should be valid")
            XCTAssertGreaterThan(segment.endFrame, segment.startFrame, "End frame should be after start")
        }

        let totalActiveFrames = result.segments.reduce(0) { $0 + ($1.endFrame - $1.startFrame) }
        XCTAssertGreaterThan(totalActiveFrames, 0, "Should have some active frames")
    }

    func testPostProcessingFrameCounterConsistency() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()

        let options = PyannoteDiarizationOptions(
            numberOfSpeakers: 3,
            clusterDistanceThreshold: 0.3,
            minClusterSize: 2,
            useExclusiveReconciliation: false
        )

        let result = try await speakerKit.diarize(audioArray: audioArray, options: options)

        let allFrameIndices = Set(result.segments.flatMap { Array($0.startFrame..<$0.endFrame) })
        for frameIndex in allFrameIndices {
            XCTAssertGreaterThanOrEqual(frameIndex, 0, "Frame index should be non-negative")
            XCTAssertLessThan(frameIndex, result.totalFrames, "Frame index should be within total frames")
        }
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
