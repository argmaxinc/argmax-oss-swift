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

    func testDiarizationReportsEmbeddingProgressBeforeClustering() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()
        let recorder = ProgressRecorder()
        _ = try await speakerKit.diarize(audioArray: audioArray) { progress in
            recorder.append(progress.completedUnitCount)
        }

        let values = recorder.values
        XCTAssertTrue(values.contains { $0 > 0 && $0 < 85 },
                      "Segmentation + embedding progress (0-85) should reach the caller before clustering, got \(values)")
        XCTAssertTrue(zip(values, values.dropFirst()).allSatisfy { $0 < $1 }, "Progress should only move forward, got \(values)")
        XCTAssertEqual(values.last, 100, "Progress should end at 100, got \(values)")
    }

    func testDiarizationStopsWhenCancelledDuringEmbedding() async throws {
        // Repeat the clip so the segmenter yields many chunks, leaving embedding work to skip after the cancel.
        let clip = try loadAudio(named: "VADAudio")
        let audioArray = Array(repeating: clip, count: 20).flatMap { $0 }
        let speakerKit = try await SpeakerKit()
        let recorder = ProgressRecorder()
        let relay = CancellationRelay()
        let diarization = Task {
            try await speakerKit.diarize(audioArray: audioArray) { progress in
                recorder.append(progress.completedUnitCount)
                if progress.completedUnitCount > 0 {
                    relay.cancel()
                }
            }
        }
        relay.attach(diarization)

        do {
            _ = try await diarization.value
            XCTFail("A cancelled diarization should throw instead of returning a result, got \(recorder.values)")
        } catch is CancellationError {}
        XCTAssertLessThan(recorder.values.max() ?? 0, 84, "Embedding should stop soon after the cancel, got \(recorder.values)")
    }
}

/// Cancels a task from a callback that may fire before the task handle is stored.
private final class CancellationRelay: @unchecked Sendable {
    private let lock = NSLock()
    private var task: Task<DiarizationResult, Error>?
    private var isCancelRequested = false

    func attach(_ task: Task<DiarizationResult, Error>) {
        lock.withLock {
            self.task = task
            if isCancelRequested { task.cancel() }
        }
    }

    func cancel() {
        lock.withLock {
            isCancelRequested = true
            task?.cancel()
        }
    }
}

private final class ProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [Int64] = []

    var values: [Int64] {
        lock.withLock { recorded }
    }

    func append(_ value: Int64) {
        lock.withLock { recorded.append(value) }
    }
}
