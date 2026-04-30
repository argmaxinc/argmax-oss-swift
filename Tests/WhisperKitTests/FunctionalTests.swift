//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import WhisperKit
import XCTest

final class FunctionalTests: XCTestCase {
    func testInitLarge() async throws {
        try await XCTAssertNoThrowAsync(
            await WhisperKit(modelFolder: largev3ModelPath(), logLevel: .error)
        )
    }

    func testRealTimeFactorTiny() async throws {
        let modelPath = try await tinyModelPath()

        let metrics: [XCTMetric] = [XCTMemoryMetric(), XCTStorageMetric(), XCTClockMetric()]

        let measureOptions = XCTMeasureOptions.default
        measureOptions.iterationCount = 5

        let audioFilePath = try XCTUnwrap(
            Bundle.current(for: self).path(forResource: "jfk", ofType: "wav"),
            "Audio file not found"
        )

        let whisperKit = try await WhisperKit(WhisperKitConfig(modelFolder: modelPath))
        let transcriptionRunner = TranscriptionRunner(whisperKit)

        measureAsync(
            metrics: metrics,
            options: measureOptions
        ) { [transcriptionRunner, audioFilePath] in
            try await transcriptionRunner.transcribe(audioPath: audioFilePath)
        } assertion: { result in
            XCTAssertGreaterThan(result.text.count, 0)
        }
    }

    func testRealTimeFactorLarge() async throws {
        let modelPath = try largev3ModelPath()

        let metrics: [XCTMetric] = [XCTMemoryMetric(), XCTStorageMetric(), XCTClockMetric()]

        let measureOptions = XCTMeasureOptions.default
        measureOptions.iterationCount = 5

        let audioFilePath = try XCTUnwrap(
            Bundle.current(for: self).path(forResource: "jfk", ofType: "wav"),
            "Audio file not found"
        )

        let whisperKit = try await WhisperKit(WhisperKitConfig(modelFolder: modelPath, verbose: false))
        let transcriptionRunner = TranscriptionRunner(whisperKit)

        measureAsync(
            metrics: metrics,
            options: measureOptions
        ) { [transcriptionRunner, audioFilePath] in
            try await transcriptionRunner.transcribe(audioPath: audioFilePath)
        } assertion: { result in
            XCTAssertGreaterThan(result.text.count, 0)
        }
    }

    func testBaseImplementation() async throws {
        let audioFilePath = try XCTUnwrap(
            Bundle.current(for: self).path(forResource: "jfk", ofType: "wav"),
            "Audio file not found"
        )

        let whisperKit = try await XCTUnwrapAsync(await WhisperKit(model: "large-v3"))
        let transcriptionResult: [TranscriptionResult] = try await whisperKit.transcribe(audioPath: audioFilePath)
        XCTAssertGreaterThan(transcriptionResult.text.count, 0)
    }

    func testAsyncImplementation() async throws {
        let audioFilePath = try XCTUnwrap(
            Bundle.current(for: self).path(forResource: "jfk", ofType: "wav"),
            "Audio file not found"
        )
        let whisperKit = try await WhisperKit(WhisperKitConfig(model: "large-v3"))
        let transcriptionResult: [TranscriptionResult] = try await whisperKit.transcribe(audioPath: audioFilePath)

        XCTAssertGreaterThan(transcriptionResult.text.count, 0)
    }

    func testBatchTranscribeAudioPaths() async throws {
        let audioPaths = try [
            XCTUnwrap(
                Bundle.current(for: self).path(forResource: "jfk", ofType: "wav"),
                "Audio file not found"
            ),
            XCTUnwrap(
                Bundle.current(for: self).path(forResource: "es_test_clip", ofType: "wav"),
                "Audio file not found"
            ),
            XCTUnwrap(
                Bundle.current(for: self).path(forResource: "ja_test_clip", ofType: "wav"),
                "Audio file not found"
            ),
        ]
        let whisperKit = try await WhisperKit(WhisperKitConfig(modelFolder: tinyModelPath()))
        let transcriptionResults: [Result<[TranscriptionResult], Swift.Error>] = await whisperKit.transcribeWithResults(audioPaths: audioPaths)

        XCTAssertEqual(transcriptionResults.count, 3)
        XCTAssertTrue(transcriptionResults.allSatisfy { $0.isSuccess })
        XCTAssertEqual(
            try transcriptionResults[0].normalizedText(prefix: 5),
            "and so my fellow americans"
        )
        XCTAssertEqual(
            try transcriptionResults[1].normalizedText(prefix: 2),
            "this is"
        )
        XCTAssertEqual(
            try transcriptionResults[2].normalizedText(prefix: 1),
            "tokyo"
        )
    }

    func testBatchTranscribeAudioPathsWithErrors() async throws {
        let audioPaths = try [
            "/path/to/file1.wav",
            XCTUnwrap(
                Bundle.current(for: self).path(forResource: "jfk", ofType: "wav"),
                "Audio file not found"
            ),
            "/path/to/file2.wav",
        ]
        let whisperKit = try await WhisperKit(WhisperKitConfig(modelFolder: tinyModelPath()))
        let transcriptionResults: [Result<[TranscriptionResult], Swift.Error>] = await whisperKit.transcribeWithResults(audioPaths: audioPaths)

        XCTAssertEqual(transcriptionResults.count, 3)
        XCTAssertEqual(
            transcriptionResults[0].whisperError(),
            .loadAudioFailed("Resource path does not exist /path/to/file1.wav")
        )
        XCTAssertEqual(
            try transcriptionResults[1].normalizedText(prefix: 5),
            "and so my fellow americans"
        )
        XCTAssertEqual(
            transcriptionResults[2].whisperError(),
            .loadAudioFailed("Resource path does not exist /path/to/file2.wav")
        )
    }

    func testBatchTranscribeAudioArrays() async throws {
        let audioPaths = try [
            XCTUnwrap(
                Bundle.current(for: self).path(forResource: "jfk", ofType: "wav"),
                "Audio file not found"
            ),
            XCTUnwrap(
                Bundle.current(for: self).path(forResource: "es_test_clip", ofType: "wav"),
                "Audio file not found"
            ),
            XCTUnwrap(
                Bundle.current(for: self).path(forResource: "ja_test_clip", ofType: "wav"),
                "Audio file not found"
            ),
        ]
        let audioArrays = try audioPaths
            .map { try AudioProcessor.loadAudio(fromPath: $0) }
            .map { AudioProcessor.convertBufferToArray(buffer: $0) }

        let whisperKit = try await WhisperKit(WhisperKitConfig(modelFolder: tinyModelPath()))
        let transcriptionResults: [Result<[TranscriptionResult], Swift.Error>] = await whisperKit.transcribeWithResults(audioArrays: audioArrays)

        XCTAssertEqual(transcriptionResults.count, 3)
        XCTAssertTrue(transcriptionResults.allSatisfy { $0.isSuccess })
        XCTAssertEqual(
            try transcriptionResults[0].normalizedText(prefix: 5),
            "and so my fellow americans"
        )
        XCTAssertEqual(
            try transcriptionResults[1].normalizedText(prefix: 2),
            "this is"
        )
        XCTAssertEqual(
            try transcriptionResults[2].normalizedText(prefix: 1),
            "tokyo"
        )
    }

    func testModelSearchPathLarge() async throws {
        let audioFilePath = try XCTUnwrap(
            Bundle.current(for: self).path(forResource: "jfk", ofType: "wav"),
            "Audio file not found"
        )

        var config = WhisperKitConfig(model: "large-v3", verbose: true, logLevel: .debug)
        let pipe1 = try await WhisperKit(config)
        let transcriptionResult1: [TranscriptionResult] = try await pipe1.transcribe(audioPath: audioFilePath)
        XCTAssertFalse(transcriptionResult1.text.isEmpty)

        config = WhisperKitConfig(model: "distil*large-v3", verbose: true, logLevel: .debug)
        let pipe2 = try await WhisperKit(config)
        let transcriptionResult2: [TranscriptionResult] = try await pipe2.transcribe(audioPath: audioFilePath)
        XCTAssertFalse(transcriptionResult2.text.isEmpty)

        config = WhisperKitConfig(model: "distil*large-v3", verbose: true, logLevel: .debug)
        let pipe3 = try await WhisperKit(config)
        let transcriptionResult3: [TranscriptionResult] = try await pipe3.transcribe(audioPath: audioFilePath)
        XCTAssertFalse(transcriptionResult3.text.isEmpty)
    }
}

// MARK: - Helper Types and Methods

private actor TranscriptionRunner {
    private let whisperKit: WhisperKit

    init(_ whisperKit: WhisperKit) { self.whisperKit = whisperKit }

    func transcribe(audioPath: String) async throws -> [TranscriptionResult] {
        try await whisperKit.transcribe(audioPath: audioPath)
    }
}

private extension FunctionalTests {
    func measureAsync<T: Sendable>(
        metrics: [XCTMetric],
        options: XCTMeasureOptions,
        timeout: TimeInterval = 120,
        operation: @escaping @Sendable () async throws -> T,
        assertion: @escaping @Sendable (T) -> Void
    ) {
        measure(metrics: metrics, options: options) {
            let task = Task { try await operation() }
            let done = expectation(description: "measure async iteration")

            Task {
                do {
                    let value = try await task.value
                    assertion(value)
                    done.fulfill()
                } catch is CancellationError {
                    // Timeout path cancels `task`; avoid reporting a duplicate failure.
                    if !task.isCancelled {
                        XCTFail("Measured async operation was cancelled unexpectedly")
                    }
                    done.fulfill()
                } catch {
                    XCTFail("Measured async operation failed: \(error)")
                    done.fulfill()
                }
            }

            let waitResult = XCTWaiter.wait(for: [done], timeout: timeout)
            guard waitResult == .completed else {
                // Cancel the in-flight task and then wait briefly for it to finish
                // so that work from this iteration does not overlap with the next one.
                task.cancel()

                // Give the cancelled task a short grace period to clean up.
                _ = XCTWaiter.wait(for: [done], timeout: 5)
                XCTFail("Timed out waiting for measured async operation (\(waitResult))")
                return
            }
        }
    }
}
