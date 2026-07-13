//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import AVFoundation
import os.lock
import XCTest
@testable import WhisperKit

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
final class AudioProcessorTests: XCTestCase {

    /// The incremental loader emits VAD windows (each ≤ one model window) and the concatenation
    /// of all chunks reconstructs the original audio with contiguous seek offsets — no samples
    /// lost, duplicated, or reordered.
    func testIncrementalChunksAreVadWindowsAndReconstructAudio() async throws {
        let mockAudioPath = try createMockAudioFile16k(duration: 100.0)
        defer { try? FileManager.default.removeItem(atPath: mockAudioPath) }

        let fullAudio = try AudioProcessor.loadAudioAsFloatArray(fromPath: mockAudioPath)
        let result = try await collectIncrementalChunks(path: mockAudioPath, chunkDurationSeconds: 30.0)
        assertChunksAreVadWindowsAndReconstruct(result, fullAudio: fullAudio)
    }

    /// Streams `path` incrementally and returns each emitted chunk's (seek offset, sample count)
    /// plus the concatenation of all emitted samples.
    private func collectIncrementalChunks(
        path: String,
        chunkDurationSeconds: Double
    ) async throws -> (chunks: [(offset: Int, count: Int)], samples: [Float]) {
        var chunks: [(offset: Int, count: Int)] = []
        var samples: [Float] = []
        let stream = try AudioProcessor.loadFileIncrementally(
            fromPath: path,
            chunkDurationSeconds: chunkDurationSeconds,
            maxBufferedChunks: 2
        )
        for try await chunk in stream {
            chunks.append((chunk.audioChunk.seekOffsetIndex, chunk.audioChunk.audioSamples.count))
            samples.append(contentsOf: chunk.audioChunk.audioSamples)
            chunk.completionSignal()
        }
        return (chunks, samples)
    }

    private func assertChunksAreVadWindowsAndReconstruct(
        _ result: (chunks: [(offset: Int, count: Int)], samples: [Float]),
        fullAudio: [Float],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let maxChunkLength = Constants.defaultWindowSamples
        XCTAssertFalse(result.chunks.isEmpty, "expected at least one chunk", file: file, line: line)

        var expectedOffset = 0
        for (i, chunk) in result.chunks.enumerated() {
            XCTAssertGreaterThan(chunk.count, 0, "chunk \(i) is empty", file: file, line: line)
            XCTAssertLessThanOrEqual(chunk.count, maxChunkLength, "chunk \(i) exceeds one model window", file: file, line: line)
            XCTAssertEqual(chunk.offset, expectedOffset, "chunk \(i) seek offset is not contiguous", file: file, line: line)
            expectedOffset += chunk.count
        }
        XCTAssertEqual(result.samples.count, fullAudio.count, "reconstructed sample count differs from source", file: file, line: line)
        XCTAssertEqual(result.samples, fullAudio, "reconstructed audio differs from source", file: file, line: line)
    }

    /// Same property on audio with explicit silence regions: chunks remain VAD windows and the
    /// concatenation reconstructs the source exactly.
    func testIncrementalChunksReconstructAudioWithSilence() async throws {
        let mockAudioPath = try createMockAudioFile16k(
            duration: 300.0,
            silenceRanges: [(80.0, 90.0), (180.0, 190.0), (280.0, 290.0)]
        )
        defer { try? FileManager.default.removeItem(atPath: mockAudioPath) }

        let fullAudio = try AudioProcessor.loadAudioAsFloatArray(fromPath: mockAudioPath)
        let result = try await collectIncrementalChunks(path: mockAudioPath, chunkDurationSeconds: 120.0)
        assertChunksAreVadWindowsAndReconstruct(result, fullAudio: fullAudio)
    }

    /// The emitted chunk boundaries must be identical regardless of the read/staging size
    /// (`chunkDurationSeconds`): buffering in 7 s, 30 s, or 120 s reads must still yield the same model
    /// chunks. Read size only affects peak memory, never the chunks fed to the model.
    func testIncrementalChunkBoundariesAreIndependentOfReadSize() async throws {
        let mockAudioPath = try createMockAudioFile16k(
            duration: 300.0,
            silenceRanges: [(80.0, 90.0), (180.0, 190.0), (280.0, 290.0)]
        )
        defer { try? FileManager.default.removeItem(atPath: mockAudioPath) }

        let readSizes: [Double] = [7.0, 30.0, 120.0]
        var boundariesPerReadSize: [[Int]] = []
        for readSize in readSizes {
            let result = try await collectIncrementalChunks(path: mockAudioPath, chunkDurationSeconds: readSize)
            // Running boundaries: each chunk's start offset, plus the final end offset.
            let bounds = result.chunks.map { $0.offset } + [result.chunks.last.map { $0.offset + $0.count } ?? 0]
            boundariesPerReadSize.append(bounds)
        }

        let reference = boundariesPerReadSize[0]
        for (i, bounds) in boundariesPerReadSize.enumerated() {
            XCTAssertEqual(bounds, reference, "chunk boundaries differ at read size \(readSizes[i]) s (must be independent of read size)")
        }
    }

    /// Tests incremental loading with single partial chunk.
    /// Verifies that audio shorter than chunk duration produces exactly 1 chunk
    func testIncrementalLoadSinglePartialChunk() async throws {
        // Create a mock audio file of 5 seconds (less than 10 second chunk duration)
        let mockAudioPath = try createMockAudioFile16k(duration: 5.0)
        let chunkDurationSeconds: Double = 10.0
        let partialChunkSize = 5 * WhisperKit.sampleRate

        defer {
            try? FileManager.default.removeItem(atPath: mockAudioPath)
        }

        var processedChunks = 0

        let stream = try AudioProcessor.loadFileIncrementally(
            fromPath: mockAudioPath,
            chunkDurationSeconds: chunkDurationSeconds,
            maxBufferedChunks: 2
        )

        for try await chunk in stream {
            processedChunks += 1
            let actualSize = chunk.audioChunk.audioSamples.count

            // Chunk should be partial-sized
            // For 5 second audio with 10 second chunks, chunk should be 5 seconds
            assertChunkSizeWithinTolerance(
                actualSize: actualSize,
                expectedSize: partialChunkSize
            )
            chunk.completionSignal()
        }

        XCTAssertEqual(processedChunks, 1, "Should process exactly 1 chunk for 5 second audio with 10 second chunks")
    }

    /// Tests incremental loading with zero-length audio.
    /// Verifies that empty audio files are handled gracefully
    func testIncrementalLoadZeroLength() async throws {
        // Create a mock audio file of 0 seconds
        let mockAudioPath = try createMockAudioFile16k(duration: 0.0)
        let chunkDurationSeconds: Double = 10.0

        defer {
            try? FileManager.default.removeItem(atPath: mockAudioPath)
        }

        var processedChunks = 0

        let stream = try AudioProcessor.loadFileIncrementally(
            fromPath: mockAudioPath,
            chunkDurationSeconds: chunkDurationSeconds,
            maxBufferedChunks: 2
        )

        for try await chunk in stream {
            processedChunks += 1
            chunk.completionSignal()
        }

        XCTAssertEqual(processedChunks, 0, "Should process 0 chunks for zero-length audio")
    }

    /// Tests incremental loading with non-existent file.
    /// Verifies that missing files are handled with appropriate error
    func testIncrementalLoadNonExistentFile() async throws {
        let nonExistentPath = "/tmp/non_existent_audio_file.wav"

        let chunkDurationSeconds: Double = 10.0

        XCTAssertThrowsError(
            try AudioProcessor.loadFileIncrementally(
                fromPath: nonExistentPath,
                chunkDurationSeconds: chunkDurationSeconds,
                maxBufferedChunks: 2
            )
        ) { error in
            XCTAssertTrue(error is WhisperError, "Should throw WhisperError when audio file does not exist")
        }
    }

    /// Verifies incremental multi-file transcription respects
    /// `concurrentWorkerCount` while keeping results aligned with the original
    /// path list, including failures that complete before successful files.
    func testIncrementalMultiFileTranscriptionRespectsConcurrentWorkerCountAndOrder() async throws {
        let firstAudioPath = try createMockAudioFile16k(duration: 1.0)
        let secondAudioPath = try createMockAudioFile16k(duration: 2.0)
        let thirdAudioPath = try createMockAudioFile16k(duration: 3.0)
        let missingAudioPath = "/path/to/missing-incremental-audio.wav"
        defer {
            try? FileManager.default.removeItem(atPath: firstAudioPath)
            try? FileManager.default.removeItem(atPath: secondAudioPath)
            try? FileManager.default.removeItem(atPath: thirdAudioPath)
        }

        let tracker = ConcurrentTranscriptionTracker()
        let whisperKit = try await ConcurrencyTrackingWhisperKit(
            tracker: tracker,
            delayNanoseconds: 100_000_000
        )
        let transcriptionResults = await whisperKit.transcribeWithResults(
            audioPaths: [
                firstAudioPath,
                missingAudioPath,
                secondAudioPath,
                thirdAudioPath,
            ],
            audioInputOptions: AudioInputOptions(audioLoadingMode: .incremental(chunkDurationSeconds: 600, maxBufferedChunks: 2)),
            decodeOptions: DecodingOptions(concurrentWorkerCount: 2)
        )

        XCTAssertEqual(transcriptionResults.count, 4)
        let firstSampleCount = try sampleCount(from: transcriptionResults[0])
        XCTAssertEqual(
            transcriptionResults[1].whisperError(),
            .loadAudioFailed("Audio file not found at path: \(missingAudioPath)")
        )
        let secondSampleCount = try sampleCount(from: transcriptionResults[2])
        let thirdSampleCount = try sampleCount(from: transcriptionResults[3])
        XCTAssertLessThan(firstSampleCount, secondSampleCount)
        XCTAssertLessThan(secondSampleCount, thirdSampleCount)

        let concurrencySnapshot = await tracker.snapshot()
        XCTAssertEqual(concurrencySnapshot.maxActiveCount, 2)
        XCTAssertEqual(concurrencySnapshot.totalStartedCount, 3)
    }

    /// Tests incremental loading with invalid chunk configuration
    /// Verifies that values which cannot make progress are rejected before stream creation
    func testIncrementalLoadRejectsInvalidParameters() throws {
        let mockAudioPath = try createMockAudioFile16k(duration: 1.0)
        defer {
            try? FileManager.default.removeItem(atPath: mockAudioPath)
        }

        XCTAssertThrowsError(
            try AudioProcessor.loadFileIncrementally(
                fromPath: mockAudioPath,
                chunkDurationSeconds: 0,
                maxBufferedChunks: 2
            )
        ) { error in
            XCTAssertTrue(error is WhisperError, "Should throw WhisperError when chunk duration is invalid")
        }

        XCTAssertThrowsError(
            try AudioProcessor.loadFileIncrementally(
                fromPath: mockAudioPath,
                chunkDurationSeconds: 10.0,
                maxBufferedChunks: 0
            )
        ) { error in
            XCTAssertTrue(error is WhisperError, "Should throw WhisperError when chunk buffer size is invalid")
        }
    }

    /// On stream termination a producer suspended waiting for a slot must be resumed (not orphaned),
    /// and any subsequent slot request must return immediately instead of suspending forever.
    func testIncrementalLoadControllerTerminationResumesWaitingProducer() async throws {
        let controller = IncrementalLoadController(maxBufferedChunks: 1)
        await controller.requestChunkSlot() // fill the only slot

        // A second request suspends because the buffer is full.
        let waiter = Task { await controller.requestChunkSlot() }
        try? await Task.sleep(nanoseconds: 50_000_000) // let the waiter suspend

        // Termination must resume the suspended producer.
        await controller.resumeWaitingProducers()
        let waiterTimedOut = await withTimeout(seconds: 2) { await waiter.value }
        XCTAssertFalse(waiterTimedOut, "Waiting producer should be resumed on termination")

        // After termination a fresh request returns immediately rather than suspending.
        let nextTimedOut = await withTimeout(seconds: 2) { await controller.requestChunkSlot() }
        XCTAssertFalse(nextTimedOut, "requestChunkSlot should return immediately after termination")
    }

    /// Verifies the VAD windows fed to the model are identical in full-file and incremental modes,
    /// so the loading strategy does not change the chunks provided to inference.
    func testIncrementalVsFullFileModelChunkParity() async throws {
        // 5 minutes of tone with a 0.8 s silence every 12 s so VAD has boundaries to cut on.
        var silences: [(start: Double, end: Double)] = []
        var t = 12.0
        while t < 300.0 { silences.append((t, t + 0.8)); t += 12.0 }
        let path = try createMockAudioFile16k(duration: 300.0, silenceRanges: silences)
        defer { try? FileManager.default.removeItem(atPath: path) }

        let maxLen = Constants.defaultWindowSamples
        let opts = DecodingOptions()

        // FULL-FILE: load the whole file and VAD-chunk once (what transcribe(.fullFile) feeds the model).
        let full = try AudioProcessor.loadAudioAsFloatArray(fromPath: path)
        let fullChunks = try await VADAudioChunker(vad: EnergyVAD())
            .chunkAll(audioArray: full, maxChunkLength: maxLen, decodeOptions: opts)
        let fullWindows = fullChunks.map { ($0.seekOffsetIndex, $0.audioSamples.count) }

        // INCREMENTAL: stream file-chunks, VAD-chunk each independently, map to global offsets.
        var incWindows: [(Int, Int)] = []
        let stream = try AudioProcessor.loadFileIncrementally(fromPath: path, chunkDurationSeconds: 120, maxBufferedChunks: 8)
        for try await chunk in stream {
            defer { chunk.completionSignal() }
            let subs = try await VADAudioChunker(vad: EnergyVAD())
                .chunkAll(audioArray: chunk.audioChunk.audioSamples, maxChunkLength: maxLen, decodeOptions: opts)
            for s in subs {
                incWindows.append((chunk.audioChunk.seekOffsetIndex + s.seekOffsetIndex, s.audioSamples.count))
            }
        }

        // The model windows must be identical between full-file and incremental modes: the same
        // audio chunks in, the same results out, with peak memory the only intended difference.
        XCTAssertEqual(incWindows.map { $0.0 }, fullWindows.map { $0.0 }, "window start offsets must be identical (full vs incremental)")
        XCTAssertEqual(incWindows.map { $0.1 }, fullWindows.map { $0.1 }, "window lengths must be identical (full vs incremental)")
    }

    /// Tests incremental loading with a file that exists but cannot be decoded as audio
    /// Verifies that async file initialization errors are surfaced through the throwing stream
    func testIncrementalLoadUnreadableFileThrows() async throws {
        let audioURL = FileManager.default.temporaryDirectory.appendingPathComponent("invalid_audio_\(UUID().uuidString).wav")
        try "not an audio file".write(to: audioURL, atomically: true, encoding: .utf8)
        defer {
            try? FileManager.default.removeItem(at: audioURL)
        }

        let stream = try AudioProcessor.loadFileIncrementally(
            fromPath: audioURL.path,
            chunkDurationSeconds: 10.0,
            maxBufferedChunks: 2
        )

        do {
            for try await chunk in stream {
                chunk.completionSignal()
            }
            XCTFail("Unreadable audio should throw from the incremental stream")
        } catch {
            XCTAssertFalse(error is CancellationError, "Unreadable audio should fail with an audio loading error")
        }
    }

    /// Tests back-pressure mechanism by not calling completionSignal
    /// and verifying the stream gets stuck after receiving maxBufferSize chunks
    func testIncrementalLoadBackPressureBlocking() async throws {
        // 100 s produces several VAD windows (> maxBufferSize), so back-pressure can engage.
        let mockAudioPath = try createMockAudioFile16k(duration: 100.0)
        defer {
            try? FileManager.default.removeItem(atPath: mockAudioPath)
        }

        let maxBufferSize = 2
        let receivedChunks = OSAllocatedUnfairLock(initialState: 0)

        let stream = try AudioProcessor.loadFileIncrementally(
            fromPath: mockAudioPath,
            chunkDurationSeconds: 10.0,
            maxBufferedChunks: maxBufferSize
        )

        // Use withTimeout to test blocking behavior
        let didTimeout = await withTimeout(seconds: 1) {
            for try await _ in stream {
                receivedChunks.withLock { count in
                    count += 1
                }
                // Deliberately NOT calling completionSignal() to test blocking
            }
        }

        // Verify that we received exactly maxBufferSize chunks before blocking
        XCTAssertEqual(receivedChunks.withLock { $0 }, maxBufferSize, "Should receive exactly \(maxBufferSize) chunks before blocking due to back-pressure")

        // Verify that the operation timed out (indicating incremental loading was blocked)
        XCTAssertTrue(didTimeout, "Incremental loading should block after \(maxBufferSize) chunks, causing timeout")
    }

    /// Validates that a chunk size is within the expected range with 10% tolerance
    /// - Parameters:
    ///   - actualSize: The actual chunk size in samples
    ///   - expectedSize: The theoretical expected size in samples
    private func assertChunkSizeWithinTolerance(
        actualSize: Int,
        expectedSize: Int
    ) {
        let tolerancePercent = 0.1
        let tolerance = Double(expectedSize) * tolerancePercent
        let lowerBound = expectedSize - Int(tolerance)
        let upperBound = expectedSize + Int(tolerance)

        XCTAssertTrue(
            actualSize >= lowerBound && actualSize <= upperBound,
            "Chunk size \(actualSize) is outside expected range [\(lowerBound)-\(upperBound)] (theoretical: \(expectedSize))"
        )
    }

    /// Creates a mock audio file at 16kHz sample rate.
    /// - Parameters:
    ///   - duration: Duration in seconds.
    ///   - silenceRanges: Optional start/end second ranges to write as silence. Samples outside these ranges use a sine wave so EnergyVAD can distinguish speech from boundary silence.
    /// - Returns: File path to the created audio file
    private func createMockAudioFile16k(
        duration: Double,
        silenceRanges: [(start: Double, end: Double)] = []
    ) throws -> String {
        let documentsPath = FileManager.default.temporaryDirectory
        let audioURL = documentsPath.appendingPathComponent("test_audio_16k_\(UUID().uuidString).wav")

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: Double(WhisperKit.sampleRate),
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: true
        ]

        let audioFile = try AVAudioFile(forWriting: audioURL, settings: settings)

        let frameCount = AVAudioFrameCount(duration * Double(WhisperKit.sampleRate))
        let buffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: frameCount)!
        buffer.frameLength = frameCount

        // Fill with simple sine wave, leaving configured ranges silent.
        let samples = buffer.floatChannelData![0]
        for i in 0..<Int(frameCount) {
            let seconds = Double(i) / Double(WhisperKit.sampleRate)
            let isSilent = silenceRanges.contains { range in
                seconds >= range.start && seconds < range.end
            }
            samples[i] = isSilent ? 0.0 : sin(2.0 * .pi * 440.0 * Float(i) / Float(WhisperKit.sampleRate)) * 0.5
        }

        try audioFile.write(from: buffer)
        return audioURL.path
    }

    /// Extracts the deterministic sample-count payload emitted by
    /// `ConcurrencyTrackingWhisperKit` from a transcription result.
    private func sampleCount(from result: Result<[TranscriptionResult], Swift.Error>) throws -> Int {
        let text = try XCTUnwrap(try result.get().first?.text)
        let prefix = "samples-"
        XCTAssertTrue(text.hasPrefix(prefix))
        return try XCTUnwrap(Int(text.dropFirst(prefix.count)))
    }
}

/// Tracks overlapping calls to the mocked transcription entry point so tests can
/// assert that file-level batching limits active work.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
private actor ConcurrentTranscriptionTracker {
    private var activeCount = 0
    private var maxActiveCount = 0
    private var totalStartedCount = 0

    /// Records that a transcription unit started and updates the observed peak
    /// active count.
    func recordStart() {
        activeCount += 1
        totalStartedCount += 1
        maxActiveCount = max(maxActiveCount, activeCount)
    }

    /// Records that a transcription unit completed and removes it from the
    /// active count.
    func recordFinish() {
        activeCount -= 1
    }

    /// Returns immutable counters for assertions after the transcription run
    /// has completed.
    func snapshot() -> (maxActiveCount: Int, totalStartedCount: Int) {
        (maxActiveCount, totalStartedCount)
    }
}

/// WhisperKit test double that bypasses model inference while preserving the
/// incremental file-loading flow used by `transcribeWithResults(audioPaths:)`.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
private final class ConcurrencyTrackingWhisperKit: WhisperKit {
    private let tracker: ConcurrentTranscriptionTracker
    private let delayNanoseconds: UInt64

    /// Creates a model-free WhisperKit instance that records concurrent
    /// `transcribe(audioArray:)` calls and delays each call long enough for
    /// overlapping task-group work to become observable.
    init(
        tracker: ConcurrentTranscriptionTracker,
        delayNanoseconds: UInt64
    ) async throws {
        self.tracker = tracker
        self.delayNanoseconds = delayNanoseconds
        try await super.init(WhisperKitConfig(verbose: false, logLevel: .error, load: false, download: false))
    }

    /// Records concurrency and returns a deterministic result derived from the
    /// loaded audio sample count instead of running Core ML inference.
    override func transcribe(
        audioArray: [Float],
        audioArrayOffset: Int = 0,
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback? = nil,
        segmentCallback: SegmentDiscoveryCallback? = nil
    ) async throws -> [TranscriptionResult] {
        await tracker.recordStart()
        do {
            try await Task.sleep(nanoseconds: delayNanoseconds)
            await tracker.recordFinish()
            let text = "samples-\(audioArray.count)"
            return [
                TranscriptionResult(
                    text: text,
                    segments: [
                        TranscriptionSegment(
                            start: 0,
                            end: Float(audioArray.count) / Float(WhisperKit.sampleRate),
                            text: text
                        )
                    ],
                    language: "en",
                    timings: TranscriptionTimings(),
                    // Relative (chunk-local); transcribeFileStream adds the chunk's absolute offset.
                    seekTime: 0
                )
            ]
        } catch {
            await tracker.recordFinish()
            throw error
        }
    }
}
