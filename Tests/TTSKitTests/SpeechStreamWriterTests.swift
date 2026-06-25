//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
@testable import TTSKit
import XCTest

/// Unit tests for `SpeechStreamWriter` — the RVQ-frame buffering / overlapped-decode /
/// drain / partial-flush logic extracted from `Qwen3GenerateTask.runGenerationLoop`.
/// Driven through a `MockSpeechDecoder` so no CoreML asset is required.
final class SpeechStreamWriterTests: XCTestCase {

    // MARK: - Test doubles

    /// Deterministic `SpeechDecoding` stub. Each call returns `samplesPerFrame`
    /// samples per input frame, valued by that frame's `code0`, so emitted audio is
    /// traceable back to specific frames (and pad frames are identifiable).
    private final class MockSpeechDecoder: SpeechDecoding, @unchecked Sendable {
        let sampleRate = 24_000
        let samplesPerFrame: Int
        let minimumBufferDuration: TimeInterval = 0.08
        let kvCacheEmbedDim = Qwen3TTSConstants.sdCacheDim
        let kvCacheMaxSequenceLength = Qwen3TTSConstants.sdMaxSeq
        let hiddenDim = Qwen3TTSConstants.sdHiddenDim
        let hiddenContextLen = Qwen3TTSConstants.sdHiddenContextLen
        let codesPerStep: Int
        var model: MLModel? { nil }

        let predictionTimePerCall: TimeInterval

        private let lock = NSLock()
        private var _recordedCalls: [[[Int32]]] = []
        /// The `codes` argument of every `decodeFrameAsync` call, in completion order.
        var recordedCalls: [[[Int32]]] { lock.withLock { _recordedCalls } }
        var callCount: Int { recordedCalls.count }

        init(samplesPerFrame: Int, codesPerStep: Int, predictionTimePerCall: TimeInterval = 0.01) {
            self.samplesPerFrame = samplesPerFrame
            self.codesPerStep = codesPerStep
            self.predictionTimePerCall = predictionTimePerCall
        }

        func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool) async throws {}
        func unloadModel() {}

        func decodeFrame(codes: [[Int32]], cache: SpeechDecoderCache) async throws -> [Float] {
            try await decodeFrameAsync(codes: codes, cache: cache).samples
        }

        func decodeFrameAsync(codes: [[Int32]], cache: SpeechDecoderCache) async throws -> SpeechDecoderTimedResult {
            lock.withLock { _recordedCalls.append(codes) }

            var samples: [Float] = []
            samples.reserveCapacity(codes.count * samplesPerFrame)
            for frame in codes {
                let value = Float(frame.first ?? -1)
                samples.append(contentsOf: repeatElement(value, count: samplesPerFrame))
            }
            var timings = SpeechTimings()
            timings.speechDecoderPredictions = predictionTimePerCall
            return SpeechDecoderTimedResult(samples: samples, timings: timings)
        }
    }

    /// Records every emitted `SpeechProgress`, optionally signalling "stop" after a
    /// given number of callbacks (to exercise callback-driven cancellation).
    private final class ProgressRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var _progresses: [SpeechProgress] = []
        private let stopAfter: Int?

        init(stopAfter: Int? = nil) { self.stopAfter = stopAfter }

        var progresses: [SpeechProgress] {
            lock.lock(); defer { lock.unlock() }
            return _progresses
        }

        func record(_ progress: SpeechProgress) -> Bool? {
            lock.lock(); defer { lock.unlock() }
            _progresses.append(progress)
            if let stopAfter, _progresses.count >= stopAfter { return false }
            return nil
        }
    }

    // MARK: - Helpers

    /// A 16-code RVQ frame whose first code (code0) is `code0`, used as the frame's
    /// signature in `MockSpeechDecoder`'s output.
    private func frame(_ code0: Int32) -> [Int32] {
        [code0] + [Int32](repeating: 0, count: 15)
    }

    private func makeWriter(
        decoder: MockSpeechDecoder,
        recorder: ProgressRecorder,
        baseTimings: SpeechTimings = SpeechTimings()
    ) throws -> SpeechStreamWriter<MockSpeechDecoder> {
        let cache = try SpeechDecoderCache(codesPerStep: decoder.codesPerStep)
        return SpeechStreamWriter(
            speechDecoder: decoder,
            sdCache: cache,
            callback: { progress in recorder.record(progress) },
            pipelineStart: CFAbsoluteTimeGetCurrent(),
            baseTimings: baseTimings,
            padFrame: [Int32](repeating: Qwen3TTSConstants.codecPAD, count: 16)
        )
    }

    /// Drive `writer` with `frames`, mimicking `runGenerationLoop`: append each frame,
    /// stopping early if a callback asks to; then `finish()` if not stopped.
    /// Returns whether the run completed without a callback-requested stop.
    @discardableResult
    private func drive(
        _ writer: SpeechStreamWriter<MockSpeechDecoder>,
        frames: [[Int32]],
        timings: inout SpeechTimings
    ) async throws -> Bool {
        for frame in frames {
            let stepStart = CFAbsoluteTimeGetCurrent()
            if try await writer.append(frame, stepStart: stepStart, loopTimings: &timings) == false {
                return false
            }
        }
        try await writer.finish(loopTimings: &timings)
        return true
    }

    // MARK: - Latency mode (codesPerStep == 1)

    func testLatencyModeEmitsOneBufferPerFrame() async throws {
        let spf = 2
        let decoder = MockSpeechDecoder(samplesPerFrame: spf, codesPerStep: 1)
        let recorder = ProgressRecorder()
        let writer = try makeWriter(decoder: decoder, recorder: recorder)

        var timings = SpeechTimings()
        let frames = (1...5).map { frame(Int32($0)) }
        let completed = try await drive(writer, frames: frames, timings: &timings)

        XCTAssertTrue(completed)
        XCTAssertEqual(decoder.callCount, 5, "One decode call per frame in latency mode")
        XCTAssertEqual(recorder.progresses.count, 5, "One callback per buffer")
        // Audio is each frame's code0 repeated `spf` times, in order, no padding.
        XCTAssertEqual(writer.collectedAudio, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    }

    // MARK: - Throughput mode (codesPerStep > 1)

    func testThroughputModeFullBuffers() async throws {
        let spf = 2
        let decoder = MockSpeechDecoder(samplesPerFrame: spf, codesPerStep: 4)
        let recorder = ProgressRecorder()
        let writer = try makeWriter(decoder: decoder, recorder: recorder)

        var timings = SpeechTimings()
        let frames = (1...8).map { frame(Int32($0)) }
        try await drive(writer, frames: frames, timings: &timings)

        XCTAssertEqual(decoder.callCount, 2, "8 frames / 4 per step = 2 decode calls")
        XCTAssertEqual(recorder.progresses.count, 2)
        XCTAssertEqual(writer.collectedAudio, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8])
    }

    func testThroughputModePartialBufferPadsAndTrims() async throws {
        let spf = 2
        let decoder = MockSpeechDecoder(samplesPerFrame: spf, codesPerStep: 4)
        let recorder = ProgressRecorder()
        let writer = try makeWriter(decoder: decoder, recorder: recorder)

        var timings = SpeechTimings()
        // 6 frames: one full buffer (f1..f4) + a partial buffer (f5, f6) padded to 4.
        let frames = (1...6).map { frame(Int32($0)) }
        try await drive(writer, frames: frames, timings: &timings)

        XCTAssertEqual(decoder.callCount, 2, "One full flush + one partial flush")
        XCTAssertEqual(recorder.progresses.count, 2)
        // The 2 pad frames' worth of trailing samples (2 frames × spf = 4 samples) are
        // trimmed, leaving exactly the 6 real frames' audio.
        XCTAssertEqual(writer.collectedAudio, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        XCTAssertEqual(writer.collectedAudio.count, 6 * spf)
        // No pad-derived samples leaked into the output.
        XCTAssertFalse(writer.collectedAudio.contains(Float(Qwen3TTSConstants.codecPAD)))
    }

    func testPartialOnlyBufferIsPaddedAndTrimmed() async throws {
        let spf = 3
        let decoder = MockSpeechDecoder(samplesPerFrame: spf, codesPerStep: 4)
        let recorder = ProgressRecorder()
        let writer = try makeWriter(decoder: decoder, recorder: recorder)

        var timings = SpeechTimings()
        // Fewer than codesPerStep frames: only the final partial flush runs.
        let frames = [frame(7), frame(9)]
        try await drive(writer, frames: frames, timings: &timings)

        XCTAssertEqual(decoder.callCount, 1)
        XCTAssertEqual(recorder.progresses.count, 1)
        XCTAssertEqual(writer.collectedAudio, [7, 7, 7, 9, 9, 9])
    }

    // MARK: - Ordering, timings, first-buffer semantics

    func testDecodeCallOrderMatchesFrameOrder() async throws {
        let decoder = MockSpeechDecoder(samplesPerFrame: 1, codesPerStep: 1)
        let recorder = ProgressRecorder()
        let writer = try makeWriter(decoder: decoder, recorder: recorder)

        var timings = SpeechTimings()
        let frames = (1...4).map { frame(Int32($0)) }
        try await drive(writer, frames: frames, timings: &timings)

        // Even though decodes after the first run on overlapped tasks, each is drained
        // before the next is submitted, so completion order matches frame order.
        let firstCodes = decoder.recordedCalls.map { $0.first?.first }
        XCTAssertEqual(firstCodes, [1, 2, 3, 4])
    }

    func testFirstBufferCarriesStepTimeAndTTFB() async throws {
        let decoder = MockSpeechDecoder(samplesPerFrame: 1, codesPerStep: 1)
        let recorder = ProgressRecorder()
        let writer = try makeWriter(decoder: decoder, recorder: recorder)

        var timings = SpeechTimings()
        try await drive(writer, frames: (1...3).map { frame(Int32($0)) }, timings: &timings)

        let progresses = recorder.progresses
        XCTAssertEqual(progresses.count, 3)
        XCTAssertNotNil(progresses.first?.stepTime, "First buffer carries stepTime for adaptive buffering")
        XCTAssertTrue(progresses.dropFirst().allSatisfy { $0.stepTime == nil },
                      "Only the first buffer carries stepTime")
        XCTAssertGreaterThan(timings.timeToFirstBuffer, 0, "TTFB recorded on first buffer")
        XCTAssertTrue(writer.hasEmittedFirstBuffer)
    }

    func testTimingsAccumulatedAcrossDecodes() async throws {
        let predictionTime: TimeInterval = 0.01
        let decoder = MockSpeechDecoder(samplesPerFrame: 1, codesPerStep: 1, predictionTimePerCall: predictionTime)
        let recorder = ProgressRecorder()
        let writer = try makeWriter(decoder: decoder, recorder: recorder)

        var timings = SpeechTimings()
        try await drive(writer, frames: (1...5).map { frame(Int32($0)) }, timings: &timings)

        XCTAssertEqual(timings.totalSpeechDecoderInvocations, 5, accuracy: 0.0001)
        XCTAssertEqual(timings.speechDecoderPredictions, 5 * predictionTime, accuracy: 1e-9)
        XCTAssertEqual(timings.speechDecoder, 5 * predictionTime, accuracy: 1e-9)
    }

    // MARK: - Callback-driven cancellation

    func testCallbackStopHaltsStreaming() async throws {
        let spf = 2
        let decoder = MockSpeechDecoder(samplesPerFrame: spf, codesPerStep: 1)
        let recorder = ProgressRecorder(stopAfter: 2) // stop after the 2nd buffer
        let writer = try makeWriter(decoder: decoder, recorder: recorder)

        var timings = SpeechTimings()
        let frames = (1...6).map { frame(Int32($0)) }
        let completed = try await drive(writer, frames: frames, timings: &timings)

        XCTAssertFalse(completed, "drive() should report an early callback stop")
        XCTAssertEqual(recorder.progresses.count, 2, "No further callbacks after stop")
        // Buffer 1 (sync) emits during frame 1; buffer 2 emits while draining during a
        // later frame and returns stop, so only those two buffers' audio is collected.
        XCTAssertEqual(writer.collectedAudio, [1, 1, 2, 2])
    }

    // MARK: - Edge cases

    func testNoFramesProducesNoAudioOrCalls() async throws {
        let decoder = MockSpeechDecoder(samplesPerFrame: 2, codesPerStep: 4)
        let recorder = ProgressRecorder()
        let writer = try makeWriter(decoder: decoder, recorder: recorder)

        var timings = SpeechTimings()
        try await drive(writer, frames: [], timings: &timings)

        XCTAssertEqual(decoder.callCount, 0)
        XCTAssertTrue(recorder.progresses.isEmpty)
        XCTAssertTrue(writer.collectedAudio.isEmpty)
        XCTAssertFalse(writer.hasEmittedFirstBuffer)
    }

    func testCancelPendingDecodeIsSafe() async throws {
        let decoder = MockSpeechDecoder(samplesPerFrame: 1, codesPerStep: 1)
        let recorder = ProgressRecorder()
        let writer = try makeWriter(decoder: decoder, recorder: recorder)

        var timings = SpeechTimings()
        // Submit a few frames (leaves an overlapped decode pending), then cancel without
        // finishing — must not crash or emit further callbacks.
        _ = try await writer.append(frame(1), stepStart: CFAbsoluteTimeGetCurrent(), loopTimings: &timings)
        _ = try await writer.append(frame(2), stepStart: CFAbsoluteTimeGetCurrent(), loopTimings: &timings)
        writer.cancelPendingDecode()
    }
}
