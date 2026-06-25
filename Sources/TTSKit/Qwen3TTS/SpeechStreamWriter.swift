//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
import Foundation

// MARK: - SpeechStreamWriter

/// Buffers decoded RVQ frames and streams them out as audio, owning the
/// overlapped-decode / drain / partial-flush logic and callback emission.
///
/// Single-use and not thread-safe: create one per generation and drive it serially
/// from one task.
final class SpeechStreamWriter<Decoder: SpeechDecoding & Sendable> {
    private let speechDecoder: Decoder
    private let sdCache: SpeechDecoderCache
    private let callback: SpeechCallback
    private let pipelineStart: CFAbsoluteTime
    private let baseTimings: SpeechTimings
    private let padFrame: [Int32]
    /// RVQ frames consumed per decode call; read from `speechDecoder`.
    private let codesPerStep: Int
    /// PCM samples produced per RVQ frame; read from `speechDecoder`.
    private let samplesPerFrame: Int

    /// Audio accumulated across every emitted buffer, in emission order.
    private(set) var collectedAudio: [Float] = []

    /// Whether the first buffer has been emitted. Drives the synchronous-first-buffer
    /// path (for minimum TTFB) and the time-to-first-buffer timing.
    private(set) var hasEmittedFirstBuffer = false

    /// RVQ frames accumulated since the last flush (length `0..<codesPerStep`).
    private var rvqBuffer: [[Int32]] = []

    /// In-flight overlapped SpeechDecoder decode awaiting its drain on the next flush.
    private var pendingDecode: (
        task: Task<SpeechDecoderTimedResult, Error>,
        padCount: Int,
        isFirstBuffer: Bool,
        stepStart: CFAbsoluteTime
    )?

    init(
        speechDecoder: Decoder,
        sdCache: SpeechDecoderCache,
        callback: SpeechCallback,
        pipelineStart: CFAbsoluteTime,
        baseTimings: SpeechTimings,
        padFrame: [Int32]
    ) {
        self.speechDecoder = speechDecoder
        self.sdCache = sdCache
        self.callback = callback
        self.pipelineStart = pipelineStart
        self.baseTimings = baseTimings
        self.padFrame = padFrame
        // Cache geometry comes straight from the decoder — no need to thread it in.
        self.codesPerStep = speechDecoder.codesPerStep
        self.samplesPerFrame = speechDecoder.samplesPerFrame
        rvqBuffer.reserveCapacity(speechDecoder.codesPerStep)
    }

    // MARK: - Driving the stream

    /// Append one decoded RVQ frame. Once `codesPerStep` frames have accumulated,
    /// drains the previous in-flight decode and submits a new one.
    ///
    /// - Returns: `false` when a callback (the drained buffer's, or the first
    ///   buffer's) asked generation to stop; `true` to continue.
    func append(
        _ frame: [Int32],
        stepStart: CFAbsoluteTime,
        loopTimings: inout SpeechTimings
    ) async throws -> Bool {
        rvqBuffer.append(frame)
        guard rvqBuffer.count == codesPerStep else { return true }
        // Drain the previous in-flight SD (if any) before kicking off a new one —
        // its execution overlapped with this step's CD/MCD work.
        guard try await drainPendingDecode(loopTimings: &loopTimings) else { return false }
        return try await submitDecode(stepStart: stepStart, loopTimings: &loopTimings)
    }

    /// Drain any in-flight decode, then flush the remaining partial buffer (padded to
    /// `codesPerStep`, trimming the pad-derived trailing samples).
    ///
    /// - Returns: `false` when the drain's callback asked to stop. The final partial
    ///   flush's callback result is intentionally ignored — there is nothing left to
    ///   stop — matching the original loop behavior.
    @discardableResult
    func finish(loopTimings: inout SpeechTimings) async throws -> Bool {
        guard try await drainPendingDecode(loopTimings: &loopTimings) else { return false }
        guard !rvqBuffer.isEmpty else { return true }

        let padCount = max(0, codesPerStep - rvqBuffer.count)
        var toSubmit = rvqBuffer
        for _ in 0..<padCount { toSubmit.append(padFrame) }
        let result = try await speechDecoder.decodeFrameAsync(codes: toSubmit, cache: sdCache)
        _ = emitDecodedBuffer(
            samples: result.samples,
            padCount: padCount,
            predictionTime: result.timings.speechDecoderPredictions,
            isFirstBuffer: !hasEmittedFirstBuffer,
            stepStart: CFAbsoluteTimeGetCurrent(),
            loopTimings: &loopTimings
        )
        rvqBuffer.removeAll(keepingCapacity: true)
        return true
    }

    /// Cancel any in-flight decode so it cannot outlive the loop. Called on every
    /// exit path (the overlapped `Task` does not inherit the loop's cancellation).
    func cancelPendingDecode() {
        pendingDecode?.task.cancel()
        pendingDecode = nil
    }

    // MARK: - Decode submission / drain

    /// Await the in-flight decode (if any) and emit its buffer.
    /// Required before submitting a new decode so `sdCache` mutations stay ordered.
    private func drainPendingDecode(loopTimings: inout SpeechTimings) async throws -> Bool {
        guard let pending = pendingDecode else { return true }
        pendingDecode = nil
        let result = try await pending.task.value
        return emitDecodedBuffer(
            samples: result.samples,
            padCount: pending.padCount,
            predictionTime: result.timings.speechDecoderPredictions,
            isFirstBuffer: pending.isFirstBuffer,
            stepStart: pending.stepStart,
            loopTimings: &loopTimings
        )
    }

    /// Snapshot the current buffer and either decode synchronously (first buffer, for
    /// minimum TTFB) or kick off an overlapped `Task` drained on the next flush.
    /// Callers must drain any previous `pendingDecode` first.
    ///
    /// - Returns: `false` when the first-buffer callback asked to stop; otherwise `true`.
    private func submitDecode(stepStart: CFAbsoluteTime, loopTimings: inout SpeechTimings) async throws -> Bool {
        let snapshot = rvqBuffer
        rvqBuffer.removeAll(keepingCapacity: true)

        if !hasEmittedFirstBuffer {
            let result = try await speechDecoder.decodeFrameAsync(codes: snapshot, cache: sdCache)
            return emitDecodedBuffer(
                samples: result.samples,
                padCount: 0,
                predictionTime: result.timings.speechDecoderPredictions,
                isFirstBuffer: true,
                stepStart: stepStart,
                loopTimings: &loopTimings
            )
        } else {
            let decoder = speechDecoder
            let cache = sdCache
            pendingDecode = (
                task: Task { try await decoder.decodeFrameAsync(codes: snapshot, cache: cache) },
                padCount: 0,
                isFirstBuffer: false,
                stepStart: stepStart
            )
            return true
        }
    }

    /// Trim trailing pad-derived audio samples, append to ``collectedAudio``, fold the
    /// decode's timing into `loopTimings`, build a `SpeechProgress`, emit the callback,
    /// and bookkeep first-buffer / TTFB state.
    ///
    /// - Returns: `false` when the callback asked generation to stop; `true` otherwise.
    private func emitDecodedBuffer(
        samples sourceSamples: [Float],
        padCount: Int,
        predictionTime: TimeInterval,
        isFirstBuffer: Bool,
        stepStart: CFAbsoluteTime,
        loopTimings: inout SpeechTimings
    ) -> Bool {
        loopTimings.speechDecoderPredictions += predictionTime
        loopTimings.speechDecoder += predictionTime
        loopTimings.totalSpeechDecoderInvocations += 1

        var samples = sourceSamples
        if padCount > 0 {
            let padSamples = padCount * samplesPerFrame
            if samples.count > padSamples {
                samples = Array(samples.prefix(samples.count - padSamples))
            } else {
                samples.removeAll()
            }
        }
        collectedAudio.append(contentsOf: samples)

        let now = CFAbsoluteTimeGetCurrent()
        if isFirstBuffer {
            loopTimings.timeToFirstBuffer = now - pipelineStart
            hasEmittedFirstBuffer = true
        }

        let progress: SpeechProgress
        if isFirstBuffer {
            progress = SpeechProgress(audio: samples, timings: loopTimings, stepTime: now - stepStart)
        } else {
            var merged = baseTimings
            merged.merge(loopTimings)
            progress = SpeechProgress(audio: samples, timings: merged, stepTime: nil)
        }
        return callback?(progress) != false
    }
}
