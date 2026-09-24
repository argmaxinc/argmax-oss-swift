//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import AVFoundation
import os.lock
@testable import WhisperKit
import XCTest

final class AudioStreamTranscriberTests: XCTestCase {
    func testTranscriptionErrorStopsRecording() async throws {
        let audio = RecordingAudioProcessor(sampleCount: WhisperKit.sampleRate * 2)
        let recordingChanges = OSAllocatedUnfairLock(initialState: [Bool]())
        let submittedSamples = OSAllocatedUnfairLock(initialState: 0)
        let transcriber = makeTranscriber(audio: audio) { oldState, newState in
            if oldState.isRecording != newState.isRecording {
                recordingChanges.withLock { [isRecording = newState.isRecording] in $0.append(isRecording) }
            }
            submittedSamples.withLock { [lastBufferSize = newState.lastBufferSize] in $0 = lastBufferSize }
        }

        // The unloaded decoder fails on the first transcription. No model download
        // or microphone permission is needed to exercise the error path.
        try await transcriber.startRecordingAndTranscribing()

        XCTAssertEqual(submittedSamples.withLock { $0 }, WhisperKit.sampleRate * 2)
        XCTAssertFalse(audio.isRecording)
        XCTAssertGreaterThan(audio.stopCount, 0)
        XCTAssertEqual(recordingChanges.withLock { $0 }, [true, false])
    }

    func testCancellationStopsRecordingWhileWaitingForAudio() async throws {
        let started = expectation(description: "Recording started")
        let audio = RecordingAudioProcessor(onStart: { started.fulfill() })
        let recordingChanges = OSAllocatedUnfairLock(initialState: [Bool]())
        let transcriber = makeTranscriber(audio: audio) { oldState, newState in
            if oldState.isRecording != newState.isRecording {
                recordingChanges.withLock { [isRecording = newState.isRecording] in $0.append(isRecording) }
            }
        }
        let task = Task { try await transcriber.startRecordingAndTranscribing() }
        await fulfillment(of: [started], timeout: 2)

        task.cancel()
        try await task.value

        XCTAssertFalse(audio.isRecording)
        XCTAssertGreaterThan(audio.stopCount, 0)
        XCTAssertEqual(recordingChanges.withLock { $0 }, [true, false])
    }

    func testExplicitStopEndsRecording() async throws {
        let started = expectation(description: "Recording started")
        let audio = RecordingAudioProcessor(onStart: { started.fulfill() })
        let recordingChanges = OSAllocatedUnfairLock(initialState: [Bool]())
        let transcriber = makeTranscriber(audio: audio) { oldState, newState in
            if oldState.isRecording != newState.isRecording {
                recordingChanges.withLock { [isRecording = newState.isRecording] in $0.append(isRecording) }
            }
        }
        let task = Task { try await transcriber.startRecordingAndTranscribing() }
        await fulfillment(of: [started], timeout: 2)

        await transcriber.stopStreamTranscription()
        try await task.value

        XCTAssertFalse(audio.isRecording)
        XCTAssertGreaterThan(audio.stopCount, 0)
        XCTAssertEqual(recordingChanges.withLock { $0 }, [true, false])
    }

    private func makeTranscriber(
        audio: RecordingAudioProcessor,
        callback: @escaping AudioStreamTranscriberCallback
    ) -> AudioStreamTranscriber {
        AudioStreamTranscriber(
            audioEncoder: AudioEncoder(),
            featureExtractor: FeatureExtractor(),
            segmentSeeker: SegmentSeeker(),
            textDecoder: TextDecoder(),
            tokenizer: CustomTokenizer(specialTokenBegin: 1000),
            audioProcessor: audio,
            decodingOptions: DecodingOptions(),
            useVAD: false,
            stateChangeCallback: callback
        )
    }
}

// Simulates capture without opening an audio device. Mutable state is protected
// by a lock because the test and the transcriber access it from different tasks.
private final class RecordingAudioProcessor: AudioProcessing, @unchecked Sendable {
    private struct State {
        var isRecording = false
        var stopCount = 0
        var relativeEnergyWindow = 20
    }

    private let state = OSAllocatedUnfairLock(initialState: State())
    private let onStart: @Sendable () -> Void
    let audioSamples: ContiguousArray<Float>
    let relativeEnergy: [Float] = []

    init(sampleCount: Int = 0, onStart: @escaping @Sendable () -> Void = {}) {
        self.audioSamples = ContiguousArray(repeating: 0.1, count: sampleCount)
        self.onStart = onStart
    }

    var isRecording: Bool { state.withLock { $0.isRecording } }
    var stopCount: Int { state.withLock { $0.stopCount } }
    var relativeEnergyWindow: Int {
        get { state.withLock { $0.relativeEnergyWindow } }
        set { state.withLock { $0.relativeEnergyWindow = newValue } }
    }

    func startRecordingLive(inputDeviceID: DeviceID?, callback: (([Float]) -> Void)?) throws {
        state.withLock { $0.isRecording = true }
        onStart()
    }

    func stopRecording() {
        state.withLock {
            $0.isRecording = false
            $0.stopCount += 1
        }
    }

    func purgeAudioSamples(keepingLast keep: Int) {}
    func pauseRecording() {}
    func resumeRecordingLive(inputDeviceID: DeviceID?, callback: (([Float]) -> Void)?) throws {}

    func startStreamingRecordingLive(inputDeviceID: DeviceID?) -> (AsyncThrowingStream<[Float], Error>, AsyncThrowingStream<[Float], Error>.Continuation) {
        AsyncThrowingStream.makeStream()
    }

    func padOrTrim(fromArray audioArray: [Float], startAt startIndex: Int, toLength frameLength: Int) -> (any AudioProcessorOutputType)? {
        AudioProcessor.padOrTrimAudio(fromArray: audioArray, startAt: startIndex, toLength: frameLength, saveSegment: false)
    }

    static func loadAudio(fromPath path: String, channelMode: ChannelMode, startTime: Double?, endTime: Double?, maxReadFrameSize: AVAudioFrameCount?) throws -> AVAudioPCMBuffer {
        try AudioProcessor.loadAudio(fromPath: path, channelMode: channelMode, startTime: startTime, endTime: endTime, maxReadFrameSize: maxReadFrameSize)
    }

    static func loadAudio(at paths: [String], channelMode: ChannelMode) async -> [Result<[Float], Error>] {
        await AudioProcessor.loadAudio(at: paths, channelMode: channelMode)
    }
}
