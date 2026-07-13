//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import AVFoundation

/// Actor that manages flow control to prevent memory overuse during incremental file loading
actor IncrementalLoadController {
    private let maxBufferedChunks: Int
    private var activeChunks = 0
    private var waitingProducers: [CheckedContinuation<Void, Never>] = []
    /// Set once the stream terminates so a producer that requests a slot afterwards returns
    /// immediately instead of suspending on a continuation that would never be resumed.
    private var isTerminated = false

    init(maxBufferedChunks: Int = 2) {
        self.maxBufferedChunks = maxBufferedChunks
    }

    /// Request permission to produce a new chunk. Will suspend if buffer is full.
    func requestChunkSlot() async {
        guard !isTerminated else { return }
        if activeChunks >= maxBufferedChunks {
            await withCheckedContinuation { continuation in
                waitingProducers.append(continuation)
            }
            // Resumed by termination rather than a freed slot; don't claim a slot.
            guard !isTerminated else { return }
        }
        activeChunks += 1
    }

    /// Signal that a chunk has been consumed, freeing up a slot
    func signalChunkConsumed() {
        guard activeChunks > 0 else { return }
        activeChunks -= 1
        if !waitingProducers.isEmpty && activeChunks < maxBufferedChunks {
            let producer = waitingProducers.removeFirst()
            producer.resume()
        }
    }

    /// Marks the controller terminated and resumes any waiting producers so task cancellation
    /// can unwind promptly. After this, `requestChunkSlot()` returns without suspending.
    func resumeWaitingProducers() {
        isTerminated = true
        let producers = waitingProducers
        waitingProducers.removeAll()
        for producer in producers {
            producer.resume()
        }
    }
}

/// Represents a chunk of audio data from an audio file for incremental loading.
///
/// This internal structure is used by the incremental file loading API to deliver audio data
/// in manageable chunks while maintaining back-pressure control to prevent memory overuse.
/// Each chunk contains the audio samples and a completion signal that must be called
/// by the consumer to maintain proper flow control.
///
/// - Important: This is an internal structure used by the audio streaming implementation.
///   The `completionSignal` closure is critical for the back-pressure mechanism and must
///   be called when the chunk has been processed to allow the next chunk to be loaded.
struct IncrementalFileChunk: Sendable {
    /// Audio chunk containing the audio samples and seek offset information
    let audioChunk: AudioChunk

    /// Completion signal used internally for back-pressure flow control.
    /// Must be called when chunk processing is complete to signal the producer
    /// that it can continue loading the next chunk.
    let completionSignal: @Sendable () -> Void

    init(audioSamples: [Float], seekOffsetIndex: Int, completionSignal: @escaping @Sendable () -> Void) {
        self.audioChunk = .init(seekOffsetIndex: seekOffsetIndex, audioSamples: audioSamples)
        self.completionSignal = completionSignal
    }
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
extension AudioProcessor {
    typealias IncrementalFileStream = AsyncThrowingStream<IncrementalFileChunk, Error>

    /// Streams an audio file in bounded-memory chunks instead of loading it all at once.
    ///
    /// Back-pressure caps how many chunks are buffered: the consumer must call each chunk's
    /// `completionSignal` when done, or loading pauses once `maxBufferedChunks` chunks are queued.
    ///
    /// - Parameters:
    ///   - audioFilePath: Path to the audio file to stream.
    ///   - channelMode: How to handle multi-channel audio. Defaults to summing all channels.
    ///   - chunkDurationSeconds: Seconds of audio to read per staging step.
    ///   - maxBufferedChunks: Max chunks buffered before back-pressure pauses loading.
    ///   - maxChunkLength: Model audio-window size used for VAD chunk boundaries.
    /// - Returns: An async throwing stream of `IncrementalFileChunk`s.
    /// - Note: Internal API; throws on validation, initialization, or read failure.
    static func loadFileIncrementally(
        fromPath audioFilePath: String,
        channelMode: ChannelMode = .sumChannels(nil),
        chunkDurationSeconds: Double,
        maxBufferedChunks: Int,
        maxChunkLength: Int = Constants.defaultWindowSamples,
        vad: VoiceActivityDetector = EnergyVAD()
    ) throws -> IncrementalFileStream {
        guard chunkDurationSeconds.isFinite, chunkDurationSeconds > 0 else {
            throw WhisperError.loadAudioFailed("Incremental chunk duration must be greater than 0")
        }
        guard maxBufferedChunks > 0 else {
            throw WhisperError.loadAudioFailed("Incremental chunk buffer size must be greater than 0")
        }
        guard FileManager.default.fileExists(atPath: audioFilePath) else {
            throw WhisperError.loadAudioFailed("Audio file not found at path: \(audioFilePath)")
        }
        return AsyncThrowingStream<IncrementalFileChunk, Error> { continuation in
            let flowController = IncrementalLoadController(maxBufferedChunks: maxBufferedChunks)
            let producerTask = Task {
                do {
                    let audioFileURL = URL(fileURLWithPath: audioFilePath)
                    let audioFile = try AVAudioFile(forReading: audioFileURL, commonFormat: .pcmFormatFloat32, interleaved: false)
                    let inputSampleRate = audioFile.fileFormat.sampleRate
                    // `audioFile.length` is an Int64 frame position; divide directly to avoid
                    // narrowing to AVAudioFrameCount (UInt32), which overflows on very long files.
                    let inputDuration = Double(audioFile.length) / inputSampleRate

                    let end = inputDuration
                    var currentTime = 0.0

                    // Chunk at VAD (silence) boundaries with the same chunker as the full-file `.vad` path,
                    // so the result matches a `.vad` transcription. `chunkDurationSeconds` only sets the read/staging
                    // size, not the boundaries; partially-decided chunks carry across staging buffers.
                    // `maxChunkLength` is the model's audio window, passed in so boundaries track the model.
                    let chunker = VADAudioChunker(vad: vad)
                    var audioBuffer: [Float] = []   // audioBuffer[0] corresponds to global sample `bufferStartSample`
                    var bufferStartSample = 0

                    while true {
                        try Task.checkCancellation()

                        // Refill with at least one window of lookahead so chunk cuts match the full-file path.
                        while audioBuffer.count <= maxChunkLength && currentTime < end {
                            let chunkEnd = min(currentTime + chunkDurationSeconds, end)
                            try autoreleasepool {
                                let buffer = try loadAudio(
                                    fromFile: audioFile,
                                    channelMode: channelMode,
                                    startTime: currentTime,
                                    endTime: chunkEnd
                                )
                                audioBuffer.append(contentsOf: Self.convertBufferToArray(buffer: buffer))
                            }
                            currentTime = chunkEnd
                        }

                        if audioBuffer.isEmpty { break }
                        let atEOF = currentTime >= end

                        let chunks = try await chunker.chunkAll(
                            audioArray: audioBuffer,
                            maxChunkLength: maxChunkLength,
                            decodeOptions: nil
                        )

                        // A chunk is final once its full decision window is staged; otherwise more
                        // audio could still move its cut, so carry it forward. At EOF all chunks are final.
                        var emitCount = chunks.count
                        if !atEOF {
                            emitCount = 0
                            for chunk in chunks {
                                if chunk.seekOffsetIndex + maxChunkLength <= audioBuffer.count {
                                    emitCount += 1
                                } else {
                                    break
                                }
                            }
                        }

                        var consumed = 0
                        for chunk in chunks.prefix(emitCount) {
                            try Task.checkCancellation()
                            // Acquire a buffer slot per emitted chunk (1:1, released by completionSignal).
                            await flowController.requestChunkSlot()
                            try Task.checkCancellation()
                            continuation.yield(
                                IncrementalFileChunk(
                                    audioSamples: chunk.audioSamples,
                                    seekOffsetIndex: bufferStartSample + chunk.seekOffsetIndex,
                                    completionSignal: {
                                        Task { await flowController.signalChunkConsumed() }
                                    }
                                )
                            )
                            consumed = chunk.seekOffsetIndex + chunk.audioSamples.count
                        }

                        if atEOF { break }
                        bufferStartSample += consumed
                        audioBuffer = Array(audioBuffer[consumed...])
                    }

                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    Logging.error("Error loading audio file incrementally: \(error)")
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                producerTask.cancel()
                Task {
                    await flowController.resumeWaitingProducers()
                }
            }
        }
    }
}
