//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import AVFoundation
import Accelerate
import ArgmaxCore
import Foundation

// MARK: - Audio Output

/// Handles audio export to file and real-time streaming playback via AVAudioEngine.
///
/// Supports adaptive pre-buffering and edge-fading to prevent audible clicks:
///
/// **Pre-buffering:** When a buffer duration is configured via `setBufferDuration(_:)`,
/// incoming audio frames accumulate until the threshold is reached, then flush to the
/// player all at once. This prevents underruns on slower devices.
///
/// **Edge-fading:** Fades are applied only at actual audio discontinuities:
/// - Fade-in on the first frame of a session, chunk, or after a detected underrun.
/// - Fade-out on the last frame of a session, chunk, or before a detected underrun.
/// Interior frames of contiguous playback are untouched.
///
/// Underrun detection uses wall-clock timing: if the current time exceeds the
/// expected playback end of all scheduled audio, the player has drained and the
/// next frame needs fade-in (and the previous tail gets fade-out).
///
/// **Buffer lifecycle:**
/// 1. `startPlayback()` - resets all state; frames accumulate until configured.
/// 2. `setBufferDuration(_:)` - configures threshold (call after start).
/// 3. `enqueueAudioChunk(_:)` - pushes frames through the buffer/tail pipeline.
/// 4. `stopPlayback()` - commits the tail with fade-out, waits, tears down.
///
/// Thread safety: used only from `TTSKit.play()` which forces sequential mode
/// (concurrency=1). `TTSKit` ensures serialized access.
/// Subclassing is intentionally not supported. Use the `TTSKit` component override
/// mechanism (`TTSKitConfig`) to swap in an alternative audio backend.
public class AudioOutput: @unchecked Sendable {
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var engineStartDeferred: Bool = false

    /// Pre-buffer threshold in seconds. `nil` means not yet configured - frames
    /// accumulate in `pendingFrames` until `setBufferDuration` is called.
    /// `0` means stream immediately. `> 0` means buffer that duration first.
    private var bufferDuration: TimeInterval?

    /// Accumulated frames waiting to be flushed to the player.
    private var pendingFrames: [[Float]] = []

    /// Total duration (seconds) of audio in `pendingFrames`.
    private var pendingDuration: TimeInterval = 0

    /// Whether the initial buffer threshold has been met and frames flow directly
    /// to the player without accumulating.
    private var bufferThresholdMet: Bool = false

    /// The most recently received frame, held back so we can apply fade-out
    /// when we learn it's the last frame before a gap or end of session.
    private var tailFrame: [Float]?

    /// Whether the next frame scheduled to the player needs a fade-in
    /// (start of session, start of chunk, or after an underrun).
    private var needsFadeIn: Bool = true

    /// Wall-clock time at which all currently scheduled audio is expected to
    /// finish playing. Used to detect underruns: if `now > expectedPlaybackEnd`,
    /// the player has drained and the next frame follows a gap.
    private var expectedPlaybackEnd: CFAbsoluteTime = 0

    /// Player time (seconds) when the first buffer was scheduled. The player
    /// node's clock starts at `.play()` but no audio plays until buffers are
    /// queued, so this offset must be subtracted from `playerTime.sampleTime`
    /// to get the true audio-out position.
    private var playbackTimeOffset: TimeInterval = 0

    /// Cumulative duration (seconds) of real audio that has been scheduled via
    /// `scheduleWithFades`. The silent sentinel buffer used for drain detection
    /// is not included. Used to clamp `currentPlaybackTime` so the reported
    /// position never advances into silence gaps between chunks or past the end
    /// of generated audio.
    public private(set) var scheduledAudioDuration: TimeInterval = 0

    /// Number of samples for the fade-in/fade-out ramp.
    /// 256 samples at 24kHz ≈ 10.7ms - imperceptible on contiguous audio
    /// but smoothly eliminates clicks at discontinuities.
    public static let fadeLengthSamples: Int = 256

    /// Output sample rate in Hz. Defaults to 24000 (Qwen3 TTS).
    /// Updated by `TTSKit.loadModels()` to match the loaded speech decoder's actual sample rate.
    public private(set) var sampleRate: Int

    /// The audio format used for playback and export (derived from `sampleRate`).
    public private(set) var audioFormat: AVAudioFormat

    public init(sampleRate: Int = 24000) {
        self.sampleRate = sampleRate
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false) else {
            preconditionFailure("AVAudioFormat init failed for PCM Float32 \(sampleRate)Hz mono - this is an invariant violation")
        }
        self.audioFormat = format
    }

    /// Update the sample rate to match the loaded speech decoder.
    /// Must be called before `startPlayback()`.
    public func configure(sampleRate newRate: Int) {
        guard newRate != sampleRate else { return }
        sampleRate = newRate
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(newRate), channels: 1, interleaved: false) else {
            preconditionFailure("AVAudioFormat init failed for PCM Float32 \(newRate)Hz mono - this is an invariant violation")
        }
        audioFormat = format
    }

    /// Current playback position in seconds, based on the audio engine's render timeline.
    /// Returns 0 if the player is not active, no audio has been scheduled yet, or
    /// the player hasn't started rendering.
    ///
    /// Clamped to `scheduledAudioDuration` so the position never advances into
    /// silence gaps between chunks, the silent drain sentinel, or the hardware
    /// pipeline tail after the last real audio frame.
    /// Returns how many seconds of audio still need to accumulate in the pre-buffer before
    /// the next chunk flushes and playback resumes. Non-zero only while in buffering
    /// mode (`bufferThresholdMet == false` and a positive `bufferDuration` is set).
    public var silentBufferRemaining: TimeInterval {
        guard !bufferThresholdMet, let duration = bufferDuration, duration > 0 else { return 0 }
        return max(0, duration - pendingDuration)
    }

    public var currentPlaybackTime: TimeInterval {
        // Guard against the engine being torn down (stopPlayback nullifies these).
        // Checking audioEngine?.isRunning prevents accessing a detached node.
        guard expectedPlaybackEnd > 0,
            audioEngine?.isRunning == true,
            let player = playerNode,
            let nodeTime = player.lastRenderTime,
            nodeTime.isSampleTimeValid,
            let playerTime = player.playerTime(forNodeTime: nodeTime)
        else { return 0 }
        let rawTime = max(0, Double(playerTime.sampleTime) / playerTime.sampleRate - playbackTimeOffset)
        return min(rawTime, scheduledAudioDuration)
    }

    // MARK: - Buffer Configuration

    /// Configure the pre-buffer duration. Call after `startPlayback()`.
    ///
    /// - If `seconds == 0`: immediately flushes any pending frames and switches
    ///   to direct streaming (no buffering).
    /// - If `seconds > 0`: sets the threshold. If enough audio has already
    ///   accumulated, flushes immediately.
    /// - Can be called multiple times (e.g., per-chunk reassessment). Any held
    ///   tail frame from the previous chunk is committed with fade-out first.
    ///
    /// - Parameter seconds: Duration of audio to accumulate before flushing.
    ///   Pass 0 for immediate streaming (fast devices).
    public func setBufferDuration(_ seconds: TimeInterval) {
        let duration = max(0, seconds)
        bufferDuration = duration

        // Commit any held tail as the last frame of the previous chunk
        if let tail = tailFrame {
            scheduleWithFades(tail, fadeIn: needsFadeIn, fadeOut: true)
            needsFadeIn = true // next chunk starts fresh with fade-in
            tailFrame = nil
        }

        if duration == 0 {
            bufferThresholdMet = true
            if !pendingFrames.isEmpty {
                flushPendingFrames()
            }
        } else {
            // Re-enter buffering mode so the new chunk accumulates frames
            // before flushing - lets the model catch up between chunks.
            bufferThresholdMet = false
            if pendingDuration >= duration {
                flushPendingFrames()
            }
        }
    }

    // MARK: - File Export

    /// Supported audio export formats.
    public enum AudioFileFormat: String, Sendable {
        case m4a
        case wav

        public var fileExtension: String { rawValue }

        /// Resolve the effective format for the current platform.
        /// On watchOS, M4A is not supported so falls back to WAV with a warning.
        public static func resolve(_ preferred: AudioFileFormat = .m4a) -> AudioFileFormat {
            #if os(watchOS)
            if preferred == .m4a {
                Logging.info("[Warning] M4A export is not available on watchOS, falling back to WAV")
                return .wav
            }
            #endif
            return preferred
        }
    }

    /// Save audio samples to a file.
    ///
    /// For M4A with metadata: writes PCM -> AAC to a temp file, then uses
    /// `AVAssetExportSession` passthrough to remux with embedded metadata atoms
    /// (no re-encode). For WAV or metadata-free M4A: writes directly.
    /// On watchOS, `.m4a` automatically falls back to `.wav`.
    ///
    /// Any extension already present in `filename` is stripped before writing.
    /// The output format is resolved in this order: explicit `format` parameter,
    /// then extension found in `filename` if it matches a supported format,
    /// then `.m4a` as the default.
    ///
    /// - Parameters:
    ///   - samples: Mono Float32 PCM samples.
    ///   - folder: Destination directory. Created if it doesn't exist.
    ///   - filename: File name, with or without extension.
    ///   - sampleRate: Sample rate in Hz.
    ///   - format: Output format. Inferred from `filename` extension when `nil`.
    ///   - metadataProvider: Optional metadata callback for items to embed into the file container for m4a formats.
    /// - Returns: The URL of the written file.
    /// - Throws: `TTSError` if audio encoding or export fails.
    @discardableResult
    public static func saveAudio(
        _ samples: [Float],
        toFolder folder: URL,
        filename: String,
        sampleRate: Int = 24000,
        format: AudioFileFormat? = nil,
        metadataProvider: (@Sendable () throws -> [AVMetadataItem])? = nil
    ) async throws -> URL {
        guard !samples.isEmpty else {
            throw TTSError.audioOutputFailed("No audio samples to export")
        }

        let filenameURL = URL(fileURLWithPath: filename)
        let baseName = filenameURL.deletingPathExtension().lastPathComponent
        let inferredFormat = AudioFileFormat(rawValue: filenameURL.pathExtension.lowercased())
        let resolvedFormat = AudioFileFormat.resolve(format ?? inferredFormat ?? .m4a)
        let outputURL = folder.appendingPathComponent(baseName)
            .appendingPathExtension(resolvedFormat.fileExtension)

        if !FileManager.default.fileExists(atPath: folder.path) {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        }
        try? FileManager.default.removeItem(at: outputURL)

        let pcmBuffer = try createPCMBuffer(from: samples, sampleRate: sampleRate)

        switch resolvedFormat {
            case .m4a:
                #if os(watchOS)
                throw TTSError.audioOutputFailed("M4A should have been resolved to WAV on watchOS")
                #else
                if let metadataProvider {
                    let metadata = try metadataProvider()
                    try await writeM4AWithMetadata(pcmBuffer, to: outputURL, sampleRate: sampleRate, metadata: metadata)
                } else {
                    try writeM4A(pcmBuffer, to: outputURL, sampleRate: sampleRate)
                }
                #endif
            case .wav:
                try writeWAV(pcmBuffer, to: outputURL, sampleRate: sampleRate)
        }

        return outputURL
    }

    /// Return the playback duration of an audio file in seconds.
    public static func duration(of url: URL) async throws -> TimeInterval {
        let asset = AVURLAsset(url: url)
        let cmDuration = try await asset.load(.duration)
        return CMTimeGetSeconds(cmDuration)
    }

    // MARK: - Crossfade assembly

    /// Assemble multiple audio chunks into one array with equal-power crossfades at each boundary.
    ///
    /// Uses `cos(t*pi/2)` fade-out and `sin(t*pi/2)` fade-in so that energy is
    /// preserved through the overlap region.  Fade curves are pre-computed once
    /// via Accelerate (`vDSP_vramp` + `vvcosf`/`vvsinf`) and reused at every
    /// chunk boundary; the per-boundary mix uses `vDSP_vmul` + `vDSP_vma`.
    ///
    /// - Parameters:
    ///   - chunks: Ordered audio chunks to concatenate.
    ///   - fadeLength: Number of overlap samples for each crossfade.
    /// - Returns: Single concatenated audio array with crossfades applied at chunk boundaries.
    public static func crossfade(_ chunks: [[Float]], fadeLength: Int) -> [Float] {
        guard !chunks.isEmpty else { return [] }
        guard chunks.count > 1 else { return chunks[0] }

        let (fadeOut, fadeIn) = equalPowerCurves(length: fadeLength)

        var result = [Float]()
        result.reserveCapacity(chunks.reduce(0) { $0 + $1.count })
        result.append(contentsOf: chunks[0])

        for i in 1..<chunks.count {
            let next = chunks[i]
            let overlap = min(fadeLength, result.count, next.count)

            if overlap == 0 {
                result.append(contentsOf: next)
                continue
            }

            let (fo, fi) =
                overlap == fadeLength
                ? (fadeOut, fadeIn)
                : equalPowerCurves(length: overlap)

            let overlapStart = result.count - overlap
            let n = vDSP_Length(overlap)

            result.withUnsafeMutableBufferPointer { buf in
                guard let bufBase = buf.baseAddress else { return }
                let dst = bufBase + overlapStart
                vDSP_vmul(dst, 1, fo, 1, dst, 1, n)
                next.withUnsafeBufferPointer { src in
                    guard let srcBase = src.baseAddress else { return }
                    vDSP_vma(srcBase, 1, fi, 1, dst, 1, dst, 1, n)
                }
            }

            if overlap < next.count {
                result.append(contentsOf: next[overlap...])
            }
        }

        return result
    }

    /// Pre-compute equal-power fade-out (cos) and fade-in (sin) curves.
    private static func equalPowerCurves(length: Int) -> ([Float], [Float]) {
        guard length > 0 else { return ([], []) }

        var ramp = [Float](repeating: 0, count: length)
        var start: Float = 0
        var step = Float.pi / 2.0 / Float(max(length - 1, 1))
        vDSP_vramp(&start, &step, &ramp, 1, vDSP_Length(length))

        var fadeOut = [Float](repeating: 0, count: length)
        var fadeIn = [Float](repeating: 0, count: length)
        var n = Int32(length)
        vvcosf(&fadeOut, ramp, &n)
        vvsinf(&fadeIn, ramp, &n)

        return (fadeOut, fadeIn)
    }

    // MARK: - Internal helpers

    private static func createPCMBuffer(from samples: [Float], sampleRate: Int) throws -> AVAudioPCMBuffer {
        guard
            let pcmFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(sampleRate),
                channels: 1,
                interleaved: false
            )
        else {
            throw TTSError.audioOutputFailed("Failed to create PCM format for sampleRate \(sampleRate)")
        }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: pcmFormat, frameCapacity: AVAudioFrameCount(samples.count)) else {
            throw TTSError.audioOutputFailed("Failed to create PCM buffer")
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)

        guard let channelData = buffer.floatChannelData else {
            throw TTSError.audioOutputFailed("Failed to access buffer channel data")
        }
        samples.withUnsafeBufferPointer { src in
            guard let srcBase = src.baseAddress else { return }
            channelData[0].update(from: srcBase, count: samples.count)
        }

        return buffer
    }

    #if !os(watchOS)
    /// Write PCM samples as AAC in an M4A container.
    ///
    /// `AVAudioFile` accepts PCM input via `write(from:)` and internally
    /// encodes to AAC when the file settings specify a compressed format,
    /// so no temp file or explicit `AVAudioConverter` is needed.
    private static func writeM4A(
        _ pcmBuffer: AVAudioPCMBuffer,
        to url: URL,
        sampleRate: Int
    ) throws {
        let aacSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 64000,
            AVEncoderAudioQualityKey: AVAudioQuality.medium.rawValue
        ]

        let file = try AVAudioFile(
            forWriting: url,
            settings: aacSettings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )
        try file.write(from: pcmBuffer)
    }

    /// Write AAC/M4A with embedded metadata atoms.
    ///
    /// `AVAudioFile` cannot embed metadata, so this does a two-step write:
    /// encode PCM -> AAC into a temp file, then use `AVAssetExportSession`
    /// passthrough to remux the bitstream into the final file with metadata
    /// atoms attached. No audio re-encoding occurs.
    private static func writeM4AWithMetadata(
        _ pcmBuffer: AVAudioPCMBuffer,
        to url: URL,
        sampleRate: Int,
        metadata: [AVMetadataItem]
    ) async throws {
        let tempURL = url.deletingLastPathComponent()
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("m4a")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try writeM4A(pcmBuffer, to: tempURL, sampleRate: sampleRate)

        let asset = AVURLAsset(url: tempURL)
        guard let exportSession = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetPassthrough) else {
            throw TTSError.audioOutputFailed("AVAssetExportSession could not be created for metadata export")
        }
        exportSession.outputURL = url
        exportSession.outputFileType = .m4a
        exportSession.metadata = metadata

        await exportSession.export()

        if let exportError = exportSession.error {
            throw exportError
        }
        guard exportSession.status == .completed else {
            throw TTSError.audioOutputFailed(
                "AVAssetExportSession finished with unexpected status \(exportSession.status.rawValue)")
        }
    }
    #endif

    private static func writeWAV(
        _ pcmBuffer: AVAudioPCMBuffer,
        to url: URL,
        sampleRate: Int
    ) throws {
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: true
        ]

        let file = try AVAudioFile(
            forWriting: url,
            settings: settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )
        try file.write(from: pcmBuffer)
    }

    // MARK: - Streaming Playback

    /// Start the audio engine for streaming playback.
    ///
    /// Resets all buffering, fade, and timing state. After calling this,
    /// configure the buffer threshold via `setBufferDuration(_:)`.
    ///
    /// - Parameter deferEngineStart: When `true`, the audio engine is created and
    ///   connected but not started. The engine will start automatically on the first
    ///   `enqueueAudioChunk` call. This avoids the render thread contending with
    ///   model predictions during the critical time-to-first-buffer path.
    /// - Throws: `TTSError` if the audio engine fails to start.
    public func startPlayback(deferEngineStart: Bool = false) throws {
        pendingFrames.removeAll()
        pendingDuration = 0
        bufferThresholdMet = false
        bufferDuration = nil
        tailFrame = nil
        needsFadeIn = true
        expectedPlaybackEnd = 0
        playbackTimeOffset = 0
        scheduledAudioDuration = 0
        engineStartDeferred = false

        // On iOS, AVAudioEngine requires an active audio session with a playback
        // category. Without this, engine.start() may silently fail or route to
        // the wrong output (e.g., airpods instead of main speaker).
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default, options: [])
        try session.setActive(true)
        #endif

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let format = audioFormat

        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)

        if deferEngineStart {
            engineStartDeferred = true
        } else {
            try engine.start()
            player.play()
        }

        self.audioEngine = engine
        self.playerNode = player
    }

    /// Enqueue a chunk of audio samples for playback.
    ///
    /// In streaming mode, detects underruns via wall-clock timing: if the player
    /// has drained since the last buffer, the held tail is committed with fade-out
    /// (it was the last frame before the gap) and the incoming frame is marked for
    /// fade-in. On contiguous playback, no fades are applied to interior frames.
    public func enqueueAudioChunk(_ samples: [Float]) {
        guard let engine = audioEngine, let player = playerNode else { return }

        if engineStartDeferred {
            engineStartDeferred = false
            do {
                try engine.start()
                player.play()
            } catch {
                Logging.error("AudioOutput: deferred engine start failed: \(error)")
                return
            }
        }

        if bufferThresholdMet {
            // Detect underrun: has the player drained since the last schedule?
            let playerDrained = CFAbsoluteTimeGetCurrent() > expectedPlaybackEnd

            if let tail = tailFrame {
                // If player drained, this tail was the last frame before silence -
                // apply fade-out to smooth the audio -> silence transition, and
                // mark the next frame for fade-in (silence -> audio transition).
                let fadeOut = playerDrained
                let fadeIn = needsFadeIn || playerDrained
                scheduleWithFades(tail, fadeIn: fadeIn, fadeOut: fadeOut)
                needsFadeIn = playerDrained
            } else if playerDrained {
                needsFadeIn = true
            }
            tailFrame = samples
        } else {
            pendingFrames.append(samples)
            pendingDuration += Double(samples.count) / Double(sampleRate)

            if let threshold = bufferDuration, pendingDuration >= threshold {
                flushPendingFrames()
            }
        }
    }

    /// Flush all accumulated frames to the player.
    ///
    /// All frames except the last are scheduled immediately (they're contiguous).
    /// The last frame becomes the tail, held back until the next frame arrives
    /// or `stopPlayback` commits it with fade-out.
    private func flushPendingFrames() {
        guard !pendingFrames.isEmpty else {
            bufferThresholdMet = true
            return
        }

        // For subsequent chunks (scheduledAudioDuration > 0), the player may have
        // been idle during the inter-chunk gap while the model generated the next
        // buffer. The raw player clock kept advancing through that silence, so
        // rawTime = playbackTimeOffset + scheduledAudioDuration + gap.
        // Absorb the gap into playbackTimeOffset so currentPlaybackTime resumes
        // from the end of the previous chunk rather than jumping ahead by gap.
        if scheduledAudioDuration > 0,
            let player = playerNode,
            let nodeTime = player.lastRenderTime,
            nodeTime.isSampleTimeValid,
            let playerTime = player.playerTime(forNodeTime: nodeTime)
        {
            let currentRawTime = Double(playerTime.sampleTime) / playerTime.sampleRate
            let expectedRawTime = playbackTimeOffset + scheduledAudioDuration
            let gap = currentRawTime - expectedRawTime
            if gap > 0.01 {
                playbackTimeOffset += gap
            }
        }

        for i in 0..<pendingFrames.count - 1 {
            scheduleWithFades(pendingFrames[i], fadeIn: i == 0 && needsFadeIn, fadeOut: false)
            if i == 0, needsFadeIn { needsFadeIn = false }
        }

        tailFrame = pendingFrames.last
        pendingFrames.removeAll()
        pendingDuration = 0
        bufferThresholdMet = true
    }

    /// Schedule a buffer on the player node, optionally applying fade-in/fade-out.
    ///
    /// Also updates `expectedPlaybackEnd` so underrun detection stays accurate.
    /// Fades are only applied when explicitly requested - interior frames of
    /// contiguous playback pass through untouched.
    private func scheduleWithFades(_ samples: [Float], fadeIn: Bool, fadeOut: Bool) {
        guard let player = playerNode else { return }
        let format = audioFormat
        let count = samples.count

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)) else {
            return
        }
        buffer.frameLength = AVAudioFrameCount(count)

        guard let channelData = buffer.floatChannelData else { return }
        samples.withUnsafeBufferPointer { srcPtr in
            guard let srcBase = srcPtr.baseAddress else { return }
            channelData[0].update(from: srcBase, count: count)
        }

        if fadeIn || fadeOut {
            let fadeLen = min(AudioOutput.fadeLengthSamples, count / 2)
            if fadeLen > 0 {
                let data = channelData[0]
                let invFade = 1.0 / Float(fadeLen)

                if fadeIn {
                    for i in 0..<fadeLen {
                        data[i] *= Float(i) * invFade
                    }
                }

                if fadeOut {
                    let fadeOutStart = count - fadeLen
                    for i in 0..<fadeLen {
                        data[fadeOutStart + i] *= Float(fadeLen - i) * invFade
                    }
                }
            }
        }

        // Capture the player's elapsed time before the first buffer is scheduled.
        // This is the dead time between .play() and actual audio output.
        if expectedPlaybackEnd == 0 {
            if let nodeTime = player.lastRenderTime,
                nodeTime.isSampleTimeValid,
                let playerTime = player.playerTime(forNodeTime: nodeTime)
            {
                playbackTimeOffset = max(0, Double(playerTime.sampleTime) / playerTime.sampleRate)
            }
        }

        // Update expected playback end for underrun detection.
        // If player was idle, this buffer starts now. If audio is still queued,
        // it starts after the currently queued audio finishes.
        let now = CFAbsoluteTimeGetCurrent()
        let bufferSeconds = Double(count) / Double(sampleRate)
        expectedPlaybackEnd = max(expectedPlaybackEnd, now) + bufferSeconds
        scheduledAudioDuration += bufferSeconds

        player.scheduleBuffer(buffer)
    }

    /// Stop playback and tear down the audio engine.
    /// Optionally waits for any remaining scheduled buffers to finish playing.
    ///
    /// The held tail frame is committed with fade-out (it's the last frame of
    /// the session). Any remaining pending frames are flushed first.
    public func stopPlayback(waitForCompletion: Bool = true) async {
        if !bufferThresholdMet, !pendingFrames.isEmpty {
            flushPendingFrames()
        }

        // Commit the tail as the final frame with fade-out
        if let tail = tailFrame {
            scheduleWithFades(tail, fadeIn: needsFadeIn, fadeOut: true)
            tailFrame = nil
            needsFadeIn = false
        }

        if waitForCompletion, let player = playerNode {
            // Schedule a silent sentinel buffer. Its completion fires once the player
            // node has dequeued it - meaning all prior audio buffers have been consumed
            // by the render thread and sent to the hardware output pipeline.
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                let format = audioFormat
                if let silentBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1) {
                    silentBuffer.frameLength = 1
                    silentBuffer.floatChannelData?[0][0] = 0
                    player.scheduleBuffer(silentBuffer) {
                        continuation.resume()
                    }
                } else {
                    continuation.resume()
                }
            }

            // Wait ~80ms after the sentinel so the hardware pipeline drains (~1-2 render cycles).
            // Prevents tail clip and CoreAudio "out of order"/overload errors across devices.
            try? await Task.sleep(for: .milliseconds(80))
        }

        let engine = audioEngine
        playerNode = nil
        audioEngine = nil

        // Stop the engine only
        engine?.stop()

        pendingFrames.removeAll()
        pendingDuration = 0
        bufferThresholdMet = false
        bufferDuration = nil
        tailFrame = nil
        needsFadeIn = true
        expectedPlaybackEnd = 0
        playbackTimeOffset = 0
    }
}
