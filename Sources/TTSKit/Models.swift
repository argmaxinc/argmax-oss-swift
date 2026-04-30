//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation

// MARK: - Generation progress

/// Per-step progress information delivered to a `SpeechCallback` during generation.
///
/// A single callback receives both streaming audio and timing metadata on every
/// generation step.
///
/// **Typical usage:**
/// ```swift
/// let result = try await tts.generate(text: "Hello!") { progress in
///     player.enqueue(progress.audio)                                  // stream audio
///     if let t = progress.stepTime { configureBufferingStrategy(t) }  // first-step setup
///     return nil                                                      // nil or true = continue, false = cancel
/// }
/// ```
public struct SpeechProgress: Sendable {
    /// Audio samples produced by this generation step (~80ms at 24 kHz).
    public var audio: [Float]

    /// Timings accumulated so far in the current chunk.
    public var timings: SpeechTimings

    /// Wall-clock duration of this generation step (seconds).
    ///
    /// Non-`nil` **only on the first step** - use it to configure adaptive playback
    /// buffers (see `PlaybackStrategy.auto`) before the first audio frame is played.
    public var stepTime: TimeInterval?

    /// Zero-based index of the chunk currently being generated (for multi-chunk requests).
    public var chunkIndex: Int?

    /// Total number of text chunks in this generation request.
    public var totalChunks: Int?

    /// Running count of generation steps completed so far across all chunks.
    /// Increases by one for every decoder step, regardless of which chunk produced it.
    public var stepsCompleted: Int?

    /// Estimated total steps for the entire request (`maxNewTokens × totalChunks`).
    /// Actual steps may be fewer if chunks finish early (EOS).
    public var totalSteps: Int?

    public init(
        audio: [Float],
        timings: SpeechTimings,
        stepTime: TimeInterval? = nil,
        chunkIndex: Int? = nil,
        totalChunks: Int? = nil,
        stepsCompleted: Int? = nil,
        totalSteps: Int? = nil
    ) {
        self.audio = audio
        self.timings = timings
        self.stepTime = stepTime
        self.chunkIndex = chunkIndex
        self.totalChunks = totalChunks
        self.stepsCompleted = stepsCompleted
        self.totalSteps = totalSteps
    }
}

/// A closure invoked on each generation step with decoded audio and timing info.
///
/// Return `false` to cancel generation early; return `nil` or `true` to continue.
/// Matches the `TranscriptionCallback` pattern in WhisperKit.
public typealias SpeechCallback = (@Sendable (SpeechProgress) -> Bool?)?

// MARK: - SpeechModel Protocol

/// The stable public contract that every TTS model family must satisfy.
///
/// Conform to this protocol to add a new TTS model family to TTSKit.
///
/// **Minimum implementation:**
/// ```swift
/// public class MyTTSModel: SpeechModel {
///     public var sampleRate: Int { 24000 }
///
///     public func generate(text: String, voice: String, language: String,
///                          options: GenerationOptions, callback: SpeechCallback) async throws -> SpeechResult { ... }
///
///     public func play(text: String, voice: String, language: String,
///                      options: GenerationOptions, playbackStrategy: PlaybackStrategy,
///                      callback: SpeechCallback) async throws -> SpeechResult { ... }
/// }
/// ```
public protocol SpeechModel: AnyObject, Sendable {
    /// Output sample rate in Hz for all audio produced by this model.
    var sampleRate: Int { get }

    /// Generate speech from text and return the complete audio result.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voice: Voice/speaker identifier. `nil` uses the model's default voice.
    ///   - language: Language identifier. `nil` uses the model's default language.
    ///   - options: Sampling and generation options. Model-specific fields are ignored by models
    ///     that do not support them.
    ///   - callback: Optional per-step callback receiving decoded audio chunks.
    ///               Return `false` to cancel; `nil` or `true` to continue.
    /// - Returns: The assembled `SpeechResult` containing all audio and timing data.
    /// - Throws: `TTSError` on generation failure or task cancellation.
    func generate(
        text: String,
        voice: String?,
        language: String?,
        options: GenerationOptions,
        callback: SpeechCallback
    ) async throws -> SpeechResult

    /// Generate speech and stream it through the device audio output in real time.
    ///
    /// Default implementation calls `generate` and plays via `AudioOutput`. Override
    /// if the model has its own streaming playback path.
    func play(
        text: String,
        voice: String?,
        language: String?,
        options: GenerationOptions,
        playbackStrategy: PlaybackStrategy,
        callback: SpeechCallback
    ) async throws -> SpeechResult
}

// MARK: - Playback Strategy

/// Controls how `play()` buffers audio before starting playback.
///
/// The TTS generation loop produces one RVQ frame (~80ms of audio) per step.
/// On fast devices, steps complete well under 80ms and audio can stream immediately.
/// On slower devices (or in debug builds), steps may exceed 80ms, causing the
/// playback buffer to drain and produce choppy audio.
///
/// `.auto` (default) measures the first generation step - which runs the full
/// pipeline (MultiCodeDecoder + SpeechDecoder + CodeDecoder) before any audio
/// is emitted - and uses that exact timing to compute the pre-buffer needed so
/// playback is less likely to underrun. The buffer is re-assessed at the start of each chunk.
public enum PlaybackStrategy: Sendable {
    /// Automatically determine buffer duration from the first step's measured
    /// wall-clock time. On fast devices this resolves to a small minimum (~240ms)
    /// to absorb per-step variance. On slower devices it pre-buffers just enough
    /// to avoid underruns for the full generation. Re-assessed per chunk.
    case auto

    /// Always stream frame-by-frame with no pre-buffer.
    /// May produce choppy audio if the device can't generate faster than real-time.
    case stream

    /// Pre-buffer a fixed duration of audio before starting playback.
    case buffered(seconds: Double)

    /// Generate all audio for each chunk first, then play.
    /// Highest latency but guaranteed smooth playback.
    case generateFirst

    // MARK: - Buffer calculation

    /// Default minimum buffer duration (seconds) applied in `.auto` mode when no
    /// model-specific value is available.
    /// 80ms ≈ 1 audio frame at 24 kHz / 1920 spf - provides headroom for scheduling jitter.
    /// The `SpeechDecoding` protocol exposes `minimumBufferDuration` so each model can
    /// override this value; pass it to `requiredBuffer(minimumBuffer:)` at the call site.
    public static let minimumBufferDuration: TimeInterval = 0.08

    /// Audio duration produced by a single generation step (seconds).
    ///
    /// - Parameters:
    ///   - samplesPerFrame: PCM samples produced per decode step (model-specific).
    ///   - sampleRate: Output sample rate in Hz (model-specific).
    /// - Returns: Audio duration in seconds for one generation step.
    public static func audioPerStep(samplesPerFrame: Int, sampleRate: Int) -> Double {
        Double(samplesPerFrame) / Double(sampleRate)
    }

    /// Compute the required pre-buffer duration (seconds) given a measured step time.
    ///
    /// Called on every generation step so the buffer duration converges toward
    /// the minimum as the measured speed approaches or exceeds real-time.
    /// No additional safety margin is applied; `maxNewTokens` is already an
    /// upper bound on actual generation length, providing natural conservatism.
    ///
    /// - Parameters:
    ///   - stepTime: Measured wall-clock time of one generation step (seconds).
    ///   - maxNewTokens: Upper bound on remaining generation steps.
    ///   - samplesPerFrame: PCM samples produced per decode step (model-specific).
    ///   - sampleRate: Output sample rate in Hz (model-specific).
    ///   - minimumBuffer: Floor on the returned buffer (defaults to
    ///     `PlaybackStrategy.minimumBufferDuration`; pass
    ///     `speechDecoder.minimumBufferDuration` for a model-specific override).
    /// - Returns: Buffer duration in seconds (≥ `minimumBuffer`).
    public static func requiredBuffer(
        stepTime: TimeInterval,
        maxNewTokens: Int,
        samplesPerFrame: Int,
        sampleRate: Int,
        minimumBuffer: TimeInterval = PlaybackStrategy.minimumBufferDuration
    ) -> TimeInterval {
        let perStep = audioPerStep(samplesPerFrame: samplesPerFrame, sampleRate: sampleRate)
        let speedRatio = perStep / stepTime
        let deficit = max(0.0, 1.0 - speedRatio)
        let maxAudioDuration = Double(maxNewTokens) * perStep
        let deficitBuffer = maxAudioDuration * deficit
        return max(minimumBuffer, deficitBuffer)
    }
}

// MARK: - Generation Options

/// Options that control the speech synthesis pipeline.
///
/// Mirrors `DecodingOptions` in WhisperKit: all fields have sensible defaults so
/// the zero-argument initializer works for most use cases.
public struct GenerationOptions: Codable, Sendable {
    // MARK: - Defaults

    public static let defaultTemperature: Float = 0.9
    public static let defaultTopK: Int = 50
    public static let defaultRepetitionPenalty: Float = 1.05
    public static let defaultMaxNewTokens: Int = 245

    public var temperature: Float
    public var topK: Int
    public var repetitionPenalty: Float
    public var maxNewTokens: Int

    /// Number of concurrent workers for multi-chunk generation.
    /// - `0`: all chunks run concurrently in one batch (default, fastest for non-streaming use cases).
    /// - `1`: sequential - one chunk at a time; required for real-time `play` streaming.
    /// - `N`: at most N chunks run concurrently.
    public var concurrentWorkerCount: Int

    /// How to split long text into chunks. Defaults to `.sentence`.
    /// Set to `.none` to force a single-pass generation without sentence splitting.
    public var chunkingStrategy: TextChunkingStrategy?

    /// Target chunk size in tokens for sentence chunking.
    /// `nil` resolves to `TextChunker.defaultTargetChunkSize` at the call site.
    public var targetChunkSize: Int?

    /// Minimum chunk size in tokens.
    /// `nil` resolves to `TextChunker.defaultMinChunkSize` at the call site.
    public var minChunkSize: Int?

    /// Optional style instruction for controlling speech characteristics
    /// (e.g., `"Very happy"`). Prepended as a text-only user prompt before the main
    /// TTS segment. For Qwen3, this is only supported by the 1.7B model variant.
    public var instruction: String?

    /// Force the legacy `[FloatType]` inference path even on macOS 15+ / iOS 18+.
    /// When `false` (default), the MLTensor path is taken on supported OS versions.
    /// Set to `true` in tests to exercise the pre-macOS-15 code path on current hardware.
    // TODO: Remove forking logic with package with min os version upgrade
    public var forceLegacyEmbedPath: Bool

    public init(
        temperature: Float = GenerationOptions.defaultTemperature,
        topK: Int = GenerationOptions.defaultTopK,
        repetitionPenalty: Float = GenerationOptions.defaultRepetitionPenalty,
        maxNewTokens: Int = GenerationOptions.defaultMaxNewTokens,
        concurrentWorkerCount: Int = 0,
        chunkingStrategy: TextChunkingStrategy? = nil,
        targetChunkSize: Int? = nil,
        minChunkSize: Int? = nil,
        instruction: String? = nil,
        forceLegacyEmbedPath: Bool = false
    ) {
        self.temperature = temperature
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.maxNewTokens = maxNewTokens
        self.concurrentWorkerCount = concurrentWorkerCount
        self.chunkingStrategy = chunkingStrategy
        self.targetChunkSize = targetChunkSize
        self.minChunkSize = minChunkSize
        self.instruction = instruction
        self.forceLegacyEmbedPath = forceLegacyEmbedPath
    }
}

// MARK: - Timings

/// All timing values are stored in seconds, matching `TranscriptionTimings` in WhisperKit.
public struct SpeechTimings: Codable, Sendable {
    // MARK: - Model loading

    public var modelLoading: TimeInterval = 0
    public var tokenizerLoadTime: TimeInterval = 0

    // MARK: - Pipeline phases

    public var tokenize: TimeInterval = 0
    public var prefill: TimeInterval = 0
    public var timeToFirstBuffer: TimeInterval = 0
    public var fullPipeline: TimeInterval = 0

    // MARK: - Generation loop

    /// Total wall-clock time spent in the autoregressive decoding loop.
    /// Mirrors `decodingLoop` in `TranscriptionTimings`.
    public var decodingLoop: TimeInterval = 0

    // MARK: - CodeDecoder

    /// Sum of `model.prediction()` call durations for the main autoregressive decoder.
    /// Mirrors `decodingPredictions` in `TranscriptionTimings`.
    public var decodingPredictions: TimeInterval = 0

    // MARK: - MultiCodeDecoder

    /// Total wall-clock time for MultiCodeDecoder across all steps.
    public var multiCodeDecoder: TimeInterval = 0
    /// Sum of `model.prediction()` calls inside MultiCodeDecoder.
    public var multiCodeDecoderPredictions: TimeInterval = 0
    public var multiCodeDecoderSampling: TimeInterval = 0
    public var multiCodeDecoderEmbedding: TimeInterval = 0
    /// Sum of KV cache update calls inside MultiCodeDecoder.
    /// Mirrors `decodingKvCaching` in `TranscriptionTimings`.
    public var decodingKvCaching: TimeInterval = 0
    public var totalMultiCodeDecoderPredictions: Double = 0

    // MARK: - SpeechDecoder

    /// Total wall-clock time for SpeechDecoder across all steps.
    public var speechDecoder: TimeInterval = 0
    /// Sum of `model.prediction()` calls inside SpeechDecoder.
    public var speechDecoderPredictions: TimeInterval = 0

    // MARK: - Non-model overhead

    /// Outer CodeDecoder KV cache update time.
    public var kvCacheUpdate: TimeInterval = 0
    /// CodeEmbedder lookup time per step.
    public var codeEmbed: TimeInterval = 0
    /// MultiCodeEmbedder + sum for next step.
    public var codecHidden: TimeInterval = 0
    /// TextProjector + combine-embeddings per step.
    public var textProjection: TimeInterval = 0
    /// Outer code-0 sampling time.
    /// Mirrors `decodingSampling` in `TranscriptionTimings`.
    public var decodingSampling: TimeInterval = 0

    // MARK: - Counters

    public var prefillTokens: Double = 0
    /// Total autoregressive steps. Mirrors `totalDecodingLoops` in `TranscriptionTimings`.
    public var totalDecodingLoops: Double = 0
    /// Duration of audio produced (seconds). Mirrors `inputAudioSeconds` in `TranscriptionTimings`.
    public var inputAudioSeconds: TimeInterval = 0

    public init() {}

    // MARK: - Computed properties

    /// Total wall-clock time in `model.prediction()` calls across all three models.
    /// Note: with async parallelism this sum may exceed `decodingLoop` (overlap).
    public var totalPredictions: TimeInterval {
        decodingPredictions + multiCodeDecoderPredictions + speechDecoderPredictions
    }

    /// Intra-step parallel overlap: SpeechDecoder time that ran concurrently with
    /// CodeDecoder within each generation step.
    public var concurrentStepOverlap: TimeInterval {
        max(0, totalPredictions - decodingLoop)
    }

    /// Non-inference overhead on the main path (excludes overlapped SpeechDecoder time).
    public var totalNonPrediction: TimeInterval {
        let mainPathPredictions = decodingPredictions + multiCodeDecoderPredictions
        return decodingLoop - mainPathPredictions
    }

    public var tokensPerSecond: Double {
        fullPipeline > 0 ? totalDecodingLoops / fullPipeline : 0
    }

    public var realTimeFactor: Double {
        inputAudioSeconds > 0 ? fullPipeline / inputAudioSeconds : 0
    }

    public var speedFactor: Double {
        inputAudioSeconds > 0 ? inputAudioSeconds / fullPipeline : 0
    }

    // MARK: - Merge

    /// Accumulate all per-chunk timing fields from `other` into this timing.
    ///
    /// `modelLoading`, `tokenizerLoadTime`, `timeToFirstBuffer`, and `fullPipeline`
    /// are intentionally excluded - callers set those separately on the combined struct.
    public mutating func merge(_ other: SpeechTimings) {
        tokenize += other.tokenize
        prefill += other.prefill
        prefillTokens += other.prefillTokens
        decodingLoop += other.decodingLoop
        decodingPredictions += other.decodingPredictions
        multiCodeDecoder += other.multiCodeDecoder
        multiCodeDecoderPredictions += other.multiCodeDecoderPredictions
        multiCodeDecoderSampling += other.multiCodeDecoderSampling
        multiCodeDecoderEmbedding += other.multiCodeDecoderEmbedding
        decodingKvCaching += other.decodingKvCaching
        totalMultiCodeDecoderPredictions += other.totalMultiCodeDecoderPredictions
        speechDecoder += other.speechDecoder
        speechDecoderPredictions += other.speechDecoderPredictions
        kvCacheUpdate += other.kvCacheUpdate
        codeEmbed += other.codeEmbed
        codecHidden += other.codecHidden
        textProjection += other.textProjection
        decodingSampling += other.decodingSampling
        totalDecodingLoops += other.totalDecodingLoops
    }
}

// MARK: - Result

/// The complete output of a speech synthesis request.
///
/// Mirrors `TranscriptionResult` in WhisperKit: holds the audio samples, timing
/// breakdown, and sample rate. Conforms to `Codable` so results can be serialized
/// for caching or logging.
public struct SpeechResult: Codable, Sendable {
    /// Raw float audio samples (mono PCM).
    public let audio: [Float]

    /// Generation performance timings.
    public let timings: SpeechTimings

    /// Sample rate of the audio in Hz.
    public let sampleRate: Int

    /// Audio duration in seconds.
    public var audioDuration: Double {
        Double(audio.count) / Double(sampleRate)
    }

    public init(audio: [Float], timings: SpeechTimings, sampleRate: Int) {
        self.audio = audio
        self.timings = timings
        self.sampleRate = sampleRate
    }

    /// Log detailed timing breakdown to console, matching WhisperKit's format.
    public func logTimings() {
        let totalLoops = timings.totalDecodingLoops
        let mcdPredRuns = timings.totalMultiCodeDecoderPredictions
        let fullPipelineDuration = max(timings.decodingLoop, timings.fullPipeline) * 1000
        let formatTime = { (duration: TimeInterval, count: Double) in
            Logging.formatTimeWithPercentage(duration, count, fullPipelineDuration)
        }

        Logging.info(
            """
            ---- Speech Timings ----
            Tokenize:                \(formatTime(timings.tokenize, 1))
            Prefill:                 \(formatTime(timings.prefill, timings.prefillTokens))
            All Predictions:         \(formatTime(timings.totalPredictions, totalLoops))
            Non-inference (main):    \(formatTime(timings.totalNonPrediction, totalLoops))
            Concurrent Step Overlap:  \(formatTime(timings.concurrentStepOverlap, totalLoops))
            CodeDecoder:             \(formatTime(timings.decodingPredictions, totalLoops))
            MultiCodeDecoder:        \(formatTime(timings.multiCodeDecoder, totalLoops))
            - Predictions:           \(formatTime(timings.multiCodeDecoderPredictions, mcdPredRuns))
            - Sampling:              \(formatTime(timings.multiCodeDecoderSampling, totalLoops))
            - Embedding:             \(formatTime(timings.multiCodeDecoderEmbedding, totalLoops))
            - KV Caching:            \(formatTime(timings.decodingKvCaching, totalLoops))
            SpeechDecoder:           \(formatTime(timings.speechDecoder, totalLoops))
            - Predictions:           \(formatTime(timings.speechDecoderPredictions, totalLoops))
            KV Cache (outer):        \(formatTime(timings.kvCacheUpdate, totalLoops))
            Code Embed (outer):      \(formatTime(timings.codeEmbed, totalLoops))
            Codec Hidden:            \(formatTime(timings.codecHidden, totalLoops))
            Text Projection:         \(formatTime(timings.textProjection, totalLoops))
            Sampling (outer):        \(formatTime(timings.decodingSampling, totalLoops))
            Decoding Loop:           \(formatTime(timings.decodingLoop, totalLoops))
            -------------------------------
            Model Load Time:               \(String(format: "%.2f", timings.modelLoading)) seconds
            - Tokenizer:                   \(String(format: "%.2f", timings.tokenizerLoadTime)) seconds
            Inference Duration (Global):   \(String(format: "%.2f", timings.fullPipeline)) seconds
            Time to first buffer:          \(String(format: "%.2f", timings.timeToFirstBuffer)) seconds
            Total Steps:                   \(Int(totalLoops))
            Steps per Second:              \(String(format: "%.2f", timings.tokensPerSecond)) steps/s
            Real Time Factor:              \(String(format: "%.3f", timings.realTimeFactor))
            Speed Factor:                  \(String(format: "%.3f", timings.speedFactor))
            Audio Duration:                \(String(format: "%.2f", audioDuration)) seconds
            """
        )
    }
}

// MARK: - Decoder Result Types

/// Result from MultiCodeDecoder.generateMultiCodes
public struct MultiCodeGenerationResult {
    public let codes: [Int32]
    public let timings: SpeechTimings
    /// Pre-computed embeddings of codes 1-14 with position offsets (offsetIndex 0-13).
    /// Produced as a side effect of multi-code generation and reused for codec-hidden
    /// computation, saving 14 embed model calls per step.
    /// The embedding of code15 (offset 14) is not included and must be fetched separately.
    public let offsetCodeEmbeds: [[FloatType]]
    /// MLTensor variant of offsetCodeEmbeds - populated by the async tensor path,
    /// avoiding the [FloatType] -> MLTensor round-trip.
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public var offsetCodeEmbedTensors: [MLTensor]? { _offsetCodeEmbedTensors as? [MLTensor] }

    let _offsetCodeEmbedTensors: Any?

    public init(codes: [Int32], timings: SpeechTimings, offsetCodeEmbeds: [[FloatType]]) {
        self.codes = codes
        self.timings = timings
        self.offsetCodeEmbeds = offsetCodeEmbeds
        self._offsetCodeEmbedTensors = nil
    }

    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public init(codes: [Int32], timings: SpeechTimings, offsetCodeEmbedTensors: [MLTensor]) {
        self.codes = codes
        self.timings = timings
        self.offsetCodeEmbeds = []
        self._offsetCodeEmbedTensors = offsetCodeEmbedTensors
    }
}

/// Result from SpeechDecoder.decodeFrameAsync
public struct SpeechDecoderTimedResult: Sendable {
    public let samples: [Float]
    public let timings: SpeechTimings
}
