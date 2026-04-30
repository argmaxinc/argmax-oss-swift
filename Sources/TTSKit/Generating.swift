//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

// MARK: - SpeechGenerating

/// Protocol for a single-chunk speech synthesis task.
///
/// `TTSKit` creates tasks via `setupGenerateTask(...)` and calls `run(...)` on each one.
/// Each task instance is independent - fresh KV caches, own RNG - so multiple tasks
/// can run concurrently without data races.
///
/// The interface uses plain `String` for `voice` and `language` to stay model-agnostic.
/// Each implementation maps these to its internal representation (e.g. token IDs).
///
/// **Reference implementation:** `Qwen3GenerateTask` in `Sources/TTSKit/Qwen3TTS/Qwen3GenerateTask.swift`.
public protocol SpeechGenerating: Sendable {
    /// Default voice identifier for this model (e.g. `"ryan"` for Qwen3 TTS).
    ///
    /// Returned by `TTSKit.generate()` and `play()` when `voice` is `nil`.
    /// Each model implementation provides its own sensible default.
    var defaultVoice: String { get }

    /// Default language identifier for this model (e.g. `"english"` for Qwen3 TTS).
    ///
    /// Returned by `TTSKit.generate()` and `play()` when `language` is `nil`.
    var defaultLanguage: String { get }

    /// Output sample rate in Hz (e.g. 24000 for Qwen3 TTS).
    var sampleRate: Int { get }

    /// Number of PCM samples produced per decoded frame (e.g. 1920 for Qwen3 TTS).
    var samplesPerFrame: Int { get }

    /// Minimum pre-buffer duration (seconds) in `.auto` playback mode.
    var minimumBufferDuration: TimeInterval { get }

    /// Generate speech for a single text chunk, delivering per-step audio via `callback`.
    ///
    /// - Parameters:
    ///   - text: The text chunk to synthesize.
    ///   - voice: Voice identifier (model-specific, e.g. `Qwen3Speaker.rawValue`).
    ///   - language: Language identifier (model-specific, e.g. `Qwen3Language.rawValue`).
    ///   - options: Generation options (temperature, top-k, chunking, concurrency, etc.)
    ///   - callback: Called on every decoded step with audio samples and running timings.
    ///               `SpeechProgress.stepTime` is non-nil only on the first step, allowing
    ///               adaptive playback buffer configuration before audio starts.
    ///               Return `false` to cancel; `nil` or `true` to continue.
    ///   - prefixCache: Optional cached prefix state to skip invariant prefill tokens.
    /// - Returns: The assembled `SpeechResult` for this text chunk.
    /// - Throws: `TTSError` on generation failure or task cancellation.
    func run(
        text: String,
        voice: String,
        language: String,
        options: GenerationOptions,
        callback: SpeechCallback,
        prefixCache: TTSPromptCache?
    ) async throws -> SpeechResult
}
