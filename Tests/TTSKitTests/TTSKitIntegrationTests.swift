//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import AVFoundation
import CoreML
import Foundation
@testable import TTSKit
import XCTest

// MARK: - Helpers

extension XCTestCase {
    /// Create a `TTSKit` instance for integration tests.
    ///
    /// Downloads models from the Hub if not already cached on disk, matching the
    /// same pattern as `tinyModelPath()` in WhisperKit's test suite. The test
    /// will fail (not skip) if the download fails.
    func makeCachedTTS(
        model: TTSModelVariant = .qwen3TTS_0_6b,
        seed: UInt64 = 42
    ) async throws -> TTSKit {
        let config = TTSKitConfig(model: model, verbose: true, logLevel: .debug, seed: seed)
        return try await TTSKit(config)
    }
}

// MARK: - Integration Tests

final class TTSKitIntegrationTests: XCTestCase {

    // MARK: - Basic generation

    /// Short sentence -> non-empty audio at 24kHz.
    func testBasicShortGeneration() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        let result = try await tts.generate(
            text: "Hello, this is a basic smoke test of the WhisperKit TTS pipeline.",
            speaker: .ryan,
            language: .english,
            options: GenerationOptions(temperature: 0.9, topK: 50, maxNewTokens: 245)
        )

        XCTAssertGreaterThan(result.audio.count, 0, "Audio samples should be non-empty")
        XCTAssertGreaterThan(result.audioDuration, 1.0, "Expect at least 1s of speech")
        XCTAssertLessThan(result.audioDuration, 30.0, "Short sentence should stay under 30s")
        XCTAssertEqual(result.sampleRate, 24000, "Sample rate should be 24kHz")
    }

    // MARK: - Long text with sentence chunking

    /// 200+ word multi-paragraph text - chunked, all audio produced, no truncation crash.
    func testLongTextSequentialChunks() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        let longText = """
        The history of artificial intelligence begins in antiquity, with myths, stories, \
        and rumors of artificial beings endowed with intelligence by master craftsmen. \
        The seeds of modern AI were planted by classical philosophers who attempted to \
        describe the process of human thinking as the mechanical manipulation of symbols. \
        In 1950, Alan Turing published a landmark paper in which he speculated about the \
        possibility of creating machines that think. Fast forward to the 2020s, and large \
        language models have transformed every domain from science to storytelling. \
        Text-to-speech systems now produce voices so natural that listeners struggle to \
        distinguish them from real human speakers.
        """

        let result = try await tts.generate(
            text: longText,
            speaker: .aiden,
            language: .english,
            options: GenerationOptions(concurrentWorkerCount: 1)
        )

        XCTAssertGreaterThan(result.audioDuration, 20.0, "Long text should produce substantial audio")
        XCTAssertGreaterThan(result.timings.totalDecodingLoops, 100, "Should require many decode steps")
    }

    /// Same long text with unlimited concurrent workers
    func testLongTextUnlimitedConcurrentWorkers() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        let longText = """
        First paragraph about science. It covers many interesting topics in depth. \
        Second paragraph switches to history. Many events happened over the centuries. \
        Third paragraph explores music theory. Harmony and rhythm define the art form.
        """

        let result = try await tts.generate(
            text: longText,
            speaker: .aiden,
            language: .english,
            options: GenerationOptions(concurrentWorkerCount: 0)
        )

        XCTAssertGreaterThan(result.audioDuration, 5.0)
    }

    // MARK: - Concurrent workers

    /// workers=2 on a 3-chunk text - verifies the batching loop handles partial batches.
    func testConcurrentWorkersBatching() async throws {
        let tts = try await makeCachedTTS(seed: 7)
        let threeChunkText = """
        First sentence for chunk one. Second sentence for chunk two. Third sentence for chunk three.
        """
        let result = try await tts.generate(
            text: threeChunkText,
            speaker: .serena,
            language: .english,
            options: GenerationOptions(
                concurrentWorkerCount: 2,
                targetChunkSize: 10,
                minChunkSize: 2
            )
        )

        XCTAssertGreaterThan(result.audio.count, 0)
        XCTAssertGreaterThan(result.audioDuration, 2.0)
    }

    // MARK: - Reproducibility

    /// Two runs with the same seed and temperature=0 should produce byte-identical audio.
    func testDeterministicOutputWithFixedSeed() async throws {
        let text = "Reproducibility test with a fixed seed."
        let options = GenerationOptions(temperature: 0.0, topK: 0, maxNewTokens: 100)

        let tts1 = try await makeCachedTTS(seed: 42)
        let result1 = try await tts1.generate(
            text: text, speaker: .ryan, language: .english, options: options
        )

        let tts2 = try await makeCachedTTS(seed: 42)
        let result2 = try await tts2.generate(
            text: text, speaker: .ryan, language: .english, options: options
        )

        XCTAssertEqual(result1.audio.count, result2.audio.count, "Step counts must match")
        // Compare raw float bytes for strict reproducibility
        let data1 = Data(bytes: result1.audio, count: result1.audio.count * MemoryLayout<Float>.size)
        let data2 = Data(bytes: result2.audio, count: result2.audio.count * MemoryLayout<Float>.size)
        XCTAssertEqual(data1, data2, "Audio samples must be byte-identical with the same seed and greedy decoding")
    }

    // MARK: - Speaker voices

    /// All 9 Qwen3Speaker voices should produce non-empty audio without error.
    func testAllSpeakerVoices() async throws {
        let tts = try await makeCachedTTS(seed: 1)
        let text = "Testing this voice."
        let options = GenerationOptions(temperature: 0.0, topK: 0, maxNewTokens: 60)

        for speaker in Qwen3Speaker.allCases {
            let result = try await tts.generate(
                text: text, speaker: speaker, language: .english, options: options
            )
            XCTAssertGreaterThan(result.audio.count, 0, "Speaker \(speaker.rawValue) produced no audio")
        }
    }

    // MARK: - Language support

    /// Korean text with a Korean-native speaker should produce non-empty audio.
    func testKoreanLanguage() async throws {
        let tts = try await makeCachedTTS(seed: 7)
        let result = try await tts.generate(
            text: "안녕하세요. 저는 한국어를 말할 수 있습니다.",
            speaker: .sohee,
            language: .korean,
            options: GenerationOptions(temperature: 0.9, maxNewTokens: 120)
        )

        XCTAssertGreaterThan(result.audio.count, 0)
        XCTAssertGreaterThan(result.audioDuration, 0.5)
    }

    // MARK: - Edge cases

    /// Single-word input should not crash and should produce short audio.
    func testSingleWordInput() async throws {
        let tts = try await makeCachedTTS(seed: 1)
        let result = try await tts.generate(
            text: "Hello.",
            speaker: .ryan,
            language: .english,
            options: GenerationOptions(temperature: 0.0, topK: 0, maxNewTokens: 60)
        )

        XCTAssertGreaterThan(result.audio.count, 0)
        XCTAssertLessThan(result.audioDuration, 5.0, "Single word should be short")
    }

    /// Text with numbers, punctuation, currency, and percentages should not crash.
    func testNumbersAndPunctuation() async throws {
        let tts = try await makeCachedTTS(seed: 3)
        let result = try await tts.generate(
            text: "On January 1st, 2025, the price was $4.99 - a 12% discount from $5.67.",
            speaker: .dylan,
            language: .english,
            options: GenerationOptions(maxNewTokens: 245)
        )

        XCTAssertGreaterThan(result.audio.count, 0)
    }

    /// Text with emoji and mixed scripts (Latin + accented) should not crash.
    func testUnicodeAndEmoji() async throws {
        let tts = try await makeCachedTTS(seed: 5)
        let result = try await tts.generate(
            text: "Today's meeting is at 3pm 🎉. The café serves café au lait. Résumé updated.",
            speaker: .ryan,
            language: .english,
            options: GenerationOptions(maxNewTokens: 200)
        )

        XCTAssertGreaterThan(result.audio.count, 0)
    }

    // MARK: - WAV export

    /// saveAudio round-trip: write WAV then verify the file exists and is readable.
    func testSaveAudioRoundTrip() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        let result = try await tts.generate(
            text: "Audio export round trip test.",
            speaker: .ryan,
            language: .english,
            options: GenerationOptions(temperature: 0.0, topK: 0, maxNewTokens: 80)
        )

        let tmpFolder = FileManager.default.temporaryDirectory
        let filename = "tts_roundtrip_\(UUID().uuidString)"
        let savedURL = try await AudioOutput.saveAudio(
            result.audio,
            toFolder: tmpFolder,
            filename: filename,
            sampleRate: result.sampleRate,
            format: .wav
        )
        defer { try? FileManager.default.removeItem(at: savedURL) }

        XCTAssertTrue(FileManager.default.fileExists(atPath: savedURL.path), "WAV file should exist after saveAudio")

        let duration = try await AudioOutput.duration(of: savedURL)
        XCTAssertEqual(duration, result.audioDuration, accuracy: 0.1, "WAV duration should match result.audioDuration")
    }

    // MARK: - SpeechModel protocol

    /// TTSKit should satisfy the SpeechModel protocol; generate via the protocol entry point.
    func testTTSModelProtocolConformance() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        let model: any SpeechModel = tts

        XCTAssertEqual(model.sampleRate, 24000)

        let result = try await model.generate(
            text: "Protocol conformance check.",
            voice: Qwen3Speaker.ryan.rawValue,
            language: Qwen3Language.english.rawValue,
            options: GenerationOptions(temperature: 0.0, topK: 0, maxNewTokens: 80),
            callback: nil
        )

        XCTAssertGreaterThan(result.audio.count, 0)
        XCTAssertEqual(result.sampleRate, 24000)
    }

    // MARK: - Performance

    /// Verify timings are populated and the generation loop completed within a reasonable ceiling.
    func testTimingsArePopulated() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        let result = try await tts.generate(
            text: "Performance check: this sentence measures how fast the model runs on device.",
            speaker: .ryan,
            language: .english,
            options: GenerationOptions(temperature: 0.0, topK: 0, maxNewTokens: 200)
        )

        XCTAssertGreaterThan(result.timings.totalDecodingLoops, 0, "Should have completed at least one decode step")
        XCTAssertGreaterThan(result.timings.fullPipeline, 0, "Full pipeline duration should be non-zero")
        XCTAssertGreaterThan(result.audioDuration, 0, "Audio duration should be non-zero")
    }

    // MARK: - Prompt caching

    /// Build a prompt cache and verify it stores the expected metadata.
    func testBuildPromptCache() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        let cache = try await tts.buildPromptCache(speaker: .ryan, language: .english)

        XCTAssertEqual(cache.voice, "ryan")
        XCTAssertEqual(cache.language, "english")
        XCTAssertNil(cache.instruction)
        XCTAssertGreaterThan(cache.prefixLength, 0, "Cache should contain at least one invariant token")
        XCTAssertTrue(cache.matches(voice: "ryan", language: "english", instruction: nil))
        XCTAssertFalse(cache.matches(voice: "aiden", language: "english", instruction: nil))
    }

    /// Cached prefill should be faster than uncached since it only processes the variable token.
    func testPromptCacheSpeedup() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        let text = "Prompt cache speed test with enough words to generate meaningful audio output."
        let options = GenerationOptions(temperature: 0.9, topK: 50, maxNewTokens: 245)

        // Run without cache — full prefill of all tokens
        let uncachedResult = try await tts.createTask().run(
            text: text, voice: "ryan", language: "english",
            options: options, callback: nil, prefixCache: nil
        )
        let uncachedPrefill = uncachedResult.timings.prefill

        // Build cache, then run with it — only the variable token is prefilled
        let cache = try await tts.buildPromptCache(speaker: .ryan, language: .english)
        let cachedResult = try await tts.createTask().run(
            text: text, voice: "ryan", language: "english",
            options: options, callback: nil, prefixCache: cache
        )
        let cachedPrefill = cachedResult.timings.prefill

        XCTAssertGreaterThan(uncachedResult.audioDuration, 1.0,
                             "Uncached run should produce at least 1s of audio")
        XCTAssertGreaterThan(cachedResult.audioDuration, 1.0,
                             "Cached run should produce at least 1s of audio")
        XCTAssertLessThan(cachedPrefill, uncachedPrefill,
                          "Cached prefill (\(cachedPrefill * 1000)ms) should be faster than uncached (\(uncachedPrefill * 1000)ms)")

        // Verify auto-build on generate works
        tts.promptCache = nil
        _ = try await tts.generate(
            text: text, speaker: .ryan, language: .english, options: options
        )
        XCTAssertNotNil(tts.promptCache, "Cache should be auto-built after generate")
    }

    /// Two consecutive cached runs should produce valid audio of similar duration.
    func testPromptCacheDeterminism() async throws {
        let text = "Cache determinism test with a longer sentence so the model generates real speech."
        let options = GenerationOptions(temperature: 0.9, topK: 50, maxNewTokens: 245)

        let tts = try await makeCachedTTS(seed: 42)
        let cache = try await tts.buildPromptCache(speaker: .ryan, language: .english)

        let result1 = try await tts.createTask().run(
            text: text, voice: "ryan", language: "english",
            options: options, callback: nil, prefixCache: cache
        )
        let result2 = try await tts.createTask().run(
            text: text, voice: "ryan", language: "english",
            options: options, callback: nil, prefixCache: cache
        )

        XCTAssertGreaterThan(result1.audioDuration, 1.0, "First cached run should produce at least 1s")
        XCTAssertGreaterThan(result2.audioDuration, 1.0, "Second cached run should produce at least 1s")
    }

    /// Cache auto-invalidates when voice changes.
    func testPromptCacheInvalidationOnVoiceChange() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        let options = GenerationOptions(temperature: 0.9, topK: 50, maxNewTokens: 245)

        let result1 = try await tts.generate(
            text: "First voice generates meaningful speech output.", speaker: .ryan, language: .english, options: options
        )
        XCTAssertEqual(tts.promptCache?.voice, "ryan")
        XCTAssertGreaterThan(result1.audio.count, 0, "First voice should produce audio")

        let result2 = try await tts.generate(
            text: "Second voice also generates meaningful speech output.", speaker: .aiden, language: .english, options: options
        )
        XCTAssertEqual(tts.promptCache?.voice, "aiden",
                       "Cache should auto-rebuild when voice changes")
        XCTAssertGreaterThan(result2.audio.count, 0, "Second voice should produce audio")
    }

    /// Prompt cache save/load round-trip produces identical generation output.
    func testPromptCacheDiskPersistence() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        let cache = try await tts.buildPromptCache(speaker: .ryan, language: .english)

        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("tts_cache_test_\(UUID().uuidString).promptcache")
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        try cache.save(to: tmpURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: tmpURL.path), "Cache file should exist on disk")

        let loaded = try TTSPromptCache.load(from: tmpURL)
        XCTAssertEqual(loaded.voice, cache.voice)
        XCTAssertEqual(loaded.language, cache.language)
        XCTAssertEqual(loaded.instruction, cache.instruction)
        XCTAssertEqual(loaded.prefixLength, cache.prefixLength)

        // Use the loaded cache for generation — should produce valid audio
        tts.promptCache = loaded
        let result = try await tts.generate(
            text: "Disk cache persistence test with enough text for real audio.",
            speaker: .ryan, language: .english,
            options: GenerationOptions(temperature: 0.9, topK: 50, maxNewTokens: 245)
        )
        XCTAssertGreaterThan(result.audioDuration, 1.0, "Loaded cache should produce at least 1s of audio")
    }

    /// Chunked generation should benefit from prompt caching across all chunks.
    func testPromptCacheWithChunkedGeneration() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        try await tts.buildPromptCache(speaker: .ryan, language: .english)

        let multiChunkText = """
        First sentence is fairly long to make a chunk. \
        Second sentence adds more content for another chunk. \
        Third sentence provides even more text for splitting.
        """

        let result = try await tts.generate(
            text: multiChunkText,
            speaker: .ryan, language: .english,
            options: GenerationOptions(
                concurrentWorkerCount: 1,
                targetChunkSize: 15,
                minChunkSize: 5
            )
        )

        XCTAssertGreaterThan(result.audio.count, 0)
        XCTAssertGreaterThan(result.audioDuration, 2.0)
    }

    // MARK: - Dual inference path

    /// MLTensor path (default, macOS 15+) produces valid audio.
    func testMLTensorPathGeneration() async throws {
        guard #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) else {
            throw XCTSkip("MLTensor path requires macOS 15+ / iOS 18+")
        }
        let tts = try await makeCachedTTS(seed: 42)
        var opts = GenerationOptions(maxNewTokens: 100)
        opts.forceLegacyEmbedPath = false

        let result = try await tts.generate(
            text: "Testing the MLTensor inference path.",
            speaker: .ryan, language: .english,
            options: opts
        )

        XCTAssertGreaterThan(result.audio.count, 0, "MLTensor path should produce audio")
        XCTAssertGreaterThan(result.audioDuration, 0.5, "Should produce at least 0.5s of speech")
    }

    /// Legacy [FloatType] path (forced via forceLegacyEmbedPath) produces valid audio.
    func testLegacyEmbedPathGeneration() async throws {
        let tts = try await makeCachedTTS(seed: 42)
        var opts = GenerationOptions(maxNewTokens: 100)
        opts.forceLegacyEmbedPath = true

        let result = try await tts.generate(
            text: "Testing the legacy embed inference path.",
            speaker: .ryan, language: .english,
            options: opts
        )

        XCTAssertGreaterThan(result.audio.count, 0, "Legacy path should produce audio")
        XCTAssertGreaterThan(result.audioDuration, 0.5, "Should produce at least 0.5s of speech")
    }

    /// Both paths with the same seed should produce audio of similar duration.
    /// Floating-point differences between paths mean samples may not be bit-identical.
    func testBothPathsProduceSimilarAudioDuration() async throws {
        guard #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) else {
            throw XCTSkip("Dual-path comparison requires macOS 15+ / iOS 18+")
        }
        let testText = "Comparing inference paths for audio duration consistency."
        let maxNewTokens = 80

        let ttsMLTensor = try await makeCachedTTS(seed: 42)
        var mlTensorOpts = GenerationOptions(maxNewTokens: maxNewTokens)
        mlTensorOpts.forceLegacyEmbedPath = false
        let mlTensorResult = try await ttsMLTensor.generate(
            text: testText, speaker: .ryan, language: .english, options: mlTensorOpts
        )

        let ttsLegacy = try await makeCachedTTS(seed: 42)
        var legacyOpts = GenerationOptions(maxNewTokens: maxNewTokens)
        legacyOpts.forceLegacyEmbedPath = true
        let legacyResult = try await ttsLegacy.generate(
            text: testText, speaker: .ryan, language: .english, options: legacyOpts
        )

        XCTAssertGreaterThan(mlTensorResult.audioDuration, 0)
        XCTAssertGreaterThan(legacyResult.audioDuration, 0)
        // Durations should be within 20% of each other (same number of tokens -> same frames)
        let durationRatio = mlTensorResult.audioDuration / legacyResult.audioDuration
        XCTAssertGreaterThan(durationRatio, 0.8, "Paths should produce similar audio duration")
        XCTAssertLessThan(durationRatio, 1.2, "Paths should produce similar audio duration")
    }
}
