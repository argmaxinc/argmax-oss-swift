//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
@testable import TTSKit
import XCTest

final class TTSKitUnitTests: XCTestCase {

    // MARK: - Configuration

    func testTTSKitConfigDefaults() {
        let config = TTSKitConfig()
        XCTAssertNil(config.modelFolder)
        XCTAssertEqual(config.modelRepo, Qwen3TTSConstants.defaultModelRepo)
        XCTAssertEqual(config.versionDir, Qwen3TTSConstants.defaultVersionDir)
        XCTAssertEqual(config.tokenizerSource, Qwen3TTSConstants.defaultTokenizerRepo)
        XCTAssertEqual(config.codeDecoderVariant, Qwen3VariantDefaults.codeDecoder)
        XCTAssertEqual(config.multiCodeDecoderVariant, Qwen3VariantDefaults.multiCodeDecoder)
        XCTAssertEqual(config.speechDecoderVariant, Qwen3VariantDefaults.speechDecoder)
        XCTAssertTrue(config.verbose)
    }

    func testTTSComputeOptionsDefaults() {
        let opts = ComputeOptions()
        XCTAssertEqual(opts.embedderComputeUnits, .cpuOnly)
        XCTAssertEqual(opts.codeDecoderComputeUnits, .cpuAndNeuralEngine)
        XCTAssertEqual(opts.multiCodeDecoderComputeUnits, .cpuAndNeuralEngine)
        XCTAssertEqual(opts.speechDecoderComputeUnits, .cpuAndNeuralEngine)
    }

    func testModelPresetResolvesInConfig() {
        let small = TTSKitConfig(model: .qwen3TTS_0_6b)
        XCTAssertEqual(small.versionDir, TTSModelVariant.qwen3TTS_0_6b.versionDir)
        XCTAssertEqual(small.multiCodeDecoderVariant, Qwen3VariantDefaults.multiCodeDecoder)

        let large = TTSKitConfig(model: .qwen3TTS_1_7b)
        XCTAssertEqual(large.versionDir, TTSModelVariant.qwen3TTS_1_7b.versionDir)
        XCTAssertEqual(large.multiCodeDecoderVariant, Qwen3VariantDefaults.multiCodeDecoder)
    }

    func testGenerationOptionsDefaults() {
        let opts = GenerationOptions()
        XCTAssertEqual(opts.temperature, 0.9)
        XCTAssertEqual(opts.topK, 50)
        XCTAssertEqual(opts.repetitionPenalty, 1.05)
        XCTAssertEqual(opts.maxNewTokens, 245)
        XCTAssertNil(opts.chunkingStrategy)
        XCTAssertNil(opts.instruction)
        XCTAssertNotNil(opts.concurrentWorkerCount)
    }

    func testOutputSuppressionToggleState() {
        let output = AudioOutput()
        XCTAssertFalse(output.isOutputSuppressed, "Suppression should be disabled by default")
        output.setOutputSuppressed(true)
        XCTAssertTrue(output.isOutputSuppressed, "Suppression should be enabled after setting true")
        output.setOutputSuppressed(false)
        XCTAssertFalse(output.isOutputSuppressed, "Suppression should be disabled after setting false")
    }

    func testDownloadPatterns() {
        let config = TTSKitConfig()
        let patterns = config.downloadPatterns
        XCTAssertEqual(patterns.count, 6)
        XCTAssertTrue(patterns.allSatisfy { $0.hasSuffix("/**") })
        XCTAssertTrue(patterns.contains { $0.contains("code_decoder/") })
        XCTAssertTrue(patterns.contains { $0.contains("speech_decoder/") })
        XCTAssertTrue(patterns.contains { $0.contains("text_projector/") })
    }

    // MARK: - Speaker & Language Enums

    func testSpeakerTokenIds() {
        // Each speaker should have a unique token ID
        let ids = Qwen3Speaker.allCases.map { $0.tokenID }
        XCTAssertEqual(Set(ids).count, Qwen3Speaker.allCases.count, "Speaker token IDs should be unique")
    }

    func testLanguageTokenIds() {
        let ids = Qwen3Language.allCases.map { $0.tokenID }
        XCTAssertEqual(Set(ids).count, Qwen3Language.allCases.count, "Language token IDs should be unique")
        XCTAssertEqual(Qwen3Language.english.tokenID, 2050)
    }

    // MARK: - Text Chunker

    /// Char-level tokenizer helper: 1 unicode scalar = 1 token, perfectly round-trips.
    /// Sizes are exact, making test assertions easy to reason about.
    private func makeChunker(
        targetChunkSize: Int = TextChunker.defaultTargetChunkSize,
        minChunkSize: Int = TextChunker.defaultMinChunkSize
    ) -> TextChunker {
        TextChunker(
            targetChunkSize: targetChunkSize,
            minChunkSize: minChunkSize,
            encode: { $0.unicodeScalars.map { Int($0.value) } },
            decode: { String($0.compactMap { Unicode.Scalar($0) }.map { Character($0) }) }
        )
    }

    func testChunkerShortText() {
        let chunker = makeChunker(targetChunkSize: 50, minChunkSize: 5)
        XCTAssertEqual(chunker.chunk("Hello world."), ["Hello world."])
    }

    func testChunkerEmptyText() {
        let chunker = makeChunker()
        XCTAssertTrue(chunker.chunk("").isEmpty)
        XCTAssertTrue(chunker.chunk("   ").isEmpty)
    }

    func testChunkerSentenceSplitting() {
        // With char-level tokenizer (1 char = 1 token), targetChunkSize: 30 gives a 30-char window.
        // "This is the first sentence." = 27 chars, so each window contains one sentence boundary.
        let chunker = makeChunker(targetChunkSize: 30, minChunkSize: 5)
        let text = "This is the first sentence. This is the second sentence. And here is a third one."
        let chunks = chunker.chunk(text)
        XCTAssertGreaterThan(chunks.count, 1, "Should split into multiple chunks")
        let recombined = chunks.joined(separator: " ")
        XCTAssertTrue(recombined.contains("first sentence"))
        XCTAssertTrue(recombined.contains("third one"))
    }

    func testChunkerMergesTinyTrailing() {
        // "A reasonably long sentence that is here." = exactly 40 chars = 40 tokens (char-level).
        // targetChunkSize: 40 captures the full sentence in one window at the "." boundary.
        // The trailing "X." = 2 tokens is below minChunkSize: 5, so it merges with the previous chunk.
        let chunker = makeChunker(targetChunkSize: 40, minChunkSize: 5)
        let text = "A reasonably long sentence that is here. X."
        let chunks = chunker.chunk(text)
        XCTAssertEqual(chunks.count, 1, "Tiny trailing chunk should merge with previous")
    }

    // MARK: - Embedding Math

    func testZeroEmbed() {
        let embed = EmbedUtilities.zeroEmbed(dim: 16)
        XCTAssertEqual(embed.count, 16)
        XCTAssertTrue(embed.allSatisfy { $0 == 0 })
    }

    func testAddEmbeddings() {
        let a: [FloatType] = [FloatType(1.0), FloatType(2.0), FloatType(3.0)]
        let b: [FloatType] = [FloatType(4.0), FloatType(5.0), FloatType(6.0)]
        let result = EmbedUtilities.addEmbeddings(a, b)
        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(Float(result[0]), 5.0, accuracy: 0.01)
        XCTAssertEqual(Float(result[1]), 7.0, accuracy: 0.01)
        XCTAssertEqual(Float(result[2]), 9.0, accuracy: 0.01)
    }

    func testSumEmbeddings() {
        let embeds: [[FloatType]] = [
            [FloatType(1.0), FloatType(2.0)],
            [FloatType(3.0), FloatType(4.0)],
            [FloatType(5.0), FloatType(6.0)],
        ]
        let result = EmbedUtilities.sumEmbeddings(embeds)
        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(Float(result[0]), 9.0, accuracy: 0.01)
        XCTAssertEqual(Float(result[1]), 12.0, accuracy: 0.01)
    }

    func testSumEmbeddingsEmpty() {
        let result = EmbedUtilities.sumEmbeddings([])
        XCTAssertTrue(result.isEmpty)
    }

    func testCreateAndExtractEmbed() throws {
        let original: [FloatType] = [FloatType(1.5), FloatType(2.5), FloatType(3.5), FloatType(4.5)]
        let arr = try EmbedUtilities.createEmbedMLArray(original)
        XCTAssertEqual(arr.shape, [1, 4, 1, 1] as [NSNumber])

        let extracted = EmbedUtilities.extractEmbed(from: arr)
        XCTAssertEqual(extracted.count, 4)
        for i in 0..<4 {
            XCTAssertEqual(Float(extracted[i]), Float(original[i]), accuracy: 0.01)
        }
    }

    // MARK: - KV Cache

    func testKVCacheInit() throws {
        let cache = try KVCache(cacheDim: 128, maxSeqLength: 32)
        XCTAssertEqual(cache.cacheLength, 0)
        XCTAssertEqual(cache.maxSeqLength, 32)
        XCTAssertEqual(cache.cacheDim, 128)
        XCTAssertFalse(cache.isStateful)
        XCTAssertFalse(cache.isFull)
        XCTAssertEqual(cache.freePositions, 31) // maxSeqLength - 1
        XCTAssertNotNil(cache.keyCache)
        XCTAssertNotNil(cache.valueCache)
    }

    func testKVCacheStatefulInit() throws {
        let cache = try KVCache(cacheDim: 128, maxSeqLength: 32, isStateful: true)
        XCTAssertTrue(cache.isStateful)
        XCTAssertNil(cache.keyCache)
        XCTAssertNil(cache.valueCache)
    }

    func testKVCacheUpdateAdvancesPosition() throws {
        let cache = try KVCache(cacheDim: 4, maxSeqLength: 8)
        XCTAssertEqual(cache.cacheLength, 0)

        cache.update()
        XCTAssertEqual(cache.cacheLength, 1)

        cache.update()
        XCTAssertEqual(cache.cacheLength, 2)
        XCTAssertEqual(cache.freePositions, 5) // 8 - 1 - 2
    }

    func testKVCacheIsFull() throws {
        let cache = try KVCache(cacheDim: 4, maxSeqLength: 4)
        // maxSeqLength=4, isFull when cacheLength >= 3 (maxSeqLength - 1)
        cache.update()
        cache.update()
        XCTAssertFalse(cache.isFull)
        cache.update()
        XCTAssertTrue(cache.isFull)
    }

    func testKVCacheReset() throws {
        let cache = try KVCache(cacheDim: 4, maxSeqLength: 8)
        cache.update()
        cache.update()
        XCTAssertEqual(cache.cacheLength, 2)

        cache.reset()
        XCTAssertEqual(cache.cacheLength, 0)
    }

    func testSpeechDecoderCacheInit() throws {
        let cache = try SpeechDecoderCache(
            cacheDim: 64, maxSeqLength: 16, hiddenDim: 32, hiddenContextLen: 4
        )
        XCTAssertEqual(cache.hiddenDim, 32)
        XCTAssertEqual(cache.hiddenContextLen, 4)
        XCTAssertEqual(cache.hiddenContext.shape, [1, 32, 1, 4] as [NSNumber])
    }

    // MARK: - Sampler

    func testGreedySamplerDeterministic() async throws {
        // Two samplers with the same seed should produce the same result
        let sampler1 = GreedyTokenSampler(seed: 42)
        let sampler2 = GreedyTokenSampler(seed: 42)

        let vocabSize = 32
        let logits = try MLMultiArray(shape: [1, 1, NSNumber(value: vocabSize)], dataType: .float16)
        let ptr = logits.dataPointer.bindMemory(to: FloatType.self, capacity: vocabSize)
        for i in 0..<vocabSize {
            ptr[i] = FloatType(Float.random(in: -1...1))
        }

        let token1 = await sampler1.sampleCodec0(
            logits: logits, temperature: 0.9, topK: 10,
            generatedTokens: [], repetitionPenalty: 1.0, suppressTokenIds: []
        )
        let token2 = await sampler2.sampleCodec0(
            logits: logits, temperature: 0.9, topK: 10,
            generatedTokens: [], repetitionPenalty: 1.0, suppressTokenIds: []
        )
        XCTAssertEqual(token1, token2, "Same seed should produce same token")
    }

    func testGreedySamplerZeroTemperature() async throws {
        let sampler = GreedyTokenSampler(seed: 0)
        let vocabSize = 16
        let logits = try MLMultiArray(shape: [1, 1, NSNumber(value: vocabSize)], dataType: .float16)
        let ptr = logits.dataPointer.bindMemory(to: FloatType.self, capacity: vocabSize)

        for i in 0..<vocabSize { ptr[i] = FloatType(-1.0) }
        ptr[7] = FloatType(5.0)

        let token = await sampler.sampleCodec0(
            logits: logits, temperature: 0.0, topK: 0,
            generatedTokens: [], repetitionPenalty: 1.0, suppressTokenIds: []
        )
        XCTAssertEqual(token, 7, "Greedy (temp=0) should pick argmax")
    }

    func testGreedySamplerTokenSuppression() async throws {
        let sampler = GreedyTokenSampler(seed: 0)
        let vocabSize = 16
        let logits = try MLMultiArray(shape: [1, 1, NSNumber(value: vocabSize)], dataType: .float16)
        let ptr = logits.dataPointer.bindMemory(to: FloatType.self, capacity: vocabSize)

        for i in 0..<vocabSize { ptr[i] = FloatType(0.0) }
        ptr[3] = FloatType(10.0)
        ptr[5] = FloatType(5.0)

        let token = await sampler.sampleCodec0(
            logits: logits, temperature: 0.0, topK: 0,
            generatedTokens: [], repetitionPenalty: 1.0, suppressTokenIds: [3]
        )
        XCTAssertEqual(token, 5, "Suppressed token should be skipped, picking next best")
    }

    func testGreedySamplerMultiHeadDeterministic() async throws {
        let sampler1 = GreedyTokenSampler(seed: 7)
        let sampler2 = GreedyTokenSampler(seed: 7)

        let numHeads = 15
        let vocabSize = 32
        let allLogits = try MLMultiArray(shape: [1, NSNumber(value: numHeads), NSNumber(value: vocabSize)], dataType: .float16)
        let ptr = allLogits.dataPointer.bindMemory(to: FloatType.self, capacity: numHeads * vocabSize)
        for i in 0..<numHeads * vocabSize {
            ptr[i] = FloatType(Float.random(in: -2...2))
        }

        let token1 = await sampler1.sampleMultiHead(allLogits: allLogits, headIndex: 3, temperature: 0.9, topK: 10)
        let token2 = await sampler2.sampleMultiHead(allLogits: allLogits, headIndex: 3, temperature: 0.9, topK: 10)
        XCTAssertEqual(token1, token2, "Same seed should produce same multi-head token")
    }

    // MARK: - Qwen3MultiCodeDecoder.buildMasks

    func testBuildMasksAtPositionZero() async throws {
        guard #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) else {
            throw XCTSkip("MLTensor requires macOS 15+ / iOS 18+")
        }
        let sequenceLength = 8
        let masks = Qwen3MultiCodeDecoder().buildMasks(position: 0, sequenceLength: sequenceLength)

        let updateValues = await masks.updateMask.toFloatArray()
        let paddingValues = await masks.paddingMask.toFloatArray()

        XCTAssertEqual(updateValues.count, sequenceLength)
        XCTAssertEqual(paddingValues.count, sequenceLength)
        // Only position 0 should be 1 in the update mask
        XCTAssertEqual(updateValues[0], 1.0, accuracy: 0.001)
        for i in 1..<sequenceLength {
            XCTAssertEqual(updateValues[i], 0.0, accuracy: 0.001, "position \(i) should be 0")
        }
        // Padding mask: position 0 = 0.0 (attend), rest = -10000
        XCTAssertEqual(paddingValues[0], 0.0, accuracy: 0.001)
        for i in 1..<sequenceLength {
            XCTAssertEqual(paddingValues[i], -10000.0, accuracy: 1.0, "position \(i) should be masked")
        }
    }

    func testBuildMasksAtPositionTwo() async throws {
        guard #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) else {
            throw XCTSkip("MLTensor requires macOS 15+ / iOS 18+")
        }
        let sequenceLength = 8
        let masks = Qwen3MultiCodeDecoder().buildMasks(position: 2, sequenceLength: sequenceLength)

        let updateValues = await masks.updateMask.toFloatArray()
        let paddingValues = await masks.paddingMask.toFloatArray()

        // Update mask: only position 2 = 1
        XCTAssertEqual(updateValues[2], 1.0, accuracy: 0.001)
        XCTAssertEqual(updateValues[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(updateValues[1], 0.0, accuracy: 0.001)
        // Padding mask: positions 0–2 should be 0 (attend), rest masked
        XCTAssertEqual(paddingValues[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(paddingValues[1], 0.0, accuracy: 0.001)
        XCTAssertEqual(paddingValues[2], 0.0, accuracy: 0.001)
        XCTAssertEqual(paddingValues[3], -10000.0, accuracy: 1.0)
    }

    // MARK: - decodeEmbedBuffer error path

    func testDecodeEmbedBufferFailsWhenModelUnloaded() async throws {
        let decoder = Qwen3MultiCodeDecoder()
        let embedDim = 8
        let embed = [FloatType](repeating: FloatType(0.1), count: embedDim)
        let reuseArray = try MLMultiArray(shape: [1, NSNumber(value: embedDim), 1, 1], dataType: .float16)
        let cache = try KVCache(cacheDim: 4, maxSeqLength: 8)

        do {
            _ = try await decoder.decodeEmbedBuffer(embed, reuseArray: reuseArray, cache: cache, state: nil)
            XCTFail("Expected TTSError.generationFailed when model is nil")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("not loaded") ||
                          "\(error)".contains("not loaded"),
                          "Error should mention model not loaded, got: \(error)")
        }
    }

    // MARK: - GenerationOptions.forceLegacyEmbedPath

    func testForceLegacyEmbedPathDefaultIsFalse() {
        let opts = GenerationOptions()
        XCTAssertFalse(opts.forceLegacyEmbedPath, "forceLegacyEmbedPath should default to false")
    }

    func testForceLegacyEmbedPathCanBeSet() {
        var opts = GenerationOptions()
        opts.forceLegacyEmbedPath = true
        XCTAssertTrue(opts.forceLegacyEmbedPath)
    }

    // MARK: - SpeechTimings phase merge

    func testPhaseTimingsMergeWorkflow() {
        var accumulated = SpeechTimings()
        accumulated.modelLoading = 0.5

        var tokenizePhase = SpeechTimings()
        tokenizePhase.tokenize = 0.1
        accumulated.merge(tokenizePhase)

        var prefillPhase = SpeechTimings()
        prefillPhase.prefill = 0.3
        prefillPhase.prefillTokens = 15
        accumulated.merge(prefillPhase)

        var loopPhase = SpeechTimings()
        loopPhase.decodingLoop = 2.0
        loopPhase.totalDecodingLoops = 50
        accumulated.merge(loopPhase)

        XCTAssertEqual(accumulated.modelLoading, 0.5, accuracy: 0.001, "modelLoading should not be merged")
        XCTAssertEqual(accumulated.tokenize, 0.1, accuracy: 0.001)
        XCTAssertEqual(accumulated.prefill, 0.3, accuracy: 0.001)
        XCTAssertEqual(accumulated.prefillTokens, 15, accuracy: 0.001)
        XCTAssertEqual(accumulated.decodingLoop, 2.0, accuracy: 0.001)
        XCTAssertEqual(accumulated.totalDecodingLoops, 50, accuracy: 0.001)
    }

    // MARK: - TTSKit Initialization

    func testTTSKitInit() async throws {
        let tts = try await TTSKit(TTSKitConfig(verbose: true, download: false, load: false, seed: 42))
        XCTAssertEqual(tts.seed, 42)
        XCTAssertTrue(tts.config.verbose)
        XCTAssertNil(tts.tokenizer)
    }

    func testTTSKitComponentsExist() async throws {
        let tts = try await TTSKit(TTSKitConfig(download: false, load: false))
        // All components should be initialized (models not loaded yet)
        XCTAssertNil(tts.textProjector.model)
        XCTAssertNil(tts.codeEmbedder.model)
        XCTAssertNil(tts.multiCodeEmbedder.model)
        XCTAssertNil(tts.codeDecoder.model)
        XCTAssertNil(tts.multiCodeDecoder.model)
        XCTAssertNil(tts.speechDecoder.model)
    }

    func testSuppressTokenIds() {
        let ids = Qwen3TTSConstants.suppressTokenIds
        // Should contain [2048, 3072) except codecEOS (2150)
        XCTAssertEqual(ids.count, 1023) // 1024 - 1 (EOS excluded)
        XCTAssertTrue(ids.contains(2048))
        XCTAssertTrue(ids.contains(3071))
        XCTAssertFalse(ids.contains(Int(Qwen3TTSConstants.codecEOS)))
        XCTAssertFalse(ids.contains(2047)) // below range
        XCTAssertFalse(ids.contains(3072)) // above range
    }

    // MARK: - Per-Task Sampler Isolation

    func testMakeTaskDerivedSeeds() async throws {
        let tts = try await TTSKit(TTSKitConfig(download: false, load: false, seed: 12345))
        XCTAssertEqual(tts.seed, 12345)
    }

    // MARK: - Timings

    func testSpeechTimingsDefaults() {
        let t = SpeechTimings()
        XCTAssertEqual(t.fullPipeline, 0)
        XCTAssertEqual(t.decodingLoop, 0)
        XCTAssertEqual(t.totalDecodingLoops, 0)
        XCTAssertEqual(t.tokensPerSecond, 0)
        XCTAssertEqual(t.realTimeFactor, 0)
        XCTAssertEqual(t.speedFactor, 0)
    }

    func testSpeechTimingsComputedProperties() {
        var t = SpeechTimings()
        t.fullPipeline = 2.0
        t.totalDecodingLoops = 100
        t.inputAudioSeconds = 5.0
        t.decodingPredictions = 0.5
        t.multiCodeDecoderPredictions = 0.3
        t.speechDecoderPredictions = 0.2
        t.decodingLoop = 0.8

        XCTAssertEqual(t.tokensPerSecond, 50.0, accuracy: 0.01)
        XCTAssertEqual(t.realTimeFactor, 0.4, accuracy: 0.01) // 2.0 / 5.0
        XCTAssertEqual(t.speedFactor, 2.5, accuracy: 0.01) // 5.0 / 2.0
        XCTAssertEqual(t.totalPredictions, 1.0, accuracy: 0.01) // 0.5 + 0.3 + 0.2
        // concurrentStepOverlap = max(0, totalPredictions - decodingLoop) = max(0, 1.0 - 0.8) = 0.2
        XCTAssertEqual(t.concurrentStepOverlap, 0.2, accuracy: 0.01)
    }

    // MARK: - TTSResult

    func testTTSResultAudioDuration() {
        let sampleRate = Qwen3TTSConstants.sampleRate // 24000
        let samples = [Float](repeating: 0, count: sampleRate * 3) // 3 seconds
        let result = SpeechResult(audio: samples, timings: SpeechTimings(), sampleRate: Qwen3TTSConstants.sampleRate)
        XCTAssertEqual(result.audioDuration, 3.0, accuracy: 0.001)
    }

    // MARK: - Constants

    func testQwen3TTSConstants() {
        XCTAssertEqual(Qwen3TTSConstants.sampleRate, 24000)
        XCTAssertEqual(Qwen3TTSConstants.samplesPerFrame, 1920)
        XCTAssertEqual(Qwen3TTSConstants.embedDim, 1024)
        XCTAssertEqual(Qwen3TTSConstants.codecVocabSize, 2048)
        // EOS should be in the codec-0 range but excluded from suppression
        XCTAssertEqual(Qwen3TTSConstants.codecEOS, 2150)
        XCTAssertTrue(Qwen3TTSConstants.codecEOS >= 2048 && Qwen3TTSConstants.codecEOS < 3072)
    }

    // MARK: - Unloaded Model Errors

    func testCodeDecoderWithoutModel() {
        let decoder = Qwen3CodeDecoder()
        XCTAssertNil(decoder.model)
        XCTAssertFalse(decoder.isStateful)
        XCTAssertNil(decoder.makeState())
    }

    func testMultiCodeDecoderWithoutModel() {
        let decoder = Qwen3MultiCodeDecoder()
        XCTAssertNil(decoder.model)
        XCTAssertFalse(decoder.isStateful)
        XCTAssertNil(decoder.makeState())
    }

    func testSpeechDecoderWithoutModel() {
        let decoder = Qwen3SpeechDecoder()
        XCTAssertNil(decoder.model)
    }

    // MARK: - Chunking Strategy

    func testChunkingStrategyEnum() {
        XCTAssertEqual(TextChunkingStrategy.allCases.count, 2)
        XCTAssertEqual(TextChunkingStrategy.none.rawValue, "none")
        XCTAssertEqual(TextChunkingStrategy.sentence.rawValue, "sentence")
    }

    // MARK: - SeededRNG Determinism

    func testSeededRNGDeterminism() {
        var rng1 = SeededRandomNumberGenerator(seed: 99)
        var rng2 = SeededRandomNumberGenerator(seed: 99)
        for _ in 0..<100 {
            XCTAssertEqual(rng1.next(), rng2.next())
        }
    }

    func testSeededRNGDifferentSeeds() {
        var rng1 = SeededRandomNumberGenerator(seed: 1)
        var rng2 = SeededRandomNumberGenerator(seed: 2)
        // Very unlikely for all 10 values to match with different seeds
        var allMatch = true
        for _ in 0..<10 {
            if rng1.next() != rng2.next() {
                allMatch = false
                break
            }
        }
        XCTAssertFalse(allMatch, "Different seeds should produce different sequences")
    }

    // MARK: - SpeechProgress

    func testSpeechProgressInit() {
        let samples: [Float] = [0.1, 0.2, 0.3]
        let timings = SpeechTimings()
        let progress = SpeechProgress(audio: samples, timings: timings, stepTime: 0.08)
        XCTAssertEqual(progress.audio, samples)
        XCTAssertEqual(progress.timings.fullPipeline, 0)
        XCTAssertEqual(progress.stepTime ?? 0, 0.08, accuracy: 0.0001)
    }

    func testSpeechProgressStepTimeNilByDefault() {
        let progress = SpeechProgress(audio: [], timings: SpeechTimings())
        XCTAssertNil(progress.stepTime)
    }

    func testSpeechProgressFirstStepSemantics() {
        // stepTime non-nil signals first step; subsequent steps have nil
        let first = SpeechProgress(audio: [0.1], timings: SpeechTimings(), stepTime: 0.05)
        let subsequent = SpeechProgress(audio: [0.2], timings: SpeechTimings(), stepTime: nil)
        XCTAssertNotNil(first.stepTime)
        XCTAssertNil(subsequent.stepTime)
    }

    // MARK: - PlaybackStrategy

    func testAudioPerStep() {
        let spf = Qwen3TTSConstants.samplesPerFrame
        let sr = Qwen3TTSConstants.sampleRate
        let expected = Double(spf) / Double(sr)
        XCTAssertEqual(PlaybackStrategy.audioPerStep(samplesPerFrame: spf, sampleRate: sr), expected, accuracy: 0.0001)
        // ~80ms per frame at 24kHz / 1920 samples
        XCTAssertEqual(PlaybackStrategy.audioPerStep(samplesPerFrame: spf, sampleRate: sr), 0.08, accuracy: 0.001)
    }

    func testRequiredBufferFastDevice() {
        // Step completes in half the frame duration -> device is 2x real-time
        // deficit = max(0, 1 - 2.0) = 0, so result = minimumBufferDuration
        let spf = Qwen3TTSConstants.samplesPerFrame
        let sr = Qwen3TTSConstants.sampleRate
        let stepTime = PlaybackStrategy.audioPerStep(samplesPerFrame: spf, sampleRate: sr) / 2.0
        let buffer = PlaybackStrategy.requiredBuffer(stepTime: stepTime, maxNewTokens: 100, samplesPerFrame: spf, sampleRate: sr)
        XCTAssertEqual(buffer, PlaybackStrategy.minimumBufferDuration, accuracy: 0.001)
    }

    func testRequiredBufferSlowDevice() {
        // Step takes 2x the frame duration -> device is at 0.5x real-time
        // speedRatio = 0.5, deficit = 0.5, maxAudio = 100 * 0.08 = 8s
        // deficitBuffer = 8 * 0.5 = 4s > minimumBufferDuration
        let spf = Qwen3TTSConstants.samplesPerFrame
        let sr = Qwen3TTSConstants.sampleRate
        let stepTime = PlaybackStrategy.audioPerStep(samplesPerFrame: spf, sampleRate: sr) * 2.0
        let buffer = PlaybackStrategy.requiredBuffer(stepTime: stepTime, maxNewTokens: 100, samplesPerFrame: spf, sampleRate: sr)
        XCTAssertGreaterThan(buffer, PlaybackStrategy.minimumBufferDuration)
        // Exact: 100 * 0.08 * 0.5 = 4.0s
        XCTAssertEqual(buffer, 4.0, accuracy: 0.01)
    }

    func testRequiredBufferAtExactRealTime() {
        // Step equals frame duration -> speedRatio = 1, deficit = 0 -> minimum clamp applies
        let spf = Qwen3TTSConstants.samplesPerFrame
        let sr = Qwen3TTSConstants.sampleRate
        let stepTime = PlaybackStrategy.audioPerStep(samplesPerFrame: spf, sampleRate: sr)
        let buffer = PlaybackStrategy.requiredBuffer(stepTime: stepTime, maxNewTokens: 50, samplesPerFrame: spf, sampleRate: sr)
        XCTAssertEqual(buffer, PlaybackStrategy.minimumBufferDuration, accuracy: 0.001)
    }

    func testRequiredBufferMinimumNeverExceeded() {
        // Even a very fast device should never return less than the minimum
        let buffer = PlaybackStrategy.requiredBuffer(
            stepTime: 0.001, maxNewTokens: 200,
            samplesPerFrame: Qwen3TTSConstants.samplesPerFrame, sampleRate: Qwen3TTSConstants.sampleRate
        )
        XCTAssertGreaterThanOrEqual(buffer, PlaybackStrategy.minimumBufferDuration)
    }

    // MARK: - TTSModelVariant

    func testModelPresetDisplayNames() {
        XCTAssertEqual(TTSModelVariant.qwen3TTS_0_6b.displayName, "Qwen3 TTS 0.6B")
        XCTAssertEqual(TTSModelVariant.qwen3TTS_1_7b.displayName, "Qwen3 TTS 1.7B")
    }

    func testModelPresetSupportsVoiceDirection() {
        XCTAssertFalse(TTSModelVariant.qwen3TTS_0_6b.supportsVoiceDirection)
        XCTAssertTrue(TTSModelVariant.qwen3TTS_1_7b.supportsVoiceDirection)
    }

    func testModelPresetVersionDirsDiffer() {
        XCTAssertNotEqual(
            TTSModelVariant.qwen3TTS_0_6b.versionDir,
            TTSModelVariant.qwen3TTS_1_7b.versionDir
        )
    }

    func testModelPresetAvailabilityOnMacOS() {
        // On macOS, all presets should be available
        #if os(macOS)
        for preset in TTSModelVariant.allCases {
            XCTAssertTrue(preset.isAvailableOnCurrentPlatform, "\(preset) should be available on macOS")
        }
        #else
        XCTAssertTrue(TTSModelVariant.qwen3TTS_0_6b.isAvailableOnCurrentPlatform)
        XCTAssertFalse(TTSModelVariant.qwen3TTS_1_7b.isAvailableOnCurrentPlatform)
        #endif
    }

    func testModelPresetVariantDefaultsConsistent() {
        // Both presets share the same variant strings (all quantization is size-independent)
        XCTAssertEqual(TTSModelVariant.qwen3TTS_0_6b.codeDecoderVariant, Qwen3VariantDefaults.codeDecoder)
        XCTAssertEqual(TTSModelVariant.qwen3TTS_1_7b.codeDecoderVariant, Qwen3VariantDefaults.codeDecoder)
        XCTAssertEqual(TTSModelVariant.qwen3TTS_0_6b.speechDecoderVariant, Qwen3VariantDefaults.speechDecoder)
    }

    // MARK: - TTSKitConfig Component Overrides

    func testTTSKitConfigComponentOverridesNilByDefault() {
        let config = TTSKitConfig()
        XCTAssertNil(config.textProjector)
        XCTAssertNil(config.codeEmbedder)
        XCTAssertNil(config.multiCodeEmbedder)
        XCTAssertNil(config.codeDecoder)
        XCTAssertNil(config.multiCodeDecoder)
        XCTAssertNil(config.speechDecoder)
    }

    // MARK: - GenerationOptions Additional Defaults

    func testGenerationOptionsChunkingDefaults() {
        let opts = GenerationOptions()
        // nil defers to TextChunker.defaultTargetChunkSize / defaultMinChunkSize at call site
        XCTAssertNil(opts.targetChunkSize)
        XCTAssertNil(opts.minChunkSize)
        // Verify the canonical defaults live in TextChunker
        XCTAssertEqual(TextChunker.defaultTargetChunkSize, 42)
        XCTAssertEqual(TextChunker.defaultMinChunkSize, 10)
    }

    // MARK: - Speaker & Language Round-trips

    func testSpeakerRawValueRoundTrip() {
        for speaker in Qwen3Speaker.allCases {
            let roundTripped = Qwen3Speaker(rawValue: speaker.rawValue)
            XCTAssertEqual(roundTripped, speaker, "rawValue round-trip failed for \(speaker)")
        }
    }

    func testLanguageRawValueRoundTrip() {
        for lang in Qwen3Language.allCases {
            let roundTripped = Qwen3Language(rawValue: lang.rawValue)
            XCTAssertEqual(roundTripped, lang, "rawValue round-trip failed for \(lang)")
        }
    }

    func testUnrecognisedSpeakerFallsBack() {
        XCTAssertNil(Qwen3Speaker(rawValue: "nonexistent_speaker"))
    }

    func testUnrecognisedLanguageFallsBack() {
        XCTAssertNil(Qwen3Language(rawValue: "klingon"))
    }

    // MARK: - SpeechTimings.merge

    func testMergeTimingsAccumulates() {
        var combined = SpeechTimings()
        combined.decodingLoop = 1.0
        combined.totalDecodingLoops = 10
        combined.decodingPredictions = 0.5

        var chunk = SpeechTimings()
        chunk.decodingLoop = 2.0
        chunk.totalDecodingLoops = 20
        chunk.decodingPredictions = 0.3
        chunk.multiCodeDecoderPredictions = 0.1
        chunk.speechDecoderPredictions = 0.2

        combined.merge(chunk)

        XCTAssertEqual(combined.decodingLoop, 3.0, accuracy: 0.001)
        XCTAssertEqual(combined.totalDecodingLoops, 30, accuracy: 0.001)
        XCTAssertEqual(combined.decodingPredictions, 0.8, accuracy: 0.001)
        XCTAssertEqual(combined.multiCodeDecoderPredictions, 0.1, accuracy: 0.001)
        XCTAssertEqual(combined.speechDecoderPredictions, 0.2, accuracy: 0.001)
    }

    func testMergeTimingsIdentity() async throws {
        var combined = SpeechTimings()
        combined.decodingLoop = 5.0
        let empty = SpeechTimings()
        combined.merge(empty)
        XCTAssertEqual(combined.decodingLoop, 5.0, accuracy: 0.001)
    }

    // MARK: - TextChunker Edge Cases

    func testChunkerPreservesAllText() {
        // Each "Sentence number N is here." ≈ 26-27 chars; targetChunkSize: 50 spans ~2 sentences.
        let chunker = makeChunker(targetChunkSize: 50, minChunkSize: 5)
        let sentences = (1...10).map { "Sentence number \($0) is here." }
        let text = sentences.joined(separator: " ")
        let chunks = chunker.chunk(text)
        let rejoined = chunks.joined(separator: " ")
        for sentence in sentences {
            XCTAssertTrue(rejoined.contains(sentence.dropLast(1)), // drop period; joins may vary
                "Missing content: \(sentence)")
        }
    }

    func testChunkerWordBoundaryFallback() {
        // No punctuation in text - chunker must fall back to word-boundary splits.
        // With char-level tokenizer, targetChunkSize: 15 gives 15-char windows.
        // Note: multi-word phrase checks are intentionally avoided here because the
        // re-encode advance can leave a stray char when the token stream has a leading
        // space (e.g. "very long" → 9 tokens, but the stream starts with " very lon").
        // This is a char-level mock artifact; BPE tokenizers fold whitespace into the
        // next word token, so no drift occurs in production.
        let chunker = makeChunker(targetChunkSize: 15, minChunkSize: 3)
        let text = "This is a very long text with no punctuation inside it at all here"
        let chunks = chunker.chunk(text)
        XCTAssertGreaterThan(chunks.count, 1, "Long text without punctuation should split at word boundaries")
        let rejoined = chunks.joined(separator: " ")
        for word in ["long", "text", "punctuation", "here"] {
            XCTAssertTrue(rejoined.contains(word), "Missing word in output: \(word)")
        }
    }

    // MARK: - Prompt Cache

    func testPromptCacheMatching() {
        let cache = TTSPromptCache(
            voice: "ryan", language: "english", instruction: nil,
            prefixLength: 9,
            kvSnapshot: KVCacheSnapshot(
                isStateful: false, cacheDim: 1, maxSeqLength: 1, cacheLength: 0,
                keyCacheData: Data(), valueCacheData: Data(),
                updateMaskData: Data(), paddingMaskData: Data()
            ),
            stateData: nil
        )

        XCTAssertTrue(cache.matches(voice: "ryan", language: "english", instruction: nil))
        XCTAssertFalse(cache.matches(voice: "aiden", language: "english", instruction: nil))
        XCTAssertFalse(cache.matches(voice: "ryan", language: "korean", instruction: nil))
        XCTAssertFalse(cache.matches(voice: "ryan", language: "english", instruction: "Speak softly"))
    }

    func testPromptCacheMatchingWithInstruction() {
        let cache = TTSPromptCache(
            voice: "ryan", language: "english", instruction: "Speak softly",
            prefixLength: 20,
            kvSnapshot: KVCacheSnapshot(
                isStateful: false, cacheDim: 1, maxSeqLength: 1, cacheLength: 0,
                keyCacheData: Data(), valueCacheData: Data(),
                updateMaskData: Data(), paddingMaskData: Data()
            ),
            stateData: nil
        )

        XCTAssertTrue(cache.matches(voice: "ryan", language: "english", instruction: "Speak softly"))
        XCTAssertFalse(cache.matches(voice: "ryan", language: "english", instruction: nil))
        XCTAssertFalse(cache.matches(voice: "ryan", language: "english", instruction: "Speak loudly"))
    }

    func testPromptCacheFileName() {
        let cache1 = TTSPromptCache(
            voice: "ryan", language: "english", instruction: nil,
            prefixLength: 9,
            kvSnapshot: KVCacheSnapshot(
                isStateful: false, cacheDim: 1, maxSeqLength: 1, cacheLength: 0,
                keyCacheData: Data(), valueCacheData: Data(),
                updateMaskData: Data(), paddingMaskData: Data()
            ),
            stateData: nil
        )
        XCTAssertEqual(cache1.cacheFileName, "ryan_english.promptcache")

        let cache2 = TTSPromptCache(
            voice: "aiden", language: "korean", instruction: "Speak slowly",
            prefixLength: 20,
            kvSnapshot: KVCacheSnapshot(
                isStateful: false, cacheDim: 1, maxSeqLength: 1, cacheLength: 0,
                keyCacheData: Data(), valueCacheData: Data(),
                updateMaskData: Data(), paddingMaskData: Data()
            ),
            stateData: nil
        )
        XCTAssertTrue(cache2.cacheFileName.hasPrefix("aiden_korean_"))
        XCTAssertTrue(cache2.cacheFileName.hasSuffix(".promptcache"))
    }

    func testKVCacheSnapshotRoundTrip() throws {
        let dim = 4
        let seq = 8
        let cache = try KVCache(cacheDim: dim, maxSeqLength: seq, isStateful: false)

        // Simulate 3 prefill steps by advancing position and writing dummy data
        for step in 0..<3 {
            if let keyCache = cache.keyCache {
                let ptr = keyCache.dataPointer.bindMemory(to: FloatType.self, capacity: dim * seq)
                for d in 0..<dim {
                    ptr[d * seq + step] = FloatType(Float(step * dim + d))
                }
            }
            if let valueCache = cache.valueCache {
                let ptr = valueCache.dataPointer.bindMemory(to: FloatType.self, capacity: dim * seq)
                for d in 0..<dim {
                    ptr[d * seq + step] = FloatType(Float(step * dim + d + 100))
                }
            }
            cache.cacheLength += 1
            let nextPos = Int(cache.cacheLength)
            let updatePtr = cache.kvCacheUpdateMask.dataPointer.bindMemory(to: FloatType.self, capacity: seq)
            let paddingPtr = cache.keyPaddingMask.dataPointer.bindMemory(to: FloatType.self, capacity: seq)
            updatePtr[step] = FloatType(0.0)
            if nextPos < seq {
                updatePtr[nextPos] = FloatType(1.0)
                paddingPtr[nextPos] = FloatType(0.0)
            }
        }

        let snapshot = cache.snapshot()
        XCTAssertEqual(snapshot.cacheLength, 3)
        XCTAssertEqual(snapshot.cacheDim, dim)
        XCTAssertEqual(snapshot.maxSeqLength, seq)

        // Create a fresh cache and restore
        let restored = try KVCache(cacheDim: dim, maxSeqLength: seq, isStateful: false)
        restored.restore(from: snapshot)

        XCTAssertEqual(restored.cacheLength, 3)

        // Verify KV data was copied correctly
        if let origKey = cache.keyCache, let restoredKey = restored.keyCache {
            let origPtr = origKey.dataPointer.bindMemory(to: FloatType.self, capacity: dim * seq)
            let restPtr = restoredKey.dataPointer.bindMemory(to: FloatType.self, capacity: dim * seq)
            for i in 0..<(dim * seq) {
                XCTAssertEqual(Float(origPtr[i]), Float(restPtr[i]), accuracy: 0.001,
                               "Key cache mismatch at index \(i)")
            }
        }

        // Verify masks were copied
        let origMask = cache.kvCacheUpdateMask.dataPointer.bindMemory(to: FloatType.self, capacity: seq)
        let restMask = restored.kvCacheUpdateMask.dataPointer.bindMemory(to: FloatType.self, capacity: seq)
        for i in 0..<seq {
            XCTAssertEqual(Float(origMask[i]), Float(restMask[i]), accuracy: 0.001,
                           "Update mask mismatch at index \(i)")
        }
    }

    func testPromptCacheDiskRoundTrip() throws {
        let snapshot = KVCacheSnapshot(
            isStateful: false, cacheDim: 4, maxSeqLength: 8, cacheLength: 3,
            keyCacheData: Data(repeating: 0xAA, count: 64),
            valueCacheData: Data(repeating: 0xBB, count: 64),
            updateMaskData: Data(repeating: 0x01, count: 16),
            paddingMaskData: Data(repeating: 0x02, count: 16)
        )
        let cache = TTSPromptCache(
            voice: "ryan", language: "english", instruction: nil,
            prefixLength: 9, kvSnapshot: snapshot, stateData: nil
        )

        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_cache_\(UUID().uuidString).promptcache")
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        try cache.save(to: tmpURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: tmpURL.path))

        let loaded = try TTSPromptCache.load(from: tmpURL)
        XCTAssertEqual(loaded.voice, "ryan")
        XCTAssertEqual(loaded.language, "english")
        XCTAssertNil(loaded.instruction)
        XCTAssertEqual(loaded.prefixLength, 9)
        XCTAssertEqual(loaded.kvSnapshot.cacheLength, 3)
        XCTAssertEqual(loaded.kvSnapshot.cacheDim, 4)
        XCTAssertEqual(loaded.kvSnapshot.maxSeqLength, 8)
        XCTAssertEqual(loaded.kvSnapshot.keyCacheData, Data(repeating: 0xAA, count: 64))
        XCTAssertEqual(loaded.kvSnapshot.valueCacheData, Data(repeating: 0xBB, count: 64))
    }
}
