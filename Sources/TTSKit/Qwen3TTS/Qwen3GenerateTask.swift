//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
import Foundation

// MARK: - Internal phase-result types

/// Output of the tokenize phase - fed into prefill and the generation loop.
struct TokenizeResult {
    let textTokenIds: [Int32]
    let trailingTextTokens: [Int32]
    let firstTextEmbed: [FloatType]
    let variableEmbed: [FloatType]
    let textPadEmbed: [FloatType]
    let timings: SpeechTimings
}

/// Output of the prefill phase - fed into the generation loop.
struct PrefillResult {
    let cdCache: KVCache
    let lastCdOutput: CodeDecoderOutput
    let timings: SpeechTimings
}

/// Output of the autoregressive generation loop.
struct GenerationLoopResult {
    let audio: [Float]
    let steps: Int
    let timings: SpeechTimings
}

// MARK: - Qwen3GenerateTask

/// Qwen3 TTS single-chunk generation task.
///
/// The core building block for Qwen3 TTS generation, analogous to `TranscribeTask`
/// in WhisperKit. Each task creates its own KV caches, MLState, and sampler, then
/// runs a complete prefill + autoregressive decode cycle.
///
/// Conforms to `SpeechGenerating` so it can be returned by
/// `TTSKit.setupGenerateTask(...)` and consumed by `TTSKit`'s generic
/// orchestration layer.
///
/// Thread safety: all stored properties are `let` (immutable after init). Each task
/// owns its own sampler (derived seed) so concurrent tasks don't share RNG state.
/// Model components are shared read-only references - `MLModel.prediction()` is
/// thread-safe. The class is `@unchecked Sendable` to permit `open` subclassing.
open class Qwen3GenerateTask: @unchecked Sendable, SpeechGenerating {
    /// Model components - concrete Qwen3 types for correct async method dispatch.
    /// Using `any Protocol` existentials would cause async extension methods to dispatch
    /// to the protocol default (sync path) instead of the Qwen3-specific MLTensor path.
    public let textProjector: Qwen3TextProjector
    public let codeEmbedder: Qwen3CodeEmbedder
    public let multiCodeEmbedder: Qwen3MultiCodeEmbedder
    public let codeDecoder: Qwen3CodeDecoder
    public let multiCodeDecoder: Qwen3MultiCodeDecoder
    public let speechDecoder: Qwen3SpeechDecoder
    public let sampler: any TokenSampling
    public let tokenizer: any TTSTokenizer
    public let suppressTokenIds: Set<Int>

    /// Timings captured at model-load time (modelLoading + tokenizerLoading populated).
    public let loadTimings: SpeechTimings

    /// Progress object for tracking generation. `totalUnitCount` is set to
    /// `maxNewTokens` at the start of generation; `completedUnitCount` is
    /// updated after each decoding step.
    public let progress: Progress

    // MARK: - Initialization

    public init(
        textProjector: Qwen3TextProjector,
        codeEmbedder: Qwen3CodeEmbedder,
        multiCodeEmbedder: Qwen3MultiCodeEmbedder,
        codeDecoder: Qwen3CodeDecoder,
        multiCodeDecoder: Qwen3MultiCodeDecoder,
        speechDecoder: Qwen3SpeechDecoder,
        sampler: any TokenSampling,
        tokenizer: any TTSTokenizer,
        suppressTokenIds: Set<Int>,
        loadTimings: SpeechTimings = SpeechTimings(),
        progress: Progress? = nil
    ) {
        self.textProjector = textProjector
        self.codeEmbedder = codeEmbedder
        self.multiCodeEmbedder = multiCodeEmbedder
        self.codeDecoder = codeDecoder
        self.multiCodeDecoder = multiCodeDecoder
        self.speechDecoder = speechDecoder
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.suppressTokenIds = suppressTokenIds
        self.loadTimings = loadTimings
        self.progress = progress ?? Progress()
    }

    // MARK: - SpeechGenerating defaults

    /// Default voice for Qwen3 TTS. Matches `Qwen3Speaker.ryan`.
    public var defaultVoice: String { Qwen3Speaker.ryan.rawValue }

    /// Default language for Qwen3 TTS. Matches `Qwen3Language.english`.
    public var defaultLanguage: String { Qwen3Language.english.rawValue }

    // MARK: - Audio format (forwarded from speechDecoder)

    public var sampleRate: Int { speechDecoder.sampleRate }
    public var samplesPerFrame: Int { speechDecoder.samplesPerFrame }
    public var minimumBufferDuration: TimeInterval { speechDecoder.minimumBufferDuration }

    // MARK: - Run

    /// Generate speech for a single text segment.
    ///
    /// Creates fresh KV caches, runs prefill, then autoregressive generation with
    /// interleaved SpeechDecoder audio output. Safe to call concurrently from
    /// multiple tasks against the same model instances.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voice: Raw string matching `Qwen3Speaker.rawValue`; falls back to `.ryan`.
    ///   - language: Raw string matching `Qwen3Language.rawValue`; falls back to `.english`.
    ///   - options: Generation options (temperature, top-k, etc.)
    ///   - callback: Per-step callback receiving decoded audio and running timings.
    ///               `TTSProgress.stepTime` is non-nil only on the first step.
    ///               Return `false` to cancel; `nil` or `true` to continue.
    ///   - prefixCache: Optional cached prefix state to skip invariant prefill tokens.
    /// - Returns: A `SpeechResult` containing the complete audio and timings for this chunk.
    /// - Throws: `TTSError` on generation failure or task cancellation.
    open func run(
        text: String,
        voice: String,
        language: String,
        options: GenerationOptions,
        callback: SpeechCallback,
        prefixCache: TTSPromptCache? = nil
    ) async throws -> SpeechResult {
        let qwen3Speaker = Qwen3Speaker(rawValue: voice) ?? .ryan
        let lang = Qwen3Language(rawValue: language) ?? .english

        var timings = loadTimings
        let pipelineStart = CFAbsoluteTimeGetCurrent()

        progress.totalUnitCount = Int64(options.maxNewTokens)
        progress.completedUnitCount = 0

        // Create task-local MLState for stateful decoders (nil for non-stateful)
        let cdState = codeDecoder.makeState()

        // Phase 1: Tokenize text and build initial embeddings
        let tokenizeResult = try await tokenizeAndBuildEmbeds(text: text)
        timings.merge(tokenizeResult.timings)

        // Phase 2: Prefill the CodeDecoder with the prompt prefix
        let prefillResult = try await prefillCodeDecoder(
            tokenizeResult: tokenizeResult,
            speaker: qwen3Speaker, lang: lang,
            options: options,
            prefixCache: prefixCache,
            voice: voice, language: language,
            cdState: cdState
        )
        timings.merge(prefillResult.timings)

        // Phase 3: Autoregressive RVQ generation with interleaved audio decode
        let loopResult = try await runGenerationLoop(
            tokenizeResult: tokenizeResult,
            prefillResult: prefillResult,
            cdState: cdState,
            options: options,
            pipelineStart: pipelineStart,
            callback: callback,
            baseTimings: timings
        )
        timings.merge(loopResult.timings)
        timings.timeToFirstBuffer = loopResult.timings.timeToFirstBuffer

        timings.fullPipeline = CFAbsoluteTimeGetCurrent() - pipelineStart
        timings.inputAudioSeconds = Double(loopResult.audio.count) / Double(speechDecoder.sampleRate)

        progress.completedUnitCount = progress.totalUnitCount

        let genMs = timings.decodingLoop * 1000
        let avgMs = loopResult.steps > 0 ? genMs / Double(loopResult.steps) : 0
        let stepsPerSec = loopResult.steps > 0 ? Double(loopResult.steps) / timings.decodingLoop : 0
        Logging.info(
            String(
                format: "Generation: %d frames in %.1fms (%.1fms/step, %.1f frames/s)",
                loopResult.steps, genMs, avgMs, stepsPerSec
            ))

        return SpeechResult(audio: loopResult.audio, timings: timings, sampleRate: speechDecoder.sampleRate)
    }

    // MARK: - Phase 1: Tokenize

    /// Tokenize `text` and pre-compute the initial embeddings needed for prefill and decoding.
    private func tokenizeAndBuildEmbeds(
        text: String
    ) async throws -> TokenizeResult {
        let start = CFAbsoluteTimeGetCurrent()

        let textTokenIds = tokenizer.encode(text: text).map { Int32($0) }
        guard !textTokenIds.isEmpty else { throw TTSError.emptyText }

        let firstTextEmbed = try await textProjector.project(tokenId: textTokenIds[0])
        let codecBOSEmbed = try await codeEmbedder.embed(tokenId: Qwen3TTSConstants.codecBOS)
        let variableEmbed = EmbedUtilities.addEmbeddings(firstTextEmbed, codecBOSEmbed)
        let textPadEmbed = try await textProjector.project(tokenId: Qwen3TTSConstants.textPAD)

        var phaseTimings = SpeechTimings()
        phaseTimings.tokenize = CFAbsoluteTimeGetCurrent() - start

        return TokenizeResult(
            textTokenIds: textTokenIds,
            trailingTextTokens: Array(textTokenIds.dropFirst()),
            firstTextEmbed: firstTextEmbed,
            variableEmbed: variableEmbed,
            textPadEmbed: textPadEmbed,
            timings: phaseTimings
        )
    }

    // MARK: - Phase 2: Prefill

    /// Prefill the CodeDecoder KV cache with the invariant prompt prefix.
    ///
    /// If `prefixCache` matches the current voice/language/instruction, restores the
    /// cached state and only decodes the variable token. Otherwise runs a full prefill.
    private func prefillCodeDecoder(
        tokenizeResult: TokenizeResult,
        speaker: Qwen3Speaker,
        lang: Qwen3Language,
        options: GenerationOptions,
        prefixCache: TTSPromptCache?,
        voice: String,
        language: String,
        cdState: Any?
    ) async throws -> PrefillResult {
        let start = CFAbsoluteTimeGetCurrent()

        let cdCache = try KVCache(
            cacheDim: codeDecoder.kvCacheEmbedDim,
            maxSeqLength: codeDecoder.kvCacheMaxSequenceLength,
            isStateful: codeDecoder.isStateful
        )

        let usedCache = prefixCache?.matches(voice: voice, language: language, instruction: options.instruction) == true
        var totalPrefillTokens: Int
        var lastCdOutput: CodeDecoderOutput?

        if usedCache, let prefixCache {
            cdCache.restore(from: prefixCache.kvSnapshot)
            if let stateData = prefixCache.stateData {
                if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), let mlState = cdState as? MLState {
                    mlState.restore(from: stateData)
                }
            }
            totalPrefillTokens = prefixCache.prefixLength + 1

            // TODO: Remove forking logic with package with min os version upgrade
            if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), !options.forceLegacyEmbedPath {
                lastCdOutput = try await codeDecoder.decode(
                    inputEmbeds: tokenizeResult.variableEmbed.asMLTensor(), cache: cdCache, state: cdState
                )
            } else {
                let embedArr = try EmbedUtilities.createEmbedMLArray(tokenizeResult.variableEmbed)
                lastCdOutput = try await codeDecoder.decode(inputEmbeds: embedArr, cache: cdCache, state: cdState)
            }
        } else {
            let embedDim = codeDecoder.embedSize
            let combinedEmbeds = try await buildCombinedEmbeddings(
                speaker: speaker, lang: lang,
                instruction: options.instruction,
                firstTextEmbed: tokenizeResult.firstTextEmbed,
                embedDim: embedDim
            )
            totalPrefillTokens = combinedEmbeds.count

            // TODO: Remove forking logic with package with min os version upgrade
            if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), !options.forceLegacyEmbedPath {
                for embed in combinedEmbeds {
                    lastCdOutput = try await codeDecoder.decode(inputEmbeds: embed.asMLTensor(), cache: cdCache, state: cdState)
                }
            } else {
                for (embedIndex, embed) in combinedEmbeds.enumerated() {
                    if embedIndex > 0, let keyUpdates = lastCdOutput?.keyCacheUpdates, let valueUpdates = lastCdOutput?.valueCacheUpdates {
                        cdCache.update(keyCacheUpdates: keyUpdates, valueCacheUpdates: valueUpdates)
                    }
                    let embedArr = try EmbedUtilities.createEmbedMLArray(embed)
                    lastCdOutput = try await codeDecoder.decode(inputEmbeds: embedArr, cache: cdCache, state: cdState)
                }
            }
        }

        guard let resolvedLastCdOutput = lastCdOutput else {
            throw TTSError.generationFailed("Prefill produced no decoder output")
        }

        var phaseTimings = SpeechTimings()
        phaseTimings.prefill = CFAbsoluteTimeGetCurrent() - start
        phaseTimings.prefillTokens = Double(totalPrefillTokens)

        let prefillMs = phaseTimings.prefill * 1000
        let prefillTokS = phaseTimings.prefill > 0 ? Double(totalPrefillTokens) / phaseTimings.prefill : 0
        let cacheTag = usedCache ? " (cache hit, restored \(prefixCache?.prefixLength ?? 0) tokens)" : ""
        Logging.info(
            String(
                format: "Prefill: %.1fms (%d tokens, %.1f tok/s)%@",
                prefillMs, totalPrefillTokens, prefillTokS, cacheTag
            ))

        return PrefillResult(cdCache: cdCache, lastCdOutput: resolvedLastCdOutput, timings: phaseTimings)
    }

    // MARK: - Phase 3: Generation loop

    /// Run the autoregressive RVQ generation loop, delivering audio frames via `callback`.
    ///
    /// `baseTimings` carries the tokenize + prefill phase timings and is used to build
    /// accurate cumulative `SpeechProgress` values for callbacks.
    /// Returns the assembled audio, the number of steps completed, and the loop-phase timings.
    private func runGenerationLoop(
        tokenizeResult: TokenizeResult,
        prefillResult: PrefillResult,
        cdState: Any?,
        options: GenerationOptions,
        pipelineStart: CFAbsoluteTime,
        callback: SpeechCallback,
        baseTimings: SpeechTimings
    ) async throws -> GenerationLoopResult {
        let cdCache = prefillResult.cdCache
        var lastCdOutput = prefillResult.lastCdOutput
        var timings = baseTimings

        let roleTokenIds = tokenizer.encode(text: "<|im_start|>assistant\n").map { Int32($0) }
        let maxStepsByPrefill = 8 * (roleTokenIds.count + tokenizeResult.textTokenIds.count)

        let sdCache = try SpeechDecoderCache(
            cacheDim: speechDecoder.kvCacheEmbedDim,
            maxSeqLength: speechDecoder.kvCacheMaxSequenceLength,
            hiddenDim: speechDecoder.hiddenDim,
            hiddenContextLen: speechDecoder.hiddenContextLen
        )

        var generatedTokens: [Int32] = []
        var code0 = await sampler.sampleCodec0(
            logits: lastCdOutput.logits,
            temperature: options.temperature, topK: options.topK,
            generatedTokens: generatedTokens,
            repetitionPenalty: options.repetitionPenalty,
            suppressTokenIds: suppressTokenIds
        )
        generatedTokens.append(code0)

        var allAudio: [Float] = []
        var stepIndex = 0
        var firstBufferEmitted = false

        // TODO: Remove forking logic with package with min os version upgrade
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), !options.forceLegacyEmbedPath {
            let textPadEmbedTensor: MLTensor = try await textProjector.project(tokenId: Qwen3TTSConstants.textPAD)

            while code0 != Qwen3TTSConstants.codecEOS
                && !cdCache.isFull
                && stepIndex < options.maxNewTokens
                && stepIndex < maxStepsByPrefill
            {
                try Task.checkCancellation()
                let stepStart = CFAbsoluteTimeGetCurrent()

                let codeEmbedStart = CFAbsoluteTimeGetCurrent()
                let code0EmbedTensor: MLTensor = try await codeEmbedder.embed(tokenId: code0)
                timings.codeEmbed += CFAbsoluteTimeGetCurrent() - codeEmbedStart

                let mcdStart = CFAbsoluteTimeGetCurrent()
                guard let hiddenStatesTensor = lastCdOutput.hiddenStates as? MLTensor else {
                    throw TTSError.generationFailed("Expected MLTensor hidden states on async path")
                }
                let mcdResult = try await multiCodeDecoder.generateMultiCodes(
                    hiddenStatesTensor: hiddenStatesTensor,
                    code0EmbedTensor: code0EmbedTensor,
                    multiCodeEmbedder: multiCodeEmbedder,
                    sampler: sampler, options: options
                )
                timings.multiCodeDecoder += CFAbsoluteTimeGetCurrent() - mcdStart
                timings.multiCodeDecoderPredictions += mcdResult.timings.multiCodeDecoderPredictions
                timings.multiCodeDecoderSampling += mcdResult.timings.multiCodeDecoderSampling
                timings.multiCodeDecoderEmbedding += mcdResult.timings.multiCodeDecoderEmbedding
                timings.decodingKvCaching += mcdResult.timings.decodingKvCaching
                timings.totalMultiCodeDecoderPredictions += mcdResult.timings.totalMultiCodeDecoderPredictions

                let rvqFrame = [code0] + mcdResult.codes

                if !firstBufferEmitted {
                    // First step: give SpeechDecoder exclusive compute access for minimum TTFB.
                    // Emit the buffer immediately, then do the remaining step work (codec hidden,
                    // text projection, CodeDecoder) which only affects the *next* step's readiness.
                    let sdResult = try await speechDecoder.decodeFrameAsync(codes: rvqFrame, cache: sdCache)
                    let stepTime = CFAbsoluteTimeGetCurrent() - stepStart
                    timings.speechDecoderPredictions += sdResult.timings.speechDecoderPredictions
                    timings.speechDecoder += sdResult.timings.speechDecoderPredictions
                    allAudio.append(contentsOf: sdResult.samples)

                    timings.timeToFirstBuffer = CFAbsoluteTimeGetCurrent() - pipelineStart
                    firstBufferEmitted = true

                    if callback?(
                        SpeechProgress(
                            audio: sdResult.samples,
                            timings: timings,
                            stepTime: stepTime
                        )
                    ) == false {
                        break
                    }

                    let codecHiddenStart = CFAbsoluteTimeGetCurrent()
                    guard let lastMcdCode = mcdResult.codes.last else {
                        throw TTSError.generationFailed("Multi-code generation result has no codes")
                    }
                    let code15OffsetId = lastMcdCode + Int32(multiCodeDecoder.codecVocabSize * 14)
                    let code15EmbedTensor: MLTensor = try await multiCodeEmbedder.embed(tokenId: code15OffsetId)
                    var allCodeEmbedTensors: [MLTensor] = [code0EmbedTensor]
                    if let tensorEmbeds = mcdResult.offsetCodeEmbedTensors {
                        allCodeEmbedTensors += tensorEmbeds
                    } else {
                        allCodeEmbedTensors += mcdResult.offsetCodeEmbeds.map { $0.asMLTensor() }
                    }
                    allCodeEmbedTensors.append(code15EmbedTensor)
                    let codecHiddenTensor = EmbedUtilities.sumEmbeddings(allCodeEmbedTensors)
                    timings.codecHidden += CFAbsoluteTimeGetCurrent() - codecHiddenStart

                    let textProjStart = CFAbsoluteTimeGetCurrent()
                    let textEmbedTensor: MLTensor =
                        stepIndex < tokenizeResult.trailingTextTokens.count
                        ? try await textProjector.project(tokenId: tokenizeResult.trailingTextTokens[stepIndex])
                        : textPadEmbedTensor
                    let combinedTensor = EmbedUtilities.addEmbeddings(codecHiddenTensor, textEmbedTensor)
                    timings.textProjection += CFAbsoluteTimeGetCurrent() - textProjStart

                    let decodingStart = CFAbsoluteTimeGetCurrent()
                    lastCdOutput = try await codeDecoder.decode(inputEmbeds: combinedTensor, cache: cdCache, state: cdState)
                    timings.decodingPredictions += CFAbsoluteTimeGetCurrent() - decodingStart - lastCdOutput.internalCacheUpdateTime
                    timings.kvCacheUpdate += lastCdOutput.internalCacheUpdateTime
                } else {
                    // Subsequent steps: overlap SpeechDecoder with CodeDecoder for throughput
                    async let speechResult = speechDecoder.decodeFrameAsync(codes: rvqFrame, cache: sdCache)

                    let codecHiddenStart = CFAbsoluteTimeGetCurrent()
                    guard let lastMcdCode = mcdResult.codes.last else {
                        throw TTSError.generationFailed("Multi-code generation result has no codes")
                    }
                    let code15OffsetId = lastMcdCode + Int32(multiCodeDecoder.codecVocabSize * 14)
                    let code15EmbedTensor: MLTensor = try await multiCodeEmbedder.embed(tokenId: code15OffsetId)
                    var allCodeEmbedTensors: [MLTensor] = [code0EmbedTensor]
                    if let tensorEmbeds = mcdResult.offsetCodeEmbedTensors {
                        allCodeEmbedTensors += tensorEmbeds
                    } else {
                        allCodeEmbedTensors += mcdResult.offsetCodeEmbeds.map { $0.asMLTensor() }
                    }
                    allCodeEmbedTensors.append(code15EmbedTensor)
                    let codecHiddenTensor = EmbedUtilities.sumEmbeddings(allCodeEmbedTensors)
                    timings.codecHidden += CFAbsoluteTimeGetCurrent() - codecHiddenStart

                    let textProjStart = CFAbsoluteTimeGetCurrent()
                    let textEmbedTensor: MLTensor =
                        stepIndex < tokenizeResult.trailingTextTokens.count
                        ? try await textProjector.project(tokenId: tokenizeResult.trailingTextTokens[stepIndex])
                        : textPadEmbedTensor
                    let combinedTensor = EmbedUtilities.addEmbeddings(codecHiddenTensor, textEmbedTensor)
                    timings.textProjection += CFAbsoluteTimeGetCurrent() - textProjStart

                    let decodingStart = CFAbsoluteTimeGetCurrent()
                    lastCdOutput = try await codeDecoder.decode(inputEmbeds: combinedTensor, cache: cdCache, state: cdState)
                    timings.decodingPredictions += CFAbsoluteTimeGetCurrent() - decodingStart - lastCdOutput.internalCacheUpdateTime
                    timings.kvCacheUpdate += lastCdOutput.internalCacheUpdateTime

                    let sdResult = try await speechResult
                    timings.speechDecoderPredictions += sdResult.timings.speechDecoderPredictions
                    timings.speechDecoder += sdResult.timings.speechDecoderPredictions
                    allAudio.append(contentsOf: sdResult.samples)

                    if callback?(
                        SpeechProgress(
                            audio: sdResult.samples,
                            timings: {
                                var merged = baseTimings; merged.merge(timings); return merged
                            }(), stepTime: nil)) == false
                    {
                        break
                    }
                }

                let samplingStart = CFAbsoluteTimeGetCurrent()
                code0 = await sampler.sampleCodec0(
                    logits: lastCdOutput.logits,
                    temperature: options.temperature, topK: options.topK,
                    generatedTokens: generatedTokens,
                    repetitionPenalty: options.repetitionPenalty,
                    suppressTokenIds: suppressTokenIds
                )
                generatedTokens.append(code0)
                timings.decodingSampling += CFAbsoluteTimeGetCurrent() - samplingStart

                timings.decodingLoop += CFAbsoluteTimeGetCurrent() - stepStart
                stepIndex += 1
                progress.completedUnitCount = Int64(stepIndex)

                if stepIndex == 1 || stepIndex % 10 == 0 {
                    let stepMs = (CFAbsoluteTimeGetCurrent() - stepStart) * 1000
                    Logging.debug(
                        String(
                            format: "  Step %d: %.1fms (avg %.1fms/step)",
                            stepIndex, stepMs, timings.decodingLoop * 1000 / Double(stepIndex)))
                }
            }
        } else {
            // Legacy path (older OS)
            while code0 != Qwen3TTSConstants.codecEOS && !cdCache.isFull && stepIndex < options.maxNewTokens && stepIndex < maxStepsByPrefill {
                try Task.checkCancellation()
                let stepStart = CFAbsoluteTimeGetCurrent()

                let cacheUpdateStart = CFAbsoluteTimeGetCurrent()
                if let keyUpdates = lastCdOutput.keyCacheUpdates, let valueUpdates = lastCdOutput.valueCacheUpdates {
                    cdCache.update(keyCacheUpdates: keyUpdates, valueCacheUpdates: valueUpdates)
                }
                timings.kvCacheUpdate += CFAbsoluteTimeGetCurrent() - cacheUpdateStart

                let codeEmbedStart = CFAbsoluteTimeGetCurrent()
                let code0Embed = try await codeEmbedder.embed(tokenId: code0)
                timings.codeEmbed += CFAbsoluteTimeGetCurrent() - codeEmbedStart

                let mcdStart = CFAbsoluteTimeGetCurrent()
                guard let hiddenStates = lastCdOutput.hiddenStates as? [FloatType] else {
                    throw TTSError.generationFailed("Expected [FloatType] hidden states on legacy path")
                }
                let mcdResult = try await multiCodeDecoder.generateMultiCodes(
                    hiddenStates: hiddenStates, code0Embed: code0Embed,
                    multiCodeEmbedder: multiCodeEmbedder, sampler: sampler, options: options
                )
                timings.multiCodeDecoder += CFAbsoluteTimeGetCurrent() - mcdStart
                timings.multiCodeDecoderPredictions += mcdResult.timings.multiCodeDecoderPredictions
                timings.multiCodeDecoderSampling += mcdResult.timings.multiCodeDecoderSampling
                timings.multiCodeDecoderEmbedding += mcdResult.timings.multiCodeDecoderEmbedding
                timings.decodingKvCaching += mcdResult.timings.decodingKvCaching
                timings.totalMultiCodeDecoderPredictions += mcdResult.timings.totalMultiCodeDecoderPredictions

                let rvqFrame = [code0] + mcdResult.codes

                if !firstBufferEmitted {
                    let sdResult = try await speechDecoder.decodeFrameAsync(codes: rvqFrame, cache: sdCache)
                    timings.speechDecoderPredictions += sdResult.timings.speechDecoderPredictions
                    timings.speechDecoder += sdResult.timings.speechDecoderPredictions
                    allAudio.append(contentsOf: sdResult.samples)

                    timings.timeToFirstBuffer = CFAbsoluteTimeGetCurrent() - pipelineStart
                    firstBufferEmitted = true
                    let stepTime = CFAbsoluteTimeGetCurrent() - stepStart

                    if callback?(
                        SpeechProgress(
                            audio: sdResult.samples,
                            timings: {
                                var merged = baseTimings; merged.merge(timings); return merged
                            }(), stepTime: stepTime)) == false
                    {
                        break
                    }

                    let codecHiddenStart = CFAbsoluteTimeGetCurrent()
                    guard let lastMcdCode = mcdResult.codes.last else {
                        throw TTSError.generationFailed("Multi-code generation result has no codes")
                    }
                    let code15OffsetId = lastMcdCode + Int32(multiCodeDecoder.codecVocabSize * 14)
                    var allCodeEmbeds: [[FloatType]] = [code0Embed]
                    allCodeEmbeds += mcdResult.offsetCodeEmbeds
                    try await allCodeEmbeds.append(multiCodeEmbedder.embed(tokenId: code15OffsetId))
                    let codecHidden = EmbedUtilities.sumEmbeddings(allCodeEmbeds)
                    timings.codecHidden += CFAbsoluteTimeGetCurrent() - codecHiddenStart

                    let textProjStart = CFAbsoluteTimeGetCurrent()
                    let textTokenEmbed: [FloatType] =
                        stepIndex < tokenizeResult.trailingTextTokens.count
                        ? try await textProjector.project(tokenId: tokenizeResult.trailingTextTokens[stepIndex])
                        : tokenizeResult.textPadEmbed
                    let combinedArr = try EmbedUtilities.createEmbedMLArray(EmbedUtilities.addEmbeddings(codecHidden, textTokenEmbed))
                    timings.textProjection += CFAbsoluteTimeGetCurrent() - textProjStart

                    let decodingStart = CFAbsoluteTimeGetCurrent()
                    lastCdOutput = try await codeDecoder.decode(inputEmbeds: combinedArr, cache: cdCache, state: cdState)
                    timings.decodingPredictions += CFAbsoluteTimeGetCurrent() - decodingStart
                } else {
                    async let speechResult = speechDecoder.decodeFrameAsync(codes: rvqFrame, cache: sdCache)

                    let codecHiddenStart = CFAbsoluteTimeGetCurrent()
                    guard let lastMcdCode = mcdResult.codes.last else {
                        throw TTSError.generationFailed("Multi-code generation result has no codes")
                    }
                    let code15OffsetId = lastMcdCode + Int32(multiCodeDecoder.codecVocabSize * 14)
                    var allCodeEmbeds: [[FloatType]] = [code0Embed]
                    allCodeEmbeds += mcdResult.offsetCodeEmbeds
                    try await allCodeEmbeds.append(multiCodeEmbedder.embed(tokenId: code15OffsetId))
                    let codecHidden = EmbedUtilities.sumEmbeddings(allCodeEmbeds)
                    timings.codecHidden += CFAbsoluteTimeGetCurrent() - codecHiddenStart

                    let textProjStart = CFAbsoluteTimeGetCurrent()
                    let textTokenEmbed: [FloatType] =
                        stepIndex < tokenizeResult.trailingTextTokens.count
                        ? try await textProjector.project(tokenId: tokenizeResult.trailingTextTokens[stepIndex])
                        : tokenizeResult.textPadEmbed
                    let combinedArr = try EmbedUtilities.createEmbedMLArray(EmbedUtilities.addEmbeddings(codecHidden, textTokenEmbed))
                    timings.textProjection += CFAbsoluteTimeGetCurrent() - textProjStart

                    let decodingStart = CFAbsoluteTimeGetCurrent()
                    lastCdOutput = try await codeDecoder.decode(inputEmbeds: combinedArr, cache: cdCache, state: cdState)
                    timings.decodingPredictions += CFAbsoluteTimeGetCurrent() - decodingStart

                    let sdResult = try await speechResult
                    timings.speechDecoderPredictions += sdResult.timings.speechDecoderPredictions
                    timings.speechDecoder += sdResult.timings.speechDecoderPredictions
                    allAudio.append(contentsOf: sdResult.samples)

                    if callback?(
                        SpeechProgress(
                            audio: sdResult.samples,
                            timings: {
                                var merged = baseTimings; merged.merge(timings); return merged
                            }(), stepTime: nil)) == false
                    {
                        break
                    }
                }

                let samplingStart = CFAbsoluteTimeGetCurrent()
                code0 = await sampler.sampleCodec0(
                    logits: lastCdOutput.logits,
                    temperature: options.temperature, topK: options.topK,
                    generatedTokens: generatedTokens,
                    repetitionPenalty: options.repetitionPenalty,
                    suppressTokenIds: suppressTokenIds
                )
                generatedTokens.append(code0)
                timings.decodingSampling += CFAbsoluteTimeGetCurrent() - samplingStart

                timings.decodingLoop += CFAbsoluteTimeGetCurrent() - stepStart
                stepIndex += 1
                progress.completedUnitCount = Int64(stepIndex)

                if stepIndex == 1 || stepIndex % 10 == 0 {
                    let stepMs = (CFAbsoluteTimeGetCurrent() - stepStart) * 1000
                    Logging.debug(
                        String(
                            format: "  Step %d: %.1fms (avg %.1fms/step)",
                            stepIndex, stepMs, timings.decodingLoop * 1000 / Double(stepIndex)))
                }
            }
        }

        let stopReason: String
        if code0 == Qwen3TTSConstants.codecEOS {
            stopReason = "EOS token"
        } else if cdCache.isFull {
            stopReason = "KV cache full (\(cdCache.cacheLength)/\(cdCache.maxSeqLength))"
        } else if stepIndex >= maxStepsByPrefill {
            stopReason = "Audio token ratio limit (\(stepIndex)/\(maxStepsByPrefill) steps)"
        } else {
            stopReason = "maxNewTokens limit (\(options.maxNewTokens))"
        }
        Logging.info("Loop stopped: \(stopReason) after \(stepIndex) steps")

        timings.totalDecodingLoops = Double(stepIndex)
        return GenerationLoopResult(audio: allAudio, steps: stepIndex, timings: timings)
    }

    // MARK: - Embedding Helpers

    /// Build the full combined embedding sequence (text track + codec track) for prefill.
    /// The returned array includes both the invariant prefix and the variable last token.
    func buildCombinedEmbeddings(
        speaker: Qwen3Speaker,
        lang: Qwen3Language,
        instruction: String?,
        firstTextEmbed: [FloatType],
        embedDim: Int
    ) async throws -> [[FloatType]] {
        let zeroCodecEmbed = EmbedUtilities.zeroEmbed(dim: embedDim)

        var instructTextEmbeds: [[FloatType]] = []
        var instructCodecEmbeds: [[FloatType]] = []
        if let instruction, !instruction.isEmpty {
            let instructPrompt = "<|im_start|>user\n\(instruction)<|im_end|>\n"
            let instructTokenIds = tokenizer.encode(text: instructPrompt).map { Int32($0) }
            for tokenId in instructTokenIds {
                try await instructTextEmbeds.append(textProjector.project(tokenId: tokenId))
                instructCodecEmbeds.append(zeroCodecEmbed)
            }
            Logging.debug("Instruction: \(instructTokenIds.count) tokens")
        }

        let rolePrefix = "<|im_start|>assistant\n"
        let roleTokenIds = tokenizer.encode(text: rolePrefix).map { Int32($0) }

        var textTrackEmbeds: [[FloatType]] = instructTextEmbeds
        for tokenId in roleTokenIds {
            try await textTrackEmbeds.append(textProjector.project(tokenId: tokenId))
        }
        let textPadEmbed = try await textProjector.project(tokenId: Qwen3TTSConstants.textPAD)
        let textBosEmbed = try await textProjector.project(tokenId: Qwen3TTSConstants.textBOS)

        let codecIds: [Int32] = [
            Qwen3TTSConstants.codecThink,
            Qwen3TTSConstants.codecThinkBos,
            lang.tokenID,
            Qwen3TTSConstants.codecThinkEos,
            speaker.tokenID,
            Qwen3TTSConstants.codecPAD,
            Qwen3TTSConstants.codecBOS
        ]
        var codecTrackEmbeds: [[FloatType]] = []
        for codecId in codecIds {
            try await codecTrackEmbeds.append(codeEmbedder.embed(tokenId: codecId))
        }

        let numPads = codecIds.count - 2
        for _ in 0..<numPads {
            textTrackEmbeds.append(textPadEmbed)
        }
        textTrackEmbeds.append(textBosEmbed)
        textTrackEmbeds.append(firstTextEmbed)

        var fullCodecTrackEmbeds: [[FloatType]] = instructCodecEmbeds
        fullCodecTrackEmbeds.append(contentsOf: Array(repeating: zeroCodecEmbed, count: roleTokenIds.count))
        fullCodecTrackEmbeds.append(contentsOf: codecTrackEmbeds)

        assert(
            textTrackEmbeds.count == fullCodecTrackEmbeds.count,
            "Track alignment mismatch: text=\(textTrackEmbeds.count) codec=\(fullCodecTrackEmbeds.count)")

        return zip(textTrackEmbeds, fullCodecTrackEmbeds).map { EmbedUtilities.addEmbeddings($0.0, $0.1) }
    }

    // MARK: - Prompt Cache Building

    /// Build a prompt cache by prefilling the invariant prefix tokens through the CodeDecoder.
    ///
    /// The invariant prefix includes: optional instruction tokens, role prefix,
    /// speaker/language control tokens, and BOS. Only the last token (first text token
    /// + codecBOS) varies per utterance and is excluded from the cache.
    ///
    /// Returns a snapshot of the KV cache state after prefilling; on cache hit the
    /// generation task restores this state and only decodes the variable token.
    open func buildPromptCache(
        voice: String,
        language: String,
        instruction: String?
    ) async throws -> TTSPromptCache {
        let qwen3Speaker = Qwen3Speaker(rawValue: voice) ?? .ryan
        let lang = Qwen3Language(rawValue: language) ?? .english
        let embedDim = codeDecoder.embedSize

        // Build invariant embeddings (everything except the last variable token).
        // Use a dummy firstTextEmbed since we drop the last element.
        let dummyFirstTextEmbed = EmbedUtilities.zeroEmbed(dim: embedDim)
        let allEmbeds = try await buildCombinedEmbeddings(
            speaker: qwen3Speaker,
            lang: lang,
            instruction: instruction,
            firstTextEmbed: dummyFirstTextEmbed,
            embedDim: embedDim
        )
        let invariantEmbeds = Array(allEmbeds.dropLast())

        // Pre-initialize the MultiCodeDecoder ANE pipeline concurrently with the
        // CodeDecoder prefill loop below. Cache build is the right place for this
        // one-time cost: it absorbs the ~150ms without affecting TTFB, and the
        // warmed pipeline persists for all subsequent generation calls.
        let mcdWarmupTask: Task<Void, Never>?
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            let mcd = multiCodeDecoder
            mcdWarmupTask = Task { try? await mcd.prewarmInference() }
        } else {
            mcdWarmupTask = nil
        }

        let cdState = codeDecoder.makeState()
        let cdCache = try KVCache(
            cacheDim: codeDecoder.kvCacheEmbedDim,
            maxSeqLength: codeDecoder.kvCacheMaxSequenceLength,
            isStateful: codeDecoder.isStateful
        )

        var lastCdOutput: CodeDecoderOutput?
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            // Async path: decoder updates cache internally
            for embed in invariantEmbeds {
                lastCdOutput = try await codeDecoder.decode(inputEmbeds: embed.asMLTensor(), cache: cdCache, state: cdState)
            }
        } else {
            for (embedIndex, embed) in invariantEmbeds.enumerated() {
                if embedIndex > 0, let keyUpdates = lastCdOutput?.keyCacheUpdates, let valueUpdates = lastCdOutput?.valueCacheUpdates {
                    cdCache.update(keyCacheUpdates: keyUpdates, valueCacheUpdates: valueUpdates)
                }
                let embedArr = try EmbedUtilities.createEmbedMLArray(embed)
                lastCdOutput = try await codeDecoder.decode(inputEmbeds: embedArr, cache: cdCache, state: cdState)
            }
            // Commit the last pending KV update so the snapshot is fully self-contained
            if let keyUpdates = lastCdOutput?.keyCacheUpdates, let valueUpdates = lastCdOutput?.valueCacheUpdates {
                cdCache.update(keyCacheUpdates: keyUpdates, valueCacheUpdates: valueUpdates)
            }
        }

        // Snapshot MLState for stateful models
        var stateData: KVStateData?
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), let mlState = cdState as? MLState {
            stateData = mlState.snapshot()
        }

        // Ensure warmup is done before returning - by this point the CodeDecoder
        // loop has run (~2700ms), so this await is a no-op in practice.
        await mcdWarmupTask?.value

        Logging.info("Built prompt cache: \(invariantEmbeds.count) invariant tokens, isStateful=\(codeDecoder.isStateful) for \(voice)/\(language)")

        return TTSPromptCache(
            voice: voice,
            language: language,
            instruction: instruction,
            prefixLength: invariantEmbeds.count,
            kvSnapshot: cdCache.snapshot(),
            stateData: stateData
        )
    }
}
