//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import Foundation

/// Responsible for transcribing audio chunk to text using the provided models and configurations.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
open class TranscribeTask {
    private var timings: TranscriptionTimings
    private let progress: Progress
    private let audioEncoder: any AudioEncoding
    private let featureExtractor: any FeatureExtracting
    private let segmentSeeker: any SegmentSeeking
    private let textDecoder: any TextDecoding
    private let audioProcessor: any AudioProcessing

    public private(set) var tokenizer: any WhisperTokenizer
    public var segmentDiscoveryCallback: SegmentDiscoveryCallback?

    public init(
        currentTimings: TranscriptionTimings,
        progress: Progress?,
        audioProcessor: (any AudioProcessing)? = nil,
        audioEncoder: any AudioEncoding,
        featureExtractor: any FeatureExtracting,
        segmentSeeker: any SegmentSeeking,
        textDecoder: any TextDecoding,
        tokenizer: any WhisperTokenizer
    ) {
        self.timings = currentTimings
        self.progress = progress ?? Progress()
        self.audioProcessor = audioProcessor ?? AudioProcessor()
        self.audioEncoder = audioEncoder
        self.featureExtractor = featureExtractor
        self.segmentSeeker = segmentSeeker
        self.textDecoder = textDecoder
        self.tokenizer = tokenizer
    }

    /// Hook for subclasses to launch work that can run alongside the main decoder pipeline.
    open func windowPreprocess(
        for paddedAudio: any AudioProcessorOutputType,
        seek: Int,
        segmentSize: Int
    ) async {}

    /// Hook for subclasses to finalize side work and optionally replace the segments for the current window.
    open func windowPostProcess(
        seek: Int,
        segmentSize: Int,
        originalSegments: [TranscriptionSegment]
    ) async -> [TranscriptionSegment] {
        originalSegments
    }

    public func run(
        audioArray: [Float],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback? = nil
    ) async throws -> TranscriptionResult {
        let interval = Logging.beginSignpost("TranscribeAudio", signposter: Logging.TranscribeTask.signposter)
        defer { Logging.endSignpost("TranscribeAudio", interval: interval, signposter: Logging.TranscribeTask.signposter) }

        timings.pipelineStart = min(CFAbsoluteTimeGetCurrent(), timings.pipelineStart)
        Logging.debug("Starting pipeline at: \(Date())")

        var options = decodeOptions ?? DecodingOptions()
        options.verbose = Logging.isLoggingEnabled

        var detectedLanguage: String?

        let contentFrames = audioArray.count
        timings.inputAudioSeconds = Double(contentFrames) / Double(WhisperKit.sampleRate) - Double(decodeOptions?.clipTimestamps.first ?? 0)

        // MARK: Init decoder inputs

        // These accumulate across windows
        var allSegments: [TranscriptionSegment] = []
        var allTokens: [Int] = []

        let startDecoderInit = CFAbsoluteTimeGetCurrent()
        var decoderInputs = try textDecoder.prepareDecoderInputs(withPrompt: [tokenizer.specialTokens.startOfTranscriptToken])
        let decoderInitTime = CFAbsoluteTimeGetCurrent() - startDecoderInit
        timings.decodingInit = decoderInitTime
        Logging.debug("Decoder init time: \(decoderInitTime)")

        // MARK: - Prefill KV Cache

        let prefillStartTime = CFAbsoluteTimeGetCurrent()
        if options.usePrefillPrompt {
            decoderInputs = try await textDecoder.prefillDecoderInputs(decoderInputs, withOptions: options)
        }
        // Update cache size based on prefill
        let prefilledCacheSize = decoderInputs.cacheLength[0].intValue
        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStartTime
        timings.prefill = prefillTime

        Logging.debug("Prefill time: \(prefillTime)")
        Logging.debug("Prefill prompt: \(decoderInputs.initialPrompt.map { tokenizer.convertIdToToken($0) ?? "" })")

        // MARK: - Main decoder loop

        // Process seek clips
        let seekClips = options.prepareSeekClips(contentFrames: contentFrames)
        Logging.debug("Decoding seek clips: \(seekClips)")

        let totalSeekDuration = seekClips.reduce(0) { $0 + ($1.end - $1.start) }
        progress.totalUnitCount = Int64(totalSeekDuration)

        let startDecodeLoopTime = CFAbsoluteTimeGetCurrent()
        for (seekClipStart, seekClipEnd) in seekClips {
            // Loop through the current clip until we reach the end
            // Typically this will be the full audio file, unless seek points are explicitly provided
            var seek: Int = seekClipStart

            let previousSeekProgress = progress.completedUnitCount

            // Prevent hallucinations at the end of the clip by stopping clip seek early
            let windowPadding = Int(options.windowClipTime * Float(WhisperKit.sampleRate))
            
            let windowSamples = featureExtractor.windowSamples ?? Constants.defaultWindowSamples
            while seek < seekClipEnd - windowPadding {
                let windowSeek = seek
                // calculate new encoder segment features
                let timeOffset = Float(seek) / Float(WhisperKit.sampleRate)
                let segmentSize = min(windowSamples, contentFrames - seek, seekClipEnd - seek)
                let timeOffsetEnd = Float(seek + segmentSize) / Float(WhisperKit.sampleRate)
                Logging.debug("Decoding Seek: \(seek) (\(Logging.formatTimestamp(timeOffset))s)")
                Logging.debug("Decoding Window Size: \(segmentSize) (\(Logging.formatTimestamp(timeOffsetEnd - timeOffset))s)")

                let audioProcessingStart = Date()
                let clipAudioSamples = Array(audioArray[seek..<(seek + segmentSize)])
                guard let audioSamples = audioProcessor.padOrTrim(fromArray: clipAudioSamples, startAt: 0, toLength: windowSamples) else {
                    throw WhisperError.transcriptionFailed("Audio samples are nil")
                }
                await windowPreprocess(for: audioSamples, seek: windowSeek, segmentSize: segmentSize)
                let processTime = Date().timeIntervalSince(audioProcessingStart)
                timings.audioProcessing += processTime
                timings.totalAudioProcessingRuns += 1

                try Task.checkCancellation()
                let melStart = Date()
                guard let melOutput = try await featureExtractor.logMelSpectrogram(fromAudio: audioSamples) else {
                    throw WhisperError.transcriptionFailed("Mel output is nil")
                }
                let melTime = Date().timeIntervalSince(melStart)
                timings.logmels += melTime
                timings.totalLogmelRuns += 1

                try Task.checkCancellation()
                let encoderStart = Date()
                guard let encoderOutput = try await audioEncoder.encodeFeatures(melOutput) else {
                    throw WhisperError.transcriptionFailed("Encoder output is nil")
                }
                let encoderTime = Date().timeIntervalSince(encoderStart)
                timings.encoding += encoderTime
                timings.totalEncodingRuns += 1

                // All features are computed, now we can decode
                Logging.info("Decoding \(Logging.formatTimestamp(timeOffset))s - \(Logging.formatTimestamp(timeOffsetEnd))s")

                // Overload progress callback to include windowId
                let windowId = Int(timings.totalDecodingWindows - timings.totalDecodingFallbacks)
                let decodingCallback: TranscriptionCallback = { progress in
                    guard let callback = callback else { return nil }
                    var windowProgress = progress
                    windowProgress.windowId = windowId
                    return callback(windowProgress)
                }

                try Task.checkCancellation()
                // Send to decoder to predict text tokens with fallback
                let decodingResult = try await decodeWithFallback(
                    encoderSegment: encoderOutput,
                    decodingOptions: options,
                    decoderInputs: &decoderInputs,
                    detectedLanguage: &detectedLanguage,
                    prefilledCacheSize: prefilledCacheSize,
                    callback: decodingCallback
                )

                // MARK: Windowing

                // At this point we have a completed window aka segment
                let windowingStart = Date()

                let previousSeek = seek
                var (newSeek, currentSegments) = segmentSeeker.findSeekPointAndSegments(
                    decodingResult: decodingResult,
                    options: options,
                    allSegmentsCount: allSegments.count,
                    currentSeek: seek,
                    segmentSize: segmentSize,
                    sampleRate: WhisperKit.sampleRate,
                    timeToken: tokenizer.specialTokens.timeTokenBegin,
                    specialToken: tokenizer.specialTokens.specialTokenBegin,
                    tokenizer: tokenizer
                )

                // Update seek point without moving backward
                seek = max(seek, newSeek)

                // Optionally add word timestamps
                if options.wordTimestamps,
                   let alignmentWeights = decodingResult.cache?.alignmentWeights
                {
                    let wordTimestampsStart = Date()
                    currentSegments = try segmentSeeker.addWordTimestamps(
                        segments: currentSegments ?? [],
                        alignmentWeights: alignmentWeights,
                        tokenizer: tokenizer,
                        seek: previousSeek,
                        segmentSize: segmentSize,
                        prependPunctuations: Constants.defaultPrependPunctuations,
                        appendPunctuations: Constants.defaultAppendPunctuations,
                        lastSpeechTimestamp: Float(Double(previousSeek) / Double(WhisperKit.sampleRate)),
                        options: options,
                        timings: timings
                    )

                    timings.decodingWordTimestamps += Date().timeIntervalSince(wordTimestampsStart)
                    timings.totalTimestampAlignmentRuns += 1

                    // Filter out zero length segments
                    currentSegments = currentSegments?.filter { $0.end > $0.start }

                    // Update seek point with new (more accurate) segments
                    if let lastSpeechTimestamp = currentSegments?.last?.end {
                        seek = max(seek, Int(lastSpeechTimestamp * Float(WhisperKit.sampleRate)))
                    }

                    if options.verbose {
                        Logging.debug("Word timestamps:")
                        for segment in currentSegments ?? [] {
                            for word in segment.words ?? [] {
                                Logging.debug("[\(word.start.formatted(.number.precision(.significantDigits(3)))) -> \(word.end.formatted(.number.precision(.significantDigits(3))))] prob: \(word.probability), word: \(word.word)")
                            }
                        }
                    }
                }
                
                // Prevent seek from exceeding previousSeek + maxWindowSeek if provided
                if let maxWindowSeek = options.maxWindowSeek {
                    let maxSeekOffset = previousSeek + maxWindowSeek
                    seek = min(seek, maxSeekOffset)
                }

                guard let currentSegments else {
                    // No current segment found, skip to next window
                    continue
                }

                let processedSegments = await windowPostProcess(
                    seek: windowSeek,
                    segmentSize: segmentSize,
                    originalSegments: currentSegments
                )

                if options.verbose {
                    let lines = TranscriptionUtilities.formatSegments(processedSegments)
                    Logging.debug("Segments for window:")
                    for line in lines {
                        Logging.debug(line)
                    }
                }

                segmentDiscoveryCallback?(processedSegments)

                // add them to the `allSegments` list
                allSegments.append(contentsOf: processedSegments)
                let allCurrentTokens = processedSegments.flatMap { $0.tokens }
                allTokens.append(contentsOf: allCurrentTokens)

                timings.decodingWindowing += Date().timeIntervalSince(windowingStart)
                timings.totalDecodingWindows += 1

                // Reset cache and move on to the next window
                decoderInputs.reset(
                    prefilledCacheSize: prefilledCacheSize,
                    maxTokenContext: decodeOptions?.sampleLength ?? Constants.maxTokenContext
                )

                // Update the progress
                let clipProgress = min(seek, seekClipEnd) - seekClipStart
                progress.completedUnitCount = previousSeekProgress + Int64(clipProgress)
            }
        }

        // Transcription completed
        progress.completedUnitCount = progress.totalUnitCount

        // MARK: Result

        timings.decodingLoop = CFAbsoluteTimeGetCurrent() - startDecodeLoopTime
        timings.fullPipeline = CFAbsoluteTimeGetCurrent() - timings.pipelineStart

        let transcriptionResult = finalizeTranscriptionResult(
            tokens: allTokens,
            segments: allSegments,
            language: detectedLanguage,
            timings: timings
        )
        return transcriptionResult
    }
    
    open func finalizeTranscriptionResult(
        tokens: [Int],
        segments allSegments: [TranscriptionSegment],
        language detectedLanguage: String?,
        timings: TranscriptionTimings
    ) -> TranscriptionResult {
        let wordTokens = tokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
        let transcription = tokenizer.decode(tokens: wordTokens).trimmingCharacters(in: .whitespaces)
        return TranscriptionResult(
            text: transcription,
            segments: allSegments,
            language: detectedLanguage ?? Constants.defaultLanguageCode,
            timings: timings
        )
    }

    // MARK: - Decode with Fallback Logic

    private func decodeWithFallback(
        encoderSegment encoderOutput: any AudioEncoderOutputType,
        decodingOptions options: DecodingOptions,
        decoderInputs: inout any DecodingInputsType,
        detectedLanguage: inout String?,
        prefilledCacheSize: Int,
        callback: TranscriptionCallback? = nil
    ) async throws -> DecodingResult {
        let interval = Logging.beginSignpost("Decode", signposter: Logging.TranscribeTask.signposter)
        defer { Logging.endSignpost("Decode", interval: interval, signposter: Logging.TranscribeTask.signposter) }

        // Fallback `options.temperatureFallbackCount` times with increasing temperatures, starting at `options.temperature`
        let temperatures = (0...options.temperatureFallbackCount).map { FloatType(options.temperature) + FloatType($0) * FloatType(options.temperatureIncrementOnFallback) }

        Logging.debug("Decoding with temperatures \(temperatures)")

        var decodingResult: DecodingResult?

        for (i, temp) in temperatures.enumerated() {
            Logging.info("Decoding Temperature: \(temp)")
            let decodeWithFallbackStart = Date()

            let tokenSampler = GreedyTokenSampler(temperature: temp, eotToken: tokenizer.specialTokens.endToken, decodingOptions: options)

            var currentDecodingOptions = options
            // For a multilingual model, if language is not passed and detectLanguage is true, detect language and set in options
            if textDecoder.isModelMultilingual, options.language == nil, options.detectLanguage {
                let languageDecodingResult: DecodingResult? = try? await textDecoder.detectLanguage(
                    from: encoderOutput,
                    using: decoderInputs,
                    sampler: tokenSampler,
                    options: options,
                    temperature: temp
                )

                // Update the language decoding options
                currentDecodingOptions.language = languageDecodingResult?.language
                detectedLanguage = languageDecodingResult?.language

                // Update prompt and KV Cache if needed
                if options.usePrefillPrompt {
                    decoderInputs = try await textDecoder.prefillDecoderInputs(decoderInputs, withOptions: currentDecodingOptions)
                }
                Logging.debug("Prefill prompt updated to: \(decoderInputs.initialPrompt.map { tokenizer.convertIdToToken($0) ?? "" })")

                // Update timings from the language detection
                if let languageDecodingTimings = languageDecodingResult?.timings {
                    timings.decodingPredictions += languageDecodingTimings.decodingPredictions
                    timings.decodingSampling += languageDecodingTimings.decodingSampling
                }
            }

            decodingResult = try await textDecoder.decodeText(
                from: encoderOutput,
                using: decoderInputs,
                sampler: tokenSampler,
                options: currentDecodingOptions,
                callback: callback
            )

            // Use the predicted language if it was not detected ahead of time
            if detectedLanguage == nil {
                detectedLanguage = decodingResult?.language
            }

            // Update timings from the decoder main loop
            if let decodingTimings = decodingResult?.timings {
                timings.firstTokenTime = min(decodingTimings.firstTokenTime, timings.firstTokenTime)
                timings.decodingPredictions += decodingTimings.decodingPredictions
                timings.totalDecodingLoops += decodingTimings.totalDecodingLoops
                timings.decodingNonPrediction += decodingTimings.decodingNonPrediction
                timings.decodingFiltering += decodingTimings.decodingFiltering
                timings.decodingSampling += decodingTimings.decodingSampling
                timings.decodingKvCaching += decodingTimings.decodingKvCaching
                timings.totalKVUpdateRuns += decodingTimings.totalKVUpdateRuns
            }

            // MARK: Fallback checks

            if let fallback = decodingResult?.fallback, fallback.needsFallback {
                // Reset decoder inputs for fallback
                timings.decodingFallback += Date().timeIntervalSince(decodeWithFallbackStart)
                timings.totalDecodingFallbacks = Double(i)
                decoderInputs.reset(
                    prefilledCacheSize: prefilledCacheSize,
                    maxTokenContext: options.sampleLength
                )
                Logging.info("Fallback #\(i + 1) (\(fallback.fallbackReason))")
            } else {
                break
            }
        }

        guard let decodingResult else {
            throw WhisperError.decodingFailed()
        }
        return decodingResult
    }
}
