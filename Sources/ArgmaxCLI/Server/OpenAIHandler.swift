//  Copyright © 2025 Argmax, Inc. All rights reserved.
//  For licensing see accompanying LICENSE.md file.

import Foundation
import Vapor
import OpenAPIRuntime
import OpenAPIVapor
@preconcurrency import WhisperKit

struct OpenAIHandler: APIProtocol {
    let whisperKit: WhisperKit
    private let logger: Logger
    private let transcribeArguments: TranscribeCLIArguments

    init(whisperKit: WhisperKit, logger: Logger, transcribeArguments: TranscribeCLIArguments) {
        self.whisperKit = whisperKit
        self.logger = logger
        self.transcribeArguments = transcribeArguments
    }

    // Helper function to create streaming response
    private func createStreamingResponse(
        whisperKit: WhisperKit,
        audioPath: String,
        decodeOptions: DecodingOptions,
        responseFormat: String,
        timestampGranularities: [String],
        requestLanguage: String?,
        includeLogprobs: Bool
    ) async throws -> OpenAPIRuntime.HTTPBody {
        // Create a streaming response that yields data immediately
        let stream = AsyncStream<ArraySlice<UInt8>> { continuation in
            Task {
                let progressCallback: TranscriptionCallback = { progress in
                    // Send progress update as SSE immediately
                    let progressEvent = Components.Schemas.TranscriptTextDeltaEvent(
                        _type: .transcript_period_text_period_delta,
                        delta: progress.text,
                        logprobs: includeLogprobs ? {
                            guard let tokenizer = whisperKit.tokenizer else { return nil }
                            let filteredTokens = progress.tokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
                            return filteredTokens.compactMap { tokenId in
                                let tokenText = tokenizer.decode(tokens: [tokenId])
                                let tokenBytes = Array(tokenText.utf8)
                                return Components.Schemas.TranscriptTextDeltaEvent.logprobsPayloadPayload(
                                    token: tokenText,
                                    logprob: progress.avgLogprob.map { Double($0) } ?? 0.0,
                                    bytes: try? OpenAPIRuntime.OpenAPIArrayContainer(unvalidatedValue: tokenBytes.map { Double($0) })
                                )
                            }
                        }() : nil
                    )

                    if let eventData = try? JSONEncoder().encode(progressEvent) {
                        let sseEvent = "data: \(String(data: eventData, encoding: .utf8) ?? "")\n\n"
                        let bytes = Array(sseEvent.utf8)[...]  // Convert to ArraySlice<UInt8>
                        continuation.yield(bytes)
                    }

                    return true // Continue transcription
                }

                do {
                    // Perform transcription
                    let transcriptionResult = try await whisperKit.transcribe(
                        audioPath: audioPath,
                        decodeOptions: decodeOptions,
                        callback: progressCallback
                    )

                    let transcription = TranscriptionUtilities.mergeTranscriptionResults(transcriptionResult)

                    let finalEvent = Components.Schemas.TranscriptTextDoneEvent(
                        _type: .transcript_period_text_period_done,
                        text: transcription.text,
                        logprobs: includeLogprobs ? {
                            guard let tokenizer = whisperKit.tokenizer else { return nil }
                            return transcription.segments.flatMap { segment in
                                // Create logprobs for each token in the final transcription
                                let filteredTokens = segment.tokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
                                return filteredTokens.enumerated().compactMap { index, tokenId in
                                    let tokenText = tokenizer.decode(tokens: [tokenId])
                                    let tokenBytes = Array(tokenText.utf8)
                                    return Components.Schemas.TranscriptTextDoneEvent.logprobsPayloadPayload(
                                        token: tokenText,
                                        logprob: Double(segment.avgLogprob),
                                        bytes: try? OpenAPIRuntime.OpenAPIArrayContainer(unvalidatedValue: tokenBytes.map { Double($0) })
                                    )
                                }
                            }
                        }() : nil
                    )

                    if let eventData = try? JSONEncoder().encode(finalEvent) {
                        let sseEvent = "data: \(String(data: eventData, encoding: .utf8) ?? "")\n\n"
                        let bytes = Array(sseEvent.utf8)[...]  // Convert to ArraySlice<UInt8>
                        continuation.yield(bytes)
                    }
                    


                    continuation.finish()
                } catch {
                    // Send error event
                    let errorEvent = "data: {\"error\": \"\(error.localizedDescription)\"}\n\n"
                    let bytes = Array(errorEvent.utf8)[...]  // Convert to ArraySlice<UInt8>
                    continuation.yield(bytes)
                    continuation.finish()
                }
            }
        }

        // Convert AsyncStream to HTTPBody with proper streaming parameters
        return OpenAPIRuntime.HTTPBody(
            stream,
            length: .unknown,
            iterationBehavior: .single
        )
    }

    func createTranscription(_ input: Operations.createTranscription.Input) async throws -> Operations.createTranscription.Output {
        let multipartBody: MultipartBody<Operations.createTranscription.Input.Body.multipartFormPayload>

        switch input.body {
        case .multipartForm(let form): multipartBody = form
        }

        var transcription: TranscriptionResult?
        var tmpURL: URL?
        var requestLanguage: String?
        var requestPrompt: String?
        var requestTemperature: Float = 0.0
        var requestModel: String?
        var requestResponseFormat: String = "verbose_json"
        var requestTimestampGranularities: [String] = ["segment"]
        var requestStream: Bool = false
        var requestIncludeLogprobs: Bool = false

        for try await part in multipartBody {
            switch part {
            case .file(let filePart):
                let data = try await Data(collecting: filePart.payload.body, upTo: Int.max)
                tmpURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString)
                    .appendingPathExtension(String(filePart.filename?.split(separator: ".").last ?? "wav"))

                if let url = tmpURL {
                    try data.write(to: url, options: .atomic)
                }

            case .model(let modelPart):
                requestModel = try await String(collecting: modelPart.payload.body, upTo: 1 << 20)

            case .language(let langPart):
                requestLanguage = try await String(collecting: langPart.payload.body, upTo: 16)

            case .prompt(let prompt):
                requestPrompt = try await String(collecting: prompt.payload.body, upTo: 4 << 20)

            case .response_format(let fmt):
                requestResponseFormat = try await String(collecting: fmt.payload.body, upTo: 64)
                logger.notice("Request response format: \(requestResponseFormat)")

            case .temperature(let temp):
                if let tempStr = try? await String(collecting: temp.payload.body, upTo: 8),
                   let tempFloat = Float(tempStr) {
                    requestTemperature = tempFloat
                }

            case .include_lbrack__rbrack_(let inc):
                let includeStr = try await String(collecting: inc.payload.body, upTo: 256)
                requestIncludeLogprobs = includeStr.components(separatedBy: ",")
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .contains("logprobs")
                logger.notice("Request include logprobs: \(requestIncludeLogprobs)")

            case .timestamp_granularities_lbrack__rbrack_(let ts):
                let granularitiesStr = try await String(collecting: ts.payload.body, upTo: 256)
                requestTimestampGranularities = granularitiesStr.components(separatedBy: ",")
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty }
                if requestTimestampGranularities.isEmpty {
                    requestTimestampGranularities = ["segment"]
                }
                logger.notice("Request timestamp granularities: \(requestTimestampGranularities)")

            case .stream(let streamFlag):
                let streamStr = try await String(collecting: streamFlag.payload.body, upTo: 8)
                requestStream = streamStr.lowercased() == "true"
                logger.notice("Request streaming: \(requestStream)")
            }
        }

        guard let audioURL = tmpURL else {
            throw Abort(.badRequest, reason: "No audio file provided")
        }

        // Create custom decoding options with request parameters
        var customOptions = TranscribeCLIUtils.createDecodingOptions(from: transcribeArguments, task: .transcribe)

        // Enable word timestamps if requested
        if requestTimestampGranularities.contains("word") {
            customOptions.wordTimestamps = true
            logger.notice("Enabling word timestamps for transcription")
        }

        // For transcription, set language if provided, otherwise let it auto-detect
        if let requestLanguage = requestLanguage {
            customOptions.language = requestLanguage
            logger.notice("Request language: \(requestLanguage)")
        } else {
            logger.notice("Auto-detecting language for transcription")
        }

        // Process prompt text
        if let requestPrompt = requestPrompt, requestPrompt.count > 0, let tokenizer = whisperKit.tokenizer {
            customOptions.promptTokens = tokenizer.encode(text: " " + requestPrompt.trimmingCharacters(in: .whitespaces))
                .filter { $0 < tokenizer.specialTokens.specialTokenBegin }
            customOptions.usePrefillPrompt = true
            if transcribeArguments.verbose {
                logger.notice("Processing prompt text: \"\(requestPrompt)\"")
                logger.notice("Encoded prompt tokens: \(customOptions.promptTokens ?? [])")
            }
        }

        if let requestModel = requestModel {
            logger.debug("Request model: \(requestModel)")
        }

        customOptions.temperature = requestTemperature
        logger.notice("Request temperature: \(requestTemperature)")

        if requestStream {
            // For streaming, we need to return a text/event-stream response
            logger.notice("Starting streaming transcription")

            let streamingResponse = try await createStreamingResponse(
                whisperKit: whisperKit,
                audioPath: audioURL.path(),
                decodeOptions: customOptions,
                responseFormat: requestResponseFormat,
                timestampGranularities: requestTimestampGranularities,
                requestLanguage: requestLanguage,
                includeLogprobs: requestIncludeLogprobs
            )

            return .ok(.init(body: .text_event_hyphen_stream(streamingResponse)))
        } else {
            // Non-streaming: perform transcription and return JSON response
            let fullTranscription = try await whisperKit.transcribe(
                audioPath: audioURL.path(),
                decodeOptions: customOptions
            )
            transcription = TranscriptionUtilities.mergeTranscriptionResults(fullTranscription)

            guard let transcription = transcription else {
                throw Abort(.internalServerError, reason: "Transcription failed")
            }

            // Determine response format and return appropriate schema
            let detectedLanguage = transcription.language
            let finalLanguage = requestLanguage ?? detectedLanguage

            switch requestResponseFormat.lowercased() {
            case "json":
                let payload = Components.Schemas.CreateTranscriptionResponseJson(
                    text: transcription.text,
                    logprobs: requestIncludeLogprobs ? {
                        guard let tokenizer = whisperKit.tokenizer else { return nil }
                        return transcription.segments.flatMap { segment in
                            // Create logprobs for each token in the segment
                            let filteredTokens = segment.tokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
                            return filteredTokens.enumerated().compactMap { index, tokenId in
                                let tokenText = tokenizer.decode(tokens: [tokenId])
                                let tokenBytes = Array(tokenText.utf8)
                                return Components.Schemas.CreateTranscriptionResponseJson.logprobsPayloadPayload(
                                    token: tokenText,
                                    logprob: Double(segment.avgLogprob),
                                    bytes: tokenBytes.map { Double($0) }
                                )
                            }
                        }
                    }() : nil,
                    _type: .CreateTranscriptionResponseJson
                )
                
                return .ok(.init(body: .json(.CreateTranscriptionResponseJson(payload))))

            case "verbose_json", _:
                // Default to verbose JSON with conditional word timing
                let includeWords = requestTimestampGranularities.contains("word")
                
                if requestIncludeLogprobs {
                    logger.warning("Logprobs requested but verbose_json format doesn't support logprobs field. Consider using 'json' format instead.")
                }
                
                let payload = Components.Schemas.CreateTranscriptionResponseVerboseJson(
                    language: finalLanguage,
                    duration: transcription.timings.inputAudioSeconds,
                    text: transcription.text,
                    words: includeWords ? transcription.allWords.map {
                        Components.Schemas.TranscriptionWord(word: $0.word, start: $0.start, end: $0.end)
                    } : [],
                    segments: transcription.segments.map {
                        Components.Schemas.TranscriptionSegment(
                            id: $0.id,
                            seek: $0.seek,
                            start: $0.start,
                            end: $0.end,
                            text: $0.text,
                            tokens: $0.tokens,
                            temperature: $0.temperature,
                            avg_logprob: $0.avgLogprob,
                            compression_ratio: $0.compressionRatio,
                            no_speech_prob: $0.noSpeechProb
                        )
                    },
                    _type: .CreateTranscriptionResponseVerboseJson
                )
                
                return .ok(.init(body: .json(.CreateTranscriptionResponseVerboseJson(payload))))
            }
        }
    }

    func createTranslation(_ input: Operations.createTranslation.Input) async throws -> Operations.createTranslation.Output {
        let multipartBody: MultipartBody<Operations.createTranslation.Input.Body.multipartFormPayload>

        switch input.body {
        case .multipartForm(let form): multipartBody = form
        }

        var translation: TranscriptionResult?
        var tmpURL: URL?
        var requestPrompt: String?
        var requestTemperature: Float = 0.0
        var requestModel: String?
        var requestResponseFormat: String = "json"
        var requestLanguage: String? = nil

        for try await part in multipartBody {
            switch part {
            case .file(let filePart):
                let data = try await Data(collecting: filePart.payload.body, upTo: Int.max)
                tmpURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString)
                    .appendingPathExtension("audio")
                try data.write(to: tmpURL!)

            case .response_format(let fmt):
                requestResponseFormat = try await String(collecting: fmt.payload.body, upTo: 64)
                logger.notice("Request response format: \(requestResponseFormat)")

            case .model(let modelPart):
                requestModel = try await String(collecting: modelPart.payload.body, upTo: 1 << 20)

            case .prompt(let prompt):
                requestPrompt = try await String(collecting: prompt.payload.body, upTo: 4 << 20)

            case .temperature(let temp):
                if let tempStr = try? await String(collecting: temp.payload.body, upTo: 8),
                   let tempFloat = Float(tempStr) {
                    requestTemperature = tempFloat
                }

            case .language(let langPart):
                requestLanguage = try await String(collecting: langPart.payload.body, upTo: 16)
            }
        }

        guard let audioURL = tmpURL else {
            throw Abort(.badRequest, reason: "No audio file provided")
        }

        // Create custom decoding options with request parameters
        var customOptions = TranscribeCLIUtils.createDecodingOptions(from: transcribeArguments, task: .translate)
        if let requestLanguage {
            customOptions.language = requestLanguage
            customOptions.usePrefillPrompt = true
            logger.notice("Request language: \(requestLanguage)")
        } else {
            logger.notice("Auto-detecting language for translation")
        }

        // Process prompt text
        if let requestPrompt = requestPrompt, requestPrompt.count > 0, let tokenizer = whisperKit.tokenizer {
            customOptions.promptTokens = tokenizer.encode(text: " " + requestPrompt.trimmingCharacters(in: .whitespaces))
                .filter { $0 < tokenizer.specialTokens.specialTokenBegin }
            customOptions.usePrefillPrompt = true
            if transcribeArguments.verbose {
                logger.notice("Processing translation prompt text: \"\(requestPrompt)\"")
                logger.notice("Encoded prompt tokens: \(customOptions.promptTokens ?? [])")
            }
        }

        if let requestModel = requestModel {
            logger.debug("Request model (ignored): \(requestModel)")
        }

        customOptions.temperature = requestTemperature
        logger.notice("Request temperature: \(requestTemperature)")

        // Use translation task instead of transcription
        let fullTranslation = try await whisperKit.transcribe(
            audioPath: audioURL.path(),
            decodeOptions: customOptions
        )
        translation = TranscriptionUtilities.mergeTranscriptionResults(fullTranslation)

        guard let translation = translation else {
            throw Abort(.internalServerError, reason: "Translation failed")
        }

        // Return appropriate response format for translation
        switch requestResponseFormat.lowercased() {
        case "json":
            let payload = Components.Schemas.CreateTranslationResponseJson(
                text: translation.text
            )
            return .ok(.init(body: .json(.CreateTranslationResponseJson(payload))))

        case "verbose_json":
            let payload = Components.Schemas.CreateTranslationResponseVerboseJson(
                language: "english",
                duration: translation.timings.inputAudioSeconds,
                text: translation.text,
                segments: translation.segments.map {
                    Components.Schemas.TranscriptionSegment(
                        id: $0.id,
                        seek: $0.seek,
                        start: $0.start,
                        end: $0.end,
                        text: $0.text,
                        tokens: $0.tokens,
                        temperature: $0.temperature,
                        avg_logprob: $0.avgLogprob,
                        compression_ratio: $0.compressionRatio,
                        no_speech_prob: $0.noSpeechProb
                    )
                }
            )
            return .ok(.init(body: .json(.CreateTranslationResponseVerboseJson(payload))))

        default:
            // Default to JSON
            let payload = Components.Schemas.CreateTranslationResponseJson(
                text: translation.text
            )
            return .ok(.init(body: .json(.CreateTranslationResponseJson(payload))))
        }
    }
}
