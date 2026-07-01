//  Copyright © 2025 Argmax, Inc. All rights reserved.
//  For licensing see accompanying LICENSE.md file.

import ArgumentParser
import CoreML
import Foundation
@preconcurrency import WhisperKit
import TTSKit
import Vapor
import OpenAPIRuntime
import OpenAPIVapor
import AVFoundation

struct ServeCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start a local server for WhisperKit transcription"
    )

    @OptionGroup
    var cliArguments: ServeCLIArguments

    mutating func run() async throws {
        try await serve()
    }

    public func configure(_ app: Application) async throws {
        let transport = VaporTransport(routesBuilder: app)

        var transcribeArguments = cliArguments.transcribe
        transcribeArguments.skipSpecialTokens = true // always skip special tokens for server responses
        if let modelPath = cliArguments.transcribe.modelPath {
            app.logger.notice("Loading model from path: \(modelPath)")
        } else if let model = cliArguments.transcribe.model {
            app.logger.notice("Loading model: \(model)")
        } else {
            let defaultModel = WhisperKit.recommendedModels().default
            app.logger.notice("Loading default model: \(defaultModel)")
            transcribeArguments.model = defaultModel
            transcribeArguments.modelPrefix = ""
        }

        let config = TranscribeCLIUtils.createWhisperKitConfig(from: transcribeArguments)
        let whisperKit = try await WhisperKit(config)
        let handler = OpenAIHandler(whisperKit: whisperKit, logger: app.logger, transcribeArguments: transcribeArguments)
        try handler.registerHandlers(on: transport, serverURL: URL(string: "/v1")!)
        
        // Register base routes after OpenAPI routes to ensure they take precedence
        app.get("") { req async throws -> EndpointInfo in
            return EndpointInfo(
                status: "ok",
                service: "Argmax Voice Server",
                endpoints: [
                    Endpoint(method: "POST", path: "/v1/audio/transcriptions", description: "Transcribe audio to text"),
                    Endpoint(method: "POST", path: "/v1/audio/translations", description: "Translate audio to English"),
                    Endpoint(method: "POST", path: "/v1/audio/speech", description: "Generate speech from text"),
                    Endpoint(method: "GET", path: "/health", description: "Health check endpoint")
                ]
            )
        }

        app.get("health") { req async throws -> [String: String] in
            return ["status": "ok"]
        }

        // --- TTS Setup ---
        app.logger.notice("Loading TTSKit (Qwen3-TTS 0.6B)...")
        let ttsConfig = TTSKitConfig(model: .qwen3TTS_0_6b, verbose: cliArguments.transcribe.verbose)
        let ttsKit = try await TTSKit(ttsConfig)
        app.logger.notice("TTSKit loaded successfully")

        app.post("v1", "audio", "speech") { req async throws -> Response in
            struct SpeechRequest: Content {
                var input: String
                var model: String?
                var voice: String?
                var language: String?
                var response_format: String?
                var speed: Float?
            }

            let speechReq = try req.content.decode(SpeechRequest.self)

            guard !speechReq.input.isEmpty else {
                throw Abort(.badRequest, reason: "input text must not be empty")
            }

            let speaker: Qwen3Speaker = {
                switch speechReq.voice?.lowercased() {
                case "alloy", nil: return .ryan
                case "echo": return .aiden
                case "nova": return .serena
                case "shimmer": return .vivian
                case "onyx": return .eric
                case "fable": return .dylan
                default:
                    return Qwen3Speaker(rawValue: speechReq.voice ?? "ryan") ?? .ryan
                }
            }()

            let language: Qwen3Language = {
                switch speechReq.language?.lowercased() {
                case "es", "spanish": return .spanish
                case "en", "english", nil: return .english
                case "fr", "french": return .french
                case "de", "german": return .german
                case "pt", "portuguese": return .portuguese
                case "it", "italian": return .italian
                case "ja", "japanese": return .japanese
                case "ko", "korean": return .korean
                case "zh", "chinese": return .chinese
                case "ru", "russian": return .russian
                default: return .english
                }
            }()

            let options = GenerationOptions(
                temperature: 0.9,
                topK: 50,
                repetitionPenalty: 1.05,
                maxNewTokens: 245,
                concurrentWorkerCount: 1
            )

            req.logger.notice("TTS request: voice=\(speaker.rawValue) language=\(language.rawValue) text=\"\(speechReq.input.prefix(80))\"")

            let result = try await ttsKit.generate(
                text: speechReq.input,
                speaker: speaker,
                language: language,
                options: options
            )

            let wavData = TTSAudioEncoder.encodeWAV(samples: result.audio, sampleRate: result.sampleRate)

            var headers = HTTPHeaders()
            headers.add(name: .contentType, value: "audio/wav")
            headers.add(name: .contentDisposition, value: "attachment; filename=\"speech.wav\"")
            return Response(status: .ok, headers: headers, body: .init(data: wavData))
        }
    }

    private func serve() async throws {
        var env = try Environment.detect()
        try LoggingSystem.bootstrap(from: &env)
        let app = try await Application.make()
        app.logger.logLevel = cliArguments.transcribe.verbose ? .debug : .info
        app.logger.notice("Starting WhisperKit Server...")
        app.environment.arguments = [""] // override arguments, handled by swift-argument-parser

        // Configure server to bind to specified host and port
        app.http.server.configuration.hostname = cliArguments.host
        app.http.server.configuration.port = cliArguments.port
        app.logger.notice("Server will bind to \(cliArguments.host):\(cliArguments.port)")

        do {
            try await configure(app)
            try await app.execute()
        } catch {
            app.logger.report(error: error)
            try? await app.asyncShutdown()
            throw error
        }
        try await app.asyncShutdown()
    }
}

// Response structs for the base endpoint
fileprivate struct Endpoint: Content {
    let method: String
    let path: String
    let description: String
}

fileprivate struct EndpointInfo: Content {
    let status: String
    let service: String
    let endpoints: [Endpoint]
}

enum TTSAudioEncoder {
    static func encodeWAV(samples: [Float], sampleRate: Int) -> Data {
        let numChannels: Int16 = 1
        let bitsPerSample: Int16 = 16
        let byteRate = Int32(sampleRate * Int(numChannels) * Int(bitsPerSample / 8))
        let blockAlign = Int16(numChannels * (bitsPerSample / 8))
        let dataSize = Int32(samples.count * Int(bitsPerSample / 8))
        let chunkSize = 36 + dataSize

        var data = Data()
        data.reserveCapacity(44 + samples.count * 2)

        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: chunkSize.littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: Int32(16).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: Int16(1).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: Int32(sampleRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16Val = Int16(clamped * Float(Int16.max))
            data.append(contentsOf: withUnsafeBytes(of: int16Val.littleEndian) { Array($0) })
        }

        return data
    }
}
