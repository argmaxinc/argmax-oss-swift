//  Copyright © 2025 Argmax, Inc. All rights reserved.
//  For licensing see accompanying LICENSE.md file.

import ArgumentParser
import CoreML
import Foundation
@preconcurrency import WhisperKit
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
                service: "WhisperKit Local Server",
                endpoints: [
                    Endpoint(method: "POST", path: "/v1/audio/transcriptions", description: "Transcribe audio to text"),
                    Endpoint(method: "POST", path: "/v1/audio/translations", description: "Translate audio to English"),
                    Endpoint(method: "GET", path: "/health", description: "Health check endpoint")
                ]
            )
        }

        app.get("health") { req async throws -> [String: String] in
            return ["status": "ok"]
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
