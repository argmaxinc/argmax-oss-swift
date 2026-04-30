//  Copyright © 2025 Argmax, Inc. All rights reserved.
//  For licensing see accompanying LICENSE.md file.

import Foundation
import ArgumentParser
import OpenAPIRuntime
import OpenAPIURLSession
import HTTPTypes

// MARK: - Main Command

@main
struct WhisperKitSwiftClient: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "whisperkit-client",
        abstract: "WhisperKit Swift Client for local server",
        subcommands: [
            Transcribe.self,
            Translate.self,
            Test.self
        ]
    )
    
    func run() async throws {
        print("Use one of the subcommands: transcribe, translate, or test")
    }
}

// MARK: - Transcribe Command

struct Transcribe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe audio file"
    )
    
    @Argument(help: "Path to audio file")
    var audioFile: String
    
    @Option(name: .shortAndLong, help: "Source language (default: auto-detect)")
    var language: String = ""
    
    @Option(name: .shortAndLong, help: "Model to use (default: tiny)")
    var model: String = "tiny"
    
    @Option(name: .shortAndLong, help: "Server URL (default: http://localhost:50060/v1)")
    var serverURL: String = "http://localhost:50060/v1"
    
    func run() async throws {
        let client = WhisperKitClient(serverURL: serverURL)
        try await client.transcribeAudio(filePath: audioFile, language: language, model: model)
    }
}

// MARK: - Translate Command

struct Translate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "translate",
        abstract: "Translate audio file to English"
    )
    
    @Argument(help: "Path to audio file")
    var audioFile: String
    
    @Option(name: .shortAndLong, help: "Source language (default: auto-detect)")
    var sourceLanguage: String = ""
    
    @Option(name: .shortAndLong, help: "Model to use (default: tiny)")
    var model: String = "tiny"
    
    @Option(name: .shortAndLong, help: "Server URL (default: http://localhost:50060/v1)")
    var serverURL: String = "http://localhost:50060/v1"
    
    func run() async throws {
        let client = WhisperKitClient(serverURL: serverURL)
        try await client.translateAudio(filePath: audioFile, sourceLanguage: sourceLanguage, model: model)
    }
}

// MARK: - Test Command

struct Test: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "test",
        abstract: "Test transcription and translation on sample files"
    )
    
    @Option(name: .shortAndLong, help: "Server URL (default: http://localhost:50060/v1)")
    var serverURL: String = "http://localhost:50060/v1"
    
    func run() async throws {
        let client = WhisperKitClient(serverURL: serverURL)
        try await client.runTests()
    }
}

// MARK: - Client Implementation

struct WhisperKitClient {
    private let serverURL: String
    private let client: Client
    
    init(serverURL: String = "http://localhost:50060/v1") {
        let normalized: String
        if serverURL.hasSuffix("/v1") {
            normalized = serverURL
        } else {
            normalized = serverURL.hasSuffix("/") ? serverURL + "v1" : serverURL + "/v1"
        }
        self.serverURL = normalized
        let url = URL(string: normalized)!
        let transport = URLSessionTransport()
        self.client = Client(serverURL: url, transport: transport)
    }
    
    // MARK: - Transcription
    
    func transcribeAudio(filePath: String, language: String, model: String) async throws {
        let fileURL = URL(fileURLWithPath: filePath)
        let data = try Data(contentsOf: fileURL)
        let filename = fileURL.lastPathComponent
        
        var parts: [Operations.createTranscription.Input.Body.multipartFormPayload] = [
            .file(.init(payload: .init(body: .init(data)), filename: filename)),
            .model(.init(payload: .init(body: .init(Data(model.utf8)))))
        ]
        
        if !language.isEmpty {
            parts.append(.language(.init(payload: .init(body: .init(Data(language.utf8))))))
        }
        
        let input = Operations.createTranscription.Input(headers: .init(), body: .multipartForm(.init(parts)))
        let output = try await client.createTranscription(input)
        
        switch output {
        case .ok(let ok):
            let body = try ok.body.json
            switch body {
            case .CreateTranscriptionResponseJson(let json):
                print("Transcription: \(json.text)")
            case .CreateTranscriptionResponseVerboseJson(let verbose):
                print("Transcription: \(verbose.text)")
            }
        case .undocumented(let status, _):
            throw WhisperKitError.serverError("Unexpected status: \(status)")
        }
    }
    
    // MARK: - Translation
    
    func translateAudio(filePath: String, sourceLanguage: String, model: String) async throws {
        let fileURL = URL(fileURLWithPath: filePath)
        let data = try Data(contentsOf: fileURL)
        let filename = fileURL.lastPathComponent
        
        var parts: [Operations.createTranslation.Input.Body.multipartFormPayload] = [
            .file(.init(payload: .init(body: .init(data)), filename: filename)),
            .model(.init(payload: .init(body: .init(Data(model.utf8)))))
        ]
        
        if !sourceLanguage.isEmpty {
            parts.append(.language(.init(payload: .init(body: .init(Data(sourceLanguage.utf8))))))
        }
        
        let input = Operations.createTranslation.Input(headers: .init(), body: .multipartForm(.init(parts)))
        let output = try await client.createTranslation(input)
        
        switch output {
        case .ok(let ok):
            let body = try ok.body.json
            switch body {
            case .CreateTranslationResponseJson(let json):
                print("Translation: \(json.text)")
            case .CreateTranslationResponseVerboseJson(let verbose):
                print("Translation: \(verbose.text)")
            }
        case .undocumented(let status, _):
            throw WhisperKitError.serverError("Unexpected status: \(status)")
        }
    }
    
    // MARK: - Testing
    
    func runTests() async throws {
        let resourcesPath = "../../../Tests/WhisperKitTests/Resources"
        let resourcesURL = URL(fileURLWithPath: resourcesPath, isDirectory: true)
        let fm = FileManager.default
        
        guard fm.fileExists(atPath: resourcesPath) else {
            print("Resources folder not found: \(resourcesPath)")
            return
        }
        
        let contents = try fm.contentsOfDirectory(at: resourcesURL, includingPropertiesForKeys: nil)
        let audioExtensions: Set<String> = ["wav", "m4a", "mp3", "flac", "aac"]
        let audioFiles = contents.filter { audioExtensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        
        print("Found \(audioFiles.count) audio files")
        
        for fileURL in audioFiles {
            let filename = fileURL.lastPathComponent
            print("\n--- Testing: \(filename) ---")
            
            // Try transcription
            do {
                try await transcribeAudio(filePath: fileURL.path, language: "", model: "tiny")
            } catch {
                print("Transcription failed: \(error)")
            }
            
            // Try translation if it's a non-English file
            if filename.contains("es_") || filename.contains("ja_") {
                do {
                    let src = filename.contains("es_") ? "es" : "ja"
                    try await translateAudio(filePath: fileURL.path, sourceLanguage: src, model: "tiny")
                } catch {
                    print("Translation failed: \(error)")
                }
            }
        }
    }
}

// MARK: - Error Types

enum WhisperKitError: Error, LocalizedError {
    case fileNotFound(String)
    case invalidResponse
    case serverError(String)
    
    var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "File not found: \(path)"
        case .invalidResponse:
            return "Invalid response from server"
        case .serverError(let message):
            return "Server error: \(message)"
        }
    }
}
