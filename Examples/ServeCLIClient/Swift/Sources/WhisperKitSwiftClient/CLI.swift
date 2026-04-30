//  Copyright © 2025 Argmax, Inc. All rights reserved.
//  For licensing see accompanying LICENSE.md file.

import Foundation
import ArgumentParser
import OpenAPIRuntime
import OpenAPIURLSession
import HTTPTypes

// MARK: - Main Command

@available(macOS 13, *)
@main
public struct WhisperKitSwiftClient: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "whisperkit-client",
        abstract: "WhisperKit Swift Client for local server",
        subcommands: [
            Transcribe.self,
            Translate.self,
            Test.self
        ]
    )
    
    public init() {}
    
    public func run() async throws {
        print("Use one of the subcommands: transcribe, translate, or test")
    }
}

// MARK: - Transcribe Command

public struct Transcribe: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe audio file"
    )
    
    public init() {}
    
    @Argument(help: "Path to audio file")
    public var audioFile: String
    
    @Option(name: .shortAndLong, help: "Source language (default: auto-detect)")
    public var language: String = ""
    
    @Option(name: .shortAndLong, help: "Model to use (default: tiny)")
    public var model: String = "tiny"
    
    @Option(name: .long, help: "Response format (default: verbose_json)")
    public var responseFormat: String = "verbose_json"
    
    @Option(name: .long, help: "Timestamp granularities (comma-separated: word,segment)")
    public var timestampGranularities: String = ""
    
    @Flag(name: .long, help: "Enable streaming output")
    public var stream: Bool = false
    
    @Option(name: .shortAndLong, help: "Server URL (default: http://localhost:50060/v1)")
    public var serverURL: String = "http://localhost:50060/v1"
    
    public func run() async throws {
        let client = WhisperKitClient(serverURL: serverURL)
        try await client.transcribeAudio(
            filePath: audioFile, 
            language: language, 
            model: model,
            responseFormat: responseFormat,
            timestampGranularities: timestampGranularities,
            stream: stream
        )
    }
}

// MARK: - Translate Command

public struct Translate: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "translate",
        abstract: "Translate audio file to English"
    )
    
    public init() {}
    
    @Argument(help: "Path to audio file")
    public var audioFile: String
    
    @Option(name: .long, help: "Language (default: auto-detect)")
    public var language: String = ""
    
    @Option(name: .long, help: "Model to use (default: tiny)")
    public var model: String = "tiny"
    
    @Option(name: .long, help: "Response format (default: verbose_json)")
    public var responseFormat: String = "verbose_json"
    
    @Option(name: .shortAndLong, help: "Server URL (default: http://localhost:50060/v1)")
    public var serverURL: String = "http://localhost:50060/v1"
    
    public func run() async throws {
        let client = WhisperKitClient(serverURL: serverURL)
        try await client.translateAudio(
            filePath: audioFile, 
            language: language, 
            model: model,
            responseFormat: responseFormat
        )
    }
}

// MARK: - Test Command

public struct Test: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "test",
        abstract: "Test transcription and translation on sample files"
    )
    
    public init() {}
    
    @Option(name: .shortAndLong, help: "Server URL (default: http://localhost:50060/v1)")
    public var serverURL: String = "http://localhost:50060/v1"
    
    public func run() async throws {
        let client = WhisperKitClient(serverURL: serverURL)
        try await client.runTests()
    }
}

// MARK: - Client Implementation

public struct WhisperKitClient {
    private let serverURL: String
    private let client: Client
    
    public init(serverURL: String = "http://localhost:50060/v1") {
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
    
    public func transcribeAudio(
        filePath: String, 
        language: String, 
        model: String,
        responseFormat: String,
        timestampGranularities: String,
        stream: Bool
    ) async throws {
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
        
        if !responseFormat.isEmpty {
            parts.append(.response_format(.init(payload: .init(body: .init(Data(responseFormat.utf8))))))
        }
        
        if !timestampGranularities.isEmpty {
            parts.append(.timestamp_granularities_lbrack__rbrack_(.init(payload: .init(body: .init(Data(timestampGranularities.utf8))))))
        }
        
        if stream {
            parts.append(.stream(.init(payload: .init(body: .init(Data("true".utf8))))))
        }
        
        let input = Operations.createTranscription.Input(headers: .init(), body: .multipartForm(.init(parts)))
        let output = try await client.createTranscription(input)
        
        if stream {
            // Handle streaming response
            try await handleStreamingTranscription(output)
        } else {
            // Handle non-streaming response
            try await handleNonStreamingTranscription(output)
        }
    }
    
    private func handleStreamingTranscription(_ output: Operations.createTranscription.Output) async throws {
        switch output {
        case .ok(let ok):
            let body = try ok.body.text_event_hyphen_stream
            print("🔄 Starting streaming transcription...")
            
            // Parse Server-Sent Events manually
            var buffer = ""
            for try await chunk in body {
                let chunkString = String(data: Data(chunk), encoding: .utf8) ?? ""
                buffer += chunkString
                
                // Process complete lines
                while let newlineIndex = buffer.firstIndex(of: "\n") {
                    let line = String(buffer[..<newlineIndex]).trimmingCharacters(in: .whitespaces)
                    buffer = String(buffer[buffer.index(after: newlineIndex)...])
                    
                    if line.hasPrefix("data: ") {
                        let dataString = String(line.dropFirst(6)) // Remove "data: " prefix
                        if !dataString.isEmpty {
                            do {
                                let eventData = try JSONSerialization.jsonObject(with: Data(dataString.utf8)) as? [String: Any]
                                if let eventType = eventData?["type"] as? String {
                                    switch eventType {
                                    case "transcript.text.delta":
                                        if let delta = eventData?["delta"] as? String {
                                            print("🔄 \(delta)")
                                        }
                                    case "transcript.text.done":
                                        if let text = eventData?["text"] as? String {
                                            print("\n✅ Final transcription: \(text)")
                                        }
                                    default:
                                        print("📝 Event: \(eventData ?? [:])")
                                    }
                                }
                            } catch {
                                print("⚠️ JSON decode error: \(error)")
                            }
                        }
                    }
                }
            }
        case .undocumented(let status, _):
            throw WhisperKitError.serverError("Unexpected status: \(status)")
        }
    }
    
    private func handleNonStreamingTranscription(_ output: Operations.createTranscription.Output) async throws {
        switch output {
        case .ok(let ok):
            let body = try ok.body.json

            switch body {
            case .CreateTranscriptionResponseJson(let json):
                print("📝 Transcription: \(json.text)")
            case .CreateTranscriptionResponseVerboseJson(let verbose):
                print("📝 Transcription: \(verbose.text)")
                
                // Display detailed information for verbose_json format
                print("\n📊 Detailed Information:")
                print("   Language: \(verbose.language)")
                print("   Duration: \(verbose.duration) seconds")
                
                // Show segments if available
                if let segments = verbose.segments, !segments.isEmpty {
                    print("   Segments: \(segments.count)")
                    for (i, segment) in segments.prefix(3).enumerated() {
                        print("     Segment \(i+1): \(String(format: "%.2f", segment.start))s - \(String(format: "%.2f", segment.end))s")
                        print("       Text: \(segment.text)")
                        print("       Confidence: \(String(format: "%.3f", segment.avg_logprob))")
                    }
                }
                
                // Show words if available
                if let words = verbose.words, !words.isEmpty {
                    print("   Words: \(words.count)")
                    print("     All words with timestamps:")
                    for word in words {
                        let start = String(format: "%.2f", word.start)
                        let end = String(format: "%.2f", word.end)
                        print("       \(start)s - \(end)s: '\(word.word)'")
                    }
                }
            }
        case .undocumented(let status, _):
            throw WhisperKitError.serverError("Unexpected status: \(status)")
        }
    }
    
    // MARK: - Translation
    
    public func translateAudio(
        filePath: String, 
        language: String, 
        model: String,
        responseFormat: String
    ) async throws {
        let fileURL = URL(fileURLWithPath: filePath)
        let data = try Data(contentsOf: fileURL)
        let filename = fileURL.lastPathComponent
        
        var parts: [Operations.createTranslation.Input.Body.multipartFormPayload] = [
            .file(.init(payload: .init(body: .init(data)), filename: filename)),
            .model(.init(payload: .init(body: .init(Data(model.utf8)))))
        ]
        
        if !language.isEmpty {
            parts.append(.language(.init(payload: .init(body: .init(Data(language.utf8))))))
        }
        
        if !responseFormat.isEmpty {
            parts.append(.response_format(.init(payload: .init(body: .init(Data(responseFormat.utf8))))))
        }
        
        let input = Operations.createTranslation.Input(headers: .init(), body: .multipartForm(.init(parts)))
        let output = try await client.createTranslation(input)
        
        switch output {
        case .ok(let ok):
            let body = try ok.body.json
            switch body {
            case .CreateTranslationResponseJson(let json):
                print("🌐 Translation: \(json.text)")
            case .CreateTranslationResponseVerboseJson(let verbose):
                print("🌐 Translation: \(verbose.text)")
                
                // Display detailed information for verbose_json format
                print("\n📊 Detailed Information:")
                print("   Output Language: \(verbose.language)")
                print("   Duration: \(verbose.duration) seconds")
                
                // Show segments if available
                if let segments = verbose.segments, !segments.isEmpty {
                    print("   Segments: \(segments.count)")
                    for (i, segment) in segments.prefix(3).enumerated() {
                        print("     Segment \(i+1): \(String(format: "%.2f", segment.start))s - \(String(format: "%.2f", segment.end))s")
                        print("       Text: \(segment.text)")
                        print("       Confidence: \(String(format: "%.3f", segment.avg_logprob))")
                    }
                }
            }
        case .undocumented(let status, _):
            throw WhisperKitError.serverError("Unexpected status: \(status)")
        }
    }
    
    // MARK: - Testing
    
    func testLogprobs() async -> Bool {
        print("🧪 Testing transcription with logprobs...")
        
        // Find test audio files
        let resourcesPath = "../../../Tests/WhisperKitTests/Resources"
        let resourcesURL = URL(fileURLWithPath: resourcesPath)
        
        guard let contents = try? FileManager.default.contentsOfDirectory(at: resourcesURL, includingPropertiesForKeys: nil),
              let testFile = contents.first(where: { $0.pathExtension.lowercased() == "wav" }) else {
            print("❌ No test audio files found")
            return false
        }
        
        print("📁 Using test file: \(testFile.lastPathComponent)")
        
        do {
            let audioData = try Data(contentsOf: testFile)
            let filename = testFile.lastPathComponent
            
            var parts: [Operations.createTranscription.Input.Body.multipartFormPayload] = [
                .file(.init(payload: .init(body: .init(audioData)), filename: filename)),
                .model(.init(payload: .init(body: .init(Data("tiny".utf8))))),
                .response_format(.init(payload: .init(body: .init(Data("json".utf8))))),
                .include_lbrack__rbrack_(.init(payload: .init(body: .init(Data("logprobs".utf8)))))
            ]
            
            let input = Operations.createTranscription.Input(headers: .init(), body: .multipartForm(.init(parts)))
            let response = try await client.createTranscription(input)
            
            switch response {
            case .ok(let okResponse):
                switch okResponse.body {
                case .json(let jsonPayload):
                    switch jsonPayload {
                    case .CreateTranscriptionResponseJson(let transcription):
                        if let logprobs = transcription.logprobs, !logprobs.isEmpty {
                            print("✅ Logprobs received: \(logprobs.count) tokens")
                            
                            // Display first few tokens with their logprobs
                            for (index, tokenInfo) in logprobs.prefix(5).enumerated() {
                                print("  Token \(index + 1): '\(tokenInfo.token)' - logprob: \(tokenInfo.logprob)")
                            }
                            
                            if logprobs.count > 5 {
                                print("  ... and \(logprobs.count - 5) more tokens")
                            }
                            
                            return true
                        } else {
                            print("❌ No logprobs in response")
                            return false
                        }
                    case .CreateTranscriptionResponseVerboseJson(let transcription):
                        // Verbose response doesn't have logprobs, only basic response does
                        print("❌ Verbose response format doesn't support logprobs")
                        return false
                    }
                case .text_event_hyphen_stream:
                    print("❌ Unexpected streaming response")
                    return false
                }
            case .undocumented(let status, _):
                print("❌ Server error: \(status)")
                return false
            }
        } catch {
            print("❌ Error testing logprobs: \(error)")
            return false
        }
    }

    public func runTests() async throws {
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
                try await transcribeAudio(
                    filePath: fileURL.path, 
                    language: "", 
                    model: "tiny",
                    responseFormat: "verbose_json",
                    timestampGranularities: "word,segment",
                    stream: false
                )
            } catch {
                print("Transcription failed: \(error)")
            }
            
            // Try translation if it's a non-English file
            if filename.contains("es_") || filename.contains("ja_") {
                do {
                    // let src = filename.contains("es_") ? "es" : "ja"
                    try await translateAudio(
                        filePath: fileURL.path, 
                        language: "en", 
                        model: "tiny",
                        responseFormat: "verbose_json"
                    )
                } catch {
                    print("Translation failed: \(error)")
                }
            }
        }
        
        // Test logprobs functionality
        print("\n--- Testing Logprobs ---")
        _ = await testLogprobs()
    }
}

// MARK: - Error Types

public enum WhisperKitError: Error, LocalizedError {
    case fileNotFound(String)
    case invalidResponse
    case serverError(String)
    
    public var errorDescription: String? {
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
