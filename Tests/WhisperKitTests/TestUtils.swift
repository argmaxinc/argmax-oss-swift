//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import AVFAudio
import Combine
import CoreML
import Foundation
@testable import WhisperKit
import XCTest

enum TestError: Error {
    case missingFile(String)
    case missingDirectory(String)
}

@discardableResult
func XCTUnwrapAsync<T>(
    _ expression: @autoclosure () async throws -> T,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line
) async throws -> T {
    let evaluated = try? await expression()
    return try XCTUnwrap(evaluated, message(), file: file, line: line)
}

@discardableResult
func XCTUnwrapAsync<T>(
    _ expression: @autoclosure () async throws -> T?,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line
) async throws -> T {
    let evaluated = try? await expression()
    return try XCTUnwrap(evaluated, message(), file: file, line: line)
}

func XCTAssertNoThrowAsync<T>(
    _ expression: @autoclosure () async throws -> T,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        _ = try await expression()
    } catch {
        XCTFail(message(), file: file, line: line)
    }
}

func XCTAssertNoThrowAsync<T>(
    _ expression: @autoclosure () async throws -> T?,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        _ = try await expression()
    } catch {
        XCTFail(message(), file: file, line: line)
    }
}

func XCTAssertNoThrowAsync(
    _ expression: @autoclosure () async throws -> Void,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        try await expression()
    } catch {
        XCTFail(message(), file: file, line: line)
    }
}

// MARK: Helpers

extension Bundle {
    static func current(for classObject: AnyObject? = nil) -> Bundle {
        #if SWIFT_PACKAGE
        return Bundle.module
        #else
        // Use bundle for class type if passed in
        if let classObject = classObject {
            return Bundle(for: type(of: classObject))
        } else {
            return Bundle.main
        }
        #endif
    }
}

extension FileManager {
    func allocatedSizeOfDirectory(at url: URL) throws -> Int64 {
        guard let enumerator = enumerator(at: url, includingPropertiesForKeys: [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey]) else {
            throw NSError(domain: NSCocoaErrorDomain, code: NSFileReadUnknownError, userInfo: nil)
        }

        var accumulatedSize: Int64 = 0
        for case let fileURL as URL in enumerator {
            let resourceValues = try fileURL.resourceValues(forKeys: [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey])
            accumulatedSize += Int64(resourceValues.totalFileAllocatedSize ?? resourceValues.fileAllocatedSize ?? 0)
        }
        return accumulatedSize
    }
}

extension MLMultiArray {
    /// Create `MLMultiArray` of shape [1, 1, arr.count] and fill up the last
    /// dimension with with values from arr.
    static func logits(_ arr: [FloatType]) throws -> MLMultiArray {
        let logits = try MLMultiArray(shape: [1, 1, arr.count] as [NSNumber], dataType: .float16)
        let ptr = UnsafeMutablePointer<FloatType>(OpaquePointer(logits.dataPointer))
        for (index, value) in arr.enumerated() {
            let linearOffset = logits.linearOffset(for: [0, 0, index])
            ptr[linearOffset] = value
        }
        return logits
    }

    /// Get the data from `MLMultiArray` for given dimension
    func data(for dimension: Int) -> [FloatType] {
        let count = shape[dimension].intValue
        let indexes = stride(from: 0, to: count, by: 1).map { [0, 0, $0] }
        var result = [FloatType]()
        let ptr = UnsafeMutablePointer<FloatType>(OpaquePointer(dataPointer))
        for index in indexes {
            let linearOffset = linearOffset(for: index)
            result.append(ptr[linearOffset])
        }
        return result
    }
}

extension XCTestCase {
    func transcribe(
        with variant: ModelVariant,
        options: DecodingOptions,
        callback: TranscriptionCallback? = nil,
        audioFile: String = "jfk.wav",
        file: StaticString = #file,
        line: UInt = #line
    ) async throws -> [TranscriptionResult] {
        let modelName: String
        switch variant {
            case .largev3:
                modelName = "large-v3"
            default:
                modelName = "tiny"
        }
        let config = WhisperKitConfig(model: modelName, verbose: true, logLevel: .debug)
        let whisperKit = try await WhisperKit(config)
        trackForMemoryLeaks(on: whisperKit, file: file, line: line)

        let audioComponents = audioFile.components(separatedBy: ".")
        guard let audioFileURL = Bundle.current(for: self).path(forResource: audioComponents.first, ofType: audioComponents.last) else {
            throw TestError.missingFile("Missing audio file")
        }
        return try await whisperKit.transcribe(audioPath: audioFileURL, decodeOptions: options, callback: callback)
    }

    func tinyModelPath() async throws -> String {
        let modelDir = try await WhisperKit.download(variant: "tiny").path()
        return modelDir
    }

    func largev3ModelPath() throws -> String {
        let modelDir = "whisperkit-coreml/openai_whisper-large-v3" // use faster to compile model for tests
        guard let modelPath = Bundle.current(for: self).urls(forResourcesWithExtension: "mlmodelc", subdirectory: modelDir)?.first?.deletingLastPathComponent().path else {
            throw TestError.missingFile("Failed to load model, ensure \"Models/\(modelDir)\" exists via Makefile command: `make download-models`")
        }
        return modelPath
    }

    func largev3TurboModelPath() throws -> String {
        let modelDir = "whisperkit-coreml/openai_whisper-large-v3_turbo"
        guard let modelPath = Bundle.current(for: self).urls(forResourcesWithExtension: "mlmodelc", subdirectory: modelDir)?.first?.deletingLastPathComponent().path else {
            throw TestError.missingFile("Failed to load model, ensure \"Models/\(modelDir)\" exists via Makefile command: `make download-models`")
        }
        return modelPath
    }

    func allModelPaths() throws -> [String] {
        let fileManager = FileManager.default
        var modelPaths: [String] = []
        let directory = "whisperkit-coreml"
        let resourceKeys: [URLResourceKey] = [.isDirectoryKey]
        guard let baseurl = Bundle.current(for: self).resourceURL?.appendingPathComponent(directory) else {
            throw TestError.missingDirectory("Base URL for directory \(directory) not found.")
        }
        let directoryContents = try fileManager.contentsOfDirectory(at: baseurl, includingPropertiesForKeys: resourceKeys, options: .skipsHiddenFiles)
        for folderURL in directoryContents {
            let resourceValues = try folderURL.resourceValues(forKeys: Set(resourceKeys))
            if resourceValues.isDirectory == true {
                // Check if the directory contains actual data files, or if it contains pointer files.
                // As a proxy, use the MelSpectrogramc.mlmodel/coredata.bin file.
                let proxyFileToCheck = folderURL.appendingPathComponent("MelSpectrogram.mlmodelc/coremldata.bin")
                if try isGitLFSPointerFile(url: proxyFileToCheck) {
                    continue
                }

                // Check if the directory name contains the quantization pattern
                // Only test large quantized models
                let dirName = folderURL.lastPathComponent
                if !(dirName.contains("q") && !dirName.contains("large")) {
                    modelPaths.append(folderURL.absoluteString)
                }
            }
        }
        return modelPaths
    }

    /// Function to check if the beginning of the file matches a Git LFS pointer pattern
    func isGitLFSPointerFile(url: URL) throws -> Bool {
        let fileHandle = try FileHandle(forReadingFrom: url)
        // Read the first few bytes of the file to get enough for the Git LFS pointer signature
        let data = fileHandle.readData(ofLength: 512) // Read first 512 bytes
        fileHandle.closeFile()
        if let string = String(data: data, encoding: .utf8),
           string.starts(with: "version https://git-lfs.github.com/")
        {
            return true
        }
        return false
    }

    func trackForMemoryLeaks(on instance: AnyObject, file: StaticString = #filePath, line: UInt = #line) {
        /// Stores only a weak reference for teardown leak assertions.
        /// `XCTestCase.addTeardownBlock` uses a sending closure, so this wrapper must be Sendable.
        final class LeakRefWrapper: @unchecked Sendable {
            weak var object: AnyObject?

            init(object: AnyObject) {
                self.object = object
            }
        }

        let wrapper = LeakRefWrapper(object: instance)
        addTeardownBlock { [wrapper, file, line] in
            XCTAssertNil(wrapper.object, "Detected potential memory leak", file: file, line: line)
        }
    }

    /// Helper to create an extended audio buffer by repeating the original buffer
    func createExtendedBuffer(from originalBuffer: AVAudioPCMBuffer, repeatCount: Int) -> AVAudioPCMBuffer {
        let frameCount = originalBuffer.frameLength
        let totalFrames = frameCount * AVAudioFrameCount(repeatCount)

        // Create new buffer with same format but longer length
        let extendedBuffer = AVAudioPCMBuffer(
            pcmFormat: originalBuffer.format,
            frameCapacity: totalFrames
        )!
        extendedBuffer.frameLength = totalFrames

        // For each channel
        for channel in 0..<originalBuffer.format.channelCount {
            if let sourceData = originalBuffer.floatChannelData?[Int(channel)],
               let targetData = extendedBuffer.floatChannelData?[Int(channel)]
            {
                // Use vDSP to fill the extended buffer with repeated copies
                for i in 0..<repeatCount {
                    let targetOffset = Int(i * Int(frameCount))

                    // Use vDSP_mmov to copy memory blocks efficiently
                    vDSP_mmov(
                        sourceData, // Source pointer
                        targetData.advanced(by: targetOffset), // Destination pointer
                        vDSP_Length(frameCount), // Frame count
                        1, // Number of channels (always 1 here since we're processing per channel)
                        1, // Source stride
                        1 // Destination stride
                    )
                }
            }
        }

        return extendedBuffer
    }

    /// Helper to create a buffer out of a multi-channel audio file preserving the number of channels
    func loadMultichannelAudio(fromPath audioFilePath: String) throws -> AVAudioPCMBuffer {
        guard FileManager.default.fileExists(atPath: audioFilePath) else {
            throw WhisperError.loadAudioFailed("Resource path does not exist \(audioFilePath)")
        }

        let audioFileURL = URL(fileURLWithPath: audioFilePath)

        // Create an audio file with original format preserved
        let audioFile = try AVAudioFile(forReading: audioFileURL)

        // Create a buffer with the original format (preserving all channels)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat,
                                            frameCapacity: AVAudioFrameCount(audioFile.length))
        else {
            throw WhisperError.loadAudioFailed("Unable to create audio buffer")
        }

        // Read the entire file into the buffer
        try audioFile.read(into: buffer)

        return buffer
    }

    /// Helper to measure channel processing operations
    func measureChannelProcessing(buffer: AVAudioPCMBuffer, mode: AudioInputConfig.ChannelMode, iterations: Int = 5) -> Double {
        // Add warm-up iterations
        for _ in 0..<3 {
            _ = AudioProcessor.convertToMono(buffer, mode: mode)
        }

        var totalTime: Double = 0
        // Then measure the actual timing
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            _ = AudioProcessor.convertToMono(buffer, mode: mode)
            let end = CFAbsoluteTimeGetCurrent()
            totalTime += (end - start)
        }

        return totalTime / Double(iterations)
    }
}

extension SpecialTokens {
    static func `default`(
        endToken: Int = 0,
        englishToken: Int = 0,
        noSpeechToken: Int = 0,
        noTimestampsToken: Int = 0,
        specialTokenBegin: Int = 0,
        startOfPreviousToken: Int = 0,
        startOfTranscriptToken: Int = 0,
        timeTokenBegin: Int = 0,
        transcribeToken: Int = 0,
        translateToken: Int = 0,
        whitespaceToken: Int = 0
    ) -> SpecialTokens {
        SpecialTokens(
            endToken: endToken,
            englishToken: englishToken,
            noSpeechToken: noSpeechToken,
            noTimestampsToken: noTimestampsToken,
            specialTokenBegin: specialTokenBegin,
            startOfPreviousToken: startOfPreviousToken,
            startOfTranscriptToken: startOfTranscriptToken,
            timeTokenBegin: timeTokenBegin,
            transcribeToken: transcribeToken,
            translateToken: translateToken,
            whitespaceToken: whitespaceToken
        )
    }
}

extension Result {
    var isSuccess: Bool {
        switch self {
            case .success:
                return true
            case .failure:
                return false
        }
    }

    func whisperError() -> WhisperError? {
        switch self {
            case .success:
                return nil
            case let .failure(error):
                return error as? WhisperError
        }
    }
}

extension Result where Success == [TranscriptionResult] {
    func normalizedText(prefix: Int) throws -> String {
        try get().text.normalized.split(separator: " ").prefix(prefix).joined(separator: " ")
    }
}

extension Collection where Element == TranscriptionResult {
    var text: String {
        map(\.text).joined(separator: " ")
    }
}

extension Collection where Element == TranscriptionResult {
    var segments: [TranscriptionSegment] {
        flatMap(\.segments)
    }
}

public extension Publisher {
    func withPrevious() -> AnyPublisher<(previous: Output?, current: Output), Failure> {
        scan((Output?, Output)?.none) { ($0?.1, $1) }
            .compactMap { $0 }
            .eraseToAnyPublisher()
    }
}
