// For licensing see accompanying LICENSE.md file.
import ArgmaxCore
import CoreML
import Foundation
import WhisperKit
import XCTest

final class TokenizerAccessTests: XCTestCase {
    func testLocalModelLoadingAssignsProducerVariantAndTokenizer() async throws {
        try await withTokenizerRequestInterception {
            let folder = try makeFolder()
            let config = try makeModelConfig(at: folder, embedding: 512)
            let tokenizers = try XCTUnwrap(config.tokenizerFolder)
            let expected = tokenizers.appendingPathComponent("models/openai/whisper-base")
            try writeTokenizer(at: expected, endToken: 1000)
            try writeTokenizer(at: tokenizers.appendingPathComponent("models/openai/whisper-tiny"), endToken: 2000)
            config.load = true
            let kit = try await LegacyLoaderClient(config)
            XCTAssertEqual(kit.modelLoads, [false])
            XCTAssertEqual(kit.tokenizerLoads, 1)
            try await kit.prewarmModels()
            XCTAssertEqual(kit.modelLoads, [false, true])
            XCTAssertEqual(kit.tokenizerLoads, 1)
            kit.tokenizer = nil
            try await kit.loadModels()
            XCTAssertEqual(kit.modelLoads, [false, true, false])
            XCTAssertEqual(kit.tokenizerLoads, 2)
            kit.tokenizer = nil
            kit.textDecoder.isModelMultilingual = false
            try await kit.loadModels(localFilesOnly: true)
            XCTAssertEqual(kit.tokenizerLoads, 2)
            XCTAssertEqual(kit.tokenizerInputs.map(\.localFilesOnly), [false, false, true])
            for input in kit.tokenizerInputs {
                XCTAssertEqual(input.modelFolder?.path, folder.path)
                XCTAssertEqual(input.tokenizerFolder?.path, config.tokenizerFolder?.path)
            }
            XCTAssertEqual(kit.modelState, .loaded)
            XCTAssertEqual(kit.modelVariant, .base)
            XCTAssertTrue(kit.textDecoder.isModelMultilingual)
            let tokenizer = try XCTUnwrap(kit.tokenizer as? WhisperTokenizerWrapper)
            // Distinct repository fixtures expose which detected variant reached the utility.
            XCTAssertEqual(tokenizer.specialTokens.endToken, 1000)
            XCTAssertTrue(tokenizer === kit.textDecoder.tokenizer as? WhisperTokenizerWrapper)

            // One real lower-loader failure proves forwarding through the normal-import API.
            let missing = try await LegacyLoaderClient(makeModelConfig(at: makeFolder()))
            do {
                try await missing.loadModels(localFilesOnly: true)
                XCTFail("Expected installed-tokenizer refusal")
            } catch {
                XCTAssertEqual(error as? WhisperError, .tokenizerUnavailable("Installed tokenizer not found for tiny."))
            }
            XCTAssertEqual(missing.tokenizerInputs.map(\.localFilesOnly), [true])
            XCTAssertNil(missing.tokenizer)
            XCTAssertEqual(missing.modelState, .unloaded)
            XCTAssertEqual(TokenizerRequestInterceptor.requestCount, 0)
        }
    }

    private func makeModelConfig(at folder: URL, logits: Int = 51865, embedding: Int = 384) throws -> WhisperKitConfig {
        // The producer checks model paths before using injected components. No Core ML assets are loaded.
        for name in ["MelSpectrogram", "AudioEncoder", "TextDecoder"] {
            try FileManager.default.createDirectory(at: folder.appendingPathComponent("\(name).mlmodelc"), withIntermediateDirectories: true)
        }
        return WhisperKitConfig(
            modelFolder: folder.path,
            tokenizerFolder: folder.appendingPathComponent("tokenizers"),
            featureExtractor: LoaderFeatureExtractor(),
            audioEncoder: LoaderAudioEncoder(embedSize: embedding),
            textDecoder: LoaderTextDecoder(logitsSize: logits),
            load: false,
            download: false
        )
    }
}

// Normal imports and the original override signatures exercise downstream source compatibility.
private final class LegacyLoaderClient: WhisperKit {
    var modelLoads: [Bool] = []
    var tokenizerLoads = 0
    var tokenizerInputs: [(localFilesOnly: Bool, modelFolder: URL?, tokenizerFolder: URL?)] = []

    override func loadModels(prewarmMode: Bool = false) async throws {
        modelLoads.append(prewarmMode)
        try await super.loadModels(prewarmMode: prewarmMode)
    }

    override func loadTokenizerIfNeeded() async throws {
        tokenizerLoads += 1
        try await super.loadTokenizerIfNeeded()
    }

    override func loadTokenizerIfNeeded(localFilesOnly: Bool) async throws {
        tokenizerInputs.append((localFilesOnly, modelFolder, tokenizerFolder))
        try await super.loadTokenizerIfNeeded(localFilesOnly: localFilesOnly)
    }
}

private struct LoaderFeatureExtractor: FeatureExtracting {
    let melCount: Int? = nil
    let windowSamples: Int? = nil
    func logMelSpectrogram(fromAudio inputAudio: any AudioProcessorOutputType) async throws -> (any FeatureExtractorOutputType)? {
        XCTFail("Loading must not run inference")
        return nil
    }
}

private struct LoaderAudioEncoder: AudioEncoding {
    let embedSize: Int?
    func encodeFeatures(_ features: any FeatureExtractorOutputType) async throws -> (any AudioEncoderOutputType)? {
        XCTFail("Loading must not run inference")
        return nil
    }
}

private final class LoaderTextDecoder: TextDecoding {
    var tokenizer: WhisperTokenizer?
    var prefillData: WhisperMLModel?
    var isModelMultilingual = false
    var logitsFilters: [any LogitsFiltering]?
    let logitsSize: Int?
    let supportsWordTimestamps = false
    let kvCacheEmbedDim: Int? = nil
    let kvCacheMaxSequenceLength: Int? = nil
    let windowSize: Int? = nil
    let embedSize: Int? = nil

    init(logitsSize: Int) { self.logitsSize = logitsSize }

    func predictLogits(_ inputs: any TextDecoderInputType) async throws -> (any TextDecoderOutputType)? {
        XCTFail("Loading must not run inference")
        return nil
    }

    func decodeText(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: any DecodingInputsType,
        sampler tokenSampler: any TokenSampling,
        options decoderOptions: DecodingOptions,
        callback: TranscriptionCallback?
    ) async throws -> DecodingResult {
        XCTFail("Loading must not run inference")
        throw WhisperError.decodingFailed("Unexpected inference during loading")
    }

    func detectLanguage(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: any DecodingInputsType,
        sampler tokenSampler: any TokenSampling,
        options: DecodingOptions,
        temperature: FloatType
    ) async throws -> DecodingResult {
        XCTFail("Loading must not run inference")
        throw WhisperError.decodingFailed("Unexpected inference during loading")
    }
}
