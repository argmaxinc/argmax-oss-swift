// Originally from: https://github.com/huggingface/swift-transformers
// Version: 1.1.6 (commit: 573e5c9036c2f136b3a8a071da8e8907322403d0)
// License: Apache 2.0 (https://github.com/huggingface/swift-transformers/blob/main/LICENSE)
// Copyright 2022 Hugging Face SAS
// Modified by Argmax, Inc. See Argmax-modification: comments for changes.
//

//
//  TokenizerTests.swift
//
//  Created by Pedro Cuenca on July 2023.
//  Based on GPT2TokenizerTests by Julien Chaumond.
//  Copyright © 2023 Hugging Face. All rights reserved.
//

import Foundation
import Testing

// Argmax-modification: consolidated the upstream swift-transformers module import(s) (Hub, Models, Tokenizers) into the single ArgmaxCore module that now vendors this code.
@testable import ArgmaxCore

// Argmax-modification: trimmed to the Whisper tokenizers only. `tokenizer(spec:)` keeps the two
// Whisper specs; the other upstream @Test cases (gemmaUnicode, phi4, legacyLlamaBehaviour,
// robertaXLMTokenizer, nllbTokenizer, deepSeekPostProcessor, llamaPostProcessor,
// localTokenizerFromPretrained, bertCased, bertCasedResaved, bertUncased, robertaEncodeDecode,
// tokenizerBackend) and the `tokenizerFromLocalFolder` case were removed. The bundled
// `tokenizer_tests.json` is subset to the two Whisper keys.

// Argmax-modification: HubApi -> HubApiWrapper (public type taken by AutoTokenizer /
// LanguageModelConfigurationFromHub), and a unique cache dir per test (matching FactoryTests) instead
// of a shared `huggingface-tests` directory. swift-testing runs the parameterized cases in parallel, so
// a shared cache let them interleave Hub snapshot writes and reuse stale state across runs — flaky.
private func makeHubApi() -> (api: HubApiWrapper, downloadDestination: URL) {
    let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    let destination = base.appending(component: "huggingface-tests-\(UUID().uuidString)")
    return (HubApiWrapper(downloadBase: destination), destination)
}

private enum TestError: Error { case unsupportedTokenizer }

private struct Dataset: Decodable {
    let text: String
    // Bad naming, not just for bpe.
    // We are going to replace this testing method anyway.
    let bpe_tokens: [String]
    let token_ids: [Int]
    let decoded_text: String
}

private func loadDataset(filename: String) throws -> Dataset {
    let url = Bundle.module.url(forResource: filename, withExtension: "json")!
    let json = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    return try decoder.decode(Dataset.self, from: json)
}

private struct EdgeCase: Decodable {
    let input: String

    struct EncodedData: Decodable {
        let input_ids: [Int]
        let token_type_ids: [Int]?
        let attention_mask: [Int]
    }

    let encoded: EncodedData
    let decoded_with_special: String
    let decoded_without_special: String
}

private func loadEdgeCases(for hubModelName: String) throws -> [EdgeCase]? {
    let url = Bundle.module.url(forResource: "tokenizer_tests", withExtension: "json")!
    let json = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    let cases = try decoder.decode([String: [EdgeCase]].self, from: json)
    return cases[hubModelName]
}

// Argmax-modification: hubApi parameter type changed from HubApi to HubApiWrapper.
private func makeTokenizer(hubModelName: String, hubApi: HubApiWrapper) async throws -> PreTrainedTokenizer {
    let config = LanguageModelConfigurationFromHub(modelName: hubModelName, hubApi: hubApi)
    guard let tokenizerConfig = try await config.tokenizerConfig else {
        // Argmax-modification: TokenizerError.tokenizerConfigNotFound -> .missingConfig
        // (the vendored TokenizerError has no `tokenizerConfigNotFound` case).
        throw TokenizerError.missingConfig
    }
    let tokenizerData = try await config.tokenizerData
    let tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    guard let pretrained = tokenizer as? PreTrainedTokenizer else {
        throw TestError.unsupportedTokenizer
    }
    return pretrained
}

// MARK: -

struct ModelSpec: Sendable, CustomStringConvertible {
    let hubModelName: String
    let encodedSamplesFilename: String
    let unknownTokenId: Int?

    var description: String {
        hubModelName
    }

    init(_ hubModelName: String, _ encodedSamplesFilename: String, _ unknownTokenId: Int? = nil) {
        self.hubModelName = hubModelName
        self.encodedSamplesFilename = encodedSamplesFilename
        self.unknownTokenId = unknownTokenId
    }
}

// MARK: -

@Suite("Tokenizer Tests")
struct TokenizerTests {
    // Argmax-modification: trimmed the upstream 8-model argument list to the two Whisper tokenizers.
    @Test(arguments: [
        ModelSpec("openai/whisper-large-v2", "whisper_large_v2_encoded", 50257),
        ModelSpec("openai/whisper-tiny.en", "whisper_tiny_en_encoded", 50256),
    ])
    func tokenizer(spec: ModelSpec) async throws {
        let (hubApi, downloadDestination) = makeHubApi()
        defer { try? FileManager.default.removeItem(at: downloadDestination) }
        let tokenizer = try await makeTokenizer(hubModelName: spec.hubModelName, hubApi: hubApi)
        let dataset = try loadDataset(filename: spec.encodedSamplesFilename)

        #expect(tokenizer.tokenize(text: dataset.text) == dataset.bpe_tokens)
        #expect(tokenizer.encode(text: dataset.text) == dataset.token_ids)
        #expect(tokenizer.decode(tokens: dataset.token_ids) == dataset.decoded_text)

        // Edge cases (if available)
        if let edgeCases = try? loadEdgeCases(for: spec.hubModelName) {
            for edgeCase in edgeCases {
                #expect(tokenizer.encode(text: edgeCase.input) == edgeCase.encoded.input_ids)
                #expect(tokenizer.decode(tokens: edgeCase.encoded.input_ids) == edgeCase.decoded_with_special)
                #expect(tokenizer.decode(tokens: edgeCase.encoded.input_ids, skipSpecialTokens: true) == edgeCase.decoded_without_special)
            }
        }

        // Unknown token checks
        let model = tokenizer.model
        #expect(model.unknownTokenId == spec.unknownTokenId)
        #expect(model.unknownTokenId == model.convertTokenToId("_this_token_does_not_exist_"))
        if let unknownTokenId = model.unknownTokenId {
            #expect(model.unknownToken == model.convertIdToToken(unknownTokenId))
        } else {
            #expect(model.unknownTokenId == nil)
        }
    }
}
