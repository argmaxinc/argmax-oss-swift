// Originally from: https://github.com/huggingface/swift-transformers
// Version: 1.1.6 (commit: 573e5c9036c2f136b3a8a071da8e8907322403d0)
// License: Apache 2.0 (https://github.com/huggingface/swift-transformers/blob/main/LICENSE)
// Copyright 2022 Hugging Face SAS
// Modified by Argmax, Inc. See Argmax-modification: comments for changes.
//

//
//  FactoryTests.swift
//
//
//  Created by Pedro Cuenca on 4/8/23.
//

import Foundation
// Argmax-modification: consolidated the upstream swift-transformers module import(s) (Hub, Tokenizers) into the single ArgmaxCore module that now vendors this code.
@testable import ArgmaxCore
import Testing


// Argmax-modification: return type changed from HubApi to the public HubApiWrapper.
// `AutoTokenizer.from(pretrained:hubApi:)` / `snapshot(from:)` now take HubApiWrapper.
private func makeHubApi() -> (api: HubApiWrapper, downloadDestination: URL) {
    let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    let destination = base.appending(component: "huggingface-tests-\(UUID().uuidString)")
    return (HubApiWrapper(downloadBase: destination), destination)
}

// Argmax-modification: removed the `fromPretrained` and `fromModelFolder` cases.
@Suite("Factory")
struct FactoryTests {
    @Test
    func whisper() async throws {
        let (hubApi, downloadDestination) = makeHubApi()
        defer { try? FileManager.default.removeItem(at: downloadDestination) }

        let tokenizer = try await AutoTokenizer.from(pretrained: "openai/whisper-large-v2", hubApi: hubApi)
        let inputIds = tokenizer("Today she took a train to the West")
        #expect(inputIds == [50258, 50363, 27676, 750, 1890, 257, 3847, 281, 264, 4055, 50257])
    }

    // Argmax-modification: removed `fromModelFolder`.

    @Test
    func whisperFromModelFolder() async throws {
        let (hubApi, downloadDestination) = makeHubApi()
        defer { try? FileManager.default.removeItem(at: downloadDestination) }

        let filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        // Argmax-modification: Hub.Repo -> HubApiWrapper.Repo (snapshot now takes the public wrapper type)
        let repo = HubApiWrapper.Repo(id: "openai/whisper-large-v2")
        let localModelFolder = try await hubApi.snapshot(from: repo, matching: filesToDownload)

        let tokenizer = try await AutoTokenizer.from(modelFolder: localModelFolder, hubApi: hubApi)
        let inputIds = tokenizer("Today she took a train to the West")
        #expect(inputIds == [50258, 50363, 27676, 750, 1890, 257, 3847, 281, 264, 4055, 50257])
    }
}
