//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

/// Public factory for loading tokenizers, mirroring the `AutoTokenizer` API.
///
/// `AutoTokenizerWrapper` is the public surface for tokenizer creation from outside `ArgmaxCore`.
/// The underlying `AutoTokenizer` enum is internal to this module.
public enum AutoTokenizerWrapper {
    /// Loads a tokenizer from a pre-trained model on the Hugging Face Hub.
    ///
    /// - Parameters:
    ///   - model: The model identifier (e.g., "openai/whisper-tiny")
    ///   - hubApi: The Hub API wrapper to use for downloading
    ///   - strict: Whether to enforce strict validation of the tokenizer class
    /// - Returns: A `TokenizerWrapper` wrapping the loaded tokenizer
    public static func from(
        pretrained model: String,
        hubApi: HubApiWrapper = .shared,
        strict: Bool = true
    ) async throws -> TokenizerWrapper {
        let tok = try await AutoTokenizer.from(pretrained: model, hubApi: hubApi, strict: strict)
        return TokenizerWrapper(tok)
    }

    /// Loads a tokenizer from a local model folder.
    ///
    /// - Parameters:
    ///   - modelFolder: The URL path to the local model folder containing tokenizer files
    ///   - hubApi: The Hub API wrapper to use for config parsing
    ///   - strict: Whether to enforce strict validation of the tokenizer class
    /// - Returns: A `TokenizerWrapper` wrapping the loaded tokenizer
    public static func from(
        modelFolder: URL,
        hubApi: HubApiWrapper = .shared,
        strict: Bool = true
    ) async throws -> TokenizerWrapper {
        let tok = try await AutoTokenizer.from(modelFolder: modelFolder, hubApi: hubApi, strict: strict)
        return TokenizerWrapper(tok)
    }
}
