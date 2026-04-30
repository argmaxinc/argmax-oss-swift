//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

// MARK: - Qwen3 TTS Constants

/// Qwen3 TTS model-specific constants: codec token IDs, vocabulary sizes,
/// model dimensions, cache geometry, and HuggingFace source locations.
///
/// These values are derived from the Qwen3 TTS architecture and must be updated
/// if the model is retrained with a different configuration.
///
/// Audio-format constants (`sampleRate`, `samplesPerFrame`) are here because
/// they match Qwen3's output format. Future model families with different audio
/// parameters should expose them via `SpeechDecoding.sampleRate` /
/// `SpeechDecoding.samplesPerFrame` rather than adding a second constants enum.
public enum Qwen3TTSConstants {
    // MARK: Codec track special tokens

    public static let codecPAD: Int32 = 2148
    public static let codecBOS: Int32 = 2149
    public static let codecEOS: Int32 = 2150
    public static let codecThink: Int32 = 2154
    public static let codecThinkBos: Int32 = 2156
    public static let codecThinkEos: Int32 = 2157

    // MARK: Text track special tokens

    public static let textPAD: Int32 = 151_671
    public static let textBOS: Int32 = 151_672

    // MARK: Vocabulary sizes

    /// Vocabulary size for the multi-code decoder heads (codes 1-15).
    public static let codecVocabSize: Int = 2048

    // MARK: Audio format

    public static let sampleRate: Int = 24000
    public static let samplesPerFrame: Int = 1920

    // MARK: Model dimensions

    /// Shared embedding dimension for all projectors and embedders.
    public static let embedDim: Int = 1024

    // MARK: KV cache geometry

    public static let cdCacheDim: Int = 28672
    public static let cdMaxSeq: Int = 256
    public static let mcdCacheDim: Int = 5120
    public static let mcdMaxSeq: Int = 64
    public static let sdCacheDim: Int = 8192
    public static let sdMaxSeq: Int = 256
    public static let sdHiddenDim: Int = 1024
    public static let sdHiddenContextLen: Int = 16

    // MARK: Default HuggingFace sources

    /// Tokenizer-compatible model on HuggingFace (must contain `tokenizer.json`).
    public static let defaultTokenizerRepo = "Qwen/Qwen3-0.6B"
    /// HuggingFace repo hosting the pre-compiled CoreML TTS models.
    public static let defaultModelRepo = "argmaxinc/ttskit-coreml"
    /// Default HuggingFace Hub endpoint. Override to point at a mirror or on-premise instance.
    public static let defaultEndpoint = "https://huggingface.co"
    /// Intermediate subdirectory inside the model repo grouping all Qwen3 TTS components.
    public static let modelFamilyDir = "qwen3_tts"
    /// Default model version directory, equivalent to `TTSModelPreset.qwen3TTS_0_6B.versionDir`.
    /// Provided as a stable exported constant; `TTSKitConfig` uses the preset's value directly.
    public static let defaultVersionDir = "12hz-0.6b-customvoice"

    // MARK: Token suppression

    /// Codec-0 token IDs suppressed during sampling: [2048, 3072) except EOS (2150).
    public static let suppressTokenIds: Set<Int> = {
        var ids = Set<Int>()
        for tokenId in 2048..<3072 where tokenId != Int(Qwen3TTSConstants.codecEOS) {
            ids.insert(tokenId)
        }
        return ids
    }()
}

// MARK: - Speaker

/// Qwen3 TTS speaker voices with their corresponding codec token IDs.
public enum Qwen3Speaker: String, CaseIterable, Sendable {
    case ryan, aiden
    case onoAnna = "ono-anna"
    case sohee, eric, dylan, serena, vivian
    case uncleFu = "uncle-fu"

    public var tokenID: Int32 {
        switch self {
            case .ryan: return 3061
            case .aiden: return 2861
            case .onoAnna: return 2873
            case .sohee: return 2864
            case .eric: return 2875
            case .dylan: return 2878
            case .serena: return 3066
            case .vivian: return 3065
            case .uncleFu: return 3010
        }
    }

    /// Human-readable display name (handles hyphenated raw values).
    public var displayName: String {
        switch self {
            case .ryan: return "Ryan"
            case .aiden: return "Aiden"
            case .onoAnna: return "Ono Anna"
            case .sohee: return "Sohee"
            case .eric: return "Eric"
            case .dylan: return "Dylan"
            case .serena: return "Serena"
            case .vivian: return "Vivian"
            case .uncleFu: return "Uncle Fu"
        }
    }

    /// Short description of the voice character and quality.
    public var voiceDescription: String {
        switch self {
            case .ryan: return "Dynamic male voice with strong rhythmic drive."
            case .aiden: return "Sunny American male voice with a clear midrange."
            case .onoAnna: return "Playful Japanese female voice with a light, nimble timbre."
            case .sohee: return "Warm Korean female voice with rich emotion."
            case .eric: return "Lively Chengdu male voice with a slightly husky brightness."
            case .dylan: return "Youthful Beijing male voice with a clear, natural timbre."
            case .serena: return "Warm, gentle young female voice."
            case .vivian: return "Bright, slightly edgy young female voice."
            case .uncleFu: return "Seasoned male voice with a low, mellow timbre."
        }
    }

    /// The speaker's native language (best quality when used with this language).
    public var nativeLanguage: String {
        switch self {
            case .ryan: return "English"
            case .aiden: return "English"
            case .onoAnna: return "Japanese"
            case .sohee: return "Korean"
            case .eric: return "Chinese (Sichuan)"
            case .dylan: return "Chinese (Beijing)"
            case .serena: return "Chinese"
            case .vivian: return "Chinese"
            case .uncleFu: return "Chinese"
        }
    }
}

// MARK: - Language

/// Qwen3 TTS supported languages with their corresponding codec token IDs.
public enum Qwen3Language: String, CaseIterable, Sendable {
    case english, chinese, japanese, korean, german, french, russian, portuguese, spanish, italian

    public var tokenID: Int32 {
        switch self {
            case .english: return 2050
            case .chinese: return 2055
            case .japanese: return 2058
            case .korean: return 2064
            case .german: return 2053
            case .french: return 2061
            case .russian: return 2069
            case .portuguese: return 2071
            case .spanish: return 2054
            case .italian: return 2070
        }
    }
}
