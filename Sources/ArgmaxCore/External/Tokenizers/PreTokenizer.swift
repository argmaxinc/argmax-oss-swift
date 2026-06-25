// Originally from: https://github.com/huggingface/swift-transformers
// Version: 1.1.6 (commit: 573e5c9036c2f136b3a8a071da8e8907322403d0)
// License: Apache 2.0 (https://github.com/huggingface/swift-transformers/blob/main/LICENSE)
// Copyright 2022 Hugging Face SAS
// Modified by Argmax, Inc. See Argmax-modification: comments for changes.
//

//
//  PreTokenizer.swift
//
//
//  Created by Pedro Cuenca on 18/7/23.
//

import Foundation
// Argmax-modification: removed `import Hub` — Hub is part of ArgmaxCore

/// Options that can be passed to pre-tokenization operations.
// Argmax-modification: removed public from PreTokenizerOption, PreTokenizerOptions, and PreTokenizer — internal types in ArgmaxCore
enum PreTokenizerOption: String {
    /// Indicates this is the first section of text being processed.
    case firstSection
}

/// A set of pre-tokenization options.
// Argmax-modification: removed public from PreTokenizerOptions — internal type in ArgmaxCore
typealias PreTokenizerOptions = Set<PreTokenizerOption>

/// A protocol for pre-tokenization operations that split text before main tokenization.
///
/// Pre-tokenizers handle the initial splitting of text into chunks that will be
/// further processed by the main tokenization algorithm. Common operations include
/// splitting on whitespace, punctuation, or applying byte-level encoding.
// Argmax-modification: removed public from PreTokenizer — internal type in ArgmaxCore
protocol PreTokenizer {
    /// Pre-tokenizes a single text string.
    ///
    /// - Parameters:
    ///   - text: The input text to pre-tokenize
    ///   - options: Options controlling the pre-tokenization behavior
    /// - Returns: An array of pre-tokenized text chunks
    func preTokenize(text: String, options: PreTokenizerOptions) -> [String]

    /// Pre-tokenizes multiple text strings.
    ///
    /// - Parameters:
    ///   - texts: The input texts to pre-tokenize
    ///   - options: Options controlling the pre-tokenization behavior
    /// - Returns: An array of pre-tokenized text chunks from all inputs
    func preTokenize(texts: [String], options: PreTokenizerOptions) -> [String]

    /// Function call syntax for pre-tokenizing multiple texts.
    ///
    /// - Parameters:
    ///   - texts: The input texts to pre-tokenize
    ///   - options: Options controlling the pre-tokenization behavior
    /// - Returns: An array of pre-tokenized text chunks
    func callAsFunction(texts: [String], options: PreTokenizerOptions) -> [String]

    /// Function call syntax for pre-tokenizing a single text.
    ///
    /// - Parameters:
    ///   - text: The input text to pre-tokenize
    ///   - options: Options controlling the pre-tokenization behavior
    /// - Returns: An array of pre-tokenized text chunks
    func callAsFunction(text: String, options: PreTokenizerOptions) -> [String]

    /// Initializes the pre-tokenizer from configuration.
    ///
    /// - Parameter config: The configuration for this pre-tokenizer
    init(config: Config)
}

extension PreTokenizer {
    func preTokenize(texts: [String], options: PreTokenizerOptions = [.firstSection]) -> [String] {
        texts.flatMap { preTokenize(text: $0, options: options) }
    }

    func callAsFunction(texts: [String], options: PreTokenizerOptions = [.firstSection]) -> [String] {
        preTokenize(texts: texts, options: options)
    }

    func callAsFunction(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        preTokenize(text: text, options: options)
    }
}

enum PreTokenizerType: String {
    case Sequence
    case ByteLevel
    case Punctuation
    case Digits
    case Split
    case Whitespace
    case WhitespaceSplit
    case Metaspace
    case BertPreTokenizer
    /// Several more to be supported
    case Unknown = ""
}

struct PreTokenizerFactory {
    static func fromConfig(config: Config?) -> PreTokenizer? {
        guard let config else { return nil }
        guard let typeName = config.type.string() else { return nil }
        let type = PreTokenizerType(rawValue: typeName)
        switch type {
        case .Sequence: return PreTokenizerSequence(config: config)
        case .ByteLevel: return ByteLevelPreTokenizer(config: config)
        case .Punctuation: return PunctuationPreTokenizer(config: config)
        case .Digits: return DigitsPreTokenizer(config: config)
        case .Split: return SplitPreTokenizer(config: config)
        case .Whitespace, .WhitespaceSplit: return WhitespacePreTokenizer(config: config)
        case .Metaspace: return MetaspacePreTokenizer(config: config)
        case .BertPreTokenizer: return BertPreTokenizer(config: config)
        default: fatalError("Unsupported PreTokenizer type: \(typeName)")
        }
    }
}

class BertPreTokenizer: PreTokenizer {
    let re: String

    required init(config: Config) {
        // Ref: https://github.com/huggingface/transformers.js/blob/27920d84831e323275b38f0b5186644b7936e1a2/src/tokenizers.js#L1002
        re = "[^\\s\(Constants.PUNCTUATION_REGEX)]+|[\(Constants.PUNCTUATION_REGEX)]"
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        text.ranges(of: re).map { String(text[$0]) }
    }
}

class PreTokenizerSequence: PreTokenizer {
    let preTokenizers: [PreTokenizer]

    required init(config: Config) {
        guard let configs = config.pretokenizers.array() else { fatalError("No pretokenizers in Sequence") }
        preTokenizers = configs.compactMap { PreTokenizerFactory.fromConfig(config: $0) }
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        preTokenizers.reduce([text]) { current, preTokenizer in
            preTokenizer(texts: current, options: options)
        }
    }
}

class WhitespacePreTokenizer: PreTokenizer {
    let re: String

    required init(config: Config) {
        re = #"\S+"#
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        text.ranges(of: re).map { String(text[$0]) }
    }
}

/// PreTokenizer that replaces spaces with the given replacement character, adds a prefix space if requested,
class MetaspacePreTokenizer: PreTokenizer {
    /// Whether to add a prefix space to the first token
    let addPrefixSpace: Bool

    /// Replacement character
    let replacement: String

    /// Optional string representation of the replacement character.
    let stringReplacement: String

    enum PrependScheme: String {
        case first
        case never
        case always

        static var defaultScheme: PrependScheme { .always }
        static func from(rawValue value: String?) -> PrependScheme {
            guard let value else { return defaultScheme }
            return PrependScheme(rawValue: value) ?? defaultScheme
        }
    }

    /// The metaspace prepend scheme, see https://github.com/huggingface/tokenizers/pull/1357
    let prependScheme: PrependScheme

    required init(config: Config) {
        addPrefixSpace = config.addPrefixSpace.boolean(or: false)
        replacement = config.replacement.string(or: " ")
        stringReplacement = config.strRep.string(or: replacement)
        prependScheme = PrependScheme.from(rawValue: config.prependScheme.string())
    }

    /// https://github.com/huggingface/tokenizers/blob/accd0650b802f2180df40ef1def3bce32156688e/tokenizers/src/pre_tokenizers/metaspace.rs#L114
    /// https://github.com/xenova/transformers.js/blob/b07336d8f7ff57453cc164cc68aead2a79cbd57e/src/tokenizers.js#L2153
    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        let normalized = text.replacingOccurrences(of: " ", with: stringReplacement)

        // We add a prefix space if:
        //  (1) The addPrefixSpace option is enabled and the normalized
        //      token does not already start with the replacement character.
        //  and (2) either:
        //  (a) prependScheme is 'always'
        //  (b) prependScheme is 'first' and this is the first section
        // FIXME: (2b) always prepends, we are not passing section info

        var prepend = ""
        if addPrefixSpace, !normalized.hasPrefix(replacement) {
            if prependScheme == .always {
                prepend = stringReplacement
            }
            if prependScheme == .first, options.contains(.firstSection) {
                prepend = stringReplacement
            }
        }

        // Split in `MergedWithNext` mode, although usually the input to this function is already pre-tokenized
        // https://github.com/huggingface/tokenizers/blob/accd0650b802f2180df40ef1def3bce32156688e/tokenizers/src/pre_tokenizers/metaspace.rs#L127
        return (prepend + normalized).split(by: replacement, behavior: .mergedWithNext)
    }
}

class ByteLevelPreTokenizer: PreTokenizer {
    let addPrefixSpace: Bool
    let trimOffsets: Bool
    let useRegex: Bool
    let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#

    required init(config: Config) {
        addPrefixSpace = config.addPrefixSpace.boolean(or: false)
        trimOffsets = config.trimOffsets.boolean(or: true)
        useRegex = config.useRegex.boolean(or: true)
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        // Split on whitespace and punctuation
        let tokens = useRegex ? text.ranges(of: RE).map { String(text[$0]) } : [text]
        return tokens.map { token in
            if addPrefixSpace, !token.hasPrefix(" ") {
                return " " + token
            }
            return token
        }.map { token in
            Array(token.utf8).map { byteEncoder[$0]! }.joined()
        }
    }
}

class PunctuationPreTokenizer: PreTokenizer {
    let re: String

    required init(config: Config) {
        re = "[^\(Constants.PUNCTUATION_REGEX)]+|[\(Constants.PUNCTUATION_REGEX)]+"
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        // Ref: https://github.com/xenova/transformers.js/blob/27920d84831e323275b38f0b5186644b7936e1a2/src/tokenizers.js#L1138
        text.ranges(of: re).map { String(text[$0]) }
    }
}

class DigitsPreTokenizer: PreTokenizer {
    let re: String

    required init(config: Config) {
        let individualDigits = config.individualDigits.boolean(or: false)
        re = "[^\\d]+|\\d\(individualDigits ? "" : "+")"
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        text.ranges(of: re).map { String(text[$0]) }
    }
}

class SplitPreTokenizer: PreTokenizer {
    let pattern: StringSplitPattern?
    let invert: Bool

    required init(config: Config) {
        pattern = StringSplitPattern.from(config: config)
        invert = config.invert.boolean(or: false)
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        guard let pattern else { return [text] }
        return pattern.split(text, invert: invert)
    }
}
