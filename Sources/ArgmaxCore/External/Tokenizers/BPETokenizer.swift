// Originally from: https://github.com/huggingface/swift-transformers
// Version: 1.1.6 (commit: 573e5c9036c2f136b3a8a071da8e8907322403d0)
// License: Apache 2.0 (https://github.com/huggingface/swift-transformers/blob/main/LICENSE)
// Copyright 2022 Hugging Face SAS
// Modified by Argmax, Inc. See Argmax-modification: comments for changes.
//

//
//  BPETokenizer.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 18/07/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation
// Argmax-modification: removed `import Hub` — Hub is part of ArgmaxCore

/// A pair of byte/token strings used in Byte-Pair Encoding (BPE) merge operations.
struct BytePair: Hashable, Sendable {
    let a: String
    let b: String
    init(_ a: String, _ b: String) {
        self.a = a
        self.b = b
    }

    init(tuple: [String]) {
        a = tuple[0]
        b = tuple[1]
    }

    static func == (lhs: BytePair, rhs: BytePair) -> Bool {
        lhs.a == rhs.a && lhs.b == rhs.b
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(a)
        hasher.combine(b)
    }
}

/// A Byte-Pair Encoding (BPE) tokenizer implementation.
///
/// BPE tokenizers learn to merge the most frequently occurring pairs of characters
/// or character sequences. This implementation supports various BPE-based models
/// including GPT-2, RoBERTa, and other transformer models.
class BPETokenizer: PreTrainedTokenizerModel, @unchecked Sendable {
    let bpeRanks: [BytePair: Int]
    private let tokensToIds: [NSString: Int]
    private let idsToTokens: [Int: NSString]

    /// The total number of tokens in the vocabulary.
    var vocabCount: Int { tokensToIds.count }

    /// The beginning-of-sequence token string, if defined.
    let bosToken: String?

    /// The numeric ID of the beginning-of-sequence token, if defined.
    let bosTokenId: Int?

    /// The end-of-sequence token string, if defined.
    let eosToken: String?

    /// The numeric ID of the end-of-sequence token, if defined.
    let eosTokenId: Int?

    /// The unknown token string used for out-of-vocabulary words.
    let unknownToken: String?

    /// The numeric ID of the unknown token.
    let unknownTokenId: Int?

    /// Whether consecutive unknown tokens should be fused together.
    let fuseUnknownTokens: Bool

    static func mergesFromConfig(_ config: Config?) -> [[String]]? {
        guard let config else { return nil }

        if let merges = config.array() {
            return merges.reduce(into: [[String]]()) { result, element in
                if let val: [String] = element.get() { // New format (pushed with tokenizers >= 0.20.0): each merge is a list of 2 items
                    result.append(val)
                }
                if let val: String = element.get() { // legacy
                    result.append(val.unicodeScalars.split(separator: " ", omittingEmptySubsequences: false).map { String($0) })
                }
            }
        }

        return nil
    }

    /// Initializes a BPE tokenizer from configuration data.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration
    ///   - tokenizerData: The tokenizer data containing vocabulary and merges
    ///   - addedTokens: Additional tokens to include in the vocabulary
    /// - Throws: `TokenizerError` if required configuration is missing
    required init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws {
        guard let merges = Self.mergesFromConfig(tokenizerData.model.merges) else { fatalError("BPETokenizer requires merges") }
        guard let vocab = tokenizerData.model.vocab.dictionary() else {
            throw TokenizerError.missingVocab
        }
        var bpeRanks: [BytePair: Int] = [:]
        for (i, merge) in merges.enumerated() {
            let bp = BytePair(tuple: merge)
            bpeRanks[bp] = i
        }
        self.bpeRanks = bpeRanks

        let addedTokens = addedTokens.reduce(into: [BinaryDistinctString: Config]()) { result, element in
            result[BinaryDistinctString(element.key)] = .init(element.value)
        }
        tokensToIds = vocab.merging(addedTokens) { $1 }.reduce(into: [NSString: Int]()) { result, element in
            result[element.key.nsString] = element.value.integer()
        }

        idsToTokens = Utils.invert(tokensToIds)

        // Populate tokens
        if let unknownToken = TokenizerModel.unknownToken(from: tokenizerConfig) {
            self.unknownToken = unknownToken
            unknownTokenId = tokensToIds[unknownToken as NSString]
        } else {
            unknownToken = nil
            unknownTokenId = nil
        }

        eosToken = addedTokenAsString(tokenizerConfig.eosToken)
        eosTokenId = eosToken == nil ? nil : tokensToIds[eosToken! as NSString]

        bosToken = addedTokenAsString(tokenizerConfig.bosToken)
        bosTokenId = bosToken == nil ? nil : tokensToIds[bosToken! as NSString]

        fuseUnknownTokens = tokenizerConfig.fuseUnk.boolean(or: false)
    }

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID, or the unknown token ID if not found
    func convertTokenToId(_ token: String) -> Int? {
        tokensToIds[token as NSString] ?? unknownTokenId
    }

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string, or nil if the ID is invalid
    func convertIdToToken(_ id: Int) -> String? {
        idsToTokens[id] as String?
    }

    func byteEncode(text: String) -> [String] {
        let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.map { token -> String in
            return Array(token.utf8).compactMap { byteEncoder[$0] }.joined()
        }
    }

    func hexaEncode(text: String) -> [String] {
        let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.flatMap { token -> [String] in
            return Array(token.utf8).map { String(format: "<0x%02X>", $0) }
        }
    }

    private func getPairs(word: [String]) -> Set<BytePair> {
        var s = Set<BytePair>()
        for i in 0..<word.count - 1 {
            let bp = BytePair(
                word[i],
                word[i + 1]
            )
            s.insert(bp)
        }
        return s
    }

    func bpe(token: String) -> String {
        if token.count <= 1 {
            return token
        }

        var word = Array(token).map { String($0) }
        var pairs = Array(getPairs(word: word))

        while true {
            let bigrams = pairs.filter { bp -> Bool in bpeRanks[bp] != nil }
            if bigrams.count == 0 {
                break
            }
            let bigram = bigrams.min { bp1, bp2 -> Bool in
                return bpeRanks[bp1]! < bpeRanks[bp2]!
            }!
            let first = bigram.a
            let second = bigram.b
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if let j = word[i..<word.count].firstIndex(of: first) {
                    newWord.append(contentsOf: word[i..<j])
                    i = j
                } else {
                    newWord.append(contentsOf: word[i..<word.count])
                    break
                }
                if word[i] == first, i < word.count - 1, word[i + 1] == second {
                    newWord.append(first + second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
            if word.count == 1 {
                break
            } else {
                pairs = Array(getPairs(word: word))
            }
        }
        return word.joined(separator: " ")
    }

    /// Tokenizes input text using the BPE algorithm.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of BPE token strings
    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        let bpeTokens = bpe(token: text).split(separator: " ").map { String($0) }
        for token in bpeTokens {
            if convertTokenToId(token) != unknownTokenId {
                tokens.append(token)
            } else {
                // TODO: if config.byte_fallback is False, append the unknown token instead
                tokens.append(contentsOf: hexaEncode(text: token))
            }
        }
        return tokens
    }
}
