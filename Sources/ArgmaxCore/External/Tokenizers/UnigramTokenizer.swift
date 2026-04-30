// Originally from: https://github.com/huggingface/swift-transformers
// Version: 1.1.6 (commit: 573e5c9036c2f136b3a8a071da8e8907322403d0)
// License: Apache 2.0 (https://github.com/huggingface/swift-transformers/blob/main/LICENSE)
// Copyright 2022 Hugging Face SAS
// Modified by Argmax, Inc. See Argmax-modification: comments for changes.
//

//
//  UnigramTokenizer.swift
//
//
//  Created by Pedro Cuenca on 20240131.
//  Copyright © 2024 Hugging Face. All rights reserved.
//

import Foundation
// Argmax-modification: removed `import Hub` — Hub is part of ArgmaxCore

/// A Unigram tokenizer implementation based on the SentencePiece algorithm.
///
/// Unigram tokenizers use a probabilistic approach where each token has a score,
/// and the tokenization process finds the most probable segmentation of the input text.
/// This is commonly used in models like T5 and XLM-RoBERTa.
class UnigramTokenizer: PreTrainedTokenizerModel, @unchecked Sendable {
    /// A token with its associated score in the Unigram model.
    struct SentencePieceToken {
        var token: String
        var score: Float
    }

    /// The complete vocabulary of tokens with their scores.
    let vocab: [SentencePieceToken]

    /// The special token used for unknown/out-of-vocabulary text.
    let unknownPiece: SentencePieceToken

    /// The score associated with the unknown token.
    var unknownTokenScore: Float { unknownPiece.score }

    /// The numeric ID of the unknown token.
    let unknownTokenId: Int?

    /// The unknown token string.
    var unknownToken: String? { unknownPiece.token }

    /// The minimum score found in the vocabulary (used for score calculations).
    let minScore: Float

    /// Mapping from token strings to their numeric IDs.
    let tokensToIds: [NSString: Int]

    /// The beginning-of-sequence token (hardcoded as space for Unigram).
    let bosToken: String? = " "

    /// The numeric ID of the beginning-of-sequence token.
    let bosTokenId: Int?

    /// The end-of-sequence token string, if defined.
    let eosToken: String?

    /// The numeric ID of the end-of-sequence token, if defined.
    let eosTokenId: Int?

    /// Whether consecutive unknown tokens should be fused (always true for Unigram).
    let fuseUnknownTokens: Bool = true

    private let trie: Trie<Character>

    /// Initializes a Unigram tokenizer from configuration data.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration
    ///   - tokenizerData: The tokenizer data containing vocabulary and scores
    ///   - addedTokens: Additional tokens to include in the vocabulary
    /// - Throws: `TokenizerError` if the vocabulary is missing or malformed
    required init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws {
        guard let configVocab = tokenizerData.model.vocab.array() else {
            throw TokenizerError.missingVocab
        }

        vocab = try configVocab.map { piece in
            let tuple = piece.array(or: [])

            guard let token = tuple.first?.string(),
                let scoreValue = tuple.last
            else {
                throw TokenizerError.malformedVocab
            }

            let score: Float
            if let floatScore = scoreValue.floating() {
                score = floatScore
            } else if let numberScore = scoreValue.integer() {
                score = Float(numberScore)

            } else {
                throw TokenizerError.malformedVocab
            }

            return SentencePieceToken(token: token, score: score)
        }

        minScore = vocab.reduce(999) { partial, token in
            min(partial, token.score)
        }

        guard let unknownTokenId = tokenizerData.model["unkId"].integer() else { throw TokenizerError.malformedVocab }
        self.unknownTokenId = unknownTokenId
        unknownPiece = SentencePieceToken(token: vocab[unknownTokenId].token, score: minScore - 10)

        tokensToIds = Dictionary(uniqueKeysWithValues: vocab.map { $0.token as NSString }.enumerated().map { ($1, $0) })
        bosTokenId = tokensToIds[bosToken! as NSString] // May be nil

        eosToken = tokenizerConfig.eosToken.string()
        eosTokenId = eosToken == nil ? nil : tokensToIds[eosToken! as NSString]

        trie = Trie()
        trie.append(contentsOf: vocab.map { $0.token })
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
    /// - Returns: The token string
    func convertIdToToken(_ id: Int) -> String? {
        vocab[id].token
    }

    /// Tokenizes input text using the Unigram algorithm with dynamic programming.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of token strings representing the most probable segmentation
    func tokenize(text: String) -> [String] {
        var lattice = TokenLattice(sentence: text, bosTokenId: bosTokenId ?? 0, eosTokenId: eosTokenId ?? 0)

        // Populate nodes
        let sentence = lattice.sentence
        var beginPos = 0
        while beginPos < sentence.count {
            let mblen = 1
            var hasSingleNode = false

            let beginIndex = sentence.index(sentence.startIndex, offsetBy: beginPos)
            for token in trie.commonPrefixSearchIterator(sentence[beginIndex...]).map({ String($0) }) {
                guard let tokenId = tokensToIds[token as NSString] else { fatalError("Token not in vocab: \(token)") }
                let tokenScore = vocab[tokenId].score
                lattice.insert(startOffset: beginPos, length: token.count, score: tokenScore, tokenId: tokenId)
                if !hasSingleNode, token.count == mblen {
                    hasSingleNode = true
                }
            }
            if !hasSingleNode {
                lattice.insert(startOffset: beginPos, length: mblen, score: unknownTokenScore, tokenId: unknownTokenId ?? 0)
            }
            beginPos += mblen
        }

        return lattice.tokens
    }
}
