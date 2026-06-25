//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

/// Public opaque wrapper around an internal `Tokenizer` implementation.
///
/// `TokenizerWrapper` is the public surface for tokenizer access from outside `ArgmaxCore`.
/// The underlying `Tokenizer` protocol and all its implementations are internal to this module.
/// Use `AutoTokenizerWrapper` to create instances.
public struct TokenizerWrapper: Sendable {
    let impl: any Tokenizer

    init(_ tokenizer: any Tokenizer) {
        self.impl = tokenizer
    }

    public func tokenize(text: String) -> [String] { impl.tokenize(text: text) }
    public func encode(text: String) -> [Int] { impl.encode(text: text) }
    public func encode(text: String, addSpecialTokens: Bool) -> [Int] { impl.encode(text: text, addSpecialTokens: addSpecialTokens) }
    public func decode(tokens: [Int]) -> String { impl.decode(tokens: tokens) }
    public func decode(tokens: [Int], skipSpecialTokens: Bool) -> String { impl.decode(tokens: tokens, skipSpecialTokens: skipSpecialTokens) }
    public func convertTokenToId(_ token: String) -> Int? { impl.convertTokenToId(token) }
    public func convertTokensToIds(_ tokens: [String]) -> [Int?] { impl.convertTokensToIds(tokens) }
    public func convertIdToToken(_ id: Int) -> String? { impl.convertIdToToken(id) }
    public func convertIdsToTokens(_ ids: [Int]) -> [String?] { impl.convertIdsToTokens(ids) }
    public var bosToken: String? { impl.bosToken }
    public var bosTokenId: Int? { impl.bosTokenId }
    public var eosToken: String? { impl.eosToken }
    public var eosTokenId: Int? { impl.eosTokenId }
    public var unknownToken: String? { impl.unknownToken }
    public var unknownTokenId: Int? { impl.unknownTokenId }
}
