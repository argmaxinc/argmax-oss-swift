//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore

/// Tokenizer protocol for TTSKit, decoupled from ArgmaxCore internals.
///
/// `TTSTokenizer` defines the subset of tokenizer functionality needed by TTSKit.
/// Use `TTSTokenizerWrapper` to bridge a `TokenizerWrapper` to this protocol.
public protocol TTSTokenizer: Sendable {
    func encode(text: String) -> [Int]
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func decode(tokens: [Int]) -> String
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String
    func convertTokenToId(_ token: String) -> Int?
    func convertIdToToken(_ id: Int) -> String?
    var bosToken: String? { get }
    var bosTokenId: Int? { get }
    var eosToken: String? { get }
    var eosTokenId: Int? { get }
    var unknownToken: String? { get }
    var unknownTokenId: Int? { get }
}

/// Bridges a `TokenizerWrapper` to the `TTSTokenizer` protocol.
public struct TTSTokenizerWrapper: TTSTokenizer {
    private let impl: TokenizerWrapper

    public init(_ tokenizer: TokenizerWrapper) {
        self.impl = tokenizer
    }

    public func encode(text: String) -> [Int] { impl.encode(text: text) }
    public func encode(text: String, addSpecialTokens: Bool) -> [Int] { impl.encode(text: text, addSpecialTokens: addSpecialTokens) }
    public func decode(tokens: [Int]) -> String { impl.decode(tokens: tokens) }
    public func decode(tokens: [Int], skipSpecialTokens: Bool) -> String { impl.decode(tokens: tokens, skipSpecialTokens: skipSpecialTokens) }
    public func convertTokenToId(_ token: String) -> Int? { impl.convertTokenToId(token) }
    public func convertIdToToken(_ id: Int) -> String? { impl.convertIdToToken(id) }
    public var bosToken: String? { impl.bosToken }
    public var bosTokenId: Int? { impl.bosTokenId }
    public var eosToken: String? { impl.eosToken }
    public var eosTokenId: Int? { impl.eosTokenId }
    public var unknownToken: String? { impl.unknownToken }
    public var unknownTokenId: Int? { impl.unknownTokenId }
}
