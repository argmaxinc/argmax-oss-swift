// Originally from: https://github.com/huggingface/swift-transformers
// Version: 1.1.6 (commit: 573e5c9036c2f136b3a8a071da8e8907322403d0)
// License: Apache 2.0 (https://github.com/huggingface/swift-transformers/blob/main/LICENSE)
// Copyright 2022 Hugging Face SAS
// Modified by Argmax, Inc. See Argmax-modification: comments for changes.
//

//
//  TrieTests.swift
//
//
//  Created by Pedro Cuenca on 12/1/24.
//

import Foundation
import Testing

// Argmax-modification: consolidated the upstream swift-transformers module import(s) (Tokenizers) into the single ArgmaxCore module that now vendors this code.
@testable import ArgmaxCore

@Suite("Trie data structure functionality")
struct TrieTests {
    @Test("Trie building and traversal")
    func trieBuilding() {
        // https://guillaume-be.github.io/2020-05-30/sentence_piece
        let trie = Trie<Character>()
        trie.insert("cat")
        trie.insert("carp")
        trie.insert("car")
        #expect(trie.root.children.count == 1)

        let c = trie.get("c")
        #expect(c != nil)
        #expect(c!.children.count == 1) // "a"

        let ca = trie.get("ca")
        #expect(ca != nil)
        #expect(ca!.children.count == 2) // "r", "t"

        let car = trie.get("car")
        #expect(car != nil)
        #expect(car!.isLeaf)
        #expect(!ca!.isLeaf)

        #expect(trie.get("card") == nil)
    }

    @Test("Trie common prefix search")
    func trieCommonPrefixSearch() {
        // https://guillaume-be.github.io/2020-05-30/sentence_piece
        let trie = Trie<Character>()
        trie.insert("cat")
        trie.insert("carp")
        trie.insert("car")

        // trie.commonPrefixSearch returns [Character] not String
        let leaves = trie.commonPrefixSearch("carpooling").map { String($0) }
        #expect(leaves == ["car", "carp"])
    }

    @Test("Trie common prefix search iterator")
    func trieCommonPrefixSearchIterator() {
        // https://guillaume-be.github.io/2020-05-30/sentence_piece
        let trie = Trie<Character>()
        trie.insert("cat")
        trie.insert("carp")
        trie.insert("car")

        var expected = Set(["car", "carp"])
        for leaf in trie.commonPrefixSearchIterator("carpooling").map({ String($0) }) {
            #expect(expected.contains(leaf))
            expected.remove(leaf)
        }
        #expect(expected.count == 0)
    }
}
