// Originally from: https://github.com/huggingface/swift-transformers
// Version: 1.1.6 (commit: 573e5c9036c2f136b3a8a071da8e8907322403d0)
// License: Apache 2.0 (https://github.com/huggingface/swift-transformers/blob/main/LICENSE)
// Copyright 2022 Hugging Face SAS
//

//
//  Trie.swift
//
//
//  Created by Pedro Cuenca on 20240112.
//  Copyright © 2024 Hugging Face. All rights reserved.
//

import Foundation

struct Trie<T: Hashable> {
    typealias Node = TrieNode<T>

    var root: Node

    init(root: Node? = nil) {
        self.root = root ?? Node()
    }
}

extension Trie {
    func insert(_ element: any Sequence<T>) {
        var node = root
        for item in element {
            if let child = node.children[item] {
                node = child
            } else {
                let child = Node()
                node.children[item] = child
                node = child
            }
        }
        node.isLeaf = true
    }

    func append(contentsOf container: any Sequence<any Sequence<T>>) {
        for t in container {
            insert(t)
        }
    }

    /// Find all leaf nodes that share a common prefix with the input sequence (usually a text)
    /// Returns an array
    func commonPrefixSearch(_ text: any Sequence<T>) -> [[T]] {
        var node = root
        var seqs: [[T]] = []
        var seq: [T] = []
        for item in text {
            seq.append(item)
            guard let child = node.children[item] else { return seqs }
            node = child
            if node.isLeaf {
                seqs.append(seq)
            }
        }
        return seqs
    }

    /// Find all leaf nodes that share a common prefix with the input sequence (usually a text)
    /// Returns an iterator
    func commonPrefixSearchIterator(_ text: any Sequence<T>) -> LeavesWithCommonPrefixIterator<T> {
        LeavesWithCommonPrefixIterator(node: root, text: text)
    }
}

extension Trie {
    /// Only used for testing, could migrate to collection
    func get(_ element: any Sequence<T>) -> Node? {
        var node = root
        for item in element {
            guard let child = node.children[item] else { return nil }
            node = child
        }
        return node
    }
}

// TODO: maybe store the scores here if it's helpful?
class TrieNode<T: Hashable> {
    var isLeaf: Bool = false
    var children: [T: TrieNode] = [:]
}

struct LeavesWithCommonPrefixIterator<T: Hashable>: Sequence, IteratorProtocol {
    var node: TrieNode<T>
    var text: any Sequence<T>
    var seq: [T] = []
    lazy var iterator = text.makeIterator() as any IteratorProtocol<T>

    mutating func next() -> [T]? {
        while true {
            guard let item = iterator.next() else { return nil }
            seq.append(item)
            guard let child = node.children[item] else { return nil }
            node = child
            if node.isLeaf {
                return seq
            }
        }
    }
}
