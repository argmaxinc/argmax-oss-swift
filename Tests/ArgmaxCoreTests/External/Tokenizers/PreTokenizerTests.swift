// Originally from: https://github.com/huggingface/swift-transformers
// Version: 1.1.6 (commit: 573e5c9036c2f136b3a8a071da8e8907322403d0)
// License: Apache 2.0 (https://github.com/huggingface/swift-transformers/blob/main/LICENSE)
// Copyright 2022 Hugging Face SAS
// Modified by Argmax, Inc. See Argmax-modification: comments for changes.
//

//
//  PreTokenizerTests.swift
//
//  Created by Jan Krukowski on 23/11/2023.
//

import Foundation
// Argmax-modification: consolidated the upstream swift-transformers module import(s) (Hub, Tokenizers) into the single ArgmaxCore module that now vendors this code.
@testable import ArgmaxCore
import Testing


@Suite("Pre-Tokenizer Tests")
struct PreTokenizerTests {
    @Test("Whitespace pre-tokenizer splits text by whitespace")
    func whitespacePreTokenizer() {
        let preTokenizer = WhitespacePreTokenizer(config: Config([String: Config]()))

        #expect(
            preTokenizer.preTokenize(text: "Hey friend!") == ["Hey", "friend!"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", "friend!", "How", "are", "you?!?",
            ]
        )
        #expect(
            preTokenizer.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                "Hey,", "friend,", "what's", "up?",
            ]
        )
    }

    @Test("Punctuation pre-tokenizer separates punctuation")
    func punctuationPreTokenizer() {
        let preTokenizer = PunctuationPreTokenizer(config: Config([String: Config]()))

        #expect(
            preTokenizer.preTokenize(text: "Hey friend!") == ["Hey friend", "!"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey friend", "!", "     How are you", "?!?",
            ]
        )
        #expect(
            preTokenizer.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                "   Hey", ",", "    friend", ",", "    what", "'", "s up", "?", "  ",
            ]
        )
    }

    @Test("Byte-level pre-tokenizer with various configurations")
    func byteLevelPreTokenizer() {
        let preTokenizer1 = ByteLevelPreTokenizer(config: Config([String: Config]()))

        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!") == ["Hey", "Ġfriend", "!"]
        )
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", "Ġfriend", "!", "ĠĠĠĠ", "ĠHow", "Ġare", "Ġyou", "?!?",
            ]
        )
        #expect(
            preTokenizer1.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                "ĠĠ", "ĠHey", ",", "ĠĠĠ", "Ġfriend", ",", "ĠĠĠ", "Ġwhat", "'s", "Ġup", "?", "ĠĠ",
            ]
        )

        let preTokenizer2 = ByteLevelPreTokenizer(config: Config(["addPrefixSpace": true]))

        #expect(
            preTokenizer2.preTokenize(text: "Hey friend!") == ["ĠHey", "Ġfriend", "Ġ!"]
        )
        #expect(
            preTokenizer2.preTokenize(text: "Hey friend!     How are you?!?") == [
                "ĠHey", "Ġfriend", "Ġ!", "ĠĠĠĠ", "ĠHow", "Ġare", "Ġyou", "Ġ?!?",
            ]
        )
        #expect(
            preTokenizer2.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                "ĠĠ", "ĠHey", "Ġ,", "ĠĠĠ", "Ġfriend", "Ġ,", "ĠĠĠ", "Ġwhat", "Ġ's", "Ġup", "Ġ?",
                "ĠĠ",
            ]
        )

        let preTokenizer3 = ByteLevelPreTokenizer(config: Config(["useRegex": false]))

        #expect(
            preTokenizer3.preTokenize(text: "Hey friend!") == ["HeyĠfriend!"]
        )
        #expect(
            preTokenizer3.preTokenize(text: "Hey friend!     How are you?!?") == [
                "HeyĠfriend!ĠĠĠĠĠHowĠareĠyou?!?"
            ]
        )
        #expect(
            preTokenizer3.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                "ĠĠĠHey,ĠĠĠĠfriend,ĠĠĠĠwhat'sĠup?ĠĠ"
            ]
        )
    }

    @Test("Digits pre-tokenizer handles numeric content")
    func digitsPreTokenizer() {
        let preTokenizer1 = DigitsPreTokenizer(config: Config([String: Config]()))

        #expect(
            preTokenizer1.preTokenize(text: "1 12 123! 1234abc") == [
                "1", " ", "12", " ", "123", "! ", "1234", "abc",
            ]
        )

        let preTokenizer2 = DigitsPreTokenizer(config: Config(["individualDigits": true]))

        #expect(
            preTokenizer2.preTokenize(text: "1 12 123! 1234abc") == [
                "1", " ", "1", "2", " ", "1", "2", "3", "! ", "1", "2", "3", "4", "abc",
            ]
        )
    }

    @Test("Split pre-tokenizer with string and regex patterns")
    func splitPreTokenizer() {
        let preTokenizer1 = SplitPreTokenizer(config: Config(["pattern": ["String": " "]]))
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!") == ["Hey", " ", "friend!"]
        )
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", " ", "friend!", " ", " ", " ", " ", " ", "How", " ", "are", " ", "you?!?",
            ]
        )
        #expect(
            preTokenizer1.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                " ", " ", " ", "Hey,", " ", " ", " ", " ", "friend,", " ", " ", " ", " ", "what's",
                " ", "up?", " ", " ",
            ]
        )

        let preTokenizer2 = SplitPreTokenizer(config: Config(["pattern": ["Regex": "\\s"]]))
        #expect(
            preTokenizer2.preTokenize(text: "Hey friend!") == ["Hey", " ", "friend!"]
        )
        #expect(
            preTokenizer2.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", " ", "friend!", " ", " ", " ", " ", " ", "How", " ", "are", " ", "you?!?",
            ]
        )
        #expect(
            preTokenizer2.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                " ", " ", " ", "Hey,", " ", " ", " ", " ", "friend,", " ", " ", " ", " ", "what's",
                " ", "up?", " ", " ",
            ]
        )

        let preTokenizer3 = SplitPreTokenizer(
            config: Config([
                "pattern": [
                    "Regex":
                        "(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                ], "invert": true,
            ]))
        #expect(
            preTokenizer3.preTokenize(text: "Hello") == ["Hello"]
        )

        #expect(
            preTokenizer3.preTokenize(text: "Hey friend!") == ["Hey", " friend", "!"]
        )
        #expect(
            preTokenizer3.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", " friend", "!", "    ", " How", " are", " you", "?!?",
            ]
        )
    }

    @Test("Split behavior merged with previous")
    func splitBehaviorMergedWithPrevious() {
        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) == ["the-", "final-", "-", "countdown"]
        )

        #expect(
            "the-final--countdown-".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) == ["the-", "final-", "-", "countdown-"]
        )

        #expect(
            "the-final--countdown--".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) == ["the-", "final-", "-", "countdown-", "-"]
        )

        #expect(
            "-the-final--countdown--".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) == ["-", "the-", "final-", "-", "countdown-", "-"]
        )

        #expect(
            "--the-final--countdown--".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) == ["-", "-", "the-", "final-", "-", "countdown-", "-"]
        )
    }

    @Test("Split behavior merged with next")
    func splitBehaviorMergedWithNext() {
        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext) == ["the", "-final", "-", "-countdown"]
        )

        #expect(
            "-the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext) == ["-the", "-final", "-", "-countdown"]
        )

        #expect(
            "--the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext) == ["-", "-the", "-final", "-", "-countdown"]
        )

        #expect(
            "--the-final--countdown-".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext) == ["-", "-the", "-final", "-", "-countdown", "-"]
        )
    }

    @Test("Split behavior other")
    func splitBehaviorOther() {
        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .isolated) == ["the", "-", "final", "-", "-", "countdown"]
        )

        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .removed) == ["the", "final", "countdown"]
        )
    }

    /// https://github.com/huggingface/tokenizers/pull/1357
    @Test("Metaspace pre-tokenizer with prefix space handling")
    func metaspacePreTokenizer() {
        // Prepend "always"
        let preTokenizer = MetaspacePreTokenizer(
            config: Config([
                "add_prefix_space": true,
                "replacement": "▁",
                "prepend_scheme": "always",
            ]))

        // TODO: different sections on <s>
        let text = "Hey my friend <s>how▁are you"
        let tokens =
            text
            .split(by: "<s>", includeSeparators: true)
            .flatMap { preTokenizer.preTokenize(text: $0) }

        #expect(
            tokens == ["▁Hey", "▁my", "▁friend", "▁", "▁<s>", "▁how", "▁are", "▁you"]
        )
    }

    @Test("BERT pre-tokenizer performs basic splitting")
    func bertPreTokenizer() {
        let preTokenizer1 = BertPreTokenizer(config: Config([String: Config]()))
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!") == ["Hey", "friend", "!"]
        )
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", "friend", "!", "How", "are", "you", "?", "!", "?",
            ]
        )
        #expect(
            preTokenizer1.preTokenize(text: "   Hey,    friend ,    what's up?  ") == [
                "Hey", ",", "friend", ",", "what", "\'", "s", "up", "?",
            ]
        )
        #expect(
            preTokenizer1.preTokenize(text: "   Hey,    friend ,  0 99  what's up?  ") == [
                "Hey", ",", "friend", ",", "0", "99", "what", "\'", "s", "up", "?",
            ]
        )
    }
}
