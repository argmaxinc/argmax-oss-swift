// Originally from: https://github.com/huggingface/swift-transformers
// Version: 1.1.6 (commit: 573e5c9036c2f136b3a8a071da8e8907322403d0)
// License: Apache 2.0 (https://github.com/huggingface/swift-transformers/blob/main/LICENSE)
// Copyright 2022 Hugging Face SAS
// Modified by Argmax, Inc. See Argmax-modification: comments for changes.
//

//
//  DecoderTests.swift
//
//  Created by Pedro Cuenca on 20231123.
//

import Foundation
// Argmax-modification: consolidated the upstream swift-transformers module import(s) (Hub, Tokenizers) into the single ArgmaxCore module that now vendors this code.
@testable import ArgmaxCore
import Testing


@Suite("Tokenizer Decoder Tests")
struct DecoderTests {
    /// https://github.com/huggingface/tokenizers/pull/1357
    @Test("Metaspace decoder with prefix space replacement")
    func metaspaceDecoder() {
        let decoder = MetaspaceDecoder(
            config: Config([
                "add_prefix_space": true,
                "replacement": "▁",
            ]))

        let tokens = ["▁Hey", "▁my", "▁friend", "▁", "▁<s>", "▁how", "▁are", "▁you"]
        let decoded = decoder.decode(tokens: tokens)

        #expect(
            decoded == ["Hey", " my", " friend", " ", " <s>", " how", " are", " you"]
        )
    }

    @Test("WordPiece decoder with prefix and cleanup")
    func wordPieceDecoder() {
        let config = Config(["prefix": "##", "cleanup": true])
        let decoder = WordPieceDecoder(config: config)

        let testCases: [([String], String)] = [
            (["##inter", "##national", "##ization"], "##internationalization"),
            (["##auto", "##mat", "##ic", "transmission"], "##automatic transmission"),
            (["who", "do", "##n't", "does", "n't", "can't"], "who don't doesn't can't"),
            (["##un", "##believ", "##able", "##fa", "##ntastic"], "##unbelievablefantastic"),
            (
                ["this", "is", "un", "##believ", "##able", "fa", "##ntastic"],
                "this is unbelievable fantastic"
            ),
            (["The", "##quick", "##brown", "fox"], "Thequickbrown fox"),
        ]

        for (tokens, expected) in testCases {
            let output = decoder.decode(tokens: tokens)
            #expect(output.joined() == expected)
        }
    }
}
