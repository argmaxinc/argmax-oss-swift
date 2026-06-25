// Originally from: https://github.com/huggingface/swift-transformers
// Version: 1.1.6 (commit: 573e5c9036c2f136b3a8a071da8e8907322403d0)
// License: Apache 2.0 (https://github.com/huggingface/swift-transformers/blob/main/LICENSE)
// Copyright 2022 Hugging Face SAS
// Modified by Argmax, Inc. See Argmax-modification: comments for changes.
//

//
//  Utils.swift
//  AudioBoloss
//
//  Created by Julien Chaumond on 07/01/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation

struct Utils {
    // Argmax-modification: removed time(_:), time(label:_:), dateNow(), clamp(_:_:_:), fakeThrowable(_:) — unused in this repo

    /// Substring
    static func substr(_ s: String, _ r: Range<Int>) -> String? {
        let stringCount = s.count
        if stringCount < r.upperBound || stringCount < r.lowerBound {
            return nil
        }
        let startIndex = s.index(s.startIndex, offsetBy: r.lowerBound)
        let endIndex = s.index(startIndex, offsetBy: r.upperBound - r.lowerBound)
        return String(s[startIndex..<endIndex])
    }

    /// Invert a (k, v) dictionary
    static func invert<K, V>(_ dict: [K: V]) -> [V: K] {
        var inverted: [V: K] = [:]
        for (k, v) in dict {
            inverted[v] = k
        }
        return inverted
    }

    /// Checks if a character is considered Chinese
    /// https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    static func isChineseChar(_ c: UnicodeScalar) -> Bool {
        (c.value >= 0x4E00 && c.value <= 0x9FFF) || (c.value >= 0x3400 && c.value <= 0x4DBF) || (c.value >= 0x20000 && c.value <= 0x2A6DF) || (c.value >= 0x2A700 && c.value <= 0x2B73F) || (c.value >= 0x2B740 && c.value <= 0x2B81F) || (c.value >= 0x2B820 && c.value <= 0x2CEAF) || (c.value >= 0xF900 && c.value <= 0xFAFF) || (c.value >= 0x2F800 && c.value <= 0x2FA1F)
    }
}

enum Constants {
    static let PUNCTUATION_REGEX = #"\p{P}\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"#
}
