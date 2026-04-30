// Originally from: https://github.com/huggingface/swift-transformers
// Version: 1.1.6 (commit: 573e5c9036c2f136b3a8a071da8e8907322403d0)
// License: Apache 2.0 (https://github.com/huggingface/swift-transformers/blob/main/LICENSE)
// Copyright 2022 Hugging Face SAS
//

//
//  JSONSerialization+BOM.swift
//  swift-transformers
//
//  Created by Pedro Cuenca on 20250912
//

import Foundation

/// Extensions to JSONSerialization for handling BOM (Byte Order Mark) sequences.
extension JSONSerialization {
    /// Parses JSON data while preserving BOM sequences that appear after quotes.
    ///
    /// This method works around issues with certain tokenizer files (like Gemma) that contain
    /// BOM sequences after quotes, which JSONSerialization incorrectly interprets as encoding markers.
    ///
    /// - Parameters:
    ///   - data: The JSON data to parse
    ///   - options: JSON reading options (defaults to empty)
    /// - Returns: The parsed JSON object
    /// - Throws: JSONSerialization errors if parsing fails
    class func bomPreservingJsonObject(with data: Data, options: JSONSerialization.ReadingOptions = []) throws -> Any {
        try JSONSerialization.jsonObject(with: data.duplicatingBOMsAfterQuotes, options: options)
    }
}

private extension Data {
    /// Workaround for BOM handling in JSON parsing.
    ///
    /// Duplicates BOM sequences that follow quotes to prevent JSONSerialization from incorrectly
    /// consuming them as encoding markers. This is needed for tokenizer files like Gemma that
    /// contain BOM sequences as part of the token content.
    ///
    /// See: https://github.com/huggingface/swift-transformers/issues/116
    ///
    /// - Returns: Data with duplicated BOM sequences after quotes
    var duplicatingBOMsAfterQuotes: Data {
        withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let src = raw.bindMemory(to: UInt8.self)
            var out = [UInt8]()
            // We expect very few matches (only 6 for Gemma)
            out.reserveCapacity(src.count + 1000)

            var i = 0
            while i < src.count {
                let b = src[i]
                out.append(b)

                // Check for \u{feff} BOM (observed in Gemma tokenizers), which is encoded as 0xef 0xbb 0xbf.
                // We may need more combinations.
                if b == 0x22, i + 3 < src.count,
                    src[i + 1] == 0xEF, src[i + 2] == 0xBB, src[i + 3] == 0xBF
                {
                    // Duplicate BOM
                    out.append(0xEF); out.append(0xBB); out.append(0xBF)
                    out.append(0xEF); out.append(0xBB); out.append(0xBF)
                    i += 4
                } else {
                    i += 1
                }
            }
            return Data(out)
        }
    }
}
