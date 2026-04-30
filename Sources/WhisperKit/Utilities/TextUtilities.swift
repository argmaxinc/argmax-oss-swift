//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

/// A utility struct providing text compression and analysis functionality
public struct TextUtilities {

    private init() {}

    /// Calculates the compression ratio of an array of text tokens using zlib compression
    /// - Parameter textTokens: Array of integer tokens to compress
    /// - Returns: The compression ratio (original size / compressed size). Returns infinity if compression fails
    public static func compressionRatio(of textTokens: [Int]) -> Float {
        // Convert the integer array to a byte array (Data)
        let dataBuffer = textTokens.compactMap { Int32($0) }
        let data = dataBuffer.withUnsafeBufferPointer { Data(buffer: $0) }

        // Compress the data using NSData compression
        do {
            let compressedData = try (data as NSData).compressed(using: .zlib)
            // Calculate and return the compression ratio
            return Float(data.count) / Float(compressedData.length)
        } catch {
            Logging.debug("Compression error: \(error.localizedDescription)")
            return Float.infinity
        }
    }

    /// Calculates the compression ratio of a text string using zlib compression
    /// - Parameter text: The text string to compress
    /// - Returns: The compression ratio (original size / compressed size). Returns infinity if text is empty or compression fails
    public static func compressionRatio(of text: String) -> Float {
        if text.isEmpty {
            return Float.infinity // TODO: throw to caller instead of return infinity
        }

        // Encode the string as UTF-8
        guard let data = text.data(using: .utf8) else {
            Logging.debug("String encoding error")
            return Float.infinity
        }

        // Compress the data using NSData compression
        do {
            let compressedData = try (data as NSData).compressed(using: .zlib)
            // Calculate and return the compression ratio
            return Float(data.count) / Float(compressedData.length)
        } catch {
            Logging.debug("Compression error: \(error.localizedDescription)")
            return Float.infinity
        }
    }
}

@available(*, deprecated, message: "Subject to removal in a future version. Use `TextUtilities.compressionRatio(of:)` instead.")
public func compressionRatio(of array: [Int]) -> Float {
    return TextUtilities.compressionRatio(of: array)
}

@available(*, deprecated, message: "Subject to removal in a future version. Use `TextUtilities.compressionRatio(of:)` instead.")
public func compressionRatio(of text: String) -> Float {
    return TextUtilities.compressionRatio(of: text)
}
