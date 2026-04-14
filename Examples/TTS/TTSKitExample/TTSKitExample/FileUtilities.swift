//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

#if canImport(PDFKit)
import PDFKit
#endif

// MARK: - File Text Extraction

/// Utility for extracting plain text from files on disk.
///
/// Supports:
/// - Plain text files (`.txt`, and any UTF-8 readable file)
/// - PDF files (via PDFKit, where available)
enum FileUtilities {
    /// Reads and returns the text content of the file at `url`.
    ///
    /// - Returns: The extracted string, or `nil` if the file cannot be read or has no
    ///   readable text content.
    static func readTextContent(at url: URL) -> String? {
        switch url.pathExtension.lowercased() {
        case "pdf":
            return readPDF(at: url)
        default:
            // Try UTF-8 first, fall back to Latin-1 for legacy files
            if let text = try? String(contentsOf: url, encoding: .utf8) {
                return text.isEmpty ? nil : text
            }
            if let text = try? String(contentsOf: url, encoding: .isoLatin1) {
                return text.isEmpty ? nil : text
            }
            return nil
        }
    }

    // MARK: - PDF

    #if canImport(PDFKit)
    private static func readPDF(at url: URL) -> String? {
        guard let document = PDFDocument(url: url) else { return nil }
        var pages: [String] = []
        for i in 0..<document.pageCount {
            if let page = document.page(at: i), let pageText = page.string, !pageText.isEmpty {
                pages.append(pageText)
            }
        }
        let text = pages.joined(separator: "\n")
        return text.isEmpty ? nil : text
    }
    #else
    private static func readPDF(at url: URL) -> String? { nil }
    #endif
}
