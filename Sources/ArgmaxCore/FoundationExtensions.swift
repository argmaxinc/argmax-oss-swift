//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

// MARK: - Float

public extension Float {
    /// Rounds to the specified number of decimal places.
    func rounded(_ decimalPlaces: Int) -> Float {
        let divisor = pow(10.0, Float(decimalPlaces))
        return (self * divisor).rounded() / divisor
    }
}

// MARK: - FileManager

public extension FileManager {
    /// Resolves an input path to an absolute path, expanding tilde and resolving
    /// relative paths against the current working directory.
    static func resolveAbsolutePath(_ inputPath: String) -> String {
        let fileManager = FileManager.default

        let pathWithTildeExpanded = NSString(string: inputPath).expandingTildeInPath

        if pathWithTildeExpanded.hasPrefix("/") {
            return pathWithTildeExpanded
        }

        if let cwd = fileManager.currentDirectoryPath as String? {
            let resolvedPath = URL(fileURLWithPath: cwd).appendingPathComponent(pathWithTildeExpanded).path
            return resolvedPath
        }

        return inputPath
    }
}

// MARK: - Array

extension Array {
    /// Splits the array into batches of the given size.
    public func batched(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}

extension Array where Element: Hashable {
    /// Returns an array with duplicates removed, preserving the original order.
    public var orderedSet: [Element] {
        var seen = Set<Element>()
        return self.filter { element in
            if seen.contains(element) {
                return false
            } else {
                seen.insert(element)
                return true
            }
        }
    }
}

// MARK: - String

extension String {
    /// Returns the text up to and including the last natural boundary in the string.
    ///
    /// Boundaries are tested in priority order: sentence boundaries (from Unicode
    /// sentence segmentation), then clause enders (, ; : - and en dash), then word
    /// boundaries (space). A candidate is only accepted when its encoded token count
    /// reaches `minTokenCount`.
    ///
    /// Sentence segmentation keeps decimals, emails, and ellipses intact. The final
    /// segment is only accepted when it ends in a terminator (. ! ?) after trailing
    /// whitespace and closing quotes/brackets are stripped, so a window truncated
    /// mid-sentence is not treated as a boundary. Clause enders only count when they
    /// terminate a word, which avoids splitting mid-word hyphens.
    ///
    /// - Parameters:
    ///   - minTokenCount: Minimum number of tokens the candidate must contain.
    ///   - encode: Closure that tokenizes a string and returns its token IDs.
    /// - Returns: The untrimmed prefix up to (and including) the last qualifying
    ///   boundary, or `nil`. The prefix is returned untrimmed because `TextChunker`
    ///   advances its token stream by the prefix's encoded length; trim only for display.
    public func lastNaturalBoundary(minTokenCount: Int, encode: (String) -> [Int]) -> String? {
        let sentenceTerminators: Set<Character> = [".", "!", "?"]
        let clauseEnders: Set<Character> = [
            ",", ";", ":", "-",
            "\u{2013}", // – En dash
        ]
        let closers: Set<Character> = [
            "\"", "'",
            "\u{201D}", // ” Right double quotation mark
            "\u{2019}", // ’ Right single quotation mark
            ")", "]",
        ]

        var sentenceRanges: [Range<String.Index>] = []
        enumerateSubstrings(in: startIndex..<endIndex, options: [.bySentences]) { _, range, _, _ in
            sentenceRanges.append(range)
        }

        for index in stride(from: sentenceRanges.count - 1, through: 0, by: -1) {
            let range = sentenceRanges[index]

            // Skip a truncated final fragment left by the caller's token window: the
            // last meaningful character (ignoring trailing whitespace and closing
            // quotes/brackets) must be a terminator. A bare trailing space is not enough.
            if index == sentenceRanges.count - 1 {
                guard let lastMeaningful = self[range].last(where: { !$0.isWhitespace && !closers.contains($0) }),
                      sentenceTerminators.contains(lastMeaningful)
                else { continue }
            }

            let prefix = String(self[..<range.upperBound])
            if encode(prefix.trimmingCharacters(in: .whitespacesAndNewlines)).count >= minTokenCount {
                return prefix
            }
        }

        // Clause enders are not sentence boundaries; only accept word-terminating ones.
        let characters = Array(self)
        var characterIndex = characters.count - 1
        while characterIndex >= 0 {
            let isAtEnd = characterIndex + 1 == characters.count
            let isFollowedByWhitespace = !isAtEnd && characters[characterIndex + 1].isWhitespace
            if clauseEnders.contains(characters[characterIndex]), isAtEnd || isFollowedByWhitespace {
                let prefix = String(characters[0...characterIndex])
                if encode(prefix.trimmingCharacters(in: .whitespacesAndNewlines)).count >= minTokenCount {
                    return prefix
                }
            }
            characterIndex -= 1
        }

        if let spaceIndex = lastIndex(of: " ") {
            let prefix = String(self[..<spaceIndex])
            if encode(prefix.trimmingCharacters(in: .whitespacesAndNewlines)).count >= minTokenCount {
                return prefix
            }
        }

        return nil
    }

    /// Trims up to `upto` occurrences of `character` from the end of the string.
    public func trimmingFromEnd(character: Character = " ", upto: Int) -> String {
        var result = self
        var trimmed = 0
        while trimmed < upto && result.last == character {
            result.removeLast()
            trimmed += 1
        }
        return result
    }
}

extension [String] {
    /// Filters strings matching a glob pattern using `fnmatch`.
    public func matching(glob: String) -> [String] {
        filter { fnmatch(glob, $0, 0) == 0 }
    }
}

// MARK: - ProcessInfo (macOS)

#if os(macOS) || targetEnvironment(simulator)
public extension ProcessInfo {
    static func stringFromSysctl(named name: String) -> String {
        var size: size_t = 0
        sysctlbyname(name, nil, &size, nil, 0)
        var machineModel = [CChar](repeating: 0, count: Int(size))
        sysctlbyname(name, &machineModel, &size, nil, 0)
        return String(cString: machineModel)
    }

    static let processor = stringFromSysctl(named: "machdep.cpu.brand_string")
    static let cores = stringFromSysctl(named: "machdep.cpu.core_count")
    static let threads = stringFromSysctl(named: "machdep.cpu.thread_count")
    static let vendor = stringFromSysctl(named: "machdep.cpu.vendor")
    static let family = stringFromSysctl(named: "machdep.cpu.family")
    static let hwModel = stringFromSysctl(named: "hw.model")
}
#endif
