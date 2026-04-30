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
    /// Boundaries are tested in priority order: sentence enders (. ! ? \n), clause
    /// enders (, ; : - –), then word boundaries (space). A candidate is only accepted
    /// when its encoded token count reaches `minTokenCount`.
    ///
    /// - Parameters:
    ///   - minTokenCount: Minimum number of tokens the candidate must contain.
    ///   - encode: Closure that tokenizes a string and returns its token IDs.
    /// - Returns: The trimmed substring up to the last qualifying boundary, or `nil`.
    public func lastNaturalBoundary(minTokenCount: Int, encode: (String) -> [Int]) -> String? {
        let sentenceEnders: [Character] = [".", "!", "?", "\n"]
        let clauseEnders: [Character] = [",", ";", ":", "-", "–"]

        for enders in [sentenceEnders, clauseEnders] {
            if let idx = lastIndex(where: { enders.contains($0) }) {
                let candidate = String(self[...idx]).trimmingCharacters(in: .whitespacesAndNewlines)
                if encode(candidate).count >= minTokenCount {
                    return candidate
                }
            }
        }

        if let idx = lastIndex(of: " ") {
            let candidate = String(self[..<idx]).trimmingCharacters(in: .whitespacesAndNewlines)
            if encode(candidate).count >= minTokenCount {
                return candidate
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
