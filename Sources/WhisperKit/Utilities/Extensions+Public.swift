//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import AVFoundation
import CoreML

public extension Array where Element == TranscriptionSegment {
    func contains(segment: TranscriptionSegment) -> Bool {
        return self.contains { $0.start == segment.start }
    }
}

public extension WhisperKit {
    static var isRunningOnSimulator: Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        return false
        #endif
    }
}

public extension String {
    var normalized: String {
        // Convert to lowercase
        let lowercaseString = self.lowercased()

        // Replace dashes with spaces
        let noDashesString = lowercaseString.replacingOccurrences(of: "-", with: " ")

        // Remove punctuation
        let noPunctuationString = noDashesString.components(separatedBy: .punctuationCharacters).joined()

        // Replace multiple spaces with a single space
        let singleSpacedString = noPunctuationString.replacingOccurrences(of: " +", with: " ", options: .regularExpression)

        // Trim whitespace and newlines
        let trimmedString = singleSpacedString.trimmingCharacters(in: .whitespacesAndNewlines)

        return trimmedString
    }

    func trimmingSpecialTokenCharacters() -> String {
        trimmingCharacters(in: Constants.specialTokenCharacters)
    }
}

@available(*, deprecated, message: "Subject to removal in a future version. Use `FileManager.resolveAbsolutePath(_:)` instead.")
public func resolveAbsolutePath(_ inputPath: String) -> String {
    return FileManager.resolveAbsolutePath(inputPath)
}


@available(*, deprecated, message: "Subject to removal in a future version. Use `ModelUtilities.formatModelFiles(_:)` instead.")
public extension WhisperKit {
    static func formatModelFiles(_ modelFiles: [String]) -> [String] {
        return ModelUtilities.formatModelFiles(modelFiles)
    }
}
