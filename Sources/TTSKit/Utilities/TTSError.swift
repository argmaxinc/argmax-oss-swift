//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

@frozen
public enum TTSError: Error, LocalizedError, Equatable {
    case emptyText
    case modelNotFound(String)
    case generationFailed(String)
    case tokenizerUnavailable(String)
    case audioOutputFailed(String)
    /// A required component URL could not be resolved from the config.
    /// Thrown at `loadModels()` time so callers fail early with a clear message.
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
            case .emptyText: return "Input text is empty"
            case let .modelNotFound(path): return "Model directory not found: \(path)"
            case let .generationFailed(msg): return "Generation failed: \(msg)"
            case let .tokenizerUnavailable(msg): return "Tokenizer unavailable: \(msg)"
            case let .audioOutputFailed(msg): return "Audio output failed: \(msg)"
            case let .invalidConfiguration(msg): return "Invalid TTSKit configuration: \(msg)"
        }
    }
}
