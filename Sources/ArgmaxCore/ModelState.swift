//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

// MARK: - ModelState

/// Lifecycle state of a loaded ML model pipeline.
///
/// Shared by both WhisperKit and TTSKit so that UI components, callbacks, and
/// utilities can reference a single canonical type.
///
/// State machine:
/// ```
/// unloaded → downloading → downloaded → loading → loaded
/// unloaded → prewarming → prewarmed
/// loaded   → unloading  → unloaded
/// ```
@frozen
public enum ModelState: CustomStringConvertible {
    case unloading
    case unloaded
    case loading
    case loaded
    case prewarming
    case prewarmed
    case downloading
    case downloaded

    public var description: String {
        switch self {
            case .unloading: return "Unloading"
            case .unloaded: return "Unloaded"
            case .loading: return "Loading"
            case .loaded: return "Loaded"
            case .prewarming: return "Specializing"
            case .prewarmed: return "Specialized"
            case .downloading: return "Downloading"
            case .downloaded: return "Downloaded"
        }
    }

    /// Returns `true` when a loading or downloading operation is in progress.
    public var isBusy: Bool {
        switch self {
            case .loading, .prewarming, .downloading, .unloading: return true
            default: return false
        }
    }
}

/// Callback invoked when the pipeline's `modelState` changes.
public typealias ModelStateCallback = @Sendable (_ oldState: ModelState?, _ newState: ModelState) -> Void
