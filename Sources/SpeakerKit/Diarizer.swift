//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import ArgmaxCore

/// Protocol for diarization backends (Pyannote, Sortformer, etc.)
///
/// This protocol defines the interface for speaker diarization implementations.
/// Concrete implementations (e.g., `PyannoteDiarizer`, `SortformerDiarizer`) handle
/// model loading, downloading, and diarization processing.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public protocol Diarizer: Sendable {
    /// The folder containing the diarization models
    var modelFolder: URL? { get }
    
    /// Current state of the models (loaded, unloaded, etc.)
    var modelState: ModelState { get }
    
    /// Download the diarization models if needed
    func downloadModels() async throws

    /// Load the diarization models into memory
    func loadModels() async throws
    
    /// Unload the diarization models from memory
    func unloadModels() async
    
    /// Perform speaker diarization on audio
    ///
    /// - Parameters:
    ///   - audioArray: Audio samples as floating-point array (16kHz, mono)
    ///   - options: Optional diarization configuration options
    ///   - progressCallback: Optional callback for progress updates
    /// - Returns: Diarization result with speaker segments
    func diarize(
        audioArray: [Float],
        options: (any DiarizationOptions)?,
        progressCallback: (@Sendable (Progress) -> Void)?
    ) async throws -> DiarizationResult
}
