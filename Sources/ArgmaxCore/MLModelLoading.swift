//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation

// MARK: - CoreML model loading protocol

/// Shared lifecycle contract for any CoreML-backed model component.
///
/// Conform to this protocol to get a unified `loadModel`/`unloadModel` interface.
/// `prewarmMode` compiles the model on-device then immediately discards it to
/// serialize compilation and cap peak memory before a final concurrent load.
public protocol MLModelLoading {
    /// Load the CoreML model bundle at `url` using the specified compute units.
    ///
    /// When `prewarmMode` is `true` the model is compiled on-device and then
    /// immediately discarded. This serializes compilation to cap peak memory before
    /// the final concurrent load.
    func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool) async throws

    /// Release the loaded model weights from memory.
    func unloadModel()
}

public extension MLModelLoading {
    /// Convenience overload — loads with `prewarmMode: false`.
    func loadModel(at url: URL, computeUnits: MLComputeUnits) async throws {
        try await loadModel(at: url, computeUnits: computeUnits, prewarmMode: false)
    }
}
