//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation

// MARK: - Compute Options

/// Per-component CoreML compute unit configuration.
///
/// Used by `TTSKitConfig` to specify which hardware accelerators each model component
/// should use. This struct is model-agnostic; any backend with multiple CoreML components
/// that need different compute targets can adopt it.
public struct ComputeOptions: Sendable {
    /// Compute units for embedding lookup models (TextProjector, CodeEmbedder, MultiCodeEmbedder).
    /// Defaults to CPU-only since these are simple table lookups with minimal compute.
    public var embedderComputeUnits: MLComputeUnits

    /// Compute units for CodeDecoder. Defaults to CPU + Neural Engine.
    public var codeDecoderComputeUnits: MLComputeUnits

    /// Compute units for MultiCodeDecoder. Defaults to CPU + Neural Engine.
    public var multiCodeDecoderComputeUnits: MLComputeUnits

    /// Compute units for SpeechDecoder. Defaults to CPU + Neural Engine.
    public var speechDecoderComputeUnits: MLComputeUnits

    public init(
        embedderComputeUnits: MLComputeUnits = .cpuOnly,
        codeDecoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        multiCodeDecoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        speechDecoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) {
        self.embedderComputeUnits = embedderComputeUnits
        self.codeDecoderComputeUnits = codeDecoderComputeUnits
        self.multiCodeDecoderComputeUnits = multiCodeDecoderComputeUnits
        self.speechDecoderComputeUnits = speechDecoderComputeUnits
    }
}
