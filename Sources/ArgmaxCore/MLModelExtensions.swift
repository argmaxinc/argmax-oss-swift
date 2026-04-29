//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

@preconcurrency import CoreML

// MARK: - Async Prediction

public extension MLModel {
    /// Async wrapper for `MLModel.prediction` that uses native async prediction
    /// on macOS 14+ / iOS 17+ and falls back to a Task-wrapped call on older OS.
    func asyncPrediction(
        from input: MLFeatureProvider,
        options: MLPredictionOptions = MLPredictionOptions()
    ) async throws -> MLFeatureProvider {
        if #available(macOS 14, iOS 17, watchOS 10, visionOS 1, *) {
            return try await prediction(from: input, options: options)
        } else {
            return try await Task {
                try prediction(from: input, options: options)
            }.value
        }
    }

    /// Async prediction with MLState for stateful models.
    /// MLState requires macOS 15+ where native async prediction is available.
    @available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func asyncPrediction(
        from input: MLFeatureProvider,
        using state: MLState,
        options: MLPredictionOptions = MLPredictionOptions()
    ) async throws -> MLFeatureProvider {
        try await prediction(from: input, using: state, options: options)
    }
}

// MARK: - Compute Units Description

public extension MLComputeUnits {
    var description: String {
        switch self {
            case .cpuOnly:
                return "cpuOnly"
            case .cpuAndGPU:
                return "cpuAndGPU"
            case .all:
                return "all"
            case .cpuAndNeuralEngine:
                return "cpuAndNeuralEngine"
            @unknown default:
                return "unknown"
        }
    }

    /// Human-readable display name suitable for UI presentation.
    var displayName: String {
        switch self {
            case .cpuOnly: return "CPU"
            case .cpuAndGPU: return "GPU"
            case .cpuAndNeuralEngine: return "Neural Engine"
            case .all: return "All"
            @unknown default: return "Unknown"
        }
    }
}
