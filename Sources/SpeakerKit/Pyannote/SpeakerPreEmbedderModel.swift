//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import ArgmaxCore

final class SpeakerPreEmbedderModel {
    private var modelURL: URL
    private var computeUnits: MLComputeUnits
    var model: MLModel?

    init(modelURL: URL) {
        self.modelURL = modelURL
        self.computeUnits = .cpuOnly
    }

    func loadModel(prewarmMode: Bool = false) async throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let loadedModel = try await MLModel.load(contentsOf: modelURL, configuration: config)
        model = prewarmMode ? nil : loadedModel
    }

    func unloadModel() {
        model = nil
    }

    func waveformSize() -> Int {
        guard let shape = model?.modelDescription.inputDescriptionsByName["waveforms"]?.multiArrayConstraint?.shape,
              shape.count > 1 else {
            Logging.error("Failed to get waveform shape for PreEmbedderModel")
            return 0
        }
        return shape[1].intValue
    }
}

// MARK: - Preprocessor Model Input / Output

final class SpeakerEmbedderPreprocessorInput: MLFeatureProvider {
    var waveforms: MLMultiArray

    var featureNames: Set<String> { ["waveforms"] }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "waveforms" {
            return MLFeatureValue(multiArray: self.waveforms)
        }
        return nil
    }

    init(waveforms: MLMultiArray) {
        self.waveforms = waveforms
    }
}

final class SpeakerEmbedderPreprocessorOutput: MLFeatureProvider, CustomDebugStringConvertible {
    private let provider: MLFeatureProvider

    var featureNames: Set<String> { provider.featureNames }

    var debugDescription: String {
        let features = featureNames.compactMap { featureName -> String? in
            guard let multiArray = provider.featureValue(for: featureName)?.multiArrayValue else { return nil }
            return "\(featureName): \(multiArray.shape)"
        }.joined(separator: ", ")
        return "[\(type(of: self)): features=\(features)]"
    }

    var preprocessorOutput: MLMultiArray? {
        provider.featureValue(for: "preprocessor_output_1")?.multiArrayValue
    }

    init(features: MLFeatureProvider) {
        self.provider = features
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        provider.featureValue(for: featureName)
    }
}
