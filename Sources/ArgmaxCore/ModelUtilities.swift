//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import Foundation

public struct ModelUtilities {
    private init() {}

    // MARK: - Model Dimension Introspection

    /// Read a dimension from a model input's multiarray constraint shape.
    public static func getModelInputDimension(_ model: MLModel?, named: String, position: Int) -> Int? {
        guard let inputDescription = model?.modelDescription.inputDescriptionsByName[named] else { return nil }
        guard inputDescription.type == .multiArray else { return nil }
        guard let shapeConstraint = inputDescription.multiArrayConstraint else { return nil }
        let shape = shapeConstraint.shape.map { $0.intValue }
        return shape[position]
    }

    /// Read a dimension from a model output's multiarray constraint shape.
    public static func getModelOutputDimension(_ model: MLModel?, named: String, position: Int) -> Int? {
        guard let inputDescription = model?.modelDescription.outputDescriptionsByName[named] else { return nil }
        guard inputDescription.type == .multiArray else { return nil }
        guard let shapeConstraint = inputDescription.multiArrayConstraint else { return nil }
        let shape = shapeConstraint.shape.map { $0.intValue }
        return shape[position]
    }

    // MARK: - Model URL Detection

    /// Recursively searches a directory tree for a named CoreML model bundle.
    ///
    /// Checks the direct path first (like `detectModelURL(inFolder:named:)`), then
    /// falls back to a recursive walk. Useful when Hub downloads place models inside
    /// nested `snapshots/` subdirectories.
    public static func detectModelURL(inFolder path: URL, named modelName: String, recursive: Bool) -> URL {
        guard recursive else { return detectModelURL(inFolder: path, named: modelName) }

        let compiledName = "\(modelName).mlmodelc"
        let direct = path.appending(path: compiledName)
        if FileManager.default.fileExists(atPath: direct.path) {
            return direct
        }

        if let enumerator = FileManager.default.enumerator(at: path, includingPropertiesForKeys: nil) {
            while let url = enumerator.nextObject() as? URL {
                if url.lastPathComponent == compiledName {
                    return url
                }
            }
        }

        return direct
    }

    /// Detects the best available CoreML model URL in a folder, preferring
    /// compiled `.mlmodelc` over `.mlpackage`.
    public static func detectModelURL(inFolder path: URL, named modelName: String) -> URL {
        let compiledUrl = path.appending(path: "\(modelName).mlmodelc")
        let packageUrl = path.appending(path: "\(modelName).mlpackage/Data/com.apple.CoreML/model.mlmodel")

        let compiledModelExists = FileManager.default.fileExists(atPath: compiledUrl.path)
        let packageModelExists = FileManager.default.fileExists(atPath: packageUrl.path)

        if packageModelExists && !compiledModelExists {
            return packageUrl
        }
        return compiledUrl
    }

    /// Scans a folder for the first CoreML model bundle when the filename is not known in advance.
    ///
    /// Prefers a compiled `.mlmodelc` bundle over an `.mlpackage` source bundle.
    public static func detectModelURL(inFolder path: URL) -> URL? {
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: path, includingPropertiesForKeys: nil
        ) else { return nil }

        if let compiled = contents.first(where: { $0.pathExtension == "mlmodelc" }) {
            return compiled
        }
        if let package = contents.first(where: { $0.pathExtension == "mlpackage" }) {
            return package.appending(path: "Data/com.apple.CoreML/model.mlmodel")
        }
        return nil
    }
}
