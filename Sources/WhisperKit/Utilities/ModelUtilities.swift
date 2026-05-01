//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML

extension ModelUtilities {

    // MARK: - WhisperKit Model Support

    public static func modelSupport(for deviceName: String, from config: ModelSupportConfig? = nil) -> ModelSupport {
        let config = config ?? Constants.fallbackModelSupportConfig
        let modelSupport = config.modelSupport(for: deviceName)
        return modelSupport
    }

    public static func loadTokenizer(
        for pretrained: ModelVariant,
        tokenizerFolder: URL? = nil,
        additionalSearchPaths: [URL] = [],
        useBackgroundSession: Bool = false
    ) async throws -> WhisperTokenizer {
        let tokenizerName = tokenizerNameForVariant(pretrained)
        let hubApi = HubApiWrapper(downloadBase: tokenizerFolder, useBackgroundSession: useBackgroundSession)
        let hubTokenizerFolder = hubApi.localRepoLocation(HubApiWrapper.Repo(id: tokenizerName))
        
        // Determine which local folder to use
        let localTokenizerFolder: URL? = {
            func hasTokenizer(in folder: URL) -> Bool {
                let tokenizerPath = folder.appendingPathComponent("tokenizer.json")
                let exists = FileManager.default.fileExists(atPath: tokenizerPath.path)
                Logging.debug("Checking for local tokenizer at \(tokenizerPath.path) - \(exists ? "found" : "not found")")
                return exists
            }

            // Search in priority order:
            // Hub-based folder structure
            var searchPaths: [URL] = [hubTokenizerFolder]
            // Top level of tokenizerFolder (if provided)
            if let tokenizerFolder {
                searchPaths.append(tokenizerFolder)
            }
            // Additional search paths
            searchPaths.append(contentsOf: additionalSearchPaths)

            let foundPath = searchPaths.first(where: hasTokenizer)
            if let found = foundPath {
                Logging.debug("Found local tokenizer at \(found.path)")
            } else {
                Logging.debug("No tokenizer found in any local search paths")
            }
            
            return foundPath
        }()

        // If we found a local folder with a tokenizer.json, try to load from it
        if let localFolder = localTokenizerFolder {
            do {
                Logging.debug("Loading tokenizer from \(localFolder.path)")
                let wrapper = try await AutoTokenizerWrapper.from(modelFolder: localFolder, hubApi: hubApi)
                return WhisperTokenizerWrapper(tokenizer: wrapper, at: localFolder)
            } catch {
                // Error during the local loading process and fall through to load from Hub
                Logging.debug("Error loading local tokenizer: \(error)")
            }
        }

        // Fallback to downloading from the Hub if local loading is not possible or fails
        Logging.debug("Downloading tokenizer from Hub at \(hubTokenizerFolder)")
        return try await WhisperTokenizerWrapper(
            tokenizer: AutoTokenizerWrapper.from(
                pretrained: tokenizerName,
                hubApi: hubApi
            ),
            at: hubTokenizerFolder
        )
    }
    
    /// Formats and sorts model file names based on model variants
    /// 
    /// Filters the provided model files to include only recognized model variants, removes path separators,
    /// and sorts them according to the predefined model size order in `ModelVariant` enum.
    /// 
    /// - Parameter modelFiles: An array of model file paths to format and sort
    /// - Returns: An array of formatted and sorted model names, with path separators removed
    public static func formatModelFiles(_ modelFiles: [String]) -> [String] {
        let modelFilters = ModelVariant.allCases.map { "\($0.description)\($0.description.contains("large") ? "" : "/")" } // Include quantized models for large
        let modelVariants = modelFiles.map { $0.components(separatedBy: "/")[0] + "/" }
        let filteredVariants = Set(modelVariants.filter { item in
            let count = modelFilters.reduce(0) { count, filter in
                let isContained = item.contains(filter) ? 1 : 0
                return count + isContained
            }
            return count > 0
        })

        let availableModels = filteredVariants.map { variant -> String in
            variant.trimmingFromEnd(character: "/", upto: 1)
        }

        // Sorting order based on enum
        let sizeOrder = ModelVariant.allCases.map { $0.description }

        let sortedModels = availableModels.sorted { firstModel, secondModel in
            // Extract the base size without any additional qualifiers
            let firstModelBase = sizeOrder.first(where: { firstModel.contains($0) }) ?? ""
            let secondModelBase = sizeOrder.first(where: { secondModel.contains($0) }) ?? ""

            if firstModelBase == secondModelBase {
                // If base sizes are the same, sort alphabetically
                return firstModel < secondModel
            } else {
                // Sort based on the size order
                return sizeOrder.firstIndex(of: firstModelBase) ?? sizeOrder.count
                    < sizeOrder.firstIndex(of: secondModelBase) ?? sizeOrder.count
            }
        }

        return sortedModels
    }

    // MARK: Internal

    static func isModelMultilingual(logitsDim: Int?) -> Bool {
        logitsDim != 51864
    }

    static func detectVariant(logitsDim: Int, encoderDim: Int) -> ModelVariant {
        // Defaults
        var modelVariant: ModelVariant = .base

        // Determine model size
        if logitsDim == 51865 {
            // Muiltilingual
            switch encoderDim {
                case 384:
                    modelVariant = .tiny
                case 512:
                    modelVariant = .base
                case 768:
                    modelVariant = .small
                case 1024:
                    modelVariant = .medium
                case 1280:
                    modelVariant = .largev2 // same for v1
                default:
                    modelVariant = .base
            }
        } else if logitsDim == 51864 {
            // English only
            switch encoderDim {
                case 384:
                    modelVariant = .tinyEn
                case 512:
                    modelVariant = .baseEn
                case 768:
                    modelVariant = .smallEn
                case 1024:
                    modelVariant = .mediumEn
                default:
                    modelVariant = .baseEn
            }

        } else if logitsDim == 51866 {
            // Large v3 has 1 additional language token
            modelVariant = .largev3
        } else {
            Logging.error("Unrecognized vocabulary size: \(logitsDim), defaulting to base variant")
            modelVariant = .base
        }

        return modelVariant
    }

    internal static func tokenizerNameForVariant(_ variant: ModelVariant) -> String {
        var tokenizerName: String
        switch variant {
            case .tiny:
                tokenizerName = "openai/whisper-tiny"
            case .tinyEn:
                tokenizerName = "openai/whisper-tiny.en"
            case .base:
                tokenizerName = "openai/whisper-base"
            case .baseEn:
                tokenizerName = "openai/whisper-base.en"
            case .small:
                tokenizerName = "openai/whisper-small"
            case .smallEn:
                tokenizerName = "openai/whisper-small.en"
            case .medium:
                tokenizerName = "openai/whisper-medium"
            case .mediumEn:
                tokenizerName = "openai/whisper-medium.en"
            case .large:
                tokenizerName = "openai/whisper-large"
            case .largev2:
                tokenizerName = "openai/whisper-large-v2"
            case .largev3:
                tokenizerName = "openai/whisper-large-v3"
        }

        return tokenizerName
    }
}

