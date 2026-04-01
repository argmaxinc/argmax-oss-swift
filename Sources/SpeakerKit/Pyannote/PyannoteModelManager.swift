//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import CoreML
import Hub
import ArgmaxCore

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
extension SpeakerKitDiarizer {
    /// Creates a ``SpeakerKitDiarizer`` configured for the Pyannote diarization backend.
    public static func pyannote(
        config: PyannoteConfig = PyannoteConfig(),
        segmenterModelInfo: ModelInfo = .segmenter(),
        embedderModelInfo: ModelInfo = .embedder(),
        pldaModelInfo: ModelInfo = .plda(),
        downloader: ModelDownloader? = nil
    ) -> SpeakerKitDiarizer {
        let loader = PyannoteModelLoader(
            config: config,
            segmenterModelInfo: segmenterModelInfo,
            embedderModelInfo: embedderModelInfo,
            pldaModelInfo: pldaModelInfo
        )
        let modelDownloader = downloader ?? ModelDownloader(config: config.modelDownloadConfig)
        return SpeakerKitDiarizer(
            loader: loader,
            downloader: modelDownloader
        )
    }
}

// MARK: - PyannoteModelLoader

/// Loads and manages Pyannote diarization models (segmenter, embedder, PLDA projector).
///
/// Conforms to ``ModelLoader`` so it can be used with ``ModelManager`` / ``SpeakerKitDiarizer``.
/// Use ``SpeakerKitDiarizer/pyannote(config:segmenterModelInfo:embedderModelInfo:pldaModelInfo:downloader:)``
/// to create a fully wired manager.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public final class PyannoteModelLoader: ModelLoader, @unchecked Sendable {
    public let pyannoteConfig: PyannoteConfig
    private let segmenterModelInfo: ModelInfo
    private let embedderModelInfo: ModelInfo
    private let pldaModelInfo: ModelInfo

    /// Loaded model container, available after a successful `load(from:prewarm:)` call.
    public private(set) var models: PyannoteModels?

    public var modelFolder: String? { pyannoteConfig.modelDownloadConfig.modelFolder }

    public init(
        config: PyannoteConfig = PyannoteConfig(),
        segmenterModelInfo: ModelInfo = .segmenter(),
        embedderModelInfo: ModelInfo = .embedder(),
        pldaModelInfo: ModelInfo = .plda()
    ) {
        self.pyannoteConfig = config
        self.segmenterModelInfo = segmenterModelInfo
        self.embedderModelInfo = embedderModelInfo
        self.pldaModelInfo = pldaModelInfo
    }

    public func resolveModels(downloader: ModelDownloader, progressCallback: ((Progress) -> Void)?) async throws -> String {
        let modelDownloadConfig = pyannoteConfig.modelDownloadConfig
        if let modelFolder = modelDownloadConfig.modelFolder {
            Logging.info("[SpeakerKit] Using local models from: \(modelFolder)")
            return modelFolder
        }

        let patterns = [
            segmenterModelInfo.downloadPattern,
            embedderModelInfo.downloadPattern,
            pldaModelInfo.downloadPattern,
        ]

        let downloadBase = modelDownloadConfig.downloadBase.map { URL(fileURLWithPath: $0) }
        let url = try await downloader.resolveRepo(
            patterns: patterns,
            downloadBase: downloadBase,
            download: pyannoteConfig.download,
            progressCallback: progressCallback
        )
        return url.path
    }

    public func load(from modelPath: String, prewarm: Bool) async throws {
        let baseURL = URL(fileURLWithPath: modelPath)
        let segmenterVersionDir = segmenterModelInfo.modelURL(baseURL: baseURL)
        let embedderVersionDir = embedderModelInfo.modelURL(baseURL: baseURL)
        let pldaVersionDir = pldaModelInfo.modelURL(baseURL: baseURL)

        let segmenterURL = ModelUtilities.detectModelURL(inFolder: segmenterVersionDir, named: "SpeakerSegmenter")
        let embedderPreprocessorURL = ModelUtilities.detectModelURL(inFolder: embedderVersionDir, named: "SpeakerEmbedderPreprocessor")
        let embedderURL = ModelUtilities.detectModelURL(inFolder: embedderVersionDir, named: "SpeakerEmbedder")
        let pldaURL = ModelUtilities.detectModelURL(inFolder: pldaVersionDir, named: "PldaProjector")

        let segmenterModel = try await SpeakerSegmenterModel(
            modelURL: segmenterURL,
            concurrentWorkers: pyannoteConfig.concurrentSegmenterWorkers,
            useFullRedundancy: pyannoteConfig.fullRedundancy,
            computeUnits: segmenterModelInfo.computeUnits
        )

        let embedderModel = SpeakerEmbedderModel(
            modelURL: embedderURL,
            preprocessorModelURL: embedderPreprocessorURL,
            pldaModelURL: pldaURL,
            computeUnits: embedderModelInfo.computeUnits
        )

        try await segmenterModel.loadModel(prewarmMode: prewarm)
        try await embedderModel.loadModel(prewarmMode: prewarm)

        if !prewarm {
            models = PyannoteModels(
                segmenter: segmenterModel,
                embedder: embedderModel,
                config: pyannoteConfig,
                segmenterModelInfo: segmenterModelInfo,
                embedderModelInfo: embedderModelInfo,
                pldaModelInfo: pldaModelInfo
            )
        }
    }

    public func unload() async {
        models?.segmenter.unloadModel()
        models?.embedder.unloadModel()
        models = nil
    }

    public func makeDiarizer() throws -> PyannoteDiarizer {
        guard let models else {
            throw SpeakerKitError.modelUnavailable("Pyannote models are not loaded")
        }
        let config = DiarizerConfig(
            segmenterModel: models.segmenter,
            embedderModel: models.embedder,
            clusterer: VBxClustering(),
            concurrentEmbedderWorkers: pyannoteConfig.concurrentEmbedderWorkers,
            models: models
        )
        // Create downloader with same config as this loader
        let downloader = ModelDownloader(config: pyannoteConfig.modelDownloadConfig)
        return PyannoteDiarizer(loader: self, downloader: downloader, config: config)
    }
}

// MARK: - PyannoteModels

/// Loaded Pyannote model bundle -- segmenter, embedder, PLDA projector, and associated configuration.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public struct PyannoteModels: SpeakerKitModels {
    public let segmenter: SpeakerSegmenterModel
    public let embedder: SpeakerEmbedderModel
    public let config: PyannoteConfig
    public let segmenterModelInfo: ModelInfo
    public let embedderModelInfo: ModelInfo
    public let pldaModelInfo: ModelInfo

    public var modelInfos: [ModelInfo] { [segmenterModelInfo, embedderModelInfo, pldaModelInfo] }

    init(
        segmenter: SpeakerSegmenterModel,
        embedder: SpeakerEmbedderModel,
        config: PyannoteConfig,
        segmenterModelInfo: ModelInfo = .segmenter(),
        embedderModelInfo: ModelInfo = .embedder(),
        pldaModelInfo: ModelInfo = .plda()
    ) {
        self.segmenter = segmenter
        self.embedder = embedder
        self.config = config
        self.segmenterModelInfo = segmenterModelInfo
        self.embedderModelInfo = embedderModelInfo
        self.pldaModelInfo = pldaModelInfo
    }
}
