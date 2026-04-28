//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import CoreML
import ArgmaxCore

// MARK: - Pyannote Model Info Defaults

public extension ModelInfo {
    static func segmenter(version: String? = nil, variant: String? = nil, computeUnits: MLComputeUnits = .cpuOnly) -> ModelInfo {
        let variant = variant ?? {
            #if targetEnvironment(simulator)
            if #available(iOS 18, macOS 15, *) {
                return "W8A16"
            }
            #else
            if #available(iOS 17, macOS 14, *) {
                return "W8A16"
            }
            #endif

            return "W32A32"
        }()
        return ModelInfo(version: version ?? "pyannote-v3", variant: variant, name: "speaker_segmenter", computeUnits: computeUnits)
    }

    static func embedder(version: String? = nil, variant: String? = nil, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) -> ModelInfo {
        let variant = variant ?? {
            if #available(iOS 17, macOS 14, *) {
                return "W8A16"
            } else {
                return "W16A16"
            }
        }()
        return ModelInfo(version: version ?? "pyannote-v3", variant: variant, name: "speaker_embedder", computeUnits: computeUnits)
    }

    static func plda() -> ModelInfo {
        return ModelInfo(version: "pyannote-v4", variant: "W32A32", name: "speaker_clusterer", computeUnits: .cpuOnly)
    }
}

// MARK: - Pyannote Configuration

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class PyannoteConfig: SpeakerKitConfig, @unchecked Sendable {
    public var fullRedundancy: Bool
    /// Number of concurrent segmenter inference workers. Defaults to 4.
    public var concurrentSegmenterWorkers: Int
    /// Number of concurrent embedder inference workers. `nil` uses the dynamic formula `min(8, max(2, audioSeconds / 30))`.
    public var concurrentEmbedderWorkers: Int?

    public init(
        modelDownloadConfig: ModelDownloadConfig? = nil,
        download: Bool = true,
        load: Bool = false,
        verbose: Bool = true,
        logLevel: Logging.LogLevel = .info,
        fullRedundancy: Bool = true,
        concurrentSegmenterWorkers: Int = 4,
        concurrentEmbedderWorkers: Int? = nil,
        diarizer: (any Diarizer)? = nil
    ) {
        self.fullRedundancy = fullRedundancy
        self.concurrentSegmenterWorkers = concurrentSegmenterWorkers
        self.concurrentEmbedderWorkers = concurrentEmbedderWorkers
        super.init(
            modelDownloadConfig: modelDownloadConfig,
            download: download,
            verbose: verbose,
            logLevel: logLevel,
            load: load,
            diarizer: diarizer
        )
    }

    /// Flattened initializer aligned with ``ModelDownloadConfig`` parameter order, then lifecycle and Pyannote runtime flags (same ordering as WhisperKit/TTSKit: download location, then `load`, then feature flags).
    ///
    /// `modelEndpoint` maps to the Hub endpoint (see ``ModelDownloadConfig/endpoint``). `downloadRevision` maps to ``ModelDownloadConfig/revision``.
    public init(
        downloadBase: String? = nil,
        modelRepo: String = "argmaxinc/speakerkit-coreml",
        modelToken: String? = nil,
        modelFolder: String? = nil,
        download: Bool = true,
        useBackgroundDownloadSession: Bool = false,
        modelEndpoint: String = "https://huggingface.co",
        downloadRevision: String = "main",
        load: Bool = false,
        verbose: Bool = true,
        logLevel: Logging.LogLevel = .info,
        fullRedundancy: Bool = true,
        concurrentSegmenterWorkers: Int = 4,
        concurrentEmbedderWorkers: Int? = nil,
        diarizer: (any Diarizer)? = nil
    ) {
        self.fullRedundancy = fullRedundancy
        self.concurrentSegmenterWorkers = concurrentSegmenterWorkers
        self.concurrentEmbedderWorkers = concurrentEmbedderWorkers
        super.init(
            modelDownloadConfig: ModelDownloadConfig(
                downloadBase: downloadBase,
                modelRepo: modelRepo,
                modelToken: modelToken,
                modelFolder: modelFolder,
                useBackgroundSession: useBackgroundDownloadSession,
                endpoint: modelEndpoint,
                revision: downloadRevision
            ),
            download: download,
            verbose: verbose,
            logLevel: logLevel,
            load: load,
            diarizer: diarizer
        )
    }
}

// MARK: - Diarization Options

public enum SpeakerCentroidSource: Sendable {
    /// Mean of all embeddings under the final post-reassignment speaker labels.
    case finalAssignment

    /// Mean of the trainable subset under the final post-reassignment speaker labels.
    ///
    /// This excludes embeddings whose `nonOverlappedFrameRatio` does not pass the clustering
    /// `minActiveRatio` filter, matching the subset used to seed VBx/k-means clustering.
    /// Speakers with no trainable embeddings are omitted from the returned centroid map.
    case trainableOnly
}

public struct PyannoteDiarizationOptions: DiarizationOptions {
    public var numberOfSpeakers: Int?
    public var minActiveOffset: Float?
    public var clusterDistanceThreshold: Float?
    public var minClusterSize: Int?
    public var useExclusiveReconciliation: Bool
    public var centroidSource: SpeakerCentroidSource
    /// Optional seek boundaries in seconds; pairs define [start, end] clips. Empty means process full audio.
    public var clipTimestamps: [Float]

    public init(
        numberOfSpeakers: Int? = nil,
        minActiveOffset: Float? = nil,
        clusterDistanceThreshold: Float? = nil,
        minClusterSize: Int? = nil,
        useExclusiveReconciliation: Bool = true,
        centroidSource: SpeakerCentroidSource = .finalAssignment,
        clipTimestamps: [Float] = []
    ) {
        self.numberOfSpeakers = numberOfSpeakers
        self.minActiveOffset = minActiveOffset
        self.clusterDistanceThreshold = clusterDistanceThreshold
        self.minClusterSize = minClusterSize
        self.useExclusiveReconciliation = useExclusiveReconciliation
        self.centroidSource = centroidSource
        self.clipTimestamps = clipTimestamps
    }
}

// MARK: - Performance Timings

public struct PyannoteDiarizationTimings: DiarizationTimings, CustomStringConvertible, CustomDebugStringConvertible {
    public init() {}

    public internal(set) var inputAudioSeconds: TimeInterval = 0
    public internal(set) var numberOfChunks: Int = 0
    public internal(set) var numberOfEmbeddings: Int = 0
    public internal(set) var numberOfSpeakers: Int = 0

    public internal(set) var numberOfSegmenterWorkers: Int = 0
    public internal(set) var numberOfEmbedderWorkers: Int = 0

    public internal(set) var pipelineStart: CFAbsoluteTime = 0
    public internal(set) var modelLoading: CFAbsoluteTime = 0
    public internal(set) var segmenterTime: CFAbsoluteTime = 0
    public internal(set) var embedderTime: CFAbsoluteTime = 0
    public internal(set) var clusteringTime: CFAbsoluteTime = 0
    public internal(set) var fullPipeline: CFAbsoluteTime = 0

    public var description: String {
        let pairs = [
            "input_length": "\(inputAudioSeconds)",
            "num_chunks": "\(numberOfChunks)",
            "num_embeddings": "\(numberOfEmbeddings)",
            "num_speakers": "\(numberOfSpeakers)",
            "num_segmenter_workers": "\(numberOfSegmenterWorkers)",
            "num_embedder_workers": "\(numberOfEmbedderWorkers)",
            "loading_time": String(format: "%.3f", modelLoading),
            "segmenter_time": String(format: "%.3f", segmenterTime),
            "embedding_time": String(format: "%.3f", embedderTime),
            "clustering_time": String(format: "%.3f", clusteringTime),
            "total_time": String(format: "%.3f", fullPipeline),
        ]
        let debugString = "{\(pairs.map { "\($0.key): \($0.value)" }.joined(separator: ", "))}"
        return "DiarizationTimings(\(debugString))"
    }

    public var debugDescription: String {
        let doubleChunks = Double(numberOfChunks)
        let avgSegmenterTime = doubleChunks > 0 ? segmenterTime / doubleChunks : 0
        let avgEmbedderTime = doubleChunks > 0 ? embedderTime / doubleChunks : 0
        return """
        ---- Diarization Timings ----
        Audio Length: \t\t\t \(inputAudioSeconds) seconds
        Model Load Time: \t\t \(String(format: "%.3f", modelLoading)) ms
        Total Time: \t\t\t \(String(format: "%.3f", fullPipeline)) ms
        Segmenter:
        - Number of Chunks: \t\t \(numberOfChunks)
        - Total Time: \t\t\t \(String(format: "%.3f", segmenterTime)) ms
        - Number of Workers: \t\t \(numberOfSegmenterWorkers)
        - Average Time per Chunk: \t \(String(format: "%.3f", avgSegmenterTime)) ms
        Embedder:
        - Total Time: \t\t\t \(String(format: "%.3f", embedderTime)) ms
        - Number of Workers: \t\t \(numberOfEmbedderWorkers)
        - Average Time per Chunk: \t \(String(format: "%.3f", avgEmbedderTime)) ms
        Clustering:
        - Number of Embeddings: \t \(numberOfEmbeddings)
        - Number of Speakers: \t\t \(numberOfSpeakers)
        - Clustering Time: \t\t \(String(format: "%.3f", clusteringTime)) ms
        """
    }
}
