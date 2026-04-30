//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import WhisperKit
import ArgmaxCore

// MARK: - SpeakerKit

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
open class SpeakerKit: @unchecked Sendable {
    public var diarizer: any Diarizer

    /// Creates SpeakerKit with the specified configuration.
    ///
    /// Downloads models if `config.download` is true.
    /// Loads models into memory if `config.load` is true.
    /// When both are false, models are loaded lazily on the first ``diarize(audioArray:options:progressCallback:)`` call.
    ///
    /// - Parameter config: Configuration specifying backend, download settings, and runtime options.
    public init(_ config: SpeakerKitConfig = PyannoteConfig()) async throws {
        Logging.shared.logLevel = config.verbose ? config.logLevel : .none

        // If custom diarizer provided via config, use it; otherwise create default
        if let customDiarizer = config.diarizer {
            self.diarizer = customDiarizer
        } else if let pyannoteConfig = config as? PyannoteConfig {
            self.diarizer = SpeakerKitDiarizer.pyannote(config: pyannoteConfig)
        } else {
            throw SpeakerKitError.invalidConfiguration("Config must be PyannoteConfig or provide custom diarizer")
        }
        
        if config.download {
            try await diarizer.downloadModels()
        }
        if config.load {
            try await diarizer.loadModels()
        }
    }
    
    /// Use ``init(_:)`` instead with a ``PyannoteConfig``.
    @available(*, unavailable, renamed: "init(_:)")
    public convenience init(models: some SpeakerKitModels) async throws { fatalError() }
    
    /// Deprecated: For backward compatibility with code using `SpeakerKitDiarizer` protocol.
    @available(*, unavailable, message: "Conform to Diarizer protocol instead of SpeakerKitDiarizer")
    public convenience init(diarizer: SpeakerKitDiarizer?) {
        fatalError("SpeakerKitDiarizer-based init is no longer supported. Use Diarizer protocol instead.")
    }

    // MARK: - Diarization

    /// Ensures models are downloaded and loaded before inference.
    public func ensureModelsLoaded() async throws {
        guard let modelManager = diarizer as? ModelManager else { return }
        try await modelManager.ensureModelsLoaded()
    }

    /// Unload SpeakerKit models from memory
    public func unloadModels() async {
        await diarizer.unloadModels()
    }

    /// Processes audio and returns labeled speaker segments.
    ///
    /// If models are not yet loaded, this method loads them automatically before running inference.
    /// Concurrent callers are safe.
    ///
    /// - Parameters:
    ///   - audioArray: 16 kHz mono PCM samples to diarize.
    ///   - options: Diarization options. Nil uses the defaults.
    ///   - progressCallback: Optional callback for progress updates.
    /// - Returns: Labeled speaker segments with timings.
    open func diarize(
        audioArray: [Float],
        options: (any DiarizationOptions)? = nil,
        progressCallback: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> DiarizationResult {
        try await ensureModelsLoaded()
        return try await diarizer.diarize(audioArray: audioArray, options: options, progressCallback: progressCallback)
    }

    /// Builds RTTM lines from a diarization result, optionally aligned to a transcription.
    /// - Parameters:
    ///   - diarizationResult: Result from `diarize(audioArray:options:)`.
    ///   - strategy: How to assign speaker info to words (e.g. `.subsegment`).
    ///   - transcription: Optional word-level transcription to align speakers to words.
    ///   - fileName: File ID used in the RTTM output (default `"audio"`).
    /// - Returns: RTTM lines ready to write or print.
    open class func generateRTTM(
        from diarizationResult: DiarizationResult,
        strategy: SpeakerInfoStrategy = .subsegment,
        transcription: [TranscriptionResult]? = nil,
        fileName: String = "audio"
    ) -> [RTTMLine] {
        if let transcription = transcription {
            let segments = diarizationResult.addSpeakerInfo(to: transcription, strategy: strategy)
            let wordsWithSpeakers = segments.flatMap { segmentGroup in
                segmentGroup.flatMap { segment in
                    segment.speakerWords.map { word in
                        WordWithSpeaker(wordTiming: word.wordTiming, speaker: word.speaker.speakerId)
                    }
                }
            }
            return RTTMLine.fromWords(wordsWithSpeakers, fileName: fileName)
        } else {
            var noOffsetResult = diarizationResult
            noOffsetResult.updateSegments(minActiveOffset: 0.0)
            return noOffsetResult.segments.map { segment in
                return RTTMLine(
                    fileId: fileName,
                    speakerId: segment.speaker.speakerId ?? -1,
                    startTime: segment.startTime,
                    duration: segment.endTime - segment.startTime
                )
            }
        }
    }
}

// MARK: - Error

public enum SpeakerKitError: Error, LocalizedError {
    case modelUnavailable(String)
    case invalidConfiguration(String)
    case invalidModelOutput(String)
    case generic(String)

    public var errorDescription: String? {
        switch self {
        case .modelUnavailable(let msg),
             .invalidConfiguration(let msg),
             .invalidModelOutput(let msg),
             .generic(let msg):
            return msg
        }
    }
}

/// Marker protocol for all diarization option types.
public protocol DiarizationOptions: Sendable {}

// MARK: - Model Containers

/// Marker protocol for loaded model containers.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public protocol SpeakerKitModels: Sendable {
    var modelInfos: [ModelInfo] { get }
}
