//  Copyright © 2025 Argmax, Inc. All rights reserved.
//  For licensing see accompanying LICENSE.md file.

import Foundation
import CoreML
@preconcurrency import WhisperKit

internal class TranscribeCLIUtils {
    
    /// Creates WhisperKit configuration from CLI arguments
    static func createWhisperKitConfig(from arguments: TranscribeCLIArguments) -> WhisperKitConfig {
        var audioEncoderComputeUnits = arguments.audioEncoderComputeUnits.asMLComputeUnits
        let textDecoderComputeUnits = arguments.textDecoderComputeUnits.asMLComputeUnits

        // Use gpu for audio encoder on macOS below 14
        if audioEncoderComputeUnits == .cpuAndNeuralEngine {
            if #unavailable(macOS 14.0) {
                audioEncoderComputeUnits = .cpuAndGPU
            }
        }

        let computeOptions = ModelComputeOptions(
            audioEncoderCompute: audioEncoderComputeUnits,
            textDecoderCompute: textDecoderComputeUnits
        )

        let downloadTokenizerFolder: URL? = arguments.downloadTokenizerPath.map { URL(filePath: $0) }
        let downloadModelFolder: URL? = arguments.downloadModelPath.map { URL(filePath: $0) }
        let modelName: String? = arguments.model.map { arguments.modelPrefix + "*" + $0 }

        return WhisperKitConfig(
            model: modelName,
            downloadBase: downloadModelFolder,
            modelEndpoint: arguments.endpoint,
            modelFolder: arguments.modelPath,
            tokenizerFolder: downloadTokenizerFolder,
            computeOptions: computeOptions,
            verbose: arguments.verbose,
            logLevel: arguments.verbose ? .debug : .info,
            prewarm: false,
            load: true,
            useBackgroundDownloadSession: false
        )
    }

    /// Creates the `AudioInputOptions` used by file-based transcription.
    ///
    /// The CLI keeps full-file loading as the default for compatibility. When
    /// `--incremental-loading` is enabled, this helper wires the CLI chunk
    /// duration and buffer size into WhisperKit's incremental file loader,
    /// falling back to the library defaults (`AudioLoadingMode.defaultChunkDurationSeconds` /
    /// `defaultMaxBufferedChunks`) when omitted.
    /// Additional input options (for example channel selection) can be added here.
    ///
    /// - Parameter arguments: Parsed transcribe CLI arguments.
    /// - Returns: The `AudioInputOptions` to pass to
    ///   `WhisperKit.transcribeWithResults(audioPaths:audioInputOptions:decodeOptions:callback:)`.
    static func createAudioInputOptions(from arguments: TranscribeCLIArguments) -> AudioInputOptions {
        let audioLoadingMode: AudioInputOptions.AudioLoadingMode = arguments.incrementalLoading
            ? .incremental(
                chunkDurationSeconds: arguments.incrementalChunkDuration ?? AudioInputOptions.AudioLoadingMode.defaultChunkDurationSeconds,
                maxBufferedChunks: arguments.incrementalChunkBufferSize ?? AudioInputOptions.AudioLoadingMode.defaultMaxBufferedChunks
            )
            : .fullFile

        return AudioInputOptions(audioLoadingMode: audioLoadingMode)
    }

    /// Creates DecodingOptions from CLI arguments and task
    static func createDecodingOptions(from arguments: TranscribeCLIArguments, task: DecodingTask) -> DecodingOptions {
        let options = DecodingOptions(
            verbose: arguments.verbose,
            task: task,
            language: arguments.language,
            temperature: arguments.temperature,
            temperatureIncrementOnFallback: arguments.temperatureIncrementOnFallback,
            temperatureFallbackCount: arguments.temperatureFallbackCount,
            topK: arguments.bestOf,
            usePrefillPrompt: arguments.usePrefillPrompt || arguments.language != nil || task == .translate,
            skipSpecialTokens: arguments.skipSpecialTokens,
            withoutTimestamps: arguments.withoutTimestamps,
            wordTimestamps: arguments.wordTimestamps,
            clipTimestamps: arguments.clipTimestamps,
            suppressTokens: arguments.suppressTokens,
            compressionRatioThreshold: arguments.compressionRatioThreshold ?? 2.4,
            logProbThreshold: arguments.logprobThreshold ?? -1.0,
            firstTokenLogProbThreshold: arguments.firstTokenLogProbThreshold,
            noSpeechThreshold: arguments.noSpeechThreshold ?? 0.6,
            concurrentWorkerCount: arguments.concurrentWorkerCount,
            chunkingStrategy: ChunkingStrategy(rawValue: arguments.chunkingStrategy)
        )

        return options
    }
}
