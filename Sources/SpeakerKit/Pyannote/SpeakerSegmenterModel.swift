//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

@preconcurrency import CoreML
import ArgmaxCore
import WhisperKit

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class SpeakerSegmenterModel: @unchecked Sendable {
    public private(set) var modelURL: URL
    public private(set) var computeUnits: MLComputeUnits
    public private(set) var model: MLModel?
    public private(set) var modelState: ModelState = .unloaded
    public let sampleRate: Int

    let concurrentWorkers: Int

    private let useFullRedundancy: Bool

    static let chunkLengthInSeconds: Float = 30.0

    init(
        modelURL: URL,
        sampleRate: Int = 16000,
        concurrentWorkers: Int = 1,
        useFullRedundancy: Bool = true,
        computeUnits: MLComputeUnits = .cpuOnly
    ) async throws {
        self.computeUnits = computeUnits
        self.modelURL = modelURL
        self.concurrentWorkers = concurrentWorkers
        self.useFullRedundancy = useFullRedundancy
        self.sampleRate = sampleRate

        Logging.info("[SpeakerSegmenter] initialized with \(modelURL.lastPathComponent) model")
    }

    // MARK: - Model Loading

    public func loadModel(prewarmMode: Bool = false) async throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let loadedModel = try await MLModel.load(contentsOf: modelURL, configuration: config)
        model = prewarmMode ? nil : loadedModel
        modelState = prewarmMode ? .prewarmed : .loaded
    }

    public func unloadModel() {
        model = nil
        modelState = .unloaded
    }

    // MARK: - Model Properties

    var windowsLength: Float {
        guard let model else { return 0.0 }
        if let outputDescription = model.modelDescription.outputDescriptionsByName["sliding_window_waveform"],
           outputDescription.type == .multiArray,
           let shape = outputDescription.multiArrayConstraint?.shape,
           shape.count == 3 {
            return shape[2].floatValue / Float(sampleRate)
        }
        return 0.0
    }

    var modelSampleRate: Float {
        guard let model else { return 0.0 }

        var framesPerWindow: Float {
            if let outputDescription = model.modelDescription.outputDescriptionsByName["speaker_ids"],
               outputDescription.type == .multiArray,
               let shape = outputDescription.multiArrayConstraint?.shape,
               shape.count == 3 {
                return shape[1].floatValue
            }
            return 0.0
        }

        guard windowsLength > 0.0 else { return 0.0 }
        return framesPerWindow / windowsLength
    }

    var modelChunkStrideOffset: Int {
        guard let model else { return 0 }

        var slidingWindowShape: [Float] {
            if let outputDescription = model.modelDescription.outputDescriptionsByName["sliding_window_waveform"],
               outputDescription.type == .multiArray,
               let shape = outputDescription.multiArrayConstraint?.shape,
               shape.count == 3 {
                return shape.map { $0.floatValue }
            }
            return []
        }

        var waveformShape: [Float] {
            if let inputDescription = model.modelDescription.inputDescriptionsByName["waveform"],
               inputDescription.type == .multiArray,
               let shape = inputDescription.multiArrayConstraint?.shape,
               shape.count == 1 {
                return shape.map { $0.floatValue }
            }
            return []
        }

        guard !waveformShape.isEmpty, !slidingWindowShape.isEmpty else {
            return 0
        }

        // Window stride = (chunk length - window length) / (windows count - 1)
        let windowLength = slidingWindowShape[2]
        let windowStride = (waveformShape[0] - windowLength) / (slidingWindowShape[0] - 1.0)

        // Stride offset = window length - window stride
        let chunkStrideOffset = windowLength - windowStride
        return Int(chunkStrideOffset)
    }

    // MARK: - Prediction

    public func predict(
        audioArray: [Float],
        outputContinuation: AsyncStream<SpeakerSegmenterOutput>.Continuation,
        windowPadding: Int = 0
    ) async throws {
        defer { outputContinuation.finish() }

        guard let model else {
            throw SpeakerKitError.modelUnavailable("Speaker segmenter model is unavailable")
        }

        let startTime = CFAbsoluteTimeGetCurrent()
        defer {
            let totalTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1_000
            Logging.debug(String(format: "[SpeakerKit] Total segmenter model inference time: %.2f ms", totalTime))
        }

        var chunkEndIndex = 0
        let audioArrayCount = audioArray.count
        let maxIndex = audioArrayCount - windowPadding

        let maxChunkLength = Int(Self.chunkLengthInSeconds) * sampleRate

        var chunks: [(index: Int, waveform: [Float])] = []
        var chunkIndex = 0
        let chunkStrideOffset = useFullRedundancy ? modelChunkStrideOffset : 0
        while chunkEndIndex < maxIndex {
            let chunkStartIndex = max(chunkEndIndex - chunkStrideOffset, 0)
            chunkEndIndex = min(chunkStartIndex + maxChunkLength, audioArrayCount)
            let chunk = Array(audioArray[chunkStartIndex..<chunkEndIndex])
            chunks.append((index: chunkIndex, waveform: chunk))
            chunkIndex += 1
        }
        Logging.debug("[SpeakerSegmenter] split \(audioArrayCount) into \(chunkIndex) chunks with stride offset \(chunkStrideOffset)")

        let chunkStream = AsyncStream<(index: Int, waveform: [Float])> { continuation in
            for chunk in chunks {
                continuation.yield(chunk)
            }
            continuation.finish()
        }

        let modelSampleRate = modelSampleRate
        let workerCount = max(1, concurrentWorkers)

        await withTaskGroup(of: Void.self) { taskGroup in
            let sampleRateFloat = Float(sampleRate)
            let chunkStride = Int(Float(maxChunkLength - chunkStrideOffset) / sampleRateFloat)
            for workerID in 0..<workerCount {
                taskGroup.addTask { [model] in
                    for await chunk in chunkStream {
                        guard !Task.isCancelled else { break }
                        Logging.debug("[SpeakerSegmenter][\(workerID)] inferring chunk \(chunk.index) count: \(chunk.waveform.count)")

                        var output: SpeakerSegmenterOutput
                        let waveformLength = Float(chunk.waveform.count) / sampleRateFloat
                        do {
                            guard let audioSamples = AudioProcessor.padOrTrimAudio(
                                fromArray: chunk.waveform,
                                startAt: 0,
                                toLength: maxChunkLength
                            ) else {
                                throw SpeakerKitError.generic("Segmentation Failed: Audio samples are nil")
                            }

                            let modelInputs = SpeakerSegmenterInput(waveform: audioSamples)
                            let start = CFAbsoluteTimeGetCurrent()
                            let outputFeatures = try await model.asyncPrediction(from: modelInputs, options: MLPredictionOptions())
                            output = SpeakerSegmenterOutput(
                                features: outputFeatures,
                                chunkIndex: chunk.index,
                                audioChunk: audioSamples,
                                chunkStride: chunkStride,
                                waveformLength: waveformLength,
                                modelSampleRate: modelSampleRate,
                                audioSampleRate: sampleRateFloat
                            )
                            Logging.debug("[SpeakerSegmenter][\(workerID)] inference for chunk \(chunk.index) took \(CFAbsoluteTimeGetCurrent() - start)")
                        } catch {
                            output = SpeakerSegmenterOutput(
                                features: NoOpMLFeatureProvider(),
                                chunkIndex: chunk.index,
                                audioChunk: MLMultiArray(),
                                chunkStride: chunkStride,
                                waveformLength: waveformLength,
                                modelSampleRate: modelSampleRate,
                                audioSampleRate: sampleRateFloat
                            )
                            Logging.debug("[SpeakerSegmenter][\(workerID)] inference for chunk \(chunk.index) encountered an error: \(error)")
                        }
                        outputContinuation.yield(output)
                    }
                    Logging.debug("[SpeakerSegmenter][\(workerID)] all chunks finished.")
                }
            }
        }
    }

    func maxChunks(for audioLength: Int) -> Int {
        let chunkLength = Double(Self.chunkLengthInSeconds) * Double(sampleRate)
        if useFullRedundancy {
            let offset = modelChunkStrideOffset
            let stride = chunkLength - Double(offset)
            return max(0, Int(ceil((Double(audioLength) - chunkLength) / stride))) + 1
        } else {
            return Int(ceil(Double(audioLength) / chunkLength))
        }
    }
}

// MARK: - Model Input / Output

class SpeakerSegmenterInput: MLFeatureProvider {
    var waveform: MLMultiArray

    var featureNames: Set<String> { ["waveform"] }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "waveform" {
            return MLFeatureValue(multiArray: self.waveform)
        }
        return nil
    }

    init(waveform: MLMultiArray) {
        self.waveform = waveform
    }
}

public class SpeakerSegmenterOutput: MLFeatureProvider, CustomDebugStringConvertible, @unchecked Sendable {
    private let provider: MLFeatureProvider

    public private(set) var chunkIndex: Int
    public let audioChunk: MLMultiArray
    public let chunkStride: Int
    public let waveformLength: Float
    public let modelSampleRate: Float
    public let audioSampleRate: Float

    public var featureNames: Set<String> { provider.featureNames }

    public var debugDescription: String {
        let features = featureNames.compactMap { featureName -> String? in
            guard let multiArray = provider.featureValue(for: featureName)?.multiArrayValue else {
                return nil
            }
            return "\(featureName): \(multiArray.shape)"
        }.joined(separator: ", ")
        return "[\(type(of: self)): chunkIndex=\(chunkIndex), features=\(features)]"
    }

    var speakerActivity: MLMultiArray? {
        provider.featureValue(for: "speaker_activity")?.multiArrayValue
    }

    var overlappingSpeakerActivity: MLMultiArray? {
        provider.featureValue(for: "overlapped_speaker_activity")?.multiArrayValue
    }

    var speakerIDs: MLMultiArray? {
        provider.featureValue(for: "speaker_ids")?.multiArrayValue
    }

    var slidingWindowWaveform: MLMultiArray? {
        provider.featureValue(for: "sliding_window_waveform")?.multiArrayValue
    }

    var windowsCount: Int {
        guard let slidingWindowWaveform, slidingWindowWaveform.shape.count > 0 else { return 0 }
        return slidingWindowWaveform.shape[0].intValue
    }

    var secondsPerWindow: Float {
        guard let slidingWindowWaveform, slidingWindowWaveform.shape.count > 2 else { return 0 }
        return slidingWindowWaveform.shape[2].floatValue / audioSampleRate
    }

    public init(features: MLFeatureProvider, chunkIndex: Int, audioChunk: MLMultiArray, chunkStride: Int, waveformLength: Float, modelSampleRate: Float, audioSampleRate: Float) {
        self.provider = features
        self.chunkIndex = chunkIndex
        self.audioChunk = audioChunk
        self.chunkStride = chunkStride
        self.waveformLength = waveformLength
        self.modelSampleRate = modelSampleRate
        self.audioSampleRate = audioSampleRate
    }

    public convenience init() {
        self.init(features: NoOpMLFeatureProvider(), chunkIndex: -1, audioChunk: MLMultiArray(), chunkStride: 0, waveformLength: 0, modelSampleRate: 0, audioSampleRate: 0)
    }

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        provider.featureValue(for: featureName)
    }

}

fileprivate class NoOpMLFeatureProvider: MLFeatureProvider {
    var featureNames: Set<String> { [] }

    func featureValue(for name: String) -> MLFeatureValue? {
        nil
    }
}

