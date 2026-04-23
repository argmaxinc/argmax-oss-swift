//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import ArgmaxCore

struct SpeakerEmbedding {
    let embedding: [Float]
    let pldaEmbedding: [Float]?
    let activeFrames: [Float]
    let windowIndex: Int
    let speakerIndex: Int
    var clusterId: Int
    let nonOverlappedFrameRatio: Float

    init(embedding: [Float],
         pldaEmbedding: [Float]? = nil,
         activeFrames: [Float],
         windowIndex: Int,
         speakerIndex: Int,
         clusterId: Int = -1,
         nonOverlappedFrameRatio: Float) {
        self.embedding = embedding
        self.pldaEmbedding = pldaEmbedding
        self.activeFrames = activeFrames
        self.windowIndex = windowIndex
        self.speakerIndex = speakerIndex
        self.clusterId = clusterId
        self.nonOverlappedFrameRatio = nonOverlappedFrameRatio
    }
}

// MARK: - SpeakerEmbedderContext

struct SpeakerEmbedderContext {
    let speakerActivity: MLMultiArray
    let speakerIds: MLMultiArray
    let overlappedSpeakerActivity: MLMultiArray

    let windowsCount: Int

    let secondsPerWindow: Float
    let secondsPerChunk: Float = 30.0

    let chunkStride: Int
    let waveformLength: Float

    var speakersCount: Int { speakerActivity.shape[1].intValue }
    var framesPerWindowCount: Int { speakerIds.shape[1].intValue }
    var secondsPerStride: Float {
        guard windowsCount > 1 else { return 0 }
        return (secondsPerChunk - secondsPerWindow) / Float(windowsCount - 1)
    }
    var secondsPerFrame: Float { secondsPerWindow / Float(framesPerWindowCount) }

    var framesPerChunk: Int { Int(Float(framesPerWindowCount) * secondsPerChunk / secondsPerWindow) }

    var chunkIndices: [[Int]] {
        var chunkIndices: [[Int]] = []
        let framesPerWindow = Float(framesPerWindowCount)
        let framesPerSecond = framesPerWindow / secondsPerWindow
        let strideInFrames = secondsPerStride * framesPerSecond
        let numWindows = 1 + ((Float(framesPerChunk) - framesPerWindow) / strideInFrames)
        for windowIdx in stride(from: 0.0, to: numWindows, by: 1.0) {
            let startFrame = Int(windowIdx * strideInFrames)
            var windowIndices: [Int] = []
            for frameIdx in 0..<Int(framesPerWindow) {
                let startIdx = startFrame + frameIdx
                windowIndices.append(startIdx)
            }
            chunkIndices.append(windowIndices)
        }

        return chunkIndices
    }

    init(speakerActivity: MLMultiArray,
         speakerIds: MLMultiArray,
         overlappedSpeakerActivity: MLMultiArray,
         windowsCount: Int,
         chunkStride: Int,
         secondsPerWindow: Float,
         waveformLength: Float
    ) {
        self.speakerActivity = speakerActivity
        self.speakerIds = speakerIds
        self.overlappedSpeakerActivity = overlappedSpeakerActivity
        self.windowsCount = windowsCount
        self.chunkStride = chunkStride
        self.secondsPerWindow = secondsPerWindow
        self.waveformLength = waveformLength
    }

    func chunkOffset(for chunkIndex: Int) -> Int {
        chunkIndex * Int(chunkStride)
    }

    func activeSpeakerIndices(for windowIdx: Int) -> [Int] {
        let minActiveDuration: Float = 2.0 * secondsPerFrame
        return (0..<speakersCount).map {
            return speakerActivity[[windowIdx, $0] as [NSNumber]].floatValue * secondsPerFrame
        }.enumerated().filter { $0.element > minActiveDuration }.map { $0.offset }
    }

    func nonOverlappedFrameRatio(for windowIdx: Int, speakerIndex: Int) -> Float {
        var nonOverlappedFrameCount: Float = 0.0
        guard framesPerWindowCount > 0 else { return nonOverlappedFrameCount }

        for frameIndex in 0..<framesPerWindowCount {
            guard speakerIds[[windowIdx, frameIndex, speakerIndex] as [NSNumber]].floatValue != 0 else { continue }
            let overlapCount = overlappedSpeakerActivity[[windowIdx, frameIndex] as [NSNumber]].floatValue
            if overlapCount == 0 {
                nonOverlappedFrameCount += 1
            }
        }

        return nonOverlappedFrameCount / Float(framesPerWindowCount)
    }

    func bounded(windowIdx: Int) -> Bool {
        guard windowIdx >= 0 && windowIdx < windowsCount else {
            return false
        }
        if windowIdx == 0 {
            return true
        }
        let endBoundary = (secondsPerStride * Float(windowIdx)) + secondsPerWindow
        let bounded = endBoundary < (waveformLength + secondsPerStride)
        return bounded
    }
}

// MARK: - SpeakerEmbedderModel

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public final class SpeakerEmbedderModel: @unchecked Sendable {
    private var modelURL: URL
    private var computeUnits: MLComputeUnits
    private var model: MLModel?
    private(set) var modelState: ModelState = .unloaded

    private let preprocessorModel: SpeakerPreEmbedderModel
    private let pldaModel: PLDAEmbedderModel?

    init(modelURL: URL, preprocessorModelURL: URL, pldaModelURL: URL? = nil, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) {
        self.modelURL = modelURL
        self.computeUnits = computeUnits
        self.preprocessorModel = SpeakerPreEmbedderModel(modelURL: preprocessorModelURL)
        self.pldaModel = pldaModelURL.map { PLDAEmbedderModel(modelURL: $0) }
    }

    func unloadModel() {
        pldaModel?.unloadModel()
        preprocessorModel.unloadModel()
        model = nil
        modelState = .unloaded
    }

    func loadModel(prewarmMode: Bool = false) async throws {
        try await pldaModel?.loadModel(prewarmMode: prewarmMode)
        try await preprocessorModel.loadModel(prewarmMode: prewarmMode)

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let loadedModel = try await MLModel.load(contentsOf: modelURL, configuration: config)
        model = prewarmMode ? nil : loadedModel
        modelState = prewarmMode ? .prewarmed : .loaded
    }

    private func getModel() throws -> MLModel {
        guard let model = model else {
            throw SpeakerKitError.modelUnavailable("SpeakerEmbedderModel not loaded. Call loadModel() first.")
        }
        return model
    }

    private func getPreprocessorModel() throws -> MLModel {
        guard let model = preprocessorModel.model else {
            throw SpeakerKitError.modelUnavailable("SpeakerPreEmbedderModel not loaded. Call loadModel() first.")
        }
        return model
    }

    private func getPLDAModel() throws -> MLModel {
        guard let pldaModel = pldaModel, let model = pldaModel.model else {
            throw SpeakerKitError.modelUnavailable("PLDAEmbedderModel not loaded. Call loadModel() first.")
        }
        return model
    }

    private func validateSegmenterOutput(_ segmenterOutput: SpeakerSegmenterOutput) throws -> (MLMultiArray, MLMultiArray, MLMultiArray, MLMultiArray) {
        guard let slidingWindowWaveform = segmenterOutput.slidingWindowWaveform,
              slidingWindowWaveform.shape.count == 3,
              let speakerActivity = segmenterOutput.speakerActivity,
              speakerActivity.shape.count == 2,
              let speakerIds = segmenterOutput.speakerIDs,
              speakerIds.shape.count == 3,
              let overlappedSpeakerActivity = segmenterOutput.overlappingSpeakerActivity,
              overlappedSpeakerActivity.shape.count == 2
        else {
            throw SpeakerKitError.invalidModelOutput("Missing required segmenter model output")
        }

        return (slidingWindowWaveform, speakerActivity, speakerIds, overlappedSpeakerActivity)
    }

    private func processChunk(
        context: SpeakerEmbedderContext,
        chunkIndex: Int,
        audioChunk: MLMultiArray
    ) async throws -> [SpeakerEmbedding] {
        let windowsCount = context.windowsCount
        let framesPerChunk = context.framesPerChunk
        Logging.debug("[SpeakerEmbedder] Processing single chunk \(chunkIndex) with \(windowsCount) windows, \(framesPerChunk) frames per chunk")

        let chunkSpeakerMasks = try MLMultiArray(shape: [1, speakerDimensionForMasks(), NSNumber(value: framesPerChunk)], dataType: .float16)

        let speakersCount = context.speakersCount
        let totalSpeakers = windowsCount * speakersCount
        for i in 0..<totalSpeakers {
            for j in 0..<framesPerChunk {
                chunkSpeakerMasks[[0, i, j] as [NSNumber]] = 0 as NSNumber
            }
        }

        let windowTimelinesForChunk = context.chunkIndices
        let speakerIds = context.speakerIds
        let framesPerWindowCount = context.framesPerWindowCount
        let overlappedSpeakerActivity = context.overlappedSpeakerActivity
        for windowIdx in 0..<windowsCount {
            let activeSpeakersIndex = context.activeSpeakerIndices(for: windowIdx)

            guard !activeSpeakersIndex.isEmpty else {
                Logging.debug("[SpeakerEmbedder] No active speakers for chunk \(chunkIndex) in window \(windowIdx), continue")
                continue
            }

            let timelineIndices = windowTimelinesForChunk[windowIdx]
            for speakerIdx in 0..<speakersCount {
                let chunkSpeakerIdx = windowIdx * speakersCount + speakerIdx
                for frameIdx in 0..<framesPerWindowCount {
                    let speakerMaskValue = speakerIds[[windowIdx, frameIdx, speakerIdx] as [NSNumber]].floatValue

                    let overlappedActivity = overlappedSpeakerActivity[[windowIdx, frameIdx] as [NSNumber]].floatValue
                    let filteredValue = speakerMaskValue * (1 - overlappedActivity)

                    let windowFrameIdx = timelineIndices[frameIdx]
                    if activeSpeakersIndex.contains(speakerIdx) {
                        chunkSpeakerMasks[[0, chunkSpeakerIdx, windowFrameIdx] as [NSNumber]] = filteredValue as NSNumber
                    }
                }
            }
        }

        let waveformCount = audioChunk.shape[0].intValue
        let waveform = try reshapeMultiArray(audioChunk, to: [1, waveformCount])

        let preEmbedderInput = SpeakerEmbedderPreprocessorInput(waveforms: waveform)
        let preprocessorModel = try getPreprocessorModel()
        let preembedderFeatures = try await preprocessorModel.asyncPrediction(from: preEmbedderInput, options: MLPredictionOptions())
        let preEmbedderOutput = SpeakerEmbedderPreprocessorOutput(features: preembedderFeatures)
        guard let preprocessorOutput = preEmbedderOutput.preprocessorOutput else {
            throw SpeakerKitError.invalidModelOutput("Missing required pre embedder model output")
        }

        let embedderInput = SpeakerEmbedderInput(speakerMasks: chunkSpeakerMasks, preprocessorOutput: preprocessorOutput)
        let embedderModel = try getModel()
        let embedderFeatures = try await embedderModel.asyncPrediction(from: embedderInput, options: MLPredictionOptions())
        let embedderOutput = SpeakerEmbedderOutput(features: embedderFeatures)
        let embeddingsOutput = SpeakerEmbedderOutput.floatArrays(from: embedderOutput.speakerEmbeddings)

        var pldaEmbeddingsOutput: [[Float]] = []
        if pldaModel != nil, let speakerEmbeddings = embedderOutput.speakerEmbeddings {
            let pldaEmbedderInput = SpeakerPLDAEmbedderInput(embedderOutput: speakerEmbeddings)
            let pldaModelInstance = try getPLDAModel()
            let pldaFeatures = try await pldaModelInstance.asyncPrediction(from: pldaEmbedderInput, options: MLPredictionOptions())
            let pldaOutput = SpeakerEmbedderOutput(features: pldaFeatures)
            pldaEmbeddingsOutput = SpeakerEmbedderOutput.floatArrays(from: pldaOutput.pldaEmbeddings)
        }

        var embeddings: [SpeakerEmbedding] = []
        let chunkIndexOffset = context.chunkOffset(for: chunkIndex)
        let secondsPerStride = context.secondsPerStride
        for windowIdx in 0..<windowsCount {
            guard context.bounded(windowIdx: windowIdx) else { continue }
            let activeSpeakerIndices = context.activeSpeakerIndices(for: windowIdx)
            for speakerIdx in activeSpeakerIndices {
                let chunkSpeakerIdx = windowIdx * speakersCount + speakerIdx

                let activeFrames = activeFrames(from: speakerIds, windowIndex: windowIdx, speakerIndex: speakerIdx)
                let embedding = embeddingsOutput[chunkSpeakerIdx]

                let pldaEmbedding = chunkSpeakerIdx < pldaEmbeddingsOutput.count ? pldaEmbeddingsOutput[chunkSpeakerIdx] : nil
                let nonOverlappedFrameRatio = context.nonOverlappedFrameRatio(for: windowIdx, speakerIndex: speakerIdx)

                let frameEmbedding = SpeakerEmbedding(
                    embedding: embedding,
                    pldaEmbedding: pldaEmbedding,
                    activeFrames: activeFrames,
                    windowIndex: chunkIndexOffset + Int(round(Float(windowIdx) * secondsPerStride)),
                    speakerIndex: speakerIdx,
                    nonOverlappedFrameRatio: nonOverlappedFrameRatio
                )

                embeddings.append(frameEmbedding)
            }
        }

        Logging.debug("[SpeakerEmbedder] Single chunk \(chunkIndex) created \(embeddings.count) embeddings")
        return embeddings
    }

    func embed(segmenterOutput: SpeakerSegmenterOutput) async throws -> [SpeakerEmbedding] {
        Logging.debug("[SpeakerEmbedder] segmenterOutput: \(segmenterOutput)")

        guard model != nil, preprocessorModel.model != nil else {
            throw SpeakerKitError.modelUnavailable("Speaker embedder model is unavailable")
        }

        let startTime = CFAbsoluteTimeGetCurrent()
        defer {
            let totalTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1_000
            Logging.debug(String(format: "[SpeakerKit] Embedder model inference for chunk \(segmenterOutput.chunkIndex) time: %.2f ms", totalTime))
        }

        let (slidingWindowWaveform, speakerActivity, speakerIds, overlappedSpeakerActivity) = try validateSegmenterOutput(segmenterOutput)

        let modelWaveformSize = preprocessorModel.waveformSize()
        let windowWaveformSize = slidingWindowWaveform.shape[2].intValue
        guard modelWaveformSize > windowWaveformSize else {
            throw SpeakerKitError.invalidModelOutput(
                "SpeakerEmbedderModel requires a per-chunk preprocessor (expected waveform size \(modelWaveformSize) must exceed sliding window size \(windowWaveformSize))"
            )
        }

        let context = SpeakerEmbedderContext(speakerActivity: speakerActivity,
                                             speakerIds: speakerIds,
                                             overlappedSpeakerActivity: overlappedSpeakerActivity,
                                             windowsCount: segmenterOutput.windowsCount,
                                             chunkStride: segmenterOutput.chunkStride,
                                             secondsPerWindow: segmenterOutput.secondsPerWindow,
                                             waveformLength: segmenterOutput.waveformLength)
        let chunkIndex = segmenterOutput.chunkIndex
        Logging.debug("Processing chunk \(context.chunkOffset(for: chunkIndex))")
        let embeddings = try await processChunk(
            context: context,
            chunkIndex: chunkIndex,
            audioChunk: segmenterOutput.audioChunk
        )

        Logging.debug("[SpeakerEmbedder] Chunk \(chunkIndex) created \(embeddings.count) embeddings")
        return embeddings
    }

    private func activeFrames(from speakerIds: MLMultiArray, windowIndex: Int, speakerIndex: Int) -> [Float] {
        let shape = speakerIds.shape

        var result: [Float] = []
        for frameIndex in 0..<shape[1].intValue {
            let value = speakerIds[[windowIndex, frameIndex, speakerIndex] as [NSNumber]].floatValue
            result.append(value)
        }

        return result
    }

    /// Reshape an MLMultiArray without copying data (zero-copy pointer view).
    private func reshapeMultiArray(_ array: MLMultiArray, to dimensions: [Int]) throws -> MLMultiArray {
        let newCount = dimensions.reduce(1, *)
        guard newCount == array.count else {
            throw SpeakerKitError.invalidModelOutput("Cannot reshape \(array.shape) to \(dimensions)")
        }

        var newStrides = [Int](repeating: 0, count: dimensions.count)
        newStrides[dimensions.count - 1] = 1
        for i in Swift.stride(from: dimensions.count - 1, to: 0, by: -1) {
            newStrides[i - 1] = newStrides[i] * dimensions[i]
        }

        return try MLMultiArray(
            dataPointer: array.dataPointer,
            shape: dimensions.map { NSNumber(value: $0) },
            dataType: array.dataType,
            strides: newStrides.map { NSNumber(value: $0) }
        )
    }

    private func speakerDimensionForMasks() -> NSNumber {
        guard let shape = model?.modelDescription.inputDescriptionsByName["speaker_masks"]?.multiArrayConstraint?.shape,
              shape.count > 1 else {
            Logging.error("Failed to get speaker_masks shape for EmbedderModel")
            return 0
        }
        return shape[1]
    }
}

// MARK: - Embedder Model Input / Output

fileprivate class SpeakerEmbedderInput: MLFeatureProvider, CustomDebugStringConvertible {
    var speakerMasks: MLMultiArray
    var preprocessorOutput: MLMultiArray

    var debugDescription: String {
        let features = featureNames.compactMap { featureName -> String? in
            guard let multiArray = featureValue(for: featureName)?.multiArrayValue else { return nil }
            return "\(featureName): \(multiArray.shape)"
        }.joined(separator: ", ")
        return "[\(type(of: self)): features=\(features)]"
    }

    var featureNames: Set<String> { ["speaker_masks", "preprocessor_output_1"] }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "speaker_masks" {
            return MLFeatureValue(multiArray: self.speakerMasks)
        }
        if featureName == "preprocessor_output_1" {
            return MLFeatureValue(multiArray: self.preprocessorOutput)
        }
        return nil
    }

    init(speakerMasks: MLMultiArray, preprocessorOutput: MLMultiArray) {
        self.speakerMasks = speakerMasks
        self.preprocessorOutput = preprocessorOutput
    }
}

fileprivate class SpeakerEmbedderOutput: MLFeatureProvider, CustomDebugStringConvertible {
    private let provider: MLFeatureProvider

    var featureNames: Set<String> { provider.featureNames }

    var debugDescription: String {
        let features = featureNames.compactMap { featureName -> String? in
            guard let multiArray = provider.featureValue(for: featureName)?.multiArrayValue else { return nil }
            return "\(featureName): \(multiArray.shape)"
        }.joined(separator: ", ")
        return "[\(type(of: self)): features=\(features)]"
    }

    var speakerEmbeddings: MLMultiArray? {
        provider.featureValue(for: "speaker_embeddings")?.multiArrayValue
    }

    var pldaEmbeddings: MLMultiArray? {
        provider.featureValue(for: "plda_embeddings")?.multiArrayValue
    }

    init(features: MLFeatureProvider) {
        self.provider = features
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        provider.featureValue(for: featureName)
    }

    static func floatArrays(from embeddings: MLMultiArray?) -> [[Float]] {
        guard let embeddings else {
            Logging.error("[SpeakerEmbedder] No embeddings to convert to float arrays")
            return []
        }

        let shape = embeddings.shape
        let numSegments = shape[1].intValue
        let embeddingSize = shape[2].intValue

        var embeddingsArray: [[Float]] = []
        for j in 0..<numSegments {
            var embedding: [Float] = Array(repeating: 0.0, count: embeddingSize)
            for k in 0..<embeddingSize {
                embedding[k] = embeddings[[0, j, k] as [NSNumber]].floatValue
            }
            embeddingsArray.append(embedding)
        }

        return embeddingsArray
    }
}

// MARK: - PLDA Embedder

fileprivate class SpeakerPLDAEmbedderInput: MLFeatureProvider, CustomDebugStringConvertible {
    var embedderOutput: MLMultiArray

    var debugDescription: String {
        let features = featureNames.compactMap { featureName -> String? in
            guard let multiArray = featureValue(for: featureName)?.multiArrayValue else { return nil }
            return "\(featureName): \(multiArray.shape)"
        }.joined(separator: ", ")
        return "[\(type(of: self)): features=\(features)]"
    }

    var featureNames: Set<String> { ["embeddings"] }

    init(embedderOutput: MLMultiArray) {
        self.embedderOutput = embedderOutput
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "embeddings" {
            return MLFeatureValue(multiArray: self.embedderOutput)
        }
        return nil
    }
}

fileprivate final class PLDAEmbedderModel {
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
}
