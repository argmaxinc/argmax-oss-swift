//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

struct VBxClusteringConfig: Sendable {
    let threshold: Float
    let speakerRelevanceFactorA: Float
    let speakerRelevanceFactorB: Float
    let speakerResponsibilityThreshold: Float
    let minActiveRatio: Float
    let maxIterations: Int
    let initialSmoothingFactor: Float
    let numSpeakers: Int?

    private static let defaultThreshold: Float = 0.6

    init(threshold: Float = defaultThreshold,
         speakerRelevanceFactorA: Float = 0.07,
         speakerRelevanceFactorB: Float = 0.8,
         speakerResponsibilityThreshold: Float = 1e-7,
         minActiveRatio: Float = 0.2,
         maxIterations: Int = 20,
         initialSmoothingFactor: Float = 7.0,
         numSpeakers: Int? = nil) {
        self.threshold = threshold
        self.speakerRelevanceFactorA = speakerRelevanceFactorA
        self.speakerRelevanceFactorB = speakerRelevanceFactorB
        self.speakerResponsibilityThreshold = speakerResponsibilityThreshold
        self.minActiveRatio = minActiveRatio
        self.maxIterations = maxIterations
        self.initialSmoothingFactor = initialSmoothingFactor
        self.numSpeakers = numSpeakers
    }

    init(from options: PyannoteDiarizationOptions) {
        self.init(
            threshold: options.clusterDistanceThreshold ?? Self.defaultThreshold,
            numSpeakers: options.numberOfSpeakers
        )
    }
}

struct ClusteringResult {
    let clusterIndices: [Int]
    let speakerEmbeddings: [SpeakerEmbedding]

    init(clusterIndices: [Int],
         speakerEmbeddings: [SpeakerEmbedding]) {
        self.clusterIndices = clusterIndices
        self.speakerEmbeddings = speakerEmbeddings
    }
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
protocol Clusterer: Sendable {
    func add(speakerEmbeddings: [SpeakerEmbedding]) async
    func speakerEmbeddings() async -> [SpeakerEmbedding]
    func update(config: VBxClusteringConfig) async -> ClusteringResult
    func reset() async
    func clusteringConfig(from options: PyannoteDiarizationOptions) -> VBxClusteringConfig
    func isEqual(to other: any Clusterer) -> Bool
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
extension Clusterer where Self: AnyObject {
    func isEqual(to other: any Clusterer) -> Bool {
        guard let other = other as? Self else { return false }
        return other === self
    }
}
