//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import ArgmaxCore

actor VBxClustering: Clusterer {
    private var _speakerEmbeddings: [SpeakerEmbedding] = []

    func add(speakerEmbeddings: [SpeakerEmbedding]) async {
        self._speakerEmbeddings.append(contentsOf: speakerEmbeddings)
    }

    func speakerEmbeddings() async -> [SpeakerEmbedding] {
        return _speakerEmbeddings
    }

    func update(config: VBxClusteringConfig) async -> ClusteringResult {
        guard !_speakerEmbeddings.isEmpty else {
            return ClusteringResult(clusterIndices: [], speakerEmbeddings: [])
        }

        _speakerEmbeddings.sort { ($0.windowIndex, $0.speakerIndex) < ($1.windowIndex, $1.speakerIndex) }

        let (clusters, _, centroids) = cluster(embeddings: _speakerEmbeddings, config: config)

        for (clusterIndex, clusterId) in clusters.enumerated() {
            _speakerEmbeddings[clusterIndex].clusterId = clusterId
        }

        return ClusteringResult(
            clusterIndices: clusters,
            speakerEmbeddings: _speakerEmbeddings,
            speakerCentroids: centroids
        )
    }

    func reset() async {
        _speakerEmbeddings.removeAll()
    }

    nonisolated func clusteringConfig(from options: PyannoteDiarizationOptions) -> VBxClusteringConfig {
        return VBxClusteringConfig(from: options)
    }

    func cluster(
        embeddings: [SpeakerEmbedding],
        config: VBxClusteringConfig
    ) -> (clusters: [Int], linkageMatrix: [[Float]], centroids: [Int: [Float]]) {
        let trainableEmbeddings = embeddings.filter { $0.nonOverlappedFrameRatio > config.minActiveRatio }
        let embeddingsFloats = trainableEmbeddings.map { $0.embedding }
        let allEmbeddingsFloats = embeddings.map { $0.embedding }

        let pldaEmbeddingsFloats = trainableEmbeddings.map { $0.pldaEmbedding ?? [] }

        let embeddingsNormalized = embeddingsFloats.map { embedding in
            let norm = (embedding.map { $0 * $0 }.reduce(0, +)).squareRoot()
            guard norm != 0.0 else { return embedding }
            return embedding.map { $0 / norm }
        }

        let linkageMatrix = ClusterAlgorithms.fastLinkage(embeddings: embeddingsNormalized)

        var clusters = assignFlatClusters(linkage: linkageMatrix, threshold: config.threshold)

        Logging.debug("Calling vbx with \(pldaEmbeddingsFloats.count) embeddings")

        let (speakerAssignmentMatrix, speakerPriorProbabilities) = VariationalBayesHiddenMarkovModel.vbx(
            ahcClusters: clusters,
            pldaEmbeddings: pldaEmbeddingsFloats,
            speakerRelevanceFactorA: config.speakerRelevanceFactorA,
            speakerRelevanceFactorB: config.speakerRelevanceFactorB,
            maxIterations: config.maxIterations,
            initialSmoothingFactor: config.initialSmoothingFactor
        )

        let filteredIndices = speakerPriorProbabilities.enumerated().compactMap { index, probability in
            probability > config.speakerResponsibilityThreshold ? index : nil
        }

        let speakerWeights: [[Float]]
        if filteredIndices.isEmpty {
            speakerWeights = MathOps.matrixTranspose(speakerAssignmentMatrix)
        } else {
            let filteredMatrix = speakerAssignmentMatrix.map { row in
                filteredIndices.map { row[$0] }
            }
            speakerWeights = MathOps.matrixTranspose(filteredMatrix)
        }

        let clusterAssignments = speakerWeights.isEmpty ? clusters : MathOps.argmax(speakerWeights, axis: 0)
        var centroids = calculateCentroids(speakerWeights: speakerWeights, embeddings: embeddingsFloats)
        // These centroids seed cluster reassignment; returned centroids are recomputed below.

        let autoSpeakerCount = centroids.count
        Logging.debug("VBx clustering completed with \(autoSpeakerCount) speakers")

        // When the caller requests an exact speaker count and VBx gives a different result,
        // re-cluster with K-Means on the normalized embeddings and recompute centroids.
        if let requestedSpeakers = config.numSpeakers, autoSpeakerCount != requestedSpeakers {
            Logging.debug("K-Means correction: VBx gave \(autoSpeakerCount) speakers, requested \(requestedSpeakers)")
            let kAssignments = ClusterAlgorithms.kMeans(embeddings: embeddingsNormalized, clusterCount: requestedSpeakers)
            centroids = centroidsFromAssignments(
                assignments: kAssignments,
                embeddings: embeddingsFloats,
                clusterCount: requestedSpeakers
            )
        }

        if !centroids.isEmpty {
            clusters = clusterReassignment(embeddings: allEmbeddingsFloats, centroids: centroids)
            Logging.debug("Cluster reassignment completed")
        } else {
            // clusterAssignments covers only trainableEmbeddings (T ≤ N). Derive centroids
            // from those AHC assignments so clusterReassignment can cover all N embeddings to match path above.
            let numClusters = (clusterAssignments.max() ?? -1) + 1
            let fallbackCentroids = numClusters > 0
                ? centroidsFromAssignments(
                    assignments: clusterAssignments,
                    embeddings: embeddingsFloats,
                    clusterCount: numClusters
                )
                : []

            if !fallbackCentroids.isEmpty {
                clusters = clusterReassignment(embeddings: allEmbeddingsFloats, centroids: fallbackCentroids)
                Logging.debug("Cluster reassignment from AHC fallback completed")
            } else {
                clusters = Array(repeating: 0, count: embeddings.count)
                Logging.debug("No trainable embeddings; assigning all to cluster 0")
            }
        }

        // Returned centroids use final assignments and the caller-selected embedding source,
        // uniform across all paths (VBx weighted, kMeans correction, AHC fallback). Empty
        // clusters (no surviving members under .trainableOnly) are not keyed.
        let finalCentroids = centroidsFromFinalAssignments(
            assignments: clusters,
            embeddings: embeddings,
            source: config.centroidSource,
            minActiveRatio: config.minActiveRatio
        )

        return (clusters, linkageMatrix, finalCentroids)
    }

    // MARK: - Internal Methods

    private func assignFlatClusters(linkage: [[Float]], threshold: Float) -> [Int] {
        let numDataPoints = linkage.count + 1
        let totalNodes = 2 * numDataPoints - 1

        var parents = Array(0..<totalNodes)

        func findRoot(_ node: Int) -> Int {
            var current = node
            while parents[current] != current {
                current = parents[current]
            }
            return current
        }

        for (i, row) in linkage.enumerated() {
            let leftCluster = Int(row[0])
            let rightCluster = Int(row[1])
            let clusterDistance = row[2]

            if clusterDistance > threshold {
                break
            }
            let leftRoot = findRoot(leftCluster)
            let rightRoot = findRoot(rightCluster)
            let newNodeIndex = numDataPoints + i

            parents[leftRoot] = newNodeIndex
            parents[rightRoot] = newNodeIndex
            parents[newNodeIndex] = newNodeIndex
        }

        var clusterIndices = Array(repeating: 0, count: numDataPoints)
        var rootToClusterMap = [Int: Int]()
        var nextClusterId = 0

        for i in 0..<numDataPoints {
            let root = findRoot(i)
            if let existingCluster = rootToClusterMap[root] {
                clusterIndices[i] = existingCluster
            } else {
                rootToClusterMap[root] = nextClusterId
                clusterIndices[i] = nextClusterId
                nextClusterId += 1
            }
        }

        return clusterIndices
    }

    func centroidsFromAssignments(assignments: [Int], embeddings: [[Float]], clusterCount: Int) -> [[Float]] {
        guard !embeddings.isEmpty, !embeddings[0].isEmpty else { return [] }
        let dim = embeddings[0].count
        var sums = Array(repeating: Array(repeating: Float(0), count: dim), count: clusterCount)
        var counts = Array(repeating: 0, count: clusterCount)
        for (i, assignment) in assignments.enumerated() {
            guard i < embeddings.count else { continue }
            counts[assignment] += 1
            for d in 0..<dim { sums[assignment][d] += embeddings[i][d] }
        }
        return (0..<clusterCount).map { ki in
            let count = counts[ki]
            guard count > 0 else { return sums[ki] }
            return sums[ki].map { $0 / Float(count) }
        }
    }

    /// Mean-pools embeddings under final post-reassignment labels, keyed by cluster id.
    /// Empty clusters (no surviving members after the `.trainableOnly` filter) never get a key.
    func centroidsFromFinalAssignments(
        assignments: [Int],
        embeddings: [SpeakerEmbedding],
        source: SpeakerCentroidSource,
        minActiveRatio: Float
    ) -> [Int: [Float]] {
        var sums: [Int: [Float]] = [:]
        var counts: [Int: Int] = [:]
        for (index, speakerEmbedding) in embeddings.enumerated() {
            guard index < assignments.count else { continue }
            let assignment = assignments[index]
            guard assignment >= 0 else { continue }
            if source == .trainableOnly, speakerEmbedding.nonOverlappedFrameRatio <= minActiveRatio {
                continue
            }
            let embedding = speakerEmbedding.embedding
            guard !embedding.isEmpty else { continue }
            if var existing = sums[assignment] {
                guard existing.count == embedding.count else { continue }
                for d in 0..<embedding.count {
                    existing[d] += embedding[d]
                }
                sums[assignment] = existing
                counts[assignment, default: 0] += 1
            } else {
                sums[assignment] = embedding
                counts[assignment] = 1
            }
        }
        var result: [Int: [Float]] = [:]
        for (clusterId, sum) in sums {
            guard let count = counts[clusterId], count > 0 else { continue }
            result[clusterId] = sum.map { $0 / Float(count) }
        }
        return result
    }

    func calculateCentroids(speakerWeights: [[Float]], embeddings: [[Float]]) -> [[Float]] {
        guard !speakerWeights.isEmpty, !embeddings.isEmpty, !embeddings[0].isEmpty else {
            return []
        }

        let numSpeakers = speakerWeights.count
        let embeddingDim = embeddings[0].count

        var centroids = Array(repeating: Array(repeating: Float(0), count: embeddingDim), count: numSpeakers)

        for speakerIdx in 0..<numSpeakers {
            let weights = speakerWeights[speakerIdx]
            let weightSum = weights.reduce(0, +)

            guard weightSum > 0 else {
                continue
            }

            var weightedSum = Array(repeating: Float(0), count: embeddingDim)

            for (embeddingIdx, weight) in weights.enumerated() {
                guard embeddingIdx < embeddings.count else { continue }
                let embedding = embeddings[embeddingIdx]

                for dim in 0..<embeddingDim {
                    weightedSum[dim] += weight * embedding[dim]
                }
            }

            for dim in 0..<embeddingDim {
                centroids[speakerIdx][dim] = weightedSum[dim] / weightSum
            }
        }

        return centroids
    }

    private func clusterReassignment(embeddings: [[Float]], centroids: [[Float]]) -> [Int] {
        guard !embeddings.isEmpty, !centroids.isEmpty else {
            return []
        }

        let e2kDistance = MathOps.cosineDistanceMatrix(embeddings: embeddings, centroids: centroids)

        let softClusters = e2kDistance.map { row in
            row.map { 2.0 - $0 }
        }

        let clusterAssignments = MathOps.argmax(softClusters, axis: 1)

        return clusterAssignments
    }
}
