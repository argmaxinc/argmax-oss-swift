//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Accelerate
import Foundation
import ArgmaxCore

// MARK: - ClusterAlgorithms

struct ClusterAlgorithms {
    /// Hierarchical clustering using UPGMC (Unweighted Pair Group Method with Centroid)
    ///
    /// - Parameters:
    ///   - embeddings: A 2D array of embeddings to cluster
    /// - Returns: A 2D array of linkage matrix where each row is 4 elements long and contains the following information:
    ///   - The two cluster IDs that are merged (first and second position)
    ///   - The distance between the two clusters (third position)
    ///   - The size of the new cluster (fourth position)
    ///
    /// - Note: This implementation is based on `fast_linkage` implementation from scipy
    /// https://github.com/scipy/scipy/blob/main/scipy/cluster/_hierarchy.pyx#L792
    static func fastLinkage(embeddings: [[Float]]) -> [[Float]] {
        let numEmbeddings = embeddings.count
        guard numEmbeddings > 1 else { return [] }

        var condensedDistance = CondensedDistanceMatrix(embeddings)
        var context = LinkageContext(numEmbeddings)
        var linkageMatrix = Array(
            repeating: Array(repeating: Float(0), count: 4), count: numEmbeddings - 1)

        for x in 0..<(numEmbeddings - 1) {
            let pair = condensedDistance.findMinDist(for: x, clusterSizes: context.sizes)
            context.updateNeighbor(clusterIndex: x, neighborIndex: pair.key, distance: pair.value)
        }

        var minDistHeap = MinHeap(context.minDistances)

        for k in 0..<(numEmbeddings - 1) {
            var leftClusterIndex = 0
            var rightClusterIndex = 0
            var dist = Float(0)

            for _ in 0..<(numEmbeddings - k) {
                let pair = minDistHeap.getMin()
                leftClusterIndex = pair.key
                dist = pair.value
                rightClusterIndex = context.getNeighbor(from: leftClusterIndex)

                if dist == condensedDistance[leftClusterIndex, rightClusterIndex] {
                    break
                }

                let newPair = condensedDistance.findMinDist(
                    for: leftClusterIndex, clusterSizes: context.sizes)
                rightClusterIndex = newPair.key
                dist = newPair.value
                context.updateNeighbor(
                    clusterIndex: leftClusterIndex, neighborIndex: rightClusterIndex, distance: dist)
                minDistHeap.changeValue(index: leftClusterIndex, value: dist)
            }
            minDistHeap.removeMin()

            let mergedResult = context.mergeClusters(
                leftClusterIndex: leftClusterIndex,
                rightClusterIndex: rightClusterIndex,
                iteration: k
            )

            linkageMatrix[k][0] = Float(mergedResult.leftClusterId)
            linkageMatrix[k][1] = Float(mergedResult.rightClusterId)
            linkageMatrix[k][2] = dist
            linkageMatrix[k][3] = Float(mergedResult.leftClusterSize + mergedResult.rightClusterSize)

            for updateClusterIndex in 0..<numEmbeddings {
                let updateClusterSize = context.getSize(from: updateClusterIndex)
                if updateClusterSize == 0 || updateClusterIndex == rightClusterIndex {
                    continue
                }

                condensedDistance.updateDistance(
                    leftClusterIndex: leftClusterIndex,
                    rightClusterIndex: rightClusterIndex,
                    updateClusterIndex: updateClusterIndex,
                    leftClusterSize: mergedResult.leftClusterSize,
                    rightClusterSize: mergedResult.rightClusterSize
                )

                if updateClusterIndex < leftClusterIndex && updateClusterSize > 0
                    && context.getNeighbor(from: updateClusterIndex) == leftClusterIndex
                {
                    context.updateNeighbor(clusterIndex: updateClusterIndex, neighborIndex: rightClusterIndex)
                }

                if updateClusterIndex < rightClusterIndex && updateClusterSize > 0 {
                    let dist = condensedDistance[updateClusterIndex, rightClusterIndex]
                    if dist < context.getMinDistance(from: updateClusterIndex) {
                        context.updateNeighbor(
                            clusterIndex: updateClusterIndex, neighborIndex: rightClusterIndex, distance: dist)
                        minDistHeap.changeValue(index: updateClusterIndex, value: dist)
                    }
                }
            }

            if rightClusterIndex < numEmbeddings - 1 {
                let pair = condensedDistance.findMinDist(
                    for: rightClusterIndex, clusterSizes: context.sizes)
                let updateClusterIndex = pair.key
                let newDist = pair.value
                if updateClusterIndex != -1 {
                    context.updateNeighbor(
                        clusterIndex: rightClusterIndex, neighborIndex: updateClusterIndex, distance: newDist)
                    minDistHeap.changeValue(index: rightClusterIndex, value: newDist)
                }
            }
        }

        return linkageMatrix
    }

    /// Deterministic Lloyd-style k-means with k-means++ initialization.
    ///
    /// Runs `numberOfRestarts` independent initializations, each seeded from `randomSeed`,
    /// and returns the assignment with the lowest inertia (sum of squared distances to centroids).
    /// Default parameters are chosen to be similar to common diarization pipeline configurations,
    /// but this is not a bit-for-bit reimplementation of any specific reference.
    ///
    /// - Parameters:
    ///   - embeddings: L2-normalized embedding vectors to cluster
    ///   - clusterCount: Target number of clusters (clamped to [1, embeddings.count])
    ///   - maxIterations: Maximum Lloyd iterations per restart
    ///   - numberOfRestarts: Number of independent initializations; best inertia is kept
    ///   - randomSeed: Base seed; each restart uses `randomSeed + restart` for an independent stream
    /// - Returns: Cluster index for each input embedding
    static func kMeans(
        embeddings: [[Float]],
        clusterCount: Int,
        maxIterations: Int = 300,
        numberOfRestarts: Int = 3,
        randomSeed: UInt64 = 42
    ) -> [Int] {
        let embeddingCount = embeddings.count
        guard embeddingCount > 0, !embeddings[0].isEmpty else { return [] }

        let targetClusterCount = max(1, min(clusterCount, embeddingCount))
        if targetClusterCount == 1 { return Array(repeating: 0, count: embeddingCount) }
        if targetClusterCount >= embeddingCount { return Array(0..<embeddingCount) }

        let dimension = embeddings[0].count

        func squaredEuclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
            var result = Float(0)
            vDSP_distancesq(a, 1, b, 1, &result, vDSP_Length(dimension))
            return result
        }

        // Assign each embedding to nearest centroid; returns (assignments, total inertia)
        func assign(centroids: [[Float]]) -> (assignments: [Int], inertia: Float) {
            var assignments = Array(repeating: 0, count: embeddingCount)
            var totalInertia = Float(0)
            for embeddingIndex in 0..<embeddingCount {
                var minimumDistance = Float.infinity
                var nearestCentroid = 0
                for centroidIndex in 0..<centroids.count {
                    let distance = squaredEuclideanDistance(embeddings[embeddingIndex], centroids[centroidIndex])
                    if distance < minimumDistance {
                        minimumDistance = distance
                        nearestCentroid = centroidIndex
                    }
                }
                assignments[embeddingIndex] = nearestCentroid
                totalInertia += minimumDistance
            }
            return (assignments, totalInertia)
        }

        // Recompute centroids as the mean of embeddings in each cluster.
        // Empty clusters retain their previous centroid to avoid collapsing to zero.
        func computeCentroids(assignments: [Int], previousCentroids: [[Float]]) -> [[Float]] {
            var sums = Array(repeating: Array(repeating: Float(0), count: dimension), count: targetClusterCount)
            var counts = Array(repeating: 0, count: targetClusterCount)
            for embeddingIndex in 0..<embeddingCount {
                let clusterIndex = assignments[embeddingIndex]
                counts[clusterIndex] += 1
                sums[clusterIndex].withUnsafeMutableBufferPointer { sumPtr in
                    embeddings[embeddingIndex].withUnsafeBufferPointer { embedPtr in
                        guard let sumBase = sumPtr.baseAddress, let embedBase = embedPtr.baseAddress else { return }
                        vDSP_vadd(sumBase, 1, embedBase, 1, sumBase, 1, vDSP_Length(dimension))
                    }
                }
            }
            return (0..<targetClusterCount).map { clusterIndex in
                let count = counts[clusterIndex]
                guard count > 0 else { return previousCentroids[clusterIndex] }
                var scale = Float(1) / Float(count)
                var centroid = Array(repeating: Float(0), count: dimension)
                vDSP_vsmul(sums[clusterIndex], 1, &scale, &centroid, 1, vDSP_Length(dimension))
                return centroid
            }
        }

        var bestClusterAssignments = Array(repeating: 0, count: embeddingCount)
        var lowestInertia = Float.infinity

        for restart in 0..<numberOfRestarts {
            // Each restart uses an independent stream derived from the base seed
            var rng = SplitMix64(seed: randomSeed &+ UInt64(restart))

            // k-means++ initialization: seed centroids with probability proportional to D²
            var centroids = [[Float]]()
            let firstIndex = Int.random(in: 0..<embeddingCount, using: &rng)
            centroids.append(embeddings[firstIndex])

            while centroids.count < targetClusterCount {
                // D²(x): squared distance from each embedding to its nearest existing centroid
                var squaredDistances = Array(repeating: Float(0), count: embeddingCount)
                for embeddingIndex in 0..<embeddingCount {
                    var minimumDistance = Float.infinity
                    for centroid in centroids {
                        let distance = squaredEuclideanDistance(embeddings[embeddingIndex], centroid)
                        if distance < minimumDistance { minimumDistance = distance }
                    }
                    squaredDistances[embeddingIndex] = minimumDistance
                }

                var totalSquaredDistance = Float(0)
                vDSP_sve(squaredDistances, 1, &totalSquaredDistance, vDSP_Length(embeddingCount))

                guard totalSquaredDistance > 0 else {
                    centroids.append(embeddings[centroids.count % embeddingCount])
                    continue
                }

                // Sample next centroid proportional to D²
                let threshold = Float.random(in: 0..<totalSquaredDistance, using: &rng)
                var cumulative = Float(0)
                var chosenIndex = embeddingCount - 1
                for embeddingIndex in 0..<embeddingCount {
                    cumulative += squaredDistances[embeddingIndex]
                    if cumulative >= threshold {
                        chosenIndex = embeddingIndex
                        break
                    }
                }
                centroids.append(embeddings[chosenIndex])
            }

            // Lloyd iterations
            var clusterAssignments = Array(repeating: 0, count: embeddingCount)
            for _ in 0..<maxIterations {
                let (newAssignments, _) = assign(centroids: centroids)
                if newAssignments == clusterAssignments { break }
                clusterAssignments = newAssignments
                centroids = computeCentroids(assignments: clusterAssignments, previousCentroids: centroids)
            }

            let (finalAssignments, inertia) = assign(centroids: centroids)
            if inertia < lowestInertia {
                lowestInertia = inertia
                bestClusterAssignments = finalAssignments
            }
        }

        return bestClusterAssignments
    }
}

// MARK: - SplitMix64

/// Seeded 64-bit PRNG by Sebastiano Vigna, conforming to `RandomNumberGenerator`.
///
/// Uses a Weyl-sequence state update followed by Stafford's "Mix13" finalizer.
/// Constants:
///  - `goldenGammaIncrement` (`0x9E3779B97F4A7C15`): floor(2^64 / phi) where phi is the golden
///    ratio. Produces a maximally-spread additive Weyl sequence over the full 64-bit range.
///  - `staffordMix13Multiplier1/2`: mixing constants from Stafford's variant "Mix13",
///    chosen by brute-force search to maximize avalanche quality (each flipped input bit
///    changes ~50% of output bits). Shift widths 30, 27, 31 are part of the same search result.
///
/// Reference: https://prng.di.unimi.it/splitmix64.c
private struct SplitMix64: RandomNumberGenerator {
    private static let goldenGammaIncrement: UInt64 = 0x9E3779B97F4A7C15
    private static let staffordMix13Multiplier1: UInt64 = 0xBF58476D1CE4E5B9
    private static let staffordMix13Multiplier2: UInt64 = 0x94D049BB133111EB

    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= Self.goldenGammaIncrement
        var mixed = state
        mixed = (mixed ^ (mixed >> 30)) &* Self.staffordMix13Multiplier1
        mixed = (mixed ^ (mixed >> 27)) &* Self.staffordMix13Multiplier2
        return mixed ^ (mixed >> 31)
    }
}

// MARK: - LinkageContext

private struct LinkageContext {
    let numClusters: Int

    var sizes: [Int]
    var ids: [Int]
    var neighbors: [Int]
    var minDistances: [Float]

    init(_ numEmbeddings: Int) {
        numClusters = numEmbeddings
        sizes = Array(repeating: 1, count: numClusters)
        ids = Array(0..<numClusters)
        neighbors = Array(repeating: 0, count: numClusters - 1)
        minDistances = Array(repeating: Float.infinity, count: numClusters - 1)
    }

    func getNeighbor(from clusterIndex: Int) -> Int {
        return neighbors[clusterIndex]
    }

    func getSize(from clusterIndex: Int) -> Int {
        return sizes[clusterIndex]
    }

    func getMinDistance(from clusterIndex: Int) -> Float {
        return minDistances[clusterIndex]
    }

    mutating func updateNeighbor(clusterIndex: Int, neighborIndex: Int, distance: Float? = nil) {
        neighbors[clusterIndex] = neighborIndex
        if let distance = distance {
            minDistances[clusterIndex] = distance
        }
    }

    mutating func mergeClusters(
        leftClusterIndex: Int,
        rightClusterIndex: Int,
        iteration: Int
    ) -> (leftClusterId: Int, rightClusterId: Int, leftClusterSize: Int, rightClusterSize: Int) {
        var leftClusterId = ids[leftClusterIndex]
        var rightClusterId = ids[rightClusterIndex]
        let leftClusterSize = sizes[leftClusterIndex]
        let rightClusterSize = sizes[rightClusterIndex]

        sizes[leftClusterIndex] = 0
        sizes[rightClusterIndex] = leftClusterSize + rightClusterSize
        ids[rightClusterIndex] = numClusters + iteration

        if leftClusterId > rightClusterId {
            swap(&leftClusterId, &rightClusterId)
        }

        return (leftClusterId, rightClusterId, leftClusterSize, rightClusterSize)
    }
}

// MARK: - CondensedDistanceMatrix

/// Container for the condensed distance matrix.
/// Allows for easy distance access with normal cluster indices (not to be confused with cluster ids)
/// and updates using Lance-Williams Formula.
private struct CondensedDistanceMatrix {
    var distances: [Float]
    let numEmbeddings: Int

    init(_ embeddings: [[Float]]) {
        assert(!embeddings.isEmpty, "Cannot create distance matrix with empty embeddings")
        assert(
            embeddings.allSatisfy { $0.count == embeddings[0].count },
            "All embeddings must have the same dimension")

        numEmbeddings = embeddings.count
        distances = Array(repeating: Float(0), count: (numEmbeddings * (numEmbeddings - 1)) / 2)

        for i in 0..<(numEmbeddings - 1) {
            for j in (i + 1)..<numEmbeddings {
                distances[condensedIndex(i: i, j: j)] = euclideanDistance(embeddings[i], embeddings[j])
            }
        }
    }

    subscript(_ i: Int, _ j: Int) -> Float {
        get {
            assert(
                i >= 0 && i < numEmbeddings && j >= 0 && j < numEmbeddings, "i and j must be valid indices")
            return distances[condensedIndex(i: i, j: j)]
        }
        set {
            assert(
                i >= 0 && i < numEmbeddings && j >= 0 && j < numEmbeddings, "i and j must be valid indices")
            distances[condensedIndex(i: i, j: j)] = newValue
        }
    }

    func condensedIndex(i: Int, j: Int) -> Int {
        if i < j {
            return (numEmbeddings * i) - (i * (i + 1) / 2) + (j - i - 1)
        } else {
            return (numEmbeddings * j) - (j * (j + 1) / 2) + (i - j - 1)
        }
    }

    func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var dist = Float()
        vDSP_distancesq(a, 1, b, 1, &dist, vDSP_Length(a.count))
        return sqrt(dist)
    }

    func findMinDist(for idx: Int, clusterSizes: [Int]) -> (key: Int, value: Float) {
        var minDist = Float.infinity
        var minY = -1

        for y in (idx + 1)..<numEmbeddings {
            if clusterSizes[y] == 0 { continue }

            let dist = self[idx, y]
            if dist < minDist {
                minDist = dist
                minY = y
            }
        }
        return (minY, minDist)
    }

    mutating func updateDistance(
        leftClusterIndex: Int,
        rightClusterIndex: Int,
        updateClusterIndex: Int,
        leftClusterSize: Int,
        rightClusterSize: Int
    ) {
        let leftSizeFloat = Float(leftClusterSize)
        let rightSizeFloat = Float(rightClusterSize)

        let distLeftRight = self[leftClusterIndex, rightClusterIndex]
        let distLeftUpdate = self[leftClusterIndex, updateClusterIndex]
        let distRightUpdate = self[rightClusterIndex, updateClusterIndex]

        let firstTerm =
            (leftSizeFloat * distLeftUpdate * distLeftUpdate)
            + (rightSizeFloat * distRightUpdate * distRightUpdate)
        let secondTerm =
            distLeftRight * distLeftRight * (leftSizeFloat * rightSizeFloat)
            / (leftSizeFloat + rightSizeFloat)

        self[rightClusterIndex, updateClusterIndex] = sqrt(
            (firstTerm - secondTerm) / (leftSizeFloat + rightSizeFloat))
    }
}

// MARK: - MinHeap

/// Min-heap for efficient minimum distance lookups and updates
private struct MinHeap {
    var heap: [(key: Int, value: Float)]
    var indices: [Int]

    init(_ values: [Float]) {
        heap = values.enumerated().map { ($0, $1) }
        indices = Array(0..<values.count)

        // Build heap bottom-up ignoring leaf nodes
        for i in stride(from: (values.count / 2) - 1, through: 0, by: -1) {
            heapify(i)
        }
    }

    func getMin() -> (key: Int, value: Float) {
        return heap[0]
    }

    mutating func removeMin() {
        let last = heap.removeLast()
        if !heap.isEmpty {
            heap[0] = last
            indices[last.key] = 0
            heapify(0)
        }
    }

    mutating func changeValue(index: Int, value: Float) {
        let i = indices[index]
        let oldValue = heap[i].value
        heap[i].value = value

        if value < oldValue {
            var current = i
            while current > 0 {
                let parent = (current - 1) / 2
                if heap[current].value < heap[parent].value {
                    heap.swapAt(current, parent)
                    indices[heap[current].key] = current
                    indices[heap[parent].key] = parent
                    current = parent
                } else {
                    break
                }
            }
        } else {
            heapify(i)
        }
    }

    mutating func heapify(_ i: Int) {
        let left = 2 * i + 1
        let right = 2 * i + 2
        var smallest = i

        if left < heap.count && heap[left].value < heap[smallest].value {
            smallest = left
        }

        if right < heap.count && heap[right].value < heap[smallest].value {
            smallest = right
        }

        if smallest != i {
            heap.swapAt(i, smallest)
            indices[heap[i].key] = i
            indices[heap[smallest].key] = smallest
            heapify(smallest)
        }
    }
}

// MARK: - VariationalBayesHiddenMarkovModel

internal struct VariationalBayesHiddenMarkovModel {

    private static let betweenClassCovariance: [Float] = [
        25.8823843, 10.64654768, 7.09749664, 5.70842102, 5.27071843,
        4.99630206, 4.25741596, 4.07776313, 3.89517645, 3.69594798,
        3.64910204, 3.4740059, 3.1161406, 2.89308777, 2.85235283,
        2.74298281, 2.69856644, 2.54895349, 2.49312298, 2.35923547,
        2.31617442, 2.25039797, 2.20650582, 2.11553732, 2.08046971,
        2.04438817, 1.99983924, 1.94495688, 1.90123046, 1.86979365,
        1.84888933, 1.81611504, 1.76659227, 1.73939854, 1.71681168,
        1.68313843, 1.63579985, 1.6291736, 1.58139228, 1.53777309,
        1.52376318, 1.50576921, 1.4852546, 1.46273286, 1.46112849,
        1.43902254, 1.41162633, 1.40358761, 1.38767215, 1.35415771,
        1.34320055, 1.31804126, 1.29211534, 1.26927315, 1.25277974,
        1.23694313, 1.21484673, 1.21013266, 1.20138393, 1.19199542,
        1.17204403, 1.14954023, 1.14245929, 1.122949, 1.11425141,
        1.09640355, 1.08456146, 1.0667317, 1.05513591, 1.04003146,
        1.02566902, 1.02010552, 1.01099642, 0.99231797, 0.98069675,
        0.97343907, 0.95881054, 0.95197792, 0.9462381, 0.92696959,
        0.91914417, 0.9136186, 0.90647712, 0.90414186, 0.8860543,
        0.88015839, 0.87319719, 0.86870833, 0.86731253, 0.85900931,
        0.84836197, 0.83159452, 0.82433101, 0.81734176, 0.80188412,
        0.79747487, 0.79064521, 0.78698437, 0.78016046, 0.76995838,
        0.76739477, 0.76181261, 0.7557517, 0.74880944, 0.73518941,
        0.73211398, 0.7256853, 0.72203483, 0.70633259, 0.70241969,
        0.69792648, 0.68882402, 0.67445369, 0.67196181, 0.66614225,
        0.65970189, 0.65231306, 0.6459088, 0.64389891, 0.63339111,
        0.62995437, 0.62304199, 0.61221797, 0.61031214, 0.60488038,
        0.6014566, 0.58401099, 0.56960536,
    ]

    // MARK: - Main VB-HMM Algorithm

    /// Variational Bayes Hidden Markov Model algorithm for speaker diarization
    static func vbx(
        ahcClusters: [Int],
        pldaEmbeddings: [[Float]],
        speakerRelevanceFactorA: Float = 0.07,
        speakerRelevanceFactorB: Float = 0.8,
        maxIterations: Int = 20,
        initialSmoothingFactor: Float = 7.0,
        verbose: Bool = false
    ) -> (speakerAssignmentMatrix: [[Float]], speakerPriorProbabilities: [Float]) {
        #if DEBUG
        let vbxStart = CFAbsoluteTimeGetCurrent()
        #endif

        guard !pldaEmbeddings.isEmpty && pldaEmbeddings[0].count == betweenClassCovariance.count else {
            return (speakerAssignmentMatrix: [], speakerPriorProbabilities: [])
        }

        guard ahcClusters.count == pldaEmbeddings.count else {
            return (speakerAssignmentMatrix: [], speakerPriorProbabilities: [])
        }

        let betweenClassCovarianceDiag = betweenClassCovariance

        let numClusters = (ahcClusters.max() ?? 0) + 1
        let numEmbeddings = pldaEmbeddings.count
        var speakerAssignmentMatrix = Array(
            repeating: Array(repeating: Float(0), count: numClusters), count: numEmbeddings)

        for (i, cluster) in ahcClusters.enumerated() {
            guard cluster >= 0 else { continue }
            speakerAssignmentMatrix[i][cluster] = 1.0
        }

        if initialSmoothingFactor >= 0 {
            speakerAssignmentMatrix = MathOps.softmax(
                speakerAssignmentMatrix.map { row in
                    row.map { $0 * initialSmoothingFactor }
                })
        }

        let embeddingDimensionality = pldaEmbeddings[0].count

        var speakerPriorProbabilities = Array(
            repeating: Float(1.0) / Float(numClusters), count: numClusters)

        let perEmbeddingLogLikelihoodConstant = pldaEmbeddings.map { embedding in
            -0.5
                * (embedding.map { $0 * $0 }.reduce(0, +) + Float(embeddingDimensionality)
                    * log(2 * Float.pi))
        }

        let embeddingScalingFactors = betweenClassCovarianceDiag.map { sqrt($0) }

        let scaledEmbeddings = pldaEmbeddings.map { embedding in
            zip(embedding, embeddingScalingFactors).map { $0 * $1 }
        }

        var evidenceLowerBoundHistory: [Float] = []
        let convergenceThreshold: Float = 1e-4

        for iteration in 0..<maxIterations {
            let speakerAssignmentSums = (0..<numClusters).map { clusterIdx in
                speakerAssignmentMatrix.map { $0[clusterIdx] }.reduce(0, +)
            }

            let speakerPrecisionInverse = (0..<numClusters).map { clusterIdx in
                betweenClassCovarianceDiag.map { covDiag in
                    1.0
                        / (1 + speakerRelevanceFactorA / speakerRelevanceFactorB
                            * speakerAssignmentSums[clusterIdx] * covDiag)
                }
            }

            let speakerModelParameters =
                VariationalBayesHiddenMarkovModel.calculateSpeakerModelParameters(
                    scaledEmbeddings: scaledEmbeddings,
                    speakerPrecisionInverse: speakerPrecisionInverse,
                    speakerAssignmentMatrix: speakerAssignmentMatrix,
                    speakerRelevanceFactorA: speakerRelevanceFactorA,
                    speakerRelevanceFactorB: speakerRelevanceFactorB
                )

            let embeddingSpeakerLogLikelihoods =
                VariationalBayesHiddenMarkovModel.calculateLogLikelihoods(
                    scaledEmbeddings: scaledEmbeddings,
                    speakerPrecisionInverse: speakerPrecisionInverse,
                    speakerModelParameters: speakerModelParameters,
                    betweenClassCovarianceDiag: betweenClassCovarianceDiag,
                    perEmbeddingLogLikelihoodConstant: perEmbeddingLogLikelihoodConstant,
                    speakerRelevanceFactorA: speakerRelevanceFactorA
                )

            let logSpeakerPriors = speakerPriorProbabilities.map {
                guard $0 > -1e-8 else { return -Float.infinity }
                return log($0 + 1e-8)
            }

            let embeddingMarginalLogLikelihoods = embeddingSpeakerLogLikelihoods.map { logLikelihoods in
                MathOps.logSumExp(logLikelihoods.indices.map { logLikelihoods[$0] + logSpeakerPriors[$0] })
            }

            let totalLogLikelihood = embeddingMarginalLogLikelihoods.reduce(0, +)

            speakerAssignmentMatrix = (0..<numEmbeddings).map { embIdx in
                let logProbs = embeddingSpeakerLogLikelihoods[embIdx].enumerated().map {
                    $0.element + logSpeakerPriors[$0.offset] - embeddingMarginalLogLikelihoods[embIdx]
                }
                let maxLogProb = logProbs.max() ?? 0
                let expProbs = logProbs.map { exp($0 - maxLogProb) }
                let sumExpProbs = expProbs.reduce(0, +)
                guard sumExpProbs != 0 else {
                    return Array(repeating: Float(1.0) / Float(numClusters), count: numClusters)
                }
                return expProbs.map { $0 / sumExpProbs }
            }

            speakerPriorProbabilities = (0..<numClusters).map { clusterIdx in
                speakerAssignmentMatrix.map { $0[clusterIdx] }.reduce(0, +)
            }
            let sumPriors = speakerPriorProbabilities.reduce(0, +)
            speakerPriorProbabilities = speakerPriorProbabilities.map { $0 / sumPriors }

            var precisionTerm: Float = 0
            for clusterIdx in 0..<numClusters {
                for dim in 0..<embeddingDimensionality {
                    let inv = speakerPrecisionInverse[clusterIdx][dim]
                    guard inv > 0 else { continue }
                    let param = speakerModelParameters[clusterIdx][dim]
                    precisionTerm += log(inv) - inv - param * param + 1
                }
            }

            let elbo = totalLogLikelihood + speakerRelevanceFactorB * 0.5 * precisionTerm
            evidenceLowerBoundHistory.append(elbo)

            if iteration > 0 {
                let improvement = elbo - evidenceLowerBoundHistory[evidenceLowerBoundHistory.count - 2]
                if improvement < convergenceThreshold {
                    if improvement < 0 && verbose {
                        Logging.error("[VBx] Value of auxiliary function has decreased!")
                    }
                    break
                }
            }
        }

        #if DEBUG
        let vbxElapsed = CFAbsoluteTimeGetCurrent() - vbxStart
        Logging.debug("[VBx] completed in \(String(format: "%.3f", vbxElapsed))s")
        #endif

        return (speakerAssignmentMatrix, speakerPriorProbabilities)
    }

    private static func calculateLogLikelihoods(
        scaledEmbeddings: [[Float]],
        speakerPrecisionInverse: [[Float]],
        speakerModelParameters: [[Float]],
        betweenClassCovarianceDiag: [Float],
        perEmbeddingLogLikelihoodConstant: [Float],
        speakerRelevanceFactorA: Float
    ) -> [[Float]] {
        let (dotFlat, numEmbeddings, numClusters) = MathOps.matrixMultiply(
            scaledEmbeddings, MathOps.matrixTranspose(speakerModelParameters))

        let sqSMP = speakerModelParameters.map { $0.map { $0 * $0 } }
        let (sqSMPFlat, _, embedDim) = MathOps.flatten(sqSMP)
        var speakerIntermediate = Array(repeating: Float(0), count: numClusters * embedDim)
        vDSP_vadd(
            speakerPrecisionInverse.flatMap { $0 }, 1,
            sqSMPFlat, 1,
            &speakerIntermediate, 1,
            vDSP_Length(numClusters * embedDim)
        )

        let (speakerIntermediateDotCov, rows, cols) = MathOps.matrixMultiply(
            MathOps.unflatten(speakerIntermediate, numClusters, embedDim),
            betweenClassCovarianceDiag.map { [$0] }
        )

        precondition(rows * cols == numClusters, "rows * cols must equal numClusters")

        var speakerIntermediateDotCovScaled = Array(repeating: Float(0), count: numClusters)
        vDSP_vsmul(
            speakerIntermediateDotCov, 1,
            [0.5],
            &speakerIntermediateDotCovScaled, 1,
            vDSP_Length(numClusters)
        )

        let speakerPerEmbeddingLogLikelihoods = (0..<numEmbeddings).map { embIdx in
            (0..<numClusters).map { clusterIdx in
                speakerIntermediateDotCovScaled[clusterIdx] + perEmbeddingLogLikelihoodConstant[embIdx]
            }
        }

        var out = Array(repeating: Float(0), count: numEmbeddings * numClusters)
        vDSP_vsub(
            speakerPerEmbeddingLogLikelihoods.flatMap { $0 }, 1,
            dotFlat, 1,
            &out, 1,
            vDSP_Length(numEmbeddings * numClusters)
        )

        var logLikelihoods = Array(repeating: Float(0), count: numEmbeddings * numClusters)
        vDSP_vsmul(
            out, 1,
            [speakerRelevanceFactorA],
            &logLikelihoods, 1,
            vDSP_Length(numEmbeddings * numClusters)
        )

        return MathOps.unflatten(logLikelihoods, numEmbeddings, numClusters)
    }

    private static func calculateSpeakerModelParameters(
        scaledEmbeddings: [[Float]],
        speakerPrecisionInverse: [[Float]],
        speakerAssignmentMatrix: [[Float]],
        speakerRelevanceFactorA: Float,
        speakerRelevanceFactorB: Float
    ) -> [[Float]] {
        let relevanceScale = speakerRelevanceFactorA / speakerRelevanceFactorB

        let (dotFlat, rows, cols) = MathOps.matrixMultiply(
            MathOps.matrixTranspose(speakerAssignmentMatrix), scaledEmbeddings)

        let spiFlat = speakerPrecisionInverse.flatMap { $0 }
        var scaledSPI = Array(repeating: Float(0), count: rows * cols)
        vDSP_vsmul(
            spiFlat, 1,
            [relevanceScale],
            &scaledSPI, 1,
            vDSP_Length(rows * cols)
        )

        var out = Array(repeating: Float(0), count: rows * cols)
        vDSP_vmul(
            scaledSPI, 1,
            dotFlat, 1,
            &out, 1,
            vDSP_Length(rows * cols)
        )

        return MathOps.unflatten(out, rows, cols)
    }
}
