//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import XCTest

@testable import SpeakerKit


final class ClusterAlgorithmsTests: XCTestCase {

    func testNegativeClusterIndicesGuard() throws {
        let ahcClusters = [0, 1, -1, 2, -5, 3]
        let pldaEmbeddings = Array(repeating: Array(repeating: Float(0.1), count: 128), count: 6)

        let result = VariationalBayesHiddenMarkovModel.vbx(
            ahcClusters: ahcClusters,
            pldaEmbeddings: pldaEmbeddings,
            maxIterations: 1
        )

        let speakerAssignmentMatrix = result.speakerAssignmentMatrix
        let speakerPriorProbabilities = result.speakerPriorProbabilities

        XCTAssertEqual(speakerAssignmentMatrix.count, 6)
        XCTAssertEqual(speakerAssignmentMatrix[0].count, 4)
        XCTAssertEqual(speakerPriorProbabilities.count, 4)

        for i in [2, 4] {
            let row = speakerAssignmentMatrix[i]
            let sum = row.reduce(0, +)
            XCTAssertEqual(sum, 1.0, accuracy: 1e-6, "Row \(i) should sum to 1.0")
            for prob in row {
                XCTAssertGreaterThanOrEqual(prob, 0.0)
            }
        }

        for row in speakerAssignmentMatrix {
            let sum = row.reduce(0, +)
            XCTAssertEqual(sum, 1.0, accuracy: 1e-6)
            for prob in row {
                XCTAssertGreaterThanOrEqual(prob, 0.0)
                XCTAssertLessThanOrEqual(prob, 1.0)
            }
        }

        let priorSum = speakerPriorProbabilities.reduce(0, +)
        XCTAssertEqual(priorSum, 1.0, accuracy: 1e-6)
        for prob in speakerPriorProbabilities {
            XCTAssertGreaterThanOrEqual(prob, 0.0)
        }
    }

    func testNegativeSpeakerPriorsGuard() throws {
        let ahcClusters = [0, 1, 2]
        let pldaEmbeddings = Array(repeating: Array(repeating: Float(0.1), count: 128), count: 3)

        let result = VariationalBayesHiddenMarkovModel.vbx(
            ahcClusters: ahcClusters,
            pldaEmbeddings: pldaEmbeddings,
            speakerRelevanceFactorA: 0.001,
            speakerRelevanceFactorB: 0.001,
            maxIterations: 5
        )

        XCTAssertEqual(result.speakerAssignmentMatrix.count, 3)
        XCTAssertEqual(result.speakerPriorProbabilities.count, 3)

        for row in result.speakerAssignmentMatrix {
            let sum = row.reduce(0, +)
            XCTAssertEqual(sum, 1.0, accuracy: 1e-6)
            for prob in row {
                XCTAssertGreaterThanOrEqual(prob, 0.0)
                XCTAssertLessThanOrEqual(prob, 1.0)
            }
        }

        let priorSum = result.speakerPriorProbabilities.reduce(0, +)
        XCTAssertEqual(priorSum, 1.0, accuracy: 1e-6)
        for prob in result.speakerPriorProbabilities {
            XCTAssertGreaterThanOrEqual(prob, 0.0)
        }
    }

    func testZeroSumExpProbsGuard() throws {
        let ahcClusters = [0, 1]
        let pldaEmbeddings = Array(repeating: Array(repeating: Float(0.1), count: 128), count: 2)

        let result = VariationalBayesHiddenMarkovModel.vbx(
            ahcClusters: ahcClusters,
            pldaEmbeddings: pldaEmbeddings,
            speakerRelevanceFactorA: 1000.0,
            speakerRelevanceFactorB: 0.001,
            maxIterations: 3
        )

        XCTAssertEqual(result.speakerAssignmentMatrix.count, 2)
        XCTAssertEqual(result.speakerAssignmentMatrix[0].count, 2)

        for (i, row) in result.speakerAssignmentMatrix.enumerated() {
            let sum = row.reduce(0, +)
            XCTAssertEqual(sum, 1.0, accuracy: 1e-6, "Row \(i) should sum to 1.0")
            for (j, prob) in row.enumerated() {
                XCTAssertGreaterThanOrEqual(prob, 0.0, "Probability at [\(i)][\(j)] should be non-negative")
                XCTAssertLessThanOrEqual(prob, 1.0)
            }
        }
    }

    func testPrecisionInverseGuard() throws {
        let ahcClusters = [0, 1, 2]
        let pldaEmbeddings = Array(repeating: Array(repeating: Float(0.1), count: 128), count: 3)

        let result = VariationalBayesHiddenMarkovModel.vbx(
            ahcClusters: ahcClusters,
            pldaEmbeddings: pldaEmbeddings,
            speakerRelevanceFactorA: 0.001,
            speakerRelevanceFactorB: 1000.0,
            maxIterations: 5
        )

        XCTAssertEqual(result.speakerAssignmentMatrix.count, 3)
        XCTAssertEqual(result.speakerPriorProbabilities.count, 3)

        for row in result.speakerAssignmentMatrix {
            let sum = row.reduce(0, +)
            XCTAssertEqual(sum, 1.0, accuracy: 1e-6)
        }

        let priorSum = result.speakerPriorProbabilities.reduce(0, +)
        XCTAssertEqual(priorSum, 1.0, accuracy: 1e-6)
    }

    func testEdgeCases() throws {
        let singleEmbedding =
            [Float(0.1), Float(0.2), Float(0.3)] + Array(repeating: Float(0.0), count: 125)
        let singleCluster = [0]

        let result1 = VariationalBayesHiddenMarkovModel.vbx(
            ahcClusters: singleCluster,
            pldaEmbeddings: [singleEmbedding]
        )

        XCTAssertEqual(result1.speakerAssignmentMatrix.count, 1)
        XCTAssertEqual(result1.speakerAssignmentMatrix[0].count, 1)
        XCTAssertEqual(result1.speakerPriorProbabilities.count, 1)

        let mixedClusters = [-1, 0, -2, 1]
        let result2 = VariationalBayesHiddenMarkovModel.vbx(
            ahcClusters: mixedClusters,
            pldaEmbeddings: Array(repeating: singleEmbedding, count: 4)
        )

        XCTAssertEqual(result2.speakerAssignmentMatrix.count, 4)
        XCTAssertEqual(result2.speakerAssignmentMatrix[0].count, 2)
        XCTAssertEqual(result2.speakerPriorProbabilities.count, 2)

        for row in result2.speakerAssignmentMatrix {
            let sum = row.reduce(0, +)
            XCTAssertEqual(sum, 1.0, accuracy: 1e-6)
        }
    }

    // MARK: - ClusterAlgorithms.kMeans Tests

    func testKMeansTwoWellSeparatedClusters() {
        // Two tightly packed groups far apart — any correct clustering must separate them.
        let groupA: [[Float]] = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]]
        let groupB: [[Float]] = [[10.0, 10.0], [10.1, 10.0], [10.0, 10.1], [10.1, 10.1]]
        let embeddings = groupA + groupB

        let assignments = ClusterAlgorithms.kMeans(embeddings: embeddings, clusterCount: 2)

        XCTAssertEqual(assignments.count, 8)
        XCTAssertEqual(Set(assignments).count, 2)
        let labelA = assignments[0]
        for i in 1..<4 { XCTAssertEqual(assignments[i], labelA, "groupA point \(i) should share cluster with point 0") }
        let labelB = assignments[4]
        XCTAssertNotEqual(labelA, labelB)
        for i in 5..<8 { XCTAssertEqual(assignments[i], labelB, "groupB point \(i) should share cluster with point 4") }
    }

    func testKMeansThreeWellSeparatedClusters() {
        // Three groups on orthogonal axes, well separated.
        let groupA: [[Float]] = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]
        let groupB: [[Float]] = [[10.0, 0.0], [10.1, 0.0], [10.0, 0.1]]
        let groupC: [[Float]] = [[0.0, 10.0], [0.1, 10.0], [0.0, 10.1]]
        let embeddings = groupA + groupB + groupC

        let assignments = ClusterAlgorithms.kMeans(embeddings: embeddings, clusterCount: 3)

        XCTAssertEqual(Set(assignments).count, 3)
        let la = assignments[0]
        for i in 1..<3 { XCTAssertEqual(assignments[i], la) }
        let lb = assignments[3]
        XCTAssertNotEqual(la, lb)
        for i in 4..<6 { XCTAssertEqual(assignments[i], lb) }
        let lc = assignments[6]
        XCTAssertNotEqual(la, lc)
        XCTAssertNotEqual(lb, lc)
        for i in 7..<9 { XCTAssertEqual(assignments[i], lc) }
    }

    func testKMeansClusterCountOne() {
        let embeddings: [[Float]] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        let assignments = ClusterAlgorithms.kMeans(embeddings: embeddings, clusterCount: 1)
        XCTAssertEqual(assignments, [0, 0, 0])
    }

    func testKMeansClusterCountEqualsEmbeddingCount() {
        let embeddings: [[Float]] = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        let assignments = ClusterAlgorithms.kMeans(embeddings: embeddings, clusterCount: 3)
        XCTAssertEqual(Set(assignments).count, 3)
    }

    func testKMeansClusterCountExceedsEmbeddingCount() {
        // clusterCount is clamped to embeddingCount — all points get distinct labels
        let embeddings: [[Float]] = [[1.0, 0.0], [0.0, 1.0]]
        let assignments = ClusterAlgorithms.kMeans(embeddings: embeddings, clusterCount: 100)
        XCTAssertEqual(Set(assignments).count, 2)
    }

    func testKMeansOutputLengthMatchesInput() {
        let embeddings: [[Float]] = (0..<15).map { i in [Float(i), 0.0] }
        let assignments = ClusterAlgorithms.kMeans(embeddings: embeddings, clusterCount: 3)
        XCTAssertEqual(assignments.count, 15)
    }

    func testKMeansAllAssignmentsInValidRange() {
        let targetClusterCount = 4
        let embeddings: [[Float]] = (0..<12).map { i in [Float(i), 0.0] }
        let assignments = ClusterAlgorithms.kMeans(embeddings: embeddings, clusterCount: targetClusterCount)
        for assignment in assignments {
            XCTAssertGreaterThanOrEqual(assignment, 0)
            XCTAssertLessThan(assignment, targetClusterCount)
        }
    }

    func testKMeansDeterministicWithFixedSeed() {
        // Fixed randomSeed must produce identical results across calls.
        let embeddings: [[Float]] = (0..<20).map { i in
            [Float(i % 5) * 0.01, Float(i / 5) * 10.0]
        }
        let first = ClusterAlgorithms.kMeans(embeddings: embeddings, clusterCount: 4, randomSeed: 42)
        let second = ClusterAlgorithms.kMeans(embeddings: embeddings, clusterCount: 4, randomSeed: 42)
        XCTAssertEqual(first, second)
    }

    func testKMeansDifferentSeedsCanDiffer() {
        // Two different seeds on ambiguous data should occasionally produce different inits.
        // We just verify both are valid (correct count, in-range), not that they must differ.
        let embeddings: [[Float]] = (0..<30).map { i in
            [Float(i % 3) * 5.0 + Float(i) * 0.001, 0.0]
        }
        let a = ClusterAlgorithms.kMeans(embeddings: embeddings, clusterCount: 3, randomSeed: 1)
        let b = ClusterAlgorithms.kMeans(embeddings: embeddings, clusterCount: 3, randomSeed: 99)
        XCTAssertEqual(a.count, 30)
        XCTAssertEqual(b.count, 30)
        for v in a { XCTAssertLessThan(v, 3) }
        for v in b { XCTAssertLessThan(v, 3) }
    }

    func testKMeansHighDimensionalEmbeddings() {
        // Verify correctness holds for embedding dimensions typical in speaker diarization.
        let dimension = 128
        let clusterCenters: [[Float]] = [
            Array(repeating: 0.0, count: dimension),
            Array(repeating: 1.0, count: dimension),
            Array(repeating: -1.0, count: dimension)
        ]
        var embeddings = [[Float]]()
        for center in clusterCenters {
            for _ in 0..<5 {
                // Add small noise around each center
                embeddings.append(center.map { $0 + Float.random(in: -0.01...0.01) })
            }
        }
        let assignments = ClusterAlgorithms.kMeans(embeddings: embeddings, clusterCount: 3)
        XCTAssertEqual(Set(assignments).count, 3)
        // Points near the same center must share the same label.
        for startIndex in stride(from: 0, to: 15, by: 5) {
            let label = assignments[startIndex]
            for i in (startIndex + 1)..<(startIndex + 5) {
                XCTAssertEqual(assignments[i], label, "Points near same center should cluster together")
            }
        }
    }

    func testKMeansUsesMultipleRestarts() {
        // With numberOfRestarts=1 vs 3 both must produce valid output.
        let embeddings: [[Float]] = (0..<12).map { i in [Float(i % 4) * 5.0, 0.0] }
        let singleRun = ClusterAlgorithms.kMeans(embeddings: embeddings, clusterCount: 4, numberOfRestarts: 1)
        let multiRun = ClusterAlgorithms.kMeans(embeddings: embeddings, clusterCount: 4, numberOfRestarts: 3)
        XCTAssertEqual(Set(singleRun).count, 4)
        XCTAssertEqual(Set(multiRun).count, 4)
    }

    func testKMeansHandlesCollidingEmbeddings() {
        // All points identical except one outlier. With k=3, at least one cluster
        // will be empty during iterations -- exercises the previousCentroids fallback.
        let embeddings: [[Float]] = Array(repeating: [1.0, 1.0], count: 10) + [[100.0, 100.0]]
        let assignments = ClusterAlgorithms.kMeans(embeddings: embeddings, clusterCount: 3)
        XCTAssertEqual(assignments.count, 11)
        for assignment in assignments {
            XCTAssertGreaterThanOrEqual(assignment, 0)
            XCTAssertLessThan(assignment, 3)
        }
    }

    // MARK: - VBxClustering Cluster Coverage

    func testVBxClusteringAllNonTrainableEmbeddingsGetAssigned() async {
        // All embeddings have nonOverlappedFrameRatio = 0, below the default minActiveRatio
        // of 0.2, so trainableEmbeddings is empty. This forces the centroids.isEmpty path.
        // The bug: clusters returned from cluster() was [], leaving all clusterId at -1 after
        // update()'s enumeration touched zero elements.
        let dim = 128
        let embeddings = (0..<5).map { i in
            SpeakerEmbedding(
                embedding: (0..<dim).map { d in Float(i * dim + d) * 0.01 },
                pldaEmbedding: Array(repeating: Float(0.1), count: 128),
                activeFrames: [0.0],
                windowIndex: i,
                speakerIndex: 0,
                nonOverlappedFrameRatio: 0.0
            )
        }

        let clustering = VBxClustering()
        await clustering.add(speakerEmbeddings: embeddings)
        let result = await clustering.update(config: VBxClusteringConfig())

        XCTAssertEqual(result.clusterIndices.count, embeddings.count,
                       "clusterIndices must cover all embeddings, not just trainable ones")
        XCTAssertEqual(result.speakerEmbeddings.count, embeddings.count)
        for embedding in result.speakerEmbeddings {
            XCTAssertGreaterThanOrEqual(embedding.clusterId, 0,
                                        "Non-trainable embeddings must not retain the default clusterId of -1")
        }
    }

    func testVBxClusteringNonTrainableEmbeddingsGetAssignedAlongWithTrainable() async {
        // Trainable embeddings (nonOverlappedFrameRatio = 1.0) drive the VBx clustering;
        // non-trainable ones (ratio = 0.0) are excluded from training but must still receive a
        // valid clusterId via clusterReassignment over all N embeddings.
        let dim = 128
        let trainable = (0..<4).map { i in
            SpeakerEmbedding(
                embedding: (0..<dim).map { _ in Float(i) * 0.5 },
                pldaEmbedding: Array(repeating: Float(i) * 0.1 + 0.1, count: 128),
                activeFrames: [1.0],
                windowIndex: i,
                speakerIndex: i % 2,
                nonOverlappedFrameRatio: 1.0
            )
        }
        let nonTrainable = (4..<7).map { i in
            SpeakerEmbedding(
                embedding: (0..<dim).map { _ in Float(i) * 0.5 },
                pldaEmbedding: Array(repeating: Float(0.1), count: 128),
                activeFrames: [0.0],
                windowIndex: i,
                speakerIndex: i % 2,
                nonOverlappedFrameRatio: 0.0
            )
        }
        let total = trainable.count + nonTrainable.count

        let clustering = VBxClustering()
        await clustering.add(speakerEmbeddings: trainable + nonTrainable)
        let result = await clustering.update(config: VBxClusteringConfig())

        XCTAssertEqual(result.clusterIndices.count, total,
                       "clusterIndices must cover all \(total) embeddings")
        XCTAssertEqual(result.speakerEmbeddings.count, total)
        for embedding in result.speakerEmbeddings {
            XCTAssertGreaterThanOrEqual(embedding.clusterId, 0,
                                        "Every embedding (trainable or not) must receive a valid clusterId")
        }
    }

    // MARK: - VBxClustering numSpeakers Config

    func testVBxClusteringConfigPassesNumSpeakers() {
        let options = PyannoteDiarizationOptions(numberOfSpeakers: 3)
        let config = VBxClusteringConfig(from: options)
        XCTAssertEqual(config.numSpeakers, 3)
    }

    func testNumericalStability() throws {
        let ahcClusters = [0, 1, 2]
        let pldaEmbeddings = Array(repeating: Array(repeating: Float(0.1), count: 128), count: 3)

        let result = VariationalBayesHiddenMarkovModel.vbx(
            ahcClusters: ahcClusters,
            pldaEmbeddings: pldaEmbeddings,
            speakerRelevanceFactorA: 0.0001,
            speakerRelevanceFactorB: 10000.0,
            maxIterations: 10
        )

        XCTAssertEqual(result.speakerAssignmentMatrix.count, 3)
        XCTAssertEqual(result.speakerPriorProbabilities.count, 3)

        for (i, row) in result.speakerAssignmentMatrix.enumerated() {
            for (j, prob) in row.enumerated() {
                XCTAssertFalse(prob.isNaN, "Probability at [\(i)][\(j)] should not be NaN")
                XCTAssertFalse(prob.isInfinite, "Probability at [\(i)][\(j)] should not be infinite")
                XCTAssertGreaterThanOrEqual(prob, 0.0)
            }
        }

        for (i, prob) in result.speakerPriorProbabilities.enumerated() {
            XCTAssertFalse(prob.isNaN, "Speaker prior \(i) should not be NaN")
            XCTAssertFalse(prob.isInfinite, "Speaker prior \(i) should not be infinite")
            XCTAssertGreaterThanOrEqual(prob, 0.0)
        }
    }
}
