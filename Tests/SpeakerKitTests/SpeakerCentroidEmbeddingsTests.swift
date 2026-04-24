//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import XCTest
import WhisperKit
@testable import SpeakerKit

final class SpeakerCentroidEmbeddingsTests: XCTestCase {

    // MARK: - Helpers

    private func loadAudio(named name: String, extension ext: String = "wav") throws -> [Float] {
        guard let url = Bundle.module.url(forResource: name, withExtension: ext) else {
            throw XCTSkip("Audio file \(name).\(ext) not found in test bundle")
        }
        let audioBuffer = try AudioProcessor.loadAudio(fromPath: url.path)
        return AudioProcessor.convertBufferToArray(buffer: audioBuffer)
    }

    private func assertVectorsEqual(
        _ actual: [Float],
        _ expected: [Float],
        accuracy: Float = 1e-5,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, "Vector lengths differ", file: file, line: line)
        for (i, (a, e)) in zip(actual, expected).enumerated() {
            XCTAssertEqual(a, e, accuracy: accuracy, "Mismatch at index \(i)", file: file, line: line)
        }
    }

    // MARK: - Unit tests: calculateCentroids (main VBx path)

    /// With one-hot responsibility per speaker, the weighted mean collapses to the
    /// arithmetic mean of the embeddings owned by that speaker.
    func testCalculateCentroids_uniformWeightsEqualsArithmeticMean() async {
        let embeddings: [[Float]] = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]
        // speakerWeights[s][e] -- 3 speakers owning embeddings 0, 1, and (2+3)
        let speakerWeights: [[Float]] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0]
        ]

        let clusterer = VBxClustering()
        let centroids = await clusterer.calculateCentroids(
            speakerWeights: speakerWeights,
            embeddings: embeddings
        )

        XCTAssertEqual(centroids.count, 3)
        assertVectorsEqual(centroids[0], embeddings[0])
        assertVectorsEqual(centroids[1], embeddings[1])
        let expectedMix = zip(embeddings[2], embeddings[3]).map { ($0 + $1) / 2 }
        assertVectorsEqual(centroids[2], expectedMix)
    }

    /// Fractional responsibility weights must produce sum(w_i * x_i) / sum(w_i).
    func testCalculateCentroids_weightedMean() async {
        let embeddings: [[Float]] = [
            [2.0, 4.0],
            [6.0, 8.0]
        ]
        let speakerWeights: [[Float]] = [
            [0.7, 0.3],
            [0.1, 0.9]
        ]

        let clusterer = VBxClustering()
        let centroids = await clusterer.calculateCentroids(
            speakerWeights: speakerWeights,
            embeddings: embeddings
        )

        XCTAssertEqual(centroids.count, 2)
        // speaker 0: (0.7*2 + 0.3*6) / 1.0 = 3.2, (0.7*4 + 0.3*8) / 1.0 = 5.2
        assertVectorsEqual(centroids[0], [3.2, 5.2])
        // speaker 1: (0.1*2 + 0.9*6) / 1.0 = 5.6, (0.1*4 + 0.9*8) / 1.0 = 7.6
        assertVectorsEqual(centroids[1], [5.6, 7.6])
    }

    /// A speaker with zero total weight must stay at the helper's sentinel (zeros) and
    /// must not divide by zero or crash.
    func testCalculateCentroids_zeroWeightSpeakerIsSkipped() async {
        let embeddings: [[Float]] = [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0]
        ]
        let speakerWeights: [[Float]] = [
            [0.5, 0.5],
            [0.0, 0.0]
        ]

        let clusterer = VBxClustering()
        let centroids = await clusterer.calculateCentroids(
            speakerWeights: speakerWeights,
            embeddings: embeddings
        )

        XCTAssertEqual(centroids.count, 2)
        assertVectorsEqual(centroids[0], [1.5, 1.5, 1.5])
        assertVectorsEqual(centroids[1], [0.0, 0.0, 0.0])
    }

    // MARK: - Unit tests: centroidsFromAssignments (kMeans + AHC fallback paths)

    /// Arithmetic mean per cluster for a simple two-cluster partition.
    func testCentroidsFromAssignments_arithmeticMean() async {
        let dim = 8
        let embeddings: [[Float]] = (0..<6).map { i in
            Array(repeating: Float(i), count: dim)
        }
        let assignments = [0, 0, 0, 1, 1, 1]

        let clusterer = VBxClustering()
        let centroids = await clusterer.centroidsFromAssignments(
            assignments: assignments,
            embeddings: embeddings,
            clusterCount: 2
        )

        XCTAssertEqual(centroids.count, 2)
        assertVectorsEqual(centroids[0], Array(repeating: 1.0, count: dim))
        assertVectorsEqual(centroids[1], Array(repeating: 4.0, count: dim))
    }

    /// A cluster with exactly one embedding must return that embedding as its centroid.
    func testCentroidsFromAssignments_singletonCluster() async {
        let embeddings: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        let assignments = [0, 1, 0]

        let clusterer = VBxClustering()
        let centroids = await clusterer.centroidsFromAssignments(
            assignments: assignments,
            embeddings: embeddings,
            clusterCount: 2
        )

        XCTAssertEqual(centroids.count, 2)
        assertVectorsEqual(centroids[0], [4.0, 5.0, 6.0])
        assertVectorsEqual(centroids[1], [4.0, 5.0, 6.0])
    }

    /// A cluster id with no members must yield a zero vector, not NaN or crash.
    func testCentroidsFromAssignments_emptyCluster() async {
        let embeddings: [[Float]] = [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0]
        ]
        let assignments = [0, 1]

        let clusterer = VBxClustering()
        let centroids = await clusterer.centroidsFromAssignments(
            assignments: assignments,
            embeddings: embeddings,
            clusterCount: 3
        )

        XCTAssertEqual(centroids.count, 3)
        assertVectorsEqual(centroids[0], [1.0, 2.0, 3.0])
        assertVectorsEqual(centroids[1], [2.0, 3.0, 4.0])
        assertVectorsEqual(centroids[2], [0.0, 0.0, 0.0])
    }

    // MARK: - Clusterer-level invariant: value matches post-reassignment mean

    /// For any embedding input that drives VBxClustering end-to-end, the surfaced
    /// `speakerCentroids[k]` must equal the arithmetic mean of the embeddings whose final
    /// `clusterIndices[i]` is `k`. This holds irrespective of which internal path
    /// (VBx weighted, kMeans correction, AHC fallback) produced the assignments, because the
    /// final recompute is always a plain `centroidsFromAssignments(...)` over the
    /// post-reassignment labels.
    func testCentroidValuesMatchFinalAssignmentMean() async {
        let dim = 128
        // two separated groups in raw embedder space (4 + 4), with an explicit two-speaker
        // request so this fixture exercises the post-reassignment centroid recompute path.
        let groupA: [[Float]] = (0..<4).map { i in
            var v = Array(repeating: Float(0), count: dim)
            v[0] = 1.0
            v[1] = Float(i) * 0.01
            return v
        }
        let groupB: [[Float]] = (0..<4).map { i in
            var v = Array(repeating: Float(0), count: dim)
            v[0] = -1.0
            v[1] = Float(i) * 0.01
            return v
        }
        let raw = groupA + groupB
        let plda = raw

        let speakerEmbeddings = (0..<raw.count).map { i in
            SpeakerEmbedding(
                embedding: raw[i],
                pldaEmbedding: plda[i],
                activeFrames: [1.0],
                windowIndex: i,
                speakerIndex: 0,
                nonOverlappedFrameRatio: 1.0
            )
        }

        let clusterer = VBxClustering()
        await clusterer.add(speakerEmbeddings: speakerEmbeddings)
        let result = await clusterer.update(config: VBxClusteringConfig(numSpeakers: 2))

        let clusters = result.clusterIndices
        XCTAssertEqual(clusters.count, raw.count)
        guard let maxCluster = clusters.max() else {
            XCTFail("expected at least one cluster")
            return
        }
        let kFinal = maxCluster + 1
        XCTAssertEqual(kFinal, 2, "expected the synthetic fixture to keep two clusters")

        let expected = await clusterer.centroidsFromAssignments(
            assignments: clusters,
            embeddings: raw,
            clusterCount: kFinal
        )

        XCTAssertFalse(result.speakerCentroids.isEmpty, "speakerCentroids must be populated")
        for k in 0..<kFinal {
            guard let actual = result.speakerCentroids[k] else {
                XCTFail("speakerCentroids missing entry for cluster \(k)")
                continue
            }
            assertVectorsEqual(actual, expected[k], accuracy: 1e-5)
        }
    }

    // MARK: - Integration tests on VADAudio

    /// keys must equal the set of speaker ids appearing in segments (keyed by `speakerIds`,
    /// not `speakerId`, since `SpeakerInfo.speakerId` is `Int?`).
    func testCentroidsMatchClusterIds() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()
        let result = try await speakerKit.diarize(
            audioArray: audioArray,
            options: PyannoteDiarizationOptions(numberOfSpeakers: 3)
        )

        XCTAssertFalse(result.speakerCentroidEmbeddings.isEmpty,
                       "speakerCentroidEmbeddings must be populated for the Pyannote backend")

        let segmentIds = Set(result.segments.flatMap { $0.speaker.speakerIds })
        let keys = Set(result.speakerCentroidEmbeddings.keys)

        XCTAssertTrue(segmentIds.isSubset(of: keys),
                      "every speakerId in segments must have a centroid")
        for key in keys {
            XCTAssertGreaterThanOrEqual(key, 0)
            XCTAssertLessThan(key, result.speakerCount)
        }

        let dims = Set(result.speakerCentroidEmbeddings.values.map { $0.count })
        XCTAssertEqual(dims.count, 1, "all centroids must share one embedding dimension")
        XCTAssertGreaterThan(dims.first ?? 0, 0, "centroid dimension must be positive")
    }

    /// Every centroid value must be finite, and centroids from real runs must not collapse
    /// to empty or zero vectors.
    func testCentroidsAreFiniteAndBounded() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()
        let result = try await speakerKit.diarize(audioArray: audioArray)

        for (id, centroid) in result.speakerCentroidEmbeddings {
            XCTAssertFalse(centroid.isEmpty, "centroid for speaker \(id) must not be empty")
            for value in centroid {
                XCTAssertTrue(value.isFinite, "centroid for speaker \(id) has non-finite value")
            }
            let norm = sqrt(centroid.reduce(0) { $0 + $1 * $1 })
            XCTAssertGreaterThan(norm, 0.0, "centroid \(id) has zero norm")
        }
    }

    /// Two identical calls on the same audio must produce matching centroids per speaker id.
    func testCentroidsStableAcrossReruns() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()
        let first = try await speakerKit.diarize(audioArray: audioArray)
        let second = try await speakerKit.diarize(audioArray: audioArray)

        XCTAssertEqual(Set(first.speakerCentroidEmbeddings.keys),
                       Set(second.speakerCentroidEmbeddings.keys),
                       "speaker ids must match across identical reruns")

        for id in first.speakerCentroidEmbeddings.keys {
            guard let a = first.speakerCentroidEmbeddings[id],
                  let b = second.speakerCentroidEmbeddings[id] else {
                XCTFail("missing centroid for speaker \(id)")
                continue
            }
            let distance = MathOps.cosineDistance(a, b)
            XCTAssertLessThan(distance, 1e-5,
                              "centroid for speaker \(id) drifted by \(distance) between reruns")
        }
    }

    /// Exercises both the default VBx + reassignment path (`numberOfSpeakers: nil`) and the
    /// kMeans correction path (`numberOfSpeakers: 2`). In both runs, centroid keys must
    /// track the final, post-reassignment cluster ids visible in `segments`.
    func testCentroidKeysSurviveClusterReassignment() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()

        let auto = try await speakerKit.diarize(audioArray: audioArray)
        let forcedTwo = try await speakerKit.diarize(
            audioArray: audioArray,
            options: PyannoteDiarizationOptions(numberOfSpeakers: 2)
        )

        for result in [auto, forcedTwo] {
            let segmentIds = Set(result.segments.flatMap { $0.speaker.speakerIds })
            let keys = Set(result.speakerCentroidEmbeddings.keys)
            XCTAssertTrue(segmentIds.isSubset(of: keys),
                          "every post-reassignment speakerId in segments must have a centroid")
            for key in keys {
                XCTAssertGreaterThanOrEqual(key, 0)
                XCTAssertLessThan(key, result.speakerCount)
            }
        }
    }

    // MARK: - centroidCosineDistance helper

    func testGenericInitAcceptsSpeakerCentroidEmbeddings() {
        let segments = [
            SpeakerSegment(speaker: .speakerId(7), startTime: 0.0, endTime: 1.0, frameRate: 100)
        ]
        let result = DiarizationResult(
            speakerCount: 1,
            totalFrames: 100,
            frameRate: 100,
            segments: segments,
            speakerCentroidEmbeddings: [7: [1.0, 0.0, 0.0]]
        )

        XCTAssertEqual(result.speakerCentroidEmbeddings[7], [1.0, 0.0, 0.0])
    }

    /// Pairwise distances must be finite, in `[0, 2]`, and delegate exactly to
    /// `MathOps.cosineDistance`. Missing ids yield `nil`.
    func testCentroidCosineDistance_sameDiarization() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()
        let result = try await speakerKit.diarize(audioArray: audioArray)

        let ids = Array(result.speakerCentroidEmbeddings.keys).sorted()
        guard ids.count >= 2 else {
            throw XCTSkip("need at least two speakers to compare centroids")
        }

        for i in 0..<ids.count {
            for j in (i + 1)..<ids.count {
                guard let distance = result.centroidCosineDistance(between: ids[i], ids[j]) else {
                    XCTFail("distance must not be nil for existing ids \(ids[i]), \(ids[j])")
                    continue
                }
                XCTAssertTrue(distance.isFinite)
                XCTAssertGreaterThanOrEqual(distance, 0.0)
                XCTAssertLessThanOrEqual(distance, 2.0)
            }
        }

        // pin that the helper delegates to MathOps.cosineDistance rather than rolling
        // its own maths.
        let a = ids[0]
        let b = ids[1]
        let helperValue = result.centroidCosineDistance(between: a, b)
        let directValue = MathOps.cosineDistance(
            result.speakerCentroidEmbeddings[a] ?? [],
            result.speakerCentroidEmbeddings[b] ?? []
        )
        XCTAssertEqual(helperValue, directValue)

        let missingId = (ids.max() ?? 0) + 10_000
        XCTAssertNil(result.centroidCosineDistance(between: missingId, a))
        XCTAssertNil(result.centroidCosineDistance(between: a, missingId))
    }

    func testNearestSpeakerCentroid() {
        let result = DiarizationResult(
            speakerCount: 4,
            totalFrames: 0,
            frameRate: 100,
            segments: [],
            speakerCentroidEmbeddings: [
                0: [1.0, 0.0, 0.0],
                1: [0.0, 1.0, 0.0],
                2: [0.9, 0.1, 0.0],
                3: [1.0, 0.0]
            ]
        )

        let nearest = result.nearestSpeakerCentroid(to: [1.0, 0.0, 0.0])
        XCTAssertEqual(nearest?.speakerId, 0)
        XCTAssertEqual(nearest?.distance, 0.0)

        XCTAssertNil(result.nearestSpeakerCentroid(to: []))
        XCTAssertNil(result.nearestSpeakerCentroid(to: [1.0, 0.0, 0.0, 0.0]))
    }
}
