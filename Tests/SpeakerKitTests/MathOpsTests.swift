//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import XCTest

@testable import SpeakerKit

final class MathOpsTests: XCTestCase {

    // MARK: - Test Data Setup

    private func assertEqual(
        _ actual: Float, _ expected: Float, accuracy: Float = 1e-6, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual, expected, accuracy: accuracy, file: file, line: line)
    }

    private func assertEqual(
        _ actual: [Float], _ expected: [Float], accuracy: Float = 1e-6, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(
            actual.count, expected.count, "Array lengths don't match", file: file, line: line)
        for (i, (a, e)) in zip(actual, expected).enumerated() {
            XCTAssertEqual(
                a, e, accuracy: accuracy, "Values at index \(i) don't match", file: file, line: line)
        }
    }

    private func assertEqual(
        _ actual: [[Float]], _ expected: [[Float]], accuracy: Float = 1e-6,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(
            actual.count, expected.count, "Matrix row counts don't match", file: file, line: line)
        for (a, e) in zip(actual, expected) {
            assertEqual(a, e, accuracy: accuracy, file: file, line: line)
        }
    }

    // MARK: - Matrix Multiplication Tests

    func testMatrixMultiply() {
        let matrixA: [[Float]] = [[1.0, 2.0], [3.0, 4.0]]
        let matrixB: [[Float]] = [[5.0, 6.0], [7.0, 8.0]]
        let result = MathOps.matrixMultiply(matrixA, matrixB)

        let expected: [Float] = [19.0, 22.0, 43.0, 50.0]
        assertEqual(result.result, expected)
        XCTAssertEqual(result.rows, 2)
        XCTAssertEqual(result.cols, 2)

        let matrixC: [[Float]] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        let matrixD: [[Float]] = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        let result2 = MathOps.matrixMultiply(matrixC, matrixD)

        let expected2: [Float] = [
            11.0, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0
        ]
        assertEqual(result2.result, expected2)
        XCTAssertEqual(result2.rows, 3)
        XCTAssertEqual(result2.cols, 4)
    }

    // MARK: - Matrix Transpose Tests

    func testMatrixTranspose() {
        let matrix: [[Float]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let result = MathOps.matrixTranspose(matrix)
        let expected: [[Float]] = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        assertEqual(result, expected)

        let matrix2: [[Float]] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        let result2 = MathOps.matrixTranspose(matrix2)
        let expected2: [[Float]] = [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
        assertEqual(result2, expected2)

        let matrix3: [[Float]] = [[1.0, 2.0], [3.0, 4.0]]
        let result3 = MathOps.matrixTranspose(matrix3)
        let expected3: [[Float]] = [[1.0, 3.0], [2.0, 4.0]]
        assertEqual(result3, expected3)
    }

    func testMatrixTransposeEmpty() {
        let emptyMatrix: [[Float]] = []
        let result = MathOps.matrixTranspose(emptyMatrix)
        XCTAssertTrue(result.isEmpty)

        let emptyRowMatrix: [[Float]] = [[]]
        let result2 = MathOps.matrixTranspose(emptyRowMatrix)
        XCTAssertTrue(result2.isEmpty)
    }

    // MARK: - Flatten/Unflatten Tests

    func testFlatten() {
        let matrix: [[Float]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let result = MathOps.flatten(matrix)

        let expectedFlat: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assertEqual(result.flatMatrix, expectedFlat)
        XCTAssertEqual(result.rows, 2)
        XCTAssertEqual(result.cols, 3)
    }

    func testUnflatten() {
        let flatMatrix: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let result = MathOps.unflatten(flatMatrix, 2, 3)
        let expected: [[Float]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        assertEqual(result, expected)

        let result2 = MathOps.unflatten(flatMatrix, 3, 2)
        let expected2: [[Float]] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        assertEqual(result2, expected2)
    }

    func testFlattenUnflattenRoundTrip() {
        let original: [[Float]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        let (flat, rows, cols) = MathOps.flatten(original)
        let unflattened = MathOps.unflatten(flat, rows, cols)
        assertEqual(unflattened, original)
    }

    // MARK: - LogSumExp Tests

    func testLogSumExp() {
        let values1: [Float] = [1.0, 2.0, 3.0]
        let result1 = MathOps.logSumExp(values1)
        let expected1: Float = log(exp(1.0) + exp(2.0) + exp(3.0))
        assertEqual(result1, expected1)

        let values2: [Float] = [100.0, 101.0, 102.0]
        let result2 = MathOps.logSumExp(values2)
        XCTAssertTrue(result2.isFinite)

        let values3: [Float] = [5.0]
        let result3 = MathOps.logSumExp(values3)
        assertEqual(result3, 5.0)

        let values4: [Float] = []
        let result4 = MathOps.logSumExp(values4)
        XCTAssertTrue(result4.isInfinite && result4 < 0)
    }

    // MARK: - Softmax Tests

    func testSoftmax() {
        let matrix: [[Float]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let result = MathOps.softmax(matrix)

        for row in result {
            let sum = row.reduce(0, +)
            assertEqual(sum, 1.0)
        }

        for row in result {
            for value in row {
                XCTAssertGreaterThan(value, 0.0)
            }
        }

        let singleRow: [[Float]] = [[1.0, 2.0, 3.0]]
        let result2 = MathOps.softmax(singleRow)
        let sum2 = result2[0].reduce(0, +)
        assertEqual(sum2, 1.0)

        let largeValues: [[Float]] = [[100.0, 101.0, 102.0]]
        let result3 = MathOps.softmax(largeValues)
        let sum3 = result3[0].reduce(0, +)
        assertEqual(sum3, 1.0)
        XCTAssertTrue(result3[0].allSatisfy { $0.isFinite })
    }

    // MARK: - Cosine Distance Tests

    func testCosineDistance() {
        let vector1: [Float] = [1.0, 2.0, 3.0]
        let vector2: [Float] = [1.0, 2.0, 3.0]
        let result1 = MathOps.cosineDistance(vector1, vector2)
        assertEqual(result1, 0.0)

        let vector3: [Float] = [1.0, 0.0, 0.0]
        let vector4: [Float] = [0.0, 1.0, 0.0]
        let result2 = MathOps.cosineDistance(vector3, vector4)
        assertEqual(result2, 1.0)

        let vector5: [Float] = [1.0, 0.0, 0.0]
        let vector6: [Float] = [-1.0, 0.0, 0.0]
        let result3 = MathOps.cosineDistance(vector5, vector6)
        assertEqual(result3, 2.0)

        let vector7: [Float] = [1.0, 2.0]
        let vector8: [Float] = [1.0, 2.0, 3.0]
        let result4 = MathOps.cosineDistance(vector7, vector8)
        assertEqual(result4, 1.0)

        let vector9: [Float] = []
        let vector10: [Float] = []
        let result5 = MathOps.cosineDistance(vector9, vector10)
        assertEqual(result5, 1.0)

        let vector11: [Float] = [0.0, 0.0, 0.0]
        let vector12: [Float] = [1.0, 2.0, 3.0]
        let result6 = MathOps.cosineDistance(vector11, vector12)
        assertEqual(result6, 1.0)
    }

    func testCosineDistanceMatrix() {
        let embeddings: [[Float]] = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        let centroids: [[Float]] = [[1.0, 0.0], [0.0, 1.0]]

        let result = MathOps.cosineDistanceMatrix(embeddings: embeddings, centroids: centroids)

        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(result[0].count, 2)

        assertEqual(result[0][0], 0.0)
        assertEqual(result[0][1], 1.0)

        assertEqual(result[1][0], 1.0)
        assertEqual(result[1][1], 0.0)

        let expectedDist1: Float = 1.0 - 1.0 / sqrt(2.0)
        assertEqual(result[2][0], expectedDist1, accuracy: 1e-6)
        let expectedDist2: Float = 1.0 - 1.0 / sqrt(2.0)
        assertEqual(result[2][1], expectedDist2, accuracy: 1e-6)
    }

    func testCosineDistanceMatrixEmpty() {
        let emptyResult = MathOps.cosineDistanceMatrix(embeddings: [], centroids: [[1.0, 2.0]])
        XCTAssertTrue(emptyResult.isEmpty)

        let emptyResult2 = MathOps.cosineDistanceMatrix(embeddings: [[1.0, 2.0]], centroids: [])
        XCTAssertTrue(emptyResult2.isEmpty)
    }

    // MARK: - Argmax Tests

    func testArgmaxAxis0() {
        let matrix: [[Float]] = [
            [1.0, 5.0, 3.0, 2.0],
            [4.0, 2.0, 6.0, 1.0],
            [2.0, 3.0, 1.0, 4.0]
        ]
        let result = MathOps.argmax(matrix, axis: 0)
        let expected = [1, 0, 1, 2]
        XCTAssertEqual(result, expected)

        let singleRow: [[Float]] = [[1.0, 5.0, 3.0, 2.0]]
        let result2 = MathOps.argmax(singleRow, axis: 0)
        let expected2 = [0, 0, 0, 0]
        XCTAssertEqual(result2, expected2)
    }

    func testArgmaxAxis1() {
        let matrix: [[Float]] = [
            [1.0, 5.0, 3.0, 2.0],
            [4.0, 2.0, 6.0, 1.0],
            [2.0, 3.0, 1.0, 4.0]
        ]
        let result = MathOps.argmax(matrix, axis: 1)
        let expected = [1, 2, 3]
        XCTAssertEqual(result, expected)

        let singleCol: [[Float]] = [[1.0], [5.0], [3.0]]
        let result2 = MathOps.argmax(singleCol, axis: 1)
        let expected2 = [0, 0, 0]
        XCTAssertEqual(result2, expected2)
    }

    func testArgmaxEmpty() {
        let emptyMatrix: [[Float]] = []
        let result = MathOps.argmax(emptyMatrix, axis: 0)
        XCTAssertTrue(result.isEmpty)

        let emptyRowMatrix: [[Float]] = [[]]
        let result2 = MathOps.argmax(emptyRowMatrix, axis: 1)
        XCTAssertTrue(result2.isEmpty)
    }

    func testArgmaxTies() {
        let matrix: [[Float]] = [
            [1.0, 5.0, 5.0, 2.0],
            [4.0, 2.0, 6.0, 6.0]
        ]
        let result0 = MathOps.argmax(matrix, axis: 0)
        let expected0 = [1, 0, 1, 1]
        XCTAssertEqual(result0, expected0)

        let result1 = MathOps.argmax(matrix, axis: 1)
        let expected1 = [1, 2]
        XCTAssertEqual(result1, expected1)
    }

    // MARK: - Edge Cases

    func testEdgeCases() {
        let smallValues: [Float] = [Float.leastNormalMagnitude, Float.leastNormalMagnitude]
        let logSumResult = MathOps.logSumExp(smallValues)
        XCTAssertTrue(logSumResult.isFinite)

        let largeValues: [Float] = [Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude]
        let logSumResult2 = MathOps.logSumExp(largeValues)
        XCTAssertTrue(logSumResult2.isFinite)

        let tinyVector1: [Float] = [Float.leastNormalMagnitude, 0.0]
        let tinyVector2: [Float] = [0.0, Float.leastNormalMagnitude]
        let cosineResult = MathOps.cosineDistance(tinyVector1, tinyVector2)
        XCTAssertTrue(cosineResult.isFinite)
        XCTAssertGreaterThanOrEqual(cosineResult, 0.0)
        XCTAssertLessThanOrEqual(cosineResult, 2.0)
    }

    // MARK: - Performance Tests

    func testPerformanceMatrixMultiply() {
        let size = 100
        let matrixA: [[Float]] = (0..<size).map { i in
            (0..<size).map { j in Float(i * size + j) }
        }
        let matrixB: [[Float]] = (0..<size).map { i in
            (0..<size).map { j in Float(i * size + j) }
        }

        measure {
            _ = MathOps.matrixMultiply(matrixA, matrixB)
        }
    }

    func testPerformanceCosineDistanceMatrix() {
        let numEmbeddings = 100
        let embeddingDim = 50
        let numCentroids = 20

        let embeddings: [[Float]] = (0..<numEmbeddings).map { _ in
            (0..<embeddingDim).map { _ in Float.random(in: -1...1) }
        }
        let centroids: [[Float]] = (0..<numCentroids).map { _ in
            (0..<embeddingDim).map { _ in Float.random(in: -1...1) }
        }

        measure {
            _ = MathOps.cosineDistanceMatrix(embeddings: embeddings, centroids: centroids)
        }
    }
}
