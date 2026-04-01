//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Accelerate
import Foundation

// MARK: - MathOps

enum MathOps {

    // MARK: - Linear Algebra Operations

    /// Matrix multiplication using vDSP: A @ B
    static func matrixMultiply(_ matrixA: [[Float]], _ matrixB: [[Float]]) -> (
        result: [Float], rows: Int, cols: Int
    ) {
        let (flatA, rowsA, colsA) = flatten(matrixA)
        let (flatB, rowsB, colsB) = flatten(matrixB)

        guard colsA == rowsB else {
            return (result: [], rows: 0, cols: 0)
        }

        var result = Array(repeating: Float(0), count: rowsA * colsB)

        vDSP_mmul(
            flatA, 1, flatB, 1, &result, 1, vDSP_Length(rowsA), vDSP_Length(colsB), vDSP_Length(colsA))

        return (result, rowsA, colsB)
    }

    /// Transpose a 2D matrix using vDSP
    static func matrixTranspose(_ matrix: [[Float]]) -> [[Float]] {
        guard !matrix.isEmpty && !matrix[0].isEmpty else { return [] }

        let rows = matrix.count
        let cols = matrix[0].count

        let flatMatrix = matrix.flatMap { $0 }
        var result = Array(repeating: Float(0), count: rows * cols)

        vDSP_mtrans(flatMatrix, 1, &result, 1, vDSP_Length(cols), vDSP_Length(rows))

        return (0..<cols).map { i in
            Array(result[i * rows..<(i + 1) * rows])
        }
    }

    /// Matrix flattening
    static func flatten(_ matrix: [[Float]]) -> (flatMatrix: [Float], rows: Int, cols: Int) {
        guard !matrix.isEmpty else { return ([], 0, 0) }
        return (matrix.flatMap { $0 }, matrix.count, matrix[0].count)
    }

    /// Matrix unflattening
    static func unflatten(_ flatMatrix: [Float], _ rows: Int, _ cols: Int) -> [[Float]] {
        return (0..<rows).map { i in
            Array(flatMatrix[i * cols..<(i + 1) * cols])
        }
    }

    // MARK: - Unary Operations

    /// Compute log-sum-exp for numerical stability
    static func logSumExp(_ values: [Float]) -> Float {
        let maxVal = values.max() ?? 0
        let expVals = values.map { exp($0 - maxVal) }
        let sumExp = expVals.reduce(0, +)
        return maxVal + log(sumExp)
    }

    /// Apply softmax function to a 2D matrix
    static func softmax(_ matrix: [[Float]]) -> [[Float]] {
        return matrix.map { row in
            let maxVal = row.max() ?? 0
            let expVals = row.map { exp($0 - maxVal) }
            let sumExp = expVals.reduce(0, +)
            return expVals.map { $0 / sumExp }
        }
    }

    // MARK: - Distance Operations

    /// Calculate cosine distance between two vectors
    /// Cosine distance = 1 - cosine similarity
    static func cosineDistance(_ vector1: [Float], _ vector2: [Float]) -> Float {
        guard vector1.count == vector2.count, !vector1.isEmpty else {
            return 1.0
        }

        let length = vDSP_Length(vector1.count)

        var dotProduct: Float = 0.0
        vDSP_dotpr(vector1, 1, vector2, 1, &dotProduct, length)

        var magnitude1: Float = 0.0
        var magnitude2: Float = 0.0
        vDSP_svesq(vector1, 1, &magnitude1, length)
        vDSP_svesq(vector2, 1, &magnitude2, length)

        magnitude1 = sqrt(magnitude1)
        magnitude2 = sqrt(magnitude2)

        guard magnitude1 > 0 && magnitude2 > 0 else {
            return 1.0
        }

        let cosineSimilarity = dotProduct / (magnitude1 * magnitude2)

        return max(0.0, min(2.0, 1.0 - cosineSimilarity))
    }

    /// Calculate cosine distance matrix between embeddings and centroids
    static func cosineDistanceMatrix(embeddings: [[Float]], centroids: [[Float]]) -> [[Float]] {
        guard !embeddings.isEmpty, !centroids.isEmpty else {
            return []
        }

        let numEmbeddings = embeddings.count
        let numCentroids = centroids.count

        var distanceMatrix = Array(
            repeating: Array(repeating: Float(0), count: numCentroids), count: numEmbeddings)

        for (embIdx, embedding) in embeddings.enumerated() {
            for (centIdx, centroid) in centroids.enumerated() {
                distanceMatrix[embIdx][centIdx] = cosineDistance(embedding, centroid)
            }
        }

        return distanceMatrix
    }

    /// Find the index of maximum value along the specified axis
    static func argmax(_ matrix: [[Float]], axis: Int) -> [Int] {
        guard !matrix.isEmpty, !matrix[0].isEmpty else {
            return []
        }
        
        if axis == 0 {
            let numCols = matrix[0].count
            var result = Array(repeating: 0, count: numCols)

            for col in 0..<numCols {
                var maxValue = matrix[0][col]
                var maxIndex = 0

                for row in 1..<matrix.count where matrix[row][col] > maxValue {
                    maxValue = matrix[row][col]
                    maxIndex = row
                }
                result[col] = maxIndex
            }

            return result
        } else {
            return matrix.map { row in
                var maxValue = row[0]
                var maxIndex = 0

                for (index, value) in row.enumerated().dropFirst() where value > maxValue {
                    maxValue = value
                    maxIndex = index
                }

                return maxIndex
            }
        }
    }
}
