//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML

// MARK: - MLTensor Conversions

#if canImport(CoreML.MLState)
@available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *)
public extension MLTensor {

    // MARK: Async (safe for cooperative thread pool)

    func toIntArray() async -> [Int] {
        await shapedArray(of: Int32.self).scalars.map { Int($0) }
    }

    func toFloatArray() async -> [Float] {
        switch scalarType {
        case is Float32.Type:
            return await shapedArray(of: Float32.self).scalars.map { Float($0) }
        case is FloatType.Type:
            return await shapedArray(of: FloatType.self).scalars.map { Float($0) }
        case is Float.Type:
            return await shapedArray(of: Float.self).scalars
        case is Int32.Type:
            return await shapedArray(of: Int32.self).scalars.map { Float($0) }
        default:
            fatalError("Unsupported scalar type: \(scalarType)")
        }
    }

    func toMLMultiArray() async -> MLMultiArray {
        switch scalarType {
        case is Float32.Type:
            return MLMultiArray(await shapedArray(of: Float32.self))
        case is FloatType.Type:
            return MLMultiArray(await shapedArray(of: FloatType.self))
        case is Float.Type:
            return MLMultiArray(await shapedArray(of: Float.self))
        case is Int32.Type:
            return MLMultiArray(await shapedArray(of: Int32.self))
        default:
            fatalError("Unsupported scalar type: \(scalarType)")
        }
    }
}
#endif
