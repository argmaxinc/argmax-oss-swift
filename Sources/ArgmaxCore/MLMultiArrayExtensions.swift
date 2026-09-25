//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML

private extension UnsafeMutableRawBufferPointer {
    /// Writes `value` into the first `count` elements of the buffer.
    ///
    /// Clamps to the buffer's own capacity, which can be larger than `count` when the backing
    /// pixel buffer pads its rows.
    func initializeElements<Element>(repeating value: Element, count: Int) {
        let typed = bindMemory(to: Element.self)
        guard let base = typed.baseAddress else { return }
        base.initialize(repeating: value, count: Swift.min(count, typed.count))
    }
}

// MARK: - MLMultiArray Creation

public extension MLMultiArray {
    /// Creates an MLMultiArray pre-filled with an initial value.
    /// Uses IOSurface-backed storage for float16 arrays.
    ///
    /// Writes go through `withUnsafeMutableBytes` rather than `dataPointer`. Touching
    /// `dataPointer` on a pixel-buffer-backed array locks that pixel buffer for as long as the
    /// array is alive, which Core ML reports as "Pixel buffer backing MLMultiArray is locked
    /// until this MultiArray is deallocated (or storage swapped) due to usage of deprecated
    /// dataPointer or bytes properties".
    convenience init(shape: [NSNumber], dataType: MLMultiArrayDataType, initialValue: Any) throws {
        switch dataType {
            case .float16:
                guard let pixelBuffer = Self.pixelBuffer(for: shape) else {
                    throw MLMultiArrayCreationError.pixelBufferFailed
                }
                self.init(pixelBuffer: pixelBuffer, shape: shape)
            default:
                try self.init(shape: shape, dataType: dataType)
        }

        let elementCount = count
        withUnsafeMutableBytes { buffer, _ in
            switch dataType {
                case .double:
                    if let value = initialValue as? Double {
                        buffer.initializeElements(repeating: value, count: elementCount)
                    }
                case .float32:
                    if let value = initialValue as? Float {
                        buffer.initializeElements(repeating: value, count: elementCount)
                    }
                case .float16:
                    if let value = initialValue as? FloatType {
                        buffer.initializeElements(repeating: value, count: elementCount)
                    }
                case .int32:
                    if let value = initialValue as? Int32 {
                        buffer.initializeElements(repeating: value, count: elementCount)
                    }
                #if compiler(>=6.2)
                case .int8:
                    if #available(macOS 26.0, iOS 26.0, watchOS 26.0, visionOS 26.0, tvOS 26.0, *),
                       let value = initialValue as? Int8 {
                        buffer.initializeElements(repeating: value, count: elementCount)
                    }
                #endif
                @unknown default:
                    break
            }
        }
    }

    /// Creates an MLMultiArray from an [Int] array.
    /// Values are stored in the last dimension (default is dims=1).
    static func from(_ array: [Int], dims: Int = 1) throws -> MLMultiArray {
        var shape = Array(repeating: 1, count: dims)
        shape[shape.count - 1] = array.count
        let output = try MLMultiArray(shape: shape as [NSNumber], dataType: .int32)
        output.withUnsafeMutableBytes { buffer, _ in
            let pointer = buffer.bindMemory(to: Int32.self)
            for (i, item) in array.enumerated() {
                pointer[i] = Int32(item)
            }
        }
        return output
    }
}

// MARK: - MLMultiArray Indexing & Fill

public extension MLMultiArray {
    /// Computes the linear offset from multi-dimensional indices using strides.
    @inline(__always)
    func linearOffset(for index: [Int], strides strideInts: [Int]? = nil) -> Int {
        var linearOffset = 0
        let strideInts = strideInts ?? strides.map { $0.intValue }
        for (dimension, stride) in zip(index, strideInts) {
            linearOffset += dimension * stride
        }
        return linearOffset
    }

    @available(*, deprecated, message: "Use linearOffset(for: [Int], strides:) instead.")
    @inline(__always)
    func linearOffset(for index: [NSNumber], strides strideInts: [Int]? = nil) -> Int {
        linearOffset(for: index.map(\.intValue), strides: strideInts)
    }

    /// Fills a range of indices in the last dimension with a value.
    /// Requires shape [1, 1, n].
    func fillLastDimension(indexes: Range<Int>, with value: FloatType) {
        precondition(shape.count == 3 && shape[0] == 1 && shape[1] == 1, "Must have [1, 1, n] shape")
        withUnsafeMutableBufferPointer(ofType: FloatType.self) { ptr, strides in
            for index in indexes {
                ptr[index * strides[2]] = value
            }
        }
    }

    /// Fills specific multi-dimensional indices with a value.
    func fill<Value>(indexes: [[Int]], with value: Value) {
        let strideInts = strides.map { $0.intValue }
        let shapeInts = shape.map { $0.intValue }
        withUnsafeMutableBytes { buffer, _ in
            let pointer = buffer.bindMemory(to: Value.self)
            for index in indexes {
                guard index.count == shapeInts.count,
                      zip(index, shapeInts).allSatisfy({ $0 >= 0 && $0 < $1 }) else {
                    continue
                }
                let linearOffset = linearOffset(for: index, strides: strideInts)
                pointer[linearOffset] = value
            }
        }
    }

    @available(*, deprecated, message: "Use fill(indexes: [[Int]], with:) instead.")
    func fill<Value>(indexes: [[NSNumber]], with value: Value) {
        fill(indexes: indexes.map { $0.map(\.intValue) }, with: value)
    }
}

// MARK: - IOSurface-backed Pixel Buffer

extension MLMultiArray {
    /// Creates a CVPixelBuffer suitable for float16 IOSurface-backed MLMultiArrays.
    public class func pixelBuffer(for shape: [NSNumber]) -> CVPixelBuffer? {
        guard let width = shape.last?.intValue else { return nil }
        let height = shape[0..<shape.count - 1].reduce(1) { $0 * $1.intValue }

        var pixelBuffer: CVPixelBuffer?
        let createReturn = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_OneComponent16Half,
            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary,
            &pixelBuffer
        )
        guard createReturn == kCVReturnSuccess else { return nil }
        return pixelBuffer
    }
}

// MARK: - Error

/// Error thrown when MLMultiArray creation fails.
@frozen
public enum MLMultiArrayCreationError: Error, LocalizedError {
    case pixelBufferFailed

    public var errorDescription: String? {
        switch self {
        case .pixelBufferFailed:
            return "Failed to create IOSurface-backed pixel buffer for MLMultiArray"
        }
    }
}

