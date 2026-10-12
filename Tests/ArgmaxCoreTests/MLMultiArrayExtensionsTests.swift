//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import CoreVideo
import XCTest

@testable import ArgmaxCore

/// Tests for the `MLMultiArray` helpers in `MLMultiArrayExtensions`.
///
/// The lock tests cover the Core ML warning "Pixel buffer backing MLMultiArray is locked until
/// this MultiArray is deallocated (or storage swapped) due to usage of deprecated dataPointer or
/// bytes properties". `CVPixelBufferGetBaseAddress` returns a non-nil address only while the
/// pixel buffer is locked, so it reads the lock state without touching the array itself.
final class MLMultiArrayExtensionsTests: XCTestCase {
    private func assertBackingPixelBufferIsUnlocked(
        _ array: MLMultiArray,
        _ message: String,
        line: UInt = #line
    ) throws {
        let pixelBuffer = try XCTUnwrap(array.pixelBuffer, "Expected an IOSurface-backed array", line: line)
        XCTAssertNil(CVPixelBufferGetBaseAddress(pixelBuffer), message, line: line)
    }

    // MARK: - Initial value

    func testInitialValueFillsEveryElement() throws {
        let float16 = try MLMultiArray(shape: [1, 4], dataType: .float16, initialValue: FloatType(-10000))
        for index in 0..<float16.count {
            XCTAssertEqual(float16[index].floatValue, -10000, accuracy: 1)
        }

        let float32 = try MLMultiArray(shape: [2, 3], dataType: .float32, initialValue: Float(1.5))
        for index in 0..<float32.count {
            XCTAssertEqual(float32[index].floatValue, 1.5, accuracy: .ulpOfOne)
        }

        let int32 = try MLMultiArray(shape: [5], dataType: .int32, initialValue: Int32(7))
        for index in 0..<int32.count {
            XCTAssertEqual(int32[index].int32Value, 7)
        }

        let double = try MLMultiArray(shape: [3], dataType: .double, initialValue: Double(2.25))
        for index in 0..<double.count {
            XCTAssertEqual(double[index].doubleValue, 2.25, accuracy: .ulpOfOne)
        }
    }

    func testInitialValueLeavesPixelBufferUnlocked() throws {
        let array = try MLMultiArray(shape: [1, 8], dataType: .float16, initialValue: FloatType(0))
        try assertBackingPixelBufferIsUnlocked(
            array,
            "Writing the initial value must not leave the pixel buffer locked"
        )
    }

    // MARK: - fill

    func testFillWritesTargetIndexesOnly() throws {
        let array = try MLMultiArray(shape: [1, 1, 6], dataType: .float16, initialValue: FloatType(0))
        array.fill(indexes: [[0, 0, 1], [0, 0, 4]], with: -FloatType.infinity)

        XCTAssertEqual(array[1].floatValue, -.infinity)
        XCTAssertEqual(array[4].floatValue, -.infinity)
        for index in [0, 2, 3, 5] {
            XCTAssertEqual(array[index].floatValue, 0)
        }
    }

    func testFillSkipsOutOfBoundsIndexes() throws {
        let array = try MLMultiArray(shape: [1, 1, 3], dataType: .float16, initialValue: FloatType(0))
        array.fill(indexes: [[0, 0, 3], [0, 0, -1], [0, 0], [0, 0, 2]], with: -FloatType.infinity)

        XCTAssertEqual(array[0].floatValue, 0)
        XCTAssertEqual(array[1].floatValue, 0)
        XCTAssertEqual(array[2].floatValue, -.infinity)
    }

    func testFillLeavesPixelBufferUnlocked() throws {
        let array = try MLMultiArray(shape: [1, 1, 6], dataType: .float16, initialValue: FloatType(0))
        array.fill(indexes: [[0, 0, 2]], with: -FloatType.infinity)
        try assertBackingPixelBufferIsUnlocked(
            array,
            "fill(indexes:with:) must not leave the pixel buffer locked"
        )
    }

    func testFillLastDimensionLeavesPixelBufferUnlocked() throws {
        let array = try MLMultiArray(shape: [1, 1, 6], dataType: .float16, initialValue: FloatType(0))
        array.fillLastDimension(indexes: 0..<3, with: -FloatType.infinity)
        try assertBackingPixelBufferIsUnlocked(
            array,
            "fillLastDimension(indexes:with:) must not leave the pixel buffer locked"
        )
    }

    // MARK: - from

    func testFromIntArray() throws {
        let array = try MLMultiArray.from([4, 5, 6], dims: 3)
        XCTAssertEqual(array.shape.map { $0.intValue }, [1, 1, 3])
        XCTAssertEqual((0..<array.count).map { array[$0].int32Value }, [4, 5, 6])
    }
}
