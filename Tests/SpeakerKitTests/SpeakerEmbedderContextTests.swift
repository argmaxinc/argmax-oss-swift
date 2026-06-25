//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import XCTest
import CoreML
@testable import SpeakerKit

final class SpeakerEmbedderContextTests: XCTestCase {

    // MARK: - Helpers

    private func makeArray(shape: [Int]) throws -> MLMultiArray {
        try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
    }

    private func makeContext(
        windowsCount: Int = 41,
        speakersCount: Int = 3,
        framesPerWindow: Int = 100,
        secondsPerWindow: Float = 10.0,
        chunkStride: Int = 30,
        waveformLength: Float = 30.0
    ) throws -> SpeakerEmbedderContext {
        let speakerActivity = try makeArray(shape: [windowsCount, speakersCount])
        let speakerIds = try makeArray(shape: [windowsCount, framesPerWindow, speakersCount])
        let overlappedSpeakerActivity = try makeArray(shape: [windowsCount, framesPerWindow])
        return SpeakerEmbedderContext(
            speakerActivity: speakerActivity,
            speakerIds: speakerIds,
            overlappedSpeakerActivity: overlappedSpeakerActivity,
            windowsCount: windowsCount,
            chunkStride: chunkStride,
            secondsPerWindow: secondsPerWindow,
            waveformLength: waveformLength
        )
    }

    // MARK: - secondsPerStride

    func testSecondsPerStrideSubSecond() throws {
        // 41 windows, 30s chunk, 10s window → (30 - 10) / 40 = 0.5s
        let ctx = try makeContext(windowsCount: 41, secondsPerWindow: 10.0)
        XCTAssertEqual(ctx.secondsPerStride, 0.5, accuracy: 1e-5)
    }

    func testSecondsPerStrideIntegerResult() throws {
        // 5 windows → (30 - 10) / 4 = 5.0
        let ctx = try makeContext(windowsCount: 5, secondsPerWindow: 10.0)
        XCTAssertEqual(ctx.secondsPerStride, 5.0, accuracy: 1e-5)
    }

    func testSecondsPerStrideIsZeroForSingleWindow() throws {
        let ctx = try makeContext(windowsCount: 1)
        XCTAssertEqual(ctx.secondsPerStride, 0.0)
    }

    // MARK: - windowIndex math

    func testWindowIndicesAreDistinctAcrossWindows() throws {
        let ctx = try makeContext(windowsCount: 41, secondsPerWindow: 10.0, chunkStride: 30)
        let stride = ctx.secondsPerStride
        XCTAssertGreaterThan(stride, 0)

        let chunkOffset = ctx.chunkOffset(for: 0)
        let indices = (0..<ctx.windowsCount).map {
            chunkOffset + Int(round(Float($0) * stride))
        }

        XCTAssertGreaterThan(indices.last!, indices.first!, "last window must have a higher index than first")
        XCTAssert(indices.allSatisfy { $0 >= 0 })
    }

    func testWindowIndexConsistencyAcrossChunks() throws {
        // chunkOffset + round(windowIdx * stride) must equal round(chunkIndex * secondsPerChunk + windowIdx * stride)
        let secondsPerChunk: Float = 30.0
        let ctx = try makeContext(windowsCount: 41, secondsPerWindow: 10.0, chunkStride: 30)
        let stride = ctx.secondsPerStride

        for chunkIndex in 0..<3 {
            let chunkOffset = ctx.chunkOffset(for: chunkIndex)
            let windowStartSeconds = Float(chunkIndex) * secondsPerChunk
            for windowIdx in 0..<ctx.windowsCount {
                let expected = Int(round(windowStartSeconds + Float(windowIdx) * stride))
                let actual = chunkOffset + Int(round(Float(windowIdx) * stride))
                XCTAssertEqual(actual, expected,
                               "chunk \(chunkIndex), window \(windowIdx): actual=\(actual) expected=\(expected)")
            }
        }
    }

    // MARK: - chunkOffset

    func testChunkOffsetIsLinearInChunkStride() throws {
        let ctx = try makeContext(chunkStride: 25)
        XCTAssertEqual(ctx.chunkOffset(for: 0), 0)
        XCTAssertEqual(ctx.chunkOffset(for: 1), 25)
        XCTAssertEqual(ctx.chunkOffset(for: 3), 75)
    }

    func testChunkOffsetRealisticValues() throws {
        let ctx = try makeContext(chunkStride: 30)
        XCTAssertEqual(ctx.chunkOffset(for: 0), 0)
        XCTAssertEqual(ctx.chunkOffset(for: 1), 30)
        XCTAssertEqual(ctx.chunkOffset(for: 2), 60)
    }

    // MARK: - chunkIndices

    func testChunkIndicesCountMatchesWindowsCount() throws {
        let ctx = try makeContext(windowsCount: 41, framesPerWindow: 100, secondsPerWindow: 10.0)
        XCTAssertEqual(ctx.chunkIndices.count, ctx.windowsCount)
    }

    func testChunkIndicesEachWindowHasCorrectFrameCount() throws {
        let framesPerWindow = 100
        let ctx = try makeContext(windowsCount: 41, framesPerWindow: framesPerWindow, secondsPerWindow: 10.0)
        for (i, window) in ctx.chunkIndices.enumerated() {
            XCTAssertEqual(window.count, framesPerWindow, "window \(i) has wrong frame count")
        }
    }

    func testChunkIndicesFirstWindowStartsAtZero() throws {
        let ctx = try makeContext(windowsCount: 41, framesPerWindow: 100, secondsPerWindow: 10.0)
        XCTAssertEqual(ctx.chunkIndices[0].first, 0)
    }

    func testChunkIndicesWindowsAdvanceByStrideInFrames() throws {
        // framesPerSecond = 100 / 10.0 = 10; strideInFrames = 0.5 * 10 = 5
        let ctx = try makeContext(windowsCount: 41, framesPerWindow: 100, secondsPerWindow: 10.0)
        let indices = ctx.chunkIndices
        let expectedStrideInFrames = 5
        XCTAssertEqual(indices[1].first! - indices[0].first!, expectedStrideInFrames)
        XCTAssertEqual(indices[2].first! - indices[1].first!, expectedStrideInFrames)
    }

    // MARK: - bounded

    func testBoundedRejectsNegativeWindowIdx() throws {
        let ctx = try makeContext(windowsCount: 41, waveformLength: 30.0)
        XCTAssertFalse(ctx.bounded(windowIdx: -1))
    }

    func testBoundedRejectsWindowIdxAtOrBeyondWindowsCount() throws {
        let ctx = try makeContext(windowsCount: 41, waveformLength: 30.0)
        XCTAssertFalse(ctx.bounded(windowIdx: 41))
        XCTAssertFalse(ctx.bounded(windowIdx: 100))
    }

    func testBoundedAlwaysAcceptsFirstWindow() throws {
        let ctx = try makeContext(windowsCount: 41, waveformLength: 1.0)
        XCTAssertTrue(ctx.bounded(windowIdx: 0))
    }

    func testBoundedAcceptsLastWindowForFullChunk() throws {
        // window 40: endBoundary = 0.5*40 + 10 = 30.0; 30.0 < (30.0 + 0.5) = 30.5 → true
        let ctx = try makeContext(windowsCount: 41, secondsPerWindow: 10.0, waveformLength: 30.0)
        XCTAssertTrue(ctx.bounded(windowIdx: 40))
    }

    func testBoundedRejectsWindowsBeyondShortWaveform() throws {
        // waveformLength = 5s; stride = 0.5s; window 1: endBoundary = 0.5 + 10 = 10.5 > 5.5 → false
        let ctx = try makeContext(windowsCount: 41, secondsPerWindow: 10.0, waveformLength: 5.0)
        XCTAssertFalse(ctx.bounded(windowIdx: 1))
    }
}
