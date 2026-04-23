//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import XCTest
import WhisperKit
@testable import SpeakerKit

/// End-to-end performance baseline for `speakerKit.diarize(...)` on the bundled VADAudio
/// fixture. This file uses only pre-existing public API (`SpeakerKit`,
/// `SpeakerKit.diarizer`, `Diarizer.loadModels`, `diarize(audioArray:)`) so it can be
/// cherry-picked onto `main` to produce a baseline number against this branch.
final class DiarizationPipelinePerformanceTests: XCTestCase {

    private func loadAudio(named name: String, extension ext: String = "wav") throws -> [Float] {
        guard let url = Bundle.module.url(forResource: name, withExtension: ext) else {
            throw XCTSkip("Audio file \(name).\(ext) not found in test bundle")
        }
        let audioBuffer = try AudioProcessor.loadAudio(fromPath: url.path)
        return AudioProcessor.convertBufferToArray(buffer: audioBuffer)
    }

    /// Measures the full diarization pipeline on `VADAudio`.
    /// Models are preloaded and a warmup call is run outside the `measure` block so the
    /// first iteration does not bake in one-off model loading or CoreML compilation time.
    func testPerformance_diarizePipeline_VADAudio() async throws {
        let audioArray = try loadAudio(named: "VADAudio")
        let speakerKit = try await SpeakerKit()

        // force model download + load outside the measure block so the first iteration
        // does not include one-off startup cost.
        try await speakerKit.diarizer.loadModels()

        // one warmup call to settle any first-run caches (CoreML compilation, etc).
        _ = try await speakerKit.diarize(audioArray: audioArray)

        let opts = XCTMeasureOptions()
        opts.iterationCount = 20

        measure(metrics: [XCTClockMetric()], options: opts) {
            let exp = expectation(description: "diarize")
            Task {
                _ = try await speakerKit.diarize(audioArray: audioArray)
                exp.fulfill()
            }
            wait(for: [exp], timeout: 120)
        }
    }
}
