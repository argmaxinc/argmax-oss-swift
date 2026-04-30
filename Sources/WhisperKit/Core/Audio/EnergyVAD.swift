//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

/// Voice activity detection based on energy threshold
public final class EnergyVAD: VoiceActivityDetector {
    public let energyThreshold: Float

    /// Initialize a new EnergyVAD instance
    /// - Parameters:
    ///   - sampleRate: Audio sample rate
    ///   - frameLength: Frame length in seconds
    ///   - frameOverlap: frame overlap in seconds, this will include `frameOverlap` length audio into the `frameLength` and is helpful to catch audio that starts exactly at chunk boundaries
    ///   - energyThreshold: minimal energy threshold
    public convenience init(
        sampleRate: Int = WhisperKit.sampleRate,
        frameLength: Float = 0.1,
        frameOverlap: Float = 0.0,
        energyThreshold: Float = 0.02
    ) {
        self.init(
            sampleRate: sampleRate,
            // Compute frame length and overlap in number of samples
            frameLengthSamples: Int(frameLength * Float(sampleRate)),
            frameOverlapSamples: Int(frameOverlap * Float(sampleRate)),
            energyThreshold: energyThreshold
        )
    }

    public required init(
        sampleRate: Int = 16000,
        frameLengthSamples: Int,
        frameOverlapSamples: Int = 0,
        energyThreshold: Float = 0.02
    ) {
        self.energyThreshold = energyThreshold
        super.init(sampleRate: sampleRate, frameLengthSamples: frameLengthSamples, frameOverlapSamples: frameOverlapSamples)
    }

    public override func voiceActivity(in waveform: [Float]) -> [Bool] {
        let chunkRatio = Double(waveform.count) / Double(frameLengthSamples)

        // Round up if uneven, the final chunk will not be a full `frameLengthSamples` long
        let count = Int(chunkRatio.rounded(.up))

        let chunkedVoiceActivity = AudioProcessor.calculateVoiceActivityInChunks(
            of: waveform,
            chunkCount: count,
            frameLengthSamples: frameLengthSamples,
            frameOverlapSamples: frameOverlapSamples,
            energyThreshold: energyThreshold
        )

        return chunkedVoiceActivity
    }
}
