//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

/// Voice activity detection based on energy threshold
public final class EnergyVAD: VoiceActivityDetector {
    public let energyThreshold: Float

    /// Optional level-aware threshold, as a fraction of the waveform's own loud level.
    ///
    /// When set, the threshold for a waveform is `relativeThreshold` times the 90th percentile of its
    /// frame energies, capped at `energyThreshold`. Loud audio keeps the fixed threshold, while quiet
    /// speech (frame RMS well under `energyThreshold`) is no longer read as silence, so the pauses
    /// found in it are real pauses. `nil` keeps the fixed threshold for every level.
    public let relativeThreshold: Float?

    /// The lowest threshold the relative mode will use, so digital silence never reads as voice.
    public static let minimumRelativeEnergyThreshold: Float = 1e-4

    /// Initialize a new EnergyVAD instance
    /// - Parameters:
    ///   - sampleRate: Audio sample rate
    ///   - frameLength: Frame length in seconds
    ///   - frameOverlap: frame overlap in seconds, this will include `frameOverlap` length audio into the `frameLength` and is helpful to catch audio that starts exactly at chunk boundaries
    ///   - energyThreshold: minimal energy threshold
    ///   - relativeThreshold: optional fraction of the waveform's loud level (90th percentile frame energy) to use as the threshold when that is lower than `energyThreshold`
    public convenience init(
        sampleRate: Int = WhisperKit.sampleRate,
        frameLength: Float = 0.1,
        frameOverlap: Float = 0.0,
        energyThreshold: Float = 0.02,
        relativeThreshold: Float? = nil
    ) {
        self.init(
            sampleRate: sampleRate,
            // Compute frame length and overlap in number of samples
            frameLengthSamples: Int(frameLength * Float(sampleRate)),
            frameOverlapSamples: Int(frameOverlap * Float(sampleRate)),
            energyThreshold: energyThreshold,
            relativeThreshold: relativeThreshold
        )
    }

    public required init(
        sampleRate: Int = 16000,
        frameLengthSamples: Int,
        frameOverlapSamples: Int = 0,
        energyThreshold: Float = 0.02,
        relativeThreshold: Float? = nil
    ) {
        self.energyThreshold = energyThreshold
        self.relativeThreshold = relativeThreshold
        super.init(sampleRate: sampleRate, frameLengthSamples: frameLengthSamples, frameOverlapSamples: frameOverlapSamples)
    }

    public override func voiceActivity(in waveform: [Float]) -> [Bool] {
        let chunkRatio = Double(waveform.count) / Double(frameLengthSamples)

        // Round up if uneven, the final chunk will not be a full `frameLengthSamples` long
        let count = Int(chunkRatio.rounded(.up))

        guard let relativeThreshold else {
            return AudioProcessor.calculateVoiceActivityInChunks(
                of: waveform,
                chunkCount: count,
                frameLengthSamples: frameLengthSamples,
                frameOverlapSamples: frameOverlapSamples,
                energyThreshold: energyThreshold
            )
        }

        let energies = AudioProcessor.calculateEnergyInChunks(
            of: waveform,
            chunkCount: count,
            frameLengthSamples: frameLengthSamples,
            frameOverlapSamples: frameOverlapSamples
        )
        let threshold = Self.levelAwareThreshold(
            energies: energies,
            energyThreshold: energyThreshold,
            relativeThreshold: relativeThreshold
        )
        return energies.map { $0 > threshold }
    }

    /// The threshold for a set of frame energies: `relativeThreshold` times their 90th percentile,
    /// capped at `energyThreshold` and floored at `minimumRelativeEnergyThreshold`.
    static func levelAwareThreshold(energies: [Float], energyThreshold: Float, relativeThreshold: Float) -> Float {
        guard !energies.isEmpty else { return energyThreshold }
        let sorted = energies.sorted()
        let loudLevel = sorted[min(sorted.count - 1, Int(Float(sorted.count) * 0.9))]
        return min(energyThreshold, max(minimumRelativeEnergyThreshold, loudLevel * relativeThreshold))
    }
}
