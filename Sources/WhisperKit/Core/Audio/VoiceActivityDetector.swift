//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

/// A base class for Voice Activity Detection (VAD), used to identify and separate segments of audio that contain human speech from those that do not.
/// Subclasses must implement the `voiceActivity(in:)` method to provide specific voice activity detection functionality.
open class VoiceActivityDetector {
    /// The sample rate of the audio signal, in samples per second.
    public let sampleRate: Int

    /// The length of each frame in samples.
    public let frameLengthSamples: Int

    /// The number of samples overlapping between consecutive frames.
    public let frameOverlapSamples: Int

    /// Initializes a new `VoiceActivityDetector` instance with the specified parameters.
    /// - Parameters:
    ///   - sampleRate: The sample rate of the audio signal in samples per second. Defaults to 16000.
    ///   - frameLengthSamples: The length of each frame in samples.
    ///   - frameOverlapSamples: The number of samples overlapping between consecutive frames. Defaults to 0.
    /// - Note: Subclasses should override the `voiceActivity(in:)` method to provide specific VAD functionality.
    public init(
        sampleRate: Int = 16000,
        frameLengthSamples: Int,
        frameOverlapSamples: Int = 0
    ) {
        self.sampleRate = sampleRate
        self.frameLengthSamples = frameLengthSamples
        self.frameOverlapSamples = frameOverlapSamples
    }

    /// Analyzes the provided audio waveform to determine which segments contain voice activity.
    /// - Parameter waveform: An array of `Float` values representing the audio waveform.
    /// - Returns: An array of `Bool` values where `true` indicates the presence of voice activity and `false` indicates silence.
    open func voiceActivity(in waveform: [Float]) -> [Bool] {
        fatalError("`voiceActivity` must be implemented by subclass")
    }

    /// Analyzes the provided audio waveform to determine which segments contain voice activity.
    /// - Parameter waveform: An array of `Float` values representing the audio waveform.
    /// - Returns: An array of `Bool` values where `true` indicates the presence of voice activity and `false` indicates silence.
    /// - Throws: An error if voice activity detection fails.
    open func voiceActivityAsync(in waveform: [Float]) async throws -> [Bool] {
        return voiceActivity(in: waveform)
    }

    /// Calculates and returns a list of active audio chunks, each represented by a start and end index.
    /// - Parameter waveform: An array of `Float` values representing the audio waveform.
    /// - Returns: An array of tuples where each tuple contains the start and end indices of an active audio chunk.
    public func calculateActiveChunks(in waveform: [Float]) -> [(startIndex: Int, endIndex: Int)] {
        let vad: [Bool] = voiceActivity(in: waveform)
        var result = [(startIndex: Int, endIndex: Int)]()

        // Temporary variables to hold the start of the current non-silent segment
        var currentStartIndex: Int?

        for (index, vadChunk) in vad.enumerated() {
            if vadChunk {
                let chunkStart = index * frameLengthSamples
                let chunkEnd = min(chunkStart + frameLengthSamples, waveform.count)

                if currentStartIndex != nil {
                    // If we already have a starting point, just update the end point in the last added segment
                    result[result.count - 1].endIndex = chunkEnd
                } else {
                    // If there is no current start, this is a new segment
                    currentStartIndex = chunkStart
                    result.append((startIndex: chunkStart, endIndex: chunkEnd))
                }
            } else {
                // Reset currentStartIndex when encountering a silent chunk
                currentStartIndex = nil
            }
        }

        return result
    }

    /// Converts a voice activity index to the corresponding audio sample index.
    /// - Parameter index: The voice activity index to convert.
    /// - Returns: The corresponding audio sample index.
    public func voiceActivityIndexToAudioSampleIndex(_ index: Int) -> Int {
        return index * frameLengthSamples
    }

    public func voiceActivityIndexToSeconds(_ index: Int) -> Float {
        return Float(voiceActivityIndexToAudioSampleIndex(index)) / Float(sampleRate)
    }

    /// Identifies the longest continuous period of silence within the provided voice activity detection results.
    /// - Parameter vadResult: An array of `Bool` values representing voice activity detection results.
    /// - Returns: A tuple containing the start and end indices of the longest silence period, or `nil` if no silence is found.
    public func findLongestSilence(in vadResult: [Bool]) -> (startIndex: Int, endIndex: Int)? {
        var longestStartIndex: Int?
        var longestEndIndex: Int?
        var longestCount = 0
        var index = 0
        while index < vadResult.count {
            let value = vadResult[index]
            if value {
                // found non-silence, skip
                index += 1
            } else {
                // found beginning of silence, find the end
                var endIndex = index
                while endIndex < vadResult.count, !vadResult[endIndex] {
                    endIndex += 1
                }
                let count = endIndex - index
                if count > longestCount {
                    longestCount = count
                    longestStartIndex = index
                    longestEndIndex = endIndex
                }
                index = endIndex
            }
        }
        if let longestStartIndex, let longestEndIndex {
            return (startIndex: longestStartIndex, endIndex: longestEndIndex)
        } else {
            return nil
        }
    }

    // MARK: - Utility

    func voiceActivityClipTimestamps(in waveform: [Float]) -> [Float] {
        let nonSilentChunks = calculateActiveChunks(in: waveform)
        var clipTimestamps = [Float]()

        for chunk in nonSilentChunks {
            let startTimestamp = Float(chunk.startIndex) / Float(sampleRate)
            let endTimestamp = Float(chunk.endIndex) / Float(sampleRate)

            clipTimestamps.append(contentsOf: [startTimestamp, endTimestamp])
        }

        return clipTimestamps
    }

    func calculateNonSilentSeekClips(in waveform: [Float]) -> [(start: Int, end: Int)] {
        let clipTimestamps = voiceActivityClipTimestamps(in: waveform)
        let options = DecodingOptions(clipTimestamps: clipTimestamps)
        let seekClips = options.prepareSeekClips(contentFrames: waveform.count)
        return seekClips
    }

    func calculateSeekTimestamps(in waveform: [Float]) -> [(startTime: Float, endTime: Float)] {
        let nonSilentChunks = calculateActiveChunks(in: waveform)
        var seekTimestamps = [(startTime: Float, endTime: Float)]()

        for chunk in nonSilentChunks {
            let startTimestamp = Float(chunk.startIndex) / Float(sampleRate)
            let endTimestamp = Float(chunk.endIndex) / Float(sampleRate)

            seekTimestamps.append(contentsOf: [(startTime: startTimestamp, endTime: endTimestamp)])
        }

        return seekTimestamps
    }
}
