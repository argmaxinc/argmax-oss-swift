//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import AVFoundation
import Foundation

/// Responsible for chunking audio into smaller pieces
public protocol AudioChunking {
    func chunkAll(audioArray: [Float], maxChunkLength: Int, decodeOptions: DecodingOptions?) async throws -> [AudioChunk]
}

public extension AudioChunking {
    func updateSeekOffsetsForResults(
        chunkedResults: [Result<[TranscriptionResult], Swift.Error>],
        audioChunks: [AudioChunk]
    ) -> [TranscriptionResult] {
        var updatedTranscriptionResults = [TranscriptionResult]()
        for (index, chunkedResult) in chunkedResults.enumerated() {
            switch chunkedResult {
                case let .success(results):
                    let seekTime = Float(audioChunks[index].seekOffsetIndex) / Float(WhisperKit.sampleRate)
                    for result in results {
                        var updatedSegments = [TranscriptionSegment]()
                        for segment in result.segments {
                            let updatedSegment = TranscriptionUtilities.updateSegmentTimings(segment: segment, seekTime: seekTime)
                            updatedSegments.append(updatedSegment)
                        }
                        let updatedResult = result
                        updatedResult.seekTime = seekTime
                        updatedResult.segments = updatedSegments
                        updatedTranscriptionResults.append(updatedResult)
                    }
                case let .failure(error):
                    Logging.debug("Error transcribing chunk \(index): \(error)")
            }
        }
        return updatedTranscriptionResults
    }
}

/// A audio chunker that splits audio into smaller pieces based on voice activity detection
open class VADAudioChunker: AudioChunking {
    /// prevent hallucinations at the end of the clip by stopping up to 1.0s early
    private let windowPadding: Int
    private let vad: VoiceActivityDetector

    public init(windowPadding: Int = 16000, vad: VoiceActivityDetector? = nil) {
        self.windowPadding = windowPadding
        self.vad = vad ?? EnergyVAD()
    }

    private func splitOnMiddleOfLongestSilence(audioArray: [Float], startIndex: Int, endIndex: Int) -> Int {
        // NOTE: we want to check just the 2nd part for the silence to attempt to get closest to a max length chunk
        let audioMidIndex = startIndex + (endIndex - startIndex) / 2
        let vadAudioSlice = Array(audioArray[audioMidIndex..<endIndex])
        let voiceActivity = vad.voiceActivity(in: vadAudioSlice)
        if let silence = vad.findLongestSilence(in: voiceActivity) {
            // if silence is detected we take the middle point of the silent chunk
            let silenceMidIndex = silence.startIndex + (silence.endIndex - silence.startIndex) / 2
            return audioMidIndex + vad.voiceActivityIndexToAudioSampleIndex(silenceMidIndex)
        }
        return endIndex
    }

    public func chunkAll(audioArray: [Float], maxChunkLength: Int, decodeOptions: DecodingOptions?) async throws -> [AudioChunk] {
        // If the audio array length is less than or equal to maxLength, return it as a single chunk
        if audioArray.count <= maxChunkLength {
            return [AudioChunk(seekOffsetIndex: 0, audioSamples: audioArray)]
        }

        // First create chunks from seek clips
        let options = decodeOptions ?? DecodingOptions()
        let seekClips = options.prepareSeekClips(contentFrames: audioArray.count)

        var chunkedAudio = [AudioChunk]()
        for (seekClipStart, seekClipEnd) in seekClips {
            // Loop through the current clip until we reach the end
            // Typically this will be the full audio file, unless seek points are explicitly provided
            var startIndex = seekClipStart
            while startIndex < seekClipEnd - windowPadding {
                guard startIndex >= 0 && startIndex < audioArray.count else {
                    throw WhisperError.audioProcessingFailed("startIndex is outside the buffer size")
                }

                // Make sure we still need chunking for this seek clip, otherwise use the original seek clip end
                var endIndex = seekClipEnd
                if startIndex + maxChunkLength < endIndex {
                    // Adjust the end index based on VAD
                    endIndex = splitOnMiddleOfLongestSilence(
                        audioArray: audioArray,
                        startIndex: startIndex,
                        endIndex: min(audioArray.count, startIndex + maxChunkLength)
                    )
                }

                guard endIndex > startIndex else {
                    break
                }
                Logging.debug("Found chunk from \(Logging.formatTimestamp(Float(startIndex) / Float(WhisperKit.sampleRate))) to \(Logging.formatTimestamp(Float(endIndex) / Float(WhisperKit.sampleRate)))")
                let audioSlice = AudioChunk(seekOffsetIndex: startIndex, audioSamples: Array(audioArray[startIndex..<endIndex]))
                chunkedAudio.append(audioSlice)
                startIndex = endIndex
            }
        }
        return chunkedAudio
    }
}
