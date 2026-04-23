//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import WhisperKit
import ArgmaxCore

/// Matching mode for finding the best speaker segment overlap
enum SegmentMatchingMode {
    /// Uses Intersection over Union (IoU) - better for comparing segments of different sizes
    case intersectionOverUnion
    /// Uses raw intersection - better for small subsegments
    case intersection
}

// MARK: - DiarizationPipelineTimings

/// Protocol for pipeline-specific timing objects produced by a diarizer backend.
///
/// Conform to this protocol when implementing a custom diarizer backend that
/// produces its own timing data. Pyannote produces ``PyannoteDiarizationTimings`` conforming to this protocol.
public protocol DiarizationTimings: CustomStringConvertible, CustomDebugStringConvertible, Sendable {}

// MARK: - DiarizationResult

public struct DiarizationResult: Sendable {
    private let binaryMatrix: [[Int]]
    public let speakerCount: Int
    public let totalFrames: Int
    public let frameRate: Float
    public private(set) var segments: [SpeakerSegment]
    public var timings: (any DiarizationTimings)?

    /// Per-speaker centroid embeddings keyed by `speakerId`, in the raw speaker-embedder output
    /// space (unnormalised, pre-PLDA). Useful for linking the same speaker across independent
    /// `diarize(...)` calls without re-running the embedder.
    ///
    /// Each centroid is the arithmetic mean of the final per-window embeddings assigned to that
    /// `speakerId` after clustering and cluster reassignment, so the centroid reflects the
    /// speaker's actual membership in this result.
    ///
    /// Compare centroids with cosine distance via `centroidCosineDistance(between:_:)`, which
    /// matches the convention used by `MathOps.cosineDistanceMatrix` elsewhere in SpeakerKit.
    ///
    /// This field is populated by the Pyannote backend (`PyannoteDiarizer`). Other backends
    /// conforming to `Diarizer` may leave it as `[:]` if they do not expose per-cluster centroids.
    public private(set) var speakerCentroidEmbeddings: [Int: [Float]]

    /// Pyannote init: builds segments from binary speaker activity matrix
    init(binaryMatrix: [[Int]], diarizationFrameRate: Float, speakerCentroidEmbeddings: [Int: [Float]] = [:]) {
        self.binaryMatrix = binaryMatrix
        self.frameRate = diarizationFrameRate
        self.speakerCount = binaryMatrix.count
        self.totalFrames = speakerCount > 0 ? binaryMatrix[0].count : 0
        self.segments = []
        self.timings = nil
        self.speakerCentroidEmbeddings = speakerCentroidEmbeddings

        self.updateSegments(minActiveOffset: 0.0)
    }

    /// Generic init: for engines that produce segments directly
    public init(speakerCount: Int, totalFrames: Int, frameRate: Float, segments: [SpeakerSegment], timings: (any DiarizationTimings)? = nil) {
        self.binaryMatrix = []
        self.speakerCount = speakerCount
        self.totalFrames = totalFrames
        self.frameRate = frameRate
        self.segments = segments
        self.timings = timings
        self.speakerCentroidEmbeddings = [:]
    }

    public mutating func updateSegments(minActiveOffset: Float) {
        guard !binaryMatrix.isEmpty else {
            Logging.debug("[SpeakerKit] updateSegments skipped (no binary matrix)")
            return
        }
        Logging.debug("[SpeakerKit] updateSegments with min \(minActiveOffset) offset")
        var segments: [SpeakerSegment] = []
        let minActiveOffsetFrames = Int(minActiveOffset * frameRate)
        for speakerId in 0..<binaryMatrix.count {
            var currentStart: Int? = nil

            for frame in 0..<binaryMatrix[speakerId].count {
                if binaryMatrix[speakerId][frame] == 1 {
                    if !segments.isEmpty,
                       let lastSegment = segments.last,
                       speakerId == lastSegment.speaker.speakerId,
                       frame - lastSegment.endFrame <= minActiveOffsetFrames
                    {
                        currentStart = lastSegment.startFrame
                        segments.removeLast()
                    }
                    if currentStart == nil {
                        currentStart = frame
                    }
                } else if let start = currentStart {
                    segments.append(SpeakerSegment(
                        speaker: .speakerId(speakerId),
                        startFrame: start,
                        endFrame: frame,
                        frameRate: frameRate
                    ))
                    currentStart = nil
                }
            }

            if let start = currentStart {
                segments.append(SpeakerSegment(
                    speaker: .speakerId(speakerId),
                    startFrame: start,
                    endFrame: binaryMatrix[speakerId].count,
                    frameRate: frameRate
                ))
            }
        }

        self.segments = segments.sorted { $0.startFrame < $1.startFrame }
    }

    // MARK: - Speaker Centroid Comparison

    /// Cosine distance in `[0.0, 2.0]` between two speaker centroids from this result.
    ///
    /// Delegates to `MathOps.cosineDistance(_:_:)`, matching the convention used by
    /// `MathOps.cosineDistanceMatrix` elsewhere in SpeakerKit. The result is clamped to
    /// `[0, 2]` to absorb floating-point error near the extremes.
    ///
    /// - Returns: `nil` if either `speakerId` is absent from
    ///   ``speakerCentroidEmbeddings``, the centroids have different dimensions, or either
    ///   vector is empty. Zero-magnitude centroids (unreachable in real diarization runs)
    ///   yield `MathOps.cosineDistance`'s sentinel of `1.0`.
    public func centroidCosineDistance(between a: Int, _ b: Int) -> Float? {
        guard let lhs = speakerCentroidEmbeddings[a],
              let rhs = speakerCentroidEmbeddings[b],
              lhs.count == rhs.count, !lhs.isEmpty else { return nil }
        return MathOps.cosineDistance(lhs, rhs)
    }

    // MARK: - Speaker Info Matching

    public func addSpeakerInfo(to transcription: [TranscriptionResult], strategy: SpeakerInfoStrategy = SpeakerInfoStrategy.subsegment) -> [[SpeakerSegment]] {
        return transcription.map { result in
            switch strategy {
            case .segment:
                return processTranscriptionWithSegmentStrategy(result)
            case .subsegment(let betweenWordThreshold):
                return processTranscriptionWithSubsegmentStrategy(result, betweenWordThreshold: betweenWordThreshold)
            }
        }
    }

    // MARK: - Private Strategy Methods

    private func processTranscriptionWithSegmentStrategy(_ result: TranscriptionResult) -> [SpeakerSegment] {
        return processSegmentsWithMatchingMode(
            result.segments,
            matchingMode: .intersectionOverUnion
        )
    }

    private func processTranscriptionWithSubsegmentStrategy(_ result: TranscriptionResult, betweenWordThreshold: Float) -> [SpeakerSegment] {
        let allSubsegments = result.segments.flatMap { segment -> [TranscriptionSegment] in
            guard let words = validateWordsInSegment(segment) else { return [] }
            return groupWordsIntoSubsegments(words, betweenWordThreshold: betweenWordThreshold)
        }

        return processSegmentsWithMatchingMode(
            allSubsegments,
            matchingMode: .intersection
        )
    }

    private func processSegmentsWithMatchingMode(
        _ transcriptionSegments: [TranscriptionSegment],
        matchingMode: SegmentMatchingMode
    ) -> [SpeakerSegment] {
        var segments: [SpeakerSegment] = []
        var previousSpeaker: SpeakerInfo = .noMatch

        for segment in transcriptionSegments {
            guard let words = validateWordsInSegment(segment) else { continue }

            let speaker = findSpeakerForSegment(segment, matchingMode: matchingMode) ?? previousSpeaker
            let speakerWords = words.map { SpeakerWordTiming(wordTiming: $0, speaker: speaker) }

            segments.append(SpeakerSegment(
                transcription: segment,
                speakerWords: speakerWords,
                diarizationFrameRate: frameRate
            ))

            previousSpeaker = speaker
        }

        return segments
    }

    // MARK: - Helper Methods

    private func groupWordsIntoSubsegments(
        _ words: [WordTiming],
        betweenWordThreshold: Float
    ) -> [TranscriptionSegment] {
        guard !words.isEmpty else { return [] }
        var subsegments: [TranscriptionSegment] = []
        var currentWords: [WordTiming] = [words[0]]
        var currentStart: Float = words[0].start

        for i in 1..<words.count {
            let prev = words[i-1]
            let curr = words[i]
            let gap = curr.start - prev.end

            if gap > betweenWordThreshold {
                guard let currentEnd = currentWords.last?.end else { continue }
                subsegments.append(TranscriptionSegment(
                    id: 0,
                    seek: 0,
                    start: currentStart,
                    end: currentEnd,
                    text: currentWords.map { $0.word }.joined(separator: ""),
                    tokens: [],
                    tokenLogProbs: [[:] ],
                    temperature: 1.0,
                    avgLogprob: 0.0,
                    compressionRatio: 1.0,
                    noSpeechProb: 0.0,
                    words: currentWords
                ))
                currentWords = [curr]
                currentStart = curr.start
            } else {
                currentWords.append(curr)
            }
        }
        if let currentEnd = currentWords.last?.end {
            subsegments.append(TranscriptionSegment(
                id: 0,
                seek: 0,
                start: currentStart,
                end: currentEnd,
                text: currentWords.map { $0.word }.joined(separator: ""),
                tokens: [],
                tokenLogProbs: [[:] ],
                temperature: 1.0,
                avgLogprob: 0.0,
                compressionRatio: 1.0,
                noSpeechProb: 0.0,
                words: currentWords
            ))
        }
        return subsegments
    }

    private func validateWordsInSegment(_ segment: TranscriptionSegment) -> [WordTiming]? {
        guard let words = segment.words, !words.isEmpty else {
            Logging.info("[SpeakerKit] No words for segment: \(segment.id). Enable `wordTimestamps` on DecodingOptions.")
            return nil
        }
        return words
    }

    private func findSpeakerForSegment(
        _ transcriptionSegment: TranscriptionSegment,
        matchingMode: SegmentMatchingMode
    ) -> SpeakerInfo? {
        let transcriptionStartFrame = Int(transcriptionSegment.start * frameRate)
        let transcriptionEndFrame = Int(transcriptionSegment.end * frameRate)

        var closestSegment: SpeakerSegment? = nil
        var maxScore: Float = 0.0

        for diarizationSegment in segments {
            let diarizationStartFrame = diarizationSegment.startFrame
            let diarizationEndFrame = diarizationSegment.endFrame

            if diarizationEndFrame < transcriptionStartFrame {
                continue
            }
            if diarizationStartFrame > transcriptionEndFrame {
                break
            }

            let intersection = max(0, min(transcriptionEndFrame, diarizationEndFrame) - max(transcriptionStartFrame, diarizationStartFrame))

            let score: Float
            switch matchingMode {
            case .intersectionOverUnion:
                let union = max(transcriptionEndFrame, diarizationEndFrame) - min(transcriptionStartFrame, diarizationStartFrame)
                guard union != 0 else { continue }
                score = Float(intersection) / Float(union)
            case .intersection:
                score = Float(intersection)
            }

            if closestSegment == nil || score > maxScore {
                maxScore = score
                closestSegment = diarizationSegment
            }
        }

        return closestSegment?.speaker
    }
}
