//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import WhisperKit

/// A segment of speech from a single speaker
public struct SpeakerSegment: Identifiable, CustomStringConvertible, CustomDebugStringConvertible, Sendable {
    public let id: UUID

    public let speaker: SpeakerInfo
    public let startTime: Float
    public let endTime: Float
    public let frameRate: Float

    /// Original start frame index (only set for frame-primary segments like Pyannote)
    private let originalStartFrame: Int?
    /// Original end frame index (only set for frame-primary segments like Pyannote)
    private let originalEndFrame: Int?

    /// Start frame index (computed from time for time-primary, preserved for frame-primary)
    public var startFrame: Int {
        originalStartFrame ?? Int(startTime * frameRate)
    }
    /// End frame index (computed from time for time-primary, preserved for frame-primary)
    public var endFrame: Int {
        originalEndFrame ?? Int(endTime * frameRate)
    }

    public let transcription: TranscriptionSegment?
    public let speakerWords: [SpeakerWordTiming]

    public var text: String {
        speakerWords.map { $0.wordTiming.word }.joined()
    }

    public var description: String {
        let start = speakerWords.first?.wordTiming.start ?? startTime
        let end = speakerWords.last?.wordTiming.end ?? endTime
        let text = text.trimmingCharacters(in: .whitespaces)
        return String(format: "\(speaker) [%.2f-%.2fs]: \(text)", start, end)
    }

    public var debugDescription: String {
        return "Segment \(id.uuidString) -- " + description
    }

    /// Time-primary init: use when the source of truth is seconds.
    public init(speaker: SpeakerInfo, startTime: Float, endTime: Float, frameRate: Float, transcription: TranscriptionSegment? = nil, speakerWords: [SpeakerWordTiming] = []) {
        self.id = UUID()
        self.speaker = speaker
        self.startTime = startTime
        self.endTime = endTime
        self.frameRate = frameRate
        self.originalStartFrame = nil
        self.originalEndFrame = nil
        self.transcription = transcription
        self.speakerWords = speakerWords
    }

    /// Frame-primary init: use when the source of truth is frame indices (e.g. Pyannote binary matrix).
    public init(speaker: SpeakerInfo, startFrame: Int, endFrame: Int, frameRate: Float, transcription: TranscriptionSegment? = nil, speakerWords: [SpeakerWordTiming] = []) {
        self.id = UUID()
        self.speaker = speaker
        self.startTime = Float(startFrame) / frameRate
        self.endTime = Float(endFrame) / frameRate
        self.frameRate = frameRate
        self.originalStartFrame = startFrame
        self.originalEndFrame = endFrame
        self.transcription = transcription
        self.speakerWords = speakerWords
    }

    /// Word-timing init: time-primary, derives segment bounds from word start/end seconds.
    public init(transcription: TranscriptionSegment, speakerWords: [SpeakerWordTiming], diarizationFrameRate: Float) {
        let speaker: SpeakerInfo
        if let speakerId = speakerWords.first?.speaker.speakerId {
            speaker = .speakerId(speakerId)
        } else {
            speaker = .noMatch
        }
        let minStart = speakerWords.map({ $0.wordTiming.start }).min() ?? 0
        let maxEnd = speakerWords.map({ $0.wordTiming.end }).max() ?? 0
        self.init(
            speaker: speaker,
            startTime: minStart,
            endTime: maxEnd,
            frameRate: diarizationFrameRate,
            transcription: transcription,
            speakerWords: speakerWords
        )
    }
}
