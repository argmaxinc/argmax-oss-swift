//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import WhisperKit

/// Word timing with speaker assignment
public struct WordWithSpeaker: Equatable, Sendable {
    public let wordTiming: WordTiming
    public let speaker: Int?

    public init(wordTiming: WordTiming, speaker: Int?) {
        self.wordTiming = wordTiming
        self.speaker = speaker
    }
}

public struct RTTMLine: CustomStringConvertible {
    public let fileId: String
    public let startTime: Float
    public let duration: Float
    public let orthography: String
    public let speakerName: String

    public init(
        fileId: String,
        speakerId: Int,
        startTime: Float,
        duration: Float,
        orthography: String = "<NA>"
    ) {
        self.fileId = fileId
        self.startTime = startTime
        self.duration = duration
        self.orthography = orthography
        self.speakerName = speakerId == -1 ? "UNKNOWN" : String(UnicodeScalar(UInt8(65 + speakerId % 26)))
    }

    public var description: String {
        let timeFormat = "%.3f"
        return "SPEAKER \(fileId) 1 \(String(format: timeFormat, startTime)) \(String(format: timeFormat, duration)) \(orthography) <NA> \(speakerName) <NA> <NA>"
    }

    /// Generate RTTM lines from words with speaker assignments.
    /// Groups consecutive words by the same speaker into segments.
    /// Words with a nil speaker are excluded from the output.
    public static func fromWords(_ wordsWithSpeakers: [WordWithSpeaker], fileName: String) -> [RTTMLine] {
        var rttmLines: [RTTMLine] = []
        var currentSpeaker: Int? = nil
        var segmentStart: Float? = nil
        var lastEnd: Float? = nil
        var currentWords: [String] = []

        for wordWithSpeaker in wordsWithSpeakers {
            let speakerId = wordWithSpeaker.speaker
            if speakerId != currentSpeaker {
                if let start = segmentStart, let end = lastEnd, let speaker = currentSpeaker {
                    rttmLines.append(RTTMLine(
                        fileId: fileName,
                        speakerId: speaker,
                        startTime: start,
                        duration: end - start,
                        orthography: currentWords.joined(separator: " ")
                    ))
                }
                currentSpeaker = speakerId
                segmentStart = wordWithSpeaker.wordTiming.start
                currentWords = [wordWithSpeaker.wordTiming.word.trimmingCharacters(in: .whitespaces)]
            } else {
                currentWords.append(wordWithSpeaker.wordTiming.word.trimmingCharacters(in: .whitespaces))
            }
            lastEnd = wordWithSpeaker.wordTiming.end
        }

        if let start = segmentStart, let end = lastEnd, let speaker = currentSpeaker {
            rttmLines.append(RTTMLine(
                fileId: fileName,
                speakerId: speaker,
                startTime: start,
                duration: end - start,
                orthography: currentWords.joined(separator: " ")
            ))
        }

        return rttmLines
    }
}
