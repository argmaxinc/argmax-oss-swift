//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import XCTest
@testable import SpeakerKit
@testable import WhisperKit

final class RTTMLineTests: XCTestCase {

    func testGenerateRTTMWithoutTranscription() {
        var binaryMatrix = [[Int]](repeating: [0], count: 2)
        let diarizationFrameRate: Float = 10.0
        let framesPerSpeaker = Int(diarizationFrameRate * 2)

        binaryMatrix[0] = [Int](repeating: 0, count: framesPerSpeaker)
        binaryMatrix[1] = [Int](repeating: 0, count: framesPerSpeaker)

        for i in 0..<Int(diarizationFrameRate) {
            binaryMatrix[0][i] = 1
        }
        for i in Int(diarizationFrameRate)..<framesPerSpeaker {
            binaryMatrix[1][i] = 1
        }

        let diarizationResult = DiarizationResult(binaryMatrix: binaryMatrix, diarizationFrameRate: diarizationFrameRate)

        let rttmLines = diarizationResult.segments.map { segment in
            RTTMLine(
                fileId: "audio",
                speakerId: segment.speaker.speakerId ?? -1,
                startTime: segment.startTime,
                duration: segment.endTime - segment.startTime
            )
        }

        XCTAssertEqual(rttmLines.count, 2)

        XCTAssertEqual(rttmLines[0].fileId, "audio")
        XCTAssertEqual(rttmLines[0].speakerName, "A")
        XCTAssertEqual(rttmLines[0].startTime, 0.0)
        XCTAssertEqual(rttmLines[0].duration, 1.0)

        XCTAssertEqual(rttmLines[1].fileId, "audio")
        XCTAssertEqual(rttmLines[1].speakerName, "B")
        XCTAssertEqual(rttmLines[1].startTime, 1.0)
        XCTAssertEqual(rttmLines[1].duration, 1.0)
    }

    func testGenerateRTTMWithTranscription() {
        var binaryMatrix = [[Int]](repeating: [0], count: 1)
        let diarizationFrameRate: Float = 10.0
        let framesPerSpeaker = Int(diarizationFrameRate * 2)

        binaryMatrix[0] = [Int](repeating: 1, count: framesPerSpeaker)

        let diarizationResult = DiarizationResult(binaryMatrix: binaryMatrix, diarizationFrameRate: diarizationFrameRate)

        let transcription = [
            TranscriptionResult(
                text: "Hello world",
                segments: [
                    TranscriptionSegment(
                        id: 0, seek: 0, start: 0.0, end: 2.0,
                        text: "Hello world", tokens: [],
                        tokenLogProbs: [[:]], temperature: 1.0,
                        avgLogprob: 0.0, compressionRatio: 1.0, noSpeechProb: 0.0,
                        words: [
                            WordTiming(word: "Hello ", tokens: [1], start: 0.0, end: 1.0, probability: 0.9),
                            WordTiming(word: "world", tokens: [2], start: 1.0, end: 2.0, probability: 0.9)
                        ]
                    )
                ],
                language: "en",
                timings: TranscriptionTimings(),
                seekTime: nil
            )
        ]

        let speakerSegments = diarizationResult.addSpeakerInfo(to: transcription, strategy: .segment)

        var rttmLines: [RTTMLine] = []
        for resultSegments in speakerSegments {
            for segment in resultSegments {
                rttmLines.append(RTTMLine(
                    fileId: "audio",
                    speakerId: segment.speaker.speakerId ?? -1,
                    startTime: segment.startTime,
                    duration: segment.endTime - segment.startTime,
                    orthography: segment.text
                ))
            }
        }

        XCTAssertEqual(rttmLines.count, 1)
        XCTAssertEqual(rttmLines[0].fileId, "audio")
        XCTAssertEqual(rttmLines[0].speakerName, "A")
        XCTAssertEqual(rttmLines[0].startTime, 0.0)
        XCTAssertEqual(rttmLines[0].duration, 2.0)
        XCTAssertEqual(rttmLines[0].orthography, "Hello world")
    }

    func testGenerateRTTMWithCustomFileName() {
        var binaryMatrix = [[Int]](repeating: [0], count: 1)
        let diarizationFrameRate: Float = 10.0
        let framesPerSpeaker = Int(diarizationFrameRate)

        binaryMatrix[0] = [Int](repeating: 1, count: framesPerSpeaker)

        let diarizationResult = DiarizationResult(binaryMatrix: binaryMatrix, diarizationFrameRate: diarizationFrameRate)

        let rttmLines = diarizationResult.segments.map { segment in
            RTTMLine(
                fileId: "custom_audio",
                speakerId: segment.speaker.speakerId ?? -1,
                startTime: segment.startTime,
                duration: segment.endTime - segment.startTime
            )
        }

        XCTAssertEqual(rttmLines.count, 1)
        XCTAssertEqual(rttmLines[0].fileId, "custom_audio")
        XCTAssertEqual(rttmLines[0].speakerName, "A")
        XCTAssertEqual(rttmLines[0].startTime, 0.0)
        XCTAssertEqual(rttmLines[0].duration, 1.0)
    }

    // MARK: - RTTMLine.fromWords Tests

    func testFromWords_SingleSpeakerMultipleWords() {
        let words: [WordWithSpeaker] = [
            WordWithSpeaker(wordTiming: WordTiming(word: "Hello ", tokens: [], start: 0.0, end: 0.5, probability: 0.9), speaker: 0),
            WordWithSpeaker(wordTiming: WordTiming(word: "world", tokens: [], start: 0.5, end: 1.0, probability: 0.9), speaker: 0)
        ]

        let rttmLines = RTTMLine.fromWords(words, fileName: "test")

        XCTAssertEqual(rttmLines.count, 1)
        XCTAssertEqual(rttmLines[0].fileId, "test")
        XCTAssertEqual(rttmLines[0].speakerName, "A")
        XCTAssertEqual(rttmLines[0].startTime, 0.0)
        XCTAssertEqual(rttmLines[0].duration, 1.0)
        XCTAssertEqual(rttmLines[0].orthography, "Hello world")
    }

    func testFromWords_MultipleSpeakers() {
        let words: [WordWithSpeaker] = [
            WordWithSpeaker(wordTiming: WordTiming(word: "Hello ", tokens: [], start: 0.0, end: 0.5, probability: 0.9), speaker: 0),
            WordWithSpeaker(wordTiming: WordTiming(word: "there", tokens: [], start: 0.5, end: 1.0, probability: 0.9), speaker: 0),
            WordWithSpeaker(wordTiming: WordTiming(word: "How ", tokens: [], start: 1.5, end: 2.0, probability: 0.9), speaker: 1),
            WordWithSpeaker(wordTiming: WordTiming(word: "are ", tokens: [], start: 2.0, end: 2.3, probability: 0.9), speaker: 1),
            WordWithSpeaker(wordTiming: WordTiming(word: "you?", tokens: [], start: 2.3, end: 2.8, probability: 0.9), speaker: 1)
        ]

        let rttmLines = RTTMLine.fromWords(words, fileName: "test")

        XCTAssertEqual(rttmLines.count, 2)

        XCTAssertEqual(rttmLines[0].fileId, "test")
        XCTAssertEqual(rttmLines[0].speakerName, "A")
        XCTAssertEqual(rttmLines[0].startTime, 0.0)
        XCTAssertEqual(rttmLines[0].duration, 1.0)
        XCTAssertEqual(rttmLines[0].orthography, "Hello there")

        XCTAssertEqual(rttmLines[1].fileId, "test")
        XCTAssertEqual(rttmLines[1].speakerName, "B")
        XCTAssertEqual(rttmLines[1].startTime, 1.5)
        XCTAssertEqual(rttmLines[1].duration, 1.3, accuracy: 0.01)
        XCTAssertEqual(rttmLines[1].orthography, "How are you?")
    }

    func testFromWords_AlternatingSpeakers() {
        let words: [WordWithSpeaker] = [
            WordWithSpeaker(wordTiming: WordTiming(word: "A: ", tokens: [], start: 0.0, end: 0.3, probability: 0.9), speaker: 0),
            WordWithSpeaker(wordTiming: WordTiming(word: "Hello", tokens: [], start: 0.3, end: 0.8, probability: 0.9), speaker: 0),
            WordWithSpeaker(wordTiming: WordTiming(word: "B: ", tokens: [], start: 1.0, end: 1.2, probability: 0.9), speaker: 1),
            WordWithSpeaker(wordTiming: WordTiming(word: "Hi", tokens: [], start: 1.2, end: 1.5, probability: 0.9), speaker: 1),
            WordWithSpeaker(wordTiming: WordTiming(word: "A: ", tokens: [], start: 2.0, end: 2.2, probability: 0.9), speaker: 0),
            WordWithSpeaker(wordTiming: WordTiming(word: "Bye", tokens: [], start: 2.2, end: 2.6, probability: 0.9), speaker: 0)
        ]

        let rttmLines = RTTMLine.fromWords(words, fileName: "conversation")

        XCTAssertEqual(rttmLines.count, 3)

        XCTAssertEqual(rttmLines[0].speakerName, "A")
        XCTAssertEqual(rttmLines[0].orthography, "A: Hello")

        XCTAssertEqual(rttmLines[1].speakerName, "B")
        XCTAssertEqual(rttmLines[1].orthography, "B: Hi")

        XCTAssertEqual(rttmLines[2].speakerName, "A")
        XCTAssertEqual(rttmLines[2].orthography, "A: Bye")
    }

    func testFromWords_EmptyArray() {
        let words: [WordWithSpeaker] = []

        let rttmLines = RTTMLine.fromWords(words, fileName: "test")

        XCTAssertEqual(rttmLines.count, 0)
    }

    func testFromWords_WordsWithoutSpeaker() {
        let words: [WordWithSpeaker] = [
            WordWithSpeaker(wordTiming: WordTiming(word: "Unknown ", tokens: [], start: 0.0, end: 0.5, probability: 0.9), speaker: nil),
            WordWithSpeaker(wordTiming: WordTiming(word: "words", tokens: [], start: 0.5, end: 1.0, probability: 0.9), speaker: nil)
        ]

        let rttmLines = RTTMLine.fromWords(words, fileName: "test")

        XCTAssertEqual(rttmLines.count, 0)
    }

    func testFromWords_MixedSpeakerAssignments() {
        let words: [WordWithSpeaker] = [
            WordWithSpeaker(wordTiming: WordTiming(word: "Hello ", tokens: [], start: 0.0, end: 0.5, probability: 0.9), speaker: 0),
            WordWithSpeaker(wordTiming: WordTiming(word: "unknown ", tokens: [], start: 0.5, end: 1.0, probability: 0.9), speaker: nil),
            WordWithSpeaker(wordTiming: WordTiming(word: "world", tokens: [], start: 1.0, end: 1.5, probability: 0.9), speaker: 0)
        ]

        let rttmLines = RTTMLine.fromWords(words, fileName: "test")

        XCTAssertEqual(rttmLines.count, 2)
        XCTAssertEqual(rttmLines[0].orthography, "Hello")
        XCTAssertEqual(rttmLines[1].orthography, "world")
    }

    func testFromWords_WhitespaceHandling() {
        let words: [WordWithSpeaker] = [
            WordWithSpeaker(wordTiming: WordTiming(word: "  Hello  ", tokens: [], start: 0.0, end: 0.5, probability: 0.9), speaker: 0),
            WordWithSpeaker(wordTiming: WordTiming(word: "  world  ", tokens: [], start: 0.5, end: 1.0, probability: 0.9), speaker: 0)
        ]

        let rttmLines = RTTMLine.fromWords(words, fileName: "test")

        XCTAssertEqual(rttmLines.count, 1)
        XCTAssertEqual(rttmLines[0].orthography, "Hello world")
    }
}
