//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import XCTest
@testable import SpeakerKit
@testable import WhisperKit

final class DiarizationResultTests: XCTestCase {

    func testSpeakerSegmentWordLevelInitialization() {
        let word1 = WordTiming(word: "hello ", tokens: [1], start: 0.0, end: 0.5, probability: 0.9)
        let word2 = WordTiming(word: "world", tokens: [2], start: 0.5, end: 1.0, probability: 0.95)

        let speaker: SpeakerInfo = .speakerId(1)
        let speakerWords = [
            SpeakerWordTiming(wordTiming: word1, speaker: speaker),
            SpeakerWordTiming(wordTiming: word2, speaker: speaker)
        ]

        let transcription = TranscriptionSegment(
            id: 100,
            seek: 0,
            start: 0.0,
            end: 1.0,
            text: "hello world",
            tokens: [1, 2],
            tokenLogProbs: [[:], [:]],
            temperature: 1.0,
            avgLogprob: 0.0,
            compressionRatio: 1.0,
            noSpeechProb: 0.1
        )
        let segment = SpeakerSegment(transcription: transcription, speakerWords: speakerWords, diarizationFrameRate: 10.0)

        XCTAssertEqual(segment.speaker, .speakerId(1))
        XCTAssertEqual(segment.startFrame, 0)
        XCTAssertEqual(segment.endFrame, 10)
        XCTAssertEqual(segment.text, "hello world")
        XCTAssertEqual(segment.speakerWords.count, 2)
        XCTAssertEqual(segment.transcription?.id, transcription.id)
    }

    func testSpeakerSegmentWithNoWords() {
        let segment = SpeakerSegment(speaker: .noMatch, startFrame: 0, endFrame: 10, frameRate: 25.0)

        XCTAssertEqual(segment.speaker, .noMatch)
        XCTAssertEqual(segment.startFrame, 0)
        XCTAssertEqual(segment.endFrame, 10)
        XCTAssertEqual(segment.text, "")
        XCTAssertTrue(segment.speakerWords.isEmpty)
        XCTAssertNil(segment.transcription)
    }

    func testSpeakerSegmentFormatting() {
        let word1 = WordTiming(word: "hello ", tokens: [1], start: 0.0, end: 0.5, probability: 0.9)
        let word2 = WordTiming(word: "world", tokens: [2], start: 0.5, end: 1.0, probability: 0.95)

        let speaker: SpeakerInfo = .speakerId(1)
        let speakerWords = [
            SpeakerWordTiming(wordTiming: word1, speaker: speaker),
            SpeakerWordTiming(wordTiming: word2, speaker: speaker)
        ]

        let transcription = TranscriptionSegment(
            id: 0,
            seek: 0,
            start: 0.0,
            end: 1.0,
            text: "hello world",
            tokens: [1, 2],
            tokenLogProbs: [[:], [:]],
            temperature: 1.0,
            avgLogprob: 0.0,
            compressionRatio: 1.0,
            noSpeechProb: 0.1
        )
        let segment = SpeakerSegment(transcription: transcription, speakerWords: speakerWords, diarizationFrameRate: 10.0)
        let formatted = segment.debugDescription
        XCTAssertEqual(formatted, "Segment \(segment.id.uuidString) -- Speaker 1 [0.00-1.00s]: hello world")
    }

    func testDiarizationResultInitialization() {
        let diarizationFrameRate: Float = 25.0
        var binaryMatrix = [[Int]](repeating: [0], count: 2)
        binaryMatrix[0] = [Int](repeating: 0, count: Int(diarizationFrameRate * 2))
        binaryMatrix[1] = [Int](repeating: 0, count: Int(diarizationFrameRate * 2))

        for i in 0..<Int(diarizationFrameRate) {
            binaryMatrix[0][i] = 1
        }

        for i in Int(diarizationFrameRate)..<Int(diarizationFrameRate * 2) {
            binaryMatrix[1][i] = 1
        }

        let result = DiarizationResult(binaryMatrix: binaryMatrix, diarizationFrameRate: diarizationFrameRate)

        XCTAssertEqual(result.speakerCount, 2)
        XCTAssertEqual(result.totalFrames, 50)
        XCTAssertEqual(result.segments.count, 2)

        XCTAssertEqual(result.segments[0].speaker, .speakerId(0))
        XCTAssertEqual(result.segments[0].speaker.speakerId, 0)
        XCTAssertEqual(result.segments[0].speaker.speakerIds, [0])
        XCTAssertEqual(result.segments[0].startFrame, 0)
        XCTAssertEqual(result.segments[0].endFrame, 25)

        XCTAssertEqual(result.segments[1].speaker, .speakerId(1))
        XCTAssertEqual(result.segments[1].speaker.speakerId, 1)
        XCTAssertEqual(result.segments[1].speaker.speakerIds, [1])
        XCTAssertEqual(result.segments[1].startFrame, 25)
        XCTAssertEqual(result.segments[1].endFrame, 50)
    }

    func testDiarizationResultWithEmptyMatrix() {
        let result = DiarizationResult(binaryMatrix: [], diarizationFrameRate: 25.0)

        XCTAssertEqual(result.speakerCount, 0)
        XCTAssertEqual(result.totalFrames, 0)
        XCTAssertTrue(result.segments.isEmpty)
    }

    func testSpeakerAssignmentWithSubsegmentStrategy() {
        let diarizationFrameRate: Float = 10.0
        var binaryMatrix = [[Int]](repeating: [0], count: 2)
        binaryMatrix[0] = [Int](repeating: 0, count: 30)
        binaryMatrix[1] = [Int](repeating: 0, count: 30)

        for i in 0..<10 {
            binaryMatrix[0][i] = 1
        }

        for i in 10..<20 {
            binaryMatrix[1][i] = 1
        }

        let result = DiarizationResult(binaryMatrix: binaryMatrix, diarizationFrameRate: diarizationFrameRate)

        let transcription = [TranscriptionResult(
            text: "Hello world",
            segments: [
                TranscriptionSegment(
                    id: 0, seek: 0, start: 0.5, end: 1.5,
                    text: "Hello world", tokens: [],
                    tokenLogProbs: [[:]], temperature: 1.0,
                    avgLogprob: 0.0, compressionRatio: 1.0, noSpeechProb: 0.0,
                    words: [
                        WordTiming(word: "Hello ", tokens: [1], start: 0.5, end: 0.8, probability: 0.9),
                        WordTiming(word: "world", tokens: [2], start: 1.2, end: 1.5, probability: 0.9)
                    ]
                )
            ],
            language: "en",
            timings: TranscriptionTimings(),
            seekTime: nil
        )]

        // Words "Hello" (0.5-0.8s) and "world" (1.2-1.5s) have a 0.4s gap > 0.15s subsegment threshold,
        // so they split into two subsegments assigned to speaker 0 and speaker 1 respectively.
        let resultWithSpeakerInfo = result.addSpeakerInfo(to: transcription, strategy: .subsegment)
        XCTAssertEqual(resultWithSpeakerInfo.count, 1)
        XCTAssertEqual(resultWithSpeakerInfo[0].count, 2)
        XCTAssertEqual(resultWithSpeakerInfo[0][0].speaker, .speakerId(0))
        XCTAssertEqual(resultWithSpeakerInfo[0][1].speaker, .speakerId(1))
    }

    func testSpeakerAssignmentWithNoSpeakers() {
        let result = DiarizationResult(binaryMatrix: [], diarizationFrameRate: 10.0)

        let transcription = [TranscriptionResult(
            text: "test",
            segments: [
                TranscriptionSegment(
                    id: 0, seek: 0, start: 0.0, end: 1.0,
                    text: "test", tokens: [],
                    tokenLogProbs: [[:]], temperature: 1.0,
                    avgLogprob: 0.0, compressionRatio: 1.0, noSpeechProb: 0.0,
                    words: [
                        WordTiming(word: "test", tokens: [1], start: 0.0, end: 1.0, probability: 0.9)
                    ]
                )
            ],
            language: "en",
            timings: TranscriptionTimings(),
            seekTime: nil
        )]

        let resultWithSpeakerInfo = result.addSpeakerInfo(to: transcription)
        XCTAssertEqual(resultWithSpeakerInfo[0][0].speakerWords[0].speaker, .noMatch)
    }

    // MARK: - Segment Strategy Tests

    func testSpeakerAssignmentSegmentStrategy() {
        let diarizationFrameRate: Float = 10.0
        var binaryMatrix = [[Int]](repeating: [0], count: 2)
        binaryMatrix[0] = [Int](repeating: 0, count: 30)
        binaryMatrix[1] = [Int](repeating: 0, count: 30)

        for i in 0..<10 {
            binaryMatrix[0][i] = 1
        }
        for i in 10..<20 {
            binaryMatrix[1][i] = 1
        }

        let result = DiarizationResult(binaryMatrix: binaryMatrix, diarizationFrameRate: diarizationFrameRate)

        let transcription = [TranscriptionResult(
            text: "Hello world",
            segments: [
                TranscriptionSegment(
                    id: 0, seek: 0, start: 0.5, end: 1.5,
                    text: "Hello world", tokens: [],
                    tokenLogProbs: [[:]], temperature: 1.0,
                    avgLogprob: 0.0, compressionRatio: 1.0, noSpeechProb: 0.0,
                    words: [
                        WordTiming(word: "Hello ", tokens: [1], start: 0.5, end: 0.8, probability: 0.9),
                        WordTiming(word: "world", tokens: [2], start: 1.2, end: 1.5, probability: 0.9)
                    ]
                )
            ],
            language: "en",
            timings: TranscriptionTimings(),
            seekTime: nil
        )]

        let resultWithSpeakerInfo = result.addSpeakerInfo(to: transcription, strategy: .segment)
        XCTAssertEqual(resultWithSpeakerInfo.count, 1)
        XCTAssertEqual(resultWithSpeakerInfo[0].count, 1)

        let segmentSpeaker = resultWithSpeakerInfo[0][0].speakerWords[0].speaker
        for speakerWord in resultWithSpeakerInfo[0][0].speakerWords {
            XCTAssertEqual(speakerWord.speaker, segmentSpeaker)
        }
    }

    func testSpeakerAssignmentSegmentStrategyWithDominantSpeaker() {
        let diarizationFrameRate: Float = 10.0
        var binaryMatrix = [[Int]](repeating: [0], count: 2)
        binaryMatrix[0] = [Int](repeating: 0, count: 30)
        binaryMatrix[1] = [Int](repeating: 0, count: 30)

        for i in 5..<25 { binaryMatrix[0][i] = 1 }
        for i in 15..<20 { binaryMatrix[1][i] = 1 }

        let result = DiarizationResult(binaryMatrix: binaryMatrix, diarizationFrameRate: diarizationFrameRate)

        let transcription = [TranscriptionResult(
            text: "This is a longer segment",
            segments: [
                TranscriptionSegment(
                    id: 0, seek: 0, start: 0.7, end: 2.3,
                    text: "This is a longer segment", tokens: [],
                    tokenLogProbs: [[:]], temperature: 1.0,
                    avgLogprob: 0.0, compressionRatio: 1.0, noSpeechProb: 0.0,
                    words: [
                        WordTiming(word: "This ", tokens: [1], start: 0.7, end: 0.9, probability: 0.9),
                        WordTiming(word: "is ", tokens: [2], start: 0.9, end: 1.1, probability: 0.9),
                        WordTiming(word: "a ", tokens: [3], start: 1.1, end: 1.3, probability: 0.9),
                        WordTiming(word: "longer ", tokens: [4], start: 1.3, end: 1.7, probability: 0.9),
                        WordTiming(word: "segment", tokens: [5], start: 1.7, end: 2.3, probability: 0.9)
                    ]
                )
            ],
            language: "en",
            timings: TranscriptionTimings(),
            seekTime: nil
        )]

        let resultWithSpeakerInfo = result.addSpeakerInfo(to: transcription, strategy: .segment)
        XCTAssertEqual(resultWithSpeakerInfo.count, 1)
        XCTAssertEqual(resultWithSpeakerInfo[0].count, 1)

        for speakerWord in resultWithSpeakerInfo[0][0].speakerWords {
            XCTAssertEqual(speakerWord.speaker, .speakerId(0))
        }
    }

    func testSpeakerAssignmentSegmentStrategyWithEqualOverlap() {
        let diarizationFrameRate: Float = 10.0
        var binaryMatrix = [[Int]](repeating: [0], count: 2)
        binaryMatrix[0] = [Int](repeating: 0, count: 20)
        binaryMatrix[1] = [Int](repeating: 0, count: 20)

        for i in 8..<12 {
            binaryMatrix[0][i] = 1
            binaryMatrix[1][i] = 1
        }

        let result = DiarizationResult(binaryMatrix: binaryMatrix, diarizationFrameRate: diarizationFrameRate)

        let transcription = [TranscriptionResult(
            text: "equal overlap",
            segments: [
                TranscriptionSegment(
                    id: 0, seek: 0, start: 0.9, end: 1.1,
                    text: "equal overlap", tokens: [],
                    tokenLogProbs: [[:]], temperature: 1.0,
                    avgLogprob: 0.0, compressionRatio: 1.0, noSpeechProb: 0.0,
                    words: [
                        WordTiming(word: "equal ", tokens: [1], start: 0.9, end: 1.0, probability: 0.9),
                        WordTiming(word: "overlap", tokens: [2], start: 1.0, end: 1.1, probability: 0.9)
                    ]
                )
            ],
            language: "en",
            timings: TranscriptionTimings(),
            seekTime: nil
        )]

        let resultWithSpeakerInfo = result.addSpeakerInfo(to: transcription, strategy: .segment)
        XCTAssertEqual(resultWithSpeakerInfo.count, 1)
        XCTAssertEqual(resultWithSpeakerInfo[0].count, 1)

        for speakerWord in resultWithSpeakerInfo[0][0].speakerWords {
            XCTAssertEqual(speakerWord.speaker, .speakerId(0))
        }
    }

    func testSpeakerAssignmentSegmentStrategyWithNoSpeakerActivity() {
        let result = DiarizationResult(binaryMatrix: [], diarizationFrameRate: 10.0)

        let transcription = [TranscriptionResult(
            text: "no speakers",
            segments: [
                TranscriptionSegment(
                    id: 0, seek: 0, start: 0.0, end: 1.0,
                    text: "no speakers", tokens: [],
                    tokenLogProbs: [[:]], temperature: 1.0,
                    avgLogprob: 0.0, compressionRatio: 1.0, noSpeechProb: 0.0,
                    words: [
                        WordTiming(word: "no ", tokens: [1], start: 0.0, end: 0.5, probability: 0.9),
                        WordTiming(word: "speakers", tokens: [2], start: 0.5, end: 1.0, probability: 0.9)
                    ]
                )
            ],
            language: "en",
            timings: TranscriptionTimings(),
            seekTime: nil
        )]

        let resultWithSpeakerInfo = result.addSpeakerInfo(to: transcription, strategy: .segment)
        XCTAssertEqual(resultWithSpeakerInfo.count, 1)
        XCTAssertEqual(resultWithSpeakerInfo[0].count, 1)

        for speakerWord in resultWithSpeakerInfo[0][0].speakerWords {
            XCTAssertEqual(speakerWord.speaker, .noMatch)
        }
    }

    func testSpeakerAssignmentSegmentStrategyWithMultipleSegments() {
        let diarizationFrameRate: Float = 10.0
        var binaryMatrix = [[Int]](repeating: [0], count: 2)
        binaryMatrix[0] = [Int](repeating: 0, count: 40)
        binaryMatrix[1] = [Int](repeating: 0, count: 40)

        for i in 5..<15 { binaryMatrix[0][i] = 1 }
        for i in 25..<35 { binaryMatrix[1][i] = 1 }

        let result = DiarizationResult(binaryMatrix: binaryMatrix, diarizationFrameRate: diarizationFrameRate)

        let transcription = [TranscriptionResult(
            text: "First segment. Second segment.",
            segments: [
                TranscriptionSegment(
                    id: 0, seek: 0, start: 0.7, end: 1.3,
                    text: "First segment.", tokens: [],
                    tokenLogProbs: [[:]], temperature: 1.0,
                    avgLogprob: 0.0, compressionRatio: 1.0, noSpeechProb: 0.0,
                    words: [
                        WordTiming(word: "First ", tokens: [1], start: 0.7, end: 0.9, probability: 0.9),
                        WordTiming(word: "segment.", tokens: [2], start: 0.9, end: 1.3, probability: 0.9)
                    ]
                ),
                TranscriptionSegment(
                    id: 1, seek: 0, start: 2.7, end: 3.3,
                    text: "Second segment.", tokens: [],
                    tokenLogProbs: [[:]], temperature: 1.0,
                    avgLogprob: 0.0, compressionRatio: 1.0, noSpeechProb: 0.0,
                    words: [
                        WordTiming(word: "Second ", tokens: [3], start: 2.7, end: 2.9, probability: 0.9),
                        WordTiming(word: "segment.", tokens: [4], start: 2.9, end: 3.3, probability: 0.9)
                    ]
                )
            ],
            language: "en",
            timings: TranscriptionTimings(),
            seekTime: nil
        )]

        let resultWithSpeakerInfo = result.addSpeakerInfo(to: transcription, strategy: .segment)
        XCTAssertEqual(resultWithSpeakerInfo.count, 1)
        XCTAssertEqual(resultWithSpeakerInfo[0].count, 2)

        for speakerWord in resultWithSpeakerInfo[0][0].speakerWords {
            XCTAssertEqual(speakerWord.speaker, .speakerId(0))
        }
        for speakerWord in resultWithSpeakerInfo[0][1].speakerWords {
            XCTAssertEqual(speakerWord.speaker, .speakerId(1))
        }
    }

    // MARK: - Subsegment Strategy Tests

    func testSubsegmentStrategySplitsOnWordGaps() {
        let diarizationFrameRate: Float = 10.0
        var binaryMatrix = [[Int]](repeating: [0], count: 2)
        binaryMatrix[0] = [Int](repeating: 0, count: 40)
        binaryMatrix[1] = [Int](repeating: 0, count: 40)

        for i in 0..<15 { binaryMatrix[0][i] = 1 }
        for i in 20..<40 { binaryMatrix[1][i] = 1 }

        let result = DiarizationResult(binaryMatrix: binaryMatrix, diarizationFrameRate: diarizationFrameRate)

        let transcription = [TranscriptionResult(
            text: "Hello there world",
            segments: [
                TranscriptionSegment(
                    id: 0, seek: 0, start: 0.5, end: 3.0,
                    text: "Hello there world", tokens: [],
                    tokenLogProbs: [[:]], temperature: 1.0,
                    avgLogprob: 0.0, compressionRatio: 1.0, noSpeechProb: 0.0,
                    words: [
                        WordTiming(word: "Hello ", tokens: [1], start: 0.5, end: 0.8, probability: 0.9),
                        WordTiming(word: "there ", tokens: [2], start: 0.9, end: 1.2, probability: 0.9),
                        WordTiming(word: "world", tokens: [3], start: 2.2, end: 2.8, probability: 0.9)
                    ]
                )
            ],
            language: "en",
            timings: TranscriptionTimings(),
            seekTime: nil
        )]

        let resultWithSpeakerInfo = result.addSpeakerInfo(to: transcription, strategy: .subsegment)
        XCTAssertEqual(resultWithSpeakerInfo.count, 1)
        XCTAssertEqual(resultWithSpeakerInfo[0].count, 2)

        XCTAssertEqual(resultWithSpeakerInfo[0][0].speakerWords.count, 2)
        XCTAssertEqual(resultWithSpeakerInfo[0][0].speaker, .speakerId(0))

        XCTAssertEqual(resultWithSpeakerInfo[0][1].speakerWords.count, 1)
        XCTAssertEqual(resultWithSpeakerInfo[0][1].speaker, .speakerId(1))
    }

    func testSubsegmentStrategyPicksMaxIntersection() {
        let diarizationFrameRate: Float = 10.0
        var binaryMatrix = [[Int]](repeating: [0], count: 2)
        binaryMatrix[0] = [Int](repeating: 0, count: 30)
        binaryMatrix[1] = [Int](repeating: 0, count: 30)

        for i in 0..<8 { binaryMatrix[0][i] = 1 }
        for i in 8..<30 { binaryMatrix[1][i] = 1 }

        let result = DiarizationResult(binaryMatrix: binaryMatrix, diarizationFrameRate: diarizationFrameRate)

        let transcription = [TranscriptionResult(
            text: "test word",
            segments: [
                TranscriptionSegment(
                    id: 0, seek: 0, start: 0.5, end: 1.2,
                    text: "test word", tokens: [],
                    tokenLogProbs: [[:]], temperature: 1.0,
                    avgLogprob: 0.0, compressionRatio: 1.0, noSpeechProb: 0.0,
                    words: [
                        WordTiming(word: "test ", tokens: [1], start: 0.5, end: 0.8, probability: 0.9),
                        WordTiming(word: "word", tokens: [2], start: 0.85, end: 1.2, probability: 0.9)
                    ]
                )
            ],
            language: "en",
            timings: TranscriptionTimings(),
            seekTime: nil
        )]

        let resultWithSpeakerInfo = result.addSpeakerInfo(to: transcription, strategy: .subsegment)
        XCTAssertEqual(resultWithSpeakerInfo[0].count, 1)
        XCTAssertEqual(resultWithSpeakerInfo[0][0].speaker, .speakerId(1))
    }

    func testSubsegmentStrategyFallsBackToPreviousSpeaker() {
        let diarizationFrameRate: Float = 10.0
        var binaryMatrix = [[Int]](repeating: [0], count: 1)
        binaryMatrix[0] = [Int](repeating: 0, count: 40)

        for i in 0..<10 { binaryMatrix[0][i] = 1 }

        let result = DiarizationResult(binaryMatrix: binaryMatrix, diarizationFrameRate: diarizationFrameRate)

        let transcription = [TranscriptionResult(
            text: "First gap second",
            segments: [
                TranscriptionSegment(
                    id: 0, seek: 0, start: 0.3, end: 3.0,
                    text: "First gap second", tokens: [],
                    tokenLogProbs: [[:]], temperature: 1.0,
                    avgLogprob: 0.0, compressionRatio: 1.0, noSpeechProb: 0.0,
                    words: [
                        WordTiming(word: "First ", tokens: [1], start: 0.3, end: 0.7, probability: 0.9),
                        WordTiming(word: "second", tokens: [2], start: 2.2, end: 2.8, probability: 0.9)
                    ]
                )
            ],
            language: "en",
            timings: TranscriptionTimings(),
            seekTime: nil
        )]

        let resultWithSpeakerInfo = result.addSpeakerInfo(to: transcription, strategy: .subsegment)
        XCTAssertEqual(resultWithSpeakerInfo[0].count, 2)
        XCTAssertEqual(resultWithSpeakerInfo[0][0].speaker, .speakerId(0))
        XCTAssertEqual(resultWithSpeakerInfo[0][1].speaker, .speakerId(0))
    }

    func testSubsegmentStrategyKeepsContiguousWordsTogether() {
        let diarizationFrameRate: Float = 10.0
        var binaryMatrix = [[Int]](repeating: [0], count: 1)
        binaryMatrix[0] = [Int](repeating: 0, count: 20)

        for i in 0..<20 { binaryMatrix[0][i] = 1 }

        let result = DiarizationResult(binaryMatrix: binaryMatrix, diarizationFrameRate: diarizationFrameRate)

        let transcription = [TranscriptionResult(
            text: "one two three four",
            segments: [
                TranscriptionSegment(
                    id: 0, seek: 0, start: 0.0, end: 2.0,
                    text: "one two three four", tokens: [],
                    tokenLogProbs: [[:]], temperature: 1.0,
                    avgLogprob: 0.0, compressionRatio: 1.0, noSpeechProb: 0.0,
                    words: [
                        WordTiming(word: "one ", tokens: [1], start: 0.0, end: 0.4, probability: 0.9),
                        WordTiming(word: "two ", tokens: [2], start: 0.45, end: 0.8, probability: 0.9),
                        WordTiming(word: "three ", tokens: [3], start: 0.9, end: 1.3, probability: 0.9),
                        WordTiming(word: "four", tokens: [4], start: 1.4, end: 1.8, probability: 0.9)
                    ]
                )
            ],
            language: "en",
            timings: TranscriptionTimings(),
            seekTime: nil
        )]

        let resultWithSpeakerInfo = result.addSpeakerInfo(to: transcription, strategy: .subsegment)
        XCTAssertEqual(resultWithSpeakerInfo[0].count, 1)
        XCTAssertEqual(resultWithSpeakerInfo[0][0].speakerWords.count, 4)
        XCTAssertEqual(resultWithSpeakerInfo[0][0].speaker, .speakerId(0))
    }
}
