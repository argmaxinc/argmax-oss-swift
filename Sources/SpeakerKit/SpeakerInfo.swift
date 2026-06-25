//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import WhisperKit

// MARK: - SpeakerInfoStrategy

public enum SpeakerInfoStrategy: Equatable, Hashable {
    case segment
    case subsegment(betweenWordThreshold: Float)

    /// Subsegment strategy with default threshold of 0.15 seconds
    public static var subsegment: SpeakerInfoStrategy {
        .subsegment(betweenWordThreshold: 0.15)
    }

    public init?(from string: String) {
        switch string.lowercased() {
        case "segment":
            self = .segment
        case "subsegment":
            self = .subsegment
        default:
            return nil
        }
    }
}

// MARK: - SpeakerInfo

public enum SpeakerInfo: Hashable, Codable, CustomStringConvertible, Sendable {
    case noMatch
    case multiple([Int])
    case speakerId(Int)

    /// Get single speaker's Cluster ID or nil if no match or multiple speakers
    public var speakerId: Int? {
        switch self {
        case .speakerId(let id):
            return id
        case .multiple, .noMatch:
            return nil
        }
    }

    /// Get all speaker IDs as an array. Empty if no match, single item for single speaker, multiple items for multiple speakers
    public var speakerIds: [Int] {
        switch self {
        case .speakerId(let id):
            return [id]
        case .multiple(let ids):
            return ids
        case .noMatch:
            return []
        }
    }

    public var description: String {
        switch self {
        case .noMatch:
            return "No Speaker Matched"
        case .multiple(let speakerIds):
            return "Multiple Speakers: \(speakerIds)"
        case .speakerId(let speakerId):
            return "Speaker \(speakerId)"
        }
    }
}

// MARK: - SpeakerWordTiming

/// A word timing with speaker information
public struct SpeakerWordTiming: Hashable, Codable, Sendable {
    public let wordTiming: WordTiming
    public let speaker: SpeakerInfo

    public init(wordTiming: WordTiming, speaker: SpeakerInfo) {
        self.wordTiming = wordTiming
        self.speaker = speaker
    }
}
