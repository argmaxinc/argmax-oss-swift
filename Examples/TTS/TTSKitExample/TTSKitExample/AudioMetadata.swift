//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import AVFoundation
import Foundation

/// Metadata for a single TTS generation, embedded inside its `.m4a` file.
///
/// The full struct is serialized as JSON and stored in the iTunes `©cmt` atom so
/// the complete generation context is recoverable from the file alone.
/// Human-readable fields (title, artist, album) are also embedded
struct AudioMetadata: Codable, Sendable {
    static let metadataTitleMaxLength = 80
    let id: UUID
    let text: String
    let speaker: String
    let language: String
    let instruction: String
    let modelName: String
    let realTimeFactor: Double
    let speedFactor: Double
    let stepsPerSecond: Double
    let timeToFirstBuffer: TimeInterval
    let date: Date

    init(
        id: UUID = UUID(),
        text: String,
        speaker: String,
        language: String,
        instruction: String,
        modelName: String,
        realTimeFactor: Double,
        speedFactor: Double,
        stepsPerSecond: Double,
        timeToFirstBuffer: TimeInterval,
        date: Date = Date()
    ) {
        self.id = id
        self.text = text
        self.speaker = speaker
        self.language = language
        self.instruction = instruction
        self.modelName = modelName
        self.realTimeFactor = realTimeFactor
        self.speedFactor = speedFactor
        self.stepsPerSecond = stepsPerSecond
        self.timeToFirstBuffer = timeToFirstBuffer
        self.date = date
    }

    // MARK: - Filename

    private static let filenameDateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        return f
    }()

    var suggestedFileName: String {
        let slug = modelName
            .lowercased()
            .replacingOccurrences(of: " ", with: "-")
        return "\(speaker)_\(slug)_\(Self.filenameDateFormatter.string(from: date))"
    }

    // MARK: - AVFoundation metadata

    /// Build the `[AVMetadataItem]` array for `AVAssetExportSession`.
    ///
    /// All fields use `commonIdentifier` variants so they work across every
    /// container format AVFoundation supports (M4A, MOV, MP3, AIFF...).
    ///
    /// Fields verified to survive `AVAssetExportSession` M4A export (confirmed via ffprobe):
    /// - `©nam` title    - first 80 chars of the generated text
    /// - `©ART` artist   - speaker name
    /// - `©alb` album    - model (e.g. "qwen3_tts_12hz-1.7b-customvoice")
    /// - `©lyr` lyrics   - full untruncated text
    /// - `©too` encoder  - "TTSKit v{appVersion}"  (shows as `encoder` in ffprobe)
    /// - `©cmt` comment  - JSON blob; the app reads this back to reconstruct the generation
    ///
    /// `.commonIdentifierCreator` and `.commonIdentifierDescription` were tried but do not
    /// survive M4A export - iTunes-specific atoms are used for those two slots instead.
    func avMetadataItems() throws -> [AVMetadataItem] {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let json = try String(data: encoder.encode(self), encoding: .utf8) ?? ""

        let appVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
        let maxLen = Self.metadataTitleMaxLength
        let title = text.count > maxLen ? String(text.prefix(maxLen - 3)) + "..." : text
        let langTag = Self.bcp47Tag(for: language)

        func item(_ identifier: AVMetadataIdentifier, _ value: String, lang: String = "und") -> AVMetadataItem {
            let i = AVMutableMetadataItem()
            i.identifier = identifier
            i.value = value as NSString
            i.extendedLanguageTag = lang
            return i
        }

        return [
            item(.commonIdentifierTitle, title, lang: langTag),
            item(.commonIdentifierArtist, speaker.capitalized),
            item(.commonIdentifierAlbumName, modelName),
            item(.iTunesMetadataLyrics, text, lang: langTag),
            item(.iTunesMetadataEncodingTool, "TTSKit v\(appVersion)"),
            item(.iTunesMetadataUserComment, json),
        ]
    }

    private static func bcp47Tag(for language: String) -> String {
        switch language.lowercased() {
            case "english": return "en"
            case "chinese": return "zh"
            case "japanese": return "ja"
            case "korean": return "ko"
            case "german": return "de"
            case "french": return "fr"
            case "russian": return "ru"
            case "portuguese": return "pt"
            case "spanish": return "es"
            case "italian": return "it"
            default: return "und"
        }
    }

    // MARK: - Loading from file

    /// Reconstruct metadata from the `©cmt` atom of an `.m4a` file.
    /// Returns `nil` if the file has no TTSKit metadata.
    static func load(from url: URL) async throws -> AudioMetadata? {
        let asset = AVURLAsset(url: url)
        let items = try await asset.load(.metadata)
        guard let commentItem = items.first(where: {
            $0.identifier == .iTunesMetadataUserComment
        }),
            let json = try await commentItem.load(.stringValue),
            let data = json.data(using: .utf8) else { return nil }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(AudioMetadata.self, from: data)
    }
}
