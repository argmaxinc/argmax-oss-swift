//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import Foundation

// MARK: - Qwen3 Voice Pack

/// A cloned-voice enrollment artifact for the Qwen3-TTS **Base** model variant.
///
/// Produced server-side by the enrollment pipeline (reference wav → speaker
/// encoder): the x-vector conditions the talker on the target voice by being
/// injected directly as a prompt embedding at the speaker slot (the Base
/// variant's speaker-encoder `enc_dim` equals the talker hidden size, so no
/// projection is needed).
///
/// `refCodes`/`refText` carry the optional in-context-learning (ICL) reference
/// for higher-fidelity cloning; they are stored in the pack format now so packs
/// don't need regenerating when the ICL prompt path lands.
public struct Qwen3VoicePack: Codable, Sendable {
    /// Embedding dimension the x-vector must match (talker hidden size).
    public static let xVectorDim = 1024
    /// Number of RVQ code groups per frame in `refCodes`.
    public static let codeGroups = 16

    /// Display / lookup name of the cloned voice.
    public var name: String

    /// 1024-dim speaker embedding from the Base model's speaker encoder.
    /// Injected untransformed as the speaker-slot prompt embedding.
    public var xVector: [Float]

    /// Optional reference codec frames (each `codeGroups` codes) for ICL mode.
    public var refCodes: [[Int32]]?

    /// Exact transcript of the reference audio (required for ICL mode —
    /// a wrong transcript poisons the conditioning).
    public var refText: String?

    public init(name: String, xVector: [Float], refCodes: [[Int32]]? = nil, refText: String? = nil) {
        self.name = name
        self.xVector = xVector
        self.refCodes = refCodes
        self.refText = refText
    }

    /// Load a voice pack from a JSON file and validate its dimensions.
    public static func load(from url: URL) throws -> Qwen3VoicePack {
        let data = try Data(contentsOf: url)
        let pack = try JSONDecoder().decode(Qwen3VoicePack.self, from: data)
        try pack.validate()
        return pack
    }

    /// Validate embedding and code-frame dimensions.
    public func validate() throws {
        guard xVector.count == Self.xVectorDim else {
            throw TTSError.generationFailed(
                "Voice pack '\(name)': xVector has \(xVector.count) dims, expected \(Self.xVectorDim)")
        }
        if let refCodes {
            for (index, frame) in refCodes.enumerated() where frame.count != Self.codeGroups {
                throw TTSError.generationFailed(
                    "Voice pack '\(name)': refCodes[\(index)] has \(frame.count) codes, expected \(Self.codeGroups)")
            }
        }
    }

    /// The x-vector converted to the embed element type used by the pipeline.
    public var speakerEmbed: [FloatType] {
        xVector.map { FloatType($0) }
    }
}
