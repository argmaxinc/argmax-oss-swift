//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

@frozen
public enum WhisperError: Error, LocalizedError, Equatable {
    case tokenizerUnavailable(String = "Tokenizer is unavailable")
    case modelsUnavailable(String = "Models are unavailable")
    case prefillFailed(String = "Prefill failed")
    case audioProcessingFailed(String = "Audio processing failed")
    case decodingLogitsFailed(String = "Unable to decode logits from the model output")
    case segmentingFailed(String = "Creating segments failed")
    case loadAudioFailed(String = "Load audio failed")
    case prepareDecoderInputsFailed(String = "Prepare decoder inputs failed")
    case transcriptionFailed(String = "Transcription failed")
    case decodingFailed(String = "Decoding failed")
    case microphoneUnavailable(String = "No available microphone to record or stream")
    case initializationError(String = "Error initializing WhisperKit")

    public var errorDescription: String? {
        switch self {
        case let .tokenizerUnavailable(message),
            let .modelsUnavailable(message),
            let .prefillFailed(message),
            let .audioProcessingFailed(message),
            let .decodingLogitsFailed(message),
            let .segmentingFailed(message),
            let .loadAudioFailed(message),
            let .prepareDecoderInputsFailed(message),
            let .transcriptionFailed(message),
            let .decodingFailed(message),
            let .microphoneUnavailable(message),
            let .initializationError(message):
            Logging.error(message)
            return message
        }
    }
}
