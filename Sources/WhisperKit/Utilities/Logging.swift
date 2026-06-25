//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgmaxCore
import OSLog

// MARK: - WhisperKit signpost categories

extension Logging {
    enum AudioEncoding {
        static let logger = Logger(
            subsystem: Constants.Logging.subsystem,
            category: "AudioEncoding"
        )
        static let signposter = OSSignposter(logger: logger)
    }

    enum FeatureExtractor {
        static let logger = Logger(
            subsystem: Constants.Logging.subsystem,
            category: "FeatureExtractor"
        )
        static let signposter = OSSignposter(logger: logger)
    }

    enum TranscribeTask {
        static let logger = Logger(
            subsystem: Constants.Logging.subsystem,
            category: "TranscribeTask"
        )
        static let signposter = OSSignposter(logger: logger)
    }

    static func beginSignpost(
        _ intervalName: StaticString,
        signposter: OSSignposter
    ) -> OSSignpostIntervalState {
        let signpostId = signposter.makeSignpostID()
        return signposter.beginInterval(intervalName, id: signpostId)
    }

    static func endSignpost(
        _ intervalName: StaticString,
        interval: OSSignpostIntervalState,
        signposter: OSSignposter
    ) {
        signposter.endInterval(intervalName, interval)
    }

    static func formatTimestamp(_ timestamp: Float) -> String {
        String(format: "%.2f", timestamp)
    }

// formatTimeWithPercentage is defined in ArgmaxCore/Logging.swift and re-exported here.
}
