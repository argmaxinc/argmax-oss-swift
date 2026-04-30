//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import ArgmaxCore

/// Base configuration for SpeakerKit initialization
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
open class SpeakerKitConfig: @unchecked Sendable {
    /// Custom diarizer instance for advanced use cases or Pro variants.
    /// When provided, SpeakerKit uses this instead of creating a new one.
    public var diarizer: (any Diarizer)?

    /// Model download configuration
    public var modelDownloadConfig: ModelDownloadConfig

    /// Enable extra verbosity for logging. When `true`, activates pipeline output at `logLevel`.
    public var verbose: Bool

    /// Log level used when `verbose` is `true`. Defaults to `.info`.
    public var logLevel: Logging.LogLevel

    /// Whether to download models during initialization.
    public var download: Bool

    /// Whether to load models into memory during initialization
    public var load: Bool

    public init(
        modelDownloadConfig: ModelDownloadConfig? = nil,
        download: Bool = true,
        verbose: Bool = true,
        logLevel: Logging.LogLevel = .info,
        load: Bool = false,
        diarizer: (any Diarizer)? = nil
    ) {
        self.modelDownloadConfig = modelDownloadConfig ?? ModelDownloadConfig(modelRepo: "argmaxinc/speakerkit-coreml")
        self.download = download
        self.verbose = verbose
        self.logLevel = logLevel
        self.load = load
        self.diarizer = diarizer
    }
}
