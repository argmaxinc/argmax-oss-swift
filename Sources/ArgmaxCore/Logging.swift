//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import OSLog

/// Shared logger for all Argmax frameworks (WhisperKit, TTSKit, etc.).
///
/// Configure the log level once at startup:
/// ```swift
/// Logging.shared.logLevel = .debug
/// ```
/// or via a config object, following the WhisperKit pattern:
/// ```swift
/// Logging.shared.logLevel = config.verbose ? config.logLevel : .none
/// ```
/// Thread-safe by construction: all mutable state lives inside `OSAllocatedUnfairLock`
/// (which is `Sendable` on its own), and every other stored property is a
/// `let`-bound value whose type is itself `Sendable`. That combination lets the
/// class conform to `Sendable` without `@unchecked`.
public final class Logging: Sendable {

    // MARK: - Helper Types

    public typealias LoggingCallback = @Sendable (_ message: String) -> Void

    /// Represents the severity threshold for emitting log messages.
    ///
    /// The `LogLevel` controls which messages are allowed to be logged. Messages are
    /// emitted when their severity is greater than or equal to the globally configured
    /// log level. For example, if the global level is set to `.info`, then `.info` and
    /// `.error` messages will be logged, while `.debug` messages will be suppressed.
    ///
    /// Ordering (from least to most severe):
    /// - `.debug` (1): Verbose diagnostic information useful during development.
    /// - `.info`  (2): High-level informational messages about app flow or state.
    /// - `.error` (3): Errors and failures that require attention.
    /// - `.none`  (4): Disables all logging.
    @frozen
    public enum LogLevel: Int, Comparable, Sendable {
        case debug = 1
        case info = 2
        case error = 3
        case none = 4

        var osLogType: OSLogType {
            switch self {
            case .debug: return .debug
            case .info: return .info
            case .error: return .error
            case .none: return .default
            }
        }

        public static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
            lhs.rawValue < rhs.rawValue
        }
    }

    private struct State {
        var logLevel: LogLevel = .none
        var loggingCallback: LoggingCallback?
    }

    // All mutable state is guarded by `lock` (itself `Sendable`);
    // every other stored property is a `let`-bound `Sendable` value.
    private let lock = OSAllocatedUnfairLock<State>(initialState: State())
    private let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "com.argmax.argmaxcore",
        category: "Argmax"
    )

    /// Shared singleton logger.
    public static let shared = Logging()

    private init() {}

    // MARK: - Configuration

    /// The current global logging level.
    ///
    /// Thread-safe; reads and writes are serialized by an internal lock.
    public var logLevel: LogLevel {
        get { lock.withLock { $0.logLevel } }
        set { lock.withLock { $0.logLevel = newValue } }
    }

    /// Optional callback to intercept log messages.
    ///
    /// When non-`nil`, messages are forwarded here instead of the system logger.
    /// Thread-safe; reads and writes are serialized by an internal lock.
    public var loggingCallback: LoggingCallback? {
        get { lock.withLock { $0.loggingCallback } }
        set { lock.withLock { $0.loggingCallback = newValue } }
    }

    /// Whether logging is enabled for any level other than `.none`.
    public var isLoggingEnabled: Bool {
        logLevel != .none
    }

    /// Whether logging is enabled for a specific level.
    public func isLoggingEnabled(for level: LogLevel) -> Bool {
        let current = logLevel
        return current != .none && current <= level
    }

    // MARK: - Core logging

    /// Logs a message at the specified `OSLogType`.
    ///
    /// Respects the current `logLevel` and routes through `loggingCallback` if set.
    public func log(_ items: Any..., separator: String = " ", terminator: String? = nil, type: OSLogType) {
        let (level, callback) = lock.withLock { ($0.logLevel, $0.loggingCallback) }
        let messageLevel = LogLevel(osLogType: type)
        guard level != .none, level <= messageLevel else { return }
        var message = items.map { "\($0)" }.joined(separator: separator)
        if let terminator {
            message += terminator
        }
        if let callback {
            callback(message)
        } else {
            logger.log(level: type, "\(message, privacy: .public)")
        }
    }

    // MARK: - Static convenience

    /// Whether logging is enabled on the shared logger.
    public static var isLoggingEnabled: Bool {
        shared.isLoggingEnabled
    }

    /// Whether logging is enabled at the given level on the shared logger.
    public static func isLoggingEnabled(for level: LogLevel) -> Bool {
        shared.isLoggingEnabled(for: level)
    }

    public static func debug(_ items: Any..., separator: String = " ", terminator: String? = nil) {
        shared.log(items, separator: separator, terminator: terminator, type: .debug)
    }

    public static func info(_ items: Any..., separator: String = " ", terminator: String? = nil) {
        shared.log(items, separator: separator, terminator: terminator, type: .info)
    }

    public static func error(_ items: Any..., separator: String = " ", terminator: String? = nil) {
        shared.log(items, separator: separator, terminator: terminator, type: .error)
    }
}

private extension Logging.LogLevel {
    init(osLogType: OSLogType) {
        switch osLogType {
        case .debug: self = .debug
        case .info: self = .info
        case .error, .fault: self = .error
        default: self = .info
        }
    }
}

// MARK: - Static mutators

public extension Logging {
    /// Update the global log level. Thread-safe.
    static func updateLogLevel(_ level: LogLevel) {
        shared.logLevel = level
    }

    /// Update the logging callback. Thread-safe.
    static func updateCallback(_ callback: LoggingCallback?) {
        shared.loggingCallback = callback
    }
}

// MARK: - Memory Usage

public extension Logging {
    static func logCurrentMemoryUsage(_ message: String) {
        let memoryUsage = getMemoryUsage()
        Logging.debug("\(message) - Memory usage: \(memoryUsage) MB")
    }

    static func getMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        guard kerr == KERN_SUCCESS else {
            return 0 // If the call fails, return 0
        }

        return info.resident_size / 1024 / 1024 // Convert to MB
    }
}

// MARK: - Formatting

public extension Logging {
    /// Format a timing entry as a human-readable string with per-run average and percentage.
    ///
    /// Output format: `  123.45 ms /    100 runs (   1.23 ms/run) 45.67%`
    ///
    /// - Parameters:
    ///   - time: Duration in seconds.
    ///   - runs: Number of calls / iterations (used for per-run average).
    ///   - fullPipelineDuration: Total pipeline duration in **milliseconds** (for percentage).
    static func formatTimeWithPercentage(_ time: Double, _ runs: Double, _ fullPipelineDuration: Double) -> String {
        let percentage = (time * 1000 / fullPipelineDuration) * 100
        let runTime = runs > 0 ? time * 1000 / Double(runs) : 0
        return String(format: "%8.2f ms / %6.0f runs (%8.2f ms/run) %5.2f%%", time * 1000, runs, runTime, percentage)
    }
}

@available(*, deprecated, message: "Subject to removal in a future version. Use `Logging.logCurrentMemoryUsage(_:)` instead.")
public func logCurrentMemoryUsage(_ message: String) {
    Logging.logCurrentMemoryUsage(message)
}

@available(*, deprecated, message: "Subject to removal in a future version. Use `Logging.getMemoryUsage()` instead.")
public func getMemoryUsage() -> UInt64 {
    return Logging.getMemoryUsage()
}
