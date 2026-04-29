//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import os.lock

// MARK: - Concurrency Utilities

public struct ConcurrencyUtilities {
    private init() {}

    /// Number of active processors on this device.
    public static var activeProcessorCount: Int {
        ProcessInfo.processInfo.activeProcessorCount
    }
}

// MARK: - Unfair Lock

/// Thin wrapper around `os_unfair_lock` that exposes a Swift-friendly
/// `withLock` helper. This lock is non-reentrant and optimized for low
/// contention, matching the semantics of Core Foundation's unfair lock.
public final class UnfairLock: @unchecked Sendable {
    var lock = os_unfair_lock()

    public init() {}

    public func withLock<T>(_ body: () throws -> T) rethrows -> T {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return try body()
    }
}

// MARK: - Property Lock

/// Serializes whole-value reads and writes with an `os_unfair_lock`.
/// Useful for properties on types marked `@unchecked Sendable`.
///
/// **Known limitation — only whole-value replacement is safe:**
///
/// ```swift
/// holder.ref = otherRef        // safe: single locked write
/// holder.count = newCount      // safe: single locked write
///
/// holder.ref.count += 1        // NOT safe: gets the reference (locked),
///                              // then mutates .count with no lock held
/// holder.count += 1            // NOT safe: get and set are two separate
///                              // lock acquisitions with a gap between them
/// ```
///
/// For read-modify-write safety, callers must replace the entire value
/// atomically or use their own external synchronisation.
@propertyWrapper
public struct PropertyLock<Value: Codable & Sendable>: Sendable, Codable {
    private let lock: UnfairLock
    private var value: Value

    public init(wrappedValue: Value) {
        self.lock = UnfairLock()
        self.value = wrappedValue
    }

    public init(from decoder: Swift.Decoder) throws {
        self.lock = UnfairLock()
        self.value = try Value(from: decoder)
    }

    public func encode(to encoder: Encoder) throws {
        try lock.withLock {
            try value.encode(to: encoder)
        }
    }

    public var wrappedValue: Value {
        get {
            lock.withLock {
                return value
            }
        }
        set {
            lock.withLock {
                value = newValue
            }
        }
    }
}

// MARK: - Weak Sendable Wrapper

/// Wraps a weak reference so it can be captured in a `@Sendable` closure.
///
/// Use when a class instance is non-Sendable but you need to reference it
/// (typically the `[weak self]` pattern) inside a Sendable callback.
/// The caller is responsible for ensuring no actual data race occurs at access sites.
public final class WeakSendableWrapper<T: AnyObject>: @unchecked Sendable {
    public weak var value: T?
    public init(_ value: T?) { self.value = value }
}

// MARK: - Early Stop Actor

/// An actor that provides thread-safe early stopping functionality using UUIDs as keys.
public actor EarlyStopActor {
    private var shouldStop = [UUID: Bool]()

    public init() {}

    /// Sets the stop flag for a given UUID
    public func set(_ value: Bool, for uuid: UUID) {
        shouldStop[uuid] = value
    }

    /// Gets the stop flag for a given UUID
    public func get(for uuid: UUID) -> Bool {
        return shouldStop[uuid] ?? false
    }

    /// Removes and returns the stop flag for a given UUID
    @discardableResult
    public func remove(for uuid: UUID) -> Bool? {
        return shouldStop.removeValue(forKey: uuid)
    }
}
