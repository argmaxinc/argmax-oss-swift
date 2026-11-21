//  Copyright © 2025 Argmax, Inc. All rights reserved.
//  For licensing see accompanying LICENSE.md file.

import Foundation
import HTTPTypes
import OpenAPIRuntime

#if compiler(>=6.2)
typealias ServerMiddlewareNext = @concurrent @Sendable (HTTPRequest, HTTPBody?, ServerRequestMetadata) async throws -> (HTTPResponse, HTTPBody?)
#else
typealias ServerMiddlewareNext = @Sendable (HTTPRequest, HTTPBody?, ServerRequestMetadata) async throws -> (HTTPResponse, HTTPBody?)
#endif

/// Bounds the request body of the OpenAPI routes.
///
/// The size limit counts the bytes of the whole body, so it also covers chunked requests without
/// `Content-Length` and requests with several file parts. Bodies above the limit fail with 413,
/// bodies that send nothing for `readTimeout` fail with 408.
struct RequestBodyLimitMiddleware: ServerMiddleware {
    let maxBytes: Int
    let readTimeout: Duration?

    func intercept(
        _ request: HTTPRequest,
        body: HTTPBody?,
        metadata: ServerRequestMetadata,
        operationID: String,
        next: ServerMiddlewareNext
    ) async throws -> (HTTPResponse, HTTPBody?) {
        guard let body else {
            return try await next(request, nil, metadata)
        }
        if case .known(let length) = body.length, length > maxBytes {
            return (HTTPResponse(status: .contentTooLarge), nil)
        }

        let counter = ByteCounter(limit: maxBytes)
        var limited = HTTPBody(
            body.map { chunk in try counter.add(chunk) },
            length: body.length,
            iterationBehavior: body.iterationBehavior
        )
        if let readTimeout {
            limited = HTTPBody(
                ReadTimeoutSequence(base: limited, timeout: readTimeout),
                length: body.length,
                iterationBehavior: body.iterationBehavior
            )
        }

        do {
            return try await next(request, limited, metadata)
        } catch {
            switch (error as? ServerError)?.underlyingError ?? error {
            case is RequestBodyTooLargeError:
                return (HTTPResponse(status: .contentTooLarge), nil)
            case is RequestBodyTimeoutError:
                return (HTTPResponse(status: .requestTimeout), nil)
            default:
                throw error
            }
        }
    }
}

struct RequestBodyTooLargeError: Error {}

struct RequestBodyTimeoutError: Error {}

private final class ByteCounter: @unchecked Sendable {
    private let lock = NSLock()
    private let limit: Int
    private var total = 0

    init(limit: Int) {
        self.limit = limit
    }

    func add(_ chunk: ArraySlice<UInt8>) throws -> ArraySlice<UInt8> {
        lock.lock()
        defer { lock.unlock() }
        total += chunk.count
        guard total <= limit else { throw RequestBodyTooLargeError() }
        return chunk
    }
}

/// Fails with `RequestBodyTimeoutError` when a single read waits longer than `timeout`.
///
/// Unlike Vapor's `idleTimeout`, this only runs while the body is being read, so long
/// non-streaming transcriptions are not affected.
private struct ReadTimeoutSequence: AsyncSequence, Sendable {
    typealias Element = ArraySlice<UInt8>

    let base: HTTPBody
    let timeout: Duration

    func makeAsyncIterator() -> Iterator {
        Iterator(reader: Reader(base.makeAsyncIterator()), timeout: timeout)
    }

    struct Iterator: AsyncIteratorProtocol {
        let reader: Reader
        let timeout: Duration

        mutating func next() async throws -> ArraySlice<UInt8>? {
            let reader = reader
            let timeout = timeout
            return try await withThrowingTaskGroup(of: ArraySlice<UInt8>?.self) { group in
                group.addTask { try await reader.next() }
                group.addTask {
                    try await Task.sleep(for: timeout)
                    throw RequestBodyTimeoutError()
                }
                defer { group.cancelAll() }
                let chunk = try await group.next()
                return chunk ?? nil
            }
        }
    }

    // Only one read is in flight at a time, so the iterator is never accessed concurrently.
    final class Reader: @unchecked Sendable {
        private var base: HTTPBody.Iterator

        init(_ base: HTTPBody.Iterator) {
            self.base = base
        }

        func next() async throws -> ArraySlice<UInt8>? {
            try await base.next()
        }
    }
}
