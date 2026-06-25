//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

@frozen
public enum ArgmaxCoreError: Error, LocalizedError {
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message):
            return message
        }
    }
}
