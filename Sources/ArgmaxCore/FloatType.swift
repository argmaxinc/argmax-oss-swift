//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import CoreML

// MARK: - Platform-aware Float Type

#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
public typealias FloatType = Float16
#else
public typealias FloatType = Float
#endif

#if (os(macOS) || targetEnvironment(macCatalyst)) && arch(arm64)

// MARK: - Float16 BNNSScalar conformance for macOS
#if os(macOS)

extension Float16: @retroactive BNNSScalar {
    public static var bnnsDataType: BNNSDataType { .float16 }
}

extension Float16: @retroactive MLShapedArrayScalar {
    public static var multiArrayDataType: MLMultiArrayDataType { .float16 }
}

#endif

// MARK: - Float16 BNNSScalar conformance for Mac Catalyst
#if targetEnvironment(macCatalyst)

// Catalyst uses its own availability domain.
@available(macCatalyst, obsoleted: 26.0)
extension Float16: @retroactive BNNSScalar {
    public static var bnnsDataType: BNNSDataType { .float16 }
}

extension Float16: @retroactive MLShapedArrayScalar {
    public static var multiArrayDataType: MLMultiArrayDataType { .float16 }
}

#endif

#endif
