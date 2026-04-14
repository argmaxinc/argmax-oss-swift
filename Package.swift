// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription
import Foundation

let approachableConcurrencySettings: [SwiftSetting] = [
    .enableUpcomingFeature("InferIsolatedConformances"),
    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
]

let package = Package(
    name: "whisperkit",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
        .watchOS(.v10),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "WhisperKit",
            targets: ["WhisperKit"]
        ),
        .library(
            name: "TTSKit",
            targets: ["TTSKit"]
        ),
        .library(
            name: "SpeakerKit",
            targets: ["SpeakerKit"]
        ),
        .executable(
            name: "whisperkit-cli",
            targets: ["WhisperKitCLI"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.7.0"),
    ] + (isServerEnabled() ? [
        .package(url: "https://github.com/vapor/vapor.git", from: "4.115.1"),
        .package(url: "https://github.com/apple/swift-openapi-generator", from: "1.10.2"),
        .package(url: "https://github.com/apple/swift-openapi-runtime", from: "1.8.2"),
        .package(url: "https://github.com/swift-server/swift-openapi-vapor", from: "1.0.1"),
    ] : []),
    targets: [
        .target(
            name: "ArgmaxCore",
            swiftSettings: approachableConcurrencySettings
        ),
        .target(
            name: "WhisperKit",
            dependencies: [
                "ArgmaxCore",
            ],
            swiftSettings: approachableConcurrencySettings
        ),
        .target(
            name: "TTSKit",
            dependencies: [
                "ArgmaxCore",
            ],
            swiftSettings: approachableConcurrencySettings
        ),
        .target(
            name: "SpeakerKit",
            dependencies: [
                "ArgmaxCore",
                "WhisperKit",
            ],
            swiftSettings: approachableConcurrencySettings
        ),
        .testTarget(
            name: "WhisperKitTests",
            dependencies: [
                "WhisperKit",
            ],
            exclude: ["UnitTestsPlan.xctestplan"],
            resources: [
                .process("Resources"),
            ],
            swiftSettings: approachableConcurrencySettings
        ),
        .testTarget(
            name: "TTSKitTests",
            dependencies: [
                "TTSKit"
            ],
            swiftSettings: approachableConcurrencySettings
        ),
        .testTarget(
            name: "SpeakerKitTests",
            dependencies: [
                "SpeakerKit",
                "WhisperKit",
            ],
            resources: [
                .process("Resources"),
            ],
            swiftSettings: approachableConcurrencySettings
        ),
        .executableTarget(
            name: "WhisperKitCLI",
            dependencies: [
                "WhisperKit",
                "TTSKit",
                "SpeakerKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ] + (isServerEnabled() ? [
                .product(name: "Vapor", package: "vapor"),
                .product(name: "OpenAPIRuntime", package: "swift-openapi-runtime"),
                .product(name: "OpenAPIVapor", package: "swift-openapi-vapor"),
            ] : []),
            exclude: (isServerEnabled() ? [] : ["Server"]),
            swiftSettings: approachableConcurrencySettings + (isServerEnabled() ? [.define("BUILD_SERVER_CLI")] : [])
        )
    ],
    swiftLanguageModes: [.v5]
)

func isServerEnabled() -> Bool {
    if let enabledValue = Context.environment["BUILD_ALL"] {
        return enabledValue.lowercased() == "true" || enabledValue == "1"
    }

    // Default disabled, change to true temporarily for local development
    return false
}
