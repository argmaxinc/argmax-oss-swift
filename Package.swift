// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription
import Foundation

let package = Package(
    name: "argmax-oss-swift",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
        .watchOS(.v10),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "ArgmaxOSS",
            targets: ["ArgmaxOSS"]
        ),
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
            name: "argmax-cli",
            targets: ["ArgmaxCLI"]
        ),
        .executable(
            name: "whisperkit-cli",
            targets: ["ArgmaxCLI"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ] + (isServerEnabled() ? [
        .package(url: "https://github.com/vapor/vapor.git", from: "4.115.1"),
        .package(url: "https://github.com/apple/swift-openapi-generator", from: "1.10.2"),
        .package(url: "https://github.com/apple/swift-openapi-runtime", from: "1.8.2"),
        .package(url: "https://github.com/swift-server/swift-openapi-vapor", from: "1.0.1"),

    ] : []),
    targets: [
        .target(
            name: "ArgmaxOSS",
            dependencies: [
                "ArgmaxCore",
                "WhisperKit",
                "TTSKit",
                "SpeakerKit",
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "ArgmaxCore",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "WhisperKit",
            dependencies: [
                "ArgmaxCore",
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "TTSKit",
            dependencies: [
                "ArgmaxCore",
            ],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .target(
            name: "SpeakerKit",
            dependencies: [
                "ArgmaxCore",
                "WhisperKit",
            ],
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
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
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "TTSKitTests",
            dependencies: [
                "TTSKit"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
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
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "ArgmaxCLI",
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
            path: "Sources/ArgmaxCLI",
            exclude: (isServerEnabled() ? [] : ["Server"]),
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
            ] + (isServerEnabled() ? [.define("BUILD_SERVER_CLI")] : [])
        )
    ],
    swiftLanguageVersions: [.v5]
)

func isServerEnabled() -> Bool {
    if let enabledValue = Context.environment["BUILD_ALL"] {
        return enabledValue.lowercased() == "true" || enabledValue == "1"
    }

    // Default disabled, change to true temporarily for local development
    return false
}
