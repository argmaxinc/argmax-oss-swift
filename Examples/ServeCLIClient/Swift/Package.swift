// swift-tools-version: 5.9
import PackageDescription

let package = Package(
	name: "WhisperKitSwiftClient",
	platforms: [
		.macOS(.v13)
	],
	products: [
		.executable(name: "whisperkit-client", targets: ["WhisperKitSwiftClient"]),
	],
	dependencies: [
		.package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),
		.package(url: "https://github.com/apple/swift-openapi-runtime", from: "1.0.0"),
		.package(url: "https://github.com/apple/swift-openapi-urlsession", from: "1.0.0"),
		.package(url: "https://github.com/apple/swift-http-types", from: "1.0.0"),
		.package(url: "https://github.com/apple/swift-openapi-generator", from: "1.0.0"),
	],
	targets: [
		.executableTarget(
			name: "WhisperKitSwiftClient",
			dependencies: [
				.product(name: "ArgumentParser", package: "swift-argument-parser"),
				.product(name: "OpenAPIRuntime", package: "swift-openapi-runtime"),
				.product(name: "OpenAPIURLSession", package: "swift-openapi-urlsession"),
				.product(name: "HTTPTypes", package: "swift-http-types"),
			],
			path: "Sources/WhisperKitSwiftClient"
		)
	]
)
