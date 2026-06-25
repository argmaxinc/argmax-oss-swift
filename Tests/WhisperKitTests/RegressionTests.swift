//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
import UniformTypeIdentifiers
import WhisperKit
import XCTest

#if os(watchOS)
import WatchKit
#endif

class RegressionTests: XCTestCase {
    var audioFileURLs: [URL]?
    var remoteFileURLs: [URL]?
    var metadataURL: URL?
    var testWERURLs: [URL]?
    var modelsToTest: [String] = []
    var modelReposToTest: [String] = []
    var modelsTested: [String] = []
    var modelReposTested: [String] = []
    var optionsToTest: [DecodingOptions] = [DecodingOptions()]

    struct TestConfig {
        let dataset: String
        let modelComputeOptions: ModelComputeOptions
        var model: String
        var modelRepo: String
        let decodingOptions: DecodingOptions
    }

    /// Located on HF https://huggingface.co/datasets/argmaxinc/whisperkit-test-data/tree/main
    let datasetRepo = "argmaxinc/whisperkit-test-data"
    var datasets = ["librispeech-10mins", "earnings22-10mins"]
    let debugDataset = ["earnings22-10mins"]
    let debugModels = ["tiny"]
    let debugRepos = ["argmaxinc/whisperkit-coreml"]

    var computeOptions: [ModelComputeOptions] = [
        ModelComputeOptions(audioEncoderCompute: .cpuAndNeuralEngine, textDecoderCompute: .cpuAndNeuralEngine),
    ]

    let defaultDecodingOptions = DecodingOptions(
        verbose: true,
        task: .transcribe
    )

    let vadDecodingOptions = DecodingOptions(
        verbose: true,
        task: .transcribe,
        chunkingStrategy: .vad
    )

    override func setUp() {
        super.setUp()
        #if canImport(UIApplication)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(didReceiveMemoryWarning),
            name: UIApplication.didReceiveMemoryWarningNotification,
            object: nil
        )
        #endif
    }

    @objc func didReceiveMemoryWarning() {
        Logging.debug("Received memory warning")

        // TODO: Record this data in the test results
        let maxMemory = SystemMemoryCheckerAdvanced.getMemoryUsage()
        Logging.debug("Max memory before warning: \(maxMemory)")
    }

    class func getModelToken() -> String? {
        // Add token here or override
        return nil
    }

    func testEnvConfigurations(defaultModels: [String]? = nil, defaultRepos: [String]? = nil) {
        if let modelSizeEnv = ProcessInfo.processInfo.environment["MODEL_NAME"], !modelSizeEnv.isEmpty {
            modelsToTest = [modelSizeEnv]
            Logging.debug("Model size: \(modelSizeEnv)")

            if let repoEnv = ProcessInfo.processInfo.environment["MODEL_REPO"] {
                modelReposToTest = [repoEnv]
                Logging.debug("Using repo: \(repoEnv)")
            }

            XCTAssertTrue(modelsToTest.count > 0, "Invalid model size: \(modelSizeEnv)")

            if modelSizeEnv == "crash_test" {
                fatalError("Crash test triggered")
            }
        } else {
            modelsToTest = defaultModels ?? debugModels
            modelReposToTest = defaultRepos ?? debugRepos
            Logging.debug("Model size not set by env")
        }
    }

    func testModelPerformanceWithDebugConfig() async throws {
        testEnvConfigurations()

        // Debug test matrix
        datasets = debugDataset
        optionsToTest = [vadDecodingOptions]
        computeOptions = [computeOptions.first!]

        let debugTestMatrix: [TestConfig] = getTestMatrix()
        Logging.debug("Running \(debugTestMatrix.count) regression tests for models: \(modelsToTest)")

        // Run the tests
        try await runRegressionTests(with: debugTestMatrix)
    }

    func testModelPerformance() async throws {
        testEnvConfigurations(defaultModels: WhisperKit.recommendedModels().supported)

        // Setup test matrix
        optionsToTest = [vadDecodingOptions]
        computeOptions = [computeOptions.first!]

        let testMatrix: [TestConfig] = getTestMatrix()
        Logging.debug("Running \(testMatrix.count) regression tests for models: \(modelsToTest)")

        // Run the tests
        try await runRegressionTests(with: testMatrix)
    }

    // MARK: - Test Pipeline

    public func runRegressionTests(with testMatrix: [TestConfig]) async throws {
        var failureInfo: [String: String] = [:]
        var attachments: [String: String] = [:]
        let device = getCurrentDevice()
        for (i, config) in testMatrix.enumerated() {
            do {
                Logging.debug("Running test \(i + 1)/\(testMatrix.count) for \(config.model) with \(config.dataset) on \(device) using encoder compute: \(config.modelComputeOptions.audioEncoderCompute.description) and decoder compute: \(config.modelComputeOptions.textDecoderCompute.description)")
                try await downloadTestData(forDataset: config.dataset)
                let attachmentName = try await testAndMeasureModelPerformance(config: config, device: device)
                attachments[config.dataset] = attachmentName
                try await Task.sleep(nanoseconds: 1_000_000_000)
            } catch {
                Logging.debug("Failed to test \(config.model): \(error)")
                failureInfo[config.model] = error.localizedDescription
            }
        }

        // Save summary
        saveSummary(failureInfo: failureInfo, attachments: attachments)
    }

    func testAndMeasureModelPerformance(config: TestConfig, device: String) async throws -> String? {
        var config = config
        var resultJSON: [RegressionStats] = []
        let audioFilePaths = try XCTUnwrap(
            self.audioFileURLs,
            "Audio files not found"
        ).map { $0.path() }

        if WhisperKit.recommendedModels().disabled.contains(where: { $0.range(of: config.model) != nil }) {
            throw WhisperError.modelsUnavailable("Skipping model \(config.model), disabled for \(device).")
        }

        // Create WhisperKit instance with checks for memory usage
        let whisperKit = try await createWithMemoryCheck(
            testConfig: config,
            verbose: true,
            logLevel: .debug
        )

        if let modelFile = whisperKit.modelFolder?.lastPathComponent {
            config.model = modelFile
            modelsTested.append(modelFile)
            modelsTested = Array(Set(modelsTested))
            modelReposTested.append(config.modelRepo)
            modelReposTested = Array(Set(modelReposTested))
        }

        for audioFilePath in audioFilePaths {
            // Process each audio file
            try await processAudioFile(
                audioFilePath: audioFilePath,
                whisperKit: whisperKit,
                config: config,
                device: device,
                resultJSON: &resultJSON
            )
        }

        do {
            let jsonData = try JSONEncoder().encode(resultJSON)
            let attachment = XCTAttachment(data: jsonData, uniformTypeIdentifier: UTType.json.identifier)
            let attachmentName = "\(device)_\(config.model)_\(Date().formatted(Date.ISO8601FormatStyle().dateSeparator(.dash).timeSeparator(.omitted)))_\(config.dataset)".replacingOccurrences(of: ".", with: "_")
            attachment.name = attachmentName + ".json"
            attachment.lifetime = .keepAlways
            add(attachment)
            return attachmentName
        } catch {
            XCTFail("Failed with error: \(error)")
            return nil
        }
    }

    func processAudioFile(
        audioFilePath: String,
        whisperKit: WhisperKit,
        config: TestConfig,
        device: String,
        resultJSON: inout [RegressionStats]
    ) async throws {
        let startTime = Date()

        // Initialize test state
        var memoryStats = MemoryStats(
            measurements: [], units: "MB",
            totalNumberOfMeasurements: 0,
            preTranscribeMemory: -1,
            postTranscribeMemory: -1
        )
        var latencyStats = LatencyStats(
            measurements: [], units: "Tokens/Sec",
            totalNumberOfMeasurements: 0
        )

        let startTimeStamp = CFAbsoluteTimeGetCurrent()
        let testState = TranscriptionTestState(
            startTime: startTime,
            startTimeStamp: startTimeStamp,
            memoryStats: memoryStats,
            latencyStats: latencyStats
        )

        let callback: TranscriptionCallback = { result in
            Task {
                await testState.update(with: result)
            }
            return true
        }

        memoryStats.preTranscribeMemory = Float(AppMemoryChecker.getMemoryUsed())

        var systemMemory: [SystemMemoryUsage] = []
        var diskSpace: [DiskSpace] = []
        var batteryLevel: [Float] = []
        var thermalState: [Int] = []
        var timerTimeElapsed: [TimeInterval] = []

        // Start your timer
        let timerQueue = DispatchQueue(label: "RegressionTimerQueue")
        let timer = DispatchSource.makeTimerSource(queue: timerQueue)
        timer.schedule(deadline: .now(), repeating: 1.0)
        timer.setEventHandler {
            systemMemory.append(SystemMemoryCheckerAdvanced.getMemoryUsage())
            diskSpace.append(DiskSpaceChecker.getDiskSpace())
            batteryLevel.append(BatteryLevelChecker.getBatteryLevel() ?? -1)
            thermalState.append(ThermalStateChecker.getThermalState())
            timerTimeElapsed.append(Date().timeIntervalSince(startTime))
        }
        timer.resume()

        // Perform transcription
        let transcriptionResults = try await whisperKit.transcribe(
            audioPath: audioFilePath,
            decodeOptions: config.decodingOptions,
            callback: callback
        )

        let tpsThreshold = 4.0
        let currentTPS = await testState.getCurrentTPS()
        if !(currentTPS != 0 && currentTPS > tpsThreshold) {
            XCTFail("Tokens per second below expected for compute unit \(currentTPS), potential CPU fallback")
        }

        let transcriptionResult = TranscriptionUtilities.mergeTranscriptionResults(transcriptionResults)

        // Store final measurements
        let (finalMemoryStats, finalLatencyStats) = await testState.processFinalMeasurements()

        memoryStats = finalMemoryStats
        latencyStats = finalLatencyStats

        memoryStats.postTranscribeMemory = Float(AppMemoryChecker.getMemoryUsed())

        let filename = String(audioFilePath.split(separator: "/").last!)
        guard let reference = getTranscript(filename: filename) else {
            Logging.debug("Reference transcript not found for \(filename)")
            return
        }

        let (wer, diff) = WERUtils.evaluate(
            originalTranscript: reference,
            generatedTranscript: transcriptionResult.text
        )

        let modelSizeMB = try? getFolderSize(atUrl: whisperKit.modelFolder)

        let testInfo = TestInfo(
            device: device,
            audioFile: URL(fileURLWithPath: audioFilePath).lastPathComponent,
            datasetDir: config.dataset,
            datasetRepo: datasetRepo,
            model: config.model,
            modelRepo: config.modelRepo,
            modelSizeMB: modelSizeMB ?? -1,
            date: startTime.formatted(Date.ISO8601FormatStyle().dateSeparator(.dash)),
            timeElapsedInSeconds: Date().timeIntervalSince(startTime),
            timings: transcriptionResult.timings,
            prediction: transcriptionResult.text,
            reference: reference,
            wer: wer,
            diff: diff
        )
        let staticAttributes = StaticAttributes(
            encoderCompute: whisperKit.modelCompute.audioEncoderCompute,
            decoderCompute: whisperKit.modelCompute.textDecoderCompute,
            decodingOptions: config.decodingOptions
        )
        let systemMeasurements = SystemMeasurements(
            systemMemory: systemMemory,
            diskSpace: diskSpace,
            batteryLevel: batteryLevel,
            thermalState: thermalState,
            timeElapsed: timerTimeElapsed
        )
        let json = RegressionStats(
            testInfo: testInfo,
            memoryStats: memoryStats,
            latencyStats: latencyStats,
            staticAttributes: staticAttributes,
            systemMeasurements: systemMeasurements
        )
        resultJSON.append(json)
    }

    // MARK: - Pipeline Tests

    func testLargeWER() async {
        let texts = await getWERTestData()

        let (simpleWER, simpleDiff) = WERUtils.evaluate(originalTranscript: "This is some basic text", generatedTranscript: "This is edited text with some words added replaced and deleted")
        XCTAssertEqual(simpleWER, 1.7, accuracy: 0.1, "Expected wer: 1.7 but computed \(simpleWER)")
        XCTAssertEqual(simpleDiff.count, 23)
        if let originalText = texts.0, let generatedText = texts.1 {
            let (werNormalized, _) = WERUtils.evaluate(originalTranscript: originalText, generatedTranscript: generatedText)
            XCTAssertEqual(werNormalized, 0.1852080123266564, accuracy: 0.001, "Expected wer: 0.1852080123266564 but computed \(werNormalized)")
            let (wer, _) = WERUtils.evaluate(originalTranscript: originalText, generatedTranscript: generatedText, normalizeOriginal: false)
            XCTAssertEqual(wer, 0.42448103078024335, accuracy: 0.001, "Expected wer: 0.42448103078024335 but computed \(wer)")
        } else {
            XCTFail("Fetching WER test data failed.")
        }
    }

    func testNormalizer() {
        let normalizer = EnglishTextNormalizer()
        let jsonText = "hello\\u2026 this is a test over GH\\u20b5 94 million in fees in H\\u00f8rsholm and Basel grew 10% to one billions, 370 millions"
        let swiftText = "hello\u{2026} this is a test over GH\u{20b5} 94 million in fees in H\u{00f8}rsholm and Basel grew 10% to one billions, 370 millions"
        let resultJson = normalizer.normalize(text: jsonText)
        let resultSwift = normalizer.normalize(text: swiftText)
        XCTAssertEqual(resultSwift, "hello . this is a test over gh 94000000 in fees in horsholm and basel grew 10% to 1000000000s 370000000s")
        XCTAssertEqual(resultJson, resultSwift)
    }

    func testHirschberg() {
        let s1 = "With a rumble that echoed through the night, thunder crashed overhead, its raw power shaking the earth beneath it, leaving in its wake an exhilarating sense of awe. As rain poured down in torrents, the thunder boomed with a rhythm that seemed to speak a secret language, intertwining nature's symphony with an innovative melody that captivated all who listened."
        let s2 = "In the midst of a summer storm, thunder erupted with a booming chorus, shaking the earth beneath our feet and electrifying the air with its powerful presence. The crackling symphony of thunderbolts danced across the darkened sky, illuminating the clouds with an innovative display of nature's raw energy."
        let ops = hirschberg(Array(s1.unicodeScalars), Array(s2.unicodeScalars))
        XCTAssertEqual(ops.count, 228)
    }

    func testLevenshtein() {
        let s1 = "With a rumble that echoed through the night, thunder crashed overhead, its raw power shaking the earth beneath it, leaving in its wake an exhilarating sense of awe. As rain poured down in torrents, the thunder boomed with a rhythm that seemed to speak a secret language, intertwining nature's symphony with an innovative melody that captivated all who listened."
        let s2 = "In the midst of a summer storm, thunder erupted with a booming chorus, shaking the earth beneath our feet and electrifying the air with its powerful presence. The crackling symphony of thunderbolts danced across the darkened sky, illuminating the clouds with an innovative display of nature's raw energy."
        let ops = levenshtein(Array(s1.unicodeScalars), Array(s2.unicodeScalars))
        XCTAssertEqual(ops.count, 495)
    }

    func testInMemoryAndDiskUsage() async throws {
        // Choose a model to test
        let modelToTest = "openai_whisper-tiny"

        // Get initial measurements
        let initialMemory = AppMemoryChecker.getMemoryUsed()
        let initialDiskSpace = DiskSpaceChecker.getDiskSpace()
        let initialCacheSize = try getCacheSize()

        // Create WhisperKit instance
        let whisperKit = try await WhisperKit(WhisperKitConfig(
            model: modelToTest,
            computeOptions: ModelComputeOptions(audioEncoderCompute: .cpuAndNeuralEngine, textDecoderCompute: .cpuAndNeuralEngine),
            verbose: true,
            logLevel: .debug,
            load: true
        ))

        // Get final measurements
        let finalMemory = AppMemoryChecker.getMemoryUsed()
        let finalDiskSpace = DiskSpaceChecker.getDiskSpace()
        let finalCacheSize = try getCacheSize()

        // Calculate differences
        let memoryUsed = finalMemory - initialMemory
        let diskSpaceUsed = initialDiskSpace.freeSpaceGB! - finalDiskSpace.freeSpaceGB!
        let cacheSpaceUsed = finalCacheSize - initialCacheSize

        // Log results
        Logging.debug("Memory used by \(modelToTest): \(memoryUsed) MB")
        Logging.debug("Disk space used by \(modelToTest): \(diskSpaceUsed) MB")
        Logging.debug("Cache space used by \(modelToTest): \(cacheSpaceUsed) MB")

        // Assert that the measurements are within expected ranges
        XCTAssertGreaterThan(memoryUsed, 0, "Model should use some memory")
        XCTAssertLessThan(memoryUsed, 1000, "Model should use less than 1GB of memory")

        XCTAssertGreaterThanOrEqual(diskSpaceUsed, 0, "Model should use some disk space unless already downloaded")
        XCTAssertLessThan(diskSpaceUsed, 5000, "Model should use less than 5GB of disk space")

        XCTAssertGreaterThanOrEqual(cacheSpaceUsed, 0, "Cache usage should not be negative")

        // Clean up
        await whisperKit.unloadModels()
    }

    // MARK: - Helper Methods

    private func downloadTestDataIfNeeded() async throws {
        guard audioFileURLs == nil || metadataURL == nil || testWERURLs == nil else { return }

        for dataset in datasets {
            try await downloadTestData(forDataset: dataset)
        }
    }

    public func getTestMatrix() -> [TestConfig] {
        var regressionTestConfigMatrix: [TestConfig] = []
        for dataset in datasets {
            for computeOption in computeOptions {
                for options in optionsToTest {
                    for repo in modelReposToTest {
                        for model in modelsToTest {
                            regressionTestConfigMatrix.append(
                                TestConfig(
                                    dataset: dataset,
                                    modelComputeOptions: computeOption,
                                    model: model,
                                    modelRepo: repo,
                                    decodingOptions: options
                                )
                            )
                        }
                    }
                }
            }
        }

        return regressionTestConfigMatrix
    }

    private func downloadTestData(forDataset dataset: String) async throws {
        Logging.debug("Available models: \(modelsToTest)")

        let testDatasetRepo = HubApiWrapper.Repo(id: datasetRepo, type: .datasets)
        let tempPath = FileManager.default.temporaryDirectory
        let downloadBase = tempPath.appending(component: "huggingface")
        let hubApi = HubApiWrapper(downloadBase: downloadBase)
        let repoURL = try await hubApi.snapshot(from: testDatasetRepo, matching: ["\(dataset)/*"]) { progress in
            Logging.debug("Downloading \(dataset) dataset: \(progress)")
        }.appending(path: dataset)

        let downloadedFiles = try FileManager.default.contentsOfDirectory(atPath: repoURL.path())
        var audioFileURLs: [URL] = []
        for file in downloadedFiles {
            if file.hasSuffix(".mp3") {
                audioFileURLs.append(repoURL.appending(component: file))
            } else if file.hasSuffix(".json") {
                self.metadataURL = repoURL.appending(component: file)
            }
        }
        self.audioFileURLs = audioFileURLs

        Logging.debug("Downloaded \(audioFileURLs.count) audio files.")
    }

    private func getTranscript(filename: String) -> String? {
        // Ensure we can access and parse the metadata
        guard let data = try? Data(contentsOf: self.metadataURL!),
              let json = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else {
            return nil
        }

        // Search for the matching audio item
        for item in json {
            // Check if the current item's audio matches the filename
            let audioFileName = filename.split(separator: ".").first!
            if let referenceFilename = item["audio"] as? String,
               referenceFilename.contains(audioFileName)
            {
                // If found, return the reference text
                return item["text"] as? String
            }
        }

        // If no matching item was found, return nil
        return nil
    }

    private func getWERTestData() async -> (String?, String?) {
        do {
            let testDataset = HubApiWrapper.Repo(id: datasetRepo, type: .datasets)
            let tempPath = FileManager.default.temporaryDirectory
            let downloadBase = tempPath.appending(component: "huggingface")
            let hubApi = HubApiWrapper(downloadBase: downloadBase)
            let testWERRepoURL = try await hubApi.snapshot(from: testDataset, matching: ["*.txt"])
            let testWERTextURLs = try FileManager.default.contentsOfDirectory(atPath: testWERRepoURL.path()).filter { $0.hasSuffix(".txt") }
            self.testWERURLs = testWERTextURLs.map { testWERRepoURL.appending(component: $0) }

            Logging.debug("Downloaded \(testWERTextURLs.count) test WER files.")

            let testFileURLs = try XCTUnwrap(
                self.testWERURLs,
                "Test files for WER verification not found"
            )
            var generatedText: String?
            var originalText: String?
            for file in testFileURLs {
                switch file.lastPathComponent {
                    case "test_generated_transcript.txt":
                        generatedText = try? String(contentsOf: file)
                    case "test_original_transcript.txt":
                        originalText = try? String(contentsOf: file)
                    default:
                        continue
                }
            }
            return (originalText, generatedText)
        } catch {
            XCTFail("Fetching test data for WER verification failed: \(error)")
        }
        return (nil, nil)
    }

    private func saveSummary(failureInfo: [String: String], attachments: [String: String]) {
        let currentDevice = getCurrentDevice()
        let osDetails = getOSDetails()
        let testReport = TestReport(
            deviceModel: currentDevice,
            osType: osDetails.osType,
            osVersion: osDetails.osVersion,
            modelsTested: modelsTested,
            modelReposTested: modelReposTested,
            failureInfo: failureInfo,
            attachments: attachments
        )

        do {
            let iso8601DateTimeString = ISO8601DateFormatter().string(from: Date())
            let jsonData = try testReport.jsonData()
            let attachment = XCTAttachment(data: jsonData, uniformTypeIdentifier: UTType.json.identifier)
            attachment.lifetime = .keepAlways
            attachment.name = "\(currentDevice)_summary_\(iso8601DateTimeString).json"
            add(attachment)
        } catch {
            XCTFail("Failed with error: \(error)")
        }
    }

    private func getCurrentDevice() -> String {
        var currentDevice = WhisperKit.deviceName()

        currentDevice = currentDevice.trimmingCharacters(in: .whitespacesAndNewlines)
        currentDevice = currentDevice.replacingOccurrences(of: " ", with: "_")

        return currentDevice
    }

    private func getOSDetails() -> (osType: String, osVersion: String) {
        #if os(iOS)
        return (UIDevice.current.systemName, UIDevice.current.systemVersion)
        #elseif os(macOS)
        let version = ProcessInfo.processInfo.operatingSystemVersion
        return ("macOS", "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)")
        #elseif os(watchOS)
        return ("watchOS", WKInterfaceDevice.current().systemVersion)
        #else
        return ("Unknown", "Unknown")
        #endif
    }

    /// Helper function to get cache size
    private func getCacheSize() throws -> Int64 {
        let fileManager = FileManager.default
        let cacheURL = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
        let cacheSize = try fileManager.allocatedSizeOfDirectory(at: cacheURL.first!)
        return cacheSize / (1024 * 1024) // Convert to MB
    }

    private func getFolderSize(atUrl folder: URL?) throws -> Double {
        guard let folder = folder else {
            return -1
        }
        let fileManager = FileManager.default
        let modelSize = try fileManager.allocatedSizeOfDirectory(at: folder)
        return Double(modelSize / (1024 * 1024)) // Convert to MB
    }

    private static func initWhisperKit(
        testConfig config: TestConfig,
        verbose: Bool,
        logLevel: Logging.LogLevel
    ) async throws -> WhisperKit {
        let whisperKit = try await WhisperKit(WhisperKitConfig(
            model: config.model,
            modelRepo: config.modelRepo,
            modelToken: Self.getModelToken(),
            computeOptions: config.modelComputeOptions,
            verbose: verbose,
            logLevel: logLevel,
            prewarm: true,
            load: true
        ))
        try Task.checkCancellation()
        return whisperKit
    }

    func createWithMemoryCheck(
        testConfig: TestConfig,
        verbose: Bool,
        logLevel: Logging.LogLevel
    ) async throws -> WhisperKit {
        return try await withThrowingTaskGroup(of: InitializationResult.self) { group in
            group.addTask {
                do {
                    let whisperKit = try await Self.initWhisperKit(
                        testConfig: testConfig,
                        verbose: verbose,
                        logLevel: logLevel
                    )
                    return .completed(WhisperKitWrapper(whisperKit))
                } catch {
                    return .failed(error.localizedDescription)
                }
            }

            group.addTask {
                while !Task.isCancelled {
                    let remainingMemory = SystemMemoryCheckerAdvanced.getMemoryUsage().totalAvailableGB
                    Logging.debug("\(remainingMemory) GB of memory left")

                    if remainingMemory <= 0.1 { // Cancel with 100MB remaining
                        Logging.debug("Cancelling due to oom")
                        return .outOfMemory
                    }

                    do {
                        try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
                    } catch {
                        return .cancelled
                    }
                }
                return .cancelled
            }

            group.addTask {
                do {
                    try await Task.sleep(nanoseconds: 300_000_000_000) // 5 minutes
                } catch {
                    return .cancelled
                }

                Logging.debug("Cancelling due to timeout")
                return .timedOut
            }

            if let signal = try await group.next() {
                switch signal {
                    case .completed(let wrapper):
                        group.cancelAll()
                        return wrapper.whisperKit
                    case .failed(let errorDescription):
                        group.cancelAll()
                        throw WhisperError.modelsUnavailable(errorDescription)
                    case .outOfMemory:
                        group.cancelAll()
                        throw WhisperError.modelsUnavailable("Memory limit exceeded during initialization")
                    case .timedOut:
                        group.cancelAll()
                        throw WhisperError.modelsUnavailable("Initialization timed out")
                    case .cancelled:
                        group.cancelAll()
                        throw WhisperError.modelsUnavailable("Initialization was cancelled")
                }
            }
            throw WhisperError.modelsUnavailable("Initialization failed without a completion result")
        }
    }
}

// MARK: - Helper Types

private final class WhisperKitWrapper: @unchecked Sendable {
    let whisperKit: WhisperKit

    init(_ whisperKit: WhisperKit) {
        self.whisperKit = whisperKit
    }
}

private enum InitializationResult: Sendable {
    case completed(WhisperKitWrapper)
    case failed(String)
    case outOfMemory
    case timedOut
    case cancelled
}
