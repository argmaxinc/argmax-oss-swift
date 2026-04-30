//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import AVFoundation
import Charts
import CoreML
import SwiftUI
import WhisperKit

struct WhisperAXWatchView: View {
    @State private var whisperKit: WhisperKit?
    @State private var currentText = "Tap below to start"
    @State private var isTranscribing = false
    @State private var isRecording = false
    @State private var energyToDisplayCount = 100
    // TODO: Make this configurable in the UI
    @State var modelStorage: String = "huggingface/models/argmaxinc/whisperkit-coreml"

    @AppStorage("selectedModel") private var selectedModel: String = WhisperKit.recommendedModels().default
    @AppStorage("selectedTab") private var selectedTab: String = "Transcribe"
    @AppStorage("selectedTask") private var selectedTask: String = "transcribe"
    @AppStorage("selectedLanguage") private var selectedLanguage: String = "english"
    @AppStorage("repoName") private var repoName: String = "argmaxinc/whisperkit-coreml"
    @AppStorage("enableTimestamps") private var enableTimestamps: Bool = false
    @AppStorage("enablePromptPrefill") private var enablePromptPrefill: Bool = true
    @AppStorage("enableCachePrefill") private var enableCachePrefill: Bool = true
    @AppStorage("enableSpecialCharacters") private var enableSpecialCharacters: Bool = false
    @AppStorage("enableEagerDecoder") private var enableEagerDecoder: Bool = false
    @AppStorage("temperatureStart") private var temperatureStart: Double = 0
    @AppStorage("fallbackCount") private var fallbackCount: Double = 4
    @AppStorage("compressionCheckWindow") private var compressionCheckWindow: Double = 20
    @AppStorage("sampleLength") private var sampleLength: Double = 224
    @AppStorage("silenceThreshold") private var silenceThreshold: Double = 0.3
    @AppStorage("useVAD") private var useVAD: Bool = true

    @State private var modelState: ModelState = .unloaded
    @State private var localModels: [String] = []
    @State private var localModelPath: String = ""
    @State private var availableModels: [String] = []
    @State private var availableLanguages: [String] = []
    @State private var disabledModels: [String] = WhisperKit.recommendedModels().disabled

    @State private var loadingProgressValue: Float = 0.0
    @State private var specializationProgressRatio: Float = 0.7
    @State private var currentLag: TimeInterval = 0
    @State private var currentFallbacks: Int = 0
    @State private var lastBufferSize: Int = 0
    @State private var lastConfirmedSegmentEndSeconds: Float = 0
    @State private var requiredSegmentsForConfirmation: Int = 2
    @State private var bufferEnergy: [EnergyValue] = []
    @State private var confirmedSegments: [TranscriptionSegment] = []
    @State private var unconfirmedSegments: [TranscriptionSegment] = []
    @State private var unconfirmedText: [String] = []

    @State private var transcriptionTask: Task<Void, Never>? = nil

    @State private var selectedCategoryId: MenuItem.ID?
    private var menu = [
        MenuItem(name: "Start Transcribing", image: "waveform.badge.mic"),
    ]

    struct MenuItem: Identifiable, Hashable {
        var id = UUID()
        var name: String
        var image: String
    }

    struct EnergyValue: Identifiable {
        let id = UUID()
        var index: Int
        var value: Float
    }

    var body: some View {
        NavigationSplitView {
            if WhisperKit.deviceName().hasPrefix("Watch7") || WhisperKit.isRunningOnSimulator {
                modelSelectorView
                    .navigationTitle("WhisperAX")
                    .navigationBarTitleDisplayMode(.automatic)

                if modelState == .loaded {
                    List(menu, selection: $selectedCategoryId) { item in
                        HStack {
                            Image(systemName: item.image)
                            Text(item.name)
                                .scaledToFit()
                                .minimumScaleFactor(0.5)
                                .font(.system(.title3))
                                .bold()
                                .padding(.horizontal)
                        }
                    }
                    .foregroundColor(.primary)
                    .navigationTitle("WhisperAX")
                    .navigationBarTitleDisplayMode(.automatic)
                }
            } else {
                VStack {
                    Image(systemName: "exclamationmark.applewatch")
                        .foregroundColor(.red)
                        .font(.system(size: 80))
                        .padding()

                    Text("Sorry, this app\nrequires Apple Watch\nSeries 9 or Ultra 2")
                        .scaledToFill()
                        .minimumScaleFactor(0.5)
                        .lineLimit(3)
                        .multilineTextAlignment(.center)
                        .padding()
                        .frame(maxWidth: .infinity)
                }
                .navigationTitle("WhisperAX")
                .navigationBarTitleDisplayMode(.inline)
            }

        } detail: {
            streamingView
        }
        .onAppear {
            fetchModels()
        }
    }

    var modelStatusView: some View {
        HStack {
            Image(systemName: "circle.fill")
                .foregroundStyle(modelState == .loaded ? .green : (modelState == .unloaded ? .red : .yellow))
                .symbolEffect(.variableColor, isActive: modelState != .loaded && modelState != .unloaded)
            Text(modelState.description)
                .font(.footnote)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    var modelSelectorView: some View {
        Group {
            VStack {
                modelStatusView
                    .padding(.top)
                HStack {
                    if availableModels.count > 0 {
                        Picker("", selection: $selectedModel) {
                            ForEach(availableModels, id: \.self) { model in
                                HStack {
                                    let modelIcon = localModels.contains { $0 == model.description } ? "checkmark.circle" : "arrow.down.circle.dotted"
                                    Text("\(Image(systemName: modelIcon)) \(model.description)").tag(model.description)
                                        .scaledToFit()
                                        .minimumScaleFactor(0.5)
                                }
                            }
                        }
                        .frame(height: 80)
                        .pickerStyle(.wheel)
                        .onChange(of: selectedModel, initial: false) { _, _ in
                            resetState()
                            modelState = .unloaded
                        }
                    } else {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle())
                            .scaleEffect(0.5)
                    }
                }

                if modelState == .unloaded {
                    Button {
                        resetState()
                        loadModel(selectedModel)
                        modelState = .loading
                    } label: {
                        Text("Load Model")
                    }
                    .buttonStyle(.bordered)
                } else if loadingProgressValue < 1.0 {
                    VStack {
                        HStack {
                            ProgressView(value: loadingProgressValue, total: 1.0)
                                .progressViewStyle(LinearProgressViewStyle())
                                .frame(maxWidth: .infinity)

                            Text(String(format: "%.1f%%", loadingProgressValue * 100))
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                        if modelState == .prewarming {
                            Text("Specializing \(selectedModel)...")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                    }
                }
            }
        }
    }

    var streamingView: some View {
        ZStack(alignment: .bottom) {
            VStack {
                Spacer()
                    .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity, alignment: Alignment.topLeading)
                HStack(alignment: .bottom) {
                    if isRecording {
                        Chart(bufferEnergy) {
                            BarMark(
                                x: .value("", $0.index),
                                y: .value("", $0.value),
                                width: 2,
                                stacking: .center
                            )
                            .cornerRadius(1)
                            .foregroundStyle($0.value > Float(silenceThreshold) ? .green : .red)
                        }
                        .chartXAxis(.hidden)
                        .chartXScale(domain: [0, energyToDisplayCount])
                        .chartYAxis(.hidden)
                        .chartYScale(domain: [-0.5, 0.5])
                        .frame(height: 45)
                        .padding(.bottom)
                    }
                }
            }
            .ignoresSafeArea()

            ScrollView(.vertical) {
                VStack(alignment: .leading) {
                    Spacer()
                        .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity, alignment: Alignment.topLeading)
                    ForEach(Array(confirmedSegments.enumerated()), id: \.element) { _, segment in
                        Text(segment.text)
                            .font(.headline)
                            .fontWeight(.bold)
                            .tint(.green)
                            .multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    ForEach(Array(unconfirmedSegments.enumerated()), id: \.element) { _, segment in
                        Text(segment.text)
                            .font(.headline)
                            .fontWeight(.light)
                            .multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            .defaultScrollAnchor(.bottom)
        }
        .toolbar {
            ToolbarItem(placement: .bottomBar) {
                let currentTranscription = (confirmedSegments.map { $0.text } + unconfirmedSegments.map { $0.text }).joined(separator: " ")
                ShareLink(item: currentTranscription, label: {
                    Image(systemName: "square.and.arrow.up")
                })
            }
            ToolbarItem(placement: .bottomBar) {
                Button {
                    withAnimation {
                        toggleRecording(shouldLoop: true)
                    }
                } label: {
                    Image(systemName: !isRecording ? "record.circle" : "stop.circle.fill")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 28, height: 28)
                        .foregroundColor(modelState != .loaded ? .gray : .red)
                }
                .contentTransition(.symbolEffect(.replace))
                .buttonStyle(BorderlessButtonStyle())
                .disabled(modelState != .loaded)
                .frame(minWidth: 0, maxWidth: .infinity)
            }
        }
    }

    // MARK: Logic

    func resetState() {
        isRecording = false
        isTranscribing = false
        whisperKit?.audioProcessor.stopRecording()
        currentText = ""
        unconfirmedText = []

        currentLag = 0
        currentFallbacks = 0
        lastBufferSize = 0
        lastConfirmedSegmentEndSeconds = 0
        requiredSegmentsForConfirmation = 2
        bufferEnergy = []
        confirmedSegments = []
        unconfirmedSegments = []
    }

    func fetchModels() {
        availableModels = ["base", "base.en", "tiny", "tiny.en"]
        let devices = MLModel.availableComputeDevices
        print("Available devices: \(devices)")
        // First check what's already downloaded
        if let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let modelPath = documents.appendingPathComponent(modelStorage).path

            // Check if the directory exists
            if FileManager.default.fileExists(atPath: modelPath) {
                localModelPath = modelPath
                do {
                    let downloadedModels = try FileManager.default.contentsOfDirectory(atPath: modelPath)
                    for model in downloadedModels where !localModels.contains(model) && model.starts(with: "openai") {
                        localModels.append(model)
                    }
                } catch {
                    print("Error enumerating files at \(modelPath): \(error.localizedDescription)")
                }
            }
        }

        localModels = WhisperKit.formatModelFiles(localModels)
        for model in localModels {
            if !availableModels.contains(model),
               !disabledModels.contains(model)
            {
                availableModels.append(model)
            }
        }

        print("Found locally: \(localModels)")
        print("Previously selected model: \(selectedModel)")

//        Task {
//            let remoteModels = try await WhisperKit.fetchAvailableModels(from: repoName)
//            for model in remoteModels {
//                if !availableModels.contains(model),
//                   !disabledModels.contains(model){
//                    availableModels.append(model)
//                }
//            }
//        }
    }

    func loadModel(_ model: String, redownload: Bool = false) {
        print("Selected Model: \(selectedModel)")

        whisperKit = nil
        Task {
            let config = WhisperKitConfig(verbose: true,
                                          logLevel: .debug,
                                          prewarm: false,
                                          load: false,
                                          download: false)

            whisperKit = try await WhisperKit(config)
            guard let whisperKit = whisperKit else {
                return
            }

            var folder: URL?

            // Check if the model is available locally
            if localModels.contains(model) && !redownload {
                // Get local model folder URL from localModels
                // TODO: Make this configurable in the UI
                // TODO: Handle incomplete downloads
                folder = URL(fileURLWithPath: localModelPath).appendingPathComponent(model)
            } else {
                // Download the model
                folder = try await WhisperKit.download(variant: model, from: repoName, progressCallback: { progress in
                    DispatchQueue.main.async {
                        loadingProgressValue = Float(progress.fractionCompleted) * specializationProgressRatio
                        modelState = .downloading
                    }
                })
            }

            if let modelFolder = folder {
                whisperKit.modelFolder = modelFolder

                await MainActor.run {
                    // Set the loading progress to 90% of the way after prewarm
                    loadingProgressValue = specializationProgressRatio
                    modelState = .prewarming
                }

                let progressBarTask = Task {
                    await updateProgressBar(targetProgress: 0.9, maxTime: 240)
                }

                // Prewarm models
                do {
                    try await whisperKit.prewarmModels()
                    progressBarTask.cancel()
                } catch {
                    print("Error prewarming models, retrying: \(error.localizedDescription)")
                    progressBarTask.cancel()
                    if !redownload {
                        loadModel(model, redownload: true)
                        return
                    } else {
                        // Redownloading failed, error out
                        modelState = .unloaded
                        return
                    }
                }

                await MainActor.run {
                    // Set the loading progress to 90% of the way after prewarm
                    loadingProgressValue = specializationProgressRatio + 0.9 * (1 - specializationProgressRatio)
                    modelState = .loading
                }

                try await whisperKit.loadModels()

                await MainActor.run {
                    availableLanguages = Constants.languages.map { $0.key }.sorted()
                    loadingProgressValue = 1.0
                    modelState = whisperKit.modelState
                }
            }
        }
    }

    func updateProgressBar(targetProgress: Float, maxTime: TimeInterval) async {
        let initialProgress = loadingProgressValue
        let decayConstant = -log(1 - targetProgress) / Float(maxTime)

        let startTime = Date()

        while true {
            let elapsedTime = Date().timeIntervalSince(startTime)

            // Break down the calculation
            let decayFactor = exp(-decayConstant * Float(elapsedTime))
            let progressIncrement = (1 - initialProgress) * (1 - decayFactor)
            let currentProgress = initialProgress + progressIncrement

            await MainActor.run {
                loadingProgressValue = currentProgress
            }

            if currentProgress >= targetProgress {
                break
            }

            do {
                try await Task.sleep(nanoseconds: 100_000_000)
            } catch {
                break
            }
        }
    }

    func toggleRecording(shouldLoop: Bool) {
        isRecording.toggle()

        if isRecording {
            resetState()
            startRecording()
        } else {
            stopRecording()
        }
    }

    func startRecording() {
        guard let whisperKit = whisperKit else { return }
        Task(priority: .userInitiated) {
//                guard await requestMicrophoneIfNeeded() else {
//                    print("Microphone access was not granted.")
//                    return
//                }

            try? whisperKit.audioProcessor.startRecordingLive { _ in
                DispatchQueue.main.async {
                    var energyToDisplay: [EnergyValue] = []
                    for (idx, val) in whisperKit.audioProcessor.relativeEnergy.suffix(energyToDisplayCount).enumerated() {
                        energyToDisplay.append(EnergyValue(index: idx, value: val))
                    }
                    bufferEnergy = energyToDisplay
                }
            }

            // Delay the timer start by 1 second
            isRecording = true
            isTranscribing = true
            realtimeLoop()
        }
    }

    func stopRecording() {
        guard let whisperKit = whisperKit else { return }
        whisperKit.audioProcessor.stopRecording()
    }

    func transcribeAudioSamples(_ samples: [Float]) async throws -> TranscriptionResult? {
        guard let whisperKit = whisperKit else { return nil }

        let languageCode = Constants.languages[selectedLanguage, default: Constants.defaultLanguageCode]
        let task: DecodingTask = selectedTask == "transcribe" ? .transcribe : .translate
        let seekClip = [lastConfirmedSegmentEndSeconds]

        let options = DecodingOptions(
            verbose: false,
            task: task,
            language: languageCode,
            temperatureFallbackCount: 3, // limit fallbacks for realtime
            sampleLength: Int(sampleLength), // reduced sample length for realtime
            skipSpecialTokens: true,
            clipTimestamps: seekClip
        )

        // Early stopping checks
        let decodingCallback: TranscriptionCallback = { progress in
            DispatchQueue.main.async {
                let fallbacks = Int(progress.timings.totalDecodingFallbacks)
                if progress.text.count < currentText.count {
                    if fallbacks == self.currentFallbacks {
                        self.unconfirmedText.append(currentText)
                    } else {
                        print("Fallback occured: \(fallbacks)")
                    }
                }
                self.currentText = progress.text
                self.currentFallbacks = fallbacks
            }
            // Check early stopping
            let currentTokens = progress.tokens
            let checkWindow = Int(compressionCheckWindow)
            if currentTokens.count > checkWindow {
                let checkTokens: [Int] = currentTokens.suffix(checkWindow)
                let compressionRatio = compressionRatio(of: checkTokens)
                if compressionRatio > options.compressionRatioThreshold! {
                    return false
                }
            }
            if progress.avgLogprob! < options.logProbThreshold! {
                return false
            }

            return nil
        }

        let transcription: [TranscriptionResult] = try await whisperKit.transcribe(
            audioArray: samples,
            decodeOptions: options,
            callback: decodingCallback
        )
        return transcription.first
    }

    // MARK: Streaming Logic

    func realtimeLoop() {
        transcriptionTask = Task {
            while isRecording && isTranscribing {
                do {
                    try await transcribeCurrentBuffer()
                } catch {
                    print("Error: \(error.localizedDescription)")
                    break
                }
            }
        }
    }

    func transcribeCurrentBuffer() async throws {
        guard let whisperKit = whisperKit else { return }

        // Retrieve the current audio buffer from the audio processor
        let currentBuffer = whisperKit.audioProcessor.audioSamples

        // Calculate the size and duration of the next buffer segment
        let nextBufferSize = currentBuffer.count - lastBufferSize
        let nextBufferSeconds = Float(nextBufferSize) / Float(WhisperKit.sampleRate)

        // Only run the transcribe if the next buffer has at least 1 second of audio
        guard nextBufferSeconds > 1 else {
            await MainActor.run {
                if currentText == "" {
                    currentText = "Waiting for speech..."
                }
            }
            try await Task.sleep(nanoseconds: 100_000_000) // sleep for 100ms for next buffer
            return
        }

        if useVAD {
            let voiceDetected = AudioProcessor.isVoiceDetected(
                in: whisperKit.audioProcessor.relativeEnergy,
                nextBufferInSeconds: nextBufferSeconds,
                silenceThreshold: Float(silenceThreshold)
            )
            // Only run the transcribe if the next buffer has voice
            guard voiceDetected else {
                await MainActor.run {
                    if currentText == "" {
                        currentText = "Waiting for speech..."
                    }
                }

                //                if nextBufferSeconds > 30 {
                //                    // This is a completely silent segment of 30s, so we can purge the audio and confirm anything pending
                //                    lastConfirmedSegmentEndSeconds = 0
                //                    whisperKit.audioProcessor.purgeAudioSamples(keepingLast: 2 * WhisperKit.sampleRate) // keep last 2s to include VAD overlap
                //                    currentBuffer = whisperKit.audioProcessor.audioSamples
                //                    lastBufferSize = 0
                //                    confirmedSegments.append(contentsOf: unconfirmedSegments)
                //                    unconfirmedSegments = []
                //                }

                // Sleep for 100ms and check the next buffer
                try await Task.sleep(nanoseconds: 100_000_000)
                return
            }
        }

        // Run transcribe
        lastBufferSize = currentBuffer.count

        let transcription = try await transcribeAudioSamples(Array(currentBuffer))

        // We need to run this next part on the main thread
        await MainActor.run {
            currentText = ""
            unconfirmedText = []
            guard let segments = transcription?.segments else {
                return
            }

//            self.tokensPerSecond = transcription?.timings?.tokensPerSecond ?? 0
//            self.realTimeFactor = transcription?.timings?.realTimeFactor ?? 0
//            self.firstTokenTime = transcription?.timings?.firstTokenTime ?? 0
//            self.pipelineStart = transcription?.timings?.pipelineStart ?? 0
//            self.currentLag = transcription?.timings?.decodingLoop ?? 0

            // Logic for moving segments to confirmedSegments
            if segments.count > requiredSegmentsForConfirmation {
                // Calculate the number of segments to confirm
                let numberOfSegmentsToConfirm = segments.count - requiredSegmentsForConfirmation

                // Confirm the required number of segments
                let confirmedSegmentsArray = Array(segments.prefix(numberOfSegmentsToConfirm))
                let remainingSegments = Array(segments.suffix(requiredSegmentsForConfirmation))

                // Update lastConfirmedSegmentEnd based on the last confirmed segment
                if let lastConfirmedSegment = confirmedSegmentsArray.last, lastConfirmedSegment.end > lastConfirmedSegmentEndSeconds {
                    lastConfirmedSegmentEndSeconds = lastConfirmedSegment.end

                    // Add confirmed segments to the confirmedSegments array
                    if !self.confirmedSegments.contains(confirmedSegmentsArray) {
                        self.confirmedSegments.append(contentsOf: confirmedSegmentsArray)
                    }
                }

                // Update transcriptions to reflect the remaining segments
                self.unconfirmedSegments = remainingSegments
            } else {
                // Handle the case where segments are fewer or equal to required
                self.unconfirmedSegments = segments
            }
        }
    }
}

#Preview {
    WhisperAXWatchView()
}
