//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import SwiftUI
import WhisperKit
import SpeakerKit
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif
import AVFoundation
import CoreML

struct ContentView: View {
    @State private var whisperKit: WhisperKit?
    #if os(macOS)
    @State private var audioDevices: [AudioDevice]?
    #endif
    @State private var isRecording: Bool = false
    @State private var isTranscribing: Bool = false
    @State private var currentText: String = ""
    @State private var currentChunks: [Int: (chunkText: [String], fallbacks: Int)] = [:]
    // TODO: Make this configurable in the UI
    @State private var modelStorage: String = "huggingface/models/argmaxinc/whisperkit-coreml"
    @State private var appStartTime = Date()

    // MARK: Model management

    @State private var modelState: ModelState = .unloaded
    @State private var localModels: [String] = []
    @State private var localModelPath: String = ""
    @State private var availableModels: [String] = []
    @State private var availableLanguages: [String] = []
    @State private var disabledModels: [String] = WhisperKit.recommendedModels().disabled
    @AppStorage("promptText") private var promptText: String?
    @AppStorage("selectedAudioInput") private var selectedAudioInput: String = "No Audio Input"
    @AppStorage("selectedModel") private var selectedModel: String = WhisperKit.recommendedModels().default
    @AppStorage("selectedTab") private var selectedTab: String = "Transcribe"
    @AppStorage("selectedTask") private var selectedTask: String = "transcribe"
    @AppStorage("selectedLanguage") private var selectedLanguage: String = "english"
    @AppStorage("repoName") private var repoName: String = "argmaxinc/whisperkit-coreml"
    @AppStorage("enableTimestamps") private var enableTimestamps: Bool = true
    @AppStorage("enablePromptPrefill") private var enablePromptPrefill: Bool = true
    @AppStorage("enableCachePrefill") private var enableCachePrefill: Bool = true
    @AppStorage("enableSpecialCharacters") private var enableSpecialCharacters: Bool = false
    @AppStorage("enableEagerDecoding") private var enableEagerDecoding: Bool = false
    @AppStorage("enableDecoderPreview") private var enableDecoderPreview: Bool = true
    @AppStorage("temperatureStart") private var temperatureStart: Double = 0
    @AppStorage("fallbackCount") private var fallbackCount: Double = 5
    @AppStorage("compressionCheckWindow") private var compressionCheckWindow: Double = 60
    @AppStorage("sampleLength") private var sampleLength: Double = 224
    @AppStorage("silenceThreshold") private var silenceThreshold: Double = 0.3
    @AppStorage("realtimeDelayInterval") private var realtimeDelayInterval: Double = 1
    @AppStorage("useVAD") private var useVAD: Bool = true
    @AppStorage("tokenConfirmationsNeeded") private var tokenConfirmationsNeeded: Double = 2
    @AppStorage("concurrentWorkerCount") private var concurrentWorkerCount: Double = 4
    @AppStorage("chunkingStrategy") private var chunkingStrategy: ChunkingStrategy = .vad
    @AppStorage("encoderComputeUnits") private var encoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
    @AppStorage("decoderComputeUnits") private var decoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine

    // MARK: Standard properties

    @State private var loadingProgressValue: Float = 0.0
    @State private var specializationProgressRatio: Float = 0.7
    @State private var isFilePickerPresented = false
    @State private var modelLoadingTime: TimeInterval = 0
    @State private var firstTokenTime: TimeInterval = 0
    @State private var pipelineStart: TimeInterval = 0
    @State private var effectiveRealTimeFactor: TimeInterval = 0
    @State private var effectiveSpeedFactor: TimeInterval = 0
    @State private var totalInferenceTime: TimeInterval = 0
    @State private var tokensPerSecond: TimeInterval = 0
    @State private var currentLag: TimeInterval = 0
    @State private var currentFallbacks: Int = 0
    @State private var currentEncodingLoops: Int = 0
    @State private var currentDecodingLoops: Int = 0
    @State private var lastBufferSize: Int = 0
    @State private var lastConfirmedSegmentEndSeconds: Float = 0
    @State private var requiredSegmentsForConfirmation: Int = 4
    @State private var bufferEnergy: [Float] = []
    @State private var bufferSeconds: Double = 0
    @State private var confirmedSegments: [TranscriptionSegment] = []
    @State private var unconfirmedSegments: [TranscriptionSegment] = []

    // MARK: Eager mode properties

    @State private var eagerResults: [TranscriptionResult?] = []
    @State private var prevResult: TranscriptionResult?
    @State private var lastAgreedSeconds: Float = 0.0
    @State private var prevWords: [WordTiming] = []
    @State private var lastAgreedWords: [WordTiming] = []
    @State private var confirmedWords: [WordTiming] = []
    @State private var confirmedText: String = ""
    @State private var hypothesisWords: [WordTiming] = []
    @State private var hypothesisText: String = ""

    // MARK: SpeakerKit

    @State private var speakerKit: SpeakerKit?
    @State private var speakerModelState: ModelState = .unloaded
    @State private var speakerModelURL: URL? = nil
    @State private var diarizationProgress: Double = 0.0
    @State private var diarizedSpeakerSegments: [SpeakerSegment] = []
    @State private var selectedDisplayTab: DisplayTab = .transcription

    @AppStorage("diarizationMode") private var diarizationModeRaw: String = "Sequential"
    @AppStorage("numberOfSpeakers") private var numberOfSpeakers: Int = 0
    @AppStorage("useExclusiveReconciliation") private var useExclusiveReconciliation: Bool = true
    @AppStorage("speakerInfoStrategy") private var speakerInfoStrategyRaw: String = "subsegment"
    @AppStorage("clustererVersion") private var clustererVersionRaw: String = "pyannote4"
    @AppStorage("segmenterComputeUnits") private var segmenterComputeUnits: MLComputeUnits = .cpuOnly
    @AppStorage("embedderComputeUnits") private var embedderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine

    @State private var lastAudioFileSamples: [Float] = []
    @State private var lastTranscriptionResult: TranscriptionResult? = nil

    enum DiarizationMode: String, CaseIterable {
        case disabled = "Disabled"
        case sequential = "Sequential"
        case concurrent = "Concurrent"
    }

    private var diarizationMode: DiarizationMode {
        DiarizationMode(rawValue: diarizationModeRaw) ?? .sequential
    }

    private var diarizationEnabled: Bool { diarizationMode != .disabled }

    private var speakerInfoStrategy: SpeakerInfoStrategy {
        speakerInfoStrategyRaw == "segment" ? .segment : .subsegment
    }

    enum DisplayTab: String, CaseIterable {
        case transcription = "Transcription"
        case speakers = "Speakers"
    }

    // MARK: UI properties

    @State private var columnVisibility: NavigationSplitViewVisibility = .all
    @State private var showWhisperKitComputeUnits: Bool = true
    @State private var showSpeakerKitComputeUnits: Bool = false
    @State private var showAdvancedOptions: Bool = false
    @State private var transcriptionTask: Task<Void, Never>?
    @State private var selectedCategoryId: MenuItem.ID?
    @State private var transcribeTask: Task<Void, Never>?
    @State private var reDiarizeTask: Task<Void, Never>?

    struct MenuItem: Identifiable, Hashable {
        var id = UUID()
        var name: String
        var image: String
    }

    private var menu = [
        MenuItem(name: "Transcribe", image: "book.pages"),
        MenuItem(name: "Stream", image: "waveform.badge.mic"),
    ]

    private var isStreamMode: Bool {
        selectedCategoryId == menu.first(where: { $0.name == "Stream" })?.id
    }

    func getComputeOptions() -> ModelComputeOptions {
        return ModelComputeOptions(audioEncoderCompute: encoderComputeUnits, textDecoderCompute: decoderComputeUnits)
    }

    // MARK: Views

    func resetState() {
        transcribeTask?.cancel()
        isRecording = false
        isTranscribing = false
        whisperKit?.audioProcessor.stopRecording()
        currentText = ""
        currentChunks = [:]

        pipelineStart = Double.greatestFiniteMagnitude
        firstTokenTime = Double.greatestFiniteMagnitude
        effectiveRealTimeFactor = 0
        effectiveSpeedFactor = 0
        totalInferenceTime = 0
        tokensPerSecond = 0
        currentLag = 0
        currentFallbacks = 0
        currentEncodingLoops = 0
        currentDecodingLoops = 0
        lastBufferSize = 0
        lastConfirmedSegmentEndSeconds = 0
        requiredSegmentsForConfirmation = 2
        bufferEnergy = []
        bufferSeconds = 0
        confirmedSegments = []
        unconfirmedSegments = []

        eagerResults = []
        prevResult = nil
        lastAgreedSeconds = 0.0
        prevWords = []
        lastAgreedWords = []
        confirmedWords = []
        confirmedText = ""
        hypothesisWords = []
        hypothesisText = ""

        diarizedSpeakerSegments = []
        diarizationProgress = 0.0
        selectedDisplayTab = .transcription
    }

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            VStack(alignment: .leading) {
                Text("WhisperKit")
                    .font(.headline)
                    .padding(.top)
                modelSelectorView
                    .padding(.bottom, 4)
                whisperKitComputeUnitsView
                    .disabled(modelState != .loaded && modelState != .unloaded)
                    .padding(.bottom)

                Divider()

                Text("SpeakerKit")
                    .font(.headline)
                    .padding(.top)
                speakerModelSelectorView
                    .padding(.bottom, 4)
                speakerKitComputeUnitsView
                    .padding(.bottom)

                List(menu, selection: $selectedCategoryId) { item in
                    HStack {
                        Image(systemName: item.image)
                        Text(item.name)
                            .font(.system(.title3))
                            .bold()
                    }
                }
                .onChange(of: selectedCategoryId) {
                    selectedTab = menu.first(where: { $0.id == selectedCategoryId })?.name ?? "Transcribe"
                }
                .disabled(modelState != .loaded)
                .foregroundColor(modelState != .loaded ? .secondary : .primary)

                Spacer()

                // New section for app and device info
                VStack(alignment: .leading, spacing: 4) {
                    let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "Unknown"
                    let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "Unknown"
                    Text("App Version: \(version) (\(build))")
                    #if os(iOS)
                    Text("Device Model: \(WhisperKit.deviceName())")
                    Text("OS Version: \(UIDevice.current.systemVersion)")
                    #elseif os(macOS)
                    Text("Device Model: \(WhisperKit.deviceName())")
                    Text("OS Version: \(ProcessInfo.processInfo.operatingSystemVersionString)")
                    #endif
                }
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(.secondary)
                .padding(.vertical)
            }
            .navigationTitle("WhisperAX")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .navigationSplitViewColumnWidth(min: 300, ideal: 350)
            .padding(.horizontal)
            Spacer()
        } detail: {
            VStack {
                #if os(iOS)
                modelSelectorView
                    .padding()
                transcriptionView
                #elseif os(macOS)
                VStack(alignment: .leading) {
                    transcriptionView
                }
                .padding()
                #endif
                controlsView
            }
            .toolbar(content: {
                ToolbarItem {
                    Button {
                        if !enableEagerDecoding {
                            let fullTranscript = TranscriptionUtilities.formatSegments(confirmedSegments + unconfirmedSegments, withTimestamps: enableTimestamps).joined(separator: "\n")
                            #if os(iOS)
                            UIPasteboard.general.string = fullTranscript
                            #elseif os(macOS)
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(fullTranscript, forType: .string)
                            #endif
                        } else {
                            #if os(iOS)
                            UIPasteboard.general.string = confirmedText + hypothesisText
                            #elseif os(macOS)
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(confirmedText + hypothesisText, forType: .string)
                            #endif
                        }
                    } label: {
                        Label("Copy Text", systemImage: "doc.on.doc")
                    }
                    .keyboardShortcut("c", modifiers: .command)
                    .foregroundColor(.primary)
                    .frame(minWidth: 0, maxWidth: .infinity)
                }
            })
        }
        .onAppear {
            #if os(macOS)
            selectedCategoryId = menu.first(where: { $0.name == selectedTab })?.id
            #else
            if UIDevice.current.userInterfaceIdiom == .pad {
                selectedCategoryId = menu.first(where: { $0.name == selectedTab })?.id
            }
            #endif
            fetchModels()
        }
    }

    // MARK: - Transcription

    var speakerBubblesView: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 8) {
                ForEach(Array(diarizedSpeakerSegments.enumerated()), id: \.offset) { index, segment in
                    let prevSpeaker = index > 0 ? diarizedSpeakerSegments[index - 1].speaker : nil
                    let speakerChanged = prevSpeaker?.speakerId != segment.speaker.speakerId

                    VStack(alignment: .leading, spacing: 2) {
                        if speakerChanged {
                            Text(speakerDisplayName(for: segment.speaker.speakerId))
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .padding(.horizontal)
                                .padding(.top, 6)
                        }
                        Text(segment.text)
                            .padding(10)
                            .background(speakerColor(for: segment.speaker.speakerId))
                            .foregroundColor(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                            .padding(.horizontal)
                    }
                }
            }
            .padding(.vertical)
        }
        .frame(maxWidth: .infinity)
    }

    func speakerColor(for speakerId: Int?) -> Color {
        guard let id = speakerId else { return Color.gray }
        switch id {
        case 0:  return Color(red: 0.224, green: 0.212, blue: 0.855)
        case 1:  return Color(red: 0.275, green: 0.639, blue: 0.408)
        case 2:  return Color(red: 0.867, green: 0.373, blue: 0.184)
        case 3:  return Color(red: 0.761, green: 0.502, blue: 0.196)
        default: return Color(red: 0.451, green: 0.439, blue: 0.400)
        }
    }

    func speakerDisplayName(for speakerId: Int?) -> String {
        guard let id = speakerId else { return "Unknown" }
        return "Speaker \(id + 1)"
    }

    var transcriptionView: some View {
        VStack {
            if !isStreamMode && diarizationEnabled && speakerKit != nil {
                Picker("", selection: $selectedDisplayTab) {
                    ForEach(DisplayTab.allCases, id: \.self) { tab in
                        Text(tab.rawValue).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)
                .disabled(diarizedSpeakerSegments.isEmpty)

                HStack {
                    Label("Speakers", systemImage: "person.2")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Spacer()
                    Picker("", selection: $numberOfSpeakers) {
                        Text("Auto").tag(0)
                        ForEach(1...6, id: \.self) { n in Text("\(n)").tag(n) }
                    }
                    .pickerStyle(MenuPickerStyle())
                    .frame(width: 80)
                    .onChange(of: numberOfSpeakers, initial: false) { _, _ in
                        reDiarizeTask?.cancel()
                        reDiarizeTask = Task { await reDiarize() }
                    }
                }
                .padding(.horizontal)
            }

            if selectedDisplayTab == .speakers && !diarizedSpeakerSegments.isEmpty {
                speakerBubblesView
            } else {
            if !bufferEnergy.isEmpty {
                ScrollView(.horizontal) {
                    HStack(spacing: 1) {
                        let startIndex = max(bufferEnergy.count - 300, 0)
                        ForEach(Array(bufferEnergy.enumerated())[startIndex...], id: \.element) { _, energy in
                            ZStack {
                                RoundedRectangle(cornerRadius: 2)
                                    .frame(width: 2, height: CGFloat(energy) * 24)
                            }
                            .frame(maxHeight: 24)
                            .background(energy > Float(silenceThreshold) ? Color.green.opacity(0.2) : Color.red.opacity(0.2))
                        }
                    }
                }
                .defaultScrollAnchor(.trailing)
                .frame(height: 24)
                .scrollIndicators(.never)
            }

            ScrollView {
                VStack(alignment: .leading) {
                    if enableEagerDecoding && isStreamMode {
                        let startSeconds = eagerResults.first??.segments.first?.start ?? 0
                        let endSeconds = lastAgreedSeconds > 0 ? lastAgreedSeconds : eagerResults.last??.segments.last?.end ?? 0
                        let timestampText = (enableTimestamps && eagerResults.first != nil) ? "[\(String(format: "%.2f", startSeconds)) --> \(String(format: "%.2f", endSeconds))]" : ""
                        Text("\(timestampText) \(Text(confirmedText).fontWeight(.bold))\(Text(hypothesisText).fontWeight(.bold).foregroundColor(.gray))")
                            .font(.headline)
                            .multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity, alignment: .leading)

                        if enableDecoderPreview {
                            Text("\(currentText)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.leading)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.top)
                        }
                    } else {
                        ForEach(Array(confirmedSegments.enumerated()), id: \.element) { _, segment in
                            let timestampText = enableTimestamps ? "[\(String(format: "%.2f", segment.start)) --> \(String(format: "%.2f", segment.end))]" : ""
                            Text(timestampText + segment.text)
                                .font(.headline)
                                .fontWeight(.bold)
                                .tint(.green)
                                .multilineTextAlignment(.leading)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        ForEach(Array(unconfirmedSegments.enumerated()), id: \.element) { _, segment in
                            let timestampText = enableTimestamps ? "[\(String(format: "%.2f", segment.start)) --> \(String(format: "%.2f", segment.end))]" : ""
                            Text(timestampText + segment.text)
                                .font(.headline)
                                .fontWeight(.bold)
                                .foregroundColor(.gray)
                                .multilineTextAlignment(.leading)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        if enableDecoderPreview {
                            Text("\(currentText)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.leading)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity)
            .defaultScrollAnchor(.bottom)
            .textSelection(.enabled)
            .padding()
            if let whisperKit,
               !isStreamMode,
               isTranscribing,
               let task = transcribeTask,
               !task.isCancelled,
               whisperKit.progress.fractionCompleted < 1
            {
                HStack {
                    ProgressView(whisperKit.progress)
                        .progressViewStyle(.linear)
                        .labelsHidden()
                        .padding(.horizontal)

                    Button {
                        transcribeTask?.cancel()
                        transcribeTask = nil
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(BorderlessButtonStyle())
                }
            }
            } // end else (speakers tab not active)

            if diarizationEnabled, diarizationProgress > 0, diarizationProgress < 1 {
                ProgressView("Diarizing...", value: diarizationProgress, total: 1)
                    .progressViewStyle(.linear)
                    .padding(.horizontal)
            }
        }
    }

    // MARK: - Models

    var modelSelectorView: some View {
        Group {
            VStack {
                HStack {
                    Image(systemName: "circle.fill")
                        .foregroundStyle(modelState == .loaded ? .green : (modelState == .unloaded ? .red : .yellow))
                        .symbolEffect(.variableColor, isActive: modelState != .loaded && modelState != .unloaded)
                    Text(modelState.description)

                    Spacer()

                    if !availableModels.isEmpty {
                        Picker("", selection: $selectedModel) {
                            ForEach(availableModels, id: \.self) { model in
                                HStack {
                                    let modelIcon = localModels.contains { $0 == model.description } ? "checkmark.circle" : "arrow.down.circle.dotted"
                                    Text("\(Image(systemName: modelIcon)) \(model.description.components(separatedBy: "_").dropFirst().joined(separator: " "))").tag(model.description)
                                }
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        .frame(minWidth: 140)
                        .onChange(of: selectedModel, initial: false) { _, _ in
                            modelState = .unloaded
                        }
                    } else {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle())
                            .scaleEffect(0.5)
                    }

                    Button(action: {
                        deleteModel()
                    }, label: {
                        Image(systemName: "trash")
                    })
                    .help("Delete model")
                    .buttonStyle(BorderlessButtonStyle())
                    .disabled(localModels.isEmpty)
                    .disabled(!localModels.contains(selectedModel))

                    #if os(macOS)
                    Button(action: {
                        let folderURL = whisperKit?.modelFolder ?? (localModels.contains(selectedModel) ? URL(fileURLWithPath: localModelPath) : nil)
                        if let folder = folderURL {
                            NSWorkspace.shared.open(folder)
                        }
                    }, label: {
                        Image(systemName: "folder")
                    })
                    .buttonStyle(BorderlessButtonStyle())
                    #endif
                    Button(action: {
                        if let url = URL(string: "https://huggingface.co/\(repoName)") {
                            #if os(macOS)
                            NSWorkspace.shared.open(url)
                            #else
                            UIApplication.shared.open(url)
                            #endif
                        }
                    }, label: {
                        Image(systemName: "link.circle")
                    })
                    .buttonStyle(BorderlessButtonStyle())
                }

                if modelState == .unloaded {
                    Divider()
                    Button {
                        resetState()
                        loadModel(selectedModel)
                        modelState = .loading
                    } label: {
                        Text("Load Model")
                            .frame(maxWidth: .infinity)
                            .frame(height: 40)
                    }
                    .buttonStyle(.borderedProminent)
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
                            Text("Specializing \(selectedModel) for your device...\nThis can take several minutes on first load")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                    }
                }
            }
        }
    }

    var speakerModelSelectorView: some View {
        HStack {
            Image(systemName: "circle.fill")
                .foregroundStyle(speakerModelState == .loaded ? .green : (speakerModelState == .unloaded ? .red : .yellow))
                .symbolEffect(.variableColor, isActive: speakerModelState != .loaded && speakerModelState != .unloaded)
            Text(speakerModelState.description)
                .foregroundColor(.secondary)

            Spacer()

            Picker("", selection: $clustererVersionRaw) {
                Text("None").tag("none")
                Text("Pyannote 4").tag("pyannote4")
            }
            .pickerStyle(MenuPickerStyle())
            .frame(minWidth: 140)
            .onChange(of: clustererVersionRaw, initial: false) { _, newValue in
                speakerKit = nil
                speakerModelState = .unloaded
                speakerModelURL = nil
                if newValue != "none" {
                    Task { await loadSpeakerKit() }
                }
            }

            Button(action: {
                if let path = speakerModelURL {
                    speakerKit = nil
                    speakerModelState = .unloaded
                    speakerModelURL = nil
                    try? FileManager.default.removeItem(at: path)
                }
            }, label: {
                Image(systemName: "trash")
            })
            .help("Delete SpeakerKit models")
            .buttonStyle(BorderlessButtonStyle())
            .disabled(speakerModelURL == nil)

            #if os(macOS)
            Button(action: {
                let folder = speakerModelURL ?? Self.speakerKitModelsURL()
                NSWorkspace.shared.open(folder)
            }, label: {
                Image(systemName: "folder")
            })
            .help("Show in Finder")
            .buttonStyle(BorderlessButtonStyle())
            .disabled(clustererVersionRaw == "none")
            #endif

            Button(action: {
                if let url = URL(string: "https://huggingface.co/argmaxinc/speakerkit-coreml") {
                    #if os(macOS)
                    NSWorkspace.shared.open(url)
                    #else
                    UIApplication.shared.open(url)
                    #endif
                }
            }, label: {
                Image(systemName: "link.circle")
            })
            .help("Open on Hugging Face")
            .buttonStyle(BorderlessButtonStyle())
            .disabled(clustererVersionRaw == "none")
        }
    }

    var whisperKitComputeUnitsView: some View {
        DisclosureGroup(isExpanded: $showWhisperKitComputeUnits) {
            VStack(alignment: .leading) {
                HStack {
                    Image(systemName: "circle.fill")
                        .foregroundStyle((whisperKit?.audioEncoder as? WhisperMLModel)?.modelState == .loaded ? .green : (modelState == .unloaded ? .red : .yellow))
                        .symbolEffect(.variableColor, isActive: modelState != .loaded && modelState != .unloaded)
                    Text("Audio Encoder")
                    Spacer()
                    Picker("", selection: $encoderComputeUnits) {
                        Text("CPU").tag(MLComputeUnits.cpuOnly)
                        Text("GPU").tag(MLComputeUnits.cpuAndGPU)
                        Text("Neural Engine").tag(MLComputeUnits.cpuAndNeuralEngine)
                    }
                    .onChange(of: encoderComputeUnits, initial: false) { _, _ in
                        loadModel(selectedModel)
                    }
                    .pickerStyle(MenuPickerStyle())
                    .frame(width: 150)
                }
                HStack {
                    Image(systemName: "circle.fill")
                        .foregroundStyle((whisperKit?.textDecoder as? WhisperMLModel)?.modelState == .loaded ? .green : (modelState == .unloaded ? .red : .yellow))
                        .symbolEffect(.variableColor, isActive: modelState != .loaded && modelState != .unloaded)
                    Text("Text Decoder")
                    Spacer()
                    Picker("", selection: $decoderComputeUnits) {
                        Text("CPU").tag(MLComputeUnits.cpuOnly)
                        Text("GPU").tag(MLComputeUnits.cpuAndGPU)
                        Text("Neural Engine").tag(MLComputeUnits.cpuAndNeuralEngine)
                    }
                    .onChange(of: decoderComputeUnits, initial: false) { _, _ in
                        loadModel(selectedModel)
                    }
                    .pickerStyle(MenuPickerStyle())
                    .frame(width: 150)
                }
            }
            .padding(.top)
        } label: {
            Button {
                showWhisperKitComputeUnits.toggle()
            } label: {
                Text("Compute Units")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            .buttonStyle(.plain)
        }
        .onChange(of: showWhisperKitComputeUnits) { _, isOpen in
            if isOpen { showSpeakerKitComputeUnits = false }
        }
    }

    var speakerKitComputeUnitsView: some View {
        DisclosureGroup(isExpanded: $showSpeakerKitComputeUnits) {
            VStack(alignment: .leading) {
                HStack {
                    Image(systemName: "circle.fill")
                        .foregroundStyle(speakerModelState == .loaded ? .green : (speakerModelState == .unloaded ? .red : .yellow))
                        .symbolEffect(.variableColor, isActive: speakerModelState != .loaded && speakerModelState != .unloaded)
                    Text("Segmenter")
                    Spacer()
                    Picker("", selection: $segmenterComputeUnits) {
                        Text("CPU").tag(MLComputeUnits.cpuOnly)
                        Text("GPU").tag(MLComputeUnits.cpuAndGPU)
                        Text("Neural Engine").tag(MLComputeUnits.cpuAndNeuralEngine)
                    }
                    .onChange(of: segmenterComputeUnits, initial: false) { _, _ in
                        speakerKit = nil
                        speakerModelState = .unloaded
                        Task { await loadSpeakerKit() }
                    }
                    .pickerStyle(MenuPickerStyle())
                    .frame(width: 150)
                }
                HStack {
                    Image(systemName: "circle.fill")
                        .foregroundStyle(speakerModelState == .loaded ? .green : (speakerModelState == .unloaded ? .red : .yellow))
                        .symbolEffect(.variableColor, isActive: speakerModelState != .loaded && speakerModelState != .unloaded)
                    Text("Embedder")
                    Spacer()
                    Picker("", selection: $embedderComputeUnits) {
                        Text("CPU").tag(MLComputeUnits.cpuOnly)
                        Text("GPU").tag(MLComputeUnits.cpuAndGPU)
                        Text("Neural Engine").tag(MLComputeUnits.cpuAndNeuralEngine)
                    }
                    .onChange(of: embedderComputeUnits, initial: false) { _, _ in
                        speakerKit = nil
                        speakerModelState = .unloaded
                        Task { await loadSpeakerKit() }
                    }
                    .pickerStyle(MenuPickerStyle())
                    .frame(width: 150)
                }
            }
            .padding(.top)
        } label: {
            Button {
                showSpeakerKitComputeUnits.toggle()
            } label: {
                Text("Compute Units")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            .buttonStyle(.plain)
        }
        .onChange(of: showSpeakerKitComputeUnits) { _, isOpen in
            if isOpen { showWhisperKitComputeUnits = false }
        }
    }

    // MARK: - Controls

    var audioDevicesView: some View {
        Group {
            #if os(macOS)
            HStack {
                if let audioDevices = audioDevices, !audioDevices.isEmpty {
                    Picker("", selection: $selectedAudioInput) {
                        ForEach(audioDevices, id: \.self) { device in
                            Text(device.name).tag(device.name)
                        }
                    }
                    .frame(width: 250)
                    .disabled(isRecording)
                }
            }
            .onAppear {
                audioDevices = AudioProcessor.getAudioDevices()
                if let audioDevices = audioDevices,
                   !audioDevices.isEmpty,
                   selectedAudioInput == "No Audio Input",
                   let device = audioDevices.first
                {
                    selectedAudioInput = device.name
                }
            }
            #endif
        }
    }

    var controlsView: some View {
        VStack {
            basicSettingsView

            if let selectedCategoryId, let item = menu.first(where: { $0.id == selectedCategoryId }) {
                switch item.name {
                    case "Transcribe":
                        VStack {
                            HStack {
                                Button {
                                    resetState()
                                } label: {
                                    Label("Reset", systemImage: "arrow.clockwise")
                                }
                                .buttonStyle(.borderless)

                                Spacer()

                                audioDevicesView

                                Spacer()

                                Button {
                                    showAdvancedOptions.toggle()
                                } label: {
                                    Label("Settings", systemImage: "slider.horizontal.3")
                                }
                                .buttonStyle(.borderless)
                            }

                            HStack {
                                let color: Color = modelState != .loaded ? .gray : .red
                                Button(action: {
                                    withAnimation {
                                        selectFile()
                                    }
                                }) {
                                    Text("FROM FILE")
                                        .font(.headline)
                                        .foregroundColor(color)
                                        .padding()
                                        .cornerRadius(40)
                                        .frame(minWidth: 70, minHeight: 70)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 40)
                                                .stroke(color, lineWidth: 4)
                                        )
                                }
                                .fileImporter(
                                    isPresented: $isFilePickerPresented,
                                    allowedContentTypes: [.audio],
                                    allowsMultipleSelection: false,
                                    onCompletion: handleFilePicker
                                )
                                .lineLimit(1)
                                .contentTransition(.symbolEffect(.replace))
                                .buttonStyle(BorderlessButtonStyle())
                                .disabled(modelState != .loaded)
                                .frame(minWidth: 0, maxWidth: .infinity)
                                .padding()

                                ZStack {
                                    Button(action: {
                                        withAnimation {
                                            toggleRecording(shouldLoop: false)
                                        }
                                    }) {
                                        if !isRecording {
                                            Text("RECORD")
                                                .font(.headline)
                                                .foregroundColor(color)
                                                .padding()
                                                .cornerRadius(40)
                                                .frame(minWidth: 70, minHeight: 70)
                                                .overlay(
                                                    RoundedRectangle(cornerRadius: 40)
                                                        .stroke(color, lineWidth: 4)
                                                )
                                        } else {
                                            Image(systemName: "stop.circle.fill")
                                                .resizable()
                                                .scaledToFit()
                                                .frame(width: 70, height: 70)
                                                .padding()
                                                .foregroundColor(modelState != .loaded ? .gray : .red)
                                        }
                                    }
                                    .lineLimit(1)
                                    .contentTransition(.symbolEffect(.replace))
                                    .buttonStyle(BorderlessButtonStyle())
                                    .disabled(modelState != .loaded)
                                    .frame(minWidth: 0, maxWidth: .infinity)
                                    .padding()

                                    if isRecording {
                                        Text("\(String(format: "%.1f", bufferSeconds)) s")
                                            .font(.caption)
                                            .foregroundColor(.gray)
                                            .offset(x: 80, y: 0)
                                    }
                                }
                            }
                        }
                    case "Stream":
                        VStack {
                            HStack {
                                Button {
                                    resetState()
                                } label: {
                                    Label("Reset", systemImage: "arrow.clockwise")
                                }
                                .frame(minWidth: 0, maxWidth: .infinity)
                                .buttonStyle(.borderless)

                                Spacer()

                                audioDevicesView

                                Spacer()

                                VStack {
                                    Button {
                                        showAdvancedOptions.toggle()
                                    } label: {
                                        Label("Settings", systemImage: "slider.horizontal.3")
                                    }
                                    .frame(minWidth: 0, maxWidth: .infinity)
                                    .buttonStyle(.borderless)
                                }
                            }

                            ZStack {
                                Button {
                                    withAnimation {
                                        toggleRecording(shouldLoop: true)
                                    }
                                } label: {
                                    Image(systemName: !isRecording ? "record.circle" : "stop.circle.fill")
                                        .resizable()
                                        .scaledToFit()
                                        .frame(width: 70, height: 70)
                                        .padding()
                                        .foregroundColor(modelState != .loaded ? .gray : .red)
                                }
                                .contentTransition(.symbolEffect(.replace))
                                .buttonStyle(BorderlessButtonStyle())
                                .disabled(modelState != .loaded)
                                .frame(minWidth: 0, maxWidth: .infinity)

                                VStack {
                                    Text("Encoder runs: \(currentEncodingLoops)")
                                        .font(.caption)
                                    Text("Decoder runs: \(currentDecodingLoops)")
                                        .font(.caption)
                                }
                                .offset(x: -120, y: 0)

                                if isRecording {
                                    Text("\(String(format: "%.1f", bufferSeconds)) s")
                                        .font(.caption)
                                        .foregroundColor(.gray)
                                        .offset(x: 80, y: 0)
                                }
                            }
                        }
                    default:
                        EmptyView()
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal)
        .sheet(isPresented: $showAdvancedOptions, content: {
            advancedSettingsView
                .presentationDetents([.medium, .large])
                .presentationBackgroundInteraction(.enabled)
                .presentationContentInteraction(.scrolls)
        })
    }

    var basicSettingsView: some View {
        VStack {
            HStack {
                Picker("", selection: $selectedTask) {
                    ForEach(DecodingTask.allCases, id: \.self) { task in
                        Text(task.description.capitalized).tag(task.description)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .disabled(!(whisperKit?.modelVariant.isMultilingual ?? false))
            }
            .padding(.horizontal)

            LabeledContent {
                Picker("", selection: $selectedLanguage) {
                    ForEach(availableLanguages, id: \.self) { language in
                        Text(language.description).tag(language.description)
                    }
                }
                .disabled(!(whisperKit?.modelVariant.isMultilingual ?? false))
            } label: {
                Label("Source Language", systemImage: "globe")
            }
            .padding(.horizontal)
            .padding(.top)

            HStack {
                Text(effectiveRealTimeFactor.formatted(.number.precision(.fractionLength(3))) + " RTF")
                    .font(.system(.body))
                    .lineLimit(1)
                Spacer()
                #if os(macOS)
                Text(effectiveSpeedFactor.formatted(.number.precision(.fractionLength(1))) + " Speed Factor")
                    .font(.system(.body))
                    .lineLimit(1)
                Spacer()
                #endif
                Text(tokensPerSecond.formatted(.number.precision(.fractionLength(0))) + " tok/s")
                    .font(.system(.body))
                    .lineLimit(1)
                Spacer()
                Text("First token: " + (firstTokenTime - pipelineStart).formatted(.number.precision(.fractionLength(2))) + "s")
                    .font(.system(.body))
                    .lineLimit(1)
            }
            .padding()
            .frame(maxWidth: .infinity)
        }
    }

    var advancedSettingsView: some View {
        #if os(iOS)
        NavigationView {
            settingsForm
                .navigationBarTitleDisplayMode(.inline)
        }
        #else
        VStack {
            Text("Decoding Options")
                .font(.title2)
                .padding()
            settingsForm
                .frame(minWidth: 500, minHeight: 500)
        }
        #endif
    }

    var settingsForm: some View {
        List {
        
                        
                    
            
            
            HStack {
                Text("Show Timestamps")
                InfoButton("Toggling this will include/exclude timestamps in both the UI and the prefill tokens.\nEither <|notimestamps|> or <|0.00|> will be forced based on this setting unless \"Prompt Prefill\" is de-selected.")
                Spacer()
                Toggle("", isOn: $enableTimestamps)
            }
            .padding(.horizontal)

            HStack {
                Text("Special Characters")
                InfoButton("Toggling this will include/exclude special characters in the transcription text.")
                Spacer()
                Toggle("", isOn: $enableSpecialCharacters)
            }
            .padding(.horizontal)

            HStack {
                Text("Show Decoder Preview")
                InfoButton("Toggling this will show a small preview of the decoder output in the UI under the transcribe. This can be useful for debugging.")
                Spacer()
                Toggle("", isOn: $enableDecoderPreview)
            }
            .padding(.horizontal)

            HStack {
                Text("Prompt Prefill")
                InfoButton("When Prompt Prefill is on, it will force the task, language, and timestamp tokens in the decoding loop. \nToggle it off if you'd like the model to generate those tokens itself instead.")
                Spacer()
                Toggle("", isOn: $enablePromptPrefill)
            }
            .padding(.horizontal)

            HStack {
                Text("Cache Prefill")
                InfoButton("When Cache Prefill is on, the decoder will try to use a lookup table of pre-computed KV caches instead of computing them during the decoding loop. \nThis allows the model to skip the compute required to force the initial prefill tokens, and can speed up inference")
                Spacer()
                Toggle("", isOn: $enableCachePrefill)
            }
            .padding(.horizontal)

            VStack {
                HStack {
                    Text("Chunking Strategy")
                    InfoButton("Select the strategy to use for chunking audio data. If VAD is selected, the audio will be chunked based on voice activity (split on silent portions).")
                    Spacer()
                    Picker("", selection: $chunkingStrategy) {
                        Text("None").tag(ChunkingStrategy.none)
                        Text("VAD").tag(ChunkingStrategy.vad)
                    }
                    .pickerStyle(SegmentedPickerStyle())
                }
                HStack {
                    Text("Workers:")
                    Slider(value: $concurrentWorkerCount, in: 0...32, step: 1)
                    Text(concurrentWorkerCount.formatted(.number))
                    InfoButton("How many workers to run transcription concurrently. Higher values increase memory usage but saturate the selected compute unit more, resulting in faster transcriptions. A value of 0 will use unlimited workers.")
                }
            }
            .padding(.horizontal)
            .padding(.bottom)
            
            TextField("Enter prompt text", text: Binding(
                get: { self.promptText ?? "" },
                set: { self.promptText = $0.isEmpty ? nil : $0 }
            ))
            .textFieldStyle(.roundedBorder)
            .padding(.horizontal)
            .padding(.bottom)

            VStack {
                Text("Starting Temperature")
                HStack {
                    Slider(value: $temperatureStart, in: 0...1, step: 0.1)
                    Text(temperatureStart.formatted(.number))
                    InfoButton("Controls the initial randomness of the decoding loop token selection.\nA higher temperature will result in more random choices for tokens, and can improve accuracy.")
                }
            }
            .padding(.horizontal)

            VStack {
                Text("Max Fallback Count")
                HStack {
                    Slider(value: $fallbackCount, in: 0...5, step: 1)
                    Text(fallbackCount.formatted(.number))
                        .frame(width: 30)
                    InfoButton("Controls how many times the decoder will fallback to a higher temperature if any of the decoding thresholds are exceeded.\n Higher values will cause the decoder to run multiple times on the same audio, which can improve accuracy at the cost of speed.")
                }
            }
            .padding(.horizontal)

            VStack {
                Text("Compression Check Tokens")
                HStack {
                    Slider(value: $compressionCheckWindow, in: 0...100, step: 5)
                    Text(compressionCheckWindow.formatted(.number))
                        .frame(width: 30)
                    InfoButton("Amount of tokens to use when checking for whether the model is stuck in a repetition loop.\nRepetition is checked by using zlib compressed size of the text compared to non-compressed value.\n Lower values will catch repetitions sooner, but too low will miss repetition loops of phrases longer than the window.")
                }
            }
            .padding(.horizontal)

            VStack {
                Text("Max Tokens Per Loop")
                HStack {
                    Slider(value: $sampleLength, in: 0...Double(min(whisperKit?.textDecoder.kvCacheMaxSequenceLength ?? Constants.maxTokenContext, Constants.maxTokenContext)), step: 10)
                    Text(sampleLength.formatted(.number))
                        .frame(width: 30)
                    InfoButton("Maximum number of tokens to generate per loop.\nCan be lowered based on the type of speech in order to further prevent repetition loops from going too long.")
                }
            }
            .padding(.horizontal)

            VStack {
                Text("Silence Threshold")
                HStack {
                    Slider(value: $silenceThreshold, in: 0...1, step: 0.05)
                    Text(silenceThreshold.formatted(.number))
                        .frame(width: 30)
                    InfoButton("Relative silence threshold for the audio. \n Baseline is set by the quietest 100ms in the previous 2 seconds.")
                }
            }
            .padding(.horizontal)

            VStack {
                Text("Realtime Delay Interval")
                HStack {
                    Slider(value: $realtimeDelayInterval, in: 0...30, step: 1)
                    Text(realtimeDelayInterval.formatted(.number))
                        .frame(width: 30)
                    InfoButton("Controls how long to wait for audio buffer to fill before running successive loops in streaming mode.\nHigher values will reduce the number of loops run per second, saving battery at the cost of higher latency.")
                }
            }
            .padding(.horizontal)

            Section(header: Text("Speaker Diarization")) {
                HStack {
                    Picker("Mode", selection: Binding(
                        get: { diarizationMode },
                        set: { diarizationModeRaw = $0.rawValue }
                    )) {
                        ForEach(DiarizationMode.allCases, id: \.self) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .disabled(speakerKit == nil)
                    Spacer()
                    HStack(spacing: 4) {
                        Image(systemName: "circle.fill")
                            .foregroundStyle(speakerModelState == .loaded ? .green : (speakerModelState == .unloaded ? .red : .yellow))
                            .font(.caption2)
                        Text(speakerModelState.description)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .padding(.horizontal)
                .padding(.top)

                if diarizationEnabled {
                    HStack {
                        Text("Exclusive Reconciliation")
                        InfoButton("When enabled, only one speaker is active per frame — prevents overlapping speaker assignments.")
                        Spacer()
                        Toggle("", isOn: $useExclusiveReconciliation)
                    }
                    .padding(.horizontal)

                    HStack {
                        Text("Speaker Info Strategy")
                        InfoButton("Segment assigns the dominant speaker to each full transcription segment. Subsegment splits segments at speaker boundaries for finer-grained attribution.")
                        Spacer()
                        Picker("", selection: $speakerInfoStrategyRaw) {
                            Text("Segment").tag("segment")
                            Text("Subsegment").tag("subsegment")
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        .fixedSize()
                    }
                    .padding(.horizontal)
                }
            }

            Section(header: Text("Experimental")) {
                HStack {
                    Text("Eager Streaming Mode")
                    InfoButton("When Eager Streaming Mode is on, the transcription will be updated more frequently, but with potentially less accurate results.")
                    Spacer()
                    Toggle("", isOn: $enableEagerDecoding)
                }
                .padding(.horizontal)
                .padding(.top)

                VStack {
                    Text("Token Confirmations")
                    HStack {
                        Slider(value: $tokenConfirmationsNeeded, in: 1...10, step: 1)
                        Text(tokenConfirmationsNeeded.formatted(.number))
                            .frame(width: 30)
                        InfoButton("Controls the number of consecutive tokens required to agree between decoder loops before considering them as confirmed in the streaming process.")
                    }
                }
                .padding(.horizontal)
            }
        }
        .navigationTitle("Decoding Options")
        .toolbar(content: {
            ToolbarItem {
                Button {
                    showAdvancedOptions = false
                } label: {
                    Label("Done", systemImage: "xmark.circle.fill")
                        .foregroundColor(.primary)
                }
            }
        })
    }

    struct InfoButton: View {
        var infoText: String
        @State private var showInfo = false

        init(_ infoText: String) {
            self.infoText = infoText
        }

        var body: some View {
            Button(action: {
                showInfo = true
            }) {
                Image(systemName: "info.circle")
                    .foregroundColor(.blue)
            }
            .popover(isPresented: $showInfo) {
                Text(infoText)
                    .padding()
            }
            .buttonStyle(BorderlessButtonStyle())
        }
    }

    // MARK: - Logic

    func fetchModels() {
        availableModels = [selectedModel]

        // First check what's already downloaded
        if let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let modelPath = documents.appendingPathComponent(modelStorage).path

            // Check if the directory exists
            if FileManager.default.fileExists(atPath: modelPath) {
                localModelPath = modelPath
                do {
                    let downloadedModels = try FileManager.default.contentsOfDirectory(atPath: modelPath)
                    for model in downloadedModels where !localModels.contains(model) {
                        localModels.append(model)
                    }
                } catch {
                    print("Error enumerating files at \(modelPath): \(error.localizedDescription)")
                }
            }
        }

        localModels = ModelUtilities.formatModelFiles(localModels)
        for model in localModels {
            if !availableModels.contains(model) {
                availableModels.append(model)
            }
        }

        print("Found locally: \(localModels)")
        print("Previously selected model: \(selectedModel)")

        Task {
            let remoteModelSupport = await WhisperKit.recommendedRemoteModels()
            await MainActor.run {
                for model in remoteModelSupport.supported {
                    if !availableModels.contains(model) {
                        availableModels.append(model)
                    }
                }
                for model in remoteModelSupport.disabled {
                    if !disabledModels.contains(model) {
                        disabledModels.append(model)
                    }
                }
            }
        }
    }

    func loadModel(_ model: String, redownload: Bool = false) {
        print("Selected Model: \(UserDefaults.standard.string(forKey: "selectedModel") ?? "nil")")
        print("""
            Computing Options:
            - Mel Spectrogram:  \(getComputeOptions().melCompute.description)
            - Audio Encoder:    \(getComputeOptions().audioEncoderCompute.description)
            - Text Decoder:     \(getComputeOptions().textDecoderCompute.description)
            - Prefill Data:     \(getComputeOptions().prefillCompute.description)
        """)

        whisperKit = nil
        Task {
            let config = WhisperKitConfig(computeOptions: getComputeOptions(),
                                          verbose: true,
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

            await MainActor.run {
                loadingProgressValue = specializationProgressRatio
                modelState = .downloaded
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
                    if !localModels.contains(model) {
                        localModels.append(model)
                    }

                    availableLanguages = Constants.languages.map { $0.key }.sorted()
                    loadingProgressValue = 1.0
                    modelState = whisperKit.modelState
                }
            }

            if clustererVersionRaw != "none" {
                await loadSpeakerKit()
            }
        }
    }

    private func loadSpeakerKit() async {
        let segCompute = segmenterComputeUnits
        let embCompute = embedderComputeUnits
        do {
            await MainActor.run { speakerModelState = .downloading }
            let config = PyannoteConfig(
                downloadBase: Self.speakerKitModelsURL().path,
                load: true
            )
            let manager = SpeakerKitDiarizer.pyannote(
                config: config,
                segmenterModelInfo: .segmenter(computeUnits: segCompute),
                embedderModelInfo: .embedder(computeUnits: embCompute)
            )
            config.diarizer = manager
            let kit = try await SpeakerKit(config)
            let resolvedURL = manager.downloader.localRepoLocation(downloadBase: Self.speakerKitModelsURL())
            await MainActor.run {
                speakerKit = kit
                speakerModelState = .loaded
                speakerModelURL = resolvedURL
            }
        } catch {
            await MainActor.run { speakerModelState = .unloaded }
            Logging.error("[WhisperAX] SpeakerKit failed to load: \(error)")
        }
    }

    private static func speakerKitModelsURL() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            .appendingPathComponent("huggingface")
    }

    func deleteModel() {
        if localModels.contains(selectedModel) {
            let modelFolder = URL(fileURLWithPath: localModelPath).appendingPathComponent(selectedModel)

            do {
                try FileManager.default.removeItem(at: modelFolder)

                if let index = localModels.firstIndex(of: selectedModel) {
                    localModels.remove(at: index)
                }

                modelState = .unloaded
            } catch {
                print("Error deleting model: \(error)")
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

    func selectFile() {
        isFilePickerPresented = true
    }

    func handleFilePicker(result: Result<[URL], Error>) {
        switch result {
            case let .success(urls):
                guard let selectedFileURL = urls.first else { return }
                if selectedFileURL.startAccessingSecurityScopedResource() {
                    do {
                        // Access the document data from the file URL
                        let audioFileData = try Data(contentsOf: selectedFileURL)

                        // Create a unique file name to avoid overwriting any existing files
                        let uniqueFileName = UUID().uuidString + "." + selectedFileURL.pathExtension

                        // Construct the temporary file URL in the app's temp directory
                        let tempDirectoryURL = FileManager.default.temporaryDirectory
                        let localFileURL = tempDirectoryURL.appendingPathComponent(uniqueFileName)

                        // Write the data to the temp directory
                        try audioFileData.write(to: localFileURL)

                        print("File saved to temporary directory: \(localFileURL)")

                        transcribeFile(path: selectedFileURL.path)
                    } catch {
                        print("File selection error: \(error.localizedDescription)")
                    }
                }
            case let .failure(error):
                print("File selection error: \(error.localizedDescription)")
        }
    }

    func transcribeFile(path: String) {
        resetState()
        whisperKit?.audioProcessor = AudioProcessor()
        transcribeTask = Task {
            isTranscribing = true
            do {
                try await transcribeCurrentFile(path: path)
            } catch {
                print("File selection error: \(error.localizedDescription)")
            }
            isTranscribing = false
        }
    }

    func toggleRecording(shouldLoop: Bool) {
        isRecording.toggle()

        if isRecording {
            resetState()
            startRecording(shouldLoop)
        } else {
            stopRecording(shouldLoop)
        }
    }

    func startRecording(_ loop: Bool) {
        if let audioProcessor = whisperKit?.audioProcessor {
            Task(priority: .userInitiated) {
                guard await AudioProcessor.requestRecordPermission() else {
                    print("Microphone access was not granted.")
                    return
                }

                var deviceId: DeviceID?
                #if os(macOS)
                if selectedAudioInput != "No Audio Input",
                   let devices = audioDevices,
                   let device = devices.first(where: { $0.name == selectedAudioInput })
                {
                    deviceId = device.id
                }

                // There is no built-in microphone
                if deviceId == nil {
                    throw WhisperError.microphoneUnavailable()
                }
                #endif

                try? audioProcessor.startRecordingLive(inputDeviceID: deviceId) { _ in
                    DispatchQueue.main.async {
                        bufferEnergy = whisperKit?.audioProcessor.relativeEnergy ?? []
                        bufferSeconds = Double(whisperKit?.audioProcessor.audioSamples.count ?? 0) / Double(WhisperKit.sampleRate)
                    }
                }

                // Delay the timer start by 1 second
                isRecording = true
                isTranscribing = true
                if loop {
                    realtimeLoop()
                }
            }
        }
    }

    func stopRecording(_ loop: Bool) {
        isRecording = false
        stopRealtimeTranscription()
        if let audioProcessor = whisperKit?.audioProcessor {
            audioProcessor.stopRecording()
        }

        // If not looping, transcribe the full recorded buffer with diarization
        if !loop {
            let recordedSamples = Array(whisperKit?.audioProcessor.audioSamples ?? [])
            transcribeTask = Task {
                isTranscribing = true
                do {
                    try await transcribeAndDiarize(recordedSamples)
                } catch {
                    print("Error: \(error.localizedDescription)")
                }
                finalizeText()
                isTranscribing = false
            }
        }

        finalizeText()
    }

    func finalizeText() {
        // Finalize unconfirmed text
        Task {
            await MainActor.run {
                if hypothesisText != "" {
                    confirmedText += hypothesisText
                    hypothesisText = ""
                }

                if !unconfirmedSegments.isEmpty {
                    confirmedSegments.append(contentsOf: unconfirmedSegments)
                    unconfirmedSegments = []
                }
            }
        }
    }

    // MARK: - Transcribe Logic

    func transcribeCurrentFile(path: String) async throws {
        Logging.debug("Loading audio file: \(path)")
        let loadingStart = Date()
        let audioFileSamples = try await Task {
            try autoreleasepool {
                try AudioProcessor.loadAudioAsFloatArray(fromPath: path)
            }
        }.value
        Logging.debug("Loaded audio file in \(Date().timeIntervalSince(loadingStart)) seconds")
        try await transcribeAndDiarize(audioFileSamples)
    }

    func transcribeAndDiarize(_ audioFileSamples: [Float]) async throws {
        await MainActor.run { lastAudioFileSamples = audioFileSamples }

        let capturedSpeakerKit: SpeakerKit? = diarizationMode == .concurrent ? speakerKit : nil
        var concurrentDiarizationTask: Task<DiarizationResult?, Never>? = nil
        defer { concurrentDiarizationTask?.cancel() }
        if let capturedSpeakerKit {
            let options = PyannoteDiarizationOptions(
                numberOfSpeakers: numberOfSpeakers == 0 ? nil : numberOfSpeakers,
                useExclusiveReconciliation: useExclusiveReconciliation
            )
            concurrentDiarizationTask = Task {
                do {
                    await MainActor.run { diarizationProgress = 0.0 }
                    return try await capturedSpeakerKit.diarize(
                        audioArray: audioFileSamples,
                        options: options,
                        progressCallback: { p in
                            Task { @MainActor in diarizationProgress = p.fractionCompleted }
                        }
                    )
                } catch {
                    Logging.error("[WhisperAX] Concurrent diarization failed: \(error)")
                    return nil
                }
            }
        }

        let transcription = try await transcribeAudioSamples(audioFileSamples)

        await MainActor.run {
            currentText = ""
            guard let segments = transcription?.segments else { return }
            tokensPerSecond = transcription?.timings.tokensPerSecond ?? 0
            effectiveRealTimeFactor = transcription?.timings.realTimeFactor ?? 0
            effectiveSpeedFactor = transcription?.timings.speedFactor ?? 0
            currentEncodingLoops = Int(transcription?.timings.totalEncodingRuns ?? 0)
            firstTokenTime = transcription?.timings.firstTokenTime ?? 0
            modelLoadingTime = transcription?.timings.modelLoading ?? 0
            pipelineStart = transcription?.timings.pipelineStart ?? 0
            currentLag = transcription?.timings.decodingLoop ?? 0
            confirmedSegments = segments
            lastTranscriptionResult = transcription
        }

        guard let transcription else { return }

        if diarizationMode == .concurrent, let task = concurrentDiarizationTask {
            if let diarizationResult = await task.value, speakerKit === capturedSpeakerKit {
                let segments = diarizationResult.addSpeakerInfo(to: [transcription], strategy: speakerInfoStrategy)
                await MainActor.run {
                    diarizedSpeakerSegments = segments.flatMap { $0 }
                    selectedDisplayTab = .speakers
                }
            }
        } else if diarizationMode == .sequential, let speakerKit {
            do {
                await MainActor.run { diarizationProgress = 0.0 }
                let options = PyannoteDiarizationOptions(
                    numberOfSpeakers: numberOfSpeakers == 0 ? nil : numberOfSpeakers,
                    useExclusiveReconciliation: useExclusiveReconciliation
                )
                let diarizationResult = try await speakerKit.diarize(
                    audioArray: audioFileSamples,
                    options: options,
                    progressCallback: { p in
                        Task { @MainActor in diarizationProgress = p.fractionCompleted }
                    }
                )
                let segments = diarizationResult.addSpeakerInfo(to: [transcription], strategy: speakerInfoStrategy)
                await MainActor.run {
                    diarizedSpeakerSegments = segments.flatMap { $0 }
                    selectedDisplayTab = .speakers
                }
            } catch {
                Logging.error("[WhisperAX] Diarization failed: \(error)")
            }
        }
    }

    func reDiarize() async {
        guard diarizationEnabled, let speakerKit, let transcription = lastTranscriptionResult, !lastAudioFileSamples.isEmpty else { return }
        do {
            let options = PyannoteDiarizationOptions(
                numberOfSpeakers: numberOfSpeakers == 0 ? nil : numberOfSpeakers,
                useExclusiveReconciliation: useExclusiveReconciliation
            )
            let diarizationResult = try await speakerKit.diarize(audioArray: lastAudioFileSamples, options: options)
            let segments = diarizationResult.addSpeakerInfo(to: [transcription], strategy: speakerInfoStrategy)
            await MainActor.run {
                diarizedSpeakerSegments = segments.flatMap { $0 }
                selectedDisplayTab = .speakers
            }
        } catch {
            Logging.error("[WhisperAX] Re-diarization failed: \(error)")
        }
    }

    func transcribeAudioSamples(_ samples: [Float]) async throws -> TranscriptionResult? {
        guard let whisperKit = whisperKit else { return nil }

        let languageCode = Constants.languages[selectedLanguage, default: Constants.defaultLanguageCode]
        let task: DecodingTask = selectedTask == "transcribe" ? .transcribe : .translate
        let seekClip: [Float] = [lastConfirmedSegmentEndSeconds]

        var options = DecodingOptions(
            verbose: true,
            task: task,
            language: languageCode,
            temperature: Float(temperatureStart),
            temperatureFallbackCount: Int(fallbackCount),
            sampleLength: Int(sampleLength),
            usePrefillPrompt: enablePromptPrefill,
            usePrefillCache: enableCachePrefill,
            skipSpecialTokens: !enableSpecialCharacters,
            withoutTimestamps: !enableTimestamps,
            wordTimestamps: true,
            clipTimestamps: seekClip,
            concurrentWorkerCount: Int(concurrentWorkerCount),
            chunkingStrategy: chunkingStrategy
        )
        
        // Prompt
        if let promptText = promptText {
            guard whisperKit.tokenizer != nil else {
                throw WhisperError.tokenizerUnavailable()
            }
            
            if promptText.count > 0, let tokenizer = whisperKit.tokenizer {
                options.promptTokens = tokenizer.encode(text: " " + promptText.trimmingCharacters(in: .whitespaces)).filter { $0 < tokenizer.specialTokens.specialTokenBegin }
                options.usePrefillPrompt = true
            }
        }


        // Early stopping checks
        let decodingCallback: ((TranscriptionProgress) -> Bool?) = { (progress: TranscriptionProgress) in
            DispatchQueue.main.async {
                let fallbacks = Int(progress.timings.totalDecodingFallbacks)
                let chunkId = isStreamMode ? 0 : progress.windowId

                // First check if this is a new window for the same chunk, append if so
                var updatedChunk = (chunkText: [progress.text], fallbacks: fallbacks)
                if var currentChunk = currentChunks[chunkId], let previousChunkText = currentChunk.chunkText.last {
                    if progress.text.count >= previousChunkText.count {
                        // This is the same window of an existing chunk, so we just update the last value
                        currentChunk.chunkText[currentChunk.chunkText.endIndex - 1] = progress.text
                        updatedChunk = currentChunk
                    } else {
                        // This is either a new window or a fallback (only in streaming mode)
                        if fallbacks == currentChunk.fallbacks && isStreamMode {
                            // New window (since fallbacks havent changed)
                            updatedChunk.chunkText = [updatedChunk.chunkText.first ?? "" + progress.text]
                        } else {
                            // Fallback, overwrite the previous bad text
                            updatedChunk.chunkText[currentChunk.chunkText.endIndex - 1] = progress.text
                            updatedChunk.fallbacks = fallbacks
                            print("Fallback occured: \(fallbacks)")
                        }
                    }
                }

                // Set the new text for the chunk
                currentChunks[chunkId] = updatedChunk
                let joinedChunks = currentChunks.sorted { $0.key < $1.key }.flatMap { $0.value.chunkText }.joined(separator: "\n")

                currentText = joinedChunks
                currentFallbacks = fallbacks
                currentDecodingLoops += 1
            }

            // Check early stopping
            let currentTokens = progress.tokens
            let checkWindow = Int(compressionCheckWindow)
            if currentTokens.count > checkWindow {
                let checkTokens: [Int] = currentTokens.suffix(checkWindow)
                let compressionRatio = TextUtilities.compressionRatio(of: checkTokens)
                if compressionRatio > options.compressionRatioThreshold! {
                    Logging.debug("Early stopping due to compression threshold")
                    return false
                }
            }
            if progress.avgLogprob! < options.logProbThreshold! {
                Logging.debug("Early stopping due to logprob threshold")
                return false
            }
            return nil
        }

        let segmentCallback: SegmentDiscoveryCallback = { segments in
            // Log segments as they are discovered from the segment discovery callback
            for segment in segments {
                Logging.debug("Discovered segment: \(segment.id) (\(segment.seek))): \(segment.start) -> \(segment.end) \(segment.text)")
            }
        }

        whisperKit.segmentDiscoveryCallback = segmentCallback

        let transcriptionResults: [TranscriptionResult] = try await whisperKit.transcribe(
            audioArray: samples,
            decodeOptions: options,
            callback: decodingCallback
        )

        let mergedResults = TranscriptionUtilities.mergeTranscriptionResults(transcriptionResults)

        return mergedResults
    }

    // MARK: Streaming Logic

    func realtimeLoop() {
        transcriptionTask = Task {
            while isRecording && isTranscribing {
                do {
                    try await transcribeCurrentBuffer(delayInterval: Float(realtimeDelayInterval))
                } catch {
                    print("Error: \(error.localizedDescription)")
                    break
                }
            }
        }
    }

    func stopRealtimeTranscription() {
        isTranscribing = false
        transcriptionTask?.cancel()
    }

    func transcribeCurrentBuffer(delayInterval: Float = 1.0) async throws {
        guard let whisperKit = whisperKit else { return }

        // Retrieve the current audio buffer from the audio processor
        let currentBuffer = whisperKit.audioProcessor.audioSamples

        // Calculate the size and duration of the next buffer segment
        let nextBufferSize = currentBuffer.count - lastBufferSize
        let nextBufferSeconds = Float(nextBufferSize) / Float(WhisperKit.sampleRate)

        // Only run the transcribe if the next buffer has at least `delayInterval` seconds of audio
        guard nextBufferSeconds > delayInterval else {
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

                // TODO: Implement silence buffer purging
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

        // Store this for next iterations VAD
        lastBufferSize = currentBuffer.count

        if enableEagerDecoding && isStreamMode {
            // Run realtime transcribe using word timestamps for segmentation
            let transcription = try await transcribeEagerMode(Array(currentBuffer))
            await MainActor.run {
                currentText = ""
                tokensPerSecond = transcription?.timings.tokensPerSecond ?? 0
                firstTokenTime = transcription?.timings.firstTokenTime ?? 0
                modelLoadingTime = transcription?.timings.modelLoading ?? 0
                pipelineStart = transcription?.timings.pipelineStart ?? 0
                currentLag = transcription?.timings.decodingLoop ?? 0
                currentEncodingLoops = Int(transcription?.timings.totalEncodingRuns ?? 0)

                let totalAudio = Double(currentBuffer.count) / Double(WhisperKit.sampleRate)
                totalInferenceTime = transcription?.timings.fullPipeline ?? 0
                effectiveRealTimeFactor = Double(totalInferenceTime) / totalAudio
                effectiveSpeedFactor = totalAudio / Double(totalInferenceTime)
            }
        } else {
            // Run realtime transcribe using timestamp tokens directly
            let transcription = try await transcribeAudioSamples(Array(currentBuffer))

            // We need to run this next part on the main thread
            await MainActor.run {
                currentText = ""
                guard let segments = transcription?.segments else {
                    return
                }

                tokensPerSecond = transcription?.timings.tokensPerSecond ?? 0
                firstTokenTime = transcription?.timings.firstTokenTime ?? 0
                modelLoadingTime = transcription?.timings.modelLoading ?? 0
                pipelineStart = transcription?.timings.pipelineStart ?? 0
                currentLag = transcription?.timings.decodingLoop ?? 0
                currentEncodingLoops += Int(transcription?.timings.totalEncodingRuns ?? 0)

                let totalAudio = Double(currentBuffer.count) / Double(WhisperKit.sampleRate)
                totalInferenceTime += transcription?.timings.fullPipeline ?? 0
                effectiveRealTimeFactor = Double(totalInferenceTime) / totalAudio
                effectiveSpeedFactor = totalAudio / Double(totalInferenceTime)

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
                        print("Last confirmed segment end: \(lastConfirmedSegmentEndSeconds)")

                        // Add confirmed segments to the confirmedSegments array
                        for segment in confirmedSegmentsArray {
                            if !confirmedSegments.contains(segment: segment) {
                                confirmedSegments.append(segment)
                            }
                        }
                    }

                    // Update transcriptions to reflect the remaining segments
                    unconfirmedSegments = remainingSegments
                } else {
                    // Handle the case where segments are fewer or equal to required
                    unconfirmedSegments = segments
                }
            }
        }
    }

    func transcribeEagerMode(_ samples: [Float]) async throws -> TranscriptionResult? {
        guard let whisperKit = whisperKit else { return nil }

        guard whisperKit.textDecoder.supportsWordTimestamps else {
            confirmedText = "Eager mode requires word timestamps, which are not supported by the current model: \(selectedModel)."
            return nil
        }

        let languageCode = Constants.languages[selectedLanguage, default: Constants.defaultLanguageCode]
        let task: DecodingTask = selectedTask == "transcribe" ? .transcribe : .translate
        print(selectedLanguage)
        print(languageCode)

        var options = DecodingOptions(
            verbose: true,
            task: task,
            language: languageCode,
            temperature: Float(temperatureStart),
            temperatureFallbackCount: Int(fallbackCount),
            sampleLength: Int(sampleLength),
            usePrefillPrompt: enablePromptPrefill,
            usePrefillCache: enableCachePrefill,
            skipSpecialTokens: !enableSpecialCharacters,
            withoutTimestamps: !enableTimestamps,
            wordTimestamps: true, // required for eager mode
            firstTokenLogProbThreshold: -1.5, // higher threshold to prevent fallbacks from running to often
            chunkingStrategy: ChunkingStrategy.none
        )
        
        // Prompt
        if let promptText = promptText {
            guard whisperKit.tokenizer != nil else {
                throw WhisperError.tokenizerUnavailable()
            }
            
            if promptText.count > 0, let tokenizer = whisperKit.tokenizer {
                options.promptTokens = tokenizer.encode(text: " " + promptText.trimmingCharacters(in: .whitespaces)).filter { $0 < tokenizer.specialTokens.specialTokenBegin }
                options.usePrefillPrompt = true
            }
        }

        // Early stopping checks
        let decodingCallback: ((TranscriptionProgress) -> Bool?) = { progress in
            DispatchQueue.main.async {
                let fallbacks = Int(progress.timings.totalDecodingFallbacks)
                if progress.text.count < currentText.count {
                    if fallbacks == currentFallbacks {
                        //                        self.unconfirmedText.append(currentText)
                    } else {
                        print("Fallback occured: \(fallbacks)")
                    }
                }
                currentText = progress.text
                currentFallbacks = fallbacks
                currentDecodingLoops += 1
            }
            // Check early stopping
            let currentTokens = progress.tokens
            let checkWindow = Int(compressionCheckWindow)
            if currentTokens.count > checkWindow {
                let checkTokens: [Int] = currentTokens.suffix(checkWindow)
                let compressionRatio = TextUtilities.compressionRatio(of: checkTokens)
                if compressionRatio > options.compressionRatioThreshold! {
                    Logging.debug("Early stopping due to compression threshold")
                    return false
                }
            }
            if progress.avgLogprob! < options.logProbThreshold! {
                Logging.debug("Early stopping due to logprob threshold")
                return false
            }

            return nil
        }

        Logging.info("[EagerMode] \(lastAgreedSeconds)-\(Double(samples.count) / 16000.0) seconds")

        let segmentCallback: SegmentDiscoveryCallback = { segments in
            // Log segments as they are discovered from the segment discovery callback
            for segment in segments {
                Logging.debug("Discovered segment: \(segment.id) (\(segment.seek))): \(segment.start) -> \(segment.end)")
            }
        }

        whisperKit.segmentDiscoveryCallback = segmentCallback

        let streamingAudio = samples
        var streamOptions = options
        streamOptions.clipTimestamps = [lastAgreedSeconds]
        let lastAgreedTokens = lastAgreedWords.flatMap { $0.tokens }
        streamOptions.prefixTokens = lastAgreedTokens
        do {
            let transcription: TranscriptionResult? = try await whisperKit.transcribe(audioArray: streamingAudio, decodeOptions: streamOptions, callback: decodingCallback).first
            await MainActor.run {
                var skipAppend = false
                if let result = transcription {
                    hypothesisWords = result.allWords.filter { $0.start >= lastAgreedSeconds }

                    if let prevResult = prevResult {
                        prevWords = prevResult.allWords.filter { $0.start >= lastAgreedSeconds }
                        let commonPrefix = TranscriptionUtilities.findLongestCommonPrefix(prevWords, hypothesisWords)
                        Logging.info("[EagerMode] Prev \"\((prevWords.map { $0.word }).joined())\"")
                        Logging.info("[EagerMode] Next \"\((hypothesisWords.map { $0.word }).joined())\"")
                        Logging.info("[EagerMode] Found common prefix \"\((commonPrefix.map { $0.word }).joined())\"")

                        if commonPrefix.count >= Int(tokenConfirmationsNeeded) {
                            lastAgreedWords = commonPrefix.suffix(Int(tokenConfirmationsNeeded))
                            lastAgreedSeconds = lastAgreedWords.first!.start
                            Logging.info("[EagerMode] Found new last agreed word \"\(lastAgreedWords.first!.word)\" at \(lastAgreedSeconds) seconds")

                            confirmedWords.append(contentsOf: commonPrefix.prefix(commonPrefix.count - Int(tokenConfirmationsNeeded)))
                            let currentWords = confirmedWords.map { $0.word }.joined()
                            Logging.info("[EagerMode] Current:  \(lastAgreedSeconds) -> \(Double(samples.count) / 16000.0) \(currentWords)")
                        } else {
                            Logging.info("[EagerMode] Using same last agreed time \(lastAgreedSeconds)")
                            skipAppend = true
                        }
                    }
                    prevResult = result
                }

                if !skipAppend {
                    eagerResults.append(transcription)
                }
            }

            await MainActor.run {
                let finalWords = confirmedWords.map { $0.word }.joined()
                confirmedText = finalWords

                // Accept the final hypothesis because it is the last of the available audio
                let lastHypothesis = lastAgreedWords + TranscriptionUtilities.findLongestDifferentSuffix(prevWords, hypothesisWords)
                hypothesisText = lastHypothesis.map { $0.word }.joined()
            }
        } catch {
            Logging.error("[EagerMode] Error: \(error)")
            finalizeText()
        }

        let mergedResult = TranscriptionUtilities.mergeTranscriptionResults(eagerResults, confirmedWords: confirmedWords)

        return mergedResult
    }
}

#Preview {
    ContentView()
    #if os(macOS)
        .frame(width: 800, height: 500)
    #endif
}
