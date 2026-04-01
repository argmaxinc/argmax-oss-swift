//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import SwiftUI
import TTSKit
import UniformTypeIdentifiers
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

struct DetailView: View {
    @Environment(ViewModel.self) private var viewModel

    @State private var isImportingFile = false
    @State private var fileImportError: String? = nil

    var body: some View {
        @Bindable var vm = viewModel

        VStack(spacing: 0) {
            // Waveform + playback controls (~top half)
            waveformSection
                .frame(minHeight: 140)

            Divider()

            // Text input + controls (bottom half)
            VStack(spacing: 16) {
                textInputSection
                metricsRow
                controlsBar
            }
            .padding(20)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .navigationTitle("TTSKit Example")
        #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
        #endif
            .toolbar {
                toolbarContent
            }
            .sheet(isPresented: $vm.showGenerationSettings) {
                GenerationSettingsView()
                    .environment(viewModel)
                    .presentationDetents([.medium, .large])
                    .presentationBackgroundInteraction(.enabled)
                    .presentationContentInteraction(.scrolls)
            }
            .onAppear {
                viewModel.onAppear()
            }
            .onChange(of: viewModel.selectedGenerationID) {
                guard !viewModel.isStreaming else { return }
                guard let gen = viewModel.selectedGeneration else { return }
                viewModel.loadWaveform(for: gen)
                // Auto-populate input fields so the user can play with the settings and regenerate
                viewModel.loadInputs(from: gen)
            }
    }

    // MARK: - Waveform Section

    @ViewBuilder
    private var waveformSection: some View {
        VStack(spacing: 12) {
            Spacer(minLength: 8)

            // All three states live in the layout simultaneously so the height
            // never changes - only opacity transitions between them.
            ZStack {
                // Idle empty placeholder
                VStack(spacing: 8) {
                    Image(systemName: "waveform")
                        .font(.system(size: 40))
                        .foregroundStyle(.quaternary)
                    Text("Waveform will appear here")
                        .font(.subheadline)
                        .foregroundStyle(.tertiary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .opacity(viewModel.currentWaveform.isEmpty && viewModel.generationState != .generating ? 1 : 0)

                // Generating progress
                VStack(spacing: 8) {
                    if viewModel.totalSteps > 0 {
                        ProgressView(
                            value: Double(viewModel.stepsCompleted),
                            total: Double(viewModel.totalSteps)
                        )
                        .progressViewStyle(.linear)
                        .frame(maxWidth: 200)

                        let pct = Int(Double(viewModel.stepsCompleted) / Double(viewModel.totalSteps) * 100)
                        let detail = viewModel.chunksTotal > 1 ? " (\(viewModel.chunksTotal) chunks)" : ""
                        Text("Generating... \(pct)%\(detail)")
                            .font(.subheadline)
                            .foregroundStyle(.tertiary)
                            .contentTransition(.numericText())
                    } else {
                        ProgressView()
                            .controlSize(.small)
                        Text(viewModel.statusMessage)
                            .font(.subheadline)
                            .foregroundStyle(.tertiary)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .opacity(viewModel.currentWaveform.isEmpty && viewModel.generationState == .generating ? 1 : 0)

                // Live / completed waveform
                WaveformView(
                    samples: viewModel.currentWaveform,
                    playbackTime: viewModel.playbackTime,
                    totalDuration: viewModel.currentDuration
                )
                .padding(.horizontal, 0)
                .opacity(viewModel.currentWaveform.isEmpty ? 0 : 1)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Time labels + playback controls
            HStack {
                HStack(spacing: 4) {
                    Text(formatTime(viewModel.playbackTime))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)

                    let bufferRemaining = viewModel.silentBufferRemaining
                    if bufferRemaining > 0.4 {
                        Text("(\(formatTime(bufferRemaining)))")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.tertiary)
                    }
                }

                Spacer()

                playbackControls

                Spacer()

                Text(formatTime(viewModel.currentDuration))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 24)

            Spacer(minLength: 8)
        }
        .background(.quaternary.opacity(0.3))
    }

    private var playbackControls: some View {
        let hasAudio = viewModel.selectedGeneration?.audioFileName != nil
        let icon = viewModel.isPlaying ? "stop.fill" : "play.fill"
        return Button {
            if viewModel.isPlaying {
                viewModel.stopPlayback()
            } else if let gen = viewModel.selectedGeneration {
                viewModel.playGeneration(gen)
            }
        } label: {
            Image(systemName: icon)
                .font(.title2)
        }
        .buttonStyle(.plain)
        .keyboardShortcut(.space, modifiers: [])
        .disabled(!hasAudio || viewModel.generationState == .generating)
        .opacity(hasAudio ? 1 : 0)
        .accessibilityLabel(viewModel.isPlaying ? "Stop playback" : "Play audio")
        .accessibilityHint(viewModel.isPlaying ? "Stops the current audio" : "Plays the selected generation")
    }

    // MARK: - Text Input

    @ViewBuilder
    private var textInputSection: some View {
        @Bindable var vm = viewModel

        VStack(alignment: .leading, spacing: 8) {
            let promptForInput = viewModel.modelState == .loaded
                && viewModel.inputText.isEmpty
                && viewModel.generationState == .idle

            ZStack(alignment: .topLeading) {
                TextEditor(text: $vm.inputText)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 8)
                    .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
                    .overlay {
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.accentColor, lineWidth: promptForInput ? 1.5 : 0)
                            .animation(.easeInOut(duration: 0.25), value: promptForInput)
                    }
                    .frame(minHeight: 80, maxHeight: 160)
//                    .overlay(alignment: .topTrailing) {
//                        Button {
//                            isImportingFile = true
//                        } label: {
//                            Image(systemName: "square.and.arrow.down")
//                                .font(.caption)
//                                .padding(6)
//                                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 6))
//                        }
//                        .buttonStyle(.plain)
//                        .padding(6)
//                        .accessibilityLabel("Import text file")
//                        .accessibilityHint("Opens a file picker to import a .txt or .pdf file into the text editor")
//                    }

                if vm.inputText.isEmpty {
                    Text("Enter text to speak...")
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .allowsHitTesting(false)
                }
            }
            .fileImporter(
                isPresented: $isImportingFile,
                allowedContentTypes: [.plainText, .pdf],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    guard let url = urls.first else { return }
                    // Security-scoped access is required for files picked outside the sandbox
                    let accessing = url.startAccessingSecurityScopedResource()
                    defer { if accessing { url.stopAccessingSecurityScopedResource() } }
                    if let text = FileUtilities.readTextContent(at: url) {
                        vm.inputText = text
                    } else {
                        fileImportError = "Could not read text from \"\(url.lastPathComponent)\"."
                    }
                case .failure(let error):
                    fileImportError = error.localizedDescription
                }
            }
            .alert("Import Failed", isPresented: .init(
                get: { fileImportError != nil },
                set: { if !$0 { fileImportError = nil } }
            )) {
                Button("OK", role: .cancel) { fileImportError = nil }
            } message: {
                Text(fileImportError ?? "")
            }

            HStack(spacing: 8) {
                Picker("Voice", selection: $vm.selectedSpeaker) {
                    ForEach(Qwen3Speaker.allCases, id: \.self) { speaker in
                        Text("\(speaker.displayName) · \(speaker.nativeLanguage)").tag(speaker)
                    }
                }
                .fixedSize()
                .accessibilityLabel("Voice")
                .accessibilityHint("Choose the speaker voice for synthesis")

                Picker("Language", selection: $vm.selectedLanguage) {
                    ForEach(Qwen3Language.allCases, id: \.self) { lang in
                        Text(lang.rawValue.capitalized).tag(lang)
                    }
                }
                .fixedSize()
                .accessibilityLabel("Language")
                .accessibilityHint("Choose the output language for the voice")

                Picker("Playback", selection: $vm.playbackStrategyTag) {
                    Text("Auto").tag("auto")
                    Text("Stream").tag("stream")
                    Text("Generate First").tag("generateFirst")
                }
                .fixedSize()
                .accessibilityLabel("Playback strategy")
                .accessibilityHint("Auto buffers adaptively; Stream plays frame-by-frame; Generate First waits for the full audio")

                Spacer(minLength: 0)

                Text(vm.inputTokenCount.map { "\($0) tok" } ?? "~\(vm.inputText.unicodeScalars.count / 5) tok")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }

            // Selected speaker description
            Text(vm.selectedSpeaker.voiceDescription)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .transition(.opacity)
                .animation(.easeInOut(duration: 0.2), value: vm.selectedSpeaker)

            // Instruction / style prompt (1.7B only)
            let instructionSupported = vm.selectedPreset.supportsVoiceDirection
            if instructionSupported {
                HStack(spacing: 6) {
                    Image(systemName: "theatermasks")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    TextField(
                        "Style instruction (e.g. cheerful and energetic)...",
                        text: $vm.instruction
                    )
                    .font(.callout)
                    .textFieldStyle(.plain)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
                .transition(.opacity.combined(with: .move(edge: .top)))
                .animation(.easeInOut(duration: 0.2), value: instructionSupported)
            }
        }
    }

    // MARK: - Metrics Row

    private var metricsRow: some View {
        let hasMetrics = viewModel.currentRTF > 0
        return HStack(spacing: 0) {
            metricItem(
                value: hasMetrics ? String(format: "%.1f×", viewModel.currentSpeedFactor) : "-",
                label: "Speed Factor"
            )
            Divider().frame(height: 24)
            metricItem(
                value: hasMetrics ? String(format: "%.0f", viewModel.currentStepsPerSecond) : "-",
                label: "steps/s"
            )
            Divider().frame(height: 24)
            metricItem(
                value: hasMetrics ? String(format: "%.2fs", viewModel.currentTimeToFirstBuffer) : "-",
                label: "First Buffer"
            )
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 6)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
        .opacity(hasMetrics ? 1 : 0.4)
        .animation(.easeInOut(duration: 0.2), value: hasMetrics)
    }

    private func metricItem(value: String, label: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(.body, design: .monospaced))
                .lineLimit(1)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Controls Bar

    @ViewBuilder
    private var controlsBar: some View {
        #if os(iOS)
        // On iOS: generate button spans full width; secondary controls on the row below.
        // This prevents the button label from being clipped on narrow screens.
        VStack(spacing: 10) {
            Button {
                if viewModel.generationState == .idle {
                    viewModel.startGeneration()
                } else {
                    viewModel.cancelGeneration()
                }
            } label: {
                Label(generateButtonTitle, systemImage: generateButtonIcon)
                    .frame(maxWidth: .infinity)
            }
            .glassButton(prominent: true)
            .tint(viewModel.generationState == .idle ? .accentColor : .red)
            .controlSize(.large)
            .disabled(!viewModel.canGenerate && viewModel.generationState == .idle)
            .accessibilityLabel(viewModel.generationState == .idle ? generateButtonTitle : "Cancel generation")
            .accessibilityHint(viewModel.generationState == .idle
                ? "Synthesizes the entered text using the loaded model"
                : "Stops the current generation immediately")

            HStack {
                Button {
                    viewModel.clearInput()
                } label: {
                    Label("Clear", systemImage: "xmark.circle")
                }
                .glassButton()
                .controlSize(.regular)
                .disabled(viewModel.inputText.isEmpty)
                .opacity(viewModel.generationState == .idle ? 1 : 0)
                .accessibilityLabel("Clear input")
                .accessibilityHint("Clears the text input and resets the waveform")

                Spacer()

                Text(viewModel.statusMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .multilineTextAlignment(.trailing)
            }
        }
        #else
        HStack(spacing: 12) {
            // Generate / Cancel button
            Button {
                if viewModel.generationState == .idle {
                    viewModel.startGeneration()
                } else {
                    viewModel.cancelGeneration()
                }
            } label: {
                Label(
                    generateButtonTitle,
                    systemImage: generateButtonIcon
                )
                .frame(maxWidth: 200)
            }
            .glassButton(prominent: true)
            .tint(viewModel.generationState == .idle ? .accentColor : .red)
            .controlSize(.large)
            .disabled(!viewModel.canGenerate && viewModel.generationState == .idle)
            .accessibilityLabel(viewModel.generationState == .idle ? generateButtonTitle : "Cancel generation")
            .accessibilityHint(viewModel.generationState == .idle
                ? "Synthesizes the entered text using the loaded model"
                : "Stops the current generation immediately")

            Button {
                viewModel.clearInput()
            } label: {
                Label("Clear", systemImage: "xmark.circle")
            }
            .glassButton()
            .controlSize(.large)
            .disabled(viewModel.inputText.isEmpty)
            .opacity(viewModel.generationState == .idle ? 1 : 0)
            .accessibilityLabel("Clear input")
            .accessibilityHint("Clears the text input and resets the waveform")

            Spacer()

            // Status
            Text(viewModel.statusMessage)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
        }
        #endif
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        #if os(macOS)
        ToolbarItem {
            Button(action: newGeneration) {
                Label("New Generation", systemImage: "plus")
            }
            .keyboardShortcut("n")
        }
        #endif

        ToolbarItem(placement: .primaryAction) {
            Button {
                viewModel.showGenerationSettings = true
            } label: {
                Label("Generation Settings", systemImage: "slider.horizontal.3")
            }
        }

        // Share / export audio file
        if let gen = viewModel.selectedGeneration,
           let url = viewModel.audioFileURL(for: gen)
        {
            ToolbarItem {
                AudioCopyButton(url: url)
            }
        }
    }

    private func newGeneration() {
        viewModel.clearInput()
        viewModel.currentWaveform = []
        viewModel.selectedGenerationID = ViewModel.newGenerationSentinel
    }

    // MARK: - Helpers

    private var generateButtonTitle: String {
        switch viewModel.generationState {
            case .generating: return "Cancel"
            case .idle:
                switch viewModel.modelState {
                    case .downloading: return "Downloading..."
                    case .prewarming: return "Specializing..."
                    case .loading: return "Loading..."
                    default: return viewModel.selectedGeneration != nil ? "Regenerate" : "Generate"
                }
        }
    }

    private var generateButtonIcon: String {
        switch viewModel.generationState {
            case .generating: return "stop.fill"
            case .idle: return "play.fill"
        }
    }

    private func formatTime(_ seconds: TimeInterval) -> String {
        let m = Int(seconds) / 60
        let s = Int(seconds) % 60
        return String(format: "%d:%02d", m, s)
    }
}

// MARK: - Audio file transferable

// MARK: - Copy to clipboard button

/// Copies the audio file to the system clipboard so it can be pasted into Mail, Finder, etc.
/// Uses `NSPasteboard` on macOS and `UIPasteboard` on iOS - both accept a file URL directly,
/// letting the receiving app decide how to handle the file.
struct AudioCopyButton: View {
    let url: URL
    @State private var copied = false

    var body: some View {
        Button {
            copyToClipboard()
            copied = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { copied = false }
        } label: {
            Label(copied ? "Copied!" : "Copy Audio", systemImage: copied ? "checkmark" : "doc.on.doc")
        }
        .accessibilityLabel(copied ? "Copied to clipboard" : "Copy audio file")
        .accessibilityHint("Copies the audio file to the clipboard so you can paste it into other apps")
    }

    private func copyToClipboard() {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.writeObjects([url as NSURL])
        #else
        guard let data = try? Data(contentsOf: url) else { return }
        let uti = UTType(filenameExtension: url.pathExtension) ?? .audio
        UIPasteboard.general.setData(data, forPasteboardType: uti.identifier)
        #endif
    }
}

// MARK: - Glass Button Style

/// Applies `.glass` on iOS/macOS 26+ and falls back to `.borderedProminent` /
/// `.bordered` on older OS versions.
struct GlassButtonModifier: ViewModifier {
    var prominent: Bool = false

    func body(content: Content) -> some View {
        if #available(iOS 26, macOS 26, watchOS 26, visionOS 26, *) {
            content
                .buttonStyle(.glass)
        } else if prominent {
            content
                .buttonStyle(.borderedProminent)
        } else {
            content
                .buttonStyle(.bordered)
        }
    }
}

extension View {
    func glassButton(prominent: Bool = false) -> some View {
        modifier(GlassButtonModifier(prominent: prominent))
    }
}
