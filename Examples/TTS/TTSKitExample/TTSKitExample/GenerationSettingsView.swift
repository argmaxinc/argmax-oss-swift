//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import SwiftUI
import TTSKit

/// Sheet for configuring GenerationOptions.
struct GenerationSettingsView: View {
    @Environment(ViewModel.self) private var viewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        #if os(iOS)
        NavigationStack {
            settingsForm
        }
        #else
        VStack(spacing: 0) {
            // Explicit header with title and close button - toolbar items
            // don't render reliably inside macOS sheet NavigationStacks.
            HStack {
                Text("Generation Options")
                    .font(.title2)
                    .fontWeight(.semibold)
                Spacer()
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title2)
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal)
            .padding(.vertical, 12)

            Divider()

            settingsForm
                .frame(minWidth: 480, minHeight: 500)
        }
        #endif
    }

    private var settingsForm: some View {
        @Bindable var vm = viewModel

        return Form {
            // MARK: Sampling

            Section("Sampling") {
                sliderRow(
                    label: "Temperature",
                    info: "Controls the randomness of token selection. Higher values produce more varied speech; lower values are more deterministic.",
                    value: $vm.temperature,
                    range: 0...1,
                    step: 0.05,
                    display: String(format: "%.2f", vm.temperature)
                )

                sliderRow(
                    label: "Top-K",
                    info: "Limits token selection to the top-K most probable tokens at each step. Lower values make output more focused.",
                    value: $vm.topK,
                    range: 1...100,
                    step: 1,
                    display: Int(vm.topK).formatted()
                )

                sliderRow(
                    label: "Repetition Penalty",
                    info: "Penalises repeating the same tokens. Values above 1.0 discourage repetition; 1.0 disables the penalty.",
                    value: $vm.repetitionPenalty,
                    range: 1.0...1.5,
                    step: 0.01,
                    display: String(format: "%.2f", vm.repetitionPenalty)
                )

                sliderRow(
                    label: "Max New Tokens",
                    info: "Upper bound on tokens generated per chunk. Longer values allow longer output but increase max generation time.",
                    value: $vm.maxNewTokens,
                    range: 50...500,
                    step: 10,
                    display: Int(vm.maxNewTokens).formatted()
                )
            }

            // MARK: Chunking

            Section("Chunking") {
                // Strategy row: label + info on the left, segmented picker on the right.
                // Using LabeledContent avoids macOS Form's two-column layout pushing
                // a plain Spacer-separated HStack to extreme edges.
                LabeledContent {
                    Picker("", selection: $vm.chunkingStrategyTag) {
                        Text("None").tag("none")
                        Text("Sentence").tag("sentence")
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 160)
                } label: {
                    HStack(spacing: 4) {
                        Text("Strategy")
                        InfoButton("Controls how long text is split before generation. Sentence-based chunking produces more natural prosody at chunk boundaries.")
                    }
                }

                sliderRow(
                    label: "Target Chunk Size",
                    info: "Maximum tokens per chunk when using sentence chunking. Chunks break at the nearest sentence boundary at or before this token count. Default (50).",
                    value: $vm.targetChunkSize,
                    range: 10...100,
                    step: 1,
                    display: "\(Int(vm.targetChunkSize)) tok",
                    displayWidth: 60
                )
                .disabled(vm.chunkingStrategyTag == "none")

                sliderRow(
                    label: "Min Chunk Size",
                    info: "Minimum tokens per chunk. Short trailing segments are merged into the previous chunk to avoid tiny segments with poor prosody.",
                    value: $vm.minChunkSize,
                    range: 1...30,
                    step: 1,
                    display: "\(Int(vm.minChunkSize)) tok",
                    displayWidth: 60
                )
                .disabled(vm.chunkingStrategyTag == "none")
            }

            // MARK: Concurrency

            Section("Concurrency") {
                sliderRow(
                    label: "Concurrent Workers",
                    info: "How many chunks to generate in parallel. 0 = all chunks at once (fastest). 1 = sequential (best for streaming playback).",
                    value: $vm.concurrentWorkerCount,
                    range: 0...16,
                    step: 1,
                    display: vm.concurrentWorkerCount == 0 ? "Max" : Int(vm.concurrentWorkerCount).formatted()
                )
            }

            // MARK: Reset

            Section {
                Button(role: .destructive) {
                    viewModel.resetGenerationSettings()
                } label: {
                    HStack {
                        Spacer()
                        Text("Reset to Defaults")
                        Spacer()
                    }
                }
            }
        }
        // Use grouped style on both platforms so macOS renders as a scrollable list
        // rather than its default two-column layout, which misaligns our custom rows.
        .formStyle(.grouped)
        #if os(iOS)
            .navigationTitle("Generation Options")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        dismiss()
                    } label: {
                        Label("Done", systemImage: "xmark.circle.fill")
                            .foregroundStyle(.primary)
                    }
                }
            }
        #endif
    }

    // MARK: - Helpers

    /// Reusable label + info + value display + slider row.
    private func sliderRow(
        label: String,
        info: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        step: Double,
        display: String,
        displayWidth: CGFloat = 44
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                InfoButton(info)
                Spacer()
                Text(display)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .frame(width: displayWidth, alignment: .trailing)
            }
            Slider(value: value, in: range, step: step)
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Info Button

/// Small info button that shows a popover with explanatory text.
/// Matches the InfoButton pattern from WhisperAX.
struct InfoButton: View {
    let text: String
    @State private var isShowing = false

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        Button {
            isShowing = true
        } label: {
            Image(systemName: "info.circle")
                .foregroundStyle(.blue)
        }
        .buttonStyle(.borderless)
        .popover(isPresented: $isShowing) {
            Text(text)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
                .padding()
                .frame(width: 260)
        }
    }
}
