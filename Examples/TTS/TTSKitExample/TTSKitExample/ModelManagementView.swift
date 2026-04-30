//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import SwiftUI
import TTSKit

/// Sidebar section for model selection, download, load/unload, and deletion.
/// Mirrors the model management pattern from WhisperAX / Argmax Playground.
struct ModelManagementView: View {
    @Environment(ViewModel.self) private var viewModel

    var body: some View {
        @Bindable var vm = viewModel

        VStack(alignment: .leading, spacing: 8) {
            // Status row: dot + state label + accessory buttons
            HStack(spacing: 6) {
                Image(systemName: "circle.fill")
                    .font(.system(size: 8))
                    .foregroundStyle(vm.modelState.color)
                    .symbolEffect(.variableColor, isActive: vm.modelState.isBusy)

                Text(vm.modelState.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                #if os(macOS)
                // On macOS the picker fits in the status row alongside the accessory buttons
                modelPicker
                #endif

                accessoryButtons
            }

            #if os(iOS)
            // On iOS the picker gets its own row so it isn't squeezed by the accessory buttons
            modelPicker
                .frame(maxWidth: .infinity, alignment: .leading)
            #endif

            // Size estimate
            Text(viewModel.modelDiskSize(for: vm.selectedPreset) ?? vm.selectedPreset.sizeEstimate)
                .font(.caption2)
                .foregroundStyle(.tertiary)

            // Action slot: button OR progress - same height, no layout shift
            if vm.modelState == .loaded && vm.loadedPreset == vm.selectedPreset {
                Button { viewModel.unloadModel() } label: {
                    Label("Unload Model", systemImage: "eject")
                        .frame(maxWidth: .infinity)
                        .frame(height: 28)
                }
                .glassButton()
                .controlSize(.small)
                .accessibilityLabel("Unload model")
                .accessibilityHint("Releases the model from memory. You can reload it later.")
            } else if vm.modelState == .downloading {
                VStack(spacing: 2) {
                    HStack {
                        ProgressView(value: vm.downloadProgress)
                            .progressViewStyle(.linear)
                        Text(String(format: "%.1f%%", vm.downloadProgress * 100))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                }
                .frame(height: 28)
            } else if vm.modelState == .prewarming || vm.modelState == .loading {
                VStack(alignment: .leading, spacing: 2) {
                    ProgressView()
                        .progressViewStyle(.linear)
                    Text(vm.modelState == .prewarming
                        ? "Specializing for your device..."
                        : "Loading...")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .frame(height: 28)
            } else {
                Button { Task { await viewModel.loadModel() } } label: {
                    Label(
                        vm.isModelDownloaded ? "Load Model" : "Download & Load",
                        systemImage: vm.isModelDownloaded ? "arrow.up.circle" : "arrow.down.circle"
                    )
                    .frame(maxWidth: .infinity)
                    .frame(height: 28)
                }
                .glassButton(prominent: true)
                .tint(.accentColor)
                .controlSize(.small)
                .accessibilityLabel(vm.isModelDownloaded ? "Load model" : "Download and load model")
                .accessibilityHint(vm.isModelDownloaded
                    ? "Loads the downloaded model into memory"
                    : "Downloads the model (~\(vm.selectedPreset.sizeEstimate)) and loads it")
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    // MARK: - Model Picker

    private var modelPicker: some View {
        @Bindable var vm = viewModel
        return Picker("Model", selection: $vm.selectedPreset) {
            ForEach(TTSModelVariant.allCases, id: \.self) { preset in
                HStack {
                    let downloaded = vm.localModelPaths[preset] != nil
                    Image(systemName: downloaded ? "checkmark.circle" : "arrow.down.circle.dotted")
                    if preset.isAvailableOnCurrentPlatform {
                        Text(preset.displayName)
                    } else {
                        Text("\(preset.displayName) - Mac only")
                            .foregroundStyle(.tertiary)
                    }
                }
                .tag(preset)
            }
        }
        .labelsHidden()
        .pickerStyle(.menu)
        .accessibilityLabel("Select model")
        .accessibilityHint("Choose the TTS model to use for generation")
        .onChange(of: vm.selectedPreset) {
            if !vm.selectedPreset.isAvailableOnCurrentPlatform {
                vm.selectedPreset = .defaultForCurrentPlatform
                vm.statusMessage = "\(vm.selectedPreset.displayName) requires macOS"
                return
            }
            if vm.loadedPreset == vm.selectedPreset { return }
            if vm.localModelPaths[vm.selectedPreset] != nil {
                vm.statusMessage = vm.modelState == .loaded
                    ? "Different model loaded"
                    : "Downloaded"
            } else {
                vm.statusMessage = "Not downloaded"
            }
        }
    }

    // MARK: - Accessory Buttons (trash, reveal, HF link)

    @ViewBuilder
    private var accessoryButtons: some View {
        let vm = viewModel

        // Delete
        Button {
            viewModel.deleteModel()
        } label: {
            Image(systemName: "trash")
        }
        .buttonStyle(.borderless)
        .disabled(!vm.isModelDownloaded || vm.modelState.isBusy)
        .help("Delete downloaded model files")
        .accessibilityLabel("Delete model")
        .accessibilityHint("Permanently deletes the downloaded model files from disk")

        #if os(macOS)
        // Reveal in Finder
        if let path = vm.localModelPaths[vm.selectedPreset] {
            Button {
                NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: path)
            } label: {
                Image(systemName: "folder")
            }
            .buttonStyle(.borderless)
            .help("Reveal in Finder")
        }
        #endif

        // Open on HuggingFace
        Button {
            if let url = viewModel.modelRepoURL {
                #if os(macOS)
                NSWorkspace.shared.open(url)
                #else
                UIApplication.shared.open(url)
                #endif
            }
        } label: {
            Image(systemName: "link.circle")
        }
        .buttonStyle(.borderless)
        .help("Open model on HuggingFace")
    }
}
