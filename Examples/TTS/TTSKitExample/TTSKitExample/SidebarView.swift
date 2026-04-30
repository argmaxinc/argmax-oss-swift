//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import SwiftUI
import WhisperKit

struct SidebarView: View {
    @Environment(ViewModel.self) private var viewModel

    var body: some View {
        @Bindable var vm = viewModel

        VStack(spacing: 0) {
            // Model management at the top of the sidebar
            ModelManagementView()

            Divider()

            // Compute units configuration
            ComputeUnitsView()
                .disabled(vm.modelState.isBusy)

            Divider()

            // Generation history list
            List(selection: $vm.selectedGenerationID) {
                if !vm.favoriteGenerations.isEmpty {
                    Section("Favorites") {
                        ForEach(vm.favoriteGenerations) { gen in
                            GenerationRow(generation: gen)
                                .tag(gen.id)
                                .contextMenu { rowContextMenu(for: gen) }
                            #if os(iOS)
                                .swipeActions(edge: .trailing) { rowSwipeActions(for: gen) }
                            #endif
                        }
                    }
                }

                Section("Recents") {
                    ForEach(vm.generations) { gen in
                        GenerationRow(generation: gen)
                            .tag(gen.id)
                            .contextMenu { rowContextMenu(for: gen) }
                        #if os(iOS)
                            .swipeActions(edge: .trailing) { rowSwipeActions(for: gen) }
                        #endif
                    }
                }
            }
            .overlay {
                if vm.isLoadingHistory {
                    ProgressView("Loading history…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if vm.generations.isEmpty {
                    ContentUnavailableView(
                        "No Generations Yet",
                        systemImage: "waveform",
                        description: Text("Generated speech will appear here.")
                    )
                }
            }

            Divider()

            // App & device info footer
            appInfoFooter
        }
        .navigationTitle("TTSKit Example")
        #if os(macOS)
            .navigationSplitViewColumnWidth(min: 260, ideal: 280, max: 400)
        #elseif os(iOS)
            .toolbar {
                ToolbarItemGroup(placement: .bottomBar) {
                    Spacer()
                    Button(action: newGeneration) {
                        Image(systemName: "plus")
                            .font(.title2)
                            .bold()
                    }
                }
            }
        #endif
    }

    // MARK: - New Generation

    private func newGeneration() {
        viewModel.clearInput()
        viewModel.currentWaveform = []
        // Use a sentinel UUID to show the detail in "new generation" mode.
        // This drives navigation uniformly on all platforms via List(selection:).
        viewModel.selectedGenerationID = ViewModel.newGenerationSentinel
    }

    // MARK: - App Info Footer

    private var appInfoFooter: some View {
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
        .font(.system(.caption2, design: .monospaced))
        .foregroundStyle(.tertiary)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    // MARK: - Context Menu / Swipe Actions

    @ViewBuilder
    private func rowContextMenu(for generation: Generation) -> some View {
        Button {
            viewModel.playGeneration(generation)
        } label: {
            Label("Play", systemImage: "play.fill")
        }

        Button {
            viewModel.toggleFavorite(generation.id)
        } label: {
            Label(
                generation.isFavorite ? "Unfavorite" : "Favorite",
                systemImage: generation.isFavorite ? "star.slash" : "star"
            )
        }

        if let url = viewModel.audioFileURL(for: generation) {
            AudioCopyButton(url: url)

            #if os(macOS)
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([url])
            } label: {
                Label("Show in Finder", systemImage: "folder")
            }
            #endif
        }

        Divider()

        Button(role: .destructive) {
            viewModel.deleteGeneration(generation.id)
        } label: {
            Label("Delete", systemImage: "trash")
        }
    }

    #if os(iOS)
    @ViewBuilder
    private func rowSwipeActions(for generation: Generation) -> some View {
        Button(role: .destructive) {
            viewModel.deleteGeneration(generation.id)
        } label: {
            Label("Delete", systemImage: "trash")
        }
    }
    #endif
}

// MARK: - Row

struct GenerationRow: View {
    @Environment(ViewModel.self) private var viewModel
    let generation: Generation
    @State private var isHovered = false

    var body: some View {
        HStack(spacing: 10) {
            if let samples = generation.waveformSamples, !samples.isEmpty {
                WaveformThumbnail(samples: samples)
            } else {
                Image(systemName: "waveform")
                    .frame(width: 48, height: 24)
                    .foregroundStyle(.tertiary)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(generation.title)
                    .font(.body)
                    .lineLimit(1)
                    .truncationMode(.tail)

                HStack(spacing: 6) {
                    Text(generation.date, style: .date)
                    Text("·")
                    Text(String(format: "%.1fs", generation.audioDuration))
                    Text("·")
                    Text(generation.speaker.capitalized)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            Spacer()

            if generation.isFavorite {
                Image(systemName: "star.fill")
                    .font(.caption)
                    .foregroundStyle(.yellow)
            }

            #if os(macOS)
            Button {
                viewModel.deleteGeneration(generation.id)
            } label: {
                Image(systemName: "trash")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .opacity(isHovered ? 1 : 0)
            .animation(.easeInOut(duration: 0.15), value: isHovered)
            #endif
        }
        .padding(.vertical, 2)
        #if os(macOS)
            .onHover { isHovered = $0 }
        #endif
    }
}
