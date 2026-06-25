//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import SwiftUI
import TTSKit

/// Sidebar picker for the SpeechDecoder mode: `.latencyOptimized` (one frame per
/// call, lowest TTFB) or `.throughputOptimized` (four frames per call). Changing it
/// triggers a model reload.
struct SpeechDecoderModeView: View {
    @Environment(ViewModel.self) private var viewModel

    var body: some View {
        @Bindable var vm = viewModel

        HStack(spacing: 8) {
            Text("Speech Decoder Mode")
                .font(.headline)
                .lineLimit(1)
                .minimumScaleFactor(0.85)
                .layoutPriority(1)

            Spacer(minLength: 4)

            Picker("", selection: Binding(
                get: { vm.speechDecoderMode },
                set: {
                    vm.speechDecoderMode = $0
                    reloadIfNeeded()
                }
            )) {
                Text("Latency").tag(Qwen3SpeechDecoderMode.latencyOptimized)
                Text("Throughput").tag(Qwen3SpeechDecoderMode.throughputOptimized)
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .fixedSize()
        }
        .disabled(viewModel.modelState.isBusy)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private func reloadIfNeeded() {
        guard viewModel.modelState == .loaded else { return }
        viewModel.reloadModelForComputeUnitChange()
    }
}
