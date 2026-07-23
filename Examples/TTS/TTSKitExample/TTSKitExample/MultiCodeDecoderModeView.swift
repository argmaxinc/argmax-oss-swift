//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import SwiftUI
import TTSKit

/// Sidebar picker for the MultiCodeDecoder mode: `.stepped` (one position per
/// call) or `.fused` (whole RVQ frame in one call). Changing it triggers a
/// model reload.
struct MultiCodeDecoderModeView: View {
    @Environment(ViewModel.self) private var viewModel

    var body: some View {
        @Bindable var vm = viewModel

        HStack(spacing: 8) {
            Text("Multi Code Decoder Mode")
                .font(.headline)
                .lineLimit(1)
                .minimumScaleFactor(0.85)
                .layoutPriority(1)

            Spacer(minLength: 4)

            Picker("", selection: Binding(
                get: { vm.multiCodeDecoderMode },
                set: {
                    vm.multiCodeDecoderMode = $0
                    reloadIfNeeded()
                }
            )) {
                Text("Stepped").tag(Qwen3MultiCodeDecoderMode.stepped)
                Text("Fused").tag(Qwen3MultiCodeDecoderMode.fused)
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
