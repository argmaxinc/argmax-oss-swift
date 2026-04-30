//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import SwiftUI
import TTSKit

/// Collapsible sidebar section for configuring per-component ML compute units.
/// Matches the pattern from WhisperAX's computeUnitsView.
struct ComputeUnitsView: View {
    @Environment(ViewModel.self) private var viewModel
    @State private var isExpanded = false

    var body: some View {
        @Bindable var vm = viewModel

        DisclosureGroup(isExpanded: $isExpanded) {
            VStack(spacing: 8) {
                computeRow(
                    label: "Embedders",
                    units: vm.embedderComputeUnits,
                    onChange: { vm.embedderComputeUnits = $0; reloadIfNeeded() }
                )
                computeRow(
                    label: "Code Decoder",
                    units: vm.codeDecoderComputeUnits,
                    onChange: { vm.codeDecoderComputeUnits = $0; reloadIfNeeded() }
                )
                computeRow(
                    label: "Multi-Code Decoder",
                    units: vm.multiCodeDecoderComputeUnits,
                    onChange: { vm.multiCodeDecoderComputeUnits = $0; reloadIfNeeded() }
                )
                computeRow(
                    label: "Speech Decoder",
                    units: vm.speechDecoderComputeUnits,
                    onChange: { vm.speechDecoderComputeUnits = $0; reloadIfNeeded() }
                )
            }
            .padding(.top, 4)
        } label: {
            Button {
                isExpanded.toggle()
            } label: {
                Text("Compute Units")
                    .font(.headline)
            }
            .buttonStyle(.plain)
        }
        .disabled(viewModel.modelState.isBusy)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    @ViewBuilder
    private func computeRow(
        label: String,
        units: MLComputeUnits,
        onChange: @escaping (MLComputeUnits) -> Void
    ) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.85)
                .layoutPriority(1)

            Spacer(minLength: 4)

            Picker("", selection: Binding(
                get: { units },
                set: { onChange($0) }
            )) {
                Text("CPU").tag(MLComputeUnits.cpuOnly)
                Text("GPU").tag(MLComputeUnits.cpuAndGPU)
                #if os(iOS)
                // Abbreviated on iOS to prevent overflow
                Text("NE").tag(MLComputeUnits.cpuAndNeuralEngine)
                #else
                Text("Neural Engine").tag(MLComputeUnits.cpuAndNeuralEngine)
                #endif
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .fixedSize()
        }
    }

    private func reloadIfNeeded() {
        guard viewModel.modelState == .loaded else { return }
        viewModel.reloadModelForComputeUnitChange()
    }
}
