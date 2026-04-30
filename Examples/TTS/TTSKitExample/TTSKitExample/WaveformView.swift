//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import SwiftUI

// MARK: - Scrolling Waveform (main detail view)

/// Waveform with a fixed center playhead.
///
/// All bars start at the center (playhead) and extend rightward.
/// As playbackTime advances, bars scroll left past the playhead.
/// During streaming generation (playbackTime=0), new bars simply
/// appear and grow to the right of the center line.
struct WaveformView: View {
    var samples: [Float]
    var playbackTime: TimeInterval
    var totalDuration: TimeInterval
    /// Total real audio scheduled to the player. When greater than `playbackTime`,
    /// bars between the playhead and this boundary are shaded at medium opacity to
    /// indicate audio that is buffered and ready but not yet played.
    /// Pass 0 (default) to use the classic two-state (played / unplayed) rendering.
    var accentColor: Color = .primary

    /// Each bar = one audio token (~80ms). Fixed pixel width per bar.
    private let barWidth: CGFloat = 3
    private let barSpacing: CGFloat = 1.5
    /// Seconds of audio each bar represents
    nonisolated static let secondsPerBar: TimeInterval = 0.08

    private var barStep: CGFloat { barWidth + barSpacing }

    /// Monotonically-increasing display time. Only moves forward, never back.
    /// Resets to 0 when the waveform is cleared (new generation) or playback
    /// explicitly resets to 0. Clamped to the last generated bar so the
    /// playhead never runs ahead of audio that hasn't been generated yet.
    @State private var displayedTime: TimeInterval = 0

    var body: some View {
        Canvas { context, size in
            let barCount = samples.count
            guard barCount > 0 else { return }

            let centerX = size.width / 2
            let midY = size.height / 2
            let maxBarHeight = size.height * 0.85

            // How many pixels displayedTime shifts the waveform left
            let playbackOffsetPx = CGFloat(displayedTime / Self.secondsPerBar) * barStep

            for i in 0..<barCount {
                // Bar i sits at: centerX + i * barStep, shifted left by playback
                let x = centerX + CGFloat(i) * barStep - playbackOffsetPx

                // Skip bars outside the visible area
                guard x + barWidth > 0 && x < size.width else { continue }

                let amplitude = CGFloat(min(abs(samples[i]), 1.0))
                let barHeight = max(1, amplitude * maxBarHeight)
                let rect = CGRect(
                    x: x,
                    y: midY - barHeight / 2,
                    width: barWidth,
                    height: barHeight
                )

                let barTime = Double(i) * Self.secondsPerBar
                let isPlayed = barTime < displayedTime
                let color = isPlayed ? accentColor : accentColor.opacity(0.3)
                context.fill(Path(roundedRect: rect, cornerRadius: 1), with: .color(color))
            }

            // Fixed playhead line at center
            let dotRadius: CGFloat = 4
            let line = Path { p in
                p.move(to: CGPoint(x: centerX, y: dotRadius))
                p.addLine(to: CGPoint(x: centerX, y: size.height - dotRadius))
            }
            context.stroke(line, with: .color(Color.accentColor), lineWidth: 1.5)

            // Playhead dots at top and bottom of the line
            let topDot = Path(ellipseIn: CGRect(x: centerX - dotRadius, y: 0, width: dotRadius * 2, height: dotRadius * 2))
            let bottomDot = Path(ellipseIn: CGRect(x: centerX - dotRadius, y: size.height - dotRadius * 2, width: dotRadius * 2, height: dotRadius * 2))
            context.fill(topDot, with: .color(Color.accentColor))
            context.fill(bottomDot, with: .color(Color.accentColor))
        }
        .onChange(of: playbackTime) { _, newTime in
            if newTime == 0 {
                // Explicit reset - new session starting
                displayedTime = 0
            } else {
                // Clamp to the last generated bar: playhead can't outrun visible audio
                let maxTime = Double(samples.count) * Self.secondsPerBar
                displayedTime = max(displayedTime, min(newTime, maxTime))
            }
        }
        .onChange(of: samples.count) { _, count in
            // Waveform cleared for a new generation
            if count == 0 { displayedTime = 0 }
        }
        .accessibilityLabel("Audio waveform")
        .accessibilityValue(
            playbackTime > 0
                ? "Playback at \(Int(playbackTime)) seconds of \(Int(totalDuration))"
                : samples.isEmpty ? "No audio" : "\(Int(totalDuration)) seconds"
        )
    }
}

// MARK: - Thumbnail (sidebar rows)

struct WaveformThumbnail: View {
    var samples: [Float]
    var color: Color = .secondary

    var body: some View {
        Canvas { context, size in
            let count = samples.count
            guard count > 0 else { return }

            let barWidth = size.width / CGFloat(count)
            let midY = size.height / 2

            for (i, sample) in samples.enumerated() {
                let amp = CGFloat(min(abs(sample), 1.0))
                let h = max(0.5, amp * size.height * 0.85)
                let x = CGFloat(i) * barWidth
                let rect = CGRect(x: x, y: midY - h / 2, width: max(0.5, barWidth - 0.3), height: h)
                context.fill(Path(roundedRect: rect, cornerRadius: 0.3), with: .color(color))
            }
        }
        .frame(width: 48, height: 24)
        .accessibilityHidden(true)
    }
}

// MARK: - Previews

#Preview("Pre-recorded - start") {
    let samples: [Float] = (0..<200).map { _ in Float.random(in: 0...1) }
    WaveformView(samples: samples, playbackTime: 0, totalDuration: 10.0)
        .frame(width: 600, height: 120).padding()
}

#Preview("Pre-recorded - mid playback") {
    let samples: [Float] = (0..<200).map { _ in Float.random(in: 0...1) }
    WaveformView(samples: samples, playbackTime: 5.0, totalDuration: 10.0)
        .frame(width: 600, height: 120).padding()
}

#Preview("Pre-recorded - near end") {
    let samples: [Float] = (0..<200).map { _ in Float.random(in: 0...1) }
    WaveformView(samples: samples, playbackTime: 9.0, totalDuration: 10.0)
        .frame(width: 600, height: 120).padding()
}
