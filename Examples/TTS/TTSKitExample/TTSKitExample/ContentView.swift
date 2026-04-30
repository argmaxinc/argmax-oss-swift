//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import SwiftUI

struct ContentView: View {
    @Environment(ViewModel.self) private var viewModel
    var body: some View {
        NavigationSplitView {
            SidebarView()
        } detail: {
            DetailView()
        }
        .navigationSplitViewStyle(.balanced)
        #if os(macOS)
            // SwiftUI frame constraints don't prevent NavigationSplitView from entering
            // overlay mode. Setting minSize directly on NSWindow is a reliable fix.
            .background(WindowMinSizeEnforcer(minSize: CGSize(width: 840, height: 560)))
        #endif
    }
}

// MARK: - macOS window minimum size enforcer

#if os(macOS)
/// Enforces `NSWindow.minSize` at the AppKit level.
///
/// SwiftUI's `.frame(minWidth:)` and `windowResizability` both fail to prevent
/// `NavigationSplitView` from shrinking into overlay mode. Going through AppKit
/// directly is a reliable way to clamp the resize handle.
///
/// A custom `NSView` subclass is used instead of `DispatchQueue.main.async` because
/// the view's `window` property is `nil` during `makeNSView`.
private struct WindowMinSizeEnforcer: NSViewRepresentable {
    let minSize: CGSize

    func makeNSView(context: Context) -> MinSizeView {
        let view = MinSizeView()
        view.minSize = minSize
        return view
    }

    func updateNSView(_ nsView: MinSizeView, context: Context) {
        nsView.minSize = minSize
    }
}

private final class MinSizeView: NSView {
    var minSize: CGSize = .zero {
        didSet { window?.minSize = minSize }
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.minSize = minSize
    }
}
#endif

#Preview {
    ContentView()
        .environment(ViewModel())
}
