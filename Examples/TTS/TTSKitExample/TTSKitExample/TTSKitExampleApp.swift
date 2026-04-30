//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import SwiftUI

@main
struct TTSKitExampleApp: App {
    @State private var viewModel = ViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(viewModel)
            #if os(iOS)
                .onAppear { installKeyboardDismissGesture() }
            #endif
                .onReceive(NotificationCenter.default.publisher(
                    for: {
                        #if os(iOS)
                        UIApplication.willResignActiveNotification
                        #else
                        NSApplication.willTerminateNotification
                        #endif
                    }()
                )) { _ in
                    viewModel.cancelAllTasks()
                }
        }
        .defaultSize(width: 1100, height: 700)
    }
}

// MARK: - iOS keyboard dismiss on tap outside

#if os(iOS)
import UIKit

/// Installs a window-level tap gesture recognizer that dismisses the keyboard whenever
/// the user taps outside a UITextView or UITextField. Uses `cancelsTouchesInView = false`
/// and a delegate check so text selection and all other controls continue to work normally.
@MainActor
private func installKeyboardDismissGesture() {
    guard let window = UIApplication.shared
        .connectedScenes
        .compactMap({ $0 as? UIWindowScene })
        .flatMap({ $0.windows })
        .first(where: { $0.isKeyWindow }) else { return }

    // Only install once
    if window.gestureRecognizers?.contains(where: { $0 is KeyboardDismissTapRecognizer }) == true {
        return
    }

    window.addGestureRecognizer(KeyboardDismissTapRecognizer())
}

private final class KeyboardDismissTapRecognizer: UITapGestureRecognizer, UIGestureRecognizerDelegate {
    init() {
        super.init(target: nil, action: nil)
        cancelsTouchesInView = false
        delegate = self
        addTarget(self, action: #selector(handleTap))
    }

    @objc private func handleTap() {
        UIApplication.shared.sendAction(
            #selector(UIResponder.resignFirstResponder),
            to: nil, from: nil, for: nil
        )
    }

    /// Only fire when the touch lands outside a text input view.
    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldReceive touch: UITouch
    ) -> Bool {
        !(touch.view is UITextView || touch.view is UITextField)
    }

    /// Always allow simultaneous recognition with every other gesture (buttons, scrolls, etc.)
    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer
    ) -> Bool {
        true
    }
}
#endif
