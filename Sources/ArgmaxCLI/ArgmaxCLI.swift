//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgumentParser
import Foundation

let VERSION: String = "development"

var subcommands: [ParsableCommand.Type] {
#if BUILD_SERVER_CLI
    [TranscribeCLI.self, TTSCLI.self, DiarizeCLI.self, ServeCLI.self]
#else
    [TranscribeCLI.self, TTSCLI.self, DiarizeCLI.self]
#endif
}

@main
struct ArgmaxCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "argmax-cli",
        abstract: "Argmax Open Core SDK CLI",
        discussion: "Swift native on-device speech recognition, text-to-speech, and speaker diarization for Apple Silicon",
        version: VERSION,
        subcommands: subcommands
    )
}
