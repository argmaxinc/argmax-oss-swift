//  Copyright © 2025 Argmax, Inc. All rights reserved.
//  For licensing see accompanying LICENSE.md file.

import ArgumentParser

struct ServeCLIArguments: ParsableArguments {
    @OptionGroup
    var transcribe: TranscribeCLIArguments
    
    @Option(name: .long, help: "Port to run the server on")
    var port: Int = 50060
    
    @Option(name: .long, help: "Host to bind the server to")
    var host: String = "localhost"
}
