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

    @Option(name: .long, help: "Maximum request body size in MiB, larger requests are rejected with 413")
    var maxBodySize: Int = 1024

    @Option(name: .long, help: "Seconds to wait for more request body data before rejecting the request with 408, 0 disables the timeout")
    var bodyReadTimeout: Int = 30

    mutating func validate() throws {
        guard maxBodySize > 0 else {
            throw ValidationError("--max-body-size must be greater than 0.")
        }
        guard bodyReadTimeout >= 0 else {
            throw ValidationError("--body-read-timeout must be 0 or greater.")
        }
    }
}
