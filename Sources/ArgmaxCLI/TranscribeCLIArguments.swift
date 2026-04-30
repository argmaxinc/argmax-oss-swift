//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgumentParser

struct TranscribeCLIArguments: ParsableArguments {
    @Option(help: "Paths to audio files")
    var audioPath = [String]()

    @Option(help: "Path to a folder containing audio files")
    var audioFolder: String?

    @Option(help: "Path of model files")
    var modelPath: String?

    @Option(help: "Model to download if no modelPath is provided")
    var model: String?

    @Option(help: "Text to add in front of the model name to specify between different types of the same variant (values: \"openai\", \"distil\")")
    var modelPrefix: String = "openai"

    @Option(help: "Path to save the downloaded model")
    var downloadModelPath: String?

    @Option(help: "Path to save the downloaded tokenizer files")
    var downloadTokenizerPath: String?

    @Option(name: .long, help: "HuggingFace Hub compatible endpoint URL")
    var endpoint: String?

    @Option(help: "Compute units for audio encoder model with {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine,random}")
    var audioEncoderComputeUnits: ComputeUnits = .cpuAndNeuralEngine

    @Option(help: "Compute units for text decoder model with {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine,random}")
    var textDecoderComputeUnits: ComputeUnits = .cpuAndNeuralEngine

    @Flag(help: "Verbose mode")
    var verbose: Bool = false

    @Option(help: "Task to perform (transcribe or translate)")
    var task: String = "transcribe"

    @Option(help: "Language spoken in the audio")
    var language: String?

    @Option(help: "Temperature to use for sampling")
    var temperature: Float = 0

    @Option(help: "Temperature to increase on fallbacks during decoding")
    var temperatureIncrementOnFallback: Float = 0.2

    @Option(help: "Number of times to increase temperature when falling back during decoding")
    var temperatureFallbackCount: Int = 5

    @Option(help: "Number of candidates when sampling with non-zero temperature")
    var bestOf: Int = 5

    @Flag(help: "Force initial prompt tokens based on language, task, and timestamp options")
    var usePrefillPrompt: Bool = false

    @Flag(help: "Use decoder prefill data for faster initial decoding")
    var usePrefillCache: Bool = false

    @Flag(help: "Skip special tokens in the output")
    var skipSpecialTokens: Bool = false

    @Flag(help: "Force no timestamps when decoding")
    var withoutTimestamps: Bool = false

    @Flag(help: "Add timestamps for each word in the output")
    var wordTimestamps: Bool = false

    @Option(help: "Force prefix text when decoding")
    var prefix: String?

    @Option(help: "Condition on this text when decoding")
    var prompt: String?

    @Option(parsing: .upToNextOption, help: "List of timestamps (in seconds) of start and end values to transcribe as seperate clips in single audio file (example: --clip-timestamps 0 10.2 34.5 60.0)")
    var clipTimestamps: [Float] = []

    @Option(parsing: .upToNextOption, help: "List of tokens to suppress in the output (example: --suppress-tokens 1 2 3)")
    var suppressTokens: [Int] = []

    @Option(help: "Gzip compression ratio threshold for decoding failure")
    var compressionRatioThreshold: Float?

    @Option(help: "Average log probability threshold for decoding failure")
    var logprobThreshold: Float?

    @Option(help: "Log probability threshold for first token decoding failure")
    var firstTokenLogProbThreshold: Float?

    @Option(help: "Probability threshold to consider a segment as silence")
    var noSpeechThreshold: Float?

    @Flag(help: "Output a report of the results")
    var report: Bool = false

    @Option(help: "Directory to save the report")
    var reportPath: String = "."

    @Flag(help: "Process audio directly from the microphone")
    var stream: Bool = false

    @Flag(help: "Simulate streaming transcription using the input audio file")
    var streamSimulated: Bool = false

    @Option(help: "Maximum concurrent inference, might be helpful when processing more than 1 audio file at the same time. 0 means unlimited. Default: 4")
    var concurrentWorkerCount: Int = 4

    @Option(help: "Chunking strategy for audio processing, `none` means no chunking, `vad` means using voice activity detection. Default: `vad`")
    var chunkingStrategy: String = "vad"
}
