//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgumentParser
import CoreML
import Foundation
import TTSKit
import WhisperKit

// MARK: - CLI-only conformances for ArgumentParser

extension Qwen3Speaker: ExpressibleByArgument {}
extension Qwen3Language: ExpressibleByArgument {}
extension TTSModelVariant: ExpressibleByArgument {}

// MARK: - CLI Command

struct TTSCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tts",
        abstract: "Generate speech from text using Qwen3-TTS"
    )

    // MARK: - Required (one of text or text-file)

    @Option(name: .long, help: "Text to synthesize")
    var text: String?

    @Option(name: .long, help: "Read text from a file (supports .txt and .md)")
    var textFile: String?

    // MARK: - Common options

    @Option(name: .long, help: "Speaker voice (aiden, ryan, ono-anna, sohee, eric, dylan, serena, vivian, uncle-fu)")
    var speaker: Qwen3Speaker = .aiden

    @Option(name: .long, help: "Language (english, chinese, japanese, korean)")
    var language: Qwen3Language = .english

    @Option(name: .long, help: "Output audio file path")
    var outputPath: String = "output"

    @Option(name: .long, help: "Output audio format (m4a or wav)")
    var outputFormat: String = "m4a"

    @Flag(name: .long, help: "Play audio through speakers in real time")
    var play: Bool = false

    @Flag(name: .long, help: "Enable verbose output")
    var verbose: Bool = false

    // MARK: - Generation options

    @Option(name: .long, help: "Sampling temperature (0.0 for greedy)")
    var temperature: Float = 0.9

    @Option(name: .long, help: "Top-k sampling (0 to disable)")
    var topK: Int = 50

    @Option(name: .long, help: "Max RVQ frames to generate")
    var maxNewTokens: Int = 245

    @Option(name: .long, help: "Concurrent chunk workers (0=max, 1=sequential, N=batch size). Defaults to 1 with --play, 0 otherwise.")
    var concurrentWorkerCount: Int?

    @Option(name: .long, help: "Target chunk size in characters for sentence splitting")
    var targetChunkSize: Int = TextChunker.defaultTargetChunkSize

    @Option(name: .long, help: "Minimum chunk size in characters (short tails merge into previous chunk)")
    var minChunkSize: Int = TextChunker.defaultMinChunkSize

    @Option(name: .long, help: "Style instruction (e.g., \"Speak slowly and softly\"). Only supported by the 1.7B model.")
    var instruction: String?

    @Option(name: .long, help: "Random seed for reproducible output")
    var seed: UInt64?

    // MARK: - Model selection

    @Option(name: .long, help: "Model preset (0.6b, 1.7b). Auto-configures version dir and variant defaults.")
    var model: TTSModelVariant = .qwen3TTS_0_6b

    // MARK: - Advanced options (auto-configured by preset, can be overridden)

    @Option(name: .long, help: "Local model directory (skips download if provided)")
    var modelsPath: String?

    @Option(name: .long, help: "HuggingFace repo for model download")
    var modelRepo: String = Qwen3TTSConstants.defaultModelRepo

    @Option(name: .long, help: "Model version directory (overrides --model preset)")
    var versionDir: String?

    @Option(name: .long, help: "HuggingFace tokenizer repo or local path")
    var tokenizer: String?

    @Option(name: .long, help: "HuggingFace API token (for private repos, or set HF_TOKEN env var)")
    var token: String?

    @Option(name: .long, help: "HuggingFace Hub compatible endpoint URL")
    var endpoint: String?

    @Option(name: .long, help: "CodeDecoder variant (overrides --model preset)")
    var codeDecoderVariant: String?

    @Option(name: .long, help: "MultiCodeDecoder variant (overrides --model preset)")
    var multiCodeDecoderVariant: String?

    @Option(name: .long, help: "SpeechDecoder variant (overrides --model preset)")
    var speechDecoderVariant: String?

    // MARK: - Compute unit options

    @Option(name: .long, help: "Compute units for embedders (TextProjector, CodeEmbedder, MultiCodeEmbedder) {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine}")
    var embedderComputeUnits: ComputeUnits = .cpuOnly

    @Option(name: .long, help: "Compute units for CodeDecoder {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine}")
    var codeDecoderComputeUnits: ComputeUnits = .cpuAndNeuralEngine

    @Option(name: .long, help: "Compute units for MultiCodeDecoder {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine}")
    var multiCodeDecoderComputeUnits: ComputeUnits = .cpuAndNeuralEngine

    @Option(name: .long, help: "Compute units for SpeechDecoder {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine}")
    var speechDecoderComputeUnits: ComputeUnits = .cpuAndNeuralEngine

    func run() async throws {
        if verbose {
            await Logging.updateCallback {
                print("[TTSKit] \($0)")
            }
        }
        // Resolve text from --text or --text-file
        let inputText: String
        if let textFile {
            let resolvedPath = FileManager.resolveAbsolutePath(textFile)
            inputText = try String(contentsOfFile: resolvedPath, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } else if let text {
            inputText = text
        } else {
            throw ValidationError("Either --text or --text-file must be provided")
        }

        guard !inputText.isEmpty else {
            throw ValidationError("Input text is empty")
        }

        // Resolve local models path if provided
        let resolvedModelFolder: URL? = modelsPath.map {
            URL(fileURLWithPath: FileManager.resolveAbsolutePath($0))
        }
        // Resolve tokenizer: pass as URL — loadTokenizer falls back to HF Hub if the path doesn't exist
        let resolvedTokenizerFolder: URL? = tokenizer.map {
            URL(fileURLWithPath: $0)
        }

        // Build config from preset — explicit CLI overrides replace preset defaults.
        let config = TTSKitConfig(
            model: model,
            modelFolder: resolvedModelFolder,
            modelRepo: modelRepo,
            tokenizerFolder: resolvedTokenizerFolder,
            modelToken: token,
            modelEndpoint: endpoint ?? Qwen3TTSConstants.defaultEndpoint,
            versionDir: versionDir,
            codeDecoderVariant: codeDecoderVariant,
            multiCodeDecoderVariant: multiCodeDecoderVariant,
            speechDecoderVariant: speechDecoderVariant,
            computeOptions: ComputeOptions(
                embedderComputeUnits: embedderComputeUnits.asMLComputeUnits,
                codeDecoderComputeUnits: codeDecoderComputeUnits.asMLComputeUnits,
                multiCodeDecoderComputeUnits: multiCodeDecoderComputeUnits.asMLComputeUnits,
                speechDecoderComputeUnits: speechDecoderComputeUnits.asMLComputeUnits
            ),
            verbose: verbose
        )

        // Default: --play uses sequential (1), file output uses unlimited (0).
        let effectiveWorkerCount = concurrentWorkerCount ?? (play ? 1 : 0)

        // Always use a seed for reproducibility -- generate one if not provided
        let effectiveSeed = seed ?? UInt64.random(in: 0...UInt64(UInt32.max))

        // Warn if instruction is used with a model that doesn't support it
        var effectiveInstruction = instruction
        if let instruction = effectiveInstruction, !instruction.isEmpty, model == .qwen3TTS_0_6b {
            print("Warning: --instruction is only supported by the 1.7B model variant. Ignoring instruction for \(model.rawValue).")
            effectiveInstruction = nil
        }

        if verbose {
            print("Qwen3-TTS Pipeline")
            if textFile != nil {
                print("  Text file: \(textFile!)")
            }
            print("  Text: \"\(inputText.prefix(80))\(inputText.count > 80 ? "..." : "")\"")
            print("  Speaker: \(speaker.rawValue)")
            print("  Language: \(language.rawValue)")
            print("  Model: \(model.rawValue)")
            if let inst = effectiveInstruction {
                print("  Instruction: \"\(inst)\"")
            }
            if let folder = resolvedModelFolder {
                print("  Models: \(folder.path)")
            } else {
                print("  Models: \(config.modelRepo) (auto-download)")
            }
            print("  Version: \(config.versionDir)")
            print("  CodeDecoder: \(config.codeDecoderVariant)")
            print("  MultiCodeDecoder: \(config.multiCodeDecoderVariant)")
            print("  SpeechDecoder: \(config.speechDecoderVariant)")
            print("  Embedder compute: \(embedderComputeUnits.rawValue)")
            print("  CodeDecoder compute: \(codeDecoderComputeUnits.rawValue)")
            print("  MultiCodeDecoder compute: \(multiCodeDecoderComputeUnits.rawValue)")
            print("  SpeechDecoder compute: \(speechDecoderComputeUnits.rawValue)")
            print("  Output: \(outputPath).\(outputFormat.lowercased())")
            print("  Temperature: \(temperature)")
            print("  Top-k: \(topK)")
            print("  Play: \(play)")
            let workerDesc = effectiveWorkerCount == 0 ? "max" : "\(effectiveWorkerCount)"
            print("  Concurrency: \(workerDesc) (chunking: sentence)")
            print("  Seed: \(effectiveSeed)")
        }

        // Initialize pipeline (downloads if needed, loads tokenizer + 6 models concurrently)
        config.seed = effectiveSeed
        let tts = try await TTSKit(config)

        let options = GenerationOptions(
            temperature: temperature,
            topK: topK,
            repetitionPenalty: 1.05,
            maxNewTokens: maxNewTokens,
            concurrentWorkerCount: effectiveWorkerCount,
            targetChunkSize: targetChunkSize,
            minChunkSize: minChunkSize,
            instruction: effectiveInstruction
        )

        let result: SpeechResult
        if play {
            result = try await tts.play(
                text: inputText,
                speaker: speaker,
                language: language,
                options: options
            )
        } else {
            result = try await tts.generate(
                text: inputText,
                speaker: speaker,
                language: language,
                options: options
            )
        }

        let format = AudioOutput.AudioFileFormat(rawValue: outputFormat.lowercased()) ?? .m4a
        let outputURL = URL(fileURLWithPath: outputPath)
        let outputFolder = outputURL.deletingLastPathComponent().path == "."
            ? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            : outputURL.deletingLastPathComponent()

        let savedURL = try await AudioOutput.saveAudio(
            result.audio,
            toFolder: outputFolder,
            filename: outputURL.lastPathComponent,
            sampleRate: result.sampleRate,
            format: format
        )

        result.logTimings()

        if verbose {
            print(String(format: "Generated %.2fs of audio -> %@", result.audioDuration, savedURL.path))
        } else {
            print(savedURL.path)
        }
    }
}
