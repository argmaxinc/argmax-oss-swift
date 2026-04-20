//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import ArgmaxCore
import CoreML

public protocol TextDecoderTensorType {}
public protocol TextDecoderInputType {}
public protocol TextDecoderOutputType {}
extension MLMultiArray: TextDecoderTensorType {}
extension MLMultiArray: TextDecoderInputType {}

public struct TextDecoderMLMultiArrayInputType: TextDecoderInputType {
    public var inputIds: MLMultiArray
    public var cacheLength: MLMultiArray
    public var keyCache: MLMultiArray
    public var valueCache: MLMultiArray
    public var kvCacheUpdateMask: MLMultiArray
    public var encoderOutputEmbeds: MLMultiArray
    public var decoderKeyPaddingMask: MLMultiArray

    public init(
        inputIds: MLMultiArray,
        cacheLength: MLMultiArray,
        keyCache: MLMultiArray,
        valueCache: MLMultiArray,
        kvCacheUpdateMask: MLMultiArray,
        encoderOutputEmbeds: MLMultiArray,
        decoderKeyPaddingMask: MLMultiArray
    ) {
        self.inputIds = inputIds
        self.cacheLength = cacheLength
        self.keyCache = keyCache
        self.valueCache = valueCache
        self.kvCacheUpdateMask = kvCacheUpdateMask
        self.encoderOutputEmbeds = encoderOutputEmbeds
        self.decoderKeyPaddingMask = decoderKeyPaddingMask
    }
}

public struct TextDecoderMLMultiArrayOutputType: TextDecoderOutputType {
    public var logits: MLMultiArray?
    public var cache: DecodingCache?

    public init(logits: MLMultiArray? = nil, cache: DecodingCache? = nil) {
        self.logits = logits
        self.cache = cache
    }
}

public protocol DecodingInputsType {
    var initialPrompt: [Int] { get set }
    var inputIds: MLMultiArray { get set }
    var cacheLength: MLMultiArray { get set }

    func reset(prefilledCacheSize: Int, maxTokenContext: Int)
}

public protocol TextDecoding {
    var tokenizer: WhisperTokenizer? { get set }
    var prefillData: WhisperMLModel? { get set }
    var isModelMultilingual: Bool { get set }
    var supportsWordTimestamps: Bool { get }
    var logitsSize: Int? { get }
    var logitsFilters: [any LogitsFiltering]? { get set }
    var kvCacheEmbedDim: Int? { get }
    var kvCacheMaxSequenceLength: Int? { get }
    var windowSize: Int? { get }
    var embedSize: Int? { get }

    func predictLogits(
        _ inputs: any TextDecoderInputType
    ) async throws -> TextDecoderOutputType?

    func prepareDecoderInputs(withPrompt initialPrompt: [Int]) throws -> any DecodingInputsType

    func prefillDecoderInputs(
        _ decoderInputs: any DecodingInputsType,
        withOptions options: DecodingOptions?
    ) async throws -> any DecodingInputsType

    func prefillKVCache(
        withTask task: MLMultiArray,
        andLanguage language: MLMultiArray
    ) async throws -> DecodingCache?

    func decodeText(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: any DecodingInputsType,
        sampler tokenSampler: TokenSampling,
        options decoderOptions: DecodingOptions,
        callback: ((TranscriptionProgress) -> Bool?)?
    ) async throws -> DecodingResult

    @available(*, deprecated, message: "Subject to removal in a future version. Use `decodeText(from:using:sampler:options:callback:) async throws -> DecodingResult` instead.")
    @_disfavoredOverload
    func decodeText(
        from encoderOutput: MLMultiArray,
        using decoderInputs: DecodingInputs,
        sampler tokenSampler: TokenSampling,
        options decoderOptions: DecodingOptions,
        callback: ((TranscriptionProgress) -> Bool?)?
    ) async throws -> [DecodingResult]

    func detectLanguage(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: any DecodingInputsType,
        sampler tokenSampler: TokenSampling,
        options: DecodingOptions,
        temperature: FloatType
    ) async throws -> DecodingResult

    @available(*, deprecated, message: "Subject to removal in a future version. Use `detectLanguage(from:using:sampler:options:temperature:) async throws -> DecodingResult` instead.")
    @_disfavoredOverload
    func detectLanguage(
        from encoderOutput: MLMultiArray,
        using decoderInputs: DecodingInputs,
        sampler tokenSampler: TokenSampling,
        options: DecodingOptions,
        temperature: FloatType
    ) async throws -> [DecodingResult]

    static func updateKVCache(
        keyTensor: MLMultiArray,
        keySlice: MLMultiArray,
        valueTensor: MLMultiArray,
        valueSlice: MLMultiArray,
        insertAtIndex index: Int
    )
}

public extension TextDecoding {
    @available(*, deprecated, message: "Subject to removal in a future version. Use `decodeText(from:using:sampler:options:callback:) async throws -> DecodingResult` instead.")
    func decodeText(
        from encoderOutput: MLMultiArray,
        using decoderInputs: DecodingInputs,
        sampler tokenSampler: TokenSampling,
        options decoderOptions: DecodingOptions,
        callback: ((TranscriptionProgress) -> Bool?)?
    ) async throws -> [DecodingResult] {
        let result: DecodingResult = try await decodeText(
            from: encoderOutput,
            using: decoderInputs,
            sampler: tokenSampler,
            options: decoderOptions,
            callback: callback
        )
        return [result]
    }

    // Add default conformance for protocol
    func decodeText(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: any DecodingInputsType,
        sampler tokenSampler: TokenSampling,
        options decoderOptions: DecodingOptions,
        callback: ((TranscriptionProgress) -> Bool?)?
    ) async throws -> DecodingResult {
        let result: DecodingResult = try await decodeText(
            from: encoderOutput,
            using: decoderInputs,
            sampler: tokenSampler,
            options: decoderOptions,
            callback: callback
        )
        return result
    }

    @available(*, deprecated, message: "Subject to removal in a future version. Use `decodeText(from:using:sampler:options:callback:) async throws -> DecodingResult` instead.")
    func decodeText(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: DecodingInputs,
        sampler tokenSampler: TokenSampling,
        options decoderOptions: DecodingOptions,
        callback: ((TranscriptionProgress) -> Bool?)?
    ) async throws -> DecodingResult {
        let result: DecodingResult = try await decodeText(
            from: encoderOutput,
            using: decoderInputs,
            sampler: tokenSampler,
            options: decoderOptions,
            callback: callback
        )
        return result
    }

    @available(*, deprecated, message: "Subject to removal in a future version. Use `detectLanguage(from:using:sampler:options:temperature:) async throws -> DecodingResult` instead.")
    func detectLanguage(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: DecodingInputs,
        sampler tokenSampler: TokenSampling,
        options: DecodingOptions,
        temperature: FloatType
    ) async throws -> DecodingResult {
        let result: DecodingResult = try await detectLanguage(
            from: encoderOutput,
            using: decoderInputs,
            sampler: tokenSampler,
            options: options,
            temperature: temperature
        )
        return result
    }

    @available(*, deprecated, message: "Subject to removal in a future version. Use `detectLanguage(from:using:sampler:options:temperature:) async throws -> DecodingResult` instead.")
    func detectLanguage(
        from encoderOutput: MLMultiArray,
        using decoderInputs: DecodingInputs,
        sampler tokenSampler: TokenSampling,
        options: DecodingOptions,
        temperature: FloatType
    ) async throws -> [DecodingResult] {
        let result: DecodingResult = try await detectLanguage(
            from: encoderOutput,
            using: decoderInputs,
            sampler: tokenSampler,
            options: options,
            temperature: temperature
        )
        return [result]
    }

    func detectLanguage(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: any DecodingInputsType,
        sampler tokenSampler: TokenSampling,
        options: DecodingOptions,
        temperature: FloatType
    ) async throws -> DecodingResult {
        let result: DecodingResult = try await detectLanguage(
            from: encoderOutput,
            using: decoderInputs,
            sampler: tokenSampler,
            options: options,
            temperature: temperature
        )
        return result
    }

    func prepareDecoderInputs(withPrompt initialPrompt: [Int]) throws -> any DecodingInputsType {
        let tokenShape = [NSNumber(value: 1), NSNumber(value: initialPrompt.count)]

        // Initialize MLMultiArray for tokens
        let tokenMultiArray = try MLMultiArray(shape: tokenShape, dataType: .int32)

        // Assign token values to the MLMultiArray
        for (index, token) in initialPrompt.enumerated() {
            tokenMultiArray[index] = NSNumber(value: token)
        }

        guard let kvCacheEmbedDim = self.kvCacheEmbedDim else {
            throw WhisperError.prepareDecoderInputsFailed("Unable to determine kvCacheEmbedDim")
        }

        guard let kvCacheMaxSequenceLength = self.kvCacheMaxSequenceLength else {
            throw WhisperError.prepareDecoderInputsFailed("Unable to determine kvCacheMaxSequenceLength")
        }

        guard let encoderOutputDim = self.windowSize else {
            throw WhisperError.prepareDecoderInputsFailed("Unable to determine encoderOutputDim")
        }

        // Initialize each MLMultiArray
        let kvCacheEmbedDimValue = NSNumber(value: kvCacheEmbedDim)
        let kvCacheMaxSequenceLengthValue = NSNumber(value: kvCacheMaxSequenceLength)
        let encoderOutputDimValue = NSNumber(value: encoderOutputDim)

        let inputIds = try MLMultiArray(shape: [1], dataType: .int32, initialValue: Int32(0))
        let cacheLength = try MLMultiArray(shape: [1], dataType: .int32, initialValue: Int32(0))
        let keyCache = try MLMultiArray(shape: [1, kvCacheEmbedDimValue, 1, kvCacheMaxSequenceLengthValue], dataType: .float16, initialValue: FloatType(0))
        let valueCache = try MLMultiArray(shape: [1, kvCacheEmbedDimValue, 1, kvCacheMaxSequenceLengthValue], dataType: .float16, initialValue: FloatType(0))
        let alignmentWeights = try MLMultiArray(shape: [kvCacheMaxSequenceLengthValue, encoderOutputDimValue], dataType: .float16, initialValue: FloatType(0))
        let kvCacheUpdateMask = try MLMultiArray(shape: [1, kvCacheMaxSequenceLengthValue], dataType: .int32, initialValue: Int32(0))
        let decoderKeyPaddingMask = try MLMultiArray(shape: [1, kvCacheMaxSequenceLengthValue], dataType: .float16, initialValue: FloatType(-10000))
        let prefillKeyCache = try! MLMultiArray(shape: [1, kvCacheEmbedDimValue, 1, kvCacheMaxSequenceLengthValue], dataType: .float16)
        let prefillValueCache = try! MLMultiArray(shape: [1, kvCacheEmbedDimValue, 1, kvCacheMaxSequenceLengthValue], dataType: .float16)

        // Initialize default masks
        kvCacheUpdateMask[0] = 1.0
        decoderKeyPaddingMask[0] = 0.0

        let decoderInputs = DecodingInputs(
            initialPrompt: initialPrompt,
            inputIds: inputIds,
            cacheLength: cacheLength,
            keyCache: keyCache,
            valueCache: valueCache,
            alignmentWeights: alignmentWeights,
            kvCacheUpdateMask: kvCacheUpdateMask,
            decoderKeyPaddingMask: decoderKeyPaddingMask,
            prefillKeyCache: prefillKeyCache,
            prefillValueCache: prefillValueCache
        )

        return decoderInputs
    }

    func prefillDecoderInputs(
        _ decoderInputs: any DecodingInputsType,
        withOptions options: DecodingOptions?
    ) async throws -> any DecodingInputsType {
        guard let tokenizer = tokenizer else {
            // Tokenizer required for prefill
            throw WhisperError.tokenizerUnavailable()
        }

        guard let prefilledDecoderInputs = decoderInputs as? DecodingInputs else {
            throw WhisperError.prepareDecoderInputsFailed("Unable to cast decoderInputs to DecodingInputs")
        }

        // Setup prefill tokens based on task and language
        var prefillTokens: [Int] = [tokenizer.specialTokens.startOfTranscriptToken] // SOT

        var languageToken: Int = tokenizer.specialTokens.englishToken
        var taskToken: Int = tokenizer.specialTokens.transcribeToken

        // Multilingual models require language and task tokens
        if let options = options {
            if isModelMultilingual {
                // Set languageToken
                let languageTokenString = "<|\(options.language ?? Constants.defaultLanguageCode)|>"
                languageToken = tokenizer.convertTokenToId(languageTokenString) ?? tokenizer.specialTokens.englishToken
                prefillTokens.append(languageToken)

                // Set taskToken
                let taskTokenString = "<|\(options.task)|>"
                taskToken = tokenizer.convertTokenToId(taskTokenString) ?? tokenizer.specialTokens.transcribeToken
                prefillTokens.append(taskToken)
            }

            // withoutTimestamps true in order to disable timestamps
            let timestampsToken = options.withoutTimestamps ? tokenizer.specialTokens.noTimestampsToken : tokenizer.specialTokens.timeTokenBegin
            prefillTokens.append(timestampsToken)

            // Add prompt tokens
            if let promptTokens = options.promptTokens {
                let maxPromptLen = (Constants.maxTokenContext / 2) - 1
                let trimmedPromptTokens = Array(promptTokens.suffix(maxPromptLen)).filter { $0 < tokenizer.specialTokens.specialTokenBegin }
                prefillTokens = [tokenizer.specialTokens.startOfPreviousToken] + trimmedPromptTokens + prefillTokens
            }

            // Add prefix tokens
            if let prefixTokens = options.prefixTokens {
                let trimmedPrefixTokens = Array(prefixTokens.suffix(Constants.maxTokenContext / 2)).filter { $0 < tokenizer.specialTokens.specialTokenBegin }
                prefillTokens.append(contentsOf: trimmedPrefixTokens)
            }
        }

        prefilledDecoderInputs.initialPrompt = prefillTokens
        var prefilledCacheSize = 0
        if options?.usePrefillCache ?? false,
           prefillData != nil,
           options?.promptTokens == nil // TODO: allow prefill cache to be used with prompt tokens, currently breaks if it starts at non-zero index
        {
            // Prefilling kv cache data requires non-nil task and language tokens, set defaults if not provided
            // Task tokens are remapped to 0->transcribe and 1->translate for the prefill lookup table
            let task = try MLMultiArray.from([taskToken == tokenizer.specialTokens.transcribeToken ? 0 : 1])
            let lang = try MLMultiArray.from([languageToken])
            guard let prefillOutput = try await self.prefillKVCache(withTask: task, andLanguage: lang) else {
                Logging.error("Unable to prefill cache")
                return prefilledDecoderInputs
            }

            // Prefill kv cache
            prefilledDecoderInputs.prefillKeyCache = prefillOutput.keyCache!
            prefilledDecoderInputs.prefillValueCache = prefillOutput.valueCache!

            TextDecoder.updateKVCache(keyTensor: prefilledDecoderInputs.keyCache,
                                      keySlice: prefilledDecoderInputs.prefillKeyCache,
                                      valueTensor: prefilledDecoderInputs.valueCache,
                                      valueSlice: prefilledDecoderInputs.prefillValueCache,
                                      insertAtIndex: prefillTokens.firstIndex(of: tokenizer.specialTokens.startOfTranscriptToken) ?? 0)
            prefilledDecoderInputs.cacheLength[0] = prefilledDecoderInputs.prefillKeyCache.shape[3]
            prefilledCacheSize = prefilledDecoderInputs.cacheLength[0].intValue
        }

        // Setup masks based on prefill values
        prefilledCacheSize += 1 // Add 1 for initial masked cache update
        for i in 0..<prefilledCacheSize {
            prefilledDecoderInputs.kvCacheUpdateMask[i] = 0.0
            prefilledDecoderInputs.decoderKeyPaddingMask[i] = 0.0
        }
        prefilledDecoderInputs.kvCacheUpdateMask[prefilledCacheSize - 1] = 1.0

        return prefilledDecoderInputs
    }

    func prefillKVCache(withTask task: MLMultiArray, andLanguage language: MLMultiArray) async throws -> DecodingCache? {
        let modelInputs = TextDecoderCachePrefillInput(
            task: task,
            language: language
        )

        guard let prefillModel = prefillData?.model else {
            return nil
        }

        try Task.checkCancellation()

        let outputFeatures = try await prefillModel.asyncPrediction(from: modelInputs, options: MLPredictionOptions())

        let output = TextDecoderCachePrefillOutput(features: outputFeatures)

        let kvCache = DecodingCache(
            keyCache: output.key_cache_prefill,
            valueCache: output.value_cache_prefill,
            alignmentWeights: nil
        )

        return kvCache
    }

    static func updateKVCache(keyTensor: MLMultiArray, keySlice: MLMultiArray,
                              valueTensor: MLMultiArray, valueSlice: MLMultiArray,
                              insertAtIndex index: Int)
    {
        let tensorShape = keyTensor.shape.map { $0.intValue }
        let sliceShape = keySlice.shape.map { $0.intValue }
        let sliceStrides = keySlice.strides.map { $0.intValue } // same for val
        let bytesPerSample = MemoryLayout<FloatType>.size

        keyTensor.withUnsafeMutableBytes { keyTensorPointer, keyTargetStrides in
            keySlice.withUnsafeBytes { keySlicePointer in
                valueTensor.withUnsafeMutableBytes { valueTensorPointer, valueTargetStrides in
                    valueSlice.withUnsafeBytes { valueSlicePointer in
                        // Assuming batch size is always 1
                        DispatchQueue.concurrentPerform(iterations: tensorShape[1]) { j in
                            // Slice size is 3 for prefill and 1 for decode loops
                            for k in 0..<sliceShape[3] {
                                // Equivalent to:
                                // `tensor[0, j, 0, k + index] = slice[0, j, 0, k + index]`
                                let keyDestIndex = j * keyTargetStrides[1] + (index + k) * keyTargetStrides[3]
                                let keyDest = keyTensorPointer.baseAddress! + keyDestIndex * bytesPerSample

                                let keySliceIndex = j * sliceStrides[1] + k * sliceStrides[3]
                                let keySlice = keySlicePointer.baseAddress! + keySliceIndex * bytesPerSample
                                memcpy(keyDest, keySlice, bytesPerSample)

                                let valDestIndex = j * valueTargetStrides[1] + (index + k) * valueTargetStrides[3]
                                let valDest = valueTensorPointer.baseAddress! + valDestIndex * bytesPerSample

                                let valSliceIndex = j * sliceStrides[1] + k * sliceStrides[3]
                                let valSlice = valueSlicePointer.baseAddress! + valSliceIndex * bytesPerSample
                                memcpy(valDest, valSlice, bytesPerSample)
                            }
                        }
                    }
                }
            }
        }
    }

    static func updateAlignmentWeights(
        alignmentTensor: MLMultiArray,
        alignmentSlice: MLMultiArray,
        insertAtIndex tokenIndex: Int
    ) {
        let tensorShape = alignmentTensor.shape.map { $0.intValue }
        let sliceStrides = alignmentSlice.strides.map { $0.intValue }
        let bytesPerSample = MemoryLayout<FloatType>.size

        alignmentTensor.withUnsafeMutableBytes { alignmentPointer, alignmentStrides in
            alignmentSlice.withUnsafeBytes { slicePointer in
                // Process each column
                for column in 0..<tensorShape[1] {
                    // Calculate source and destination indices
                    let destIndex = (tokenIndex + 1) * alignmentStrides[0] + column * alignmentStrides[1]
                    let sourceIndex = column * sliceStrides[1]

                    // Copy the weight value
                    let dest = alignmentPointer.baseAddress! + destIndex * bytesPerSample
                    let source = slicePointer.baseAddress! + sourceIndex * bytesPerSample
                    memcpy(dest, source, bytesPerSample)
                }
            }
        }
    }
}

public class TextDecoderContextPrefill: WhisperMLModel {
    public var model: MLModel?
}

open class TextDecoder: TextDecoding, WhisperMLModel {
    public var model: MLModel?
    public var tokenizer: WhisperTokenizer?
    public var prefillData: WhisperMLModel?
    public var isModelMultilingual: Bool = false
    public var logitsFilters: [any LogitsFiltering]? = []
    private let earlyStopActor = EarlyStopActor()
    private var languageLogitsFilter: LanguageLogitsFilter?

    public init() {}

    public var supportsWordTimestamps: Bool {
        return ModelUtilities.getModelOutputDimention(model, named: "alignment_heads_weights", position: 0) != nil
    }

    public var logitsSize: Int? {
        return ModelUtilities.getModelOutputDimention(model, named: "logits", position: 2)
    }

    public var kvCacheEmbedDim: Int? {
        return ModelUtilities.getModelInputDimention(model, named: "key_cache", position: 1)
    }

    public var kvCacheMaxSequenceLength: Int? {
        return ModelUtilities.getModelInputDimention(model, named: "key_cache", position: 3)
    }

    public var windowSize: Int? {
        return ModelUtilities.getModelInputDimention(model, named: "encoder_output_embeds", position: 3)
    }

    public var embedSize: Int? {
        return ModelUtilities.getModelInputDimention(model, named: "encoder_output_embeds", position: 1)
    }

    /// Override default so we an unload the prefill data as well
    public func unloadModel() {
        model = nil
        prefillData = nil
        languageLogitsFilter = nil
    }

    func debugCaches(decoderInputs: DecodingInputs, tokenIndex: Int, prefillSize: Int) {
        Logging.debug("--------------- DECODER INPUTS DEBUG ---------------")
        Logging.debug(
            String(
                format: "Cache Length: %2.0f Input Token: %4.0f",
                decoderInputs.cacheLength[0].floatValue,
                decoderInputs.inputIds[0].floatValue
            )
        )
        Logging.debug("Key Cache | Val Cache | Align Cache | Update Mask | Decoder Mask | Position")

        for i in 0..<min(prefillSize + 4, Constants.maxTokenContext) {
            let formattedString = String(format: "%9.6f | %9.6f | %9.6f | %11.0f | %12.0f | %d",
                                         decoderInputs.keyCache[i].floatValue,
                                         decoderInputs.valueCache[i].floatValue,
                                         decoderInputs.alignmentWeights[i * 1500].floatValue,
                                         decoderInputs.kvCacheUpdateMask[i].floatValue,
                                         decoderInputs.decoderKeyPaddingMask[i].floatValue,
                                         i)
            Logging.debug(formattedString)
        }
    }

    public func predictLogits(
        _ inputs: TextDecoderInputType
    ) async throws -> TextDecoderOutputType? {
        guard let inputs = inputs as? TextDecoderMLMultiArrayInputType else {
            throw WhisperError.transcriptionFailed("Input must be TextDecoderMLMultiArrayInputType")
        }

        let result = try await predictLogits(
            inputIds: inputs.inputIds,
            cacheLength: inputs.cacheLength,
            keyCache: inputs.keyCache,
            valueCache: inputs.valueCache,
            kvCacheUpdateMask: inputs.kvCacheUpdateMask,
            encoderOutputEmbeds: inputs.encoderOutputEmbeds,
            decoderKeyPaddingMask: inputs.decoderKeyPaddingMask
        )

        return TextDecoderMLMultiArrayOutputType(logits: result?.logits, cache: result?.cache)
    }

    public func predictLogits(
        inputIds: MLMultiArray,
        cacheLength: MLMultiArray,
        keyCache: MLMultiArray,
        valueCache: MLMultiArray,
        kvCacheUpdateMask: MLMultiArray,
        encoderOutputEmbeds: MLMultiArray,
        decoderKeyPaddingMask: MLMultiArray
    ) async throws -> (logits: MLMultiArray?, cache: DecodingCache?)? {
        guard let model = model else {
            return nil
        }

        let modelInputs = TextDecoderInput(
            input_ids: inputIds,
            cache_length: cacheLength,
            key_cache: keyCache,
            value_cache: valueCache,
            kv_cache_update_mask: kvCacheUpdateMask,
            encoder_output_embeds: encoderOutputEmbeds,
            decoder_key_padding_mask: decoderKeyPaddingMask
        )

        try Task.checkCancellation()

        let outputFeatures = try await model.asyncPrediction(from: modelInputs, options: MLPredictionOptions())

        let output = TextDecoderOutput(features: outputFeatures)

        let logits = output.logits
        let cache = DecodingCache(
            keyCache: output.key_cache_updates,
            valueCache: output.value_cache_updates,
            alignmentWeights: output.alignment_heads_weights
        )

        return (logits, cache)
    }

    public func detectLanguage(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: any DecodingInputsType,
        sampler tokenSampler: TokenSampling,
        options: DecodingOptions,
        temperature: FloatType
    ) async throws -> DecodingResult {
        // Predict logits for 1 iteration with sot
        // 1. LanguageLogitsFilter for only language tokens
        // 2. GreedyTokenSampler for most likely language
        guard let tokenizer = tokenizer else {
            // Tokenizer required for decoding
            throw WhisperError.tokenizerUnavailable()
        }
        guard let logitsSize = logitsSize else {
            throw WhisperError.modelsUnavailable("Failed to read logits size from model")
        }

        var timings = TranscriptionTimings()
        let prefilledIndex = 0
        let currentTokens: [Int] = [tokenizer.specialTokens.startOfTranscriptToken]
        var logProbs: [Float] = Array(repeating: 0, count: prefilledIndex + 1)

        // Logits filters
        let languageLogitsFilter = self.languageLogitsFilter ?? LanguageLogitsFilter(
            allLanguageTokens: tokenizer.allLanguageTokens,
            logitsDim: logitsSize,
            sampleBegin: prefilledIndex
        )
        self.languageLogitsFilter = languageLogitsFilter

        let tokenIndex = 0
        let prefillToken = currentTokens[tokenIndex]
        var nextToken = prefillToken

        // Set the current token as model input
        decoderInputs.inputIds[0] = NSNumber(value: nextToken)
        decoderInputs.cacheLength[0] = NSNumber(value: tokenIndex)

        // MARK: Decoding Inference

        // Predict next token
        let inferenceTime = Date()

        Logging.debug("Detecting language...")
        guard let encoderOutput = encoderOutput as? MLMultiArray else {
            throw WhisperError.prepareDecoderInputsFailed("Input must be MLMultiArray")
        }
        guard let decoderInputs = decoderInputs as? DecodingInputs else {
            throw WhisperError.prepareDecoderInputsFailed("DecodingInputsType must be DecodingInputs")
        }
        let predictedLogits = try await self.predictLogits(
            TextDecoderMLMultiArrayInputType(
                inputIds: decoderInputs.inputIds,
                cacheLength: decoderInputs.cacheLength,
                keyCache: decoderInputs.keyCache,
                valueCache: decoderInputs.valueCache,
                kvCacheUpdateMask: decoderInputs.kvCacheUpdateMask,
                encoderOutputEmbeds: encoderOutput,
                decoderKeyPaddingMask: decoderInputs.decoderKeyPaddingMask
            )
        ) as? TextDecoderMLMultiArrayOutputType

        guard let decoderOutput = predictedLogits else {
            Logging.error("Unable to decode logits")
            throw WhisperError.decodingLogitsFailed()
        }

        let decodingInferenceTime = Date().timeIntervalSince(inferenceTime)
        timings.decodingPredictions += decodingInferenceTime

        // MARK: Non-inference

        // Update predicted token as current
        guard let outputLogits = decoderOutput.logits else {
            throw WhisperError.decodingLogitsFailed("Logits unavailable in language detection output")
        }
        let logits = languageLogitsFilter.filterLogits(outputLogits, withTokens: currentTokens)

        // MARK: Sampling

        let samplingStartTime = Date()

        let sampleResult = tokenSampler.update(tokens: currentTokens, logits: logits, logProbs: logProbs)

        guard let sampledToken = sampleResult.tokens.last else {
            throw WhisperError.decodingFailed("Language detection sampler returned empty tokens")
        }
        nextToken = sampledToken
        logProbs = sampleResult.logProbs

        let samplingTime = Date().timeIntervalSince(samplingStartTime)
        timings.decodingSampling += samplingTime

        var languageProbs = [String: Float]()
        for (tokenIndex, token) in sampleResult.tokens.enumerated() {
            if tokenizer.allLanguageTokens.contains(token) {
                let language = tokenizer.decode(tokens: [token]).trimmingSpecialTokenCharacters()
                languageProbs[language] = sampleResult.logProbs[tokenIndex]
            }
        }

        let sampledLanguage = tokenizer.decode(tokens: [nextToken]).trimmingSpecialTokenCharacters()
        let detectedLanguage: String
        if Constants.languageCodes.contains(sampledLanguage) {
            detectedLanguage = sampledLanguage
            Logging.debug("Detected language: \(sampledLanguage)")
        } else {
            detectedLanguage = Constants.defaultLanguageCode
            Logging.error("Detected language \(sampledLanguage) is not supported, defaulting to \(Constants.defaultLanguageCode)")
        }
        return DecodingResult(
            language: detectedLanguage,
            languageProbs: languageProbs,
            tokens: [],
            tokenLogProbs: [],
            text: "",
            avgLogProb: 0.0,
            noSpeechProb: 0.0,
            temperature: 0.0,
            compressionRatio: 0.0,
            cache: nil,
            timings: timings,
            fallback: nil
        )
    }

    public func decodeText(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: any DecodingInputsType,
        sampler tokenSampler: TokenSampling,
        options: DecodingOptions,
        callback: TranscriptionCallback = nil
    ) async throws -> DecodingResult {
        guard let tokenizer else {
            // Tokenizer required for decoding
            throw WhisperError.tokenizerUnavailable()
        }

        // Single loop variables
        var timings = TranscriptionTimings()
        let prefilledIndex = decoderInputs.cacheLength[0].intValue
        let initialPromptIndex = decoderInputs.initialPrompt.count
        var currentTokens: [Int] = decoderInputs.initialPrompt
        guard let lastPromptToken = decoderInputs.initialPrompt.last else {
            throw WhisperError.prepareDecoderInputsFailed("Initial prompt is empty")
        }
        var nextToken: Int = lastPromptToken
        var logProbs: [Float] = Array(repeating: 0, count: currentTokens.count)

        // Logits filters
        let logitsFilters = createLogitsFilters(options: options, prefilledIndex: prefilledIndex, initialPromptIndex: initialPromptIndex, tokenizer: tokenizer)

        // MARK: Main loop

        let loopCount = min(options.sampleLength, Constants.maxTokenContext - 1)
        Logging.debug("Running main loop for a maximum of \(loopCount) iterations, starting at index \(prefilledIndex)")
        var hasAlignment = false
        var isFirstTokenLogProbTooLow = false
        let windowUUID = UUID()
        await earlyStopActor.set(false, for: windowUUID)

        for tokenIndex in prefilledIndex..<loopCount {
            let loopStart = Date()

            let isPrefill = tokenIndex < initialPromptIndex - 1 // Prefill stops at the last token of the initial prompt
            let isLastPrefillToken = tokenIndex == initialPromptIndex - 1
            let isFirstToken = tokenIndex == prefilledIndex

            // Check if current index is part of the initial prompt
            if tokenIndex < initialPromptIndex {
                let isTimestampToken = currentTokens[tokenIndex] >= tokenizer.specialTokens.timeTokenBegin
                let modelPredictedTimestamp = nextToken >= tokenizer.specialTokens.timeTokenBegin

                // Force the token unless it's the last prefill token and both are timestamps
                if !(isLastPrefillToken && isTimestampToken && modelPredictedTimestamp) {
                    nextToken = currentTokens[tokenIndex]
                    Logging.debug("Forcing prompt tokenIndex: \(tokenIndex), token: \(nextToken), text: \(tokenizer.decode(tokens: [nextToken]))")
                } else {
                    // Last prefill was a timestamp but the model predicted a timestamp
                    currentTokens[tokenIndex] = nextToken
                    Logging.debug("Skipping prompt tokenIndex: \(tokenIndex), token: \(nextToken), text: \(tokenizer.decode(tokens: [nextToken]))")
                }
            }

            guard let decoderInputs = decoderInputs as? DecodingInputs else {
                throw WhisperError.prepareDecoderInputsFailed("DecodingInputsType must be DecodingInputs")
            }

            // Set the current token as model input
            decoderInputs.inputIds[0] = NSNumber(value: nextToken)
            decoderInputs.cacheLength[0] = NSNumber(value: tokenIndex)

            if tokenIndex <= prefilledIndex + 3 {
                debugCaches(decoderInputs: decoderInputs, tokenIndex: tokenIndex, prefillSize: prefilledIndex)
            }

            // MARK: Decoding Inference

            // Predict next token
            let inferenceTime = Date()

            guard let encoderOutput = encoderOutput as? MLMultiArray else {
                throw WhisperError.prepareDecoderInputsFailed("Input must be MLMultiArray")
            }
            let predictedLogits = try await self.predictLogits(
                TextDecoderMLMultiArrayInputType(
                    inputIds: decoderInputs.inputIds,
                    cacheLength: decoderInputs.cacheLength,
                    keyCache: decoderInputs.keyCache,
                    valueCache: decoderInputs.valueCache,
                    kvCacheUpdateMask: decoderInputs.kvCacheUpdateMask,
                    encoderOutputEmbeds: encoderOutput,
                    decoderKeyPaddingMask: decoderInputs.decoderKeyPaddingMask
                )
            ) as? TextDecoderMLMultiArrayOutputType

            guard let decoderOutput = predictedLogits else {
                throw WhisperError.decodingLogitsFailed("Unable to decode logits")
            }

            let decodingInferenceTime = Date().timeIntervalSince(inferenceTime)
            timings.decodingPredictions += decodingInferenceTime

            // MARK: Non-inference

            let nonInferenceStartTime = Date()

            // Update predicted token as current
            guard let outputLogits = decoderOutput.logits else {
                throw WhisperError.decodingLogitsFailed("Logits unavailable in decoder output")
            }
            var logits = outputLogits
            for filter in logitsFilters {
                logits = filter.filterLogits(logits, withTokens: currentTokens)
            }

            let filteringTime = Date().timeIntervalSince(nonInferenceStartTime)
            timings.decodingFiltering += filteringTime

            // MARK: Sampling

            let samplingStartTime = Date()

            let sampleResult = tokenSampler.update(tokens: currentTokens, logits: logits, logProbs: logProbs)

            guard let sampledToken = sampleResult.tokens.last else {
                throw WhisperError.decodingFailed("Token sampler returned empty tokens")
            }
            guard let sampledLogProb = sampleResult.logProbs.last else {
                throw WhisperError.decodingFailed("Token sampler returned empty logProbs")
            }
            nextToken = sampledToken
            let nextTokenLogProb = sampledLogProb

            Logging.debug("Predicted next tokenIndex: \(tokenIndex + 1), token: \(nextToken), text: \(tokenizer.decode(tokens: [nextToken]))")

            let samplingTime = Date().timeIntervalSince(samplingStartTime)
            timings.decodingSampling += samplingTime

            isFirstTokenLogProbTooLow =
                if isFirstToken, let firstTokenLogProbThreshold = options.firstTokenLogProbThreshold, nextTokenLogProb < firstTokenLogProbThreshold {
                    true
                } else {
                    false
                }
            let isSegmentCompleted =
                sampleResult.completed ||
                currentTokens.count >= Constants.maxTokenContext - 1 ||
                isFirstTokenLogProbTooLow

            if isSegmentCompleted {
                // Completed segment, stop the loop
                timings.decodingNonPrediction += Date().timeIntervalSince(nonInferenceStartTime)
                timings.decodingLoop += Date().timeIntervalSince(loopStart)
                timings.totalDecodingLoops += 1
                break
            } else {
                // MARK: KV Caching

                if !isPrefill {
                    // Found the next token, store it
                    currentTokens.append(nextToken)
                    logProbs.append(nextTokenLogProb)
                }

                // Update KV cache for this token
                guard let decoderCache = decoderOutput.cache,
                      let newKeyCache = decoderCache.keyCache,
                      let newValueCache = decoderCache.valueCache
                else {
                    fatalError("Invalid model output")
                }

                // tensor: [1, kvCacheEmbedDim, 1, kvCacheMaxSequenceLength], slice: [1, kvCacheEmbedDim, 1, 1]
                let kvStartTime = Date()
                TextDecoder.updateKVCache(keyTensor: decoderInputs.keyCache,
                                          keySlice: newKeyCache,
                                          valueTensor: decoderInputs.valueCache,
                                          valueSlice: newValueCache,
                                          insertAtIndex: tokenIndex)

                decoderInputs.decoderKeyPaddingMask[tokenIndex + 1] = 0

                decoderInputs.kvCacheUpdateMask[tokenIndex] = 0
                decoderInputs.kvCacheUpdateMask[tokenIndex + 1] = 1

                // Update alignment weights for token if present
                if let newAlignmentWeights = decoderOutput.cache?.alignmentWeights {
                    hasAlignment = true
                    TextDecoder.updateAlignmentWeights(
                        alignmentTensor: decoderInputs.alignmentWeights,
                        alignmentSlice: newAlignmentWeights,
                        insertAtIndex: tokenIndex
                    )
                }

                let kvTime = Date().timeIntervalSince(kvStartTime)
                timings.decodingKvCaching += kvTime
                timings.totalKVUpdateRuns += 1

                // Prepare results
                let wordTokens = currentTokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
                let slicedTextTokens = options.skipSpecialTokens ? wordTokens : currentTokens
                let currentTranscript = tokenizer.decode(tokens: slicedTextTokens)
                let averageLogProb = logProbs.reduce(0, +) / Float(logProbs.count)
                let compressionRatio = TextUtilities.compressionRatio(of: currentTokens)

                let result = TranscriptionProgress(timings: timings, text: currentTranscript, tokens: currentTokens, avgLogprob: averageLogProb, compressionRatio: compressionRatio)

                // Call the callback if it is provided on a background thread
                if let callback = callback {
                    Task.detached(priority: .low) { [weak self] in
                        guard let self = self else { return }
                        let shouldContinue = callback(result)
                        if let shouldContinue = shouldContinue, !shouldContinue, !isPrefill {
                            Logging.debug("Early stopping")
                            await self.earlyStopActor.set(true, for: windowUUID)
                        }
                    }
                }
            }

            timings.decodingNonPrediction += Date().timeIntervalSince(nonInferenceStartTime)
            timings.decodingLoop += Date().timeIntervalSince(loopStart)
            timings.totalDecodingLoops += 1

            if tokenIndex == prefilledIndex {
                Logging.debug("Found first token at: \(Date())")
                timings.firstTokenTime = CFAbsoluteTimeGetCurrent()
            }

            // Check if early stopping is triggered
            if await earlyStopActor.get(for: windowUUID) {
                break
            }
        }

        // Cleanup after loop completion
        if await earlyStopActor.remove(for: windowUUID) == nil {
            Logging.error("Early stop flag not found for window: \(windowUUID)")
        }

        var cache: DecodingCache?
        if let inputs = decoderInputs as? DecodingInputs {
            cache = DecodingCache(
                keyCache: inputs.keyCache,
                valueCache: inputs.valueCache,
                alignmentWeights: hasAlignment ? inputs.alignmentWeights : nil
            )
        }

        // NOTE:
        // While `currentTokens` and `logProbs` are usually the same length
        // `currentTokens` does not always contain an end of text token at the end (it is added by this finalize function),
        let finalSamplingResult = tokenSampler.finalize(tokens: currentTokens, logProbs: logProbs)
        let segmentTokens = finalSamplingResult.tokens
        let segmentLogProbs = finalSamplingResult.logProbs

        let startIndex = segmentTokens.firstIndex(of: tokenizer.specialTokens.startOfTranscriptToken) ?? 0
        let endIndex = segmentTokens.firstIndex(of: tokenizer.specialTokens.endToken) ?? segmentTokens.count
        let filteredTokens = Array(segmentTokens[startIndex...endIndex])
        let filteredLogProbs = Array(segmentLogProbs[startIndex...endIndex])

        let sumLogProbs = filteredLogProbs.reduce(0, +)
        let avgLogProbs = sumLogProbs / Float(filteredLogProbs.count)

        var tokenProbs = [[Int: Float]]()
        for (index, token) in filteredTokens.enumerated() {
            tokenProbs.append([token: filteredLogProbs[index]])
        }

        let wordTokens = filteredTokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
        let finalCompressionRatio = TextUtilities.compressionRatio(of: wordTokens)

        var temperature = options.temperature
        if let sampler = tokenSampler as? GreedyTokenSampler {
            // Convert Float16 temperature to Float with 3 decimal places
            temperature = Float(sampler.temperature).rounded(3)
        }

        let noSpeechProb: Float = 0 // TODO: implement no speech prob

        // If language is still nil here, check language can be inferred from tokens
        var language = options.language ?? Constants.defaultLanguageCode
        var languageProbs = [String: Float]()
        if options.language == nil {
            // Find the first token that is a recognized language token
            if let predictedLanguageIndex = filteredTokens.firstIndex(where: { tokenizer.allLanguageTokens.contains($0) }),
               predictedLanguageIndex < tokenProbs.count
            {
                let predictedLanguageToken = filteredTokens[predictedLanguageIndex]
                // Decode the predicted language token to get the language
                language = tokenizer.decode(tokens: [predictedLanguageToken]).trimmingSpecialTokenCharacters()

                // Fetch the corresponding probability for the predicted language
                let probsDict = tokenProbs[predictedLanguageIndex]
                languageProbs[language] = probsDict[predictedLanguageToken] ?? 0.0
            } else {
                // Set default values if no language token is found
                languageProbs[language] = 0.0
            }
        } else {
            // If language is provided, set the logprob to 0.0
            languageProbs[language] = 0.0
        }

        let transcript = tokenizer.decode(tokens: filteredTokens)

        Logging.debug("Completed window: \(transcript)")

        let decodingFallback = DecodingFallback(
            options: options,
            isFirstTokenLogProbTooLow: isFirstTokenLogProbTooLow,
            noSpeechProb: noSpeechProb,
            compressionRatio: finalCompressionRatio,
            avgLogProb: avgLogProbs
        )

        let decodingResult = DecodingResult(
            language: language,
            languageProbs: languageProbs,
            tokens: filteredTokens,
            tokenLogProbs: tokenProbs,
            text: transcript,
            avgLogProb: avgLogProbs,
            noSpeechProb: noSpeechProb,
            temperature: temperature,
            compressionRatio: finalCompressionRatio,
            cache: cache,
            timings: timings,
            fallback: decodingFallback
        )
        return decodingResult
    }
    
    internal func createLogitsFilters(
        options: DecodingOptions,
        prefilledIndex: Int,
        initialPromptIndex: Int,
        tokenizer: WhisperTokenizer
    ) -> [any LogitsFiltering] {
        // Start with custom logits filters
        var allFilters: [any LogitsFiltering] = logitsFilters ?? []

        // Add built-in filters based on options
        if options.suppressBlank {
            allFilters.append(
                SuppressBlankFilter(
                    specialTokens: tokenizer.specialTokens,
                    sampleBegin: prefilledIndex
                )
            )
        }

        if !options.supressTokens.isEmpty {
            let filteredSupressTokens = options.supressTokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
            allFilters.append(SuppressTokensFilter(suppressTokens: filteredSupressTokens))
        }

        if !options.withoutTimestamps {
            let maxInitialTimestampIndex: Int? =
                if let maxInitialTimestamp = options.maxInitialTimestamp {
                    Int(maxInitialTimestamp / WhisperKit.secondsPerTimeToken)
                } else {
                    nil
                }
            allFilters.append(
                TimestampRulesFilter(
                    specialTokens: tokenizer.specialTokens,
                    sampleBegin: initialPromptIndex,
                    maxInitialTimestampIndex: maxInitialTimestampIndex,
                    isModelMultilingual: isModelMultilingual
                )
            )
        }
        
        return allFilters
    }
}
