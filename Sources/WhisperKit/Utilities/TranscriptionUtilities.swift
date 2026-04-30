//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

/// A utility struct providing transcription-related functionality for processing and manipulating transcription results
public struct TranscriptionUtilities {

    private init() {}

    // MARK: Public

    /// Formats transcription segments into an array of strings, optionally including timestamps
    /// - Parameters:
    ///   - segments: Array of transcription segments to format
    ///   - withTimestamps: Whether to include timestamps in the output (default: true)
    /// - Returns: Array of formatted strings, each containing a segment's text and optionally its timestamps
    public static func formatSegments(_ segments: [TranscriptionSegment], withTimestamps: Bool = true) -> [String] {
        var lines = [String]()
        for segment in segments {
            let start = segment.start
            let end = segment.end
            let text = segment.text
            let timestamps = withTimestamps ? "[\(Logging.formatTimestamp(start)) --> \(Logging.formatTimestamp(end))] " : ""
            let line = "\(timestamps)\(text)"
            lines.append(line)
        }
        return lines
    }

    /// Finds the longest common prefix between two arrays of word timings
    /// - Parameters:
    ///   - words1: First array of word timings
    ///   - words2: Second array of word timings
    /// - Returns: Array of word timings representing the longest common prefix
    public static func findLongestCommonPrefix(_ words1: [WordTiming], _ words2: [WordTiming]) -> [WordTiming] {
        let commonPrefix = zip(words1, words2).prefix(while: { $0.word.normalized == $1.word.normalized })
        return commonPrefix.map { $0.1 }
    }

    /// Finds the longest different suffix between two arrays of word timings
    /// - Parameters:
    ///   - words1: First array of word timings
    ///   - words2: Second array of word timings
    /// - Returns: Array of word timings representing the longest different suffix
    public static func findLongestDifferentSuffix(_ words1: [WordTiming], _ words2: [WordTiming]) -> [WordTiming] {
        let commonPrefix = findLongestCommonPrefix(words1, words2)
        let remainingWords = words2[commonPrefix.count...]
        return Array(remainingWords)
    }

    /// Updates the timings of a transcription segment by adding a seek time offset
    /// - Parameters:
    ///   - segment: The transcription segment to update
    ///   - seekTime: The time offset to add to all timings
    /// - Returns: Updated transcription segment with adjusted timings
    public static func updateSegmentTimings(segment: TranscriptionSegment, seekTime: Float) -> TranscriptionSegment {
        var updatedSegment = segment
        let seekOffsetIndex = Int(seekTime * Float(WhisperKit.sampleRate))
        updatedSegment.seek += seekOffsetIndex
        updatedSegment.start += seekTime
        updatedSegment.end += seekTime
        if var words = updatedSegment.words {
            for wordIndex in 0..<words.count {
                words[wordIndex].start += seekTime
                words[wordIndex].end += seekTime
            }
            updatedSegment.words = words
        }
        return updatedSegment
    }

    /// Merges multiple transcription results into a single result
    /// - Parameters:
    ///   - results: Array of transcription results to merge
    ///   - confirmedWords: Optional array of confirmed word timings to use instead of merging text
    /// - Returns: A single merged transcription result
    public static func mergeTranscriptionResults(_ results: [TranscriptionResult?], confirmedWords: [WordTiming]? = nil) -> TranscriptionResult {
        var mergedText = ""
        if let words = confirmedWords {
            mergedText = words.map { $0.word }.joined()
        } else {
            mergedText = results.map { $0?.text ?? "" }.joined(separator: " ")
        }

        // Merge segments
        let validResults = results.compactMap { $0 }
        var mergedSegments = [TranscriptionSegment]()
        var previousSeek: Float = 0.0
        for (resultIndex, result) in validResults.enumerated() {
            let seekTime = result.seekTime ?? previousSeek
            for (segmentIndex, segment) in result.segments.enumerated() {
                var updatedSegment = segment
                updatedSegment.id = resultIndex + segmentIndex
                mergedSegments.append(updatedSegment)
            }
            // Update previousSeek only if seekTime is nil
            if result.seekTime == nil {
                previousSeek += Float(result.timings.inputAudioSeconds)
            } else {
                previousSeek = seekTime + Float(result.timings.inputAudioSeconds)
            }
        }

        let language = validResults.first?.language ?? Constants.defaultLanguageCode

        // Calculate the earliest start and latest end times
        let earliestPipelineStart = validResults.map { $0.timings.pipelineStart }.min() ?? 0
        let earliestTokenTime = validResults.map { $0.timings.firstTokenTime }.min() ?? 0
        let latestPipelineEnd = validResults.map { $0.timings.pipelineStart + $0.timings.fullPipeline }.max() ?? 0

        // Calculate the "user" pipeline time, excluding the time spent in concurrent pipelines
        let userPipelineDuration = latestPipelineEnd - earliestPipelineStart
        let systemPipelineDuration = validResults.map { $0.timings.fullPipeline }.reduce(0, +)
        let fullPipelineDuration = min(userPipelineDuration, systemPipelineDuration)

        // Update the merged timings with non-overlapping time values
        var mergedTimings = TranscriptionTimings(
            modelLoading: validResults.map { $0.timings.modelLoading }.max() ?? 0,
            prewarmLoadTime: validResults.map { $0.timings.prewarmLoadTime }.max() ?? 0,
            encoderLoadTime: validResults.map { $0.timings.encoderLoadTime }.max() ?? 0,
            decoderLoadTime: validResults.map { $0.timings.decoderLoadTime }.max() ?? 0,
            tokenizerLoadTime: validResults.map { $0.timings.tokenizerLoadTime }.max() ?? 0,
            audioLoading: validResults.map { $0.timings.audioLoading }.reduce(0, +),
            audioProcessing: validResults.map { $0.timings.audioProcessing }.reduce(0, +),
            logmels: validResults.map { $0.timings.logmels }.reduce(0, +),
            encoding: validResults.map { $0.timings.encoding }.reduce(0, +),
            prefill: validResults.map { $0.timings.prefill }.reduce(0, +),
            decodingInit: validResults.map { $0.timings.decodingInit }.reduce(0, +),
            decodingLoop: validResults.map { $0.timings.decodingLoop }.reduce(0, +),
            decodingPredictions: validResults.map { $0.timings.decodingPredictions }.reduce(0, +),
            decodingFiltering: validResults.map { $0.timings.decodingFiltering }.reduce(0, +),
            decodingSampling: validResults.map { $0.timings.decodingSampling }.reduce(0, +),
            decodingFallback: validResults.map { $0.timings.decodingFallback }.reduce(0, +),
            decodingWindowing: validResults.map { $0.timings.decodingWindowing }.reduce(0, +),
            decodingKvCaching: validResults.map { $0.timings.decodingKvCaching }.reduce(0, +),
            decodingTimestampAlignment: validResults.map { $0.timings.decodingWordTimestamps }.reduce(0, +),
            decodingNonPrediction: validResults.map { $0.timings.decodingNonPrediction }.reduce(0, +),
            totalAudioProcessingRuns: validResults.map { $0.timings.totalAudioProcessingRuns }.reduce(0, +),
            totalLogmelRuns: validResults.map { $0.timings.totalLogmelRuns }.reduce(0, +),
            totalEncodingRuns: validResults.map { $0.timings.totalEncodingRuns }.reduce(0, +),
            totalDecodingLoops: validResults.map { $0.timings.totalDecodingLoops }.reduce(0, +),
            totalKVUpdateRuns: validResults.map { $0.timings.totalKVUpdateRuns }.reduce(0, +),
            totalTimestampAlignmentRuns: validResults.map { $0.timings.totalTimestampAlignmentRuns }.reduce(0, +),
            totalDecodingFallbacks: validResults.map { $0.timings.totalDecodingFallbacks }.reduce(0, +),
            totalDecodingWindows: validResults.map { $0.timings.totalDecodingWindows }.reduce(0, +),
            fullPipeline: fullPipelineDuration
        )

        mergedTimings.pipelineStart = earliestPipelineStart
        mergedTimings.firstTokenTime = earliestTokenTime
        mergedTimings.inputAudioSeconds = validResults.map { $0.timings.inputAudioSeconds }.reduce(0, +)

        return TranscriptionResult(
            text: mergedText,
            segments: mergedSegments,
            language: language,
            timings: mergedTimings
        )
    }
}

@available(*, deprecated, message: "Subject to removal in a future version. Use `TranscriptionUtilities.formatSegments(_:withTimestamps:)` instead.")
public func formatSegments(_ segments: [TranscriptionSegment], withTimestamps: Bool = true) -> [String] {
    return TranscriptionUtilities.formatSegments(segments, withTimestamps: withTimestamps)
}

@available(*, deprecated, message: "Subject to removal in a future version. Use `TranscriptionUtilities.findLongestCommonPrefix(_:_:)` instead.")
public func findLongestCommonPrefix(_ words1: [WordTiming], _ words2: [WordTiming]) -> [WordTiming] {
    return TranscriptionUtilities.findLongestCommonPrefix(words1, words2)
}

@available(*, deprecated, message: "Subject to removal in a future version. Use `TranscriptionUtilities.findLongestDifferentSuffix(_:_:)` instead.")
public func findLongestDifferentSuffix(_ words1: [WordTiming], _ words2: [WordTiming]) -> [WordTiming] {
    TranscriptionUtilities.findLongestDifferentSuffix(words1, words2)
}

@available(*, deprecated, message: "Subject to removal in a future version. Use `TranscriptionUtilities.mergeTranscriptionResults(_:confirmedWords:)` instead.")
public func mergeTranscriptionResults(_ results: [TranscriptionResult?], confirmedWords: [WordTiming]? = nil) -> TranscriptionResult {
    return TranscriptionUtilities.mergeTranscriptionResults(results, confirmedWords: confirmedWords)
}

@available(*, deprecated, message: "Subject to removal in a future version. Use `TranscriptionUtilities.updateSegmentTimings(segment:seekTime:)` instead.")
public func updateSegmentTimings(segment: TranscriptionSegment, seekTime: Float) -> TranscriptionSegment {
    return TranscriptionUtilities.updateSegmentTimings(segment: segment, seekTime: seekTime)
}
