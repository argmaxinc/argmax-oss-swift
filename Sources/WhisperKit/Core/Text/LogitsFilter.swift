//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import CoreML
import Foundation

public protocol LogitsFiltering {
    func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray
}

open class SuppressTokensFilter: LogitsFiltering {
    let suppressTokens: [Int]
    private let suppressTokenIndexes: [[Int]]

    public init(suppressTokens: [Int]) {
        self.suppressTokens = suppressTokens
        self.suppressTokenIndexes = suppressTokens.map { [0, 0, $0] }
    }

    public func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray {
        logits.fill(indexes: suppressTokenIndexes, with: -FloatType.infinity)
        return logits
    }
}

open class SuppressBlankFilter: LogitsFiltering {
    let specialTokens: SpecialTokens
    let sampleBegin: Int
    private let suppressTokenIndexes: [[Int]]

    public init(
        specialTokens: SpecialTokens,
        sampleBegin: Int
    ) {
        self.specialTokens = specialTokens
        self.sampleBegin = sampleBegin
        self.suppressTokenIndexes = [
            [0, 0, specialTokens.whitespaceToken],
            [0, 0, specialTokens.endToken],
        ]
    }

    public func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray {
        guard tokens.count == sampleBegin else {
            return logits
        }
        logits.fill(indexes: suppressTokenIndexes, with: -FloatType.infinity)
        return logits
    }
}

/// Implementation based on https://github.com/openai/whisper/blob/master/whisper/decoding.py#L441
open class TimestampRulesFilter: LogitsFiltering {
    let specialTokens: SpecialTokens
    let sampleBegin: Int
    let maxInitialTimestampIndex: Int?
    let isModelMultilingual: Bool

    public init(
        specialTokens: SpecialTokens,
        sampleBegin: Int,
        maxInitialTimestampIndex: Int?,
        isModelMultilingual: Bool
    ) {
        self.specialTokens = specialTokens
        self.sampleBegin = sampleBegin
        self.maxInitialTimestampIndex = maxInitialTimestampIndex
        self.isModelMultilingual = isModelMultilingual
    }

    public func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray {
        guard let sampleBegin = sampleBegin(for: tokens),
              sampleBegin <= tokens.count
        else {
            // Early return if we are still prefilling the prompt
            return logits
        }

        // suppress <|notimestamps|> which is handled by `withoutTimestamps`
        logits.fill(indexes: [[0, 0, specialTokens.noTimestampsToken]], with: -FloatType.infinity)

        if tokens.count > sampleBegin {
            // timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
            let sampledTokens = tokens[sampleBegin...]
            let lastWasTimestamp = sampledTokens.count >= 1 && sampledTokens.last! >= specialTokens.timeTokenBegin
            let penultimateWasTimestamp = sampledTokens.count < 2 || sampledTokens.dropLast().last! >= specialTokens.timeTokenBegin
            if lastWasTimestamp {
                if penultimateWasTimestamp {
                    // has to be non-timestamp
                    logits.fillLastDimension(indexes: specialTokens.timeTokenBegin..<logits.count, with: -FloatType.infinity)
                } else {
                    // cannot be normal text tokens
                    logits.fillLastDimension(indexes: 0..<specialTokens.endToken, with: -FloatType.infinity)
                }
            }

            let timestamps = sampledTokens.filter { $0 >= specialTokens.timeTokenBegin }
            if let lastTimestamp = timestamps.last {
                // timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
                // also force each segment to have a nonzero length, to prevent infinite looping
                let timestampLast =
                    if lastWasTimestamp && !penultimateWasTimestamp {
                        lastTimestamp
                    } else {
                        lastTimestamp + 1
                    }
                logits.fillLastDimension(indexes: specialTokens.timeTokenBegin..<timestampLast, with: -FloatType.infinity)
            }
        }

        // TODO: Allow model to predict initial timestamp
        // Currently initial timestamp is forced to <|0.00|> every time
//       if tokens.count == sampleBegin {
//           // suppress generating non-timestamp tokens at the beginning
//           logits.fillLastDimension(indexes: 0..<specialTokens.timeTokenBegin, with: -FloatType.infinity)
//           if let maxInitialTimestampIndex {
//               // apply the `maxInitialTimestamp` option
//               let lastAllowed = specialTokens.timeTokenBegin + maxInitialTimestampIndex + 1
//               logits.fillLastDimension(indexes: lastAllowed..<logits.count, with: -FloatType.infinity)
//           }
//       }

        // if sum of probability over timestamps is above any other token, sample timestamp
        if sumOfProbabilityOverTimestampsIsAboveAnyOtherToken(logits: logits, timeTokenBegin: specialTokens.timeTokenBegin) {
            logits.fillLastDimension(indexes: 0..<specialTokens.timeTokenBegin, with: -FloatType.infinity)
        }
        return logits
    }

    private func sampleBegin(for tokens: [Int]) -> Int? {
        if isModelMultilingual {
            // NOTE: for multilingual model we don't want to suppress "<|transcribe|>" or "<|translate|>" tokens
            if let taskTokenIndex = tokens.prefix(3).firstIndex(where: { $0 == specialTokens.transcribeToken || $0 == specialTokens.translateToken }) {
                return max(taskTokenIndex + 1, sampleBegin)
            } else {
                return nil
            }
        } else {
            return sampleBegin
        }
    }

    private func sumOfProbabilityOverTimestampsIsAboveAnyOtherToken(logits: MLMultiArray, timeTokenBegin: Int) -> Bool {
        let timeTokenBeginOffset = logits.linearOffset(for: [0, 0, timeTokenBegin])

        let logprobsInputPointer = UnsafeMutableRawBufferPointer(
            start: logits.dataPointer,
            count: logits.count * MemoryLayout<FloatType>.stride
        )

        guard let logprobsInputDescriptor = BNNSNDArrayDescriptor(
            data: logprobsInputPointer,
            scalarType: FloatType.self,
            shape: .vector(logits.count, stride: 1)
        ) else {
            Logging.error("Cannot create `logprobsInputDescriptor`")
            return false
        }

        let logprobs = BNNSNDArrayDescriptor.allocateUninitialized(
            scalarType: FloatType.self,
            shape: .vector(logits.count, stride: 1)
        )
        defer { logprobs.deallocate() }

        do {
            try BNNS.applyActivation(
                activation: BNNS.ActivationFunction.logSoftmax,
                input: logprobsInputDescriptor,
                output: logprobs,
                batchSize: 1
            )

            let timeTokenCount = logits.count - timeTokenBeginOffset
            let noTimeTokenCount = timeTokenBeginOffset
            let logSumExpInputPointer = UnsafeMutableRawBufferPointer(
                start: logprobs.data!.advanced(by: timeTokenBeginOffset * MemoryLayout<FloatType>.stride),
                count: timeTokenCount * MemoryLayout<FloatType>.stride
            )

            guard let logSumExpInputDescriptor = BNNSNDArrayDescriptor(
                data: logSumExpInputPointer,
                scalarType: FloatType.self,
                shape: .vector(timeTokenCount, stride: 1)
            ) else {
                Logging.error("Cannot create `logSumExpInputDescriptor`")
                return false
            }

            let timestampLogProb = BNNSNDArrayDescriptor.allocateUninitialized(
                scalarType: FloatType.self,
                shape: .vector(1, stride: 1)
            )
            defer { timestampLogProb.deallocate() }

            try BNNS.applyReduction(
                .logSumExp,
                input: logSumExpInputDescriptor,
                output: timestampLogProb,
                weights: nil
            )

            let maxTextTokenLogProbInputPointer = UnsafeMutableRawBufferPointer(
                start: logprobs.data,
                count: noTimeTokenCount * MemoryLayout<FloatType>.stride
            )

            guard let maxTextTokenLogProbInputDescriptor = BNNSNDArrayDescriptor(
                data: maxTextTokenLogProbInputPointer,
                scalarType: FloatType.self,
                shape: .vector(noTimeTokenCount, stride: 1)
            ) else {
                Logging.error("Cannot create `maxTextTokenLogProbInputDescriptor`")
                return false
            }

            let maxTextTokenLogProb = BNNSNDArrayDescriptor.allocateUninitialized(
                scalarType: FloatType.self,
                shape: .vector(1, stride: 1)
            )
            defer { maxTextTokenLogProb.deallocate() }

            try BNNS.applyReduction(
                .max,
                input: maxTextTokenLogProbInputDescriptor,
                output: maxTextTokenLogProb,
                weights: nil
            )

            guard let timestampLogProbValue = timestampLogProb.makeArray(of: FloatType.self)?.first,
                  let maxTextTokenLogProbValue = maxTextTokenLogProb.makeArray(of: FloatType.self)?.first
            else {
                Logging.error("Cannot create logProb arrays")
                return false
            }
            return timestampLogProbValue > maxTextTokenLogProbValue
        } catch {
            Logging.error("TimestampRulesFilter error: \(error)")
            return false
        }
    }
}

open class LanguageLogitsFilter: LogitsFiltering {
    let allLanguageTokens: Set<Int>
    let logitsDim: Int
    let sampleBegin: Int
    let nonLanguageTokenIndexes: [[Int]]

    public init(allLanguageTokens: Set<Int>, logitsDim: Int, sampleBegin: Int) {
        self.allLanguageTokens = allLanguageTokens
        self.logitsDim = logitsDim
        self.sampleBegin = sampleBegin
        self.nonLanguageTokenIndexes = LanguageLogitsFilter.getNonLanguageTokenIndexes(logitsDim: self.logitsDim, allLanguageTokens: self.allLanguageTokens)
    }

    /// Retain the logits that correspond to language tokens and suppress non-language tokens
    public func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray {
        guard tokens.count >= sampleBegin else {
            return logits
        }
        logits.fill(indexes: nonLanguageTokenIndexes, with: -FloatType.infinity)
        return logits
    }

    private static func getNonLanguageTokenIndexes(logitsDim: Int, allLanguageTokens: Set<Int>) -> [[Int]] {
        var indexes: [[Int]] = []
        for i in 0..<logitsDim {
            if !allLanguageTokens.contains(i) {
                indexes.append([0, 0, i])
            }
        }
        return indexes
    }
}
