import Foundation
import Accelerate

// MARK: - BNNS Replacement Helpers
//
// BNNS.applyActivation and BNNS.ActivationFunction were deprecated in
// iOS 26 / macOS 26 in favor of BNNSGraph. These helpers provide
// numerically stable softmax / log-softmax using Accelerate directly.

enum ActivationHelper {
    /// Numerically stable log-softmax: log(exp(x_i) / sum(exp(x)))
    static func logSoftmax(_ input: [Float]) -> [Float] {
        guard !input.isEmpty else { return [] }
        let count = input.count

        // Find max for numerical stability
        var maxVal: Float = 0
        vDSP_maxv(input, 1, &maxVal, vDSP_Length(count))

        // Subtract max, exponentiate
        var shifted = [Float](repeating: 0, count: count)
        var negMax = -maxVal
        vDSP_vsadd(input, 1, &negMax, &shifted, 1, vDSP_Length(count))

        var expVals = [Float](repeating: 0, count: count)
        var count32 = Int32(count)
        vvexpf(&expVals, &shifted, &count32)

        // Sum of exp values
        var sum: Float = 0
        vDSP_sve(expVals, 1, &sum, vDSP_Length(count))

        // log-softmax = shifted - log(sum)
        let logSum = logf(sum)
        var negLogSum = -logSum
        var result = [Float](repeating: 0, count: count)
        vDSP_vsadd(&shifted, 1, &negLogSum, &result, 1, vDSP_Length(count))

        return result
    }

    /// Numerically stable softmax: exp(x_i) / sum(exp(x))
    static func softmax(_ input: [Float]) -> [Float] {
        guard !input.isEmpty else { return [] }
        let count = input.count

        var maxVal: Float = 0
        vDSP_maxv(input, 1, &maxVal, vDSP_Length(count))

        var shifted = [Float](repeating: 0, count: count)
        var negMax = -maxVal
        vDSP_vsadd(input, 1, &negMax, &shifted, 1, vDSP_Length(count))

        var expVals = [Float](repeating: 0, count: count)
        var count32 = Int32(count)
        vvexpf(&expVals, &shifted, &count32)

        var sum: Float = 0
        vDSP_sve(expVals, 1, &sum, vDSP_Length(count))

        var result = [Float](repeating: 0, count: count)
        var invSum = 1.0 / sum
        vDSP_vsmul(&expVals, 1, &invSum, &result, 1, vDSP_Length(count))

        return result
    }

    /// Scale input by alpha (replaces BNNS.ActivationFunction.linear(alpha:))
    static func linearScale(_ input: [Float], alpha: Float) -> [Float] {
        guard !input.isEmpty else { return [] }
        var result = [Float](repeating: 0, count: input.count)
        var a = alpha
        vDSP_vsmul(input, 1, &a, &result, 1, vDSP_Length(input.count))
        return result
    }

    /// Read Float values from a BNNSNDArrayDescriptor into an array.
    static func readArray(from descriptor: BNNSNDArrayDescriptor, count: Int) -> [Float] {
        guard let data = descriptor.data else { return [] }
        return data.withMemoryRebound(to: Float.self, capacity: count) { ptr in
            Array(UnsafeBufferPointer(start: ptr, count: count))
        }
    }

    /// Write Float values from an array into a BNNSNDArrayDescriptor.
    static func writeArray(_ array: [Float], to descriptor: BNNSNDArrayDescriptor) {
        guard let data = descriptor.data, !array.isEmpty else { return }
        data.withMemoryRebound(to: Float.self, capacity: array.count) { ptr in
            for i in 0..<array.count { ptr[i] = array[i] }
        }
    }
}
