//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgumentParser
import CoreML
import Foundation

enum ComputeUnits: String, ExpressibleByArgument, CaseIterable {
    case all, cpuAndGPU, cpuOnly, cpuAndNeuralEngine, random
    var asMLComputeUnits: MLComputeUnits {
        switch self {
            case .all: return .all
            case .cpuAndGPU: return .cpuAndGPU
            case .cpuOnly: return .cpuOnly
            case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
            case .random: return Bool.random() ? .cpuAndGPU : .cpuAndNeuralEngine
        }
    }
}
