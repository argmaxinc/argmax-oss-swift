//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import AVFoundation
import CoreML

extension Array where Element == Result<[TranscriptionResult], Swift.Error> {
    /// Convenience method to convert the `Result` object into an array of optional arrays of `TranscriptionResult`.
    /// - Returns: An array of optional arrays containing `TranscriptionResult`.
    func toOptionalArrays() -> [[TranscriptionResult]?] {
        return self.map { try? $0.get() }
    }
}

extension AVAudioPCMBuffer {
    /// Converts the buffer to a float array
    func asFloatArray() throws -> [Float] {
        guard let data = floatChannelData?.pointee else {
            throw WhisperError.audioProcessingFailed("Error converting audio, missing floatChannelData")
        }
        return Array(UnsafeBufferPointer(start: data, count: Int(frameLength)))
    }

    /// Appends the contents of another buffer to the current buffer
    func appendContents(of buffer: AVAudioPCMBuffer) -> Bool {
        return appendContents(of: buffer, startingFrame: 0, frameCount: buffer.frameLength)
    }

    /// Appends a specific range of frames from another buffer to the current buffer
    func appendContents(of buffer: AVAudioPCMBuffer, startingFrame: AVAudioFramePosition, frameCount: AVAudioFrameCount) -> Bool {
        guard format == buffer.format else {
            Logging.debug("Format mismatch")
            return false
        }

        guard startingFrame + AVAudioFramePosition(frameCount) <= AVAudioFramePosition(buffer.frameLength) else {
            Logging.error("Insufficient audio in buffer")
            return false
        }

        guard let destination = floatChannelData, let source = buffer.floatChannelData else {
            Logging.error("Failed to access float channel data")
            return false
        }

        var calculatedFrameCount = frameCount
        if frameLength + frameCount > frameCapacity {
            Logging.debug("Insufficient space in buffer, reducing frame count to fit")
            calculatedFrameCount = frameCapacity - frameLength
        }

        let calculatedStride = stride
        let destinationPointer = destination.pointee.advanced(by: calculatedStride * Int(frameLength))
        let sourcePointer = source.pointee.advanced(by: calculatedStride * Int(startingFrame))

        memcpy(destinationPointer, sourcePointer, Int(calculatedFrameCount) * calculatedStride * MemoryLayout<Float>.size)

        frameLength += calculatedFrameCount
        return true
    }

    /// Convenience initializer to concatenate multiple buffers into one
    convenience init?(concatenating buffers: [AVAudioPCMBuffer]) {
        guard !buffers.isEmpty else {
            Logging.debug("Buffers array should not be empty")
            return nil
        }

        let totalFrames = buffers.reduce(0) { $0 + $1.frameLength }

        guard let firstBuffer = buffers.first else {
            Logging.debug("Failed to get the first buffer")
            return nil
        }

        self.init(pcmFormat: firstBuffer.format, frameCapacity: totalFrames)

        for buffer in buffers {
            if !appendContents(of: buffer) {
                Logging.debug("Failed to append buffer")
                return nil
            }
        }
    }

    /// Computed property to determine the stride for float channel data
    private var stride: Int {
        return Int(format.streamDescription.pointee.mBytesPerFrame) / MemoryLayout<Float>.size
    }
}

// MARK: - WhisperKit Components

extension AudioProcessing {
    static func getDownloadsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)
        return paths[0]
    }

    static func saveBuffer(_ buffer: AVAudioPCMBuffer, to url: URL) throws {
        // create folder
        let folderURL = url.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: folderURL.path) {
            try FileManager.default.createDirectory(at: folderURL, withIntermediateDirectories: true, attributes: nil)
        }
        let audioFile = try AVAudioFile(forWriting: url, settings: buffer.format.settings)
        try audioFile.write(from: buffer)
    }
}

extension DecodingOptions {
    func prepareSeekClips(contentFrames: Int) -> [(start: Int, end: Int)] {
        var seekPoints: [Int] = clipTimestamps.map { Int(round($0 * Float(WhisperKit.sampleRate))) }
        if seekPoints.count == 0 {
            seekPoints.append(0)
        }

        if seekPoints.count % 2 == 1 {
            seekPoints.append(contentFrames)
        }

        var seekClips: [(start: Int, end: Int)] = []
        for i in stride(from: 0, to: seekPoints.count, by: 2) {
            let start = seekPoints[i]
            let end = i + 1 < seekPoints.count ? seekPoints[i + 1] : contentFrames
            seekClips.append((start, end))
        }

        return seekClips
    }
}
