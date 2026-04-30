# WhisperKit Swift Client

A simple Swift client for the WhisperKit local server.

## Quick Start

1. **Start the WhisperKit server** (in another terminal):
   ```bash
   whisperkit-cli serve
   ```

2. **Build the client**:
   ```bash
   swift build
   ```

3. **Run commands**:
   ```bash
   # Transcribe an audio file
   swift run whisperkit-client transcribe audio.wav
   
   # Translate an audio file to English
   swift run whisperkit-client translate audio.wav
   
   # Test with sample files
   swift run whisperkit-client test
   ```

## Available Commands

- `transcribe <audio-file>` - Transcribe audio to text
- `translate <audio-file>` - Translate audio to English
- `test` - Test transcription and translation on sample files

## Options

- `--language, -l` - Source language for transcription (default: auto-detect)
- `--model, -m` - Model to use (default: tiny)
- `--response-format` - Response format: json, verbose_json (default: verbose_json)
- `--timestamp-granularities` - Comma-separated: word,segment (default: segment)
- `--stream` - Enable streaming output
- `--server-url, -s` - Server URL (default: http://localhost:50060/v1)

## Examples

```bash
# Transcribe in Spanish
swift run whisperkit-client transcribe -l es audio.wav

# Transcribe with word-level timestamps
swift run whisperkit-client transcribe --timestamp-granularities "word,segment" audio.wav

# Translate from Spanish to English
swift run whisperkit-client translate -l es audio.wav

# Use custom server
swift run whisperkit-client transcribe -s http://192.168.1.100:50060 audio.wav

# Stream transcription
swift run whisperkit-client transcribe --stream audio.wav
```

## Project Structure

```
Sources/
├── CLI.swift           # All CLI commands and client logic
└── Generated/          # Auto-generated OpenAPI client code
    ├── Client.swift
    └── Types.swift
```

## Current Limitations

- **Response Format**: The `--response-format` parameter is not fully working due to OpenAPI schema discrimination issues. The client always receives basic JSON responses instead of verbose JSON with segments and word timestamps.
- **Word Timestamps**: Word-level timestamps are not displayed due to the response format issue above.
- **Basic Functionality**: Basic transcription and translation work correctly.

> **Note**: This is a known issue with the Swift OpenAPI generator's handling of `oneOf` schemas with discriminators. The server correctly sends verbose JSON responses, but the Swift client cannot properly parse them. Consider using the Python client or CurlClient for full functionality.

## Updating Generated Code

When the server spec changes, regenerate the client code:

```bash
./updateClient.sh
```

This will update the files in `Sources/Generated/` from `scripts/specs/localserver_openapi.yaml`.
