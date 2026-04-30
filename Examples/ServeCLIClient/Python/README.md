# WhisperKit Python Client

A simple Python client for the WhisperKit local server using OpenAI's SDK.

## Quick Start

1. **Start the WhisperKit server** (in another terminal):
   ```bash
   whisperkit-cli serve
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Run commands**:
   ```bash
   # Transcribe an audio file
   python whisperkit_client.py transcribe audio.wav
   
   # Translate an audio file to English
   python whisperkit_client.py translate audio.wav
   
   # Test with sample files
   python whisperkit_client.py test
   ```

## Available Commands

- `transcribe <audio-file>` - Transcribe audio to text
- `translate <audio-file>` - Translate audio to English
- `test` - Test transcription and translation on sample files

## Options

- `--server, -s` - Server URL (default: http://localhost:50060)
- `--model, -m` - Model to use (default: tiny)
- `--language, -l` - Source language for transcription (default: auto-detect)
- `--response-format` - Response format: json, verbose_json (default: verbose_json)
- `--timestamp-granularities` - Comma-separated: word,segment (default: segment)
- `--stream` - Enable streaming output
- `--debug` - Show raw JSON response for debugging


## Examples

```bash
# Transcribe in Spanish
python whisperkit_client.py transcribe -l es audio.wav

# Translate to English (auto-detects source language)
python whisperkit_client.py translate audio.wav

# Use custom server and model
python whisperkit_client.py -s http://192.168.1.100:50060 -m large transcribe audio.wav

# Transcribe with word-level timestamps
python whisperkit_client.py transcribe --timestamp-granularities "word,segment" audio.wav

# Stream transcription
python whisperkit_client.py transcribe --stream audio.wav

# Debug mode to see raw JSON
python whisperkit_client.py transcribe --debug audio.wav

# Test with sample files
python whisperkit_client.py test
```

## Project Structure

```
Examples/ServeCLIClient/Python/
├── whisperkit_client.py    # Main CLI script with all functionality
├── test_transcribe.py      # Test script for transcription
├── test_translate.py       # Test script for translation
├── requirements.txt         # Python dependencies
├── uv.lock                 # Locked dependency versions
└── README.md               # This file
```

## Dependencies

- `openai` - OpenAI Python SDK for API communication
- `uv` - Fast Python package manager

## Testing

The client automatically finds test audio files from `Tests/WhisperKitTests/Resources/` in the main project directory.

```bash
# Run tests on sample files
python whisperkit_client.py test

# Or run individual test scripts
python test_transcribe.py
python test_translate.py
```

## Alternative Clients

For lightweight testing without Python dependencies, see the [CurlClient](../Curl/README.md) which provides shell script implementations using curl.
