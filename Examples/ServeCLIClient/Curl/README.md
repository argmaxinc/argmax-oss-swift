# WhisperKit CurlClient

A simple, lightweight client for the WhisperKit Local Server using shell scripts and curl.

## Quick Start

1. **Make scripts executable:**
   ```bash
   chmod +x *.sh
   ```

2. **Start the WhisperKit server:**
   ```bash
   whisperkit-cli serve --model tiny
   ```

3. **Use the scripts:**
   ```bash
   # Transcribe audio
   ./transcribe.sh audio.wav
   
   # Translate audio to English
   ./translate.sh audio.wav --language es
   
   # Run test suite
   ./test.sh
   ```

## Scripts

### `transcribe.sh`
Transcribes audio files to text.

**Basic usage:**
```bash
./transcribe.sh audio.wav
./transcribe.sh audio.wav --language en --timestamp-granularities word,segment
./transcribe.sh audio.wav --stream true --logprobs
```

### `translate.sh`
Translates audio files to English.

**Basic usage:**
```bash
./translate.sh audio.wav
./translate.sh audio.wav --language es
./translate.sh audio.wav --stream true --logprobs
```

### `test.sh`
Runs comprehensive tests on sample files.

## Options

- `-h, --help` - Show help
- `-s, --server <url>` - Server URL (default: http://localhost:50060/v1)
- `-l, --language <lang>` - Source language (e.g., en, es, ja)
- `-f, --response-format <format>` - Response format: json, verbose_json
- `--timestamp-granularities <types>` - Timestamp granularities: word,segment
- `--stream <true|false>` - Enable streaming (default: false)
- `--logprobs` - Include logprobs in response (default: false)
- `--temperature <value>` - Sampling temperature 0.0-1.0 (default: 0.0)
- `--verbose` - Show verbose curl output

## Prerequisites

- `curl` (usually pre-installed)
- `bash` shell
- WhisperKit Local Server running
