#!/bin/bash

#  Copyright © 2025 Argmax, Inc. All rights reserved.
#  For licensing see accompanying LICENSE.md file.

# WhisperKit Local Server Transcription Client
# Usage: ./transcribe.sh <audio_file> [options]

set -e

# Default values
AUDIO_FILE=""
MODEL="tiny"
LANGUAGE=""
PROMPT=""
RESPONSE_FORMAT="verbose_json"
TIMESTAMP_GRANULARITIES="segment"
TEMPERATURE="0.0"
STREAM="false"
VERBOSE="false"
LOGPROBS="false"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --response-format)
            RESPONSE_FORMAT="$2"
            shift 2
            ;;
        --timestamp-granularities)
            TIMESTAMP_GRANULARITIES="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --stream)
            STREAM="$2"
            shift 2
            ;;
        --logprobs)
            LOGPROBS="true"
            shift
            ;;
        --verbose)
            VERBOSE="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 <audio_file> [options]"
            echo ""
            echo "Options:"
            echo "  --model <model>                    Model to use (default: tiny)"
            echo "  --language <language>              Language code (e.g., en, es, fr)"
            echo "  --prompt <text>                    Prompt text for transcription"
            echo "  --response-format <format>         Response format: json, verbose_json (default: verbose_json)"
            echo "  --timestamp-granularities <list>   Comma-separated list: word,segment (default: segment)"
            echo "  --temperature <float>               Temperature for sampling (default: 0.0)"
            echo "  --stream <true|false>              Enable streaming (default: false)"
            echo "  --logprobs                        Include logprobs in transcription (default: false)"
            echo "  --verbose                          Show verbose output"
            echo "  -h, --help                         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 audio.wav"
            echo "  $0 audio.wav --model base --language en"
            echo "  $0 audio.wav --timestamp-granularities word,segment --stream true"
            exit 0
            ;;
        *)
            if [[ -z "$AUDIO_FILE" ]]; then
                AUDIO_FILE="$1"
            else
                echo "Error: Unknown option $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if audio file is provided
if [[ -z "$AUDIO_FILE" ]]; then
    echo "Error: Audio file is required"
    echo "Usage: $0 <audio_file> [options]"
    exit 1
fi

# Check if audio file exists
if [[ ! -f "$AUDIO_FILE" ]]; then
    echo "Error: Audio file '$AUDIO_FILE' not found"
    exit 1
fi

# Build curl command
CURL_CMD="curl -s -S"

# Add verbose flag if requested
if [[ "$VERBOSE" == "true" ]]; then
    CURL_CMD="$CURL_CMD -v"
fi

CURL_CMD="$CURL_CMD -X POST http://localhost:50060/v1/audio/transcriptions"

# Build multipart form data
CURL_CMD="$CURL_CMD -F file=@\"$AUDIO_FILE\""
CURL_CMD="$CURL_CMD -F model=\"$MODEL\""
CURL_CMD="$CURL_CMD -F response_format=\"$RESPONSE_FORMAT\""
CURL_CMD="$CURL_CMD -F timestamp_granularities[]=\"$TIMESTAMP_GRANULARITIES\""
CURL_CMD="$CURL_CMD -F temperature=\"$TEMPERATURE\""
CURL_CMD="$CURL_CMD -F stream=\"$STREAM\""
# Add logprobs if specified
if [ "$LOGPROBS" = "true" ]; then
    CURL_CMD="$CURL_CMD -F \"include[]=logprobs\""
fi

if [[ -n "$LANGUAGE" ]]; then
    CURL_CMD="$CURL_CMD -F language=\"$LANGUAGE\""
fi

if [[ -n "$PROMPT" ]]; then
    CURL_CMD="$CURL_CMD -F prompt=\"$PROMPT\""
fi

echo "🔄 Transcribing: $AUDIO_FILE"
echo "📋 Options: model=$MODEL, format=$RESPONSE_FORMAT, granularities=$TIMESTAMP_GRANULARITIES, stream=$STREAM"
echo ""

# Execute curl command
if [[ "$STREAM" == "true" ]]; then
    # For streaming, process line by line with timestamps
    echo "📡 Starting streaming transcription..."
    echo "⏰ Timestamps show when each piece of data arrives:"
    echo ""
    
    # Use a function to add timestamps to each line
    timestamp_stream() {
        while IFS= read -r line; do
            if [[ -n "$line" ]]; then
                timestamp=$(date '+%H:%M:%S.%3N')
                echo "[$timestamp] $line"
            fi
        done
    }
    
    eval "$CURL_CMD" | timestamp_stream
else
    # For non-streaming, just execute normally
    eval "$CURL_CMD"
fi

echo ""
echo "✅ Transcription complete"
