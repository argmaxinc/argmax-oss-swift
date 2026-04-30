#!/bin/bash

#  Copyright © 2025 Argmax, Inc. All rights reserved.
#  For licensing see accompanying LICENSE.md file.

# WhisperKit CurlClient - Translate Audio
# Usage: ./translate.sh <audio-file> [options]

set -e

# Default values
SERVER_URL="http://localhost:50060/v1"
MODEL="tiny"
LANGUAGE=""
PROMPT=""
RESPONSE_FORMAT="verbose_json"
TEMPERATURE="0.0"
VERBOSE="false"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help function
show_help() {
    echo "Usage: $0 <audio-file> [options]"
    echo ""
    echo "Arguments:"
    echo "  audio-file                    Path to audio file (wav, mp3, m4a, flac, etc.)"
    echo ""
    echo "Options:"
    echo "  -h, --help                    Show this help message"
    echo "  -s, --server <url>            Server URL (default: http://localhost:50060/v1)"
    echo "  -m, --model <model>           Model to use (default: tiny)"
    echo "  -l, --language <lang>         Source language code (e.g., es, ja, fr)"
    echo "  -p, --prompt <text>           Text to guide translation (should be in English)"
    echo "  -f, --response-format <format>         Response format: json, verbose_json (default: verbose_json)"
    echo "  -t, --temperature <value>     Sampling temperature 0.0-1.0 (default: 0.0)"
    echo "  --verbose                     Show verbose curl output"
    echo ""
    echo "Examples:"
    echo "  $0 audio.wav"
    echo "  $0 audio.wav --language es --response-format json"
    echo "  $0 audio.wav --language ja --prompt \"This is a formal conversation\""
    echo ""
}

# Parse command line arguments
AUDIO_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--server)
            SERVER_URL="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -l|--language)
            LANGUAGE="$2"
            shift 2
            ;;
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -f|--response-format)
            RESPONSE_FORMAT="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;

        --verbose)
            VERBOSE="true"
            shift
            ;;
        -*)
            echo -e "${RED}Error: Unknown option $1${NC}"
            show_help
            exit 1
            ;;
        *)
            if [[ -z "$AUDIO_FILE" ]]; then
                AUDIO_FILE="$1"
            else
                echo -e "${RED}Error: Multiple audio files specified${NC}"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if audio file is provided
if [[ -z "$AUDIO_FILE" ]]; then
    echo -e "${RED}Error: Audio file is required${NC}"
    show_help
    exit 1
fi

# Check if audio file exists
if [[ ! -f "$AUDIO_FILE" ]]; then
    echo -e "${RED}Error: Audio file '$AUDIO_FILE' not found${NC}"
    exit 1
fi

# Build curl command
CURL_CMD="curl -X POST \"$SERVER_URL/audio/translations\""
CURL_CMD="$CURL_CMD -H \"Content-Type: multipart/form-data\""
CURL_CMD="$CURL_CMD -F \"file=@$AUDIO_FILE\""
CURL_CMD="$CURL_CMD -F \"model=$MODEL\""

if [[ -n "$LANGUAGE" ]]; then
    CURL_CMD="$CURL_CMD -F \"language=$LANGUAGE\""
fi

if [[ -n "$PROMPT" ]]; then
    CURL_CMD="$CURL_CMD -F \"prompt=$PROMPT\""
fi

CURL_CMD="$CURL_CMD -F \"response_format=$RESPONSE_FORMAT\""
CURL_CMD="$CURL_CMD -F \"temperature=$TEMPERATURE\""

# Add output flags based on verbose setting
if [[ "$VERBOSE" == "true" ]]; then
    CURL_CMD="$CURL_CMD -v"
else
    CURL_CMD="$CURL_CMD -s -S"
fi

echo -e "${BLUE}🚀 Starting translation...${NC}"
echo -e "${YELLOW}📁 Audio file:${NC} $AUDIO_FILE"
echo -e "${YELLOW}🌐 Server:${NC} $SERVER_URL"
echo -e "${YELLOW}🤖 Model:${NC} $MODEL"
echo -e "${YELLOW}📝 Response format:${NC} $RESPONSE_FORMAT"
echo -e "${YELLOW}🌡️ Temperature:${NC} $TEMPERATURE"
if [[ -n "$LANGUAGE" ]]; then
    echo -e "${YELLOW}🌍 Source language:${NC} $LANGUAGE"
fi
if [[ -n "$PROMPT" ]]; then
    echo -e "${YELLOW}💡 Prompt:${NC} $PROMPT"
fi
echo ""

# Execute curl command
echo -e "${BLUE}📡 Sending request...${NC}"
echo ""
eval $CURL_CMD

echo ""
echo -e "${GREEN}✅ Translation complete!${NC}"
