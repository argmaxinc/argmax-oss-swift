#!/bin/bash

#  Copyright © 2025 Argmax, Inc. All rights reserved.
#  For licensing see accompanying LICENSE.md file.

# WhisperKit CurlClient - Test Script
# This script demonstrates various features of the CurlClient

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test audio files (adjust paths as needed)
TEST_FILES=(
    "../../../Tests/WhisperKitTests/Resources/jfk.wav"
    "../../../Tests/WhisperKitTests/Resources/es_test_clip.wav"
    "../../../Tests/WhisperKitTests/Resources/ja_test_clip.wav"
)

# Server URL
SERVER_URL="http://localhost:50060"

echo -e "${BLUE}🧪 WhisperKit CurlClient Test Suite${NC}"
echo -e "${YELLOW}Testing against server:${NC} $SERVER_URL"
echo ""

# Check if server is running
echo -e "${BLUE}🔍 Checking server status...${NC}"
if curl -s "$SERVER_URL" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Server is running${NC}"
else
    echo -e "${RED}❌ Server is not running at $SERVER_URL${NC}"
    echo -e "${YELLOW}Please start the server first:${NC}"
    echo "  whisperkit-cli serve --model tiny"
    exit 1
fi

echo ""

test_logprobs() {
    echo "🧪 Testing transcription with logprobs..."
    
    # Find test audio files - use absolute path
    local test_file=""
    if [ -f "../../../Tests/WhisperKitTests/Resources/jfk.wav" ]; then
        test_file="$(cd ../../../Tests/WhisperKitTests/Resources && pwd)/jfk.wav"
    elif [ -f "../../../Tests/WhisperKitTests/Resources/es_test_clip.wav" ]; then
        test_file="$(cd ../../../Tests/WhisperKitTests/Resources && pwd)/es_test_clip.wav"
    elif [ -f "../../../Tests/WhisperKitTests/Resources/ja_test_clip.wav" ]; then
        test_file="$(cd ../../../Tests/WhisperKitTests/Resources && pwd)/ja_test_clip.wav"
    fi
    
    if [ -z "$test_file" ]; then
        echo "❌ No test audio files found"
        return 1
    fi
    
    echo "📁 Using test file: $(basename "$test_file")"
    echo "🔍 Full path: $test_file"
    
    # Test with logprobs enabled
    echo "🔍 Testing with file: $test_file"
    echo "🔍 Server URL: $SERVER_URL"
    
    local response=$(curl -s -X POST "$SERVER_URL/v1/audio/transcriptions" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@$test_file" \
        -F "model=tiny" \
        -F "response_format=json" \
        -F "include[]=logprobs")
    
    # Debug: Show response length and first part
    echo "🔍 Response length: ${#response}"
    echo "🔍 Response preview: ${response:0:200}..."
    
    if echo "$response" | grep -q "logprobs"; then
        echo "✅ Logprobs received in response"
        
        # Extract and display logprobs info
        local logprobs_count=$(echo "$response" | jq -r '.logprobs | length' 2>/dev/null || echo "0")
        echo "📊 Found $logprobs_count logprob entries"
        
        # Show first few logprobs
        if [ "$logprobs_count" -gt 0 ]; then
            echo "🔍 First few logprobs:"
            echo "$response" | jq -r '.logprobs[0:3][] | "  Token: \(.token) - Logprob: \(.logprob)"' 2>/dev/null || echo "  Could not parse logprobs"
        fi
        
        return 0
    else
        echo "❌ No logprobs in response"
        echo "Available keys: $(echo "$response" | jq -r 'keys | join(", ")' 2>/dev/null || echo "Could not parse response")"
        return 1
    fi
}

# Test 1: Basic transcription
echo -e "${BLUE}📝 Test 1: Basic Transcription (verbose_json)${NC}"
echo -e "${YELLOW}File:${NC} ${TEST_FILES[0]}"
echo ""
./transcribe.sh "${TEST_FILES[0]}" --response-format verbose_json
echo ""

# Test 2: Basic transcription with JSON format
echo -e "${BLUE}📝 Test 2: Basic Transcription (json)${NC}"
echo -e "${YELLOW}File:${NC} ${TEST_FILES[0]}"
echo ""
./transcribe.sh "${TEST_FILES[0]}" --response-format json
echo ""

# Test 3: Transcription with word timestamps
echo -e "${BLUE}📝 Test 3: Transcription with Word Timestamps${NC}"
echo -e "${YELLOW}File:${NC} ${TEST_FILES[0]}"
echo ""
./transcribe.sh "${TEST_FILES[0]}" --timestamp-granularities "word,segment"
echo ""

# Test 4: Spanish transcription
echo -e "${BLUE}📝 Test 4: Spanish Transcription${NC}"
echo -e "${YELLOW}File:${NC} ${TEST_FILES[1]}"
echo -e "${YELLOW}Language:${NC} es"
echo ""
./transcribe.sh "${TEST_FILES[1]}" --language es
echo ""

# Test 5: Japanese transcription
echo -e "${BLUE}📝 Test 5: Japanese Transcription${NC}"
echo -e "${YELLOW}File:${NC} ${TEST_FILES[2]}"
echo -e "${YELLOW}Language:${NC} ja"
echo ""
./transcribe.sh "${TEST_FILES[2]}" --language ja
echo ""

# Test 6: Translation (Spanish to English)
echo -e "${BLUE}🌐 Test 6: Translation (Spanish to English)${NC}"
echo -e "${YELLOW}File:${NC} ${TEST_FILES[1]}"
echo -e "${YELLOW}Source Language:${NC} es"
echo ""
./translate.sh "${TEST_FILES[1]}" --language es
echo ""

# Test 7: Translation (Japanese to English)
echo -e "${BLUE}🌐 Test 7: Translation (Japanese to English)${NC}"
echo -e "${YELLOW}File:${NC} ${TEST_FILES[2]}"
echo -e "${YELLOW}Source Language:${NC} ja"
echo ""
./translate.sh "${TEST_FILES[2]}" --language ja
echo ""

# Test 7.5: Translation with basic JSON format
echo -e "${BLUE}🌐 Test 7.5: Translation with JSON Format${NC}"
echo -e "${YELLOW}File:${NC} ${TEST_FILES[1]}"
echo -e "${YELLOW}Source Language:${NC} es"
echo ""
./translate.sh "${TEST_FILES[1]}" --language es --response-format json
echo ""

# Test 8: Streaming transcription
echo -e "${BLUE}📡 Test 8: Streaming Transcription${NC}"
echo -e "${YELLOW}File:${NC} ${TEST_FILES[0]}"
echo ""
./transcribe.sh "${TEST_FILES[0]}" --stream true
echo ""

# Test 8.5: Translation with prompt
echo -e "${BLUE}📝 Test 8.5: Translation with Prompt${NC}"
echo -e "${YELLOW}File:${NC} ${TEST_FILES[1]}"
echo -e "${YELLOW}Source Language:${NC} es"
echo ""
./translate.sh "${TEST_FILES[1]}" --language es --prompt "This is a formal conversation"
echo ""

# Test 9: Logprobs functionality
echo -e "${BLUE}🧪 Test 9: Logprobs Functionality${NC}"
if test_logprobs; then
    echo -e "${GREEN}✅ Logprobs test passed${NC}"
else
    echo -e "${RED}❌ Logprobs test failed${NC}"
fi
echo ""

# Test 10: Translation with different temperature
echo -e "${BLUE}🌡️ Test 10: Translation with Temperature${NC}"
echo -e "${YELLOW}File:${NC} ${TEST_FILES[1]}"
echo -e "${YELLOW}Source Language:${NC} es"
echo ""
./translate.sh "${TEST_FILES[1]}" --language es --temperature 0.2
echo ""

echo -e "${GREEN}🎉 All tests completed!${NC}"
echo ""
echo -e "${BLUE}📚 For more examples, see:${NC}"
echo "  ./transcribe.sh --help"
echo "  ./translate.sh --help"
