#  Copyright © 2025 Argmax, Inc. All rights reserved.
#  For licensing see accompanying LICENSE.md file.

"""
WhisperKit Python Client CLI
A simple client for the WhisperKit local server
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import openai
import requests


class WhisperKitClient:
    def __init__(self, server_url: str = "http://localhost:50060"):
        self.client = openai.OpenAI(
            base_url=f"{server_url}/v1",
            api_key="dummy-key"  # Not used by local server
        )
        self.server_url = server_url

    def test_connection(self) -> bool:
        """Test connection to the server"""
        try:
            # Try a simple transcription with minimal data
            response = self.client.audio.transcriptions.create(
                model="tiny",
                file=("test.wav", b"test", "audio/wav"),
                language="en"
            )
            print(f"✅ Connection successful to {self.server_url}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def transcribe(self, audio_path: str, language: str = "", model: str = "tiny", stream: bool = False, 
                   response_format: str = "verbose_json", timestamp_granularities: List[str] = None, debug: bool = False, include: List[str] = None) -> Optional[str]:
        """Transcribe an audio file"""
        try:
            if stream:
                return self._transcribe_streaming(audio_path, language, model, response_format, timestamp_granularities)
            else:
                return self._transcribe_non_streaming(audio_path, language, model, response_format, timestamp_granularities, debug, include)
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return None

    def _transcribe_streaming(self, audio_path: str, language: str = "", model: str = "tiny", 
                             response_format: str = "verbose_json", timestamp_granularities: List[str] = None) -> Optional[str]:
        """Transcribe an audio file with streaming"""
        try:
            with open(audio_path, "rb") as audio_file:
                files = {"file": audio_file}
                data = {
                    "model": model,
                    "response_format": response_format,
                    "stream": "true"
                }
                if language:
                    data["language"] = language
                
                if timestamp_granularities:
                    data["timestamp_granularities[]"] = ",".join(timestamp_granularities)
                
                print("🔄 Starting streaming transcription...")
                
                response = requests.post(
                    f"{self.server_url}/v1/audio/transcriptions",
                    files=files,
                    data=data,
                    headers={"Accept": "text/event-stream"},
                    stream=True
                )
                
                if response.status_code != 200:
                    print(f"❌ Streaming failed: {response.status_code} {response.text}")
                    return None
                
                final_text = ""
                print("📡 Receiving stream data...")
                
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # Remove 'data: ' prefix
                            try:
                                event_data = json.loads(data_str)
                                
                                # Handle different event types based on OpenAI API format
                                if event_data.get('type') == 'transcript.text.delta':
                                    delta = event_data.get('delta', '')
                                    if delta:
                                        # Clean up WhisperKit's internal timestamps for better readability
                                        clean_delta = self._clean_whisperkit_text(delta)
                                        print(f"🔄 {clean_delta}")
                                        final_text += clean_delta
                                elif event_data.get('type') == 'transcript.text.done':
                                    # Final completion event
                                    final_text = event_data.get('text', final_text)
                                    print(f"\n✅ Final transcription: {final_text}")
                                    
                                    # Show additional info if available
                                    if 'language' in event_data:
                                        print(f"🌍 Language: {event_data.get('language')}")
                                    if 'duration' in event_data:
                                        print(f"⏱️  Duration: {event_data.get('duration')} seconds")
                                    
                                    # Show timestamp granularities info
                                    if timestamp_granularities:
                                        print(f"📊 Timestamp Granularities: {', '.join(timestamp_granularities)}")
                                        if "word" in timestamp_granularities:
                                            print("   Word-level timestamps enabled")
                                        if "segment" in timestamp_granularities:
                                            print("   Segment-level timestamps enabled")
                                else:
                                    print(f"📝 Event: {event_data}")
                            except json.JSONDecodeError as e:
                                print(f"⚠️  JSON decode error: {e} for data: {data_str}")
                                # Try to extract text directly if it's not JSON
                                if data_str.strip():
                                    final_text = data_str.strip()
                                    print(f"📝 Extracted text: {final_text}")
                
                if not final_text:
                    print("⚠️  No text received from stream, falling back to non-streaming")
                    # Fall back to non-streaming if streaming didn't work
                    return self._transcribe_non_streaming(audio_path, language, model, response_format, timestamp_granularities)
                
                return final_text
                
        except Exception as e:
            print(f"❌ Streaming transcription failed: {e}")
            return None

    def _transcribe_non_streaming(self, audio_path: str, language: str = "", model: str = "tiny", 
                                 response_format: str = "verbose_json", timestamp_granularities: List[str] = None, debug: bool = False, include: List[str] = None) -> Optional[str]:
        """Transcribe an audio file without streaming"""
        try:
            with open(audio_path, "rb") as audio_file:
                files = {"file": audio_file}
                data = {
                    "model": model,
                    "response_format": response_format
                }
                
                if language:
                    data["language"] = language
                
                if timestamp_granularities:
                    data["timestamp_granularities[]"] = ",".join(timestamp_granularities)
                
                if include:
                    for include_item in include:
                        data[f"include[]"] = include_item
                
                response = requests.post(
                    f"{self.server_url}/v1/audio/transcriptions",
                    files=files,
                    data=data
                )
                
                if response.status_code != 200:
                    print(f"❌ Transcription failed: {response.status_code} {response.text}")
                    return None
                
                result = response.json()
                
                # Debug: Print raw JSON response
                if debug:
                    import json
                    print(f"\n🔍 DEBUG: Raw JSON Response:")
                    print("=" * 60)
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                    print("=" * 60)
                
                # Display transcription text
                text = result.get("text", "")
                print(f"📝 Transcription: {text}")
                
                # Display detailed information for verbose_json format
                if response_format == "verbose_json":
                    print(f"\n📊 Detailed Information:")
                    print(f"   Language: {result.get('language', 'Unknown')}")
                    print(f"   Duration: {result.get('duration', 'Unknown')} seconds")
                    
                    # Show segments if available
                    segments = result.get('segments', [])
                    if segments:
                        print(f"   Segments: {len(segments)}")
                        for i, segment in enumerate(segments[:3]):  # Show first 3 segments
                            print(f"     Segment {i+1}: {segment.get('start', 0):.2f}s - {segment.get('end', 0):.2f}s")
                            print(f"       Text: {segment.get('text', '')}")
                            if 'avg_logprob' in segment:
                                print(f"       Confidence: {segment.get('avg_logprob', 0):.3f}")
                    
                    # Show words if word timestamps are requested
                    if timestamp_granularities and "word" in timestamp_granularities:
                        words = result.get('words', [])
                        if words and len(words) > 0:
                            print(f"   Words: {len(words)}")
                            print("     All words with timestamps:")
                            for i, word in enumerate(words):
                                start = word.get('start', 0)
                                end = word.get('end', 0)
                                word_text = word.get('word', '')
                                print(f"       {start:.2f}s - {end:.2f}s: '{word_text}'")
                        else:
                            print("   Words: No word-level timestamps available")
                            print("     Note: Server may not support word-level timestamps yet")
                            print("     Raw response keys: " + ", ".join(result.keys()))
                

                
                return text
                
        except Exception as e:
            print(f"❌ Non-streaming transcription failed: {e}")
            return None

    def _clean_whisperkit_text(self, text: str) -> str:
        """Clean up WhisperKit's internal timestamps and formatting for better readability"""
        import re
        # Remove WhisperKit's internal timestamps like <|0.00|>, <|4.00|>, etc.
        cleaned = re.sub(r'<\|[^>]+\|>', '', text)
        # Remove special tokens like <|startoftranscript|>, <|en|>, <|transcribe|>
        cleaned = re.sub(r'<\|[^>]+\|>', '', cleaned)
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def translate(self, audio_path: str, model: str = "tiny", 
                  response_format: str = "verbose_json", timestamp_granularities: List[str] = None) -> Optional[str]:
        """Translate an audio file to English"""
        try:
            with open(audio_path, "rb") as audio_file:
                files = {"file": audio_file}
                data = {
                    "model": model,
                    "response_format": response_format
                }
                
                # Note: Translation endpoint doesn't support timestamp_granularities
                # if timestamp_granularities:
                #     data["timestamp_granularities[]"] = timestamp_granularities
                
                response = requests.post(
                    f"{self.server_url}/v1/audio/translations",
                    files=files,
                    data=data
                )
                
                if response.status_code != 200:
                    print(f"❌ Translation failed: {response.status_code} {response.text}")
                    return None
                
                result = response.json()
                
                # Display translation text
                text = result.get("text", "")
                print(f"🌐 Translation: {text}")
                
                # Display detailed information for verbose_json format
                if response_format == "verbose_json":
                    print(f"\n📊 Translation Details:")
                    print(f"   Output Language: {result.get('language', 'Unknown')}")
                    print(f"   Duration: {result.get('duration', 'Unknown')} seconds")
                    
                    # Show segments if available
                    segments = result.get('segments', [])
                    if segments:
                        print(f"   Segments: {len(segments)}")
                        for i, segment in enumerate(segments[:3]):  # Show first 3 segments
                            print(f"     Segment {i+1}: {segment.get('start', 0):.2f}s - {segment.get('end', 0):.2f}s")
                            print(f"       Text: {segment.get('text', '')}")
                
                # Show format differences
                if response_format == "json":
                    print(f"\n📋 Response Format: {response_format}")
                    print("   Contains only the translated text")
                elif response_format == "verbose_json":
                    print(f"\n📋 Response Format: {response_format}")
                    print("   Contains detailed information including:")
                    print("   - Output language (should be English)")
                    print("   - Duration")
                    print("   - Segments with timing")
                
                return text
                
        except Exception as e:
            print(f"❌ Translation failed: {e}")
            return None

    def show_format_comparison(self, audio_path: str, language: str = "", model: str = "tiny") -> None:
        """Show the difference between json and verbose_json response formats"""
        print(f"\n🔍 Format Comparison for: {audio_path}")
        print("=" * 60)
        
        # Test with json format
        print("\n📋 JSON Format (minimal):")
        print("-" * 30)
        json_result = self._transcribe_non_streaming(audio_path, language, model, "json")
        
        # Test with verbose_json format
        print("\n📊 Verbose JSON Format (detailed):")
        print("-" * 30)
        verbose_result = self._transcribe_non_streaming(audio_path, language, model, "verbose_json")
        
        print("\n" + "=" * 60)
        print("💡 Key Differences:")
        print("   • JSON: Only contains the transcribed text")
        print("   • Verbose JSON: Contains text + language + duration + segments + confidence scores")
        print("   • Use JSON for simple text extraction")
        print("   • Use Verbose JSON for detailed analysis and timing information")



    def run_tests(self, model: str = "tiny", response_format: str = "verbose_json", 
                   timestamp_granularities: List[str] = None, stream: bool = False) -> None:
        """Run tests on sample audio files"""
        # Find test audio files from the main project's test resources
        project_root = Path(__file__).parent.parent.parent.parent
        resources_path = project_root / "Tests" / "WhisperKitTests" / "Resources"
        
        if not resources_path.exists():
            print(f"❌ Resources folder not found: {resources_path}")
            return
        
        # Find audio files
        audio_extensions = {".wav", ".m4a", ".mp3", ".flac", ".aac"}
        audio_files = [
            f for f in resources_path.iterdir()
            if f.suffix.lower() in audio_extensions
        ]
        
        if not audio_files:
            print("❌ No audio files found in resources")
            return
        
        print(f"🔍 Found {len(audio_files)} audio files")
        if stream:
            print("🔄 Testing with streaming enabled")
        
        for audio_file in sorted(audio_files):
            filename = audio_file.name
            print(f"\n--- Testing: {filename} ---")
            
            # Try transcription
            self.transcribe(str(audio_file), model=model, response_format=response_format, 
                           timestamp_granularities=timestamp_granularities, stream=stream)
            
            # Try translation for non-English files
            if "es_" in filename or "ja_" in filename:
                self.translate(str(audio_file), model=model, response_format=response_format, 
                              timestamp_granularities=timestamp_granularities)
        
        # Test logprobs functionality
        print(f"\n--- Testing Logprobs ---")
        self.test_logprobs()

    def test_logprobs(self):
        """Test transcription with logprobs enabled"""
        print("🧪 Testing transcription with logprobs...")
        
        # Find test audio files from the main project's test resources
        project_root = Path(__file__).parent.parent.parent.parent
        resources_path = project_root / "Tests" / "WhisperKitTests" / "Resources"
        
        if not resources_path.exists():
            print(f"❌ Resources folder not found: {resources_path}")
            return False
        
        test_files = list(resources_path.glob("*.wav"))
        if not test_files:
            print("❌ No test audio files found")
            return False
        
        test_file = test_files[0]
        print(f"📁 Using test file: {test_file.name}")
        
        # Test with logprobs enabled
        try:
            # Use internal method to get full response
            with open(str(test_file), "rb") as audio_file:
                files = {"file": audio_file}
                data = {
                    "model": "tiny",
                    "response_format": "json",
                    "include[]": "logprobs"
                }
                
                response = requests.post(
                    f"{self.server_url}/v1/audio/transcriptions",
                    files=files,
                    data=data
                )
                
                if response.status_code != 200:
                    print(f"❌ Logprobs test failed: {response.status_code} {response.text}")
                    return False
                
                result = response.json()
                
                if "logprobs" in result:
                    logprobs = result["logprobs"]
                    print(f"✅ Logprobs received: {len(logprobs)} tokens")
                    
                    # Display first few tokens with their logprobs
                    for i, token_info in enumerate(logprobs[:5]):
                        print(f"  Token {i+1}: '{token_info.get('token', 'N/A')}' - logprob: {token_info.get('logprob', 'N/A')}")
                    
                    if len(logprobs) > 5:
                        print(f"  ... and {len(logprobs) - 5} more tokens")
                    
                    return True
                else:
                    print("❌ No logprobs in response")
                    print(f"Available keys: {list(result.keys())}")
                    return False
                 
        except Exception as e:
            print(f"❌ Error testing logprobs: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="WhisperKit Python Client for local server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s transcribe audio.wav
  %(prog)s translate audio.wav
  %(prog)s test
  %(prog)s compare audio.wav
  %(prog)s --server http://192.168.1.100:50060 transcribe audio.wav
        """
    )
    
    parser.add_argument(
        "--server", "-s",
        default="http://localhost:50060",
        help="Server URL (default: http://localhost:50060)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="tiny",
        help="Model to use (default: tiny)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Transcribe command
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio file")
    transcribe_parser.add_argument("audio_file", help="Path to audio file")
    transcribe_parser.add_argument("--language", "-l", default="", help="Source language (default: auto-detect)")
    transcribe_parser.add_argument("--stream", action="store_true", help="Enable streaming output")
    transcribe_parser.add_argument("--response-format", "-f", default="verbose_json", 
                                  choices=["json", "verbose_json"], 
                                  help="Response format (default: verbose_json)")
    transcribe_parser.add_argument("--timestamp-granularities", "-t", 
                                  help="Timestamp granularities as comma-separated values (e.g., 'word,segment')")
    transcribe_parser.add_argument("--debug", action="store_true", help="Show raw JSON response for debugging")
    
    # Translate command
    translate_parser = subparsers.add_parser("translate", help="Translate audio file to English")
    translate_parser.add_argument("audio_file", help="Path to audio file")
    translate_parser.add_argument("--response-format", "-f", default="verbose_json", 
                                 choices=["json", "verbose_json"], 
                                 help="Response format (default: verbose_json)")
    translate_parser.add_argument("--timestamp-granularities", "-t", 
                                 help="Timestamp granularities as comma-separated values (e.g., 'word,segment')")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test transcription and translation on sample files")
    test_parser.add_argument("--model", "-m", default="tiny", help="Model to use (default: tiny)")
    test_parser.add_argument("--response-format", "-f", default="verbose_json", 
                            choices=["json", "verbose_json"], 
                            help="Response format (default: verbose_json)")
    test_parser.add_argument("--timestamp-granularities", "-t", 
                            help="Timestamp granularities as comma-separated values (e.g., 'word,segment')")
    test_parser.add_argument("--stream", action="store_true", help="Enable streaming output for tests")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare json vs verbose_json response formats")
    compare_parser.add_argument("audio_file", help="Path to audio file")
    compare_parser.add_argument("--language", "-l", default="", help="Source language (default: auto-detect)")
    compare_parser.add_argument("--model", "-m", default="tiny", help="Model to use (default: tiny)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create client
    client = WhisperKitClient(args.server)
    
    # Execute command
    if args.command == "transcribe":
        # Parse timestamp granularities from comma-separated string
        timestamp_granularities = None
        if args.timestamp_granularities:
            timestamp_granularities = [x.strip() for x in args.timestamp_granularities.split(',')]
        
        client.transcribe(args.audio_file, args.language, args.model, args.stream, 
                         args.response_format, timestamp_granularities, args.debug)
    elif args.command == "translate":
        # Parse timestamp granularities from comma-separated string
        timestamp_granularities = None
        if args.timestamp_granularities:
            timestamp_granularities = [x.strip() for x in args.timestamp_granularities.split(',')]
        
        client.translate(args.audio_file, args.model, args.response_format, timestamp_granularities)
    elif args.command == "test":
        # Parse timestamp granularities from comma-separated string
        timestamp_granularities = None
        if hasattr(args, 'timestamp_granularities') and args.timestamp_granularities:
            timestamp_granularities = [x.strip() for x in args.timestamp_granularities.split(',')]
        
        client.run_tests(args.model, args.response_format, timestamp_granularities, args.stream)
    elif args.command == "compare":
        client.show_format_comparison(args.audio_file, args.language, args.model)


if __name__ == "__main__":
    main()

