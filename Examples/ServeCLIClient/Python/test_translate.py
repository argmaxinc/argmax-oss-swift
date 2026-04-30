#  Copyright © 2025 Argmax, Inc. All rights reserved.
#  For licensing see accompanying LICENSE.md file.

"""
Test translation with audio files from Tests/WhisperKitTests/Resources/

This script tests translation functionality using the actual test audio files
from the WhisperKit test suite.
"""

import os
import sys
import argparse
from pathlib import Path
from openai import OpenAI


def get_test_audio_files():
    """
    Get list of available test audio files from Tests/WhisperKitTests/Resources/
    
    Returns:
        List of audio file paths
    """
    # Path to test resources relative to project root
    resources_dir = Path(__file__).parent.parent.parent.parent / "Tests" / "WhisperKitTests" / "Resources"
    
    if not resources_dir.exists():
        print(f"Error: Test resources directory not found: {resources_dir}")
        return []
    
    audio_extensions = {'.wav', '.m4a', '.mp3', '.flac', '.aac'}
    
    audio_files = []
    for file_path in resources_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            audio_files.append(file_path)
    
    return sorted(audio_files)


def translate_test_file(client, audio_file_path, prompt=None):
    """
    Translate a test audio file using the local WhisperKit server.
    
    Args:
        client: OpenAI client instance
        audio_file_path: Path to the audio file
        prompt: Optional prompt to guide translation
    
    Returns:
        Translation result or None if failed
    """
    try:
        print(f"Translating: {audio_file_path.name}")
        
        with open(audio_file_path, "rb") as audio_file:
            response = client.audio.translations.create(
                model="tiny",
                file=audio_file,
                prompt=prompt,
                response_format="verbose_json"
            )
            return response
    except Exception as e:
        print(f"Error translating {audio_file_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test translation with WhisperKit test audio files"
    )
    parser.add_argument(
        "--prompt", 
        help="Optional prompt to guide translation"
    )
    parser.add_argument(
        "--server-url", 
        default="http://localhost:50060/v1",
        help="WhisperKit server URL (default: http://localhost:50060/v1)"
    )
    parser.add_argument(
        "--file", 
        help="Specific test file to translate (e.g., 'es_test_clip.wav')"
    )
    parser.add_argument(
        "--target-language", 
        default="en",
        help="Target language for translation (default: 'en')"
    )
    
    args = parser.parse_args()
    
    # Get available test audio files
    test_files = get_test_audio_files()
    
    if not test_files:
        print("No test audio files found!")
        sys.exit(1)
    
    print("Available test audio files:")
    for i, file_path in enumerate(test_files, 1):
        print(f"  {i}. {file_path.name}")
    
    # Initialize OpenAI client with local server
    client = OpenAI(
        base_url=args.server_url,
        api_key="dummy-key"
    )
    
    print(f"\nConnecting to WhisperKit server at: {args.server_url}")
    print(f"Target language: {args.target_language}")
    
    if args.prompt:
        print(f"Prompt: {args.prompt}")
    
    if args.file:
        target_file = None
        for file_path in test_files:
            if file_path.name == args.file:
                target_file = file_path
                break
        
        if not target_file:
            print(f"Error: Test file '{args.file}' not found")
            print("Available files:", [f.name for f in test_files])
            sys.exit(1)
        
        files_to_process = [target_file]
    else:
        # Process all files
        files_to_process = test_files
    
    print(f"\nProcessing {len(files_to_process)} file(s)...")
    
    # Process each file
    for i, audio_file in enumerate(files_to_process, 1):
        print(f"\n{'='*50}")
        print(f"File {i}/{len(files_to_process)}: {audio_file.name}")
        print(f"{'='*50}")
        
        result = translate_test_file(
            client, 
            audio_file, 
            prompt=args.prompt
        )
        
        if result:
            print(f"\n✓ Translation successful!")
            print(f"Translated text: {result.text}")
            
            if hasattr(result, 'segments') and result.segments:
                print(f"\nSegments ({len(result.segments)}):")
                for segment in result.segments:
                    print(f"  [{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
            
            if hasattr(result, 'language') and result.language:
                print(f"\nSource language: {result.language}")
                
            # File size info
            file_size = audio_file.stat().st_size / 1024  # KB
            print(f"\nFile size: {file_size:.1f} KB")
            
        else:
            print(f"✗ Translation failed for {audio_file.name}")
    
    print(f"\n{'='*50}")
    print("Test translation complete!")
    print(f"Processed {len(files_to_process)} file(s)")
    print(f"All audio translated to: {args.target_language}")


if __name__ == "__main__":
    main()
