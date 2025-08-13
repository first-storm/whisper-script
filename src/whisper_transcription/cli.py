# -*- coding: utf-8 -*-
"""
This module contains the command-line interface (CLI) logic,
including argument parsing and dispatching to other modules.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Optional

from openai import OpenAI
from groq import Groq

from .config import load_config, select_api_profile
from .transcription import transcribe_audio, TranscriptionProvider
from .services import download_youtube_audio
from .processing import ensure_tool_exists
from .providers import OpenAIProvider, GroqProvider


def process_item(item: str, provider_service: TranscriptionProvider, config, temp_dir: Path, output_to_console=False):
    """Process a single file path or URL."""
    audio_path = None
    is_temp_file = False
    output_path = None

    if item.startswith("http") and ("youtube.com" in item or "youtu.be" in item):
        audio_path = download_youtube_audio(item, temp_dir)
        is_temp_file = True
        # Always set output_path for YouTube downloads if not outputting to console
        if audio_path and not output_to_console:
            # For YouTube downloads, save to current working directory
            output_path = Path.cwd() / f"{audio_path.stem}.txt"
    elif Path(item).is_file():
        audio_path = Path(item)
        if not output_to_console:
            # For local files, save to current working directory
            output_path = Path.cwd() / f"{audio_path.stem}.txt"
    else:
        print(f"\nSkipping invalid input: '{item}' (not a valid file or YouTube link)")
        return

    if audio_path:
        # Always check file size and process through transcribe_audio (handles chunking if needed)
        transcription_text = transcribe_audio(
            audio_path,
            provider_service,
            config,
            output_to_file=not output_to_console,
            output_path=output_path
        )
        if is_temp_file and audio_path:
            try:
                os.remove(audio_path)
                print(f"Temporary file deleted: {audio_path.name}")
            except OSError as e:
                print(f"Failed to delete temporary file: {e}")


def interactive_mode(provider_service: TranscriptionProvider, config, temp_dir: Path, output_to_console=False):
    """Run the script in an interactive command-line mode."""
    print("\n--- Interactive Transcription Mode ---")
    print("Enter a file path or YouTube URL. Type 'quit' or 'exit' to finish.")
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
        if not user_input:
            continue
        process_item(user_input, provider_service, config, temp_dir, output_to_console)
    print("\nExiting interactive mode.")


def main():
    """Main execution function for the CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Speech-to-text for local files and YouTube links, with auto-chunking on silence for large files.\n"
            "Now supports multiple API profiles via config."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "inputs",
        nargs='*',
        help=(
            "One or more input sources.\n"
            "Can be:\n  - Local audio file path (e.g., audio.mp3)\n  - YouTube video URL (e.g., 'https://www.youtube.com/watch?v=...')"
        )
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode to process files one by one."
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Output transcription to console instead of saving to text files."
    )
    # NEW: allow choosing a profile from config
    parser.add_argument(
        "--profile",
        help=(
            "Choose an API profile defined in config.yaml (overrides default_profile / WHISPER_PROFILE)."
        )
    )

    args = parser.parse_args()

    config = load_config()

    # NEW: resolve and select API profile (writes back to config['api'])
    try:
        select_api_profile(config, cli_profile=args.profile)
    except SystemExit as e:
        print(e)
        sys.exit(1)

    # Optional: log which profile is used
    sel = config.get('selected_profile_name', '?')
    api_cfg = config['api']
    provider_name = api_cfg.get('provider', 'openai').lower() # Default to 'openai' if not specified
    print(
        f"Using API profile: {sel} | provider={provider_name} | model={api_cfg.get('model','?')}"
    )

    # Instantiate the correct client based on the provider
    client = None
    provider_service = None
    if provider_name == 'openai':
        base_url = api_cfg.get('base_url')
        api_key = api_cfg.get('api_key')
        if not api_key:
            print("Error: Missing 'api_key' for the selected profile. "
                  "Set it in config.yaml or via environment variable.")
            sys.exit(1)
        client = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI(api_key=api_key)
        provider_service = OpenAIProvider(client, config)
    elif provider_name == 'groq':
        api_key = api_cfg.get('api_key')
        if not api_key:
            print("Error: Missing 'api_key' for the selected profile. "
                  "Set it in config.yaml or via environment variable.")
            sys.exit(1)
        client = Groq(api_key=api_key)
        provider_service = GroqProvider(client, config)
    else:
        print(f"Error: Unknown provider '{provider_name}' specified in profile '{sel}'.")
        sys.exit(1)

    # Prepare temp dir
    temp_dir = Path("./whisper_temp_audio")
    temp_dir.mkdir(exist_ok=True)

    # Provide a hint if FFmpeg/FFprobe is missing (only needed for chunking)
    if not ensure_tool_exists(config['processing']['ffmpeg_path']):
        print("Note: FFmpeg not found. Large files over the size limit cannot be auto-chunked.")
    if not ensure_tool_exists(config['processing']['ffprobe_path']):
        print("Note: FFprobe not found. Large files over the size limit cannot be analyzed for chunking.")

    if args.interactive:
        interactive_mode(provider_service, config, temp_dir, args.console)
    elif args.inputs:
        print(f"Found {len(args.inputs)} item(s) to process.")
        for item in args.inputs:
            process_item(item, provider_service, config, temp_dir, args.console)
    else:
        parser.print_help()
        print("\nError: No inputs provided. Please provide file paths/URLs or use interactive mode (-i).")
        sys.exit(1)

    # Clean up empty temp directory (best-effort)
    try:
        # remove empty chunk subdirs
        chunks_root = temp_dir / "chunks"
        if chunks_root.exists():
            for p in chunks_root.glob("*"):
                try:
                    if p.is_dir() and not any(p.iterdir()):
                        p.rmdir()
                except Exception:
                    pass
        if not any(temp_dir.iterdir()):
            temp_dir.rmdir()
        else:
            # If temp_dir is not empty, remove it recursively
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    except OSError:
        pass  # Don't delete if not empty

    print("\nAll tasks completed.")