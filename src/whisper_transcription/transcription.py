# -*- coding: utf-8 -*-
"""
This module handles transcription logic, interacting with the Whisper API.
It includes functions for single file and chunked transcription.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ui import Spinner
from .processing import split_on_silence_to_chunks, human_readable_size, ensure_tool_exists
from .providers.openai_provider import OpenAIProvider
from .providers.groq_provider import GroqProvider

# Define a type hint for the provider service
TranscriptionProvider = Any # Union[OpenAIProvider, GroqProvider] etc.


def save_transcription_to_file(transcription_text, output_path: Path):
    """Save transcription text to a file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription_text.strip())
        print(f"Transcription saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving transcription to file: {e}")
        return False


def transcribe_single_file(file_path: Path, provider_service: TranscriptionProvider) -> str:
    """Transcribe a single small file directly via the selected provider API."""
    return provider_service.transcribe_audio_file(file_path)


def transcribe_chunk(file_path: Path, provider_service: TranscriptionProvider, index: int) -> Tuple[int, str]:
    """Transcribe a chunk and return (index, text)."""
    text = transcribe_single_file(file_path, provider_service)
    return index, text


def transcribe_large_file(file_path: Path, provider_service: TranscriptionProvider, config: Dict[str, Any], temp_dir: Path) -> str:
    """
    Split the file on silence into chunks <= max_file_mb and transcribe chunks in parallel.
    Concatenate results in chronological order.
    """
    spinner = Spinner('Preparing chunks (FFmpeg silencedetect)...', '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')
    spinner.start()
    try:
        chunks = split_on_silence_to_chunks(file_path, temp_dir, config)
        spinner.succeed(f"Created {len(chunks)} chunk(s).")
    except Exception as e:
        spinner.fail(f"Chunking failed: {e}")
        raise

    workers = int(config['processing']['workers'] or 4)
    print(f"Transcribing {len(chunks)} chunk(s) with {workers} worker(s)...")

    # Spinner for chunk transcription progress
    transcription_spinner = Spinner(f'Transcribing chunks (0/{len(chunks)} completed)...', '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')
    transcription_spinner.start()

    results: Dict[int, str] = {}
    completed_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(transcribe_chunk, chunk, provider_service, idx): (idx, chunk)
            for idx, chunk in enumerate(chunks, start=1)
        }
        for fut in as_completed(futures):
            idx, chunk_path = futures[fut]
            try:
                i, text = fut.result()
                results[i] = text
                completed_count += 1
                transcription_spinner.message = f'Transcribing chunks ({completed_count}/{len(chunks)} completed)...'
            except Exception as e:
                print(f"\nError transcribing chunk {idx}: {e}")
                results[idx] = ""
                completed_count += 1
                transcription_spinner.message = f'Transcribing chunks ({completed_count}/{len(chunks)} completed)...'

    transcription_spinner.succeed(f"All {len(chunks)} chunk(s) transcribed successfully!")

    # Clean up chunk files
    cleanup_spinner = Spinner('Cleaning up temporary chunk files...', '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')
    cleanup_spinner.start()
    for c in chunks:
        try:
            c.unlink(missing_ok=True)
        except Exception:
            pass
    cleanup_spinner.succeed("Temporary files cleaned up.")

    # Combine in order
    combined = "\n\n".join(results[i] for i in sorted(results.keys()))
    return combined.strip()


def transcribe_audio(file_path: Path, provider_service: TranscriptionProvider, config: Dict[str, Any], output_to_file=True, output_path: Optional[Path]=None) -> Optional[str]:
    """Dispatch to direct or chunked transcription based on file size and config limit."""
    print(f"\nTranscribing: {file_path.name}")

    # Tools check (only required for chunking path)
    ffmpeg_ok = ensure_tool_exists(config['processing']['ffmpeg_path'])
    ffprobe_ok = ensure_tool_exists(config['processing']['ffprobe_path'])

    max_mb = float(config['limits']['max_file_mb'])
    max_bytes = int(max_mb * 1024 * 1024)
    fsize = file_path.stat().st_size

    spinner = Spinner('Transcription in progress...', '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')

    api_config = config['api']
    chunking_strategy = api_config.get('chunking', 'auto') # Default to 'auto' if not specified

    try:
        if chunking_strategy == 'local':
            # Force local chunking
            if not (ffmpeg_ok and ffprobe_ok):
                print("Error: Local chunking requested, but FFmpeg/FFprobe is not available.")
                return None
            text = transcribe_large_file(file_path, provider_service, config, temp_dir=Path("./whisper_temp_audio"))
        elif chunking_strategy == 'disable':
            # Force single file transcription, fail if too large
            if fsize > max_bytes:
                print("Error: Chunking disabled and file exceeds size limit.")
                return None
            spinner.start()
            text = transcribe_single_file(file_path, provider_service)
            spinner.succeed("Transcription successful!")
        else: # 'auto' chunking strategy
            if fsize <= max_bytes:
                spinner.start()
                text = transcribe_single_file(file_path, provider_service)
                spinner.succeed("Transcription successful!")
            else:
                if not (ffmpeg_ok and ffprobe_ok):
                    print("Error: File exceeds size limit and FFmpeg/FFprobe is not available for chunking.")
                    return None
                text = transcribe_large_file(file_path, provider_service, config, temp_dir=Path("./whisper_temp_audio"))
        
        # Output handling
        if output_to_file and output_path:
            save_transcription_to_file(text, output_path)
        else:
            print("-" * 20)
            print(text)
            print("-" * 20)
        return text
    except Exception as e:
        spinner.fail(f"Error: API or processing failed: {e}")
        return None