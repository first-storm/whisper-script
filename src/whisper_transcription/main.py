#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import uuid
import subprocess
import argparse
from pathlib import Path
import yaml
from openai import OpenAI
import boto3
from botocore.exceptions import NoCredentialsError
from halo import Halo
import re  # added for domain normalization

# --- Configuration Loading ---
def load_config():
    """Loads configuration from XDG config directory (whisper/config.yaml)."""
    # Use XDG_CONFIG_HOME if set, otherwise fall back to ~/.config
    xdg_config_home = os.environ.get('XDG_CONFIG_HOME')
    if xdg_config_home:
        config_dir = Path(xdg_config_home)
    else:
        config_dir = Path.home() / ".config"
    
    config_path = config_dir / "whisper" / "config.yaml"
    
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        print("Please create a configuration file according to the documentation.")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error: Failed to parse configuration file: {e}")
            sys.exit(1)

# --- S3 File Upload ---
def upload_to_s3(file_path, config):
    """Uploads a file to S3-compatible storage."""
    if not config.get('s3', {}).get('enabled', False):
        return None
        
    s3_config = config['s3']
    s3_client = boto3.client(
        's3',
        endpoint_url=s3_config['endpoint_url'],
        aws_access_key_id=s3_config['access_key_id'],
        aws_secret_access_key=s3_config['secret_access_key'],
    )
    
    file_extension = Path(file_path).suffix
    random_filename = f"{uuid.uuid4()}{file_extension}"
    object_name = f"audio-uploads/{random_filename}"
    
    spinner = Halo(text=f"Uploading to S3 Bucket '{s3_config['bucket_name']}'...", spinner='dots')
    try:
        spinner.start()
        s3_client.upload_file(str(file_path), s3_config['bucket_name'], object_name)
        spinner.succeed("Upload successful.")
        
        public_domain = s3_config.get('public_domain')
        public_url = None
        if public_domain:
            # Normalize: strip duplicated protocols and trailing slashes to avoid double https://
            raw = public_domain.strip()
            raw = re.sub(r'^(https?://)+', '', raw)  # remove any leading protocol repetitions
            raw = raw.strip('/')  # remove trailing slash
            public_url = f"https://{raw}/{object_name}"
            print(f"Public URL: {public_url}")
        
        # Always return structure containing object_key for downstream deletion
        return {
            "object_key": object_name,
            "public_url": public_url
        }
    except FileNotFoundError:
        spinner.fail(f"Error: Local file not found at {file_path}")
        return None
    except NoCredentialsError:
        spinner.fail("Error: S3 credentials not found or invalid.")
        return None
    except Exception as e:
        spinner.fail(f"S3 upload failed: {e}")
        return None

# --- S3 File Deletion ---
def delete_from_s3(object_key, config):
    """Deletes a file from S3-compatible storage."""
    if not config.get('s3', {}).get('enabled', False) or not config.get('s3', {}).get('auto_delete', False):
        return False
        
    s3_config = config['s3']
    s3_client = boto3.client(
        's3',
        endpoint_url=s3_config['endpoint_url'],
        aws_access_key_id=s3_config['access_key_id'],
        aws_secret_access_key=s3_config['secret_access_key'],
    )
    
    spinner = Halo(text=f"Deleting from S3: {object_key}", spinner='dots')
    try:
        spinner.start()
        s3_client.delete_object(Bucket=s3_config['bucket_name'], Key=object_key)
        spinner.succeed("S3 file deleted successfully.")
        return True
    except Exception as e:
        spinner.fail(f"S3 file deletion failed: {e}")
        return False

# --- Text Output Function ---
def save_transcription_to_file(transcription_text, output_path):
    """Saves transcription text to a file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription_text.strip())
        print(f"Transcription saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving transcription to file: {e}")
        return False

# --- Core Transcription Function ---
def transcribe_audio(file_path, client, config, output_to_file=True, output_path=None):
    """Calls the API to perform speech-to-text transcription."""
    print(f"\n----- Transcribing: {file_path.name} -----")
    api_config = config['api']
    spinner = Halo(text='Transcription in progress...', spinner='bouncingBar')
    
    try:
        spinner.start()
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=api_config.get('model', 'whisper-1'),
                file=audio_file,
                language=api_config.get('language'),
                temperature=float(api_config.get('temperature', 0.0)),
                prompt=api_config.get('prompt'),
                response_format=api_config.get('response_format', 'text')
            )
        spinner.succeed("Transcription successful!")
        
        # Get transcription text
        if isinstance(transcription, str):
            transcription_text = transcription.strip()
        else:
            transcription_text = str(transcription).strip()
        
        # Output handling
        if output_to_file and output_path:
            save_transcription_to_file(transcription_text, output_path)
        else:
            print("-" * 20)
            print(transcription_text)
            print("-" * 20)
            
        return transcription_text
    except Exception as e:
        spinner.fail(f"Error: API call failed: {e}")
        return None

# --- YouTube Audio Download ---
def download_youtube_audio(url, output_dir):
    """Downloads audio from a YouTube video using yt-dlp."""
    spinner = Halo(text=f"Downloading audio from YouTube: {url}", spinner='dots')
    random_id = uuid.uuid4()
    output_template = str(output_dir / f"yt_{random_id}.%(ext)s")
    
    command = ["yt-dlp", "-x", "--audio-format", "mp3", "-o", output_template, url]
    
    try:
        spinner.start()
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        spinner.succeed("Audio download complete.")
        actual_file = next(Path(output_dir).glob(f"yt_{random_id}.*"))
        return actual_file
    except FileNotFoundError:
        spinner.fail("Error: 'yt-dlp' command not found. Please ensure it's installed and in your PATH.")
        print("Installation: pip install yt-dlp")
        return None
    except subprocess.CalledProcessError as e:
        spinner.fail(f"Error: yt-dlp execution failed (Code {e.returncode}).")
        print(e.stderr)
        return None

# --- Process a single input item ---
def process_item(item, client, config, temp_dir, output_to_console=False):
    """Processes a single file path or URL."""
    audio_path = None
    is_temp_file = False
    output_path = None
    
    if item.startswith("http") and ("youtube.com" in item or "youtu.be" in item):
        audio_path = download_youtube_audio(item, temp_dir)
        is_temp_file = True
        if audio_path and not output_to_console:
            # Generate output filename based on downloaded file
            output_path = audio_path.with_suffix('.txt')
    elif Path(item).is_file():
        audio_path = Path(item)
        if not output_to_console:
            # Generate output filename based on input file
            output_path = audio_path.with_suffix('.txt')
    else:
        print(f"\nSkipping invalid input: '{item}' (not a valid file or YouTube link)")
        return
        
    if audio_path:
        s3_upload_info = upload_to_s3(audio_path, config)
        transcription_text = transcribe_audio(
            audio_path, 
            client, 
            config, 
            output_to_file=not output_to_console,
            output_path=output_path
        )
        # Updated deletion logic: always use object_key if auto_delete enabled
        if s3_upload_info and config.get('s3', {}).get('auto_delete', False):
            if isinstance(s3_upload_info, dict):
                key = s3_upload_info.get("object_key")
            else:  # backward fallback
                key = s3_upload_info
            if key:
                delete_from_s3(key, config)
        if is_temp_file:
            try:
                os.remove(audio_path)
                print(f"Temporary file deleted: {audio_path.name}")
            except OSError as e:
                print(f"Failed to delete temporary file: {e}")

# --- Interactive Mode ---
def interactive_mode(client, config, temp_dir, output_to_console=False):
    """Runs the script in an interactive command-line mode."""
    print("\n--- Interactive Transcription Mode ---")
    print("Enter a file path or YouTube URL. Type 'quit' or 'exit' to finish.")
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
        if not user_input:
            continue
        process_item(user_input, client, config, temp_dir, output_to_console)
    print("\nExiting interactive mode.")

# --- Main Function ---
def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="A unified speech-to-text script supporting local files and YouTube links.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "inputs", 
        nargs='*', 
        help="One or more input sources.\nCan be:\n  - Local audio file path (e.g., audio.mp3)\n  - YouTube video URL (e.g., 'https://www.youtube.com/watch?v=...')"
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
    args = parser.parse_args()
    
    config = load_config()
    
    try:
        client = OpenAI(
            base_url=config['api']['base_url'],
            api_key=config['api']['api_key']
        )
    except KeyError as e:
        print(f"Error: Missing required API configuration key in config file: {e}")
        sys.exit(1)
        
    temp_dir = Path("./whisper_temp_audio")
    temp_dir.mkdir(exist_ok=True)

    if args.interactive:
        interactive_mode(client, config, temp_dir, args.console)
    elif args.inputs:
        print(f"Found {len(args.inputs)} items to process.")
        for item in args.inputs:
            process_item(item, client, config, temp_dir, args.console)
    else:
        parser.print_help()
        print("\nError: No inputs provided. Please provide file paths/URLs or use interactive mode (-i).")
        sys.exit(1)

    # Clean up empty temp directory
    try:
        if not any(temp_dir.iterdir()):
            temp_dir.rmdir()
    except OSError:
        pass  # Don't delete if not empty
        
    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()
