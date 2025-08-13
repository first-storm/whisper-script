# -*- coding: utf-8 -*-
"""
This module provides functions for interacting with external services,
such as downloading audio from YouTube.
"""

import subprocess
import uuid
from pathlib import Path
from typing import Optional

from .ui import Spinner

def download_youtube_audio(url, output_dir: Path) -> Optional[Path]:
    """Download audio from a YouTube video using yt-dlp."""
    spinner = Spinner(f"Downloading audio from YouTube: {url}", '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')
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