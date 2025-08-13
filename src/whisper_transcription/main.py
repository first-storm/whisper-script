#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speech-to-text CLI for local audio files and YouTube links.

This version adds **multi-profile API support** via config:
- Define multiple API profiles under `profiles:` (provider, base_url, api_key, model, temperature, etc.)
- Select profile order of precedence: `--profile` > env `WHISPER_PROFILE` > `default_profile` in config > first profile.
- Backward compatible with legacy `api:` root section when `profiles:` is absent.

Notes:
- All comments are written in English per user request.
"""

import os
import sys
import uuid
import math
import re
import subprocess
import argparse
import threading
import time
from pathlib import Path
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict

from openai import OpenAI


# -----------------------------
# Custom Spinner Implementation
# -----------------------------
class Spinner:
    def __init__(self, message='Loading...', spinner_chars='|/-\\'):
        self.message = message
        self.spinner_chars = spinner_chars
        self.spinning = False
        self.thread = None
        self.idx = 0

    def _spin(self):
        while self.spinning:
            print(f'\r{self.spinner_chars[self.idx % len(self.spinner_chars)]} {self.message}', end='', flush=True)
            self.idx += 1
            time.sleep(0.1)

    def start(self):
        if not self.spinning:
            self.spinning = True
            self.thread = threading.Thread(target=self._spin)
            self.thread.start()

    def stop(self, final_message=None):
        if self.spinning:
            self.spinning = False
            if self.thread:
                self.thread.join()
            if final_message:
                print(f'\r✓ {final_message}')
            else:
                print(f'\r✓ {self.message}')

    def fail(self, error_message):
        if self.spinning:
            self.spinning = False
            if self.thread:
                self.thread.join()
        print(f'\r✗ {error_message}')

    def succeed(self, success_message):
        if self.spinning:
            self.spinning = False
            if self.thread:
                self.thread.join()
        print(f'\r✓ {success_message}')


# -----------------------------
# Configuration Loading
# -----------------------------

def load_config():
    """Load configuration from XDG config directory (`~/.config/whisper/config.yaml`).

    The function keeps backward compatibility with a single `api:` block.
    Defaults are still applied so the old config remains usable.
    """
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
            cfg = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"Error: Failed to parse configuration file: {e}")
            sys.exit(1)

    # Back-compat defaults for legacy single-profile configs
    cfg.setdefault('api', {})
    cfg['api'].setdefault('model', 'whisper-1')
    cfg['api'].setdefault('response_format', 'text')
    cfg['api'].setdefault('temperature', 0.0)

    # Global limits / processing defaults
    cfg.setdefault('limits', {})
    cfg['limits'].setdefault('max_file_mb', 25)

    cfg.setdefault('processing', {})
    cfg['processing'].setdefault('workers', 4)
    cfg['processing'].setdefault('ffmpeg_path', 'ffmpeg')
    cfg['processing'].setdefault('ffprobe_path', 'ffprobe')
    cfg['processing'].setdefault('silence', {})
    cfg['processing']['silence'].setdefault('noise_db', -35)         # dB threshold for silence
    cfg['processing']['silence'].setdefault('min_silence_sec', 0.6)  # min silence duration
    cfg['processing']['silence'].setdefault('padding_sec', 0.1)      # padding around cuts
    cfg['processing']['silence'].setdefault('max_chunk_sec_cap', 1200.0)  # 20 minutes cap

    return cfg


# -----------------------------
# Profile Resolving (NEW)
# -----------------------------

def _apply_api_defaults(api: dict) -> dict:
    """Apply sane defaults to an API profile dict."""
    api = dict(api or {})
    api.setdefault('model', 'whisper-1')
    api.setdefault('response_format', 'text')
    api.setdefault('temperature', 0.0)
    # Optional keys: language, prompt, base_url, api_key, provider
    return api


def _expand_env_in_obj(obj):
    """Recursively expand environment variables like ${ENV_VAR} in strings."""
    if isinstance(obj, dict):
        return {k: _expand_env_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_in_obj(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def select_api_profile(cfg: dict, cli_profile: Optional[str] = None) -> None:
    """Resolve and select the API profile, writing the result back to cfg['api'].

    Selection precedence:
        1) --profile (CLI)
        2) WHISPER_PROFILE (env)
        3) cfg['default_profile']
        4) 'default' if present in profiles
        5) First profile available

    Backward compatibility:
        If no `profiles:` exists, use cfg['api'] as-is (with defaults applied).
    """
    profiles = cfg.get('profiles') or {}
    env_profile = os.environ.get('WHISPER_PROFILE')
    chosen = cli_profile or env_profile or cfg.get('default_profile')

    # Legacy path: no profiles configured
    if not profiles:
        cfg['api'] = _apply_api_defaults(cfg.get('api', {}))
        cfg['api'] = _expand_env_in_obj(cfg['api'])
        cfg['selected_profile_name'] = 'api'
        return

    # If not explicitly chosen, try 'default' else first available key
    if not chosen:
        chosen = 'default' if 'default' in profiles else next(iter(profiles.keys()))

    if chosen not in profiles:
        available = ", ".join(sorted(profiles.keys()))
        raise SystemExit(f"Error: profile '{chosen}' not found. Available: {available}")

    # Merge defaults, expand env vars and store back
    api = _apply_api_defaults(profiles[chosen])
    api = _expand_env_in_obj(api)

    if isinstance(api, dict) and not api.get('api_key'):
        print(
            f"Warning: profile '{chosen}' has no 'api_key'. "
            f"If your provider requires it, set it in config or via environment."
        )

    cfg['api'] = api
    cfg['selected_profile_name'] = chosen


# -----------------------------
# Utilities
# -----------------------------

def human_readable_size(bytes_val: int) -> str:
    units = ['B', 'KB', 'MB', 'GB']
    size = float(bytes_val)
    for u in units:
        if size < 1024.0 or u == 'GB':
            return f"{size:.2f} {u}"
        size /= 1024.0
    # Fallback in case bytes_val is extremely large
    return f"{size:.2f} TB"


def ensure_tool_exists(tool: str) -> bool:
    try:
        subprocess.run([tool, '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except FileNotFoundError:
        return False


def run_subprocess(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)


def get_media_info(ffprobe: str, input_path: Path) -> Tuple[Optional[float], Optional[int]]:
    """
    Return (duration_sec, bitrate_bps) using ffprobe.
    """
    cmd = [
        ffprobe, "-v", "error", "-hide_banner",
        "-show_entries", "format=duration,bit_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path)
    ]
    proc = run_subprocess(cmd)
    if proc.returncode != 0:
        return None, None
    lines = [l.strip() for l in proc.stdout.strip().splitlines() if l.strip()]
    duration = None
    bitrate = None
    if len(lines) >= 1:
        try:
            duration = float(lines[0])
        except Exception:
            duration = None
    if len(lines) >= 2:
        try:
            bitrate = int(float(lines[1]))
        except Exception:
            bitrate = None
    return duration, bitrate


def detect_silences(ffmpeg: str, input_path: Path, noise_db: int, min_silence_sec: float) -> List[Tuple[float, float]]:
    """
    Use ffmpeg silencedetect to find silence start/end times.
    Return list of (silence_start, silence_end) pairs.
    The track start/end will be handled by caller.
    """
    cmd = [
        ffmpeg, "-hide_banner", "-nostats", "-i", str(input_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_sec}",
        "-f", "null", "-"
    ]
    proc = run_subprocess(cmd)
    stderr = proc.stderr or ""
    starts = []
    ends = []
    for line in stderr.splitlines():
        m1 = re.search(r"silence_start:\s*([0-9\.]+)", line)
        if m1:
            starts.append(float(m1.group(1)))
        m2 = re.search(r"silence_end:\s*([0-9\.]+)", line)
        if m2:
            ends.append(float(m2.group(1)))
    # Pair starts to ends in order, truncating extras if mismatch
    pairs = []
    i = j = 0
    while i < len(starts) and j < len(ends):
        if ends[j] <= starts[i]:
            # Stray end before start; skip
            j += 1
            continue
        pairs.append((starts[i], ends[j]))
        i += 1
        j += 1
    return pairs


def derive_voice_segments(duration: float, silence_pairs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    From silence intervals, derive non-silence (voice) segments as (start, end) within [0, duration].
    """
    segments = []
    cur = 0.0
    for (s_start, s_end) in silence_pairs:
        s_start = max(0.0, min(s_start, duration))
        s_end = max(0.0, min(s_end, duration))
        if s_start > cur:
            segments.append((cur, s_start))
        cur = max(cur, s_end)
    if cur < duration:
        segments.append((cur, duration))
    # Filter out zero or negative lengths
    segments = [(a, b) for (a, b) in segments if b - a > 1e-3]
    return segments


def group_segments(voice_segments: List[Tuple[float, float]],
                   max_group_sec: float
                   ) -> List[Tuple[float, float]]:
    """
    Merge consecutive voice segments into groups not exceeding max_group_sec.
    """
    groups = []
    acc_start = None
    acc_end = None
    acc_len = 0.0

    for (v_start, v_end) in voice_segments:
        v_len = v_end - v_start
        if acc_start is None:
            acc_start, acc_end, acc_len = v_start, v_end, v_len
            continue
        if (acc_len + v_len) <= max_group_sec:
            acc_end = v_end
            acc_len += v_len
        else:
            groups.append((acc_start, acc_end))
            acc_start, acc_end, acc_len = v_start, v_end, v_len

    if acc_start is not None:
        groups.append((acc_start, acc_end))

    # Clean any zero-lengths
    groups = [(a, b) for (a, b) in groups if (b - a) > 1e-3]
    return groups


def clip_with_padding(a: float, b: float, padding: float, duration: float) -> Tuple[float, float]:
    start = max(0.0, a - padding)
    end = min(duration, b + padding)
    if end <= start:
        end = min(duration, start + 0.1)
    return (start, end)


def export_chunk(ffmpeg: str, input_path: Path, out_path: Path, start: float, end: float) -> bool:
    """
    Export a chunk via stream copy for speed (audio only).
    Place -ss/-to after -i for better accuracy.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-hide_banner", "-y",
        "-i", str(input_path),
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-c", "copy",
        str(out_path)
    ]
    proc = run_subprocess(cmd)
    return proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


def reencode_shrink(ffmpeg: str, input_path: Path, out_path: Path, target_bitrate_kbps: int = 96) -> bool:
    """
    Re-encode the chunk to AAC M4A at target bitrate to shrink under size limit if needed.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-hide_banner", "-y",
        "-i", str(input_path),
        "-vn",
        "-c:a", "aac",
        "-b:a", f"{target_bitrate_kbps}k",
        str(out_path)
    ]
    proc = run_subprocess(cmd)
    return proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


def split_on_silence_to_chunks(input_path: Path,
                               temp_dir: Path,
                               config) -> List[Path]:
    """
    High-level: determine chunking based on silences and size limit.
    Return a list of chunk files.
    """
    ffmpeg = config['processing']['ffmpeg_path']
    ffprobe = config['processing']['ffprobe_path']
    silence_cfg = config['processing']['silence']
    max_mb = float(config['limits']['max_file_mb'])
    max_bytes = int(max_mb * 1024 * 1024)
    cap_sec = float(silence_cfg.get('max_chunk_sec_cap', 1200.0))
    padding_sec = float(silence_cfg['padding_sec'])

    # Gather media info
    duration, bitrate_bps = get_media_info(ffprobe, input_path)
    if duration is None or duration <= 0:
        raise RuntimeError("Unable to read media duration via ffprobe.")
    if not bitrate_bps or bitrate_bps <= 0:
        # Fallback assumption: 128 kbps
        bitrate_bps = 128_000

    # Compute safe max duration to stay below size limit
    # Add headroom factor to reduce risk of overshooting due to VBR/container overhead
    headroom = 0.95
    max_group_sec_by_size = max_bytes * headroom / bitrate_bps
    max_group_sec = max(5.0, min(max_group_sec_by_size, cap_sec))

    # Detect silences and derive voice segments
    silences = detect_silences(
        ffmpeg,
        input_path,
        noise_db=int(silence_cfg['noise_db']),
        min_silence_sec=float(silence_cfg['min_silence_sec'])
    )
    voice_segments = derive_voice_segments(duration, silences)
    if not voice_segments:
        # No clear voice areas; fallback to naive fixed-time grouping to respect size limit
        voice_segments = [(0.0, duration)]

    groups = group_segments(
        voice_segments,
        max_group_sec=max_group_sec
    )

    # Export chunks
    chunks_dir = temp_dir / "chunks" / input_path.stem
    out_ext = input_path.suffix or ".mp3"
    chunk_files = []
    for idx, (a, b) in enumerate(groups, start=1):
        s, e = clip_with_padding(a, b, padding_sec, duration)
        out_path = chunks_dir / f"{input_path.stem}_chunk{idx:03d}{out_ext}"
        ok = export_chunk(ffmpeg, input_path, out_path, s, e)
        if not ok:
            raise RuntimeError(f"FFmpeg failed to export chunk {idx}")
        # Enforce size limit if still too large: re-encode and replace
        if out_path.stat().st_size > max_bytes:
            shrunk = out_path.with_suffix(".m4a")
            if reencode_shrink(ffmpeg, out_path, shrunk, target_bitrate_kbps=96):
                if shrunk.stat().st_size <= max_bytes:
                    try:
                        out_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    out_path = shrunk
                else:
                    # If still too big, try an even lower bitrate
                    shrunk2 = out_path.with_name(out_path.stem + "_64k.m4a")
                    if reencode_shrink(ffmpeg, shrunk, shrunk2, target_bitrate_kbps=64):
                        try:
                            out_path.unlink(missing_ok=True)
                            shrunk.unlink(missing_ok=True)
                        except Exception:
                            pass
                        out_path = shrunk2
            # Final guard: if still oversized, warn (rare)
            if out_path.stat().st_size > max_bytes:
                size_str = human_readable_size(out_path.stat().st_size)
                print(f"Warning: a chunk remains over the size limit ({size_str} > {max_mb} MB). Proceeding anyway.")
        chunk_files.append(out_path)

    return chunk_files


# -----------------------------
# Text Output
# -----------------------------

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


# -----------------------------
# API Transcription
# -----------------------------

def transcribe_single_file(file_path: Path, client: OpenAI, config) -> str:
    """Transcribe a single small file directly via API."""
    api_config = config['api']
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=api_config.get('model', 'whisper-1'),
            file=audio_file,
            language=api_config.get('language'),
            temperature=float(api_config.get('temperature', 0.0)),
            prompt=api_config.get('prompt'),
            response_format=api_config.get('response_format', 'text')
        )
    if isinstance(transcription, str):
        return transcription.strip()
    else:
        return str(transcription).strip()


def transcribe_chunk(file_path: Path, client: OpenAI, config, index: int) -> Tuple[int, str]:
    """Transcribe a chunk and return (index, text)."""
    text = transcribe_single_file(file_path, client, config)
    return index, text


def transcribe_large_file(file_path: Path, client: OpenAI, config, temp_dir: Path) -> str:
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
            executor.submit(transcribe_chunk, chunk, client, config, idx): (idx, chunk)
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


def transcribe_audio(file_path: Path, client: OpenAI, config, output_to_file=True, output_path: Optional[Path]=None) -> Optional[str]:
    """Dispatch to direct or chunked transcription based on file size and config limit."""
    print(f"\nTranscribing: {file_path.name}")

    # Tools check (only required for chunking path)
    ffmpeg_ok = ensure_tool_exists(config['processing']['ffmpeg_path'])
    ffprobe_ok = ensure_tool_exists(config['processing']['ffprobe_path'])

    max_mb = float(config['limits']['max_file_mb'])
    max_bytes = int(max_mb * 1024 * 1024)
    fsize = file_path.stat().st_size

    spinner = Spinner('Transcription in progress...', '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')

    try:
        if fsize <= max_bytes:
            spinner.start()
            text = transcribe_single_file(file_path, client, config)
            spinner.succeed("Transcription successful!")
        else:
            if not (ffmpeg_ok and ffprobe_ok):
                print("Error: File exceeds size limit and FFmpeg/FFprobe is not available for chunking.")
                return None
            text = transcribe_large_file(file_path, client, config, temp_dir=Path("./whisper_temp_audio"))
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


# -----------------------------
# YouTube Audio Download
# -----------------------------

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


# -----------------------------
# Process a single input item
# -----------------------------

def process_item(item: str, client: OpenAI, config, temp_dir: Path, output_to_console=False):
    """Process a single file path or URL."""
    audio_path = None
    is_temp_file = False
    output_path = None

    if item.startswith("http") and ("youtube.com" in item or "youtu.be" in item):
        audio_path = download_youtube_audio(item, temp_dir)
        is_temp_file = True
        if audio_path and not output_to_console:
            output_path = audio_path.with_suffix('.txt')
    elif Path(item).is_file():
        audio_path = Path(item)
        if not output_to_console:
            output_path = audio_path.with_suffix('.txt')
    else:
        print(f"\nSkipping invalid input: '{item}' (not a valid file or YouTube link)")
        return

    if audio_path:
        transcription_text = transcribe_audio(
            audio_path,
            client,
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


# -----------------------------
# Interactive Mode
# -----------------------------

def interactive_mode(client: OpenAI, config, temp_dir: Path, output_to_console=False):
    """Run the script in an interactive command-line mode."""
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


# -----------------------------
# Main
# -----------------------------

def main():
    """Main execution function."""
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
    print(
        f"Using API profile: {sel} | provider={api_cfg.get('provider','?')} | model={api_cfg.get('model','?')}"
    )

    # Construct OpenAI-compatible client from selected profile
    base_url = api_cfg.get('base_url')
    api_key = api_cfg.get('api_key')

    if not api_key:
        print("Error: Missing 'api_key' for the selected profile. "
              "Set it in config.yaml or via environment variable.")
        sys.exit(1)

    if base_url:
        client = OpenAI(base_url=base_url, api_key=api_key)
    else:
        # No base_url => use official default
        client = OpenAI(api_key=api_key)

    # Prepare temp dir
    temp_dir = Path("./whisper_temp_audio")
    temp_dir.mkdir(exist_ok=True)

    # Provide a hint if FFmpeg/FFprobe is missing (only needed for chunking)
    if not ensure_tool_exists(config['processing']['ffmpeg_path']):
        print("Note: FFmpeg not found. Large files over the size limit cannot be auto-chunked.")
    if not ensure_tool_exists(config['processing']['ffprobe_path']):
        print("Note: FFprobe not found. Large files over the size limit cannot be analyzed for chunking.")

    if args.interactive:
        interactive_mode(client, config, temp_dir, args.console)
    elif args.inputs:
        print(f"Found {len(args.inputs)} item(s) to process.")
        for item in args.inputs:
            process_item(item, client, config, temp_dir, args.console)
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
    except OSError:
        pass  # Don't delete if not empty

    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()