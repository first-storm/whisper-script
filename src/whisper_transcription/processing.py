# -*- coding: utf-8 -*-
"""
This module handles audio processing, including chunking, silence detection,
and file operations using FFmpeg/FFprobe.
"""

import subprocess
import re
import math
from pathlib import Path
from typing import List, Tuple, Optional

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


def convert_to_aac(ffmpeg: str, input_path: Path, output_path: Path) -> bool:
    """
    Convert any audio format to AAC using ffmpeg.
    Returns True if conversion was successful, False otherwise.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-hide_banner", "-y",
        "-i", str(input_path),
        "-vn",  # No video
        "-c:a", "aac",  # AAC codec
        "-b:a", "128k",  # 128 kbps bitrate
        str(output_path)
    ]
    proc = run_subprocess(cmd)
    return proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0


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