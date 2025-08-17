# -*- coding: utf-8 -*-
"""
google_provider.py

A Google Gemini transcription provider modeled after openai_provider.py / groq_provider.py.

Features
- Inline audio (<= 20 MB total request) or Files API upload (> 20 MB or forced)
- Prompt configurable via config or method arg
- Optional segment transcription using MM:SS timestamps
- Optional token counting helper
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import tempfile
import os

import json

from google import genai
from google.genai import types

from ..processing import convert_to_aac, ensure_tool_exists


# Gemini limits: inline total request size must be <= 20 MB.
# We keep a small headroom to account for prompt/system text.
DEFAULT_INLINE_LIMIT_BYTES = 20 * 1024 * 1024 - 64 * 1024

_SUPPORTED_MIME_BY_EXT = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".aiff": "audio/aiff",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


class GoogleProvider:
    """
    Google Gemini provider.

    Expected config shape (mirrors OpenAI/Groq style):
    {
        "api": {
            "model": "gemini-2.5-flash",
            "prompt": "Generate a transcript of the speech.",
            "temperature": 0.0,        # optional
            "force_upload": False,     # optional; force Files API even if small
            "inline_limit_bytes": 20930560  # optional; override DEFAULT_INLINE_LIMIT_BYTES
        }
    }
    """

    def __init__(self, client: genai.Client, config: Dict[str, Any]):
        self.client = client
        self.config = config

    # ---------- Public API ----------

    def transcribe_audio_file(
        self,
        file_path: Path,
        response_format: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Transcribe a single audio file with Gemini.
        - Uses inline bytes by default when the total request would be <= ~20 MB.
        - Falls back to Files API upload when larger (or when force_upload=True).

        :param file_path: Path to the audio file.
        :param response_format: "text" | "json" | "verbose_json" (kept for parity with other providers).
        :param prompt: Optional override for the prompt; defaults from config.
        :return: Transcribed text or formatted string per response_format.
        """
        api = self.config.get("api", {})
        model = api.get("model", "gemini-2.5-flash")
        gen_temperature = api.get("temperature", None)

        use_upload = bool(api.get("force_upload", False))
        inline_limit = int(api.get("inline_limit_bytes", DEFAULT_INLINE_LIMIT_BYTES))

        # Check if format is supported, convert if not
        converted_file = None
        try:
            mime_type = self._mime_type_for(file_path)
            actual_file_path = file_path
        except ValueError:
            # Unsupported format, convert to AAC
            print(f"Unsupported audio format: {file_path.suffix}. Converting to AAC...")
            ffmpeg_path = self.config.get('processing', {}).get('ffmpeg_path', 'ffmpeg')
            
            if not ensure_tool_exists(ffmpeg_path):
                raise RuntimeError(f"FFmpeg not found at '{ffmpeg_path}'. Please install FFmpeg or update the path in config.")
            
            # Create temporary AAC file
            with tempfile.NamedTemporaryFile(suffix='.aac', delete=False) as tmp_file:
                converted_file = Path(tmp_file.name)
            
            if not convert_to_aac(ffmpeg_path, file_path, converted_file):
                if converted_file.exists():
                    converted_file.unlink()
                raise RuntimeError(f"Failed to convert {file_path} to AAC format")
            
            actual_file_path = converted_file
            mime_type = "audio/aac"
            print(f"Successfully converted to AAC: {converted_file}")

        # Choose prompt
        final_prompt = prompt or api.get("prompt") or "Generate a transcript of the speech."

        try:
            # Decide inline vs upload
            if not use_upload and actual_file_path.stat().st_size <= inline_limit:
                contents = [final_prompt, self._make_inline_part(actual_file_path, mime_type)]
            else:
                uploaded = self.client.files.upload(file=str(actual_file_path))
                contents = [final_prompt, uploaded]

            # Build optional generation config only if provided
            kwargs: Dict[str, Any] = {"model": model, "contents": contents}
            if gen_temperature is not None:
                # Use types.GenerateContentConfig for proper configuration
                kwargs["config"] = types.GenerateContentConfig(
                    temperature=float(gen_temperature)
                )

            response = self.client.models.generate_content(**kwargs)
            return self._coerce_response(response, response_format)
        
        finally:
            # Clean up temporary converted file
            if converted_file and converted_file.exists():
                try:
                    converted_file.unlink()
                except Exception:
                    pass  # Ignore cleanup errors

    def transcribe_audio_segment(
        self,
        file_path: Path,
        start: Optional[Tuple[int, int]] = None,
        end: Optional[Tuple[int, int]] = None,
        response_format: Optional[str] = None,
        prompt_prefix: str = "Provide a transcript of the speech",
    ) -> str:
        """
        Transcribe a specific segment using MM:SS timestamp hints in the prompt.
        Automatically converts unsupported formats to AAC.

        :param file_path: Path to the audio file.
        :param start: (MM, SS) tuple or None.
        :param end: (MM, SS) tuple or None.
        :param response_format: "text" | "json" | "verbose_json".
        :param prompt_prefix: Customizable instruction prefix.
        """
        segment_instr = self._segment_prompt(start, end)
        prompt = f"{prompt_prefix}{segment_instr}."
        return self.transcribe_audio_file(file_path, response_format=response_format, prompt=prompt)

    def count_tokens(self, file_path: Path) -> Any:
        """
        Use Files API + count_tokens to estimate tokens for an audio file.
        Returns the raw response object from the SDK (shape may change with SDK versions).
        """
        model = self.config.get("api", {}).get("model", "gemini-2.5-flash")
        uploaded = self.client.files.upload(file=str(file_path))
        return self.client.models.count_tokens(model=model, contents=[uploaded])

    # ---------- Helpers ----------

    @staticmethod
    def _mime_type_for(file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext not in _SUPPORTED_MIME_BY_EXT:
            raise ValueError(f"Unsupported audio file format: {ext}")
        return _SUPPORTED_MIME_BY_EXT[ext]

    @staticmethod
    def _make_inline_part(file_path: Path, mime_type: str) -> types.Part:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        return types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

    @staticmethod
    def _mmss_tuple_to_str(t: Tuple[int, int]) -> str:
        mm, ss = t
        if mm < 0 or ss < 0 or ss >= 60:
            raise ValueError("Timestamp must be (MM >= 0, 0 <= SS < 60).")
        return f"{mm:02d}:{ss:02d}"

    def _segment_prompt(
        self,
        start: Optional[Tuple[int, int]],
        end: Optional[Tuple[int, int]],
    ) -> str:
        if start is None and end is None:
            return ""
        pieces = []
        if start is not None:
            pieces.append(f" from {self._mmss_tuple_to_str(start)}")
        if end is not None:
            # If only end is set, we'll still read as "to 00:SS/MM:SS"
            pieces.append(f" to {self._mmss_tuple_to_str(end)}")
        return "".join(pieces)

    @staticmethod
    def _coerce_response(response: Any, response_format: Optional[str]) -> str:
        """
        Normalize Gemini responses to match the provider interface used elsewhere.
        - "text": return response.text (default)
        - "json": return {"text": ...} as JSON string
        - "verbose_json": best-effort JSON dump of the whole response (fallback to str)
        """
        fmt = (response_format or "text").lower()
        # response.text is the canonical convenience accessor
        text = getattr(response, "text", None)
        if text is None:
            # Fallback if SDK changes; stringify the response
            text = str(response)

        if fmt == "text":
            return text.strip()

        if fmt == "json":
            return json.dumps({"text": text.strip()}, ensure_ascii=False)

        if fmt == "verbose_json":
            # Try structured dump if available; else fallback to str
            for attr in ("to_dict", "model_dump", "model_dump_json"):
                if hasattr(response, attr):
                    try:
                        val = getattr(response, attr)()
                        if isinstance(val, str):
                            return val
                        return json.dumps(val, ensure_ascii=False)
                    except Exception:
                        pass
            return json.dumps({"raw": str(response)}, ensure_ascii=False)

        # Unknown format → default to text
        return text.strip()
