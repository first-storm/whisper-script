# -*- coding: utf-8 -*-
"""
This module provides transcription services for Groq API.
"""

from pathlib import Path
from groq import Groq
from typing import Dict, Any, Optional

class GroqProvider:
    def __init__(self, client: Groq, config: Dict[str, Any]):
        self.client = client
        self.config = config

    def transcribe_audio_file(self, file_path: Path) -> str:
        """Transcribe a single small file directly via Groq API."""
        api_config = self.config['api']
        
        # Extract parameters, providing defaults or None if not present
        model = api_config.get('model', 'whisper-large-v3-turbo')
        temperature = float(api_config.get('temperature', 0.0))
        response_format = api_config.get('response_format', 'text')
        language = api_config.get('language') # Optional
        prompt = api_config.get('prompt')     # Optional

        with open(file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                file=audio_file,
                model=model,
                temperature=temperature,
                response_format=response_format,
                language=language, # Pass language if available
                prompt=prompt      # Pass prompt if available
            )
        if isinstance(transcription, str):
            return transcription.strip()
        else:
            return str(transcription).strip()