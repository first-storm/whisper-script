# -*- coding: utf-8 -*-
"""
This module provides transcription services for OpenAI-compatible APIs.
"""

from pathlib import Path
from openai import OpenAI
from typing import Dict, Any

class OpenAIProvider:
    def __init__(self, client: OpenAI, config: Dict[str, Any]):
        self.client = client
        self.config = config

    def transcribe_audio_file(self, file_path: Path) -> str:
        """Transcribe a single small file directly via OpenAI API."""
        api_config = self.config['api']
        
        params = {
            'model': api_config.get('model', 'whisper-1'),
            'file': open(file_path, "rb"),
            'language': api_config.get('language'),
            'temperature': float(api_config.get('temperature', 0.0)),
            'prompt': api_config.get('prompt'),
            'response_format': api_config.get('response_format', 'text'),
        }

        # Only add chunking_strategy if it's 'auto' and provider is openai
        if api_config.get('chunking') == 'auto':
            params['chunking_strategy'] = 'auto'

        with open(file_path, "rb") as audio_file:
            params['file'] = audio_file
            transcription = self.client.audio.transcriptions.create(**params)

        if isinstance(transcription, str):
            return transcription.strip()
        else:
            return str(transcription).strip()