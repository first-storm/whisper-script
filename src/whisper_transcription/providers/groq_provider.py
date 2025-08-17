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

    def transcribe_audio_file(self, file_path: Path, response_format: Optional[str] = None) -> str:
        """Transcribe a single small file directly via Groq API."""
        api_config = self.config['api']
        
        # Extract parameters, providing defaults or None if not present
        model = api_config.get('model', 'whisper-large-v3-turbo')
        temperature = float(api_config.get('temperature', 0.0))
        language = api_config.get('language') # Optional
        prompt = api_config.get('prompt')     # Optional

        params = {
            'file': None, # This will be set below
            'model': model,
            'temperature': temperature,
            'language': language,
            'prompt': prompt,
        }
        
        # Determine the response_format to use
        final_response_format = response_format if response_format else api_config.get('response_format')
        if final_response_format:
            # Only add if it's one of the expected literal values or can be coerced
            if final_response_format in ['json', 'text', 'verbose_json']:
                params['response_format'] = final_response_format
            else:
                # Default to 'text' if the configured format is not valid
                params['response_format'] = 'text'
        else:
            # Default to 'text' if no response_format is specified in argument or config
            params['response_format'] = 'text'

        with open(file_path, "rb") as audio_file:
            params['file'] = audio_file
            transcription = self.client.audio.transcriptions.create(**params)
        if isinstance(transcription, str):
            return transcription.strip()
        else:
            return str(transcription).strip()