# -*- coding: utf-8 -*-
"""
This module handles loading and managing configuration, including API profiles.
"""

import os
import sys
from pathlib import Path
import yaml
from typing import Optional

def load_config():
    """Load configuration from XDG config directory (~/.config/whisper/config.yaml).

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
    cfg['api'].setdefault('provider', 'openai') # Default provider
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


def _apply_api_defaults(api: dict) -> dict:
    """Apply sane defaults to an API profile dict."""
    api = dict(api or {})
    api.setdefault('provider', 'openai') # Default provider
    api.setdefault('model', 'whisper-1')
    api.setdefault('response_format', 'text')
    api.setdefault('temperature', 0.0)
    # Optional keys: language, prompt, base_url, api_key, provider

    # Set default chunking strategy based on provider
    if api['provider'] == 'openai':
        api.setdefault('chunking', 'auto')
    else:
        api.setdefault('chunking', 'local')

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