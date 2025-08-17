# Whisper Transcription Tool

A unified speech-to-text transcription tool supporting local audio files and YouTube links with multiple AI providers (OpenAI, Groq, Google Gemini).

## Features

- 🎵 **Local Audio Files**: Process MP3, WAV, M4A, and other audio formats
- 🎬 **YouTube Support**: Automatically download and transcribe YouTube videos
- 🤖 **Multiple AI Providers**: Support for OpenAI Whisper, Groq, and Google Gemini
- ☁️ **S3 Integration**: Optional S3-compatible storage upload/deletion
- 🔄 **Interactive Mode**: Process files one by one interactively
- 📝 **Flexible Output**: Save to files or output to console
- ⚙️ **Configurable**: XDG-compliant configuration management

## Installation

### Using pipx (Recommended)

```bash
# Install directly from the current directory
pipx install .

# Or install from git repository
pipx install git+https://github.com/first-storm/whisper-script.git
```

### Using pip

```bash
# Install in a virtual environment
pip install .

# Or install from git repository
pip install git+https://github.com/first-storm/whisper-script.git
```

## Configuration

The tool supports multiple AI providers through profile-based configuration. Create a configuration file at `~/.config/whisper/config.yaml`:

```yaml
# Profiles let you define multiple API configurations and switch between them
profiles:
  default:
    provider: openai
    chunking: auto  # auto, local, disable
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}   # environment variable expansion supported
    model: whisper-1
    response_format: text
    temperature: 0.0
    language: en

  google:
    provider: google
    chunking: auto
    base_url: https://generativelanguage.googleapis.com  # optional: custom API endpoint
    api_key: ${GOOGLE_API_KEY}
    model: gemini-2.5-flash
    prompt: "Generate a transcript of the speech."
    temperature: 0.0
    force_upload: false  # optional: force Files API even for small files

  groq:
    provider: groq
    chunking: auto
    api_key: ${GROQ_API_KEY}
    model: whisper-large-v3
    temperature: 0.0

# Optional: pick the default profile if `--profile` and env `WHISPER_PROFILE` are unset
default_profile: default

limits:
  max_file_mb: 25

processing:
  workers: 4
  ffmpeg_path: ffmpeg
  ffprobe_path: ffprobe
  silence:
    noise_db: -35
    min_silence_sec: 0.6
    padding_sec: 0.1
    max_chunk_sec_cap: 1200.0

s3:
  enabled: false  # set to true to enable S3 uploads
  endpoint_url: "https://s3.amazonaws.com"  # or your S3-compatible endpoint
  access_key_id: "your-access-key"
  secret_access_key: "your-secret-key"
  bucket_name: "your-bucket-name"
  public_domain: "your-domain.com"  # optional: for public URLs
  auto_delete: false  # set to true to auto-delete after transcription
```

## Usage

Once installed, you can use the `transcribe` command:

### Basic Usage

```bash
# Transcribe a local audio file
transcribe audio.mp3

# Transcribe multiple files
transcribe audio1.mp3 audio2.wav

# Transcribe a YouTube video
transcribe "https://www.youtube.com/watch?v=VIDEO_ID"

# Output to console instead of files
transcribe --console audio.mp3

# Use a specific provider profile
transcribe --profile google audio.mp3

# Output in SRT format (OpenAI and Google providers only)
transcribe --srt audio.mp3
```

### Interactive Mode

```bash
# Enter interactive mode
transcribe -i

# Interactive mode with console output
transcribe -i --console

# Interactive mode with specific provider
transcribe -i --profile google
```

### Help

```bash
transcribe --help
```

## Requirements

- Python 3.8+
- API key for your chosen provider:
  - OpenAI API key for OpenAI Whisper
  - Groq API key for Groq
  - Google API key for Google Gemini
- `yt-dlp` for YouTube support (automatically installed)

## Dependencies

- `openai` - OpenAI API client
- `groq` - Groq API client
- `google-genai` - Google Gemini API client
- `boto3` - AWS/S3 integration
- `halo` - CLI progress indicators
- `PyYAML` - Configuration file parsing
- `yt-dlp` - YouTube video downloading