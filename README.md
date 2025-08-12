# Whisper Transcription Tool

A unified speech-to-text transcription tool supporting local audio files and YouTube links using OpenAI Whisper API.

## Features

- 🎵 **Local Audio Files**: Process MP3, WAV, M4A, and other audio formats
- 🎬 **YouTube Support**: Automatically download and transcribe YouTube videos
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

Create a configuration file at `~/.config/whisper/config.yaml`:

```yaml
api:
  base_url: "https://api.openai.com/v1"  # or your custom API endpoint
  api_key: "your-api-key-here"
  model: "whisper-1"
  language: null  # optional: specify language code (e.g., "en", "zh")
  temperature: 0.0
  prompt: null  # optional: provide context/prompt
  response_format: "text"  # or "json", "srt", "verbose_json", "vtt"

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
```

### Interactive Mode

```bash
# Enter interactive mode
transcribe -i

# Interactive mode with console output
transcribe -i --console
```

### Help

```bash
transcribe --help
```

## Requirements

- Python 3.8+
- OpenAI API key (or compatible API endpoint)
- `yt-dlp` for YouTube support (automatically installed)

## Dependencies

- `openai` - OpenAI API client
- `boto3` - AWS/S3 integration
- `halo` - CLI progress indicators
- `PyYAML` - Configuration file parsing
- `yt-dlp` - YouTube video downloading