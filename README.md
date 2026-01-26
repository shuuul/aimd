<div align="center">
  <img src="assets/banner.png" alt="aimd">

  ![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
  ![uv](https://img.shields.io/badge/uv-ready-blue)
  ![Version](https://img.shields.io/badge/version-0.0.3-blue)
  ![License](https://img.shields.io/badge/license-MIT-green)
</div>

# aimd

Context preparation tool for LLM workflows - Transcribe audio/video and convert documents to markdown.

## Features

- **Auto-Detection**: Automatically detects input type (audio, video, URL, document)
- **Multi-Engine Audio Transcription**: Support for yap (macOS), MLX-Whisper (Apple Silicon), faster-whisper (CUDA/CPU)
- **Unified Video Platform Support**: Extract content from 1000+ video platforms including YouTube, Bilibili using yt-dlp
- **Document Conversion**: Convert EPUB, PDF, TXT, and other formats to markdown using Pandoc

## Installation

### Quick Install

```bash
uv tool install git+https://github.com/shuuul/aimd
```

### System Requirements

**macOS**: Requires Apple Silicon (M1/M2/M3/M4) for optimal performance.

**Linux/Windows**: For GPU acceleration, requires CUDA 12.x and cuDNN 9.x.

### Install via uv (Recommended)

```bash
# Install from GitHub repository
uv tool install git+https://github.com/shuuul/aimd

# Verify installation
aimd --help
```

### Development Setup

```bash
# Clone and install for development
git clone https://github.com/shuuul/aimd.git
cd aimd
uv sync --all-packages --all-extras --dev
```

## Quick Start

```bash
# Process any input - auto-detects type
aimd audio.mp3                    # Transcribe audio
aimd "https://youtube.com/watch?v=..."  # Extract video subtitles
aimd book.epub                    # Convert EPUB to markdown
aimd document.txt                 # Convert text to markdown

# With options
aimd audio.mp3 -o output.md       # Custom output file
aimd audio.mp3 -e mlx             # Specify engine
aimd interview.wav -l zh_CN       # Specify locale
```

## Usage

### Audio/Video Transcription

```bash
# Automatic engine selection (recommended)
aimd audio.mp3 --engine auto

# Specify transcription engine explicitly
aimd audio.wav --engine mlx       # Apple Silicon
aimd audio.wav --engine yap       # macOS
aimd audio.wav --engine cuda      # NVIDIA GPU
aimd audio.wav --engine cpu       # Cross-platform

# Process with specific locale
aimd interview.m4a --locale zh_CN

# Custom output file
aimd lecture.mp3 -o meeting_notes.md
```

#### Available Transcription Engines

- **`auto`** (default): Automatically selects the best engine for your platform
- **`yap`**: macOS-only, fastest for supported languages (requires [yap CLI](https://github.com/finnvoor/yap))
- **`mlx`**: Apple Silicon only, 4-5x faster than CPU (uses mlx-whisper)
- **`cuda`**: NVIDIA GPU acceleration (requires CUDA 12 + cuDNN 9)
- **`cpu`**: Cross-platform CPU fallback (uses faster-whisper)

### Video URL Processing

Extract content from video platforms:

```bash
# YouTube video
aimd "https://www.youtube.com/watch?v=I3WUiD8HYn8"

# Bilibili video
aimd "https://www.bilibili.com/video/BV1Rz4y127jd"

# Xiaoyuzhou Podcast
aimd "https://www.xiaoyuzhoufm.com/episode/69277ae50084e2631deb56e0"
```

### Document Conversion

Convert documents to markdown:

```bash
# Convert EPUB to markdown (extracts images and chapters)
aimd book.epub

# Convert text file
aimd notes.txt

# Convert with custom output
aimd document.epub -o output.md
```

#### EPUB Output Structure

When processing EPUB files, aimd extracts:
- **Main file**: `{book_name}.md` - Combined content from all chapters
- **Chapters directory**: `chapters/` - Individual chapter markdown files
- **Images directory**: `images/` - All images from the EPUB

```
book_name/
├── book_name.md      # Combined content
├── chapters/
│   ├── chapter_001.md
│   ├── chapter_002.md
│   └── ...
└── images/
    └── *.jpg, *.png, etc.
```

## Supported Formats

### Video Platforms

Thanks to yt-dlp integration, aimd supports content extraction from:

- **YouTube** - Videos, subtitles, metadata
- **Bilibili** - Chinese video platform, subtitles, metadata
- **1000+ other platforms** - See [yt-dlp supported sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)

### Audio Files

- **Common Formats**: MP3, WAV, M4A, FLAC, OGG, AAC
- **Video Audio**: MP4 (extracts audio for transcription)
- **Languages**: Multi-language transcription with auto-detection

### Document Formats

- **EPUB**: `.epub` ebooks
- **Markdown**: `.md` files
- **Plain Text**: `.txt` files
- **And 40+ other formats** via Pandoc

> **Tip**: For scanned PDFs, we recommend using [MinerU](https://github.com/opendatalab/MinerU) or the [MinerU Desktop App](https://mineru.net) for high-quality OCR and layout extraction, then pass the extracted markdown to aimd for further processing.

## Configuration

### Environment Variables

```bash
# Optional: yt-dlp configuration
export YT_DLP_CONFIG_HOME="/path/to/config"
export YT_DLP_CACHE_DIR="/path/to/cache"

# Optional: yap engine configuration (macOS only)
export YAP_MODEL_PATH="/path/to/custom/model"
```

## Development

### Setup Development Environment

```bash
# Clone from the official repository
git clone https://github.com/shuuul/aimd.git
cd aimd

# Install dependencies
uv sync --all-packages --all-extras --dev

# Run code quality checks
uv run ruff check --fix && uv run ruff format
uv run pre-commit run --all-files
```

### Version Management

```bash
# Bump version
uv version --bump patch  # or minor, major
```

### Testing

```bash
# Run tests
uv run pytest

# Test CLI functionality
aimd --help
```

### Project Structure

```
aimd/
├── src/aimd/
│   ├── cli.py              # Unified CLI with auto-detection
│   ├── const.py            # Constants (extensions, engines, languages)
│   ├── utils.py            # Utility functions
│   ├── types.py            # Pydantic models (TextContext)
│   └── tool/
│       ├── audio.py        # Audio transcription engines
│       ├── file.py         # Document conversion with Pandoc
│       └── url.py          # Video URL processing with yt-dlp
├── tests/
├── pyproject.toml
└── AGENTS.md
```

## Architecture

### Core Components

- **Unified Processing**: Single command with auto-detection
- **Multi-Engine Audio Transcription**: Auto-selection between yap, MLX-Whisper, faster-whisper
- **Tool-Based Architecture**: Modular design with file, audio, and URL processing tools
- **Pandoc Integration**: Document conversion supporting 40+ formats

### Processing Pipelines

#### Audio/Video Processing

1. **Input Detection**: Auto-detect audio, video, or URL
2. **Engine Selection**: Choose optimal transcription engine
3. **Content Extraction**: Speech-to-text or subtitle extraction
4. **Markdown Formatting**: Structured output

#### Document Conversion

1. **Format Detection**: Identify document type via extension
2. **Pandoc Conversion**: Convert to markdown
3. **Title Extraction**: Extract clean titles from content
4. **Chunking**: Split large documents automatically

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the code style
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- **Issues**: Report bugs and request features on GitHub
- **Documentation**: See AGENTS.md for detailed architecture information
