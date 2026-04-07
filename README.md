<div align="center">
  <img src="assets/banner.png" alt="aimd">

  ![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
  ![uv](https://img.shields.io/badge/uv-ready-blue)
  ![Version](https://img.shields.io/badge/version-0.7.2-blue)
  ![License](https://img.shields.io/badge/license-MIT-green)
</div>

# aimd

Context preparation tool for LLM workflows - Transcribe audio/video and convert documents to markdown.

## Features

- **Auto-Detection**: Automatically detects input type (audio, video, URL, document)
- **Multi-Engine Audio Transcription**: Support for mlx-audio (Apple Silicon), Qwen3-ASR (Linux/CUDA), FunASR (CPU/CUDA), yap (macOS)
- **Unified Video Platform Support**: Extract content from 1000+ video platforms including YouTube, Bilibili using yt-dlp
- **Document Conversion**: Convert EPUB, PDF, TXT, and other formats to markdown using Pandoc

## Architecture

The runtime now uses a ports/adapters layout:

- `aimd.application` for use-cases and request/response models
- `aimd.infrastructure` for concrete transcription/URL/document processors
- `aimd.adapters` for CLI, HTTP API, and MCP interface layers

Entrypoints are exposed via `aimd.cli`, `aimd.api`, and `aimd.mcp`, backed by the adapter/application/infrastructure layers.

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
uv sync --dev --upgrade --all-extras


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
aimd interview.wav -l zh          # Specify language
```

## Usage

### Audio/Video Transcription

```bash
# Automatic engine selection (recommended)
aimd audio.mp3 --engine auto

# Specify transcription engine explicitly
aimd audio.wav --engine mlx       # Apple Silicon (default on macOS)
aimd audio.wav --engine qwen      # Linux/CUDA (Qwen3-ASR, default on Linux)
aimd audio.wav --engine funasr    # CPU/CUDA (SenseVoiceSmall, cross-platform fallback)
aimd audio.wav --engine yap       # macOS (Apple Speech framework)

# Select a specific model (mlx engine)
aimd audio.wav -e mlx -m mlx-community/Qwen3-ASR-0.6B-8bit

# Process with specific language
aimd interview.m4a --language zh

# Custom output file
aimd lecture.mp3 -o meeting_notes.md
```

#### Available Transcription Engines

- **`auto`** (default): Automatically selects the best engine for your platform
- **`mlx`**: Apple Silicon only, uses [mlx-audio](https://github.com/Blaizzy/mlx-audio) STT (Qwen3-ASR-1.7B-8bit by default). Highest priority on macOS.
- **`qwen`**: Linux + CUDA, uses [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) via `qwen-asr` (Qwen3-ASR-1.7B by default). Highest priority on Linux.
- **`funasr`**: CPU/CUDA, uses [FunASR](https://github.com/modelscope/FunASR) (SenseVoiceSmall by default). Cross-platform fallback engine.
- **`yap`**: macOS-only, uses Apple Speech framework (requires [yap CLI](https://github.com/finnvoor/yap))

#### Auto-Selection Priority

| Platform | Priority |
|----------|----------|
| macOS (Apple Silicon) | `mlx` → `yap` → `funasr` |
| Linux | `qwen` → `funasr` |

#### MLX Model Selection

The `mlx` engine supports multiple models via `--model` / `-m`:

| Model | Description |
|-------|-------------|
| `mlx-community/Qwen3-ASR-1.7B-8bit` | Qwen3-ASR 1.7B, 8-bit quantized **(default)** |
| `mlx-community/Qwen3-ASR-0.6B-8bit` | Qwen3-ASR 0.6B, 8-bit quantized (faster, lighter) |
| `mlx-community/parakeet-tdt-0.6b-v3` | Parakeet TDT 0.6B v3 (multilingual) |

Supported languages for Qwen3-ASR (mlx): Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Portuguese, Russian.

#### Qwen3-ASR Model Selection (Linux)

The `qwen` engine supports models via `--model` / `-m`:

| Model | Description |
|-------|-------------|
| `Qwen/Qwen3-ASR-1.7B` | Qwen3-ASR 1.7B **(default)** |
| `Qwen/Qwen3-ASR-0.6B` | Qwen3-ASR 0.6B (faster, lighter) |

The `qwen` engine supports 52 languages with auto-detection. Specify a language with `-l` (e.g. `zh`, `en`, `ja`, `ko`, `de`, `fr`, `es`, `ar`, `hi`, `th`, `vi`, etc.).

#### FunASR Model Selection

The `funasr` engine supports models via `--model` / `-m`:

| Model | Description |
|-------|-------------|
| `FunAudioLLM/SenseVoiceSmall` | SenseVoice Small, 234M params, multilingual **(default)** |
| `FunAudioLLM/Fun-ASR-Nano-2512` | Fun-ASR-Nano, 800M params, 31 languages, lyric recognition |

SenseVoiceSmall supports ASR, language identification, speech emotion recognition, and audio event detection. Fun-ASR-Nano supports 31 languages with mixed-language recognition and regional accent support.

The `funasr` engine automatically uses CUDA when available, otherwise falls back to CPU.

> **Note on MPS (Apple Silicon GPU)**: FunASR does not officially support MPS. While a source-code patch can enable MPS with ~2.4x speedup, this requires modifying FunASR internals. On macOS, use the `mlx` engine instead for native Apple Silicon acceleration.

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

By default, downloaded subtitles are simplified to plain text (SRT/VTT timestamps and sequence numbers are stripped). Use `--raw-transcript` to preserve the original subtitle formatting:

```bash
# Default: clean plain text output
aimd "https://www.youtube.com/watch?v=..."

# Preserve original SRT/VTT formatting
aimd "https://www.youtube.com/watch?v=..." --raw-transcript
```

#### Authenticated Access

For premium or age-restricted content, you may need to export your browser cookies:

```bash
# Export Chrome cookies to a file
yt-dlp --cookies-from-browser chrome --cookies cookies.txt

# Use cookies file with aimd
aimd "https://youtube.com/watch?v=..." --cookies cookies.txt

# Or read cookies directly from browser profile
aimd "https://www.bilibili.com/video/BV..." --cookies-from-browser "chrome:default"
```

> **Note**: Cookie files are in Netscape format and can also be created using browser extensions like "Get cookies.txt" (Chrome) or "Cookie-Editor" (Firefox).

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

When processing EPUB files, aimd:
- Reads the EPUB **spine** for correct chapter ordering (falls back to alphabetical if spine is unavailable)
- Converts each chapter via pandoc (`-f html -t markdown_mmd-raw_html --wrap=none`)
- Applies post-processing cleanup (heading normalisation, footnote conversion, image path fixup, TOC flattening)
- Extracts all images into a flat `images/` directory

Output layout:

```
book_name/
├── book_name.md      # Combined content (chapters separated by ---)
├── chapters/
│   ├── intro.md      # Named after original HTML stems
│   ├── chapter01.md
│   └── ...
└── images/
    └── *.jpg, *.png, etc.
```

## HTTP API (FastAPI)

Run the API service:

```bash
aimd-api
# or: uv run uvicorn aimd.api:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

Inspect transcription engine availability (preflight):

```bash
curl http://127.0.0.1:8000/v1/engines
```

Process any supported input:

```bash
curl -X POST http://127.0.0.1:8000/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "input_source": "https://www.youtube.com/watch?v=I3WUiD8HYn8",
    "transcribe_engine": "auto",
    "language": "en"
  }'
```

OpenAPI docs are available at:
- `/docs`
- `/redoc`

## MCP Server

Run an MCP server over stdio:

```bash
aimd-mcp
```

Available MCP tools:
- `healthz`
- `list_engines`
- `process_input`

`process_input` mirrors the API/CLI behavior and supports:
- `input_source`
- `transcribe_engine`
- `model` (mlx-audio model path, qwen-asr model, or FunASR model)
- `language`
- `output_file`
- `save_original`
- `cookies`
- `raw_transcript` (preserve original subtitle formatting, default: `false`)

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

# Optional: custom temporary directory for intermediate files
# Useful for sandboxed environments where /tmp is not writable
export AIMD_TEMP_DIR="/path/to/writable/tmp"
```

### Sandboxed Environments

By default, aimd uses the system temporary directory (`/tmp`) for intermediate files such as downloaded audio and EPUB extraction. In sandboxed environments where `/tmp` may not be writable, you can redirect temp I/O:

```bash
# Via environment variable (works for CLI, MCP, and HTTP API)
export AIMD_TEMP_DIR="/path/to/writable/tmp"

# Or via CLI option
aimd audio.mp3 --temp-dir /path/to/writable/tmp
```

The directory will be created automatically if it does not exist.

## Development

### Setup Development Environment

```bash
# Clone from the official repository
git clone https://github.com/shuuul/aimd.git
cd aimd

# Install dependencies
uv sync --dev --upgrade --all-extras

# Run code quality checks
uv run ruff check --fix && uv run ruff format
uv run prek --all-files
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
│   ├── cli.py                    # CLI entrypoint
│   ├── api.py                    # FastAPI entrypoint
│   ├── mcp.py                    # MCP entrypoint
│   ├── errors.py                 # Domain error types
│   ├── const.py                  # Constants (extensions, engines, languages)
│   ├── utils.py                  # URL/file utility helpers
│   ├── types.py                  # TextContext model
│   ├── application/              # Use-cases and dependency wiring
│   ├── infrastructure/           # Concrete processing implementations
│   └── adapters/                 # CLI/API/MCP interface adapters
├── tests/
├── docs/
├── pyproject.toml
└── AGENTS.md
```

## Architecture

### Core Components

- **Application Layer**: `application/use_cases/*` owns orchestration and flow decisions.
- **Infrastructure Layer**: `infrastructure/*` contains integrations (yt-dlp, pandoc, transcription runtimes).
- **Adapter Layer**: `adapters/*` maps CLI/API/MCP inputs to use-cases and maps outputs back.
- **Typed Error Contract**: `AimdError` subclasses provide consistent interface behavior.

### Processing Pipelines

#### Audio/Video Processing

1. **Adapter Input Mapping**: CLI/API/MCP request -> `ProcessInput`
2. **Use-case Orchestration**: task type detection and transcript/convert routing
3. **Infra Execution**: engine preflight + subtitle/audio/document extraction
4. **Output Mapping**: `ProcessResult` serialized to CLI text/API JSON/MCP response

#### Document Conversion

1. **Format Detection**: extension + supported format checks
2. **Pandoc Conversion**: source document -> markdown (`markdown_mmd-raw_html`, `--wrap=none`)
3. **Title Extraction**: normalized title resolution from content
4. **Chunking / EPUB Layout**: markdown splitting and EPUB spine-ordered chapter/image extraction with post-processing cleanup

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
