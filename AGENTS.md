# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Development Commands

**Environment Setup:**

```bash
# Install dependencies and setup development environment
uv sync --all-packages --all-extras --dev

# Activate virtual environment (if needed)
source .venv/bin/activate
```

**Code Quality:**

```bash
# Lint and format code
uv run ruff check --fix && uv run ruff format

# Run pre-commit checks
uv run pre-commit run --all-files
```

**Version Management:**

```bash
# Bump version (major, minor, patch)
uv version --bump patch
```

**Testing and Usage:**

```bash
# Run the CLI tool
aimd --help

# Process any input - auto-detects type
aimd audio.mp3                    # Transcribe audio
aimd "https://www.youtube.com/watch?v=VIDEO_ID"  # Extract subtitles
aimd book.epub                    # Convert to markdown
aimd document.txt                 # Convert text file

# With options
aimd audio.mp3 -o output.md       # Custom output
aimd audio.mp3 -e mlx             # Specify engine (mlx, yap, cuda, cpu)
aimd interview.wav -l zh_CN       # Specify locale
```

## Architecture Overview

### Core Components

**Context Preparation Tool**: A unified CLI tool that automatically detects input type and processes accordingly.

**Auto-Detection Logic** (`cli.py:_get_task_type`):
- URL → Transcript extraction
- Audio/Video file → Transcription
- Document file → Pandoc conversion

**Tool-Based Architecture**: Specialized tools in `tool/` directory:

- `tool/file.py`: Document processing with Pandoc
- `tool/audio.py`: Multi-engine audio transcription
- `tool/url.py`: Video platform content extraction using yt-dlp

**Core Utilities** (`utils.py`):

- `save_result()`: File output handling
- `sanitize_filename()`: Create safe filenames
- `create_output_path_from_title()`: Generate output paths
- `is_url()` / `is_supported_url()`: URL validation

### Key Patterns

**Async-First Design**: All core processing functions are async.

**TextContext**: Pydantic model carrying processing context:
- `title`: Document/video title
- `chunk_list`: List of text chunks
- `split_header_level`: Header level used for splitting

**CLI Interface**: Single command `aimd` with auto-detection.

## Important Details

**Output File Handling**: When no output specified, `create_output_path_from_title()` generates filenames with template suffix (`.md` extension). For video URLs, the video title is extracted and sanitized.

**Video Platform Support** (`tool/url.py`):

- **Supported**: YouTube, Bilibili, 1000+ platforms via yt-dlp
- **Content**: Metadata, subtitles, audio streams
- **No authentication** required for most content

**Audio Transcription** (`tool/audio.py`):

- **Formats**: MP3, WAV, M4A, FLAC, OGG, AAC, MP4
- **Engines**:
  - `auto`: Platform-optimized selection
  - `yap`: macOS only, requires yap CLI
  - `mlx`: Apple Silicon (M1-M4), uses mlx-whisper
  - `cuda`: NVIDIA GPU with CUDA 12
  - `cpu`: Cross-platform fallback (faster-whisper)
- **Languages**: Auto-detection or manual locale

**Document Conversion** (`tool/file.py`):

- **Formats**: EPUB, PDF, TXT, MD, 40+ others via Pandoc
- **Features**: Title extraction, automatic chunking for large files
- **Output**: Clean markdown optimized for LLM context

**EPUB Processing** (`tool/file.py`):

- **EPUB to Markdown**: Extracts images and chapter files from EPUB
- **Output Structure**:
  ```
  book_name/
  ├── book_name.md      # Combined content from all chapters
  ├── chapters/
  │   ├── chapter_001.md
  │   ├── chapter_002.md
  │   └── ...
  └── images/
      └── *.jpg, *.png, *.svg, etc.
  ```

**Configuration** (optional env vars):

- `YT_DLP_CONFIG_HOME`: Custom yt-dlp config
- `YT_DLP_CACHE_DIR`: Custom yt-dlp cache
- `YAP_MODEL_PATH`: Custom yap model (macOS)

**Error Handling**:

- Runtime engine compatibility checks
- Graceful fallbacks between engines
- Cached cookie jar failures
