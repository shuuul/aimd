<div align="center">
  <img src="assets/aimd-banner-sm.png" alt="aimd">

  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white" alt="Python 3.10, 3.11, 3.12">
  </a>
  <a href="https://docs.astral.sh/uv/">
    <img src="https://img.shields.io/badge/uv-package-654FF0?logo=uv&logoColor=white" alt="uv package">
  </a>
  <a href="https://github.com/shuuul/aimd/actions/workflows/ci.yml">
    <img src="https://github.com/shuuul/aimd/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://github.com/shuuul/aimd/actions/workflows/release.yml">
    <img src="https://github.com/shuuul/aimd/actions/workflows/release.yml/badge.svg" alt="Release">
  </a>
  <a href="https://github.com/shuuul/aimd/releases">
    <img src="https://img.shields.io/badge/version-0.10.0-blue" alt="Version 0.10.0">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  </a>
</div>

# aimd

Prepare LLM-ready context from URLs, audio/video, and documents.

`aimd` gives you one command that auto-detects input type, extracts or transcribes the content, converts it to Markdown, and returns a chunked text context suitable for downstream AI workflows.

## Highlights

- **One input command** for URLs, audio/video files, ebooks, PDFs, Markdown, text, and other MarkItDown-supported documents.
- **Media extraction** through bundled `aimd.media`: yt-dlp URLs such as podcasts, YouTube, Bilibili, and local audio/video files.
- **Subtitle-first fallback**: download subtitles when available; otherwise download audio and transcribe with `mlx-audio` or Qwen3-ASR through Transformers.
- **Document conversion** through MarkItDown, with dedicated ebook chapter/image extraction in the bundled `aimd.book` plugin.
- **Three interfaces**: CLI (`aimd`), HTTP API (`aimd-api`), and MCP server (`aimd-mcp`).

> OCR for scanned PDFs and images is planned. The `aimd.ocr` module scaffold is bundled in the single distribution so OCR can land without growing the transcript or document conversion code paths.

## Install

Install the CLI after release:

```bash
uv tool install aimd-tool
aimd --help
```

Install from GitHub `main` before a release:

```bash
uv tool install --force \
  "aimd-tool @ git+https://github.com/shuuul/aimd.git@main"
```

Install the full tool set from GitHub, including the API and MCP runtime dependencies:

```bash
uv tool install --force \
  "aimd-tool[all] @ git+https://github.com/shuuul/aimd.git@main"
```

Install API/MCP dependencies from PyPI when needed:

```bash
# HTTP API + MCP server
uv tool install "aimd-tool[all]"

# Or install into an existing Python environment
uv pip install "aimd-tool[all]"
```

The public PyPI release is intentionally a single distribution (`aimd-tool`). The installed command remains `aimd`.

From a source checkout, use the project directly:

```bash
git clone https://github.com/shuuul/aimd.git
cd aimd
uv sync --dev
uv run aimd --help
```

Platform notes:

- macOS transcription is optimized for Apple Silicon through `mlx-audio`.
- Linux transcription uses Qwen3-ASR through the Transformers backend and requires a CUDA-capable GPU.
- Local file conversion is powered by MarkItDown. Ebook conversion is handled by the bundled `aimd.book` MarkItDown plugin; today it supports EPUB-compatible ZIP/spine books and still shells out to the Pandoc CLI for chapter HTML conversion.

## Quick start

```bash
# Auto-detect input type
aimd audio.mp3
aimd "https://youtube.com/watch?v=..."
aimd book.epub
aimd notes.txt

# Common options
aimd audio.mp3 --output transcript.md
aimd audio.wav --engine mlx --language zh
aimd "https://youtube.com/watch?v=..." --cookies-from-browser chrome
aimd "https://youtube.com/watch?v=..." --raw-transcript
```

## CLI usage

### Audio and video files

```bash
aimd interview.m4a
aimd lecture.mp3 --engine auto
aimd audio.wav --engine mlx
aimd audio.wav --engine qwen
aimd audio.wav --model mlx-community/Qwen3-ASR-1.7B-4bit
```

Engines:

| Engine | Platform | Notes |
|--------|----------|-------|
| `auto` | macOS/Linux | Selects the best available backend. |
| `mlx` | Apple Silicon | Uses `mlx-audio`; default local backend on macOS when available. |
| `qwen` | Linux/CUDA | Uses Qwen3-ASR through Transformers; default local backend on Linux when available. |

### URLs

```bash
aimd "https://www.youtube.com/watch?v=..."
aimd "https://www.bilibili.com/video/BV..."
aimd "https://www.xiaoyuzhoufm.com/episode/..."
```

Subtitles are simplified to plain text by default. Use `--raw-transcript` to preserve SRT/VTT formatting.

For authenticated or restricted content:

```bash
aimd "https://youtube.com/watch?v=..." --cookies cookies.txt
aimd "https://www.bilibili.com/video/BV..." --cookies-from-browser "chrome:default"
```

### Documents

```bash
aimd book.epub
aimd book.mobi
aimd book.azw3
aimd document.pdf
aimd notes.md
aimd document.epub --output output.md
```

Book files are expanded into a structured output directory:

```text
book_name/
├── book_name.md
├── chapters/
└── images/
```

The book pipeline preserves spine order, extracts images, converts chapters through Pandoc, and applies Markdown cleanup.

Current note: `aimd.book` owns the ebook converter and routes `.epub`, `.mobi`, and `.azw3` as book inputs. The implemented extraction pipeline is EPUB-compatible; non-EPUB ebook formats may still need a later format-specific conversion step before the same cleanup pipeline can succeed.

### MarkItDown plugins

`aimd` uses MarkItDown for local files with plugins enabled. The bundled `aimd.media` and `aimd.book` modules register standard `markitdown.plugin` entry points, so installing `aimd-tool` also installs the ASR and ebook converters.

You can also use the plugins from MarkItDown directly:

```bash
markitdown --list-plugins
markitdown --use-plugins book.epub -o book.md
markitdown --use-plugins audio.mp3 -o transcript.md
```

## HTTP API

Run the API server:

```bash
aimd-api
```

Endpoints:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/engines

curl -X POST http://127.0.0.1:8000/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "input_source": "https://www.youtube.com/watch?v=...",
    "transcribe_engine": "auto",
    "language": "en"
  }'
```

OpenAPI docs are available at `/docs` and `/redoc`.

## MCP server

Run the stdio MCP server:

```bash
aimd-mcp
```

Tools:

- `healthz`
- `list_engines`
- `process_input`

`process_input` mirrors the CLI/API flow and accepts options such as `input_source`, `transcribe_engine`, `model`, `language`, `output_file`, `save_original`, `cookies`, `cookies_from_browser`, and `raw_transcript`. For MCP, temporary files are controlled by the `AIMD_TEMP_DIR` environment variable.

## Configuration

```bash
# Optional yt-dlp configuration
export YT_DLP_CONFIG_HOME="/path/to/config"
export YT_DLP_CACHE_DIR="/path/to/cache"

# Optional temporary directory for downloads, transcoding, and ebook extraction
export AIMD_TEMP_DIR="/path/to/writable/tmp"
```

You can also pass `--temp-dir` on the CLI.

## Development

```bash
uv sync --dev --upgrade
uv run prek --all-files
uv run pytest -q
```

Useful maintenance commands:

```bash
uv run prek autoupdate
uv build
uv version --bump patch
```

Release a tagged version:

```bash
# Make sure the package version matches the tag, then push a v-prefixed tag.
git tag v0.10.0
git push origin v0.10.0
```

The release workflow builds the single `aimd-tool` distribution, smoke-installs the tool on Linux and macOS, creates a GitHub Release, and publishes that distribution to PyPI using the `UV_PUBLISH_TOKEN` repository secret.

Project layout:

```text
src/
└── aimd/            # Single aimd-tool package: CLI, API, MCP, media, book, OCR/clip modules
```

## Architecture

`aimd` is a single published distribution, `aimd-tool`. The package uses MarkItDown as the local-file conversion contract and follows a ports/adapters layout:

- `application` owns orchestration and task routing.
- `process_input.py` acts as the facade/router; local file processors call `MarkItDown(enable_plugins=True)`.
- Bundled modules register MarkItDown plugins: `aimd.media` for local audio/video ASR and `aimd.book` for ebooks. `aimd.ocr` is the next plugin scaffold, and `aimd.clip` wraps Defuddle-backed HTML extraction.
- `aimd.media` owns URL media extraction: yt-dlp metadata, subtitle download, cookie handling, audio download fallback, and ASR.
- `aimd.core.infrastructure` wraps media/MarkItDown markdown results into `TextContext` chunks.
- CLI/API/MCP modules translate interface requests into application use-cases. Output file persistence is adapter-owned and shared through `aimd.core.application.services.output_writer`; it is not part of `ProcessInput`.

All processing returns a shared `TextContext` shape: title, chunks, and split metadata.

## License

MIT. See [LICENSE](LICENSE).
