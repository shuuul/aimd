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
    <img src="https://img.shields.io/badge/version-0.10.2-blue" alt="Version 0.10.2">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  </a>
</div>

# aimd

Prepare LLM-ready context from URLs, audio/video, and documents.

`aimd` gives you one command that auto-detects input type, extracts or transcribes the content, converts it to Markdown, and returns a chunked text context suitable for downstream AI workflows.

## Highlights

- **One input command** for URLs, audio/video files, ebooks, PDFs, scanned PDFs/images, Markdown, text, and other MarkItDown-supported documents.
- **Media extraction** through bundled `aimd.media`: yt-dlp URLs such as podcasts, YouTube, and Bilibili.
- **ASR transcription** through bundled `aimd.asr`: local audio/video transcription with `mlx-audio` on Apple Silicon or Qwen3-ASR on Linux/CUDA.
- **Subtitle-first fallback**: download subtitles when available; otherwise download audio and transcribe with `mlx-audio` or Qwen3-ASR through Transformers.
- **Document conversion** through MarkItDown, with dedicated ebook chapter/image extraction in the bundled `aimd.book` plugin.
- **OCR task** for scanned PDFs and images, with `mlx4ocr` on macOS/Apple Silicon and CUDA VLM OCR models through Transformers on Linux.
- **Three interfaces**: CLI (`aimd`), HTTP API (`aimd-api`), and MCP server (`aimd-mcp`).

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
- macOS OCR uses `mlx4ocr` on Python 3.12+ and downloads OCR model weights on first use.
- Linux transcription uses Qwen3-ASR through the Transformers backend and requires a CUDA-capable GPU.
- Linux OCR uses the Transformers backend with CUDA.
- Local file conversion is powered by MarkItDown. Ebook conversion is handled by the bundled `aimd.book` MarkItDown plugin; today it supports EPUB-compatible ZIP/spine books and still shells out to the Pandoc CLI for chapter HTML conversion.

## Quick start

```bash
# Auto-detect input type
aimd audio.mp3
aimd "https://youtube.com/watch?v=..."
aimd book.epub
aimd notes.txt
aimd scan.pdf
aimd page.png

# Common options
aimd audio.mp3 --output transcript.md
aimd audio.wav --engine mlx --language zh
aimd scan.pdf --engine mlx4ocr --model paddleocr_v6 --start 0 --end 2
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

### Supported models

`--model` is interpreted by the selected task engine. The table below lists the model aliases and model IDs that `aimd` validates directly. Linux/CUDA OCR also accepts an explicit Hugging Face model ID, but only the listed aliases have model-specific handling.

| Task | Engine | Platform | `--model` value | Upstream model / runtime | Default | Notes |
|------|--------|----------|-----------------|--------------------------|---------|-------|
| Transcription | `mlx` | macOS Apple Silicon | `mlx-community/Qwen3-ASR-1.7B-4bit` | mlx-audio STT | Yes | Qwen3-ASR 1.7B, 4-bit quantized. |
| Transcription | `mlx` | macOS Apple Silicon | `mlx-community/Qwen3-ASR-1.7B-6bit` | mlx-audio STT | No | Qwen3-ASR 1.7B, 6-bit quantized. |
| Transcription | `mlx` | macOS Apple Silicon | `mlx-community/Qwen3-ASR-1.7B-8bit` | mlx-audio STT | No | Qwen3-ASR 1.7B, 8-bit quantized. |
| Transcription | `mlx` | macOS Apple Silicon | `mlx-community/Qwen3-ASR-0.6B-4bit` | mlx-audio STT | No | Qwen3-ASR 0.6B, 4-bit quantized. |
| Transcription | `mlx` | macOS Apple Silicon | `mlx-community/Qwen3-ASR-0.6B-6bit` | mlx-audio STT | No | Qwen3-ASR 0.6B, 6-bit quantized. |
| Transcription | `mlx` | macOS Apple Silicon | `mlx-community/Qwen3-ASR-0.6B-8bit` | mlx-audio STT | No | Qwen3-ASR 0.6B, 8-bit quantized. |
| Transcription | `mlx` | macOS Apple Silicon | `mlx-community/whisper-large-v3-turbo-asr-fp16` | mlx-audio STT | No | Whisper large-v3-turbo ASR, fp16. |
| Transcription | `mlx` | macOS Apple Silicon | `distil-whisper/distil-large-v3` | mlx-audio STT | No | Distil-Whisper large-v3. |
| Transcription | `mlx` | macOS Apple Silicon | `mlx-community/parakeet-tdt-0.6b-v3` | mlx-audio STT | No | NVIDIA Parakeet TDT 0.6B v3. |
| Transcription | `mlx` | macOS Apple Silicon | `mlx-community/nemotron-3.5-asr-streaming-0.6b` | mlx-audio STT | No | NVIDIA Nemotron 3.5 ASR streaming 0.6B. |
| Transcription | `mlx` | macOS Apple Silicon | `mlx-community/Voxtral-Mini-3B-2507-bf16` | mlx-audio STT | No | Voxtral Mini 3B, bf16. |
| Transcription | `mlx` | macOS Apple Silicon | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` | mlx-audio STT | No | Voxtral Mini 4B Realtime, 4-bit. |
| Transcription | `mlx` | macOS Apple Silicon | `mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16` | mlx-audio STT | No | Voxtral Mini 4B Realtime, fp16. |
| Transcription | `mlx` | macOS Apple Silicon | `mlx-community/VibeVoice-ASR-bf16` | mlx-audio STT | No | VibeVoice-ASR, bf16; upstream model may include diarization/timestamps. |
| Transcription | `mlx` | macOS Apple Silicon | `mlx-community/Qwen2-Audio-7B-Instruct-4bit` | mlx-audio STT | No | Qwen2-Audio 7B Instruct, 4-bit. |
| Transcription | `qwen` | Linux/CUDA | `Qwen/Qwen3-ASR-1.7B` | Transformers | Yes | Default Linux/CUDA ASR model. |
| Transcription | `qwen` | Linux/CUDA | `Qwen/Qwen3-ASR-0.6B` | Transformers | No | Lower-memory Qwen3-ASR option. |
| OCR | `mlx4ocr` | macOS Apple Silicon | `paddleocr_v6`, `ppocrv6`, `pp_ocrv6` | mlx4ocr `ppocrv6` | Yes | Uses the `medium` PP-OCRv6 variant by default. |
| OCR | `mlx4ocr` | macOS Apple Silicon | `tiny`, `small`, `medium` | mlx4ocr `ppocrv6` | No | Explicit PP-OCRv6 variants. |
| OCR | `mlx4ocr` | macOS Apple Silicon | `glm_ocr` | mlx4ocr `glm-ocr` | No | Optional mlx4ocr VLM backend. |
| OCR | `mlx4ocr` | macOS Apple Silicon | `paddleocr_vl` | mlx4ocr `paddleocr-vl` | No | Optional mlx4ocr VLM backend. |
| OCR | `transformers` | Linux/CUDA | `got_ocr`, `got-ocr`, `got_ocr2`, `got-ocr2`, `stepfun-ai/GOT-OCR-2.0-hf` | `stepfun-ai/GOT-OCR-2.0-hf` | Yes | Default Linux/CUDA OCR model. |
| OCR | `transformers` | Linux/CUDA | `unlimited_ocr`, `unlimited-ocr`, `baidu/Unlimited-OCR` | `baidu/Unlimited-OCR` | No | Uses Baidu Unlimited-OCR remote code with CUDA and `save_results=True`. |
| OCR | `transformers` | Linux/CUDA | `glm_ocr`, `glm-ocr`, `zai-org/GLM-OCR` | `zai-org/GLM-OCR` | No | May require a newer Transformers build than the PyPI baseline. |
| OCR | `transformers` | Linux/CUDA | `paddleocr_vl`, `paddleocr-vl`, `PaddlePaddle/PaddleOCR-VL-1.5` | `PaddlePaddle/PaddleOCR-VL-1.5` | No | May require optional runtime packages expected by upstream model code. |

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

### OCR for scanned PDFs and images

```bash
aimd page.png
aimd scan.pdf                                  # OCR if no extractable PDF text is found
aimd scan.pdf --engine mlx4ocr                 # macOS/Apple Silicon
aimd scan.pdf --engine transformers --model got_ocr       # Linux/CUDA VLM OCR
aimd scan.pdf --engine transformers --model unlimited_ocr # Linux/CUDA Baidu Unlimited-OCR
aimd scan.pdf --model paddleocr_v6             # default PP-OCRv6 detector/recognizer
aimd scan.pdf --model glm_ocr                  # optional mlx4ocr VLM backend
aimd scan.pdf --model paddleocr_vl             # optional mlx4ocr VLM backend
aimd scan.pdf --start 0 --end 2                # 0-based inclusive OCR PDF page range
```

OCR keeps the same Markdown/TextContext output contract as transcript and convert tasks. Images route to OCR automatically. PDFs with an extractable text layer route to normal document conversion; scanned PDFs route to OCR when the local PDF text-layer check is available. On macOS, OCR `auto` resolves to `mlx4ocr`, and the default OCR model is `paddleocr_v6` (mapped to mlx4ocr `ppocrv6` with the `medium` variant). `glm_ocr` and `paddleocr_vl` require mlx4ocr's optional VLM dependencies. On Linux, OCR `auto` resolves to the CUDA Transformers backend. Its default is `got_ocr` (`stepfun-ai/GOT-OCR-2.0-hf`) because it works with the current PyPI Transformers release. `unlimited_ocr` maps to `baidu/Unlimited-OCR` and uses the model's custom `infer`/`infer_multi` API. `glm_ocr` maps to `zai-org/GLM-OCR` when a new-enough Transformers build is installed, and `paddleocr_vl` maps to `PaddlePaddle/PaddleOCR-VL-1.5` when its optional runtime requirements are present. Traditional PP-OCRv6/`paddleocr_v6` is intentionally not routed through Transformers. PDF OCR on Linux uses the system `pdftoppm` executable from poppler when available.

### MarkItDown plugins

`aimd` uses MarkItDown for local files with plugins enabled. The bundled `aimd.media` and `aimd.book` modules register standard `markitdown.plugin` entry points, so installing `aimd-tool` also installs the media/ASR and ebook converters. The media plugin delegates transcription to `aimd.asr`.

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

curl -X POST http://127.0.0.1:8000/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "input_source": "scan.pdf",
    "task_type": "ocr",
    "transcribe_engine": "mlx4ocr",
    "start": 0,
    "end": 2
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

`process_input` mirrors the CLI/API flow and accepts options such as `input_source`, `task_type`, `transcribe_engine`, `model`, `language`, `start`, `end`, `output_file`, `save_original`, `cookies`, `cookies_from_browser`, and `raw_transcript`. For MCP, temporary files are controlled by the `AIMD_TEMP_DIR` environment variable.

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

`aimd` is a single published distribution, `aimd-tool`. See [docs/architecture.md](docs/architecture.md) for the full architecture notes and [docs/performance.md](docs/performance.md) for performance expectations and measurement guidance.

The package uses MarkItDown as the local-file conversion contract and follows a ports/adapters layout:

- `application` owns orchestration and task routing.
- `process_input.py` acts as the facade/router; convert-task local file processors call `MarkItDown(enable_plugins=True)`, while OCR routes explicitly to `aimd.ocr`.
- Bundled modules register MarkItDown plugins: `aimd.media` for local audio/video inputs and `aimd.book` for ebooks. `aimd.ocr` provides the explicit OCR task for images and scanned PDFs, and `aimd.clip` wraps Defuddle-backed HTML extraction.
- `aimd.media` owns URL media extraction: yt-dlp metadata, subtitle download, cookie handling, and audio download fallback. `aimd.asr` owns transcription engines and model validation.
- `aimd.core.infrastructure` wraps media/MarkItDown markdown results into `TextContext` chunks.
- CLI/API/MCP modules translate interface requests into application use-cases. Output file persistence is adapter-owned and shared through `aimd.core.application.services.output_writer`; it is not part of `ProcessInput`.

All processing returns a shared `TextContext` shape: title, chunks, and split metadata.

## License

MIT. See [LICENSE](LICENSE).
