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
    <img src="https://img.shields.io/badge/version-0.12.1-blue" alt="Version 0.12.1">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  </a>
</div>

# aimd

Prepare LLM-ready context from URLs, audio/video, and documents.

`aimd` gives you one command that auto-detects input type, extracts or transcribes the content, converts it to Markdown, and returns a chunked text context suitable for downstream AI workflows.

## Highlights

- **One input command** for URLs, audio/video files, EPUB documents, PDFs, scanned PDFs/images, Markdown, text, and other MarkItDown-supported documents.
- **URL extraction** through bundled `aimd.plugins.url`: yt-dlp transcript URLs such as podcasts, YouTube, and Bilibili, plus opt-in readable HTML extraction through Defuddle.
- **ASR transcription** through bundled `aimd.plugins.asr`: local audio/video transcription with `mlx-audio` by default on Apple Silicon, or Qwen3-ASR through native Transformers (`transformers>=5.13`) on CUDA-capable non-Darwin platforms and as an explicit opt-in model path on macOS.
- **Subtitle-first fallback**: download subtitles when available; otherwise download audio and transcribe with `mlx-audio` or Qwen3-ASR through Transformers.
- **Document conversion** through MarkItDown, with dedicated EPUB chapter/image extraction in the bundled `aimd.plugins.doc` plugin.
- **OCR task** for scanned PDFs and images, with `mlx-vlm` on macOS/Apple Silicon and CUDA VLM OCR models through Transformers on Linux.
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

- macOS transcription is optimized for Apple Silicon through `mlx-audio`; this remains the default backend.
- macOS OCR uses `mlx-vlm` on Python 3.12+ and downloads OCR model weights on first use.
- Linux transcription uses Qwen3-ASR through the Transformers backend and requires a CUDA-capable GPU.
- Explicit `Qwen/Qwen3-ASR-*-hf` (or legacy `Qwen/Qwen3-ASR-*`) transcription model IDs use native Transformers Qwen3-ASR (`transformers>=5.13`); on macOS this is an opt-in MPS path, not the default.
- Linux OCR uses the Transformers backend with CUDA.
- Local file conversion is powered by MarkItDown. Pandoc-backed document conversion is handled by the bundled `aimd.plugins.doc` MarkItDown plugin; EPUB uses a custom ZIP/spine pipeline for stable chapter ordering and image extraction, while other Pandoc-supported formats go through the Pandoc CLI directly.

## Quick start

```bash
# Auto-detect input type
aimd audio.mp3
aimd "https://youtube.com/watch?v=..."
aimd document.epub
aimd notes.txt
aimd scan.pdf
aimd page.png

# Common options
aimd audio.mp3 --output transcript.md
aimd audio.wav --language zh
aimd scan.pdf --model glm_ocr --start 0 --end 2
aimd "https://youtube.com/watch?v=..." --cookies-from-browser chrome
aimd "https://youtube.com/watch?v=..." --raw-transcript
```

## CLI usage

### Audio and video files

```bash
aimd interview.m4a
aimd lecture.mp3
aimd audio.wav --language zh
aimd audio.wav --model mlx-community/Qwen3-ASR-1.7B-4bit
```

Backend selection is automatic:

| Platform | Backend | Notes |
|--------|----------|-------|
| macOS Apple Silicon | MLX | Uses `mlx-audio` when available. |
| CUDA-capable non-Darwin | Transformers | Uses Qwen3-ASR through Transformers when CUDA is available. |

Passing an explicit transcription model ID such as `Qwen/Qwen3-ASR-0.6B-hf` (or the legacy alias `Qwen/Qwen3-ASR-0.6B`) opts into the native Transformers Qwen3-ASR backend, including on macOS/MPS. The automatic macOS backend remains MLX because the quantized MLX models are usually faster and lower-maintenance on Apple Silicon.

### Supported models

`--model` is interpreted by the platform-selected backend. The table below lists the model aliases and model IDs that `aimd` validates directly. Linux/CUDA OCR also accepts an explicit Hugging Face model ID, but only the listed aliases have model-specific handling.

| Task | Platform/backend | `--model` value | Upstream model / runtime | Default | Notes |
|------|------------------|-----------------|--------------------------|---------|-------|
| Transcription | macOS Apple Silicon / MLX | `mlx-community/Qwen3-ASR-1.7B-4bit` | mlx-audio STT | Yes | Qwen3-ASR 1.7B, 4-bit quantized. |
| Transcription | macOS Apple Silicon / MLX | `mlx-community/Qwen3-ASR-1.7B-6bit` | mlx-audio STT | No | Qwen3-ASR 1.7B, 6-bit quantized. |
| Transcription | macOS Apple Silicon / MLX | `mlx-community/Qwen3-ASR-1.7B-8bit` | mlx-audio STT | No | Qwen3-ASR 1.7B, 8-bit quantized. |
| Transcription | macOS Apple Silicon / MLX | `mlx-community/Qwen3-ASR-0.6B-4bit` | mlx-audio STT | No | Qwen3-ASR 0.6B, 4-bit quantized. |
| Transcription | macOS Apple Silicon / MLX | `mlx-community/Qwen3-ASR-0.6B-6bit` | mlx-audio STT | No | Qwen3-ASR 0.6B, 6-bit quantized. |
| Transcription | macOS Apple Silicon / MLX | `mlx-community/Qwen3-ASR-0.6B-8bit` | mlx-audio STT | No | Qwen3-ASR 0.6B, 8-bit quantized. |
| Transcription | macOS Apple Silicon / MLX | `mlx-community/whisper-large-v3-turbo-asr-fp16` | mlx-audio STT | No | Whisper large-v3-turbo ASR, fp16. |
| Transcription | macOS Apple Silicon / MLX | `distil-whisper/distil-large-v3` | mlx-audio STT | No | Distil-Whisper large-v3. |
| Transcription | macOS Apple Silicon / MLX | `mlx-community/parakeet-tdt-0.6b-v3` | mlx-audio STT | No | NVIDIA Parakeet TDT 0.6B v3. |
| Transcription | macOS Apple Silicon / MLX | `mlx-community/nemotron-3.5-asr-streaming-0.6b` | mlx-audio STT | No | NVIDIA Nemotron 3.5 ASR streaming 0.6B. |
| Transcription | macOS Apple Silicon / MLX | `mlx-community/Voxtral-Mini-3B-2507-bf16` | mlx-audio STT | No | Voxtral Mini 3B, bf16. |
| Transcription | macOS Apple Silicon / MLX | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` | mlx-audio STT | No | Voxtral Mini 4B Realtime, 4-bit. |
| Transcription | macOS Apple Silicon / MLX | `mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16` | mlx-audio STT | No | Voxtral Mini 4B Realtime, fp16. |
| Transcription | macOS Apple Silicon / MLX | `mlx-community/VibeVoice-ASR-bf16` | mlx-audio STT | No | VibeVoice-ASR, bf16; upstream model may include diarization/timestamps. |
| Transcription | macOS Apple Silicon / MLX | `mlx-community/Qwen2-Audio-7B-Instruct-4bit` | mlx-audio STT | No | Qwen2-Audio 7B Instruct, 4-bit. |
| Transcription | CUDA-capable non-Darwin / Transformers; explicit opt-in on macOS/MPS | `Qwen/Qwen3-ASR-1.7B-hf` | Native Transformers Qwen3-ASR | Yes on CUDA Transformers | Default CUDA Transformers ASR model; explicit opt-in on macOS. Legacy `Qwen/Qwen3-ASR-1.7B` resolves here. |
| Transcription | CUDA-capable non-Darwin / Transformers; explicit opt-in on macOS/MPS | `Qwen/Qwen3-ASR-0.6B-hf` | Native Transformers Qwen3-ASR | No | Lower-memory Qwen3-ASR option; explicit opt-in on macOS. Legacy `Qwen/Qwen3-ASR-0.6B` resolves here. |
| OCR | macOS Apple Silicon / mlx-vlm | `glm_ocr`, `glm-ocr`, `mlx-community/GLM-OCR-bf16` | `mlx-community/GLM-OCR-bf16` | Yes | Default macOS MLX VLM OCR model. |
| OCR | macOS Apple Silicon / mlx-vlm | Explicit Hugging Face model ID | Provided model ID | No | Any local `mlx-vlm` compatible OCR/image-text model. |
| OCR | Linux/CUDA / Transformers | `got_ocr`, `got-ocr`, `got_ocr2`, `got-ocr2`, `stepfun-ai/GOT-OCR-2.0-hf` | `stepfun-ai/GOT-OCR-2.0-hf` | Yes | Default Linux/CUDA OCR model. |
| OCR | Linux/CUDA / Transformers | `unlimited_ocr`, `unlimited-ocr`, `baidu/Unlimited-OCR` | `baidu/Unlimited-OCR` | No | Uses Baidu Unlimited-OCR remote code with CUDA and `save_results=True`. |
| OCR | Linux/CUDA / Transformers | `glm_ocr`, `glm-ocr`, `zai-org/GLM-OCR` | `zai-org/GLM-OCR` | No | May require a newer Transformers build than the PyPI baseline. |

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

When no cookie option is provided, URL extraction may try available browser cookie sources before falling back to unauthenticated access. Keep this behavior because it makes restricted media work out of the box, but explicit cookie options should fail fast when invalid. Transcript requests should also fail when subtitles and audio transcription both produce no text, instead of returning metadata-only Markdown as a successful transcript.

### Documents

```bash
aimd document.epub
aimd document.pdf
aimd notes.md
aimd document.epub --output output.md
```

Asset-bearing documents such as EPUB are expanded into a structured output directory:

```text
document_name/
├── document_name.md
├── chapters/
└── images/
```

The EPUB pipeline preserves spine order, extracts images, converts chapters through Pandoc, and applies Markdown cleanup. Other Pandoc-supported formats are converted by Pandoc directly. Pandoc does not support MOBI/AZW3 as input formats, so AIMD no longer advertises them as supported document inputs.

### OCR for scanned PDFs and images

```bash
aimd page.png
aimd scan.pdf                                  # OCR if no extractable PDF text is found
aimd scan.pdf --model got_ocr                  # Linux/CUDA VLM OCR
aimd scan.pdf --model unlimited_ocr            # Linux/CUDA Baidu Unlimited-OCR
aimd scan.pdf --model glm_ocr                  # macOS mlx-vlm default, or Linux/CUDA GLM-OCR
aimd scan.pdf --start 0 --end 2                # 0-based inclusive OCR PDF page range
```

OCR keeps the same Markdown/TextContext output contract as transcript and convert tasks. Images route to OCR automatically. PDFs with an extractable text layer route to normal document conversion; scanned PDFs route to OCR when the local PDF text-layer check is available. macOS uses `mlx-vlm` directly, where the default OCR model is `glm_ocr` (`mlx-community/GLM-OCR-bf16`), and explicit `mlx-vlm` compatible Hugging Face model IDs are accepted. Linux/CUDA uses the Transformers backend. Its default is `got_ocr` (`stepfun-ai/GOT-OCR-2.0-hf`) because it works with the current PyPI Transformers release. `unlimited_ocr` maps to `baidu/Unlimited-OCR` and uses the model's custom `infer`/`infer_multi` API. `glm_ocr` maps to `zai-org/GLM-OCR` when a new-enough Transformers build is installed. PaddleOCR aliases are intentionally not supported. PDF OCR renders pages with PyMuPDF before model inference.

### MarkItDown plugins

`aimd` uses MarkItDown with plugins enabled. The bundled `aimd.plugins.url`, `aimd.plugins.asr`, `aimd.plugins.doc`, and `aimd.plugins.ocr` modules register standard `markitdown.plugin` entry points, so installing `aimd-tool` also installs URL transcript/readable HTML, local audio/video ASR, Pandoc document, and OCR converters.

You can also use the plugins from MarkItDown directly:

```bash
markitdown --list-plugins
markitdown --use-plugins document.epub -o doc.md
markitdown --use-plugins audio.mp3 -o transcript.md
```

The Defuddle-backed `aimd.plugins.url` readable HTML converter is opt-in so normal HTML conversion does not require Node.js/npm:

```python
from markitdown import MarkItDown

result = MarkItDown(enable_plugins=True).convert("article.html", defuddle=True)
print(result.markdown)
```

## HTTP API

Run the API server:

```bash
aimd-api
```

Endpoints:

```bash
curl http://127.0.0.1:8000/healthz

curl -X POST http://127.0.0.1:8000/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "input_source": "https://www.youtube.com/watch?v=...",
    "language": "en"
  }'

curl -X POST http://127.0.0.1:8000/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "input_source": "scan.pdf",
    "task_type": "ocr",
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
- `process_input`

`process_input` mirrors the CLI/API flow and accepts options such as `input_source`, `task_type`, `model`, `language`, `start`, `end`, `output_file`, `save_original`, `cookies`, `cookies_from_browser`, and `raw_transcript`. For MCP, temporary files are controlled by the `AIMD_TEMP_DIR` environment variable.

## Configuration

```bash
# Optional yt-dlp configuration
export YT_DLP_CONFIG_HOME="/path/to/config"
export YT_DLP_CACHE_DIR="/path/to/cache"

# Optional temporary directory for downloads, transcoding, and document extraction
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
git tag v0.12.1
git push origin v0.12.1
```

The release workflow builds the single `aimd-tool` distribution, smoke-installs the tool on Linux and macOS, creates a GitHub Release, and publishes that distribution to PyPI using the `UV_PUBLISH_TOKEN` repository secret.

Project layout:

```text
src/
└── aimd/            # core, interfaces/{cli,api,mcp,output.py}, and bundled MarkItDown plugins
```

## Architecture

`aimd` is a single published distribution, `aimd-tool`. See [docs/architecture.md](docs/architecture.md) for the full architecture notes and [docs/performance.md](docs/performance.md) for performance expectations and measurement guidance.

The package uses MarkItDown as the URL/local-file conversion contract and keeps core as a small interface-independent processing service:

- `aimd.core.models` and `aimd.core.process` own canonical request/response models, input routing, and processing; ASR/OCR backends are selected internally from the current platform.
- `aimd.core.process.process_input()` sends URL and local-file work through `MarkItDown(enable_plugins=True)`.
- Bundled modules register MarkItDown plugins: `aimd.plugins.url` for URL transcript extraction and opt-in Defuddle-backed HTML extraction, `aimd.plugins.asr` for local audio/video inputs, `aimd.plugins.doc` for Pandoc-backed documents, and `aimd.plugins.ocr` for explicit OCR of images and scanned PDFs.
- Console entry points are `aimd.interfaces.cli:main`, `aimd.interfaces.api:main`, and `aimd.interfaces.mcp.app:main`; MarkItDown plugin entry points are `aimd.plugins.asr`, `aimd.plugins.url`, `aimd.plugins.doc`, and `aimd.plugins.ocr`.
- `aimd.plugins.url` owns URL extraction: yt-dlp metadata, subtitle download, cookie handling, audio download fallback, and readable HTML extraction. It intentionally keeps automatic browser-cookie probing for convenience, while explicit cookie arguments and transcript/audio fallback failures should surface clear errors. `aimd.plugins.asr` owns transcription backend selection, model validation, and the local audio/video MarkItDown plugin.
- `aimd.core.process` wraps MarkItDown markdown results into `TextContext` chunks.
- CLI/API/MCP modules translate interface requests into core processing. Output file persistence is interface-owned and shared through `aimd.interfaces.output`; it is not part of `ProcessInput`.

All processing returns a shared `TextContext` shape: title, chunks, and split metadata.

## License

MIT. See [LICENSE](LICENSE).
