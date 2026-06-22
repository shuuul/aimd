---
name: aimd
description: Runs and develops the aimd context-preparation toolkit. Use when extracting LLM-ready Markdown from URLs, audio/video, documents, PDFs, scanned images, or when working on the aimd CLI/API/MCP codebase.
license: MIT
compatibility: Requires Python >=3.10,<3.13 and uv. Platform-specific inference: mlx-audio/mlx4ocr on macOS Apple Silicon; Transformers OCR/ASR expects Linux with CUDA. Local development should use uv commands, not pip or poetry.
metadata:
  source: https://github.com/shuuul/aimd
---

# aimd

Use this skill to guide agents through local context extraction and development with the `aimd` toolkit.

## Constraints

- Prefer the packaged CLI/API/MCP entry points instead of ad hoc Python scripts.
- Use `uv` for all project commands: `uv run`, `uv sync`, `uv build`, and `uv tool install`.
- Validate local input paths before running long OCR or ASR jobs.
- Treat `aimd.core` as interface-independent; do not import CLI/API/MCP modules from core.
- Keep URL, ASR, document, and OCR integrations behind MarkItDown plugins in `aimd.plugins.*`.
- Keep output persistence in interfaces through `aimd.interfaces.output`; `ProcessInput` and `ProcessResult` do not own `output_file`.
- Do not promise unsupported platform behavior: macOS local ML uses MLX; Linux local ASR/OCR expects CUDA-capable Transformers backends.

## One-off use from PyPI or GitHub

Run the released CLI:

```bash
uvx --from aimd-tool aimd --help
uvx --from aimd-tool aimd /path/to/input
```

Run the current GitHub `main` before a release:

```bash
uvx --from "aimd-tool @ git+https://github.com/shuuul/aimd.git@main" \
  aimd /path/to/input
```

Install the full API/MCP runtime when needed:

```bash
uvx --from "aimd-tool[all]" aimd-api
uvx --from "aimd-tool[all]" aimd-mcp
```

## Repeated use with uv tool

Install once:

```bash
uv tool install aimd-tool
aimd --help
```

Install API/MCP extras:

```bash
uv tool install "aimd-tool[all]"
```

Upgrade or remove later:

```bash
uv tool upgrade aimd-tool
uv tool uninstall aimd-tool
```

## Source checkout workflow

From this repository:

```bash
uv sync --dev
uv run aimd --help
uv run aimd /path/to/input
```

Targeted checks:

```bash
uv run pytest -q
uv run ruff check
uv run ruff format --check
```

Maintenance checks:

```bash
uv run prek --all-files
uv build
```

## Common CLI tasks

Auto-detect input type:

```bash
aimd audio.mp3
aimd "https://youtube.com/watch?v=..."
aimd document.epub
aimd scan.pdf
aimd page.png
```

Save Markdown output:

```bash
aimd document.epub --output output.md
aimd audio.mp3 --output transcript.md
```

Use URL cookies or preserve raw subtitles:

```bash
aimd "https://youtube.com/watch?v=..." --cookies cookies.txt
aimd "https://youtube.com/watch?v=..." --cookies-from-browser chrome
aimd "https://youtube.com/watch?v=..." --raw-transcript
```

Redirect temporary downloads, transcoding, and document extraction:

```bash
AIMD_TEMP_DIR=/path/to/writable/tmp aimd audio.mp3
aimd audio.mp3 --temp-dir /path/to/writable/tmp
```

## Audio and video transcription

Use `auto` unless the user asks for a specific backend:

```bash
aimd lecture.mp3 --engine auto
aimd lecture.mp3 --engine mlx      # macOS Apple Silicon
aimd lecture.mp3 --engine qwen     # Linux/CUDA Transformers
aimd lecture.mp3 --language zh
```

Common models:

```bash
aimd audio.wav --engine mlx --model mlx-community/Qwen3-ASR-1.7B-4bit
aimd audio.wav --engine qwen --model Qwen/Qwen3-ASR-1.7B
```

## OCR for scanned PDFs and images

Images route to OCR automatically. PDFs with extractable text route to document conversion; scanned PDFs route to OCR when text-layer detection is available.

```bash
aimd page.png
aimd scan.pdf --engine mlx4ocr                 # macOS Apple Silicon
aimd scan.pdf --engine transformers --model got_ocr       # Linux/CUDA
aimd scan.pdf --model paddleocr_v6             # default mlx4ocr PP-OCRv6 alias
aimd scan.pdf --model glm_ocr
aimd scan.pdf --model paddleocr_vl
aimd scan.pdf --start 0 --end 2                # zero-based inclusive PDF page range
```

## Document and MarkItDown plugin use

Local files and URLs flow through MarkItDown with bundled plugins enabled.

```bash
aimd document.epub
aimd document.pdf
aimd notes.md
markitdown --list-plugins
markitdown --use-plugins document.epub -o doc.md
markitdown --use-plugins audio.mp3 -o transcript.md
```

Readable HTML extraction through Defuddle is opt-in:

```python
from markitdown import MarkItDown

result = MarkItDown(enable_plugins=True).convert("article.html", defuddle=True)
print(result.markdown)
```

## API and MCP interfaces

Run the HTTP API:

```bash
aimd-api
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/engines
```

Process input through the API:

```bash
curl -X POST http://127.0.0.1:8000/v1/process \
  -H "Content-Type: application/json" \
  -d '{"input_source":"https://www.youtube.com/watch?v=...","transcribe_engine":"auto"}'
```

Run the MCP stdio server:

```bash
aimd-mcp
```

MCP tools are `healthz`, `list_engines`, and `process_input`. `process_input` accepts `input_source`, `task_type`, `transcribe_engine`, `model`, `language`, `start`, `end`, `output_file`, `save_original`, `cookies`, `cookies_from_browser`, and `raw_transcript`.

## Development map

- CLI: `src/aimd/interfaces/cli/app.py`
- HTTP API: `src/aimd/interfaces/api/app.py`
- MCP server: `src/aimd/interfaces/mcp/app.py`
- Shared output persistence: `src/aimd/interfaces/output.py`
- Models and routing: `src/aimd/core/models.py`, `src/aimd/core/router.py`
- Processing and Markdown shaping: `src/aimd/core/process.py`
- URL extraction plugin: `src/aimd/plugins/url/`
- Audio/video ASR plugin: `src/aimd/plugins/asr/`
- Document conversion plugin: `src/aimd/plugins/doc/`
- OCR plugin: `src/aimd/plugins/ocr/`

## Recommended checks before running

1. Confirm `uv` is installed: `uv --version`.
2. For local files, confirm the input exists: `test -e /path/to/input`.
3. For ASR/OCR, pick a backend compatible with the platform.
4. For sandboxed environments, set `AIMD_TEMP_DIR` to a writable directory.
5. For code changes, run the narrowest relevant pytest/ruff check, then broaden only when shared contracts changed.
