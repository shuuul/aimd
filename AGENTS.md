# PROJECT KNOWLEDGE BASE

**Generated:** 2026-06-22
**Version:** 0.10.0
**Branch:** main

## OVERVIEW

`aimd` is a Python context-preparation toolkit for LLM workflows. It turns URLs, audio/video, and documents into Markdown-like text context.

Interfaces:

- **CLI**: `aimd`
- **HTTP API**: `aimd-api` / FastAPI
- **MCP server**: `aimd-mcp` / stdio

Published distribution:

- **PyPI package**: `aimd-tool`
- **Import package**: `aimd`
- **Layout**: single root project using `src/aimd/`

## CURRENT STRUCTURE

```text
src/aimd/
├── core/           # CLI, use-cases, routing, shared contracts, infrastructure wrappers
├── api/            # FastAPI HTTP API, entrypoint aimd-api
├── mcp/            # MCP stdio server, entrypoint aimd-mcp
├── media/          # yt-dlp URLs, subtitles, audio fallback, ASR plugin
├── book/           # MarkItDown plugin: ebook extraction and Markdown cleanup
├── ocr/            # OCR scaffold for scanned PDFs/images
└── clip/           # Defuddle CLI wrapper for readable HTML extraction
```

Tests live under `tests/`; architecture notes live in `docs/architecture.md`.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Input routing | `src/aimd/core/application/use_cases/input_routing.py` | Maps source_kind + task_type into an `InputRoute`. |
| Core facade/router | `src/aimd/core/application/use_cases/process_input.py` | Dispatches `InputRoute` to configured task processors. |
| Local file conversion | `src/aimd/core/infrastructure/markitdown_processor.py` | Calls `MarkItDown(enable_plugins=True)` and wraps Markdown into `TextContext`. |
| Transcript task | `src/aimd/core/application/use_cases/processors/transcript.py` | URL/audio transcript flow and engine resolution. |
| Convert task | `src/aimd/core/application/use_cases/processors/convert.py` | Local document conversion through MarkItDown. |
| Request/result models | `src/aimd/core/application/models.py` | `ProcessInput`, `ProcessResult`, `TaskType`; output files are adapter-owned. |
| Interface mapping helpers | `src/aimd/core/application/services/interface_payloads.py` | Shared API/MCP request/result/engine payload mapping and request-time `AIMD_TEMP_DIR`. |
| Output persistence | `src/aimd/core/application/services/output_writer.py` | Shared CLI/API/MCP markdown persistence helpers. |
| Dependency wiring | `src/aimd/core/application/bootstrap.py` | Binds use-cases to infrastructure callables. |
| CLI adapter | `src/aimd/core/adapters/cli/app.py` | Typer command, user output, local file persistence. |
| HTTP API | `src/aimd/api/app.py` | `/healthz`, `/v1/engines`, `/v1/process`. |
| MCP server | `src/aimd/mcp/app.py` | `healthz`, `list_engines`, `process_input`. |
| Engine preflight | `src/aimd/media/capabilities.py` | Audio engine availability and auto-resolution. |
| Media processing | `src/aimd/media/` | yt-dlp URLs, subtitles, audio fallback, ASR, ffmpeg transcoding. |
| URL extraction | `src/aimd/media/url/` | yt-dlp cookies/subtitles/audio fallback. |
| Markdown chunking | `src/aimd/core/infrastructure/documents/` | Chunking and title extraction for MarkItDown output. |
| Book conversion | `src/aimd/book/` | MarkItDown plugin, EPUB-compatible ebook pipeline, cleanup, image extraction. |
| OCR scaffold | `src/aimd/ocr/` | Placeholder OCR entrypoint for next feature. |
| HTML clipping | `src/aimd/clip/` | Defuddle CLI wrapper. |

## NEXT MAJOR FEATURE: OCR

The next planned package capability is OCR for scanned PDFs/images.

Recommended smallest pre-OCR changes:

1. Extend `TaskType` with an explicit OCR path, e.g. `"ocr"`, and `SourceKind` with OCR-relevant sources such as `"image_file"` if OCR will support images/scanned PDFs or OCR-specific options.
2. Implement OCR as a MarkItDown `DocumentConverter` in `src/aimd/ocr/`.
3. Register OCR via `[project.entry-points."markitdown.plugin"]` when implemented.
4. Extend centralized support/routing in `input_routing.py` for OCR inputs if MarkItDown extension coverage is insufficient.
5. Keep `aimd.core` as the router/facade; local OCR conversion should go through MarkItDown.
6. Keep `/v1/engines` transcription-oriented for now; add OCR capability reporting later only if needed.

OCR non-goals for the first pass:

- No custom aimd plugin registry; use MarkItDown entry points.
- No rewrite of `ProcessInput` unless OCR introduces multiple user-facing options.
- No changes to URL subtitle/audio fallback ordering.
- No changes to the `TextContext(title, chunk_list, split_header_level)` contract.

## CONVENTIONS

- **Async-first**: processing paths are async through use-cases and infrastructure.
- **Use-case centric orchestration**: routing belongs in `aimd.core.application.use_cases`, not adapters.
- **Route model**: classify inputs with `InputRoute(source_kind, task_type)`; `source_kind` is the supplied input kind, `task_type` selects processing logic.
- **MarkItDown contract**: local file conversion goes through `MarkItDown(enable_plugins=True)`; bundled feature modules register MarkItDown plugins.
- **Output ownership**: `output_file` belongs to CLI/API/MCP adapters, not `ProcessInput` or `ProcessResult`; use `aimd.core.application.services.output_writer` for shared persistence.
- **Fail-fast preflight**: validate selected engines before expensive work.
- **Typed errors**: raise `AimdError` subclasses where possible for predictable CLI/API/MCP mapping.
- **Stable output contract**: processors return `TextContext(title, chunk_list, split_header_level)`.
- **uv only**: use `uv run`, `uv sync`; avoid poetry/pip for local development workflows.
- **Platform-conditional audio deps**: `mlx-audio` on Darwin; Qwen3-ASR runs through the Transformers backend on Linux/CUDA.
- **Module boundaries**: `aimd.core` owns interface adapters, routing, and `TextContext` wrapping; `aimd.media` owns media URL/local audio-video extraction; `aimd.book` and future OCR follow MarkItDown plugin contracts.

## ANTI-PATTERNS

- Embedding orchestration directly in adapters.
- Importing infrastructure from adapters when an application-level dependency would work.
- Raising generic exceptions where a domain error exists.
- Changing URL subtitle/cookie/audio fallback ordering without explicit tests.
- Hard-coding stale audio model allow-lists or help text that blocks newer `mlx-audio` STT models.
- Adding broad abstractions before OCR requirements prove they are needed.

## COMMANDS

```bash
# Setup / dependency refresh
uv sync --dev --upgrade

# Code quality
uv run ruff check --fix
uv run ruff format
uv run prek --all-files
uv run prek autoupdate

# Tests
uv run pytest -q

# Build package
uv build

# CLI examples
aimd audio.mp3
aimd "https://youtube.com/..."
aimd book.epub
aimd audio.mp3 --engine mlx --model mlx-community/parakeet-tdt-0.6b-v3
aimd "https://youtube.com/..." --cookies-from-browser chrome --raw-transcript
aimd audio.mp3 --temp-dir ./tmp --log-level DEBUG

# API
aimd-api
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/engines

# MCP
aimd-mcp
```

## DAILY MAINTENANCE / RELEASE CHECKLIST

```bash
# 1) Refresh environment and lockfile
uv sync --dev --upgrade

# 2) Refresh pre-commit hooks when doing maintenance
uv run prek autoupdate

# 3) Run hooks
uv run prek --all-files

# 4) If hooks changed files, stage/re-run once
git add -A
uv run prek --all-files

# 5) Run tests
uv run pytest -q
```

For version bumps:

```bash
uv version --bump patch  # or minor/major
```

## PACKAGE / DEPENDENCY NOTES

- Single PyPI distribution: `aimd-tool`.
- Import namespace: `aimd`.
- Console scripts: `aimd`, `aimd-api`, `aimd-mcp`.
- `aimd-tool[all]` installs API/MCP runtime dependencies.
- Dev group includes API/MCP dependencies so tests run with `uv sync --dev`.
- `torch`/`torchaudio` resolve from the PyTorch CPU wheel index on Darwin.
- `defuddle` is a Node/TypeScript package; `aimd.clip` wraps `npx defuddle parse` rather than vendoring Node code.
- URL extraction supports Netscape cookie files and browser cookie sources.
- `--raw-transcript` preserves original SRT/VTT subtitle formatting; default strips subtitles to plain text.
- `--temp-dir` and `AIMD_TEMP_DIR` redirect temporary downloads, transcoding, and ebook extraction.
- Local file conversion uses MarkItDown and installed `markitdown.plugin` entry points.
- Ebook pipeline lives in the `aimd.book` MarkItDown plugin and uses spine ordering, Pandoc CLI (`-f html -t markdown_mmd-raw_html --wrap=none`), post-processing cleanup, and flat image extraction.
- `aimd.book` currently implements an EPUB-compatible ZIP/spine extraction pipeline while routing `.epub`, `.mobi`, and `.azw3` as book inputs; true non-EPUB handling should be added inside `aimd.book`, not in the core router.
