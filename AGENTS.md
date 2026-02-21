# PROJECT KNOWLEDGE BASE

**Generated:** 2026-02-21
**Commit:** working-tree
**Branch:** main

## OVERVIEW

Python context-preparation toolkit with dual interfaces:
- **CLI** (`aimd`) for local batch usage
- **HTTP API** (`aimd-api`, FastAPI) for service integration

Core capabilities remain: transcribe audio/video or URLs, convert docs to markdown, and process EPUB with chapters/images.

## STRUCTURE

```
aimd/
├── src/aimd/
│   ├── cli.py              # Typer entrypoint
│   ├── api.py              # FastAPI app (/healthz, /v1/engines, /v1/process)
│   ├── service.py          # Shared async orchestration for CLI/API
│   ├── capabilities.py     # Engine preflight + auto engine resolution
│   ├── errors.py           # Typed domain errors with status mapping
│   ├── const.py            # Extensions, engines, language mappings
│   ├── utils.py            # URL/file utilities
│   ├── types.py            # TextContext model
│   └── tool/
│       ├── audio.py        # yap/mlx/faster-whisper transcription
│       ├── file.py         # Pandoc conversion + EPUB extraction
│       └── url.py          # yt-dlp subtitle/audio fallback
├── tests/
│   ├── test_api.py
│   ├── test_capabilities.py
│   └── test_epub.py
├── pyproject.toml
└── AGENTS.md
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Shared processing flow | `src/aimd/service.py` | `ensure_supported_input`, transcript/convert orchestration |
| API endpoints | `src/aimd/api.py` | Health, engine preflight introspection, process endpoint |
| Engine preflight logic | `src/aimd/capabilities.py` | Availability checks + auto priority |
| Typed error model | `src/aimd/errors.py` | Domain-level errors for CLI/API consistency |
| CLI command behavior | `src/aimd/cli.py` | Delegates to service layer |
| Audio transcription internals | `src/aimd/tool/audio.py` | Runtime engine execution; lazy `torch` usage |
| URL extraction | `src/aimd/tool/url.py` | Subtitle-first + audio fallback with cookies handling |
| Document conversion | `src/aimd/tool/file.py` | Pandoc + EPUB image/chapter extraction |

## CONVENTIONS (THIS PROJECT)

- **Async-first**: processing functions are async and shared by CLI/API.
- **Single orchestration layer**: business flow goes through `service.py`.
- **Fail-fast preflight**: transcription engine choice validated before costly work.
- **Typed errors**: use `AimdError` subclasses for predictable status handling.
- **TextContext contract**: downstream processing returns `TextContext(title, chunk_list, split_header_level)`.
- **uv only**: use `uv run`, `uv sync`; avoid pip/poetry.

## ANTI-PATTERNS (THIS PROJECT)

- **Headerless split failure remains**: `file.py` still errors if no usable markdown headers for chunking.
- **Platform locks**: `yap`/`mlx` remain macOS-constrained; `mlx` requires Apple Silicon.
- **External runtime tools**: `yap`, `pandoc`, `yt-dlp` environment issues surface at runtime.
- **Subtitle constraints**: danmaku blocked via `FORBIDDEN_SUBTITLE_LANGUAGES`.

## COMMANDS

```bash
# Setup
uv sync --all-packages --all-extras --dev

# Code quality
uv run ruff check --fix && uv run ruff format
uv run pre-commit run --all-files

# CLI
aimd audio.mp3
aimd "https://youtube.com/..."
aimd book.epub

# API
aimd-api
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/engines

# Test
uv run pytest
```

## NOTES

- **API route summary**: `/healthz`, `/v1/engines`, `/v1/process`.
- **Engine auto priority**:
  - macOS: `yap -> mlx -> cpu`
  - non-macOS: `cuda -> cpu`
- **Lazy heavyweight import**: `torch` is no longer imported at module import time in `audio.py`.
- **EPUB output layout**: `book_name/{book_name.md, chapters/, images/}`.
- **Cookies support**: URL extraction supports explicit Netscape cookies file to bypass keyring issues.
