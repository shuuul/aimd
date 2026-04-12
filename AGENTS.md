# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-08
**Commit:** c7d5e24 (v0.6.5)
**Branch:** main

## OVERVIEW

Python context-preparation toolkit with three interfaces:
- **CLI** (`aimd`) for local usage
- **HTTP API** (`aimd-api`, FastAPI)
- **MCP Server** (`aimd-mcp`, stdio)

Architecture is ports/adapters:
- `application` for orchestration use-cases and canonical models
- `infrastructure` for concrete processing engines and integrations
- `adapters` for interface binding (CLI/API/MCP)

## STRUCTURE

```
aimd/
├── src/aimd/
│   ├── cli.py                    # CLI entrypoint
│   ├── api.py                    # FastAPI entrypoint
│   ├── mcp.py                    # MCP entrypoint
│   ├── errors.py                 # Typed domain errors with status mapping
│   ├── const.py                  # Extensions, engines, language mappings
│   ├── utils.py                  # URL/file utilities
│   ├── types.py                  # TextContext model
│   ├── application/
│   │   ├── models.py             # ProcessInput/ProcessResult/TaskType
│   │   ├── bootstrap.py          # Explicit dependency wiring
│   │   ├── use_cases/
│   │   │   ├── process_input.py  # Main orchestration use-case
│   │   │   └── list_engines.py   # Engine introspection use-case
│   │   └── services/
│   │       └── output_writer.py  # Shared output persistence
│   ├── infrastructure/
│   │   ├── capabilities/
│   │   │   └── detector.py       # Engine preflight checks
│   │   ├── transcription/        # mlx-audio-plus Fun-ASR-Nano / FunASR PyTorch
│   │   ├── documents/            # Pandoc + EPUB + chunking pipeline
│   │   │   ├── epub_processor.py # Spine-ordered EPUB extraction via pandoc CLI
│   │   │   ├── epub_cleaner.py   # Post-processing cleanup for EPUB markdown
│   │   │   ├── pandoc_reader.py  # General pandoc-backed document conversion
│   │   │   ├── chunking.py       # Markdown splitting by headers / paragraphs
│   │   │   ├── title_extractor.py# Title extraction from markdown content
│   │   │   └── processor.py      # Document processing orchestration
│   │   └── url/                  # yt-dlp cookies/subtitles/audio fallback
│   └── adapters/
│       ├── cli/app.py            # Typer interface adapter
│       ├── http/app.py           # FastAPI interface adapter
│       └── mcp/server.py         # MCP interface adapter
├── tests/
│   ├── test_api.py
│   ├── test_cli.py
│   ├── test_use_cases.py
│   ├── test_url.py
│   ├── test_file_processing.py
│   ├── test_capabilities.py
│   └── test_mcp_server.py
├── docs/
│   └── architecture.md
├── pyproject.toml
└── AGENTS.md
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Main process orchestration | `src/aimd/application/use_cases/process_input.py` | task type detection + transcript/convert routing |
| Dependency wiring | `src/aimd/application/bootstrap.py` | binds use-cases to infra implementations |
| API routes | `src/aimd/adapters/http/app.py` | `/healthz`, `/v1/engines`, `/v1/process` |
| CLI behavior | `src/aimd/adapters/cli/app.py` | Typer command and output persistence |
| MCP tools | `src/aimd/adapters/mcp/server.py` | `healthz`, `list_engines`, `process_input` |
| Engine preflight | `src/aimd/infrastructure/capabilities/detector.py` | availability + auto-resolution |
| Audio transcription | `src/aimd/infrastructure/transcription/` | mlx-audio-plus Fun-ASR-Nano (MLX), FunASR PyTorch (Fun-ASR-Nano / SenseVoiceSmall) |
| URL extraction | `src/aimd/infrastructure/url/` | cookie source fallback + subtitle/audio logic |
| Document conversion | `src/aimd/infrastructure/documents/` | Pandoc conversion (`markdown_mmd-raw_html`), split logic, EPUB extraction |
| EPUB pipeline | `src/aimd/infrastructure/documents/epub_processor.py` | Spine-ordered chapters, pandoc CLI subprocess, flat image extraction |
| EPUB markdown cleanup | `src/aimd/infrastructure/documents/epub_cleaner.py` | Footnotes, TOC, heading normalisation, image path fixup |

## CONVENTIONS (THIS PROJECT)

- **Async-first**: processing paths are async throughout use-cases/infrastructure.
- **Use-case centric orchestration**: task routing lives in `application/use_cases`.
- **Fail-fast preflight**: engine selection validated before expensive work.
- **Typed errors**: use `AimdError` subclasses for predictable mapping.
- **TextContext contract**: processing returns `TextContext(title, chunk_list, split_header_level)`.
- **uv only**: use `uv run`, `uv sync`; avoid pip/poetry.

## ANTI-PATTERNS (THIS PROJECT)

- Embedding business orchestration directly in adapters.
- Cross-layer imports from infrastructure into adapters.
- Raising generic exceptions where a domain error exists.
- Changing URL subtitle/cookie fallback ordering without explicit tests.

## COMMANDS

```bash
# Setup (dev group includes fastapi/uvicorn/mcp for tests; core package is CLI-first)
uv sync --dev --upgrade

# Code quality
uv run ruff check --fix && uv run ruff format
uv run prek --all-files
uv run prek autoupdate

# CLI
aimd audio.mp3
aimd "https://youtube.com/..."
aimd book.epub

# API
aimd-api
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/engines

# MCP
aimd-mcp

# Test
uv run pytest
```

## DAILY COMMIT CHECKLIST

```bash
# 1) Sync environment (if needed)
uv sync --dev --upgrade

# 2) Run hooks on all files
uv run prek --all-files

# 3) If hooks modified files (e.g., ruff-format), re-stage and re-run once
git add -A
uv run prek --all-files

# 4) Run tests
uv run pytest -q

# 5) Commit
git add -A
git commit -m "<type>: <summary>"
```

Recommended maintenance:
- Run `uv run prek autoupdate` periodically (e.g., weekly) and commit hook version bumps separately.

## NOTES

- **Dependencies**: Core install is **CLI-first** (`aimd`). Optional extras: `api` (FastAPI + uvicorn → `aimd-api`), `mcp` (`aimd-mcp`). Development uses `uv sync --dev`, which includes those packages so API/MCP tests run without extra flags.
- API routes: `/healthz`, `/v1/engines`, `/v1/process`.
- Engine auto priority:
  - macOS: `mlx -> funasr`
  - non-macOS: `funasr`
- mlx engine uses [mlx-audio-plus](https://huggingface.co/mlx-community/Fun-ASR-Nano-2512-4bit) Fun-ASR-Nano (`mlx_audio.stt.models.funasr`) by default on Apple Silicon.
- funasr engine uses [FunASR](https://github.com/modelscope/FunASR) with Fun-ASR-Nano-2512 by default (CPU/CUDA). Also supports SenseVoiceSmall.
- FunASR does not officially support MPS (Apple Silicon GPU). On macOS, use `mlx` instead.
- `--model` / `-m` CLI option selects the transcription model (mlx Fun-ASR-Nano id or FunASR model id).
- `--raw-transcript` CLI option preserves original subtitle formatting (SRT/VTT); default strips to plain text.
- EPUB pipeline: spine-based chapter ordering (container.xml → OPF → manifest + spine), pandoc CLI (`-f html -t markdown_mmd-raw_html --wrap=none`), post-processing via `epub_cleaner.clean_markdown`.
- EPUB output layout: `book_name/{book_name.md, chapters/, images/}`. Chapters named after original HTML stems; combined file uses `---` separators.
- Non-EPUB document conversion uses the `pandoc` Python library with `markdown_mmd-raw_html` format and `--wrap=none`.
- URL extraction supports explicit Netscape cookies file and browser cookie source.
- URL subtitle content is simplified to plain text by default (SRT/VTT/TTML formatting stripped via `strip_subtitle_formatting`).
