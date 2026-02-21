# PROJECT KNOWLEDGE BASE

**Generated:** 2026-02-21
**Commit:** working-tree
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
│   │   ├── transcription/        # yap/mlx/faster-whisper execution
│   │   ├── documents/            # Pandoc + EPUB + chunking pipeline
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
| Audio transcription | `src/aimd/infrastructure/transcription/` | engine-specific runtime paths |
| URL extraction | `src/aimd/infrastructure/url/` | cookie source fallback + subtitle/audio logic |
| Document conversion | `src/aimd/infrastructure/documents/` | Pandoc conversion, split logic, EPUB extraction |

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

# MCP
aimd-mcp

# Test
uv run pytest
```

## NOTES

- API routes: `/healthz`, `/v1/engines`, `/v1/process`.
- Engine auto priority:
  - macOS: `yap -> mlx -> cpu`
  - non-macOS: `cuda -> cpu`
- EPUB output layout: `book_name/{book_name.md, chapters/, images/}`.
- URL extraction supports explicit Netscape cookies file and browser cookie source.
