# PROJECT KNOWLEDGE BASE

**Generated:** 2026-06-17
**Version:** 0.8.5
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
│   │   ├── transcription/        # mlx-audio/qwen-asr engines, resolver, audio transcoding
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
│   ├── test_audio_transcoding.py
│   ├── test_file_processing.py
│   ├── test_epub.py
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
| Audio transcription | `src/aimd/infrastructure/transcription/` | mlx-audio STT (Apple Silicon), qwen-asr Qwen3-ASR (Linux/CUDA), ffmpeg fallback transcoding |
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
- **Platform-conditional core audio deps**: `mlx-audio` installs on Darwin; `qwen-asr` installs on Linux.

## ANTI-PATTERNS (THIS PROJECT)

- Embedding business orchestration directly in adapters.
- Cross-layer imports from infrastructure into adapters.
- Raising generic exceptions where a domain error exists.
- Changing URL subtitle/cookie fallback ordering without explicit tests.
- Blocking newer mlx-audio STT models by hard-coded stale help text.

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
aimd audio.mp3 --engine mlx --model mlx-community/parakeet-tdt-0.6b-v3
aimd "https://youtube.com/..." --cookies-from-browser chrome --raw-transcript
aimd audio.mp3 --temp-dir ./tmp --log-level DEBUG

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
- Core audio dependencies are platform-conditional in `pyproject.toml`: `mlx-audio>=0.4.4` on Darwin, `qwen-asr>=0.0.6` on Linux, and `torch`/`torchaudio` resolve from the PyTorch CPU wheel index on Darwin.
- API routes: `/healthz`, `/v1/engines`, `/v1/process`.
- Engine auto priority:
  - macOS Apple Silicon with `mlx_audio`: `mlx`
  - Linux with `qwen_asr`, `torch`, and CUDA: `qwen`
- mlx engine uses [mlx-audio](https://github.com/Blaizzy/mlx-audio) STT on Apple Silicon. Default model remains `mlx-community/Qwen3-ASR-1.7B-4bit`; `MLX_AUDIO_MODELS` also allows newer mlx-audio 0.4.4 STT model IDs such as Whisper large-v3-turbo, Distil-Whisper, Parakeet v3, Nemotron ASR, Voxtral, VibeVoice-ASR, and Qwen2-Audio. Forced aligner models are not transcription models because they require reference text.
- qwen engine uses [qwen-asr](https://github.com/QwenLM/Qwen3-ASR) with `Qwen/Qwen3-ASR-1.7B` by default; `Qwen/Qwen3-ASR-0.6B` is the faster/lower-memory option. Requires Linux with a CUDA-capable GPU. Upstream Qwen3-ASR supports 30 languages plus 22 Chinese dialects; local language mapping currently exposes a smaller explicit code set.
- `--model` / `-m` CLI option selects the transcription model for local audio fallback and direct audio inputs.
- `--language` maps short codes to model language names where the selected backend accepts language hints. mlx Qwen3-ASR defaults to Chinese when omitted; non-Qwen mlx-audio STT models are left on model default/auto-detection.
- `--raw-transcript` CLI option preserves original subtitle formatting (SRT/VTT); default strips to plain text.
- `--temp-dir` and `AIMD_TEMP_DIR` redirect temporary downloads, transcoding, and EPUB extraction.
- URL extraction supports explicit Netscape cookies file and browser cookie source.
- EPUB pipeline: spine-based chapter ordering (container.xml → OPF → manifest + spine), pandoc CLI (`-f html -t markdown_mmd-raw_html --wrap=none`), post-processing via `epub_cleaner.clean_markdown`.
- EPUB output layout: `book_name/{book_name.md, chapters/, images/}`. Chapters named after original HTML stems; combined file uses `---` separators.
- Non-EPUB document conversion uses the `pandoc` Python library with `markdown_mmd-raw_html` format and `--wrap=none`.
- URL subtitle content is simplified to plain text by default (SRT/VTT/TTML formatting stripped via `strip_subtitle_formatting`).
