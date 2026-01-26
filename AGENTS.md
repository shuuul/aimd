# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-26
**Commit:** 3755051293cef5b08fbac2e9b7db6ba32388e199
**Branch:** main

## OVERVIEW

Python CLI tool for LLM context preparation - transcribes audio/video and converts documents to markdown.

## STRUCTURE

```
aimd/
├── src/aimd/
│   ├── cli.py              # Entry point (typer CLI)
│   ├── const.py            # Constants (extensions, engines, locales)
│   ├── utils.py            # URL/file utilities
│   ├── types.py            # TextContext Pydantic model
│   └── tool/
│       ├── audio.py        # Multi-engine transcription
│       ├── file.py         # Document conversion (Pandoc)
│       └── url.py          # Video URL extraction (yt-dlp)
├── tests/
│   └── test_epub.py
├── pyproject.toml
└── AGENTS.md
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| CLI logic | `src/aimd/cli.py` | Auto-detection, task dispatch |
| Audio transcription | `src/aimd/tool/audio.py` | yap/mlx/cuda/cpu engines |
| Document conversion | `src/aimd/tool/file.py` | Pandoc, EPUB extraction |
| URL processing | `src/aimd/tool/url.py` | yt-dlp, subtitle extraction |
| Constants | `src/aimd/const.py` | Extensions, engines, locales |

## CONVENTIONS (THIS PROJECT)

- **Async-first**: All processing functions are `async`
- **uv only**: Use `uv run`, `uv sync`, not pip/poetry
- **TextContext**: Tools return `TextContext(title, chunk_list, split_header_level)`
- **40k char limit**: Documents auto-split to stay under LLM context limit
- **Engine selection**: `auto` picks platform-optimal (yap→mlx→cuda→cpu)

## ANTI-PATTERNS (THIS PROJECT)

- **Danmaku forbidden**: Subtitle type blocked in `FORBIDDEN_SUBTITLE_LANGUAGES`
- **Platform locks**: `yap`/`mlx` require macOS; `mlx` needs Apple Silicon
- **No fallback splitter**: `file.py` throws if doc has no markdown headers
- **Fragile encoding**: `audio.py` yap fallback tries UTF-8→GB2312→Latin-1
- **Runtime imports**: Engine libs imported inside functions (late failure)

## COMMANDS

```bash
# Setup
uv sync --all-packages --all-extras --dev

# Code quality
uv run ruff check --fix && uv run ruff format
uv run pre-commit run --all-files

# Run
aimd audio.mp3                    # Auto-detect
aimd "https://youtube.com/..."   # URL
aimd book.epub                   # Document

# Test
uv run pytest
```

## NOTES

- **EPUB output**: Creates `book_name/{book_name.md, chapters/, images/}`
- **Cookie dependency**: `url.py` reads Chrome cookies for YouTube/Bilibili
- **Title extraction priority**: H1 → YAML → Setext → First line
