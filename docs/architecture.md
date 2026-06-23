# Architecture

## Overview

The repository has one published distribution, `aimd-tool`, with source code under `src/aimd`.
The `aimd` package uses MarkItDown as the URL/local-file conversion contract and keeps core as a small interface-independent processing service:

- `aimd.core.models`: canonical request/response models.
- `aimd.core.process`: input routing and processing.
- `aimd.core.process`: processing orchestration, MarkItDown runner, and Markdown shaping helpers.
- `aimd.interfaces.output`: shared output persistence helper for interfaces.
- `aimd.interfaces.cli`: Typer CLI interface (`aimd.interfaces.cli:main`).
- `aimd.interfaces.api`: FastAPI-backed HTTP API module (`aimd.interfaces.api:main`).
- `aimd.interfaces.mcp`: MCP stdio server module (`aimd.interfaces.mcp.app:main`).
- `aimd.plugins.url`: MarkItDown plugin for URL transcript extraction, yt-dlp subtitle-first/audio fallback, cookie handling, and opt-in Defuddle readable HTML extraction. Logic lives directly under `aimd.plugins.url/` (flattened).
- `aimd.plugins.asr`: MarkItDown plugin for local audio/video transcription, audio preprocessing, and platform backend selection (backends flattened, no separate models/ subdir).
- `aimd.plugins.doc`: MarkItDown plugin for Pandoc-supported documents (cleaner inlined). EPUB uses a custom spine/image extraction pipeline; other supported formats use direct Pandoc conversion.
- `aimd.plugins.ocr`: MarkItDown plugin and OCR task implementation for images and scanned PDFs (models/ collapsed to single models.py), with `mlx-vlm` on macOS/Apple Silicon and CUDA Transformers OCR models on Linux.

Bundled MarkItDown plugin entry points are `aimd.plugins.asr`, `aimd.plugins.url`, `aimd.plugins.doc`, and `aimd.plugins.ocr`.

## Dependency Rules

- Interface modules depend on the core processing service, not plugin implementation internals.
- URL and local-file processing should go through `MarkItDown(enable_plugins=True)` rather than calling bundled plugin processors directly.
- Plugins and core processing do not import interfaces.
- Feature modules that handle local files should follow MarkItDown's plugin/converter contract.
- Output destinations are interface concerns. `ProcessInput` does not carry `output_file`; CLI/API/MCP persist requested outputs through `aimd.interfaces.output`.
- API/MCP response/request payload shaping lives in the API/MCP modules, not in core.

## Primary Flow

1. CLI/API/MCP receives request payload/options.
2. The interface builds `ProcessInput` with shared mapping helpers and calls `aimd.core.process.process_input()`.
3. Core routes request by `InputRoute(source_kind, task_type)`.
4. URL and local-file tasks go through MarkItDown plus bundled plugins (`aimd.plugins.url`, `aimd.plugins.asr`, `aimd.plugins.doc`, `aimd.plugins.ocr`).
5. The interface maps `ProcessResult` to interface-specific response/output and persists `output_file` if requested.

```diagram
╭──────────────╮     ╭──────────────────────╮     ╭──────────────────────╮
│ CLI/API/MCP  │────▶│ process_input()      │────▶│ MarkItDown plugins  │
│ interfaces   │     │ route + dispatch     │     │ URL/local conversion│
╰──────┬───────╯     ╰──────────────────────╯     ╰──────────┬───────────╯
       │                                                     │
       │                                                     ▼
       │                                          ╭──────────────────────╮
       │                                          │ aimd.plugins.url, aimd.plugins.asr, │
       │                                          │ aimd.plugins.doc, aimd.plugins.ocr │
       │                                          ╰──────────┬───────────╯
       ▼                                                     ▼
╭────────────────────╮                           ╭──────────────────────╮
│ aimd.interfaces.output +      │◀──────────────────────────│ ProcessResult        │
│ interface mapping  │                           │ TextContext shape    │
╰────────────────────╯                           ╰──────────────────────╯
```

## Backend And Model Boundaries

Model selection is task-specific and flows through `ProcessInput.model` to the selected processor:

| Task | Backend boundary | Supported model source |
|------|-----------------|------------------------|
| Transcript | `aimd.plugins.url` for URL/subtitle/audio fallback, `aimd.plugins.asr` for transcription | `mlx-audio` STT models by default on Apple Silicon; Qwen3-ASR via AIMD's local Transformers 5.x backend on CUDA-capable non-Darwin platforms and as an explicit `Qwen/Qwen3-ASR-*` opt-in on macOS/MPS. |
| Convert | MarkItDown | MarkItDown built-ins plus bundled `aimd.plugins.url`, `aimd.plugins.asr`, `aimd.plugins.doc`, and `aimd.plugins.ocr` plugin entry points. |
| OCR | MarkItDown + `aimd.plugins.ocr` plugin | `mlx-vlm` models on macOS/Apple Silicon; CUDA Transformers OCR aliases and explicit Hugging Face model IDs on Linux. |

The README is the user-facing source of truth for supported `--model` values. Implementation constants live in `aimd.plugins.asr.const` and `aimd.plugins.ocr.backends`.

For performance expectations and benchmarking guidance, see [Performance](performance.md).

## Error Handling

Domain errors in `aimd.core.errors` are preserved and used end-to-end:

- `UnsupportedInputError` -> 400 in HTTP API
- `BackendUnavailableError` -> 422 in HTTP API
- `InputNotFoundError` -> 404 in HTTP API
- `ProcessingFailedError` -> 500 in HTTP API
