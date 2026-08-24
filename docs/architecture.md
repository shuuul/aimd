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
- `aimd.plugins.asr`: MarkItDown plugin for audio/video transcription, audio preprocessing, and backend selection. Backend implementations live under `aimd.plugins.asr.models` (`mlx`, `transformers`, `remote`).
- `aimd.plugins.doc`: MarkItDown plugin for document conversion. EPUB uses a custom spine/image extraction pipeline; other Pandoc-supported formats use direct Pandoc conversion; text-layer PDFs use pdf-inspector (`AimdPdfConverter`) ahead of MarkItDown's built-in `PdfConverter`.
- `aimd.plugins.ocr`: MarkItDown plugin and OCR task implementation for images and scanned PDFs. Engine implementations include local MLX/Transformers models and an OpenAI-compatible remote Unlimited-OCR adapter.

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
5. Core returns lossless Markdown and its asset base alongside the shaped `TextContext`.
6. The interface maps `ProcessResult` to interface-specific response/output and persists `output_file` if requested.

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
│ interface mapping  │                           │ Markdown + asset base│
│                    │                           │ + TextContext shape  │
╰────────────────────╯                           ╰──────────────────────╯
```

`ProcessResult.markdown` is the exact MarkItDown output before context splitting.
`TextContext.chunk_list` remains the context-window representation and is not a lossless
viewer artifact. `asset_base_uri` is the original URL for URL processing, the generated
asset directory for document conversions that emit one, the input parent directory for
other local convert/OCR work, and `None` for local transcriptions without related assets.
Local directory URIs always end in `/` so standards-based URL joining preserves directory
semantics.

Asset-producing document conversions create `output_dir` beside the local source. The
directory is durable caller-owned output, not request-scoped temporary storage, and AIMD
does not clean it up when an interface call completes. Interfaces therefore ignore a
separate requested `output_file` for these results so relative links remain valid. Other
requested files are persisted byte-for-byte from `ProcessResult.markdown`; interfaces
must not reconstruct them from `TextContext.chunk_list`.

## Backend And Model Boundaries

Model selection is task-specific and flows through `ProcessInput.model` to the selected processor:

| Task | Backend boundary | Supported model source |
|------|-----------------|------------------------|
| Transcript | `aimd.plugins.url` for URL/subtitle/audio fallback, `aimd.plugins.asr` for transcription | Qwen3-ASR 1.7B/0.6B through local MLX/Transformers, or remote `/v1/audio/transcriptions` when `AIMD_ASR_BASE_URL` is set. |
| Convert | MarkItDown | MarkItDown built-ins plus bundled `aimd.plugins.url`, `aimd.plugins.asr`, `aimd.plugins.doc`, and `aimd.plugins.ocr` plugin entry points. |
| OCR | MarkItDown + `aimd.plugins.ocr` plugin | Local MLX/Transformers models, or remote Unlimited-OCR chat completions when `AIMD_OCR_BASE_URL` is set. |

The README is the user-facing source of truth for supported `--model` values. Implementation constants live in `aimd.plugins.asr.const` and `aimd.plugins.ocr.backends`.

Remote settings flow through `ProcessInput` and MarkItDown kwargs so CLI, API,
MCP, local-file work, and URL audio fallback share one selection path. Explicit
request values override `AIMD_{ASR,OCR}_{BASE_URL,MODEL,API_KEY}`. A configured
remote URL bypasses local CUDA/MLX preflight; without one, local behavior is
unchanged.

For performance expectations and benchmarking guidance, see [Performance](performance.md).
For the authenticated loopback process, path, job, and output-lifetime boundary used by
desktop clients, see [Local desktop sidecar contract](sidecar.md).

## MarkItDown Worker And Domain-Error Normalization

Core runs MarkItDown conversion off the event loop via `_run_markitdown()` (`asyncio` executor). Failures are normalized by `_raise_from_markitdown_failure()`:

1. Bare `AimdError` subclasses re-raise unchanged.
2. MarkItDown `FileConversionException` aggregates are scanned in converter-attempt order, and the first nested `AimdError` cause/context is restored.
3. Any other exception is wrapped as `ProcessingFailedError`.

Feature plugins own mapping at their boundary:

- Missing external backends (`pandoc`, `npx`, ASR engines) -> `BackendUnavailableError`
- Missing input paths -> `InputNotFoundError`
- Conversion/transcription failures -> `ProcessingFailedError`

## Error Handling

Domain errors in `aimd.core.errors` are preserved and used end-to-end:

- `UnsupportedInputError` -> 400 in HTTP API
- `BackendUnavailableError` -> 422 in HTTP API
- `InputNotFoundError` -> 404 in HTTP API
- `ProcessingFailedError` -> 500 in HTTP API
