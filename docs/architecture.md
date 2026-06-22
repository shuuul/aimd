# Architecture

## Overview

The repository has one published distribution, `aimd-tool`, with source code under `src/aimd`.
The `aimd` package uses MarkItDown as the local-file conversion contract and follows a ports/adapters structure:

- `aimd.core.application`: use-cases, canonical request/response models, and bootstrap wiring.
- `aimd.core.infrastructure`: the MarkItDown runner, media package adapter, and Markdown chunking helpers.
- `aimd.core.adapters`: CLI interface layer.
- `aimd.api`: FastAPI-backed HTTP API module.
- `aimd.mcp`: MCP stdio server module.
- `aimd.asr`: local audio/video transcription, audio preprocessing, ASR model validation, and engine capability preflight.
- `aimd.media`: yt-dlp URL extraction, subtitle-first/audio fallback, and MarkItDown plugin wiring for local audio/video inputs.
- `aimd.book`: MarkItDown plugin for ebook spine/image extraction and Markdown cleanup. The current implementation is EPUB-compatible and routes `.epub`, `.mobi`, and `.azw3` as book inputs for future format-specific handling.
- `aimd.ocr`: explicit OCR task for images and scanned PDFs, with `mlx4ocr` on macOS/Apple Silicon and CUDA Transformers OCR models on Linux.
- `aimd.clip`: Python wrapper around the Defuddle CLI.

## Dependency Rules

- Interface modules/adapters depend on application use-cases, not infrastructure processing modules.
- Convert-task local-file processing should go through `MarkItDown(enable_plugins=True)` rather than calling bundled plugin processors directly. OCR-task local-file processing is routed explicitly to `aimd.ocr`.
- Infrastructure does not import adapters.
- Feature modules that are MarkItDown extensions should follow MarkItDown's plugin/converter contract. Explicit task modules such as `aimd.ocr` expose application-facing processor functions instead.
- Output destinations are interface concerns. `ProcessInput` does not carry `output_file`; CLI/API/MCP persist requested outputs through shared helpers in `aimd.core.application.services.output_writer`.
- API/MCP response/request payload shaping lives in `aimd.core.application.services.interface_payloads` as plain mapping helpers; it must not depend on FastAPI or MCP types.

## Primary Flow

1. CLI/API/MCP adapter receives request payload/options.
2. Adapter builds `ProcessInput` with shared mapping helpers and calls `ProcessInputUseCase.execute`.
3. Use-case routes request by `InputRoute(source_kind, task_type)` to a task processor.
4. Transcript URL tasks call `aimd.media`; transcript local audio/video conversion delegates to `aimd.asr` through the media MarkItDown plugin; convert tasks call MarkItDown plus bundled plugins; OCR tasks call `aimd.ocr`.
5. Adapter maps `ProcessResult` to interface-specific response/output and persists `output_file` if requested.

```diagram
╭──────────────╮     ╭──────────────────────╮     ╭──────────────────────╮
│ CLI/API/MCP  │────▶│ ProcessInputUseCase  │────▶│ TaskProcessor        │
│ adapters     │     │ route + dispatch     │     │ transcript/convert/ │
╰──────┬───────╯     ╰──────────────────────╯     │ ocr                  │
       │                                          ╰──────────┬───────────╯
       │                                                     │
       │                                                     ▼
       │                                          ╭──────────────────────╮
       │                                          │ aimd.media/aimd.asr, │
       │                                          │ MarkItDown plugins, │
       │                                          │ or aimd.ocr          │
       │                                          ╰──────────┬───────────╯
       ▼                                                     ▼
╭────────────────────╮                           ╭──────────────────────╮
│ output_writer +    │◀──────────────────────────│ ProcessResult        │
│ interface_payloads │                           │ TextContext shape    │
╰────────────────────╯                           ╰──────────────────────╯
```

## Engine And Model Boundaries

Model selection is task-specific and flows through `ProcessInput.model` to the selected processor:

| Task | Engine boundary | Supported model source |
|------|-----------------|------------------------|
| Transcript | `aimd.media` for URL/subtitle/audio fallback, `aimd.asr` for transcription | `mlx-audio` STT models on Apple Silicon; Qwen3-ASR Transformers models on Linux/CUDA. |
| Convert | MarkItDown | MarkItDown built-ins plus bundled `aimd.media` and `aimd.book` plugin entry points. |
| OCR | `aimd.ocr` | `mlx4ocr` models on macOS/Apple Silicon; CUDA Transformers OCR aliases and explicit Hugging Face model IDs on Linux. |

The README is the user-facing source of truth for supported `--model` values. Implementation constants live in `aimd.asr.const`, `aimd.ocr.mlx4ocr_engine`, and `aimd.ocr.transformers_engine`.

For performance expectations and benchmarking guidance, see [Performance](performance.md).

## Error Handling

Domain errors in `aimd.core.errors` are preserved and used end-to-end:

- `UnsupportedInputError` -> 400 in HTTP adapter
- `EngineUnavailableError` -> 422 in HTTP adapter
- `InputNotFoundError` -> 404 in HTTP adapter
- `ProcessingFailedError` -> 500 in HTTP adapter
