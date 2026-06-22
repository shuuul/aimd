# Architecture

## Overview

The repository has one published distribution, `aimd-tool`, with source code under `src/aimd`.
The `aimd` package uses MarkItDown as the local-file conversion contract and follows a ports/adapters structure:

- `aimd.core.application`: use-cases, canonical request/response models, and bootstrap wiring.
- `aimd.core.infrastructure`: the MarkItDown runner, media package adapter, and Markdown chunking helpers.
- `aimd.core.adapters`: CLI interface layer.
- `aimd.api`: FastAPI-backed HTTP API module.
- `aimd.mcp`: MCP stdio server module.
- `aimd.media`: yt-dlp URL extraction, subtitle-first/audio-ASR fallback, MarkItDown plugin for local audio/video, transcoding, and engine capability preflight.
- `aimd.book`: MarkItDown plugin for ebook spine/image extraction and Markdown cleanup. The current implementation is EPUB-compatible and routes `.epub`, `.mobi`, and `.azw3` as book inputs for future format-specific handling.
- `aimd.ocr`: OCR plugin scaffold for the next feature.
- `aimd.clip`: Python wrapper around the Defuddle CLI.

## Dependency Rules

- Interface modules/adapters depend on application use-cases, not infrastructure processing modules.
- Application local-file processing should go through `MarkItDown(enable_plugins=True)` rather than calling bundled plugin processors directly.
- Infrastructure does not import adapters.
- Feature modules should follow MarkItDown's plugin/converter contract and avoid depending on `aimd` internals.
- Output destinations are interface concerns. `ProcessInput` does not carry `output_file`; CLI/API/MCP persist requested outputs through shared helpers in `aimd.core.application.services.output_writer`.
- API/MCP response/request payload shaping lives in `aimd.core.application.services.interface_payloads` as plain mapping helpers; it must not depend on FastAPI or MCP types.

## Primary Flow

1. CLI/API/MCP adapter receives request payload/options.
2. Adapter builds `ProcessInput` with shared mapping helpers and calls `ProcessInputUseCase.execute`.
3. Use-case routes request by `InputRoute(source_kind, task_type)` to a task processor.
4. `aimd.media` performs URL media extraction; local files are converted by MarkItDown plus bundled plugins.
5. Adapter maps `ProcessResult` to interface-specific response/output and persists `output_file` if requested.

```diagram
╭──────────────╮     ╭──────────────────────╮     ╭────────────────────╮
│ CLI/API/MCP │────▶│ ProcessInputUseCase  │────▶│ TaskProcessor      │
│ adapters     │     │ route + dispatch     │     │ transcript/convert │
╰──────┬───────╯     ╰──────────────────────╯     ╰─────────┬──────────╯
       │                                                     │
       │                                                     ▼
       │                                          ╭────────────────────╮
       │                                          │ media or      │
       │                                          │ MarkItDown plugins │
       │                                          ╰─────────┬──────────╯
       ▼                                                    ▼
╭────────────────────╮                           ╭────────────────────╮
│ output_writer +    │◀──────────────────────────│ ProcessResult      │
│ interface_payloads │                           │ TextContext shape  │
╰────────────────────╯                           ╰────────────────────╯
```

## Error Handling

Domain errors in `aimd.core.errors` are preserved and used end-to-end:

- `UnsupportedInputError` -> 400 in HTTP adapter
- `EngineUnavailableError` -> 422 in HTTP adapter
- `InputNotFoundError` -> 404 in HTTP adapter
- `ProcessingFailedError` -> 500 in HTTP adapter
