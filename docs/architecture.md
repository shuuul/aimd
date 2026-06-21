# Architecture

## Overview

The repository is a uv workspace with feature packages under `packages/`.
The main `aimd` package uses MarkItDown as the local-file conversion contract and follows a ports/adapters structure:

- `aimd.application`: use-cases, canonical request/response models, and bootstrap wiring.
- `aimd.infrastructure`: the MarkItDown runner, media package adapter, and Markdown chunking helpers.
- `aimd.adapters`: CLI interface layer.
- `aimd-api`: FastAPI service package.
- `aimd-mcp`: MCP stdio server package.
- `aimd-media`: yt-dlp URL extraction, subtitle-first/audio-ASR fallback, MarkItDown plugin for local audio/video, transcoding, and engine capability preflight.
- `aimd-book`: MarkItDown plugin for ebook spine/image extraction and Markdown cleanup. The current implementation is EPUB-compatible and routes `.epub`, `.mobi`, and `.azw3` as book inputs for future format-specific handling.
- `aimd-ocr`: OCR plugin scaffold for the next feature.
- `aimd-html`: Python wrapper around the Defuddle CLI.

## Dependency Rules

- Interface packages/adapters depend on application use-cases, not infrastructure processing modules.
- Application local-file processing should go through `MarkItDown(enable_plugins=True)` rather than calling feature package processors directly.
- Infrastructure does not import adapters.
- Feature packages should follow MarkItDown's plugin/converter contract and avoid depending on `aimd` internals.
- Output destinations are interface concerns. `ProcessInput` does not carry `output_file`; CLI/API/MCP persist requested outputs through shared helpers in `aimd.application.services.output_writer`.
- API/MCP response/request payload shaping lives in `aimd.application.services.interface_payloads` as plain mapping helpers; it must not depend on FastAPI or MCP types.

## Primary Flow

1. CLI/API/MCP adapter receives request payload/options.
2. Adapter builds `ProcessInput` with shared mapping helpers and calls `ProcessInputUseCase.execute`.
3. Use-case routes request by `InputRoute(source_kind, task_type)` to a task processor.
4. `aimd-media` performs URL media extraction; local files are converted by MarkItDown plus installed `aimd-*` plugins.
5. Adapter maps `ProcessResult` to interface-specific response/output and persists `output_file` if requested.

```diagram
╭──────────────╮     ╭──────────────────────╮     ╭────────────────────╮
│ CLI/API/MCP  │────▶│ ProcessInputUseCase  │────▶│ TaskProcessor      │
│ adapters     │     │ route + dispatch     │     │ transcript/convert │
╰──────┬───────╯     ╰──────────────────────╯     ╰─────────┬──────────╯
       │                                                     │
       │                                                     ▼
       │                                          ╭────────────────────╮
       │                                          │ aimd-media or      │
       │                                          │ MarkItDown plugins │
       │                                          ╰─────────┬──────────╯
       ▼                                                    ▼
╭────────────────────╮                           ╭────────────────────╮
│ output_writer +    │◀──────────────────────────│ ProcessResult      │
│ interface_payloads │                           │ TextContext shape  │
╰────────────────────╯                           ╰────────────────────╯
```

## Error Handling

Domain errors in `aimd.errors` are preserved and used end-to-end:

- `UnsupportedInputError` -> 400 in HTTP adapter
- `EngineUnavailableError` -> 422 in HTTP adapter
- `InputNotFoundError` -> 404 in HTTP adapter
- `ProcessingFailedError` -> 500 in HTTP adapter
