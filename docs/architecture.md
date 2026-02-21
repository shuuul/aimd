# Architecture

## Overview

The codebase follows a ports/adapters structure:

- `aimd.application`: use-cases, canonical request/response models, and bootstrap wiring.
- `aimd.infrastructure`: concrete implementations for transcription, URL extraction, document conversion, and engine capabilities.
- `aimd.adapters`: interface layers for CLI, HTTP API, and MCP.

## Dependency Rules

- Adapters may depend on application and infrastructure.
- Application may depend on infrastructure through explicit wiring in `application/bootstrap.py`.
- Infrastructure does not import adapters.

## Primary Flow

1. Adapter receives request payload/options.
2. Adapter builds `ProcessInput` and calls `ProcessInputUseCase.execute`.
3. Use-case routes request to transcript or convert pipeline.
4. Infrastructure modules perform external-tool work (yt-dlp, pandoc, whisper backends).
5. Adapter maps `ProcessResult` to interface-specific response/output.

## Error Handling

Domain errors in `aimd.errors` are preserved and used end-to-end:

- `UnsupportedInputError` -> 400 in HTTP adapter
- `EngineUnavailableError` -> 422 in HTTP adapter
- `InputNotFoundError` -> 404 in HTTP adapter
- `ProcessingFailedError` -> 500 in HTTP adapter
