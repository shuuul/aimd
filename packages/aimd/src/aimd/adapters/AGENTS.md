# packages/aimd/src/aimd/adapters

Interface adapters for the core CLI package.

## Responsibilities

- Parse interface-level input.
- Map payloads/options to `application.models.ProcessInput`.
- Map use-case output/errors to CLI output format.
- Expose engine/model choices without duplicating backend validation logic.

## Current Interfaces

- CLI (`cli/app.py`): `aimd <input_source>` with `--output`, `--engine`, `--model`, `--language`, `--save-original`, `--cookies`, `--cookies-from-browser`, `--log-level`, `--raw-transcript`, and `--temp-dir` / `AIMD_TEMP_DIR`.
- HTTP lives in the `aimd-api` package.
- MCP lives in the `aimd-mcp` package.

## Rules

- Do not implement core processing logic in adapters.
- Reuse shared output persistence helpers from `application/services`.
- Keep CLI option descriptions aligned with ASR plugin model constants and `ProcessInput`.
- Do not import infrastructure processing modules directly; adapters should call application use-cases.
