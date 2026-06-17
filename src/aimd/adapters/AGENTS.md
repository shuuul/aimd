# src/aimd/adapters

Interface adapters for CLI, HTTP API, and MCP.

## Responsibilities

- Parse interface-level input.
- Map payloads/options to `application.models.ProcessInput`.
- Map use-case output/errors to interface response format.
- Expose engine/model choices without duplicating backend validation logic.

## Current Interfaces

- CLI (`cli/app.py`): `aimd <input_source>` with `--output`, `--engine`, `--model`, `--language`, `--save-original`, `--cookies`, `--cookies-from-browser`, `--log-level`, `--raw-transcript`, and `--temp-dir` / `AIMD_TEMP_DIR`.
- HTTP (`http/app.py`): `/healthz`, `/v1/engines`, `/v1/process`; temp directory comes from `AIMD_TEMP_DIR`, not request payload.
- MCP (`mcp/server.py`): `healthz`, `list_engines`, `process_input`; temp directory comes from `AIMD_TEMP_DIR`.

## Rules

- Do not implement core processing logic in adapters.
- Reuse shared output persistence helpers from `application/services`.
- Keep CLI/API/MCP option descriptions aligned with `const.MLX_AUDIO_MODELS`, `const.QWEN_ASR_MODELS`, and `ProcessInput`.
- Do not import infrastructure processing modules directly; adapters should call application use-cases.
