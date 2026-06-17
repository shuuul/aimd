# src/aimd/application

Application layer: orchestration and canonical models.

## Responsibilities

- Define request/response models used across interfaces.
- Implement use-cases (`process_input`, `list_engines`).
- Wire dependencies explicitly in `bootstrap.py`.
- Keep output persistence in `services/output_writer.py` so CLI/API/MCP share identical file behavior.

## Current Flow

- `ProcessInput` carries `input_source`, output path, engine/model/language, URL cookie options, `save_original`, `raw_transcript`, and `temp_dir`.
- `ProcessInputUseCase` detects URL/file and transcript/convert task type, resolves transcription engine early, then dispatches to infrastructure processors.
- `ListEnginesUseCase` returns capability detector output for adapter `/v1/engines` and MCP `list_engines`.

## Rules

- No direct adapter imports.
- Keep business flow in use-cases; avoid embedding third-party integration logic here.
- Do not hard-code platform/model availability in application code; use infrastructure capability detection and constants.
- Preserve the `TextContext(title, chunk_list, split_header_level)` contract across interfaces.
