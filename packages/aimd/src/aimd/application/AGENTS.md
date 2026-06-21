# packages/aimd/src/aimd/application

Application layer: orchestration and canonical models.

## Responsibilities

- Define request/response models used across interfaces.
- Implement use-cases (`process_input`, `input_routing`, `list_engines`) and task processors under `use_cases/processors/`.
- Wire dependencies explicitly in `bootstrap.py`.
- Keep output persistence in `services/output_writer.py` so CLI/API/MCP share identical file behavior.

## Current Flow

- `ProcessInput` carries `input_source`, output path, engine/model/language, URL cookie options, `save_original`, `raw_transcript`, and `temp_dir`.
- `input_routing.py` returns `InputRoute(source_kind, task_type)`.
- `source_kind` describes what the user supplied: `url`, `audio_file`, `video_file`, `document_file`, or `unknown`.
- `task_type` describes which processing task runs: currently `transcript` or `convert`.
- `process_input.py` is the facade/router: it dispatches `InputRoute` to configured `TaskProcessor` objects.
- `processors/transcript.py` owns URL/audio transcript flow and resolves transcription engines early.
- `processors/convert.py` owns local document conversion flow through MarkItDown.
- `ListEnginesUseCase` returns capability detector output for adapter `/v1/engines` and MCP `list_engines`.

## Rules

- No direct adapter imports.
- Keep business flow in use-cases; avoid embedding third-party integration logic here.
- Keep each task implementation in its own processor module; local file conversion should call the MarkItDown runner rather than feature package internals.
- Route by the pair `(source_kind, task_type)`; avoid collapsing source classification into task naming.
- Keep dependency registration explicit in `bootstrap.py`; do not add a custom aimd plugin registry or priority routing until requirements justify it.
- Do not hard-code platform/model availability in application code; use infrastructure capability detection and constants.
- Preserve the `TextContext(title, chunk_list, split_header_level)` contract across interfaces.
