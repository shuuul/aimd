# src/aimd

Core package organized with ports/adapters architecture.

## STRUCTURE

- `application/` — use-cases, canonical request/response models, bootstrap wiring
- `infrastructure/` — concrete implementations (capabilities, transcription, URL, documents)
- `adapters/` — CLI, HTTP API, MCP interface adapters
- `platform_utils.py` — platform probes used by capability preflight
- `cli.py`, `api.py`, `mcp.py` — runtime entrypoints

## CONVENTIONS

- Keep orchestration in `application/use_cases/*`.
- Keep IO/third-party integrations in `infrastructure/*`.
- Keep interface-specific request/response mapping in `adapters/*`.

- Keep platform-dependent dependency use behind capability checks; `mlx-audio` is Darwin-only and `qwen-asr` is Linux-only.

## TEMP DIRECTORY

All temporary file operations (audio downloads, EPUB extraction) use
Python's `tempfile` module with a configurable base directory:

- **CLI**: `--temp-dir` option (also reads `AIMD_TEMP_DIR` env var)
- **MCP / HTTP**: reads `AIMD_TEMP_DIR` env var at request time
- **Default**: when unset, falls back to the system temp directory (`/tmp`)

In sandboxed environments where `/tmp` may not be writable, set `AIMD_TEMP_DIR`
to redirect temp I/O. The `temp_dir` field flows through `ProcessInput` →
use-cases → infrastructure functions via the `dir=` parameter of
`tempfile.TemporaryDirectory` and `tempfile.NamedTemporaryFile`.

## SUBTITLE FORMATTING

URL-sourced subtitles (SRT/VTT/TTML) are simplified to plain text by default.
The `raw_transcript` field on `ProcessInput` (default `False`) controls this:

- **CLI**: `--raw-transcript` flag
- **HTTP API**: `raw_transcript` field in `ProcessRequest`
- **MCP**: `raw_transcript` parameter on `process_input` tool

The stripping is performed by `strip_subtitle_formatting()` in
`infrastructure/url/formatter.py`, applied in `processor.py` before
`format_content()` embeds the text into the markdown output.

## TRANSCRIPTION MODELS

- Engine names are fixed in `const.TRANSCRIPTION_ENGINES`: `auto`, `mlx`, `qwen`.
- `mlx` uses `mlx_audio.stt.load()` on Apple Silicon. The default remains `mlx-community/Qwen3-ASR-1.7B-4bit`; `const.MLX_AUDIO_MODELS` also tracks newer mlx-audio 0.4.4 STT IDs. Do not add forced-aligner models to this list unless the calling code also supplies reference text.
- `qwen` uses `qwen_asr.Qwen3ASRModel` on Linux/CUDA with `Qwen/Qwen3-ASR-1.7B` default and `Qwen/Qwen3-ASR-0.6B` as the lower-memory option.
- mlx Qwen3-ASR defaults omitted language to `Chinese`; other mlx-audio STT models stay on their own default/auto language behavior.
