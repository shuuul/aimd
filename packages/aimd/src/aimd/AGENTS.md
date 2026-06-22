# packages/aimd/src/aimd

Core package organized with ports/adapters architecture.

## STRUCTURE

- `application/` — use-cases, canonical request/response models, bootstrap wiring
- `infrastructure/` — MarkItDown runner, media package adapter, and Markdown chunking helpers
- `adapters/` — CLI interface adapter
- `cli.py` — runtime entrypoint

Feature packages live beside this package:

- `packages/aimd-media/` — yt-dlp URLs, subtitles, audio fallback, ASR plugin, and capability checks
- `packages/aimd-api/` — FastAPI service package
- `packages/aimd-mcp/` — MCP stdio server package
- `packages/aimd-book/` — MarkItDown plugin for ebook extraction and cleanup
- `packages/aimd-ocr/` — OCR plugin scaffold
- `packages/aimd-html/` — Defuddle CLI wrapper

## CONVENTIONS

- Keep orchestration in `application/use_cases/*`.
- Keep IO/third-party integrations in `infrastructure/*`.
- Keep interface-specific request/response mapping in `adapters/*`.

- Keep platform-dependent dependency use behind capability checks; `mlx-audio` is Darwin-only and Qwen3-ASR runs through Transformers on Linux/CUDA.
- Keep heavy local-file integrations behind MarkItDown plugins; keep this package as the facade/router and `TextContext` wrapper.
- Keep output file persistence in adapters via `application/services/output_writer.py`; `ProcessInput` and `ProcessResult` do not carry `output_file`.
- Keep API/MCP payload mapping in `application/services/interface_payloads.py` as plain helpers with no FastAPI/MCP imports.

## BOOK CONVERSION

`aimd-book` owns ebook conversion through the MarkItDown plugin entry point. Core routing treats `.epub`, `.mobi`, and `.azw3` as book inputs via `BOOK_EXTENSIONS`, but the current book pipeline is EPUB-compatible ZIP/spine extraction. Add true non-EPUB handling inside `aimd-book` rather than special-casing it in `aimd.infrastructure` or adapters.

## TEMP DIRECTORY

All temporary file operations (audio downloads, ebook extraction) use
Python's `tempfile` module with a configurable base directory:

- **CLI**: `--temp-dir` option (also reads `AIMD_TEMP_DIR` env var)
- **MCP / HTTP**: reads `AIMD_TEMP_DIR` env var at request time
- **Default**: when unset, falls back to the system temp directory (`/tmp`)

In sandboxed environments where `/tmp` may not be writable, set `AIMD_TEMP_DIR`
to redirect temp I/O. The `temp_dir` field flows through `ProcessInput` →
use-cases → infrastructure functions via the `dir=` parameter of
`tempfile.TemporaryDirectory` and `tempfile.NamedTemporaryFile`. ASR temp files are implemented in `aimd-media`; ebook extraction temp files are implemented in `aimd-book`.

## SUBTITLE FORMATTING

URL-sourced subtitles (SRT/VTT/TTML) are simplified to plain text by default.
The `raw_transcript` field on `ProcessInput` (default `False`) controls this:

- **CLI**: `--raw-transcript` flag
- **HTTP API**: `raw_transcript` field in `ProcessRequest`
- **MCP**: `raw_transcript` parameter on `process_input` tool

The stripping is performed by `strip_subtitle_formatting()` in
`aimd_media.url.formatter`, applied in `aimd_media.url.processor` before
`format_content()` embeds the text into the markdown output.

## TRANSCRIPTION MODELS

- Engine names are fixed in `const.TRANSCRIPTION_ENGINES`: `auto`, `mlx`, `qwen`.
- `mlx` is implemented in `aimd_media.mlx_engine` and uses `mlx_audio.stt.load()` on Apple Silicon. The default remains `mlx-community/Qwen3-ASR-1.7B-4bit`; `const.MLX_AUDIO_MODELS` also tracks newer mlx-audio 0.4.4 STT IDs. Do not add forced-aligner models to this list unless the calling code also supplies reference text.
- `qwen` is implemented in `aimd_media.qwen_engine` and uses a direct Transformers backend on Linux/CUDA with `Qwen/Qwen3-ASR-1.7B` default and `Qwen/Qwen3-ASR-0.6B` as the lower-memory option.
- mlx Qwen3-ASR defaults omitted language to `Chinese`; other mlx-audio STT models stay on their own default/auto language behavior.
