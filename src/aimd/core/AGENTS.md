# src/aimd/core

Core package organized as a small interface-independent processing service.

## STRUCTURE

- `models.py` — canonical request/response models
- `process.py` — input routing, processing orchestration, MarkItDown runner, and Markdown shaping helpers

Feature modules are bundled in the same `aimd-tool` distribution:

- `aimd.plugins.url` — MarkItDown plugin for URL transcript extraction, yt-dlp subtitles/cookies/audio fallback, and opt-in Defuddle readable HTML
- `aimd.plugins.asr` — MarkItDown plugin for local audio/video transcription, ASR backends, and capability checks
- `aimd.interfaces.api` — FastAPI service module
- `aimd.interfaces.mcp` — MCP stdio server module
- `aimd.plugins.doc` — MarkItDown plugin for Pandoc-backed document conversion
- `aimd.plugins.ocr` — MarkItDown plugin for OCR extraction

## CONVENTIONS

- Keep routing and orchestration in `process.py`.
- Keep IO/third-party integrations in `process.py` and owning feature packages.
- Keep interface-specific request/response mapping in the interface module (`aimd.interfaces.cli`, `aimd.interfaces.api`, or `aimd.interfaces.mcp`).

- Keep platform-dependent dependency use behind capability checks; `mlx-audio` is Darwin-only and Qwen3-ASR runs through Transformers on CUDA-capable non-Darwin platforms.
- Keep URL and local-file capabilities behind MarkItDown plugins in their owning submodules (`aimd.plugins.url`, `aimd.plugins.asr`, `aimd.plugins.doc`, `aimd.plugins.ocr`); core should call MarkItDown instead of feature internals.
- Keep output file persistence in interfaces via `aimd.interfaces.output`; `ProcessInput` and `ProcessResult` do not carry `output_file`.
- Keep API/MCP payload mapping in the API/MCP modules.

## DOCUMENT CONVERSION

`aimd.plugins.doc` owns Pandoc-backed document conversion through the MarkItDown plugin entry point. Core sends Pandoc-supported local document extensions through MarkItDown. EPUB gets an image/chapter output directory and uses the custom ZIP/spine pipeline; DOCX/ODT may also receive an asset directory for extracted media.

## TEMP DIRECTORY

All temporary file operations (audio downloads, document extraction) use
Python's `tempfile` module with a configurable base directory:

- **CLI**: `--temp-dir` option (also reads `AIMD_TEMP_DIR` env var)
- **MCP / HTTP**: reads `AIMD_TEMP_DIR` env var at request time
- **Default**: when unset, falls back to the system temp directory (`/tmp`)

In sandboxed environments where `/tmp` may not be writable, set `AIMD_TEMP_DIR`
to redirect temp I/O. The `temp_dir` field flows through `ProcessInput` →
use-cases → infrastructure functions via the `dir=` parameter of
`tempfile.TemporaryDirectory` and `tempfile.NamedTemporaryFile`. ASR temp files are implemented in `aimd.plugins.asr`; document extraction temp files are implemented in `aimd.plugins.doc`.

## SUBTITLE FORMATTING

URL-sourced subtitles (SRT/VTT/TTML) are simplified to plain text by default.
The `raw_transcript` field on `ProcessInput` (default `False`) controls this:

- **CLI**: `--raw-transcript` flag
- **HTTP**: `raw_transcript` field in `ProcessRequest`
- **MCP**: `raw_transcript` parameter on `process_input` tool

The stripping is performed by `strip_subtitle_formatting()` in
`aimd.plugins.url.markdown`, applied in `aimd.plugins.url._plugin` before
`format_content()` embeds the text into the markdown output.

## TRANSCRIPTION MODELS

- Backend selection is platform-driven and not user-configurable.
- The MLX backend is implemented in `aimd.plugins.asr.models.mlx` and uses `mlx_audio.stt.load()` on Apple Silicon. The default remains `mlx-community/Qwen3-ASR-1.7B-4bit`; `const.MLX_AUDIO_MODELS` also tracks newer mlx-audio 0.4.4 STT IDs. Do not add forced-aligner models to this list unless the calling code also supplies reference text.
- The Transformers ASR backend is implemented in `aimd.plugins.asr.models.transformers` and uses Qwen3-ASR on CUDA-capable non-Darwin platforms with `Qwen/Qwen3-ASR-1.7B` default and `Qwen/Qwen3-ASR-0.6B` as the lower-memory option.
- mlx Qwen3-ASR defaults omitted language to `Chinese`; other mlx-audio STT models stay on their own default/auto language behavior.
