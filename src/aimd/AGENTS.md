# src/aimd

Core package organized with ports/adapters architecture.

## STRUCTURE

- `application/` — use-cases, canonical request/response models, bootstrap wiring
- `infrastructure/` — concrete implementations (capabilities, transcription, URL, documents)
- `adapters/` — CLI, HTTP API, MCP interface adapters
- `cli.py`, `api.py`, `mcp.py` — runtime entrypoints

## CONVENTIONS

- Keep orchestration in `application/use_cases/*`.
- Keep IO/third-party integrations in `infrastructure/*`.
- Keep interface-specific request/response mapping in `adapters/*`.

## TEMP DIRECTORY

All temporary file operations (audio downloads, yap output, EPUB extraction) use
Python's `tempfile` module with a configurable base directory:

- **CLI**: `--temp-dir` option (also reads `AIMD_TEMP_DIR` env var)
- **MCP / HTTP**: reads `AIMD_TEMP_DIR` env var at request time
- **Default**: when unset, falls back to the system temp directory (`/tmp`)

This is critical for sandboxed environments (e.g. spacebot) where `/tmp` may not
be writable. The `temp_dir` field flows through `ProcessInput` → use-cases →
infrastructure functions via the `dir=` parameter of `tempfile.TemporaryDirectory`
and `tempfile.NamedTemporaryFile`.

## SUBTITLE FORMATTING

URL-sourced subtitles (SRT/VTT/TTML) are simplified to plain text by default.
The `raw_transcript` field on `ProcessInput` (default `False`) controls this:

- **CLI**: `--raw-transcript` flag
- **HTTP API**: `raw_transcript` field in `ProcessRequest`
- **MCP**: `raw_transcript` parameter on `process_input` tool

The stripping is performed by `strip_subtitle_formatting()` in
`infrastructure/url/formatter.py`, applied in `processor.py` before
`format_content()` embeds the text into the markdown output.
