# src/aimd/infrastructure

Infrastructure layer: concrete integrations and processing pipelines.

## Responsibilities

- Capabilities detection and engine resolution.
- Transcription backends (mlx-audio Qwen3-ASR / yap / faster-whisper).
- URL extraction (yt-dlp, cookies, subtitles, audio fallback).
- Document processing (pandoc, EPUB extraction, chunking/title logic).

## Rules

- Do not import adapters.
- Keep modules focused and small by pipeline concern.
- Raise domain errors from `aimd.errors`.
- When using `tempfile.TemporaryDirectory` or `tempfile.NamedTemporaryFile`,
  always pass the `dir=temp_dir` parameter so callers can redirect temp I/O
  to a sandbox-safe location via `AIMD_TEMP_DIR`.
