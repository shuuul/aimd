# src/aimd/infrastructure

Infrastructure layer: concrete integrations and processing pipelines.

## Responsibilities

- Capabilities detection and engine resolution.
- Transcription backends (yap/mlx/faster-whisper).
- URL extraction (yt-dlp, cookies, subtitles, audio fallback).
- Document processing (pandoc, EPUB extraction, chunking/title logic).

## Rules

- Do not import adapters.
- Keep modules focused and small by pipeline concern.
- Raise domain errors from `aimd.errors`.
