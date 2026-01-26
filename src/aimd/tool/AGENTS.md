# src/aimd/tool

Processing modules for audio, file, and URL input types.

## OVERVIEW

Three async processing modules. Each returns `TextContext(title, chunk_list, split_header_level)`.

## STRUCTURE

```
src/aimd/tool/
├── __init__.py   # Public API exports
├── audio.py      # Transcription engines (370 lines)
├── file.py       # Pandoc conversion, EPUB extraction (631 lines)
└── url.py        # yt-dlp integration, subtitle extraction (630 lines)
```

## WHERE TO LOOK

| Task | File | Key Functions |
|------|------|---------------|
| Audio transcription | `audio.py` | `get_text_from_audio()`, `transcribe_audio_yap/mlx/cuda/cpu()` |
| Document conversion | `file.py` | `get_text_from_file()`, `process_epub_with_images()` |
| URL extraction | `url.py` | `get_text_from_url()`, `_extract_subtitles()` |

## CONVENTIONS

- **Return type**: All functions return `TextContext`
- **Async only**: No synchronous processing in tool modules
- **Chunking**: Large outputs split into ~40k char chunks
- **EPUB structure**: `book_name/{book_name.md, chapters/, images/}`

## ANTI-PATTERNS (THIS PROJECT)

- **Fragile encoding**: `yap` fallback tries UTF-8→GB2312→Latin-1
- **No header fallback**: `file.py` throws `RuntimeError` if no H1-H6 found
- **Hardcoded priorities**: Subtitle formats/languages not configurable
- **Runtime imports**: `mlx_whisper`, `faster_whisper` imported inside functions

## ENGINE MATRIX

| Engine | Platform | Hardware | Dependency |
|--------|----------|----------|------------|
| `yap` | macOS | Any | External CLI |
| `mlx` | macOS | Apple Silicon (M1-M4) | mlx-whisper |
| `cuda` | Linux/Windows | NVIDIA GPU | faster-whisper + CUDA |
| `cpu` | All | CPU | faster-whisper |

## NOTES

- `url.py` uses Chrome cookies for YouTube/Bilibili extraction
- `file.py` uses Pandoc for 40+ format conversions
- Subtitle extraction skipped if not YouTube/Bilibili/unknown platform
