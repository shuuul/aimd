---
name: aimd
description: Converts URLs, audio/video, documents, PDFs, and images into LLM-ready Markdown with the aimd CLI. Use when an agent needs to prepare readable text context from local files or web/media sources.
license: MIT
compatibility: Requires uv and local file or network access for the requested input. macOS Apple Silicon uses MLX backends for local ASR/OCR; Linux ASR/OCR expects CUDA-capable Transformers backends.
metadata:
  source: https://github.com/shuuul/aimd
---

# aimd

Use this skill when you need to turn source material into Markdown context for an agent or LLM workflow.

## Constraints

- Use the `aimd` CLI as the primary interface.
- Prefer `uvx --from aimd-tool aimd ...` for one-off runs when `aimd` is not installed.
- Validate local input paths before running OCR, document conversion, or transcription.
- Save output with `--output` when the generated Markdown will be reused by another agent step.
- Choose `--raw-transcript` only when timestamps/cues from subtitles are useful; otherwise use the default cleaned text.
- For restricted URLs, prefer explicit `--cookies` or `--cookies-from-browser` when the user knows the needed login source. AIMD may auto-probe browser cookies when no cookie option is supplied, but invalid explicit cookie options should be treated as errors.
- Treat a URL transcript result with no subtitle text and no audio transcription as a failure to prepare transcript context, not as useful metadata-only context.
- For sandboxed environments, set `AIMD_TEMP_DIR` or pass `--temp-dir` to a writable directory.
- Do not claim unsupported local inference: macOS local ML uses MLX; Linux local ASR/OCR expects CUDA.

## Recommended command shape

If `aimd` is installed:

```bash
aimd INPUT --output output.md
```

For one-off use without installing:

```bash
uvx --from aimd-tool aimd INPUT --output output.md
```

For the latest GitHub version before a release:

```bash
uvx --from "aimd-tool @ git+https://github.com/shuuul/aimd.git@main" \
  aimd INPUT --output output.md
```

Install for repeated use:

```bash
uv tool install aimd-tool
aimd --help
```

## Prepare Markdown from common inputs

Auto-detect input type:

```bash
aimd audio.mp3 --output transcript.md
aimd video.mp4 --output transcript.md
aimd "https://youtube.com/watch?v=..." --output page.md
aimd document.epub --output document.md
aimd document.pdf --output document.md
aimd scan.pdf --output scan.md
aimd page.png --output page.md
aimd notes.txt --output notes.md
```

Use `uvx` by prefixing the same command:

```bash
uvx --from aimd-tool aimd document.pdf --output document.md
```

## URL and transcript sources

Extract readable Markdown or subtitles from supported URLs:

```bash
aimd "https://youtube.com/watch?v=..." --output transcript.md
aimd "https://www.bilibili.com/video/BV..." --output transcript.md
aimd "https://www.xiaoyuzhoufm.com/episode/..." --output episode.md
```

For authenticated or restricted media:

```bash
aimd "https://youtube.com/watch?v=..." --cookies cookies.txt --output transcript.md
aimd "https://youtube.com/watch?v=..." --cookies-from-browser chrome --output transcript.md
```

If the user does not provide cookies, AIMD may try available browser cookie sources automatically for convenience. Do not hide an invalid explicit cookie option or a failed audio transcription fallback; surface the error so the user can choose the right credential source or backend.

Preserve original subtitle formatting only when needed:

```bash
aimd "https://youtube.com/watch?v=..." --raw-transcript --output subtitles.md
```

## Audio and video transcription

Transcription backend selection is automatic based on the current platform:

```bash
aimd lecture.mp3 --output lecture.md
aimd interview.m4a --language zh --output interview.md
aimd video.mp4 --output video.md
```

Model examples:

```bash
aimd audio.wav --model mlx-community/Qwen3-ASR-1.7B-4bit --output transcript.md # macOS Apple Silicon
aimd audio.wav --model Qwen/Qwen3-ASR-1.7B --output transcript.md               # Linux/CUDA
```

## Documents, PDFs, and text files

Convert documents to Markdown:

```bash
aimd document.epub --output document.md
aimd document.pdf --output document.md
aimd notes.md --output notes.md
aimd notes.txt --output notes.md
```

EPUB and asset-bearing documents may produce a structured output directory with Markdown plus extracted assets. Use an explicit output path so later agent steps know where to read from.

## OCR for scanned PDFs and images

Images route to OCR automatically. PDFs with a text layer are converted as documents; scanned PDFs route to OCR when text-layer detection is available.

```bash
aimd page.png --output page.md
aimd scan.pdf --output scan.md
aimd scan.pdf --start 0 --end 2 --output scan-pages.md
```

OCR backend selection is automatic based on the current platform:

```bash
aimd scan.pdf --model paddleocr_v6 --output scan.md       # macOS Apple Silicon
aimd scan.pdf --model got_ocr --output scan.md            # Linux/CUDA
aimd scan.pdf --model unlimited_ocr --output scan.md      # Linux/CUDA
```

## Temporary directory handling

Use a writable temp directory when processing large media, sandboxed filesystems, or environments where `/tmp` may be restricted:

```bash
AIMD_TEMP_DIR=/path/to/writable/tmp aimd audio.mp3 --output transcript.md
aimd scan.pdf --temp-dir /path/to/writable/tmp --output scan.md
```

## Before running

1. Confirm `uv` or `aimd` is available:

   ```bash
   uv --version
   aimd --help
   ```

2. For local files, confirm the input exists:

   ```bash
   test -e /path/to/input
   ```

3. Pick the smallest useful output target, usually a Markdown file via `--output`.
4. For ASR/OCR, choose a backend compatible with the current machine.
5. After the command completes, pass the saved Markdown path to the next agent step.
