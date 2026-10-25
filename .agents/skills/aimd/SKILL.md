---
name: aimd
description: Converts URLs, audio/video, documents, PDFs, and images into LLM-ready Markdown with the aimd CLI. Use when an agent needs to prepare readable text context from local files or web/media sources.
license: Apache-2.0
compatibility: Requires the Homebrew `aimd` formula (macOS Apple Silicon) and local file or network access for the requested input. Local ASR/OCR runs on MLX (Metal GPU); elsewhere use the OpenAI-compatible remote backends.
metadata:
  source: https://github.com/shuuul/aimd
---

# aimd

Use this skill when you need to turn source material into Markdown context for an agent or LLM workflow.

## Constraints

- Use the `aimd` CLI as the primary interface.
- Install with Homebrew when `aimd` is missing: `brew install shuuul/tap/aimd`
  (pulls in mlx, pandoc, ffmpeg, mupdf, and yt-dlp).
- Validate local input paths before running OCR, document conversion, or transcription.
- Save output with `--output` when the generated Markdown will be reused by another agent step.
- Choose `--raw-transcript` only when timestamps/cues from subtitles are useful; otherwise use the default cleaned text.
- For restricted URLs, prefer explicit `--cookies` or `--cookies-from-browser` when the user knows the needed login source. AIMD may auto-probe browser cookies when no cookie option is supplied, but invalid explicit cookie options should be treated as errors.
- Treat a URL transcript result with no subtitle text and no audio transcription as a failure to prepare transcript context, not as useful metadata-only context.
- For sandboxed environments, set `AIMD_TEMP_DIR` or pass `--temp-dir` to a writable directory.
- Do not claim unsupported local inference: local ASR/OCR needs the MLX Metal GPU. Without it, point `--asr-base-url` / `--ocr-base-url` (or `AIMD_ASR_BASE_URL` / `AIMD_OCR_BASE_URL`) at an OpenAI-compatible server.
- Supported precisions are `4bit` (default), `6bit`, and `8bit`; `bf16` is not supported by the native engine yet.

## Recommended command shape

```bash
aimd INPUT --output output.md
```

Install for repeated use:

```bash
brew install shuuul/tap/aimd
aimd --help
```

Model weights download to the Hugging Face cache on first use.

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

Force a route with `--task transcript|convert|ocr` when auto-routing picks the wrong one.

## URL and transcript sources

Extract subtitles (or transcribe audio as a fallback) from supported URLs:

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

If the user does not provide cookies, AIMD may try available browser cookie sources automatically for convenience. Do not hide an invalid explicit cookie option or a failed audio transcription fallback; surface the error so the user can choose the right credential source or backend. If yt-dlp itself is outdated, `brew upgrade yt-dlp`.

Preserve original subtitle formatting only when needed:

```bash
aimd "https://youtube.com/watch?v=..." --raw-transcript --output subtitles.md
```

## Audio and video transcription

```bash
aimd lecture.mp3 --language en --output lecture.md
aimd interview.m4a --language zh --output interview.md
aimd video.mp4 --context "Names: Alice Zhang, Kubernetes" --output video.md
```

Pass `--language` explicitly; local ASR assumes Chinese when it is omitted.

Model examples:

```bash
aimd audio.wav --model qwen3-asr-1.7b --precision 4bit --output transcript.md
aimd audio.wav --model qwen3-asr-0.6b --precision 8bit --output transcript.md
```

## Documents, PDFs, and text files

```bash
aimd document.epub --output document.md
aimd report.docx --output report.md
aimd document.pdf --output document.md
aimd notes.md --output notes.md
```

Pandoc-readable formats (docx/pptx/xlsx/odt/epub/html/rtf/rst/…) are supported; legacy binary `.doc`/`.ppt`/`.xls` are not. EPUB output may include extracted images next to the Markdown; use an explicit output path so later agent steps know where to read from.

## OCR for scanned PDFs and images

Images route to OCR automatically. PDFs with a clean text layer are converted as documents; scanned PDFs and PDFs with broken font encodings route to OCR automatically.

```bash
aimd page.png --output page.md
aimd scan.pdf --output scan.md
aimd scan.pdf --start 0 --end 2 --output scan-pages.md
aimd scan.pdf --model unlimited-ocr --precision 8bit --output scan.md
```

## Temporary directory handling

```bash
AIMD_TEMP_DIR=/path/to/writable/tmp aimd audio.mp3 --output transcript.md
aimd scan.pdf --temp-dir /path/to/writable/tmp --output scan.md
```

## Before running

1. Confirm `aimd` is available: `aimd --help` (otherwise `brew install shuuul/tap/aimd`).
2. For local files, confirm the input exists: `test -e /path/to/input`.
3. Pick the smallest useful output target, usually a Markdown file via `--output`.
4. For ASR/OCR, check the machine has the MLX GPU or configure a remote backend.
5. After the command completes, pass the saved Markdown path to the next agent step.
