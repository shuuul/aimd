<div align="center">
  <img src="https://raw.githubusercontent.com/shuuul/aimd/main/assets/aimd-banner-sm.png" alt="aimd">

  <a href="https://github.com/shuuul/aimd/releases">
    <img src="https://img.shields.io/github/v/release/shuuul/aimd" alt="Latest release">
  </a>
  <a href="https://github.com/shuuul/aimd/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache License 2.0">
  </a>
</div>

# aimd

Prepare LLM-ready context from URLs, audio/video, and documents.

`aimd` takes one input, works out what kind it is, extracts or transcribes the
content, and writes Markdown. Results also carry a chunked text context for
downstream AI workflows. It is a native Rust tool: ASR and OCR run locally on
Apple Silicon through MLX (Metal), with no Python runtime.

## Highlights

- **One command** for URLs, audio/video files, EPUB and other Pandoc
  documents, text-layer PDFs, scanned PDFs and images, Markdown, and text.
- **URL transcripts** through yt-dlp (YouTube, Bilibili, podcasts, and other
  yt-dlp sites). Subtitles are used first; without them, aimd downloads the
  audio and transcribes it locally.
- **Local ASR** with Qwen3-ASR (1.7B or 0.6B) on MLX.
- **Local OCR** with Unlimited-OCR on MLX, for images and scanned PDFs.
- **Document conversion** through Pandoc. EPUB keeps spine order and extracts
  chapters and images. Text-layer PDFs are converted by
  [pdf-inspector](https://github.com/firecrawl/pdf-inspector) without OCR.
- **Remote backends**: optional OpenAI-compatible ASR/OCR servers.
- **Two interfaces**: CLI (`aimd`) and MCP stdio server (`aimd-mcp`).

## Install

```bash
brew install shuuul/tap/aimd
aimd --help
```

The formula lives in [shuuul/homebrew-tap](https://github.com/shuuul/homebrew-tap).
Bottles are built for macOS on Apple Silicon (arm64). On Linux (x86_64 or
aarch64 with an NVIDIA GPU) the formula builds from source against the tap's
`mlx-cuda`, MLX's CUDA backend. The NVIDIA driver, CUDA 13, cuDNN 9, and NCCL 2
come from the system and must be on the dynamic loader path. Arch's
`/opt/cuda` is found automatically; elsewhere set `CUDA_HOME` if the toolkit is
not in `/usr/local/cuda`.

Homebrew also installs the runtime tools aimd uses: MLX (`mlx` on macOS,
`mlx-cuda` on Linux), `ffmpeg`, `mupdf`
(`mutool`, for rendering scanned PDF pages), `pandoc`, and `yt-dlp`. Sites
change often, so keep yt-dlp current with `brew upgrade yt-dlp`. aimd suggests
this when yt-dlp's error output looks like an outdated extractor.

Model weights are not bundled. The first local ASR or OCR run downloads the
selected checkpoint from Hugging Face into the standard HF cache
(`HF_HUB_CACHE`, else `HF_HOME/hub`, else `~/.cache/huggingface/hub`).
Checkpoints already in that cache are reused.

## Quick start

```bash
# Input type is detected automatically
aimd audio.mp3
aimd "https://www.youtube.com/watch?v=..."
aimd document.epub
aimd notes.txt
aimd scan.pdf
aimd page.png

# Common options
aimd audio.mp3 --output transcript.md
aimd audio.wav --language en
aimd scan.pdf --task ocr --start 0 --end 2
aimd "https://www.youtube.com/watch?v=..." --cookies-from-browser chrome
aimd "https://www.youtube.com/watch?v=..." --raw-transcript
```

Without `--output`, the file name comes from the result title:
`<title>_transcript.md` in the current directory for transcripts, and
`<title>_converted.md` or `<title>_ocr.md` next to the input for conversions
and OCR.

## CLI usage

```text
aimd [OPTIONS] INPUT_SOURCE
```

| Option | Description |
|--------|-------------|
| `--task TEXT` | Explicit task: `transcript`, `convert`, or `ocr`. Defaults to auto-routing. |
| `-o, --output PATH` | Output file. Generated from the title when omitted. |
| `-m, --model TEXT` | ASR or OCR model (see [Models and precision](#models-and-precision)). |
| `--precision TEXT` | `4bit`, `6bit`, or `8bit` (dash variants such as `4-bit` are accepted). |
| `-l, --language TEXT` | ASR language (`zh`, `en`, `ja`, `yue`, ...). Omit it to auto-detect. OCR ignores it. |
| `--context TEXT` | ASR biasing text: proper nouns, names, domain vocabulary. |
| `--no-context` | Do not build ASR context from URL page metadata. |
| `--start INTEGER`, `--end INTEGER` | 0-based inclusive page range for OCR of PDFs. |
| `-s, --save-original PATH` | Save the downloaded audio/video of a URL to a file or directory. |
| `-c, --cookies PATH` | Netscape-format cookies file for URL extraction. |
| `--cookies-from-browser TEXT` | Browser cookie source for URL extraction, e.g. `chrome` or `chrome:default`. |
| `--raw-transcript` | Keep original SRT/VTT subtitle formatting instead of plain text. |
| `--temp-dir PATH` | Directory for temporary downloads, transcoding, and extraction (env `AIMD_TEMP_DIR`). |
| `--log-level TEXT` | `DEBUG`, `INFO` (default), `WARNING`, `ERROR`, `CRITICAL`. Logs go to stderr. |
| `--asr-base-url`, `--asr-model`, `--asr-api-key` | Remote ASR backend (env `AIMD_ASR_*`). |
| `--ocr-base-url`, `--ocr-model`, `--ocr-api-key` | Remote OCR backend (env `AIMD_OCR_*`). |

The CLI also reads `KEY=VALUE` pairs from a `.env` file in the current
directory. Variables already set in the environment take precedence.

### Audio and video files

```bash
aimd interview.m4a
aimd lecture.mp4
aimd audio.wav --language en
aimd audio.wav --model qwen3-asr-0.6b --precision 8bit
aimd interview.mp3 --context "Vocabulary: Qwen, MLX, LoRA."
```

ffmpeg decodes audio and extracts it from video. Inputs longer than 10
minutes are transcribed in segments. Without `--language`, Qwen3-ASR detects
the language itself (per segment), which also handles mixed-language audio.
Pass `--language` only to force one: `zh`, `en`, `ja`, `ko`, `de`, `es`,
`fr`, `it`, `pt`, `ru`, `ar`, `hi`, `th`, `vi`, `id`, `tr`, `nl`, `pl`,
`sv`, `da`, `fi`, `ms`, `yue`, `cs`, `fil`, `fa`, `el`, `hu`, `mk`, `ro`, or
the full language name. If a 4-bit run gets stuck in a repetition loop, aimd
retries that segment once with the matching 8-bit checkpoint.

### URLs

```bash
aimd "https://www.youtube.com/watch?v=..."
aimd "https://www.bilibili.com/video/BV..."
aimd "https://www.xiaoyuzhoufm.com/episode/..."
```

aimd uses subtitles when the page has them and simplifies them to plain text
unless you pass `--raw-transcript`. Otherwise it downloads the audio and
transcribes it. In that case the page metadata (title, author, description,
tags, chapters; at most 2000 characters) becomes ASR context so that names
and terms are recognized correctly. `--no-context` turns this off. `--context`
adds your own text, and works for local files too.

For restricted or signed-in content:

```bash
aimd "https://www.youtube.com/watch?v=..." --cookies cookies.txt
aimd "https://www.bilibili.com/video/BV..." --cookies-from-browser "chrome:default"
```

Without a cookie option, aimd tries the available browser cookie sources
before falling back to unauthenticated access. An explicit cookie option that
yt-dlp rejects fails immediately. If neither subtitles nor audio
transcription produce text, the request fails; aimd does not return
metadata-only Markdown as a transcript.

### ASR context: names and terms from descriptions

Qwen3-ASR takes free-form context text as a hint for spelling names, brands,
and jargon. It is not an instruction prompt: give it the words you expect to
hear, not commands. Context does not change which language is detected or
what gets transcribed; it only nudges ambiguous words toward the spellings
you supply.

**YouTube videos and podcast pages.** For URLs, aimd builds the context from
the page metadata automatically whenever it has to transcribe audio:

```text
Title: <title>
Author: <uploader or channel>
Description: <first 800 characters of the description>
Tags: <up to 20 tags>
Chapters: <chapter titles>
```

The whole block is capped at 2000 characters. The description is usually
the most useful part: video descriptions and podcast show notes list guest
names, products, papers, and links that the speaker then says out loud. Your
own `--context` goes in front of the metadata, so put the terms that matter
most there:

```bash
# YouTube talk: metadata context is automatic; add the names it may miss
aimd "https://www.youtube.com/watch?v=..." \
  --context "Speakers: Andrej Karpathy. Terms: nanoGPT, RLHF, tokenizer."

# Podcast episode page
aimd "https://www.xiaoyuzhoufm.com/episode/..." \
  --context "嘉宾：张三、李四。话题：MLX、Apple Silicon、端侧推理。"

# Turn metadata off when the description is unrelated (sponsor copy, link lists)
aimd "https://www.youtube.com/watch?v=..." --no-context --context "Qwen, MLX"
```

When the page already has subtitles, aimd uses them and does not run ASR,
so context has no effect there.

**Downloaded audio and local recordings** get no metadata, so pass the show
notes yourself. Keep it short and term-dense: a list of names and terms
works better than the full notes.

```bash
aimd episode.mp3 --context "Guests: Jane Doe (Acme). Terms: LoRA, KV cache, Metal."
aimd episode.mp3 --context "$(head -c 1500 shownotes.txt)"
```

Tips:

- Lead with proper nouns and domain terms; drop links, timestamps, and
  sponsor copy.
- Write names in the script the speaker uses (for example `张三` in a
  Chinese podcast, not a romanization).
- Long-audio segments all receive the same context.
- Never paste a previous transcript as context; ASR errors in it get
  reinforced.

### Documents

```bash
aimd document.epub
aimd report.docx
aimd slides.pptx
aimd paper.pdf
aimd notes.md
```

Pandoc converts every format it reads (docx, pptx, xlsx, odt, epub, html,
rtf, rst, LaTeX, and more). `.txt` and `.md` pass through unchanged. Legacy
binary `.doc`, `.ppt`, and `.xls` files are not supported because Pandoc has
no reader for them. MOBI/AZW3 are not supported either.

EPUB, DOCX, and ODT conversions write a directory next to the source, and
`--output` is ignored for them:

```text
document_name/
├── document_name.md
├── chapters/
└── images/
```

### OCR for scanned PDFs and images

```bash
aimd page.png
aimd scan.pdf                      # OCR when the PDF has no usable text layer
aimd scan.pdf --task ocr           # force OCR
aimd scan.pdf --precision 8bit     # Unlimited-OCR 8-bit checkpoint
aimd scan.pdf --start 0 --end 2    # 0-based inclusive page range
```

Images always go to OCR. For PDFs, pdf-inspector checks the text layer. PDFs
with clean text are converted as documents. Scanned PDFs, and PDFs whose
fonts cannot be decoded, go to OCR. aimd renders those pages with `mutool`
and runs OCR one page at a time. Unlimited-OCR uses single-image gundam mode
(`document parsing.` prompt), a sliding-window no-repeat n-gram guard
(n=35, window 128), and at most 8192 tokens per page.

## Models and precision

The model (`--model`) and the precision (`--precision`) are separate options.

| Task | `--model` | Precisions | Default checkpoint |
|------|-----------|------------|--------------------|
| Transcription | `qwen3-asr-1.7b` (default), `qwen3-asr-0.6b` | `4bit` (default), `6bit`, `8bit` | `mlx-community/Qwen3-ASR-1.7B-4bit` |
| OCR | `unlimited-ocr` (default) | `4bit` (default), `6bit`, `8bit` | `mlx-community/Unlimited-OCR-4bit` |

```bash
aimd audio.wav --model qwen3-asr-1.7b --precision 4bit
aimd scan.pdf --model unlimited-ocr --precision 8bit
```

Explicit `mlx-community/Qwen3-ASR-*` and `mlx-community/Unlimited-OCR-*`
checkpoint IDs are also accepted. If a checkpoint ID already contains a
precision, a different `--precision` is an error. Other models are rejected.
bf16 (dense) checkpoints are not supported by the native engine yet.

Local inference needs the MLX GPU (Metal) device. With `AIMD_MLX_DEVICE=cpu`,
or on a machine without a Metal GPU, local ASR/OCR fails early. Use a remote
backend in that case.

## Remote ASR and OCR

Remote inference is opt-in. It sends audio or page images to an
OpenAI-compatible server (`/v1/audio/transcriptions` for ASR, chat
completions for OCR), and no local model is loaded for that task. If you give
a bare server URL, aimd appends `/v1`. The server decides the weight
precision, so `--precision` is ignored with a warning.

```bash
AIMD_ASR_BASE_URL=http://192.168.100.114:8000 \
AIMD_ASR_MODEL=Qwen3-ASR-1.7B \
  aimd audio.wav

AIMD_OCR_BASE_URL=http://192.168.100.114:10000 \
AIMD_OCR_MODEL=Unlimited-OCR \
  aimd scan.png
```

| Variable | CLI option | Default |
|----------|------------|---------|
| `AIMD_ASR_BASE_URL` | `--asr-base-url` | unset (local ASR) |
| `AIMD_ASR_MODEL` | `--asr-model` | `Qwen3-ASR-1.7B` |
| `AIMD_ASR_API_KEY` | `--asr-api-key` | `not-needed` |
| `AIMD_OCR_BASE_URL` | `--ocr-base-url` | unset (local OCR) |
| `AIMD_OCR_MODEL` | `--ocr-model` | `Unlimited-OCR` |
| `AIMD_OCR_API_KEY` | `--ocr-api-key` | `not-needed` |

Explicit options override the environment. MCP tool parameters use the same
names in snake_case (`asr_base_url`, ...).

## MCP server

```bash
aimd-mcp
```

A stdio MCP server with two tools:

- `healthz`
- `process_input`: the same flow and fields as the CLI
  (`input_source`, `task_type`, `model`, `precision`, `language`, `context`,
  `metadata_context`, `start`, `end`, `output_file`, `save_original`,
  `cookies`, `cookies_from_browser`, `raw_transcript`, and the remote backend
  fields).

Example client configuration:

```json
{
  "mcpServers": {
    "aimd": { "command": "aimd-mcp" }
  }
}
```

Set `AIMD_TEMP_DIR` in the server environment to control where temporary
files go.

## Configuration

| Variable | Purpose |
|----------|---------|
| `AIMD_TEMP_DIR` | Temporary downloads, transcoding, and extraction (CLI: `--temp-dir`). |
| `AIMD_ASR_*`, `AIMD_OCR_*` | Remote backends (see above). |
| `AIMD_MLX_DEVICE` | `gpu` or `cpu`. Local inference needs the GPU; `cpu` is for tests. |
| `AIMD_YT_DLP` | Path to the `yt-dlp` executable (default: `yt-dlp` on `PATH`). |
| `HF_HUB_CACHE`, `HF_HOME` | Hugging Face cache location for model weights. |
| `HF_TOKEN` | Hugging Face token for checkpoint downloads. |
| `HF_ENDPOINT` | Alternative Hugging Face endpoint. |
| `HF_HUB_OFFLINE` | Use only cached checkpoints and never download. |
| `YT_DLP_CONFIG_HOME`, `YT_DLP_CACHE_DIR` | Read by yt-dlp itself. |

## Building from source

This needs Rust (stable, edition 2024) and Homebrew `mlx`. To use the
binaries, you also need the runtime tools.

```bash
brew install mlx ffmpeg mupdf pandoc yt-dlp
git clone https://github.com/shuuul/aimd.git
cd aimd
cargo build --release -p aimd-cli -p aimd-mcp
./target/release/aimd --help
```

`crates/aimd-mlx/build.rs` finds MLX through `brew --prefix mlx`. Set
`MLX_PREFIX` to use a different MLX install. See [AGENTS.md](AGENTS.md) for
the crate layout, test tiers, and release process, and
[docs/architecture.md](docs/architecture.md) for the processing flow.

Release notes are on the GitHub Releases page.

## License

Apache-2.0. See [LICENSE](https://github.com/shuuul/aimd/blob/main/LICENSE).
