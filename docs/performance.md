# Performance

This document records practical performance expectations for `aimd` processing paths. It is not a leaderboard: real throughput depends on input length, page resolution, model size, first-run model download time, GPU memory, and whether the path can reuse cached model weights in the current Python process.

## High-level expectations

| Path | Typical bottleneck | Warm-cache behavior | Notes |
|------|--------------------|---------------------|-------|
| URL with subtitles | Network and yt-dlp metadata/subtitle download | Usually fast after metadata is available | `aimd` prefers subtitles over ASR whenever usable subtitles are found. |
| URL without subtitles | Media download, audio extraction, then ASR | Model reuse helps only inside the same long-running process | CLI invocations pay process startup costs each time; API/MCP can reuse loaded models. |
| Local audio/video transcription | Audio decode/transcode in `aimd.plugins.asr` and ASR model inference | Better in API/MCP because model objects are cached in-process | Non-WAV inputs may be converted to WAV before transcription. |
| MarkItDown document conversion | Parser/Pandoc/document structure | Mostly CPU and file I/O bound | EPUB conversion also writes chapter/image output trees. |
| Image OCR | OCR model inference | API/MCP can reuse loaded model objects | VLM OCR models are heavier than classic detector/recognizer OCR. |
| PDF OCR | PDF rasterization plus one or more OCR passes | Model reuse helps; page rendering still scales with page count and DPI | Linux PDF OCR currently uses `pdftoppm`; macOS mlx4ocr delegates PDF handling to the `mlx4ocr` CLI. |

## Model/runtime tradeoffs

| Task | Engine | Model family | Expected performance profile |
|------|--------|--------------|------------------------------|
| Transcription | `mlx` | Quantized Qwen3-ASR | Best for Apple Silicon local transcription; lower-bit and 0.6B models trade quality/capability for lower memory and faster startup. |
| Transcription | `mlx` | Whisper, Distil-Whisper, Parakeet, Nemotron, Voxtral, VibeVoice, Qwen2-Audio | Performance is delegated to `mlx-audio`; model size and upstream generation behavior dominate. |
| Transcription | `qwen` | Qwen3-ASR | Requires Linux/CUDA. The 0.6B model is the lower-memory option; 1.7B is the default quality-oriented option. |
| OCR | `mlx4ocr` | PP-OCRv6 | Classic OCR path on macOS; generally lighter than VLM OCR. `tiny`, `small`, and `medium` variants trade speed and accuracy. |
| OCR | `mlx4ocr` | GLM-OCR / PaddleOCR-VL | macOS VLM OCR path; heavier than PP-OCRv6 and may require optional `mlx4ocr` VLM dependencies. |
| OCR | `transformers` | GOT-OCR | Default Linux/CUDA OCR path; uses a generic Transformers image-text generation flow. |
| OCR | `transformers` | Unlimited-OCR | Linux/CUDA VLM OCR path using Baidu's custom `infer`/`infer_multi` API and trusted remote code. Results are read from the model's saved output files. |
| OCR | `transformers` | GLM-OCR / PaddleOCR-VL | Linux/CUDA VLM OCR path; availability depends on upstream model-code requirements and compatible Transformers/runtime versions. |

## Current smoke-test record

The following is a functional smoke test, not a benchmark. It verifies that the CUDA path can load and execute a model in the development environment.

| Date | Hardware | Command/path | Input | Observed result |
|------|----------|--------------|-------|-----------------|
| 2026-06-22 | NVIDIA GeForce RTX 5090, Linux/WSL2 | `process_ocr(..., engine="transformers", model="unlimited_ocr")` | Generated 512×160 PNG containing `Hello OCR 123` | Returned `HelloOCR 123` after first loading `baidu/Unlimited-OCR`. |

## Measurement guidance

When comparing models, measure warm and cold runs separately:

1. **Cold run**: clear model cache/process state and include model download/load time.
2. **Warm run**: run multiple inputs in one API/MCP process or one Python process so model objects can be reused.
3. **Separate stages** when possible: download/extraction, PDF rasterization, audio conversion, and model inference.
4. **Record input shape**: audio duration, page count, image dimensions, PDF DPI, language, and model ID.
5. **Record environment**: OS, Python version, CPU/GPU, CUDA/Metal runtime, dependency versions, and whether weights were already cached.

A minimal local timing harness can wrap the public CLI:

```bash
/usr/bin/time -p aimd input.mp3 --engine qwen --model Qwen/Qwen3-ASR-0.6B --output /tmp/out.md
/usr/bin/time -p aimd scan.pdf --engine transformers --model unlimited_ocr --output /tmp/ocr.md
```

For API/MCP model-reuse measurements, start the server once and send repeated requests against the same process.
