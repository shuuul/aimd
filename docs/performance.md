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
| PDF OCR | PDF rasterization plus one or more OCR passes | Model reuse helps; page rendering still scales with page count and DPI | PDF pages render with PyMuPDF before OCR model inference. |

## Model/runtime tradeoffs

| Task | Backend | Model family | Expected performance profile |
|------|--------|--------------|------------------------------|
| Transcription | `mlx` | Quantized Qwen3-ASR | Best for Apple Silicon local transcription; lower-bit and 0.6B models trade quality/capability for lower memory and faster startup. |
| Transcription | `mlx` | Whisper, Distil-Whisper, Parakeet, Nemotron, Voxtral, VibeVoice, Qwen2-Audio | Performance is delegated to `mlx-audio`; model size and upstream generation behavior dominate. |
| Transcription | `transformers` | Qwen3-ASR | Default on CUDA-capable non-Darwin platforms and explicit opt-in on macOS/MPS when a `Qwen/Qwen3-ASR-*-hf` (or legacy `Qwen/Qwen3-ASR-*`) model ID is provided. Uses native Transformers Qwen3-ASR (`transformers>=5.14.1`); no vendored model code, `qwen-asr`, vLLM, or SGLang runtime. The 0.6B-hf model is the lower-memory option; 1.7B-hf is the default quality-oriented CUDA option. |
| OCR | `mlx-vlm` | GLM-OCR or explicit mlx-vlm compatible VLMs | macOS VLM OCR path. Heavier than classic detector/recognizer OCR but avoids a separate OCR wrapper layer. |
| OCR | `transformers` | GOT-OCR | Default Linux/CUDA OCR path; uses a generic Transformers image-text generation flow. |
| OCR | `transformers` | Unlimited-OCR | Linux/CUDA VLM OCR path using Baidu's custom `infer`/`infer_multi` API and trusted remote code. Results are read from the model's saved output files. |
| OCR | `transformers` | GLM-OCR or explicit Hugging Face image-text models | Linux/CUDA VLM OCR path; availability depends on upstream model-code requirements and compatible Transformers/runtime versions. |

## Apple Silicon ASR comparison

MPS can run Qwen3-ASR through native Transformers (`transformers>=5.14.1`), but macOS still defaults to MLX. In our smoke/latency checks, the MPS path was viable and useful as an explicit `Qwen/Qwen3-ASR-*-hf` opt-in, but the quantized MLX models remained the better default for Apple Silicon because they are simpler to maintain, smaller, and at least as fast in warm-cache runs.

Input shapes differed slightly between runs, so treat these as practical observations rather than a strict leaderboard.

| Date | Hardware/runtime | Backend/model | Input | Warm latency | Memory observation | Result / decision |
|------|------------------|---------------|-------|--------------|--------------------|-------------------|
| 2026-06-23 | Apple Silicon, macOS, MLX | `mlx-community/Qwen3-ASR-0.6B-4bit` | 7.24s generated Chinese WAV | ~0.18s | MLX peak ~1.46 GB | Fastest/lower-memory observed Qwen3-ASR path; suitable Apple Silicon opt-in for lower-memory MLX use. |
| 2026-06-23 | Apple Silicon, macOS, MLX | `mlx-community/Qwen3-ASR-1.7B-4bit` | 7.24s generated Chinese WAV | ~0.39s | MLX peak ~2.5 GB | Balanced default-quality Apple Silicon path; macOS default remains MLX. |
| 2026-06-23 | Apple Silicon, macOS, MLX | `mlx-community/Qwen3-ASR-1.7B-8bit` | 7.24s generated Chinese WAV | ~0.56s | MLX peak ~3.32 GB | Higher memory and slower than 4-bit in this smoke. |
| 2026-06-23 | Apple Silicon, macOS, Transformers 5.12.1 / MPS | `Qwen/Qwen3-ASR-0.6B` (legacy ID; now maps to `...-0.6B-hf`) | generated Chinese WAV | ~0.25s after local SDPA + KV-cache adaptation on a short smoke; earlier 7.24s run was ~0.45s | MPS allocated ~1.5 GB in earlier run | Historical smoke on vendored backend. Current path uses native Transformers `Qwen/Qwen3-ASR-*-hf` and still does not replace MLX as the macOS default. |

Operational policy from these measurements:

- macOS Apple Silicon automatic transcription backend remains `mlx`.
- Passing `--model Qwen/Qwen3-ASR-0.6B-hf` or `--model Qwen/Qwen3-ASR-1.7B-hf` (legacy non-`-hf` IDs still work) explicitly opts into the native Transformers backend, including on MPS.
- Dependency upgrades that affect Transformers should run the opt-in Qwen3-ASR integration smoke below before release.

## Current smoke-test record

The following are functional smoke tests, not benchmarks. They verify that heavyweight model paths can load and execute in development environments.

| Date | Hardware | Command/path | Input | Observed result |
|------|----------|--------------|-------|-----------------|
| 2026-06-22 | NVIDIA GeForce RTX 5090, Linux/WSL2 | `process_ocr(..., model="unlimited_ocr")` | Generated 512×160 PNG containing `Hello OCR 123` | Returned `HelloOCR 123` after first loading `baidu/Unlimited-OCR`. |
| 2026-06-23 | Apple Silicon, macOS, Transformers 5.12.1 / MPS | `transcribe_file(..., model="Qwen/Qwen3-ASR-0.6B", language="zh")` | Generated short Chinese WAV | Historical smoke: non-empty Chinese text via vendored Transformers backend. Prefer `Qwen/Qwen3-ASR-0.6B-hf` with `transformers>=5.14.1` now. |

## Measurement guidance

When comparing models, measure warm and cold runs separately:

1. **Cold run**: clear model cache/process state and include model download/load time.
2. **Warm run**: run multiple inputs in one API/MCP process or one Python process so model objects can be reused.
3. **Separate stages** when possible: download/extraction, PDF rasterization, audio conversion, and model inference.
4. **Record input shape**: audio duration, page count, image dimensions, PDF DPI, language, and model ID.
5. **Record environment**: OS, Python version, CPU/GPU, CUDA/Metal runtime, dependency versions, and whether weights were already cached.

A minimal local timing harness can wrap the public CLI:

```bash
/usr/bin/time -p aimd input.mp3 --model Qwen/Qwen3-ASR-0.6B-hf --output /tmp/out.md
/usr/bin/time -p aimd scan.pdf --model unlimited_ocr --output /tmp/ocr.md
```

For API/MCP model-reuse measurements, start the server once and send repeated requests against the same process.

## Dependency-upgrade smoke tests

Qwen3-ASR uses native Transformers support (`transformers>=5.14.1`, `Qwen/Qwen3-ASR-*-hf`). Before upgrading Transformers, run the normal tests and the opt-in real inference smoke:

```bash
uv run pytest -q tests/test_transformers_asr.py tests/test_capabilities.py
AIMD_RUN_QWEN3_ASR_INTEGRATION=1 uv run pytest -q tests/test_transformers_asr.py -k real_inference_smoke
```

Set `AIMD_QWEN3_ASR_TEST_AUDIO=/path/to/audio.wav` to use a fixed local audio fixture; otherwise the macOS smoke creates a short synthetic speech file with `say` and `ffmpeg`.
