# packages/aimd/src/aimd/infrastructure

Infrastructure layer in the main `aimd` package.

## Responsibilities

- MarkItDown runner for local file conversion (`markitdown_processor.py`).
- Media adapter that wraps `aimd-media` URL extraction results as `TextContext`.
- Markdown chunking/title helpers for MarkItDown output.

## Model Notes

- `aimd_media.const.MLX_AUDIO_MODELS` is a curated allow-list for mlx-audio STT IDs. It includes Qwen3-ASR quantized variants plus newer mlx-audio 0.4.4 STT models such as Whisper, Distil-Whisper, Parakeet, Nemotron ASR, Voxtral, VibeVoice-ASR, and Qwen2-Audio.
- Do not add mlx-audio forced aligner models to transcription until the product/API accepts reference text and timestamp output.
- mlx Qwen3-ASR gets a default `Chinese` language hint for existing behavior. Other mlx-audio STT models should receive no language hint unless the caller provided one and the model `generate()` signature accepts `language`.
- `aimd_media.const.QWEN_ASR_MODELS` tracks official Qwen3-ASR models only: `Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3-ASR-0.6B`.
- Qwen3-ASR upstream supports more languages/dialects than the local `LANGUAGE_CODE_TO_NAME` table exposes; expanding the table is a valid small follow-up if the adapters need those codes.

## Rules

- Do not import adapters.
- Keep modules focused and small by pipeline concern.
- Raise domain errors from `aimd.errors`.
- Keep capability checks fail-fast before expensive model work.
- When using `tempfile.TemporaryDirectory` or `tempfile.NamedTemporaryFile`, always pass the `dir=temp_dir` parameter so callers can redirect temp I/O to a sandbox-safe location via `AIMD_TEMP_DIR`.
- Do not add compatibility aliases for old internal paths; local file processors should use MarkItDown contracts.
