//! Qwen3-ASR-1.7B on the MLX backend (WS-06 / WS-11).
//!
//! The engine is MLX-only (2026-10-25 decision): there is no
//! hand-written CPU compute path. The graph runs on Metal (macOS, or the MLX
//! CPU device as a fallback) or the `mlx` CUDA backend (Linux) through the
//! [`aimd_mlx`] shim, op-for-op against mlx-audio.
//!
//! # Module layout
//!
//! - [`config`]: typed checkpoint config (`thinker_config.audio_config` +
//!   `text_config` + audio token ids), parsed from the checkpoint
//!   `config.json`. Shared by all submodules; do not duplicate.
//! - [`layout`]: mel frames -> `<|audio_pad|>` count, needed by the prompt
//!   builder before the tower runs.
//! - [`generate`]: backend-agnostic generation pieces: settings/outcome
//!   types, EOS resolution, repetition-loop detection, and the
//!   `parse_qwen_output` port.
//! - [`pipeline`]: the public entry point. [`pipeline::Qwen3AsrPipeline`]
//!   loads a checkpoint (mel -> 24-layer windowed audio tower -> Qwen3 dense
//!   decoder over 4-bit/8-bit affine weights -> greedy loop) and
//!   [`pipeline::transcribe_with_retry`] adds the 4-bit -> 8-bit retry when
//!   the repetition-loop detector trips.
//!
//! # Reference implementations
//!
//! - MLX graph: `.venv/lib/python3.12/site-packages/mlx_audio/stt/models/
//!   qwen3_asr/qwen3_asr.py` (+ `config.py`) — the porting source of truth.
//! - Python behavior contracts: `src/aimd/plugins/asr/models/transformers.py`
//!   (template/retry/parse) and the mel parameter table in
//!   `crate::audio`.
//! - Golden fixtures under `native/golden/asr_encoder` and
//!   `native/golden/asr_decoder` are exported from mlx-audio and gate the
//!   tower output and the forced argmax stream.

pub mod config;
pub mod generate;
pub mod layout;
pub mod pipeline;
