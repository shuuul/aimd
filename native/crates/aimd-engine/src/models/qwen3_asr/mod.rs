//! Qwen3-ASR-1.7B graph (WS-06).
//!
//! # Module layout
//!
//! - [`config`]: typed checkpoint config (`thinker_config.audio_config` +
//!   `text_config` + audio token ids), parsed from the checkpoint
//!   `config.json`. Shared by all submodules; do not duplicate.
//! - `encoder` (WS-06a): mel -> conv front-end -> 24-layer windowed encoder
//!   -> output projection. Weights are UNQUANTIZED bf16 under `audio_tower.*`
//!   in the mlx-community checkpoint (Whisper-style blocks: biased
//!   LayerNorm, biased q/k/v/out projections, GELU FFN, windowed attention
//!   `n_window=50` / `n_window_infer=800`).
//! - `decoder` + `generate` (WS-06b): Qwen3 dense decoder (28 layers, GQA
//!   16/8, head_dim 128, SwiGLU 6144, RoPE theta 1e6, tied embeddings, vocab
//!   151936) over 4-bit affine weights (`model.*`), greedy generation with
//!   repetition-loop detection and the 8-bit retry hook, plus the
//!   `parse_qwen_output` port.
//!
//! # Reference implementations
//!
//! - MLX graph: `.venv/lib/python3.12/site-packages/mlx_audio/stt/models/
//!   qwen3_asr/qwen3_asr.py` (+ `config.py`) — the porting source of truth.
//! - Python behavior contracts: `src/aimd/plugins/asr/models/transformers.py`
//!   (template/retry/parse) and the mel parameter table in
//!   `crate::audio`.
//!
//! Submodules are developed in parallel: `encoder` must not import
//! `decoder`/`generate` and vice versa; pipeline integration (mel ->
//! encoder -> decoder -> generate) is coordinator-owned in a later file.

pub mod config;
pub mod decoder;
pub mod encoder;
pub mod generate;
#[cfg(feature = "mlx")]
pub mod mlx_backend;
pub mod pipeline;
