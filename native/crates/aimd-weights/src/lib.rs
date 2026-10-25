//! Weight loading and MLX affine quantization (WS-02).
//!
//! # Contract
//!
//! - [`load_checkpoint`] maps a local HF-style snapshot directory (containing
//!   `*.safetensors` + `config.json`), parses the safetensors headers, and
//!   returns [`WeightStore`] with zero-copy views into mapped bytes.
//!   Multi-shard checkpoints are merged; `config.json`'s `quantization`
//!   block (including per-tensor overrides) is parsed into
//!   [`QuantizationConfig`].
//! - MLX affine quantization: `bits=4|8`, `group_size=64`. A quantized
//!   tensor is stored as the triple `<m>.weight` / `<m>.scales` /
//!   `<m>.biases` for each quantized module `<m>`. The exact packing
//!   (LSB-first nibble/byte order within little-endian `uint32` payload
//!   words, row-major `(rows, cols/group_size)` f16-or-bf16 scales) is
//!   documented in [`mlx_quant`] and locked bit-exactly by golden fixtures
//!   exported via `scripts/export_quant_reference.py`.
//! - [`dequantize_affine`] must be bit-exact against those fixtures (same
//!   f16/bf16 values, same layout). Bit-exactness is the gate: any tolerance
//!   means the packing is wrong, not "close enough".
//! - 8-bit affine (ASR repetition retry) shares the same path with `bits=8`.
//! - Errors: I/O and parse failures are typed ([`WeightsError`]); no panics
//!   on malformed input.

pub mod mlx_quant;
pub mod store;

pub use mlx_quant::{DequantizedTensor, QuantizedTensor, ScaleDtype, dequantize_affine};
pub use store::{PerTensorQuant, QuantizationConfig, WeightStore, load_checkpoint};

/// Loading/parse failures for checkpoint directories and safetensors files.
#[derive(Debug, thiserror::Error)]
pub enum WeightsError {
    #[error("checkpoint directory not found or empty: {0}")]
    MissingCheckpoint(String),
    #[error("safetensors parse error: {0}")]
    Safetensors(String),
    #[error("unsupported quantization (expected affine 4/8-bit, group 64): {0}")]
    UnsupportedQuant(String),
    #[error("I/O error: {0}")]
    Io(String),
    #[error("config.json parse error: {0}")]
    Json(String),
    #[error("tensor not found: {0}")]
    MissingTensor(String),
    #[error("invalid tensor (shape/layout/offset): {0}")]
    InvalidTensor(String),
}
