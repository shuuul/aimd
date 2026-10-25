//! Tensor core for the aimd native engine.
//!
//! This crate owns dtype/shape handling, CPU-side storage, and the
//! hand-written CPU operators shared by both models. GPU backends (Metal,
//! CUDA) consume the same layouts; nothing here may depend on a device API.
//!
//! # Contract (WS-03)
//!
//! - [`DType`] covers `F16`, `BF16`, `F32`, and `I32`; `F16`/`BF16` use
//!   [`half`] on the wire. Decode is little-endian, row-major, contiguous
//!   unless a stride is explicitly documented on the API.
//! - [`Tensor`] is a view over an owned, immutable byte buffer
//!   (`Arc<Vec<u8>>`); mutation happens by producing new tensors. Typed
//!   constructors (`Tensor::from_f16_vec` and friends) pack element slices,
//!   and the typed accessors (`Tensor::as_f16`, ...) borrow the buffer
//!   directly as element slices via `half` — no copies, no in-place ops.
//!   The generation loop owns a small set of reusable scratch buffers
//!   instead of general in-place ops.
//! - Ops are free functions in [`ops`] taking/returning [`Tensor`] (`f16`
//!   with `f32` accumulate; one `f16` rounding on store), validated against
//!   naive scalar `f64` references in unit tests:
//!   - [`ops::rmsnorm`] over the last dim with a `[d]` weight and `eps`.
//!   - [`ops::softmax_lastdim`] with an optional row-major `bool` keep-mask
//!     slice (`true` = keep); [`ops::causal_mask`] /
//!     [`ops::sliding_window_mask`] build the causal/window masks that
//!     WS-06/07 wire in.
//!   - [`ops::apply_rope`] on `q`/`k` `[seq, heads, head_dim]` with `theta:
//!     f64` and absolute `positions: &[u32]` (NeoX/rotate_half convention,
//!     Qwen3-compatible; GQA and sliding-window position offsets need no
//!     special handling).
//!   - [`ops::swiglu`] (`silu(gate) * up`).
//!   - [`ops::matmul_f16_f32acc`] `[m,k] x [k,n]`, cache-blocked, rayon
//!     over output row panels, deterministic accumulation order.
//!   - [`ops::gqa_decode_attention`]: single new query token against cache
//!     slices with contiguous `q_heads -> kv_heads` mapping and optional
//!     last-`window` rows visibility.
//!   - [`ops::causal_prefill_attention`]: no cache, `q_len <= k_len`
//!     (absolute query position `k_len - q_len + i`), same GQA mapping.
//! - [`kv`] holds per-layer preallocated contiguous `f16` KV caches
//!   ([`KvCache`]/[`KvLayerCache`]/[`KvWindow`], `max_tokens` sized) with
//!   `append` and sliding-window views; views feed
//!   [`ops::gqa_decode_attention`] directly.
//! - Quantized matmul lives in `aimd-weights`; this crate provides the
//!   dequantized GEMM ([`ops::matmul_f16_f32acc`]) it feeds.

pub mod dtype;
pub mod error;
pub mod kv;
pub mod ops;
pub mod shape;
pub mod tensor;

#[cfg(test)]
pub(crate) mod testutil;

pub use dtype::DType;
pub use error::TensorError;
pub use half::{bf16, f16};
pub use kv::{KvCache, KvLayerCache, KvWindow};
pub use shape::Shape;
pub use tensor::Tensor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_smoke() {
        let shape = Shape::from_dims(&[2, 3]);
        assert_eq!(shape.elem_count(), 6);
    }
}
