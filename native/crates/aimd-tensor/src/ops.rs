//! Hand-written CPU operators (WS-03).
//!
//! Conventions used across every op:
//!
//! - Inputs/outputs are [`Tensor`]s in `F16` unless documented otherwise;
//!   all math accumulates in `f32` (the unit-test references use `f64`), and
//!   outputs round to `f16` exactly once, on store.
//! - Tensors are contiguous row-major; ops validate dtype and shape and
//!   return [`TensorError`] on mismatches.
//! - This phase is deliberately naive and portable: plain scalar loops with
//!   cache-friendly panel blocking in [`matmul_f16_f32acc`] (plus rayon
//!   parallelism over independent output row blocks, which does not change
//!   any accumulation order). aarch64 NEON specialization is a later perf
//!   phase and must not change results beyond f16 rounding.
//!
//! Each op is validated against a naive scalar `f64` reference in the
//! `tests` module below.

use crate::dtype::DType;
use crate::error::TensorError;
use crate::shape::Shape;
use crate::tensor::Tensor;
use half::f16;
use rayon::prelude::*;

/// Row-block height for the `matmul_f16_f32acc` panel blocking.
const MC: usize = 32;
/// Inner-dimension block depth for the `matmul_f16_f32acc` panel blocking.
const KC: usize = 256;
/// Column-block width for the `matmul_f16_f32acc` panel blocking.
const NC: usize = 128;
/// At or below this many rows, matmul parallelism switches from row panels
/// to per-row column blocks (decode steps run `m == 1`).
const SMALL_M: usize = 8;

/// Root-mean-square layer norm over the last dimension.
///
/// `x` is `F16` with shape `[..., d]` (any rank >= 1) and `weight` is `F16`
/// with shape `[d]`. Each output row is `weight * x / sqrt(mean(x^2) + eps)`
/// with the mean-of-squares and the multiplies accumulated in `f32`. This is
/// the MLX/HF `rms_norm` definition used by Qwen3 and the OCR decoder (no
/// mean subtraction, no bias).
pub fn rmsnorm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor, TensorError> {
    if x.dtype() != DType::F16 {
        return Err(TensorError::DTypeMismatch {
            expected: DType::F16,
            actual: x.dtype(),
        });
    }
    if weight.dtype() != DType::F16 {
        return Err(TensorError::DTypeMismatch {
            expected: DType::F16,
            actual: weight.dtype(),
        });
    }
    let dims = x.shape().dims();
    let Some(&d) = dims.last() else {
        return Err(TensorError::InvalidShape {
            shape: dims.to_vec(),
            reason: "rmsnorm needs rank >= 1".to_string(),
        });
    };
    if weight.shape().rank() != 1 || weight.shape().dims()[0] != d {
        return Err(TensorError::ShapeMismatch {
            expected: format!("[{d}]"),
            actual: format!("{:?}", weight.shape().dims()),
        });
    }

    let xs = x.as_f16()?;
    let w = weight.as_f16()?;
    let rows: usize = dims[..dims.len() - 1].iter().product();
    if d == 0 {
        return Tensor::from_f16_vec(x.shape().clone(), &[]);
    }
    let mut out = vec![f16::ZERO; rows * d];
    for (r, orow) in out.chunks_exact_mut(d).enumerate() {
        let row = &xs[r * d..(r + 1) * d];
        let sum_sq: f32 = row
            .iter()
            .map(|&v| {
                let v = v.to_f32();
                v * v
            })
            .sum();
        let inv = 1.0 / (sum_sq / d as f32 + eps).sqrt();
        for ((o, &xv), &wv) in orow.iter_mut().zip(row).zip(w) {
            *o = f16::from_f32(xv.to_f32() * inv * wv.to_f32());
        }
    }
    Tensor::from_f16_vec(x.shape().clone(), &out)
}

/// Softmax over the last dimension, computed in `f32`.
///
/// `x` is `F16` with shape `[..., k]`. `mask`, when present, is a row-major
/// boolean slice of shape `[rows, k]` where `rows == x.elem_count() / k`:
/// `mask[r * k + j] == true` **keeps** key `j` for row `r`, `false` masks it
/// out (its probability becomes exactly zero). Use [`causal_mask`] /
/// [`sliding_window_mask`] to build the masks the ASR encoder and OCR
/// decoder need once WS-06/07 wire them in.
///
/// Numerically stable (max-subtracted). A row with no kept positions yields
/// all-zero probabilities instead of NaN.
pub fn softmax_lastdim(x: &Tensor, mask: Option<&[bool]>) -> Result<Tensor, TensorError> {
    if x.dtype() != DType::F16 {
        return Err(TensorError::DTypeMismatch {
            expected: DType::F16,
            actual: x.dtype(),
        });
    }
    let dims = x.shape().dims();
    let Some(&k) = dims.last() else {
        return Err(TensorError::InvalidShape {
            shape: dims.to_vec(),
            reason: "softmax_lastdim needs rank >= 1".to_string(),
        });
    };
    let rows: usize = dims[..dims.len() - 1].iter().product();
    if let Some(m) = mask.filter(|m| m.len() != rows * k) {
        return Err(TensorError::ShapeMismatch {
            expected: format!("mask of [{rows}, {k}]"),
            actual: format!("len {}", m.len()),
        });
    }

    let xs = x.as_f16()?;
    if k == 0 {
        return Tensor::from_f16_vec(x.shape().clone(), &[]);
    }
    let mut out = vec![f16::ZERO; rows * k];
    let mut exps = vec![0f32; k];
    for (r, orow) in out.chunks_exact_mut(k).enumerate() {
        let row = &xs[r * k..(r + 1) * k];
        let allowed = |j: usize| mask.is_none_or(|m| m[r * k + j]);
        let mut max = f32::NEG_INFINITY;
        for (j, &xv) in row.iter().enumerate() {
            if allowed(j) {
                max = max.max(xv.to_f32());
            }
        }
        if max.is_infinite() {
            // No kept position in this row: documented all-zero output.
            continue;
        }
        let mut sum = 0f32;
        for (j, &xv) in row.iter().enumerate() {
            if allowed(j) {
                let e = (xv.to_f32() - max).exp();
                exps[j] = e;
                sum += e;
            } else {
                exps[j] = 0.0;
            }
        }
        let inv = 1.0 / sum;
        for (o, &e) in orow.iter_mut().zip(exps.iter()) {
            *o = f16::from_f32(e * inv);
        }
    }
    Tensor::from_f16_vec(x.shape().clone(), &out)
}

/// Causal boolean mask for [`softmax_lastdim`]: row-major `[q_len, k_len]`,
/// entry `(i, j)` is `true` (keep) iff `j <= i`.
pub fn causal_mask(q_len: usize, k_len: usize) -> Vec<bool> {
    (0..q_len)
        .flat_map(|i| (0..k_len).map(move |j| j <= i))
        .collect()
}

/// Sliding-window + causal boolean mask for [`softmax_lastdim`]: row-major
/// `[q_len, k_len]`, entry `(i, j)` is `true` (keep) iff `j <= i` and
/// `i - j < window` (the `window` most recent keys including the current
/// position). `window == 0` keeps nothing, which [`softmax_lastdim`] maps to
/// all-zero rows.
pub fn sliding_window_mask(q_len: usize, k_len: usize, window: usize) -> Vec<bool> {
    (0..q_len)
        .flat_map(|i| (0..k_len).map(move |j| j <= i && i - j < window))
        .collect()
}

/// Standard attention softmax scale `1 / sqrt(head_dim)`.
pub fn attention_scale(head_dim: usize) -> f32 {
    (head_dim as f32).sqrt().recip()
}

/// Rotary position embedding (RoPE) applied to query and key tensors.
///
/// Shapes: `q` is `F16 [seq, q_heads, head_dim]`, `k` is
/// `F16 [seq, kv_heads, head_dim]` (head counts may differ — GQA needs no
/// special handling because the rotation is per head), `head_dim` must be
/// even, and `positions.len() == seq`. Returns freshly rotated tensors with
/// the same shapes.
///
/// Convention (matches the HF/Qwen3 `rotary_emb`, aka the "rotate_half" /
/// NeoX style): with `h = head_dim / 2`, frequency `i` has
/// `inv_freq[i] = theta^(-2i / head_dim)` and the pair `(x[i], x[i + h])` is
/// rotated by `angle = position * inv_freq[i]`:
///
/// ```text
/// out[i]     = x[i]     * cos(angle) - x[i + h] * sin(angle)
/// out[i + h] = x[i + h] * cos(angle) + x[i]     * sin(angle)
/// ```
///
/// `positions` holds absolute token positions, so sliding-window decode
/// offsets for the OCR decoder just supply the real positions; attention
/// itself stays position-free. `theta` is `f64` (1e6 for Qwen3-ASR) and
/// angles/cos/sin are computed in `f64` before rounding to `f32` math, so
/// long sequences and repeated decode steps stay stable.
pub fn apply_rope(
    q: &Tensor,
    k: &Tensor,
    positions: &[u32],
    theta: f64,
) -> Result<(Tensor, Tensor), TensorError> {
    for t in [q, k] {
        if t.dtype() != DType::F16 {
            return Err(TensorError::DTypeMismatch {
                expected: DType::F16,
                actual: t.dtype(),
            });
        }
    }
    let qd = q.shape().dims();
    let kd = k.shape().dims();
    if qd.len() != 3 {
        return Err(TensorError::InvalidShape {
            shape: qd.to_vec(),
            reason: "q must be [seq, q_heads, head_dim]".to_string(),
        });
    }
    if kd.len() != 3 {
        return Err(TensorError::InvalidShape {
            shape: kd.to_vec(),
            reason: "k must be [seq, kv_heads, head_dim]".to_string(),
        });
    }
    let (q_seq, q_heads, head_dim) = (qd[0], qd[1], qd[2]);
    let (k_seq, k_heads, k_head_dim) = (kd[0], kd[1], kd[2]);
    if q_seq != k_seq {
        return Err(TensorError::ShapeMismatch {
            expected: format!("k seq = {q_seq}"),
            actual: format!("k seq = {k_seq}"),
        });
    }
    if head_dim != k_head_dim {
        return Err(TensorError::ShapeMismatch {
            expected: format!("k head_dim = {head_dim}"),
            actual: format!("k head_dim = {k_head_dim}"),
        });
    }
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(TensorError::InvalidShape {
            shape: vec![q_seq, q_heads, head_dim],
            reason: "head_dim must be nonzero and even".to_string(),
        });
    }
    if positions.len() != q_seq {
        return Err(TensorError::ShapeMismatch {
            expected: format!("positions len = {q_seq}"),
            actual: format!("positions len = {}", positions.len()),
        });
    }

    let half = head_dim / 2;
    let inv_freq: Vec<f64> = (0..half)
        .map(|i| theta.powf(-2.0 * i as f64 / head_dim as f64))
        .collect();
    let qs = q.as_f16()?;
    let ks = k.as_f16()?;
    let mut q_out = vec![f16::ZERO; qs.len()];
    let mut k_out = vec![f16::ZERO; ks.len()];
    rotate_rows(&mut q_out, qs, q_heads, head_dim, &inv_freq, positions);
    rotate_rows(&mut k_out, ks, k_heads, head_dim, &inv_freq, positions);
    Ok((
        Tensor::from_f16_vec(q.shape().clone(), &q_out)?,
        Tensor::from_f16_vec(k.shape().clone(), &k_out)?,
    ))
}

/// Applies the half-split rotation to every `[seq, heads, head_dim]` row,
/// writing into `out`. cos/sin are recomputed per position (f64 -> f32) and
/// shared across heads.
fn rotate_rows(
    out: &mut [f16],
    input: &[f16],
    heads: usize,
    head_dim: usize,
    inv_freq: &[f64],
    positions: &[u32],
) {
    let half = head_dim / 2;
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for (t, &pos) in positions.iter().enumerate() {
        for (i, &inv) in inv_freq.iter().enumerate() {
            let angle = f64::from(pos) * inv;
            cos[i] = angle.cos() as f32;
            sin[i] = angle.sin() as f32;
        }
        let base = t * heads * head_dim;
        for h in 0..heads {
            let off = base + h * head_dim;
            let row = &input[off..off + head_dim];
            let orow = &mut out[off..off + head_dim];
            for (i, &c) in cos.iter().enumerate() {
                let s = sin[i];
                let a = row[i].to_f32();
                let b = row[i + half].to_f32();
                orow[i] = f16::from_f32(a * c - b * s);
                orow[i + half] = f16::from_f32(b * c + a * s);
            }
        }
    }
}

/// SwiGLU feed-forward activation: `silu(gate) * up`, elementwise in `f32`.
///
/// `gate` and `up` are `F16` with identical shapes (any rank); the result is
/// `F16` with that shape. `silu(x) = x * sigmoid(x)` uses the stable
/// two-sided formulation so large negative gates do not overflow `exp`.
pub fn swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor, TensorError> {
    for t in [gate, up] {
        if t.dtype() != DType::F16 {
            return Err(TensorError::DTypeMismatch {
                expected: DType::F16,
                actual: t.dtype(),
            });
        }
    }
    if gate.shape() != up.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: format!("{:?}", gate.shape().dims()),
            actual: format!("{:?}", up.shape().dims()),
        });
    }
    let g = gate.as_f16()?;
    let u = up.as_f16()?;
    let mut out = vec![f16::ZERO; g.len()];
    for ((o, &gv), &uv) in out.iter_mut().zip(g).zip(u) {
        let silu = silu_f32(gv.to_f32());
        *o = f16::from_f32(silu * uv.to_f32());
    }
    Tensor::from_f16_vec(gate.shape().clone(), &out)
}

/// Numerically stable `x * sigmoid(x)` in `f32`.
fn silu_f32(x: f32) -> f32 {
    if x >= 0.0 {
        x / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        x * e / (1.0 + e)
    }
}

/// Naive-but-cache-blocked matrix multiply: `a [m, k] x b [k, n] -> [m, n]`,
/// all `F16` row-major, accumulation in `f32`, one rounding to `F16` on
/// store.
///
/// Panel blocking (`MC x NC` output panels, `KC`-deep k slabs) keeps the
/// f32 accumulator and the `b` panel L1/L2-resident; the inner loop is a
/// plain axpy over contiguous rows. Work is parallelized with rayon over
/// row panels when `m` is large, and over `NC` column blocks within each
/// row when `m` is small (decode steps run `m == 1`, where row-panel
/// parallelism would be serial). The per-element accumulation order is
/// fixed either way, so results are deterministic. NEON microkernels
/// replace the inner loops in the perf phase.
pub fn matmul_f16_f32acc(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    for t in [a, b] {
        if t.dtype() != DType::F16 {
            return Err(TensorError::DTypeMismatch {
                expected: DType::F16,
                actual: t.dtype(),
            });
        }
        if t.shape().rank() != 2 {
            return Err(TensorError::InvalidShape {
                shape: t.shape().dims().to_vec(),
                reason: "matmul needs rank-2 inputs".to_string(),
            });
        }
    }
    let (m, k) = (a.shape().dims()[0], a.shape().dims()[1]);
    let (k2, n) = (b.shape().dims()[0], b.shape().dims()[1]);
    if k != k2 {
        return Err(TensorError::ShapeMismatch {
            expected: format!("b rows = {k}"),
            actual: format!("b rows = {k2}"),
        });
    }
    let out_shape = Shape::from_dims(&[m, n]);
    if m == 0 || n == 0 || k == 0 {
        return Tensor::from_f16_vec(out_shape, &[]);
    }

    let a_s = a.as_f16()?;
    let b_s = b.as_f16()?;
    let mut out = vec![f16::ZERO; m * n];
    if m <= SMALL_M {
        // Decode-step regime (m == 1..SMALL_M): parallelize column blocks
        // within each row instead of row panels. Same KC-slab accumulation
        // order and single f32->f16 rounding as the panel path, so
        // per-element results are identical.
        for i in 0..m {
            let orow = &mut out[i * n..(i + 1) * n];
            orow.par_chunks_mut(NC)
                .enumerate()
                .for_each(|(blk, oblock)| {
                    let jj = blk * NC;
                    let jb = oblock.len();
                    let a_off = i * k;
                    let mut acc = vec![0f32; jb];
                    for kk in (0..k).step_by(KC) {
                        let kb = KC.min(k - kk);
                        let a_row = &a_s[a_off + kk..a_off + kk + kb];
                        for (t, &av) in a_row.iter().enumerate() {
                            let avf = av.to_f32();
                            let b_off = (kk + t) * n + jj;
                            let b_row = &b_s[b_off..b_off + jb];
                            crate::simd::axpy_f16_f32(avf, b_row, &mut acc);
                        }
                    }
                    for (o, &v) in oblock.iter_mut().zip(&acc) {
                        *o = f16::from_f32(v);
                    }
                });
        }
        return Tensor::from_f16_vec(out_shape, &out);
    }
    out.par_chunks_mut(MC * n)
        .enumerate()
        .for_each(|(block, orows)| {
            let ii = block * MC;
            let ib = orows.len() / n;
            for jj in (0..n).step_by(NC) {
                let jb = NC.min(n - jj);
                let mut acc = vec![0f32; ib * jb];
                for kk in (0..k).step_by(KC) {
                    let kb = KC.min(k - kk);
                    for i in 0..ib {
                        let a_off = (ii + i) * k + kk;
                        let a_row = &a_s[a_off..a_off + kb];
                        let acc_row = &mut acc[i * jb..(i + 1) * jb];
                        for (t, &av) in a_row.iter().enumerate() {
                            let avf = av.to_f32();
                            let b_off = (kk + t) * n + jj;
                            let b_row = &b_s[b_off..b_off + jb];
                            crate::simd::axpy_f16_f32(avf, b_row, acc_row);
                        }
                    }
                }
                for i in 0..ib {
                    let dst = &mut orows[i * n + jj..i * n + jj + jb];
                    let acc_row = &acc[i * jb..(i + 1) * jb];
                    for (o, &v) in dst.iter_mut().zip(acc_row) {
                        *o = f16::from_f32(v);
                    }
                }
            }
        });
    Tensor::from_f16_vec(out_shape, &out)
}

/// Transposed-B matrix multiply for the tied-embedding logits path:
/// `a [m, k] x bt^T` where `bt` is `F16 [n, k]` row-major (row `v` of `bt`
/// is the `v`-th output column), returning `F32 [m, n]`.
///
/// This avoids materializing the `[k, n]` transpose of an embedding table
/// (the Qwen3-ASR table would be a ~600 MB copy) and keeps full `f32`
/// output precision for faithful argmax ties. Parallelized over `NC` row
/// blocks of `bt` (the vocab axis) with rayon — the decode-time `m` is 1
/// to a few rows. Deterministic: each output element is a single-task
/// `KC`-blocked accumulation.
pub fn matmul_bt_f16_f32out(a: &Tensor, bt: &Tensor) -> Result<Tensor, TensorError> {
    for t in [a, bt] {
        if t.dtype() != DType::F16 {
            return Err(TensorError::DTypeMismatch {
                expected: DType::F16,
                actual: t.dtype(),
            });
        }
        if t.shape().rank() != 2 {
            return Err(TensorError::InvalidShape {
                shape: t.shape().dims().to_vec(),
                reason: "matmul_bt needs rank-2 inputs".to_string(),
            });
        }
    }
    let (m, k) = (a.shape().dims()[0], a.shape().dims()[1]);
    let (n, k2) = (bt.shape().dims()[0], bt.shape().dims()[1]);
    if k != k2 {
        return Err(TensorError::ShapeMismatch {
            expected: format!("bt cols = {k}"),
            actual: format!("bt cols = {k2}"),
        });
    }
    let out_shape = Shape::from_dims(&[m, n]);
    if m == 0 || n == 0 || k == 0 {
        return Tensor::from_f32_vec(out_shape, &[]);
    }

    let a_s = a.as_f16()?;
    let bt_s = bt.as_f16()?;
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        let orow = &mut out[i * n..(i + 1) * n];
        orow.par_chunks_mut(NC)
            .enumerate()
            .for_each(|(blk, oblock)| {
                let jj = blk * NC;
                let jb = oblock.len();
                let a_off = i * k;
                let mut acc = vec![0f32; jb];
                for kk in (0..k).step_by(KC) {
                    let kb = KC.min(k - kk);
                    let a_row = &a_s[a_off + kk..a_off + kk + kb];
                    for (v, ov) in acc.iter_mut().enumerate() {
                        let bt_row = &bt_s[(jj + v) * k + kk..(jj + v) * k + kk + kb];
                        *ov += crate::simd::dot_f16_f32(a_row, bt_row);
                    }
                }
                oblock.copy_from_slice(&acc);
            });
    }
    Tensor::from_f32_vec(out_shape, &out)
}

/// Single-token GQA decode attention against a contiguous KV cache.
///
/// - `q`: `F16 [q_heads, head_dim]` — the current token's query (after
///   RoPE).
/// - `k_cache` / `v_cache`: `F16` slices of `kv_len` cached rows, each row
///   `[kv_heads, head_dim]` flattened — exactly the layout of
///   [`crate::KvLayerCache::keys_view`] / [`crate::KvLayerCache::values_view`].
/// - GQA head mapping is contiguous: query head `h` reads kv head
///   `h / (q_heads / kv_heads)` (Qwen3-ASR: 16 query heads -> 8 kv heads,
///   groups of 2). `q_heads` must be a multiple of `kv_heads` (equal for
///   MHA).
/// - `scale`: softmax scale; `None` uses [`attention_scale`].
/// - `window`: when `Some(w)`, only the last `min(w, kv_len)` cached rows
///   are visible (sliding-window decode for the OCR decoder); `None`
///   attends to the whole cache.
///
/// Scores, softmax, and the value mix accumulate in `f32`; the output is
/// `F16 [q_heads, head_dim]`. An empty cache yields zeros (defensive; the
/// engine appends the current token's k/v before decoding).
pub fn gqa_decode_attention(
    q: &Tensor,
    k_cache: &[f16],
    v_cache: &[f16],
    kv_heads: usize,
    head_dim: usize,
    scale: Option<f32>,
    window: Option<usize>,
) -> Result<Tensor, TensorError> {
    if q.dtype() != DType::F16 {
        return Err(TensorError::DTypeMismatch {
            expected: DType::F16,
            actual: q.dtype(),
        });
    }
    if q.shape().rank() != 2 {
        return Err(TensorError::InvalidShape {
            shape: q.shape().dims().to_vec(),
            reason: "q must be [q_heads, head_dim]".to_string(),
        });
    }
    let (q_heads, q_head_dim) = (q.shape().dims()[0], q.shape().dims()[1]);
    if kv_heads == 0 {
        return Err(TensorError::InvalidShape {
            shape: vec![kv_heads, head_dim],
            reason: "kv_heads must be nonzero".to_string(),
        });
    }
    if head_dim == 0 {
        return Err(TensorError::InvalidShape {
            shape: vec![kv_heads, head_dim],
            reason: "head_dim must be nonzero".to_string(),
        });
    }
    if q_head_dim != head_dim {
        return Err(TensorError::ShapeMismatch {
            expected: format!("q head_dim = {head_dim}"),
            actual: format!("q head_dim = {q_head_dim}"),
        });
    }
    if q_heads % kv_heads != 0 {
        return Err(TensorError::ShapeMismatch {
            expected: format!("q_heads multiple of {kv_heads}"),
            actual: format!("q_heads = {q_heads}"),
        });
    }
    let row = kv_heads * head_dim;
    if !k_cache.len().is_multiple_of(row) || !v_cache.len().is_multiple_of(row) {
        return Err(TensorError::InvalidShape {
            shape: vec![k_cache.len()],
            reason: "cache length must be a multiple of kv_heads * head_dim".to_string(),
        });
    }
    let kv_len = k_cache.len() / row;
    if v_cache.len() / row != kv_len {
        return Err(TensorError::ShapeMismatch {
            expected: format!("v cache rows = {kv_len}"),
            actual: format!("v cache rows = {}", v_cache.len() / row),
        });
    }

    let group = q_heads / kv_heads;
    let scale = scale.unwrap_or_else(|| attention_scale(head_dim));
    let visible = window.unwrap_or(kv_len).min(kv_len);
    let skip = kv_len - visible;
    let qs = q.as_f16()?;
    let mut out = vec![f16::ZERO; q_heads * head_dim];
    if visible == 0 {
        return Tensor::from_f16_vec(Shape::from_dims(&[q_heads, head_dim]), &out);
    }

    let mut scores = vec![0f32; visible];
    let mut acc = vec![0f32; head_dim];
    for h in 0..q_heads {
        let g = h / group;
        let q_row = &qs[h * head_dim..(h + 1) * head_dim];
        for (idx, s) in scores.iter_mut().enumerate() {
            let off = (skip + idx) * row + g * head_dim;
            let k_row = &k_cache[off..off + head_dim];
            let dot: f32 = q_row
                .iter()
                .zip(k_row)
                .map(|(&qv, &kv)| qv.to_f32() * kv.to_f32())
                .sum();
            *s = dot * scale;
        }
        softmax_in_place(&mut scores);
        acc.fill(0.0);
        for (idx, &p) in scores.iter().enumerate() {
            let off = (skip + idx) * row + g * head_dim;
            let v_row = &v_cache[off..off + head_dim];
            for (a, &vv) in acc.iter_mut().zip(v_row) {
                *a += p * vv.to_f32();
            }
        }
        for (o, &a) in out[h * head_dim..(h + 1) * head_dim].iter_mut().zip(&acc) {
            *o = f16::from_f32(a);
        }
    }
    Tensor::from_f16_vec(Shape::from_dims(&[q_heads, head_dim]), &out)
}

/// Causal self-attention for prefill (no KV cache).
///
/// Shapes: `q` `F16 [q_len, q_heads, head_dim]`, `k`/`v`
/// `F16 [k_len, kv_heads, head_dim]` with `q_len <= k_len`, `v` matching
/// `k`, and `q_heads` a multiple of `kv_heads` (equal for MHA). Query row
/// `i` sits at absolute position `k_len - q_len + i` and attends to key rows
/// `0..=pos`; the plain no-cache prefill is the `q_len == k_len` case, and
/// `q_len < k_len` also covers cache-aware chunked prefill. GQA uses the
/// same contiguous head mapping as [`gqa_decode_attention`].
///
/// `scale: None` uses [`attention_scale`]. All math accumulates in `f32`;
/// the output is `F16 [q_len, q_heads, head_dim]`.
pub fn causal_prefill_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: Option<f32>,
) -> Result<Tensor, TensorError> {
    for t in [q, k, v] {
        if t.dtype() != DType::F16 {
            return Err(TensorError::DTypeMismatch {
                expected: DType::F16,
                actual: t.dtype(),
            });
        }
        if t.shape().rank() != 3 {
            return Err(TensorError::InvalidShape {
                shape: t.shape().dims().to_vec(),
                reason: "prefill attention needs rank-3 inputs".to_string(),
            });
        }
    }
    let (q_len, q_heads, head_dim) = (
        q.shape().dims()[0],
        q.shape().dims()[1],
        q.shape().dims()[2],
    );
    let (k_len, kv_heads, k_head_dim) = (
        k.shape().dims()[0],
        k.shape().dims()[1],
        k.shape().dims()[2],
    );
    if v.shape().dims() != k.shape().dims() {
        return Err(TensorError::ShapeMismatch {
            expected: format!("{:?}", k.shape().dims()),
            actual: format!("{:?}", v.shape().dims()),
        });
    }
    if kv_heads == 0 || head_dim == 0 {
        return Err(TensorError::InvalidShape {
            shape: vec![k_len, kv_heads, head_dim],
            reason: "kv_heads and head_dim must be nonzero".to_string(),
        });
    }
    if k_head_dim != head_dim {
        return Err(TensorError::ShapeMismatch {
            expected: format!("k head_dim = {head_dim}"),
            actual: format!("k head_dim = {k_head_dim}"),
        });
    }
    if q_heads % kv_heads != 0 {
        return Err(TensorError::ShapeMismatch {
            expected: format!("q_heads multiple of {kv_heads}"),
            actual: format!("q_heads = {q_heads}"),
        });
    }
    if q_len > k_len {
        return Err(TensorError::ShapeMismatch {
            expected: format!("q_len <= k_len = {k_len}"),
            actual: format!("q_len = {q_len}"),
        });
    }

    let group = q_heads / kv_heads;
    let scale = scale.unwrap_or_else(|| attention_scale(head_dim));
    let offset = k_len - q_len;
    let qs = q.as_f16()?;
    let ks = k.as_f16()?;
    let vs = v.as_f16()?;
    let mut out = vec![f16::ZERO; q_len * q_heads * head_dim];
    let mut scores = vec![0f32; k_len.max(1)];
    let mut acc = vec![0f32; head_dim];
    for i in 0..q_len {
        let pos = offset + i;
        for h in 0..q_heads {
            let g = h / group;
            let q_off = (i * q_heads + h) * head_dim;
            let q_row = &qs[q_off..q_off + head_dim];
            let mut max = f32::NEG_INFINITY;
            for (j, s_slot) in scores[..=pos].iter_mut().enumerate() {
                let k_off = (j * kv_heads + g) * head_dim;
                let dot: f32 = q_row
                    .iter()
                    .zip(&ks[k_off..k_off + head_dim])
                    .map(|(&qv, &kv)| qv.to_f32() * kv.to_f32())
                    .sum();
                let s = dot * scale;
                *s_slot = s;
                max = max.max(s);
            }
            let mut sum = 0f32;
            for s in &mut scores[..=pos] {
                *s = (*s - max).exp();
                sum += *s;
            }
            let inv = 1.0 / sum;
            acc.fill(0.0);
            for (j, &sp) in scores[..=pos].iter().enumerate() {
                let p = sp * inv;
                let v_off = (j * kv_heads + g) * head_dim;
                let v_row = &vs[v_off..v_off + head_dim];
                for (a, &vv) in acc.iter_mut().zip(v_row) {
                    *a += p * vv.to_f32();
                }
            }
            for (o, &a) in out[q_off..q_off + head_dim].iter_mut().zip(&acc) {
                *o = f16::from_f32(a);
            }
        }
    }
    Tensor::from_f16_vec(Shape::from_dims(&[q_len, q_heads, head_dim]), &out)
}

/// Numerically stable in-place `f32` softmax. Used by attention where masked
/// lanes never enter the buffer, so `values` is non-empty and the sum cannot
/// be zero (the max lane contributes `exp(0) == 1`).
fn softmax_in_place(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in values.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in values.iter_mut() {
        *v *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{Lcg, assert_close};

    const ATOL: f64 = 1e-2;
    const RTOL: f64 = 1e-3;

    // ---------- naive f64 reference implementations ----------

    fn ref_rmsnorm(x: &[f16], w: &[f16], rows: usize, d: usize, eps: f64) -> Vec<f64> {
        let mut out = vec![0.0; rows * d];
        for r in 0..rows {
            let row = &x[r * d..(r + 1) * d];
            let mean_sq: f64 = row.iter().map(|&v| v.to_f64() * v.to_f64()).sum::<f64>() / d as f64;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for j in 0..d {
                out[r * d + j] = row[j].to_f64() * inv * w[j].to_f64();
            }
        }
        out
    }

    /// Masked softmax reference (max-subtracted; masked entries 0; a
    /// fully-masked row yields zeros).
    fn ref_softmax(x: &[f16], rows: usize, k: usize, mask: Option<&[bool]>) -> Vec<f64> {
        let mut out = vec![0.0; rows * k];
        for r in 0..rows {
            let row = &x[r * k..(r + 1) * k];
            let allowed = |j: usize| mask.is_none_or(|m| m[r * k + j]);
            let mut max = f64::NEG_INFINITY;
            for (j, &xv) in row.iter().enumerate() {
                if allowed(j) {
                    max = max.max(xv.to_f64());
                }
            }
            if max.is_infinite() {
                continue;
            }
            let mut exps = vec![0.0; k];
            let mut sum = 0.0;
            for (j, &xv) in row.iter().enumerate() {
                if allowed(j) {
                    exps[j] = (xv.to_f64() - max).exp();
                    sum += exps[j];
                }
            }
            for (o, &e) in out[r * k..(r + 1) * k].iter_mut().zip(exps.iter()) {
                *o = e / sum;
            }
        }
        out
    }

    /// Fixed-offset softmax (subtract the first element, not the row max)
    /// used to prove the implementation is numerically stable for large
    /// magnitudes without relying on the same max-subtraction trick.
    fn ref_softmax_raw(x: &[f16], rows: usize, k: usize) -> Vec<f64> {
        let mut out = vec![0.0; rows * k];
        for r in 0..rows {
            let row = &x[r * k..(r + 1) * k];
            let c = row[0].to_f64();
            let sum: f64 = row.iter().map(|&v| (v.to_f64() - c).exp()).sum();
            for (j, &v) in row.iter().enumerate() {
                out[r * k + j] = (v.to_f64() - c).exp() / sum;
            }
        }
        out
    }

    fn ref_rope(
        x: &[f16],
        heads: usize,
        head_dim: usize,
        positions: &[u32],
        theta: f64,
    ) -> Vec<f64> {
        let half = head_dim / 2;
        let mut out = vec![0.0; x.len()];
        for (t, &pos) in positions.iter().enumerate() {
            for h in 0..heads {
                let off = (t * heads + h) * head_dim;
                for i in 0..half {
                    let angle = f64::from(pos) * theta.powf(-2.0 * i as f64 / head_dim as f64);
                    let (c, s) = (angle.cos(), angle.sin());
                    let a = x[off + i].to_f64();
                    let b = x[off + i + half].to_f64();
                    out[off + i] = a * c - b * s;
                    out[off + i + half] = b * c + a * s;
                }
            }
        }
        out
    }

    fn ref_swiglu(g: &[f16], u: &[f16]) -> Vec<f64> {
        g.iter()
            .zip(u)
            .map(|(&gv, &uv)| {
                let x = gv.to_f64();
                let silu = x / (1.0 + (-x).exp());
                silu * uv.to_f64()
            })
            .collect()
    }

    fn ref_matmul(a: &[f16], b: &[f16], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let dot: f64 = (0..k)
                    .map(|t| a[i * k + t].to_f64() * b[t * n + j].to_f64())
                    .sum();
                out[i * n + j] = dot;
            }
        }
        out
    }

    // Naive references spell out every dimension explicitly on purpose.
    #[allow(clippy::too_many_arguments)]
    fn ref_decode_attention(
        q: &[f16],
        k: &[f16],
        v: &[f16],
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        scale: f64,
        window: Option<usize>,
    ) -> Vec<f64> {
        let group = q_heads / kv_heads;
        let visible = window.unwrap_or(kv_len).min(kv_len);
        let skip = kv_len - visible;
        let mut out = vec![0.0; q_heads * head_dim];
        if visible == 0 {
            return out;
        }
        for h in 0..q_heads {
            let g = h / group;
            let mut scores: Vec<f64> = (0..visible)
                .map(|idx| {
                    let off = (skip + idx) * kv_heads * head_dim + g * head_dim;
                    let dot: f64 = (0..head_dim)
                        .map(|d| q[h * head_dim + d].to_f64() * k[off + d].to_f64())
                        .sum();
                    dot * scale
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for s in &mut scores {
                *s = (*s - max).exp();
            }
            let sum: f64 = scores.iter().sum();
            for s in &mut scores {
                *s /= sum;
            }
            for (idx, &p) in scores.iter().enumerate() {
                let off = (skip + idx) * kv_heads * head_dim + g * head_dim;
                for d in 0..head_dim {
                    out[h * head_dim + d] += p * v[off + d].to_f64();
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn ref_prefill_attention(
        q: &[f16],
        k: &[f16],
        v: &[f16],
        q_len: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        k_len: usize,
        scale: f64,
    ) -> Vec<f64> {
        let group = q_heads / kv_heads;
        let offset = k_len - q_len;
        let mut out = vec![0.0; q_len * q_heads * head_dim];
        for i in 0..q_len {
            let pos = offset + i;
            for h in 0..q_heads {
                let g = h / group;
                let mut scores: Vec<f64> = (0..=pos)
                    .map(|j| {
                        let off = (j * kv_heads + g) * head_dim;
                        let dot: f64 = (0..head_dim)
                            .map(|d| {
                                q[(i * q_heads + h) * head_dim + d].to_f64() * k[off + d].to_f64()
                            })
                            .sum();
                        dot * scale
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for s in &mut scores {
                    *s = (*s - max).exp();
                }
                let sum: f64 = scores.iter().sum();
                for s in &mut scores {
                    *s /= sum;
                }
                for (j, &p) in scores.iter().enumerate() {
                    let off = (j * kv_heads + g) * head_dim;
                    for d in 0..head_dim {
                        out[(i * q_heads + h) * head_dim + d] += p * v[off + d].to_f64();
                    }
                }
            }
        }
        out
    }

    // ---------- reference-vs-reference sanity ----------

    #[test]
    fn ref_softmax_sums_to_one_and_ratios_match_exp() {
        let x = [f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
        let p = ref_softmax(&x, 1, 3, None);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        for i in 0..3 {
            for j in 0..3 {
                let ratio = p[i] / p[j];
                let expected = (f64::from(x[i].to_f32()) - f64::from(x[j].to_f32())).exp();
                assert!((ratio - expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn ref_rmsnorm_constant_vector_is_unit_rms() {
        // x = [3, 4], w = [1, 1]: mean sq = 12.5, out = x / sqrt(12.5 + eps)
        let x = [f16::from_f32(3.0), f16::from_f32(4.0)];
        let w = [f16::ONE, f16::ONE];
        let eps = 1e-6;
        let out = ref_rmsnorm(&x, &w, 1, 2, eps);
        let expected_inv = 1.0 / (12.5 + eps).sqrt();
        assert!((out[0] - 3.0 * expected_inv).abs() < 1e-9);
        assert!((out[1] - 4.0 * expected_inv).abs() < 1e-9);
        // mean(out^2) == 12.5 / (12.5 + eps)  (exactly 1 only when eps == 0)
        let mean_sq = (out[0] * out[0] + out[1] * out[1]) / 2.0;
        assert!((mean_sq - 12.5 / (12.5 + eps)).abs() < 1e-9);
    }

    #[test]
    fn ref_rope_known_rotation() {
        // x = [1, 0, 0, 0], head_dim 4, theta 10000, position p:
        // pair (x0, x2) rotates by p * theta^0 = p; pair (x1, x3) by
        // p * theta^-0.5 but both entries are zero.
        let theta = 10000.0f64;
        let x = [f16::ONE, f16::ZERO, f16::ZERO, f16::ZERO];
        let p = 7u32;
        let out = ref_rope(&x, 1, 4, &[p], theta);
        let angle = f64::from(p);
        assert!((out[0] - angle.cos()).abs() < 1e-12);
        assert!((out[2] - angle.sin()).abs() < 1e-12);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn ref_matmul_hand_2x2() {
        let a = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let b = [
            f16::from_f32(5.0),
            f16::from_f32(6.0),
            f16::from_f32(7.0),
            f16::from_f32(8.0),
        ];
        let out = ref_matmul(&a, &b, 2, 2, 2);
        assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
    }

    // ---------- rmsnorm ----------

    #[test]
    fn rmsnorm_matches_reference() {
        let mut rng = Lcg::new(0xA11CE);
        let (rows, d) = (4usize, 16usize);
        let mut xd = vec![f16::ZERO; rows * d];
        rng.fill_f16(&mut xd);
        let mut wd = vec![f16::ZERO; d];
        rng.fill_f16(&mut wd);
        let x = Tensor::from_f16_vec(Shape::from_dims(&[rows, d]), &xd).unwrap();
        let w = Tensor::from_f16_vec(Shape::from_dims(&[d]), &wd).unwrap();

        let out = rmsnorm(&x, &w, 1e-6).unwrap();
        assert_eq!(out.shape().dims(), &[rows, d]);
        let reference = ref_rmsnorm(&xd, &wd, rows, d, 1e-6);
        assert_close(out.as_f16().unwrap(), &reference, ATOL, RTOL);
    }

    #[test]
    fn rmsnorm_handles_leading_dims() {
        let mut rng = Lcg::new(0xB0B);
        let dims = [2usize, 3usize, 8usize];
        let mut xd = vec![f16::ZERO; 2 * 3 * 8];
        rng.fill_f16(&mut xd);
        let mut wd = vec![f16::ZERO; 8];
        rng.fill_f16(&mut wd);
        let x = Tensor::from_f16_vec(Shape::from_dims(&dims), &xd).unwrap();
        let w = Tensor::from_f16_vec(Shape::from_dims(&[8]), &wd).unwrap();

        let out = rmsnorm(&x, &w, 1e-5).unwrap();
        let reference = ref_rmsnorm(&xd, &wd, 6, 8, 1e-5);
        assert_close(out.as_f16().unwrap(), &reference, ATOL, RTOL);
        assert_eq!(out.shape().dims(), &dims);
    }

    #[test]
    fn rmsnorm_analytic_and_op_agree_with_reference() {
        // Constant row c: out_j = w_j * c / sqrt(c^2 + eps)
        let c = 2.0f32;
        let eps = 1e-6f32;
        let x = Tensor::from_f16_vec(
            Shape::from_dims(&[1, 3]),
            &[f16::from_f32(c), f16::from_f32(c), f16::from_f32(c)],
        )
        .unwrap();
        let w = Tensor::from_f16_vec(
            Shape::from_dims(&[3]),
            &[f16::from_f32(0.5), f16::ONE, f16::from_f32(1.5)],
        )
        .unwrap();
        let out = rmsnorm(&x, &w, eps).unwrap();
        let inv_rms = 1.0 / (c as f64 * c as f64 + eps as f64).sqrt();
        let expected = [
            c as f64 * inv_rms * 0.5,
            c as f64 * inv_rms,
            c as f64 * inv_rms * 1.5,
        ];
        assert_close(out.as_f16().unwrap(), &expected, ATOL, RTOL);
        // Reference-vs-reference: the naive ref must agree with the analytic
        // value for the same inputs (pure f64 comparison).
        let xd = [f16::from_f32(c), f16::from_f32(c), f16::from_f32(c)];
        let wd = [f16::from_f32(0.5), f16::ONE, f16::from_f32(1.5)];
        let reference = ref_rmsnorm(&xd, &wd, 1, 3, eps as f64);
        for (r, e) in reference.iter().zip(&expected) {
            assert!((r - e).abs() < 1e-9, "ref {r} vs analytic {e}");
        }
    }

    #[test]
    fn rmsnorm_weight_shape_mismatch_is_an_error() {
        let x = Tensor::from_f16_vec(Shape::from_dims(&[2, 4]), &[f16::ONE; 8]).unwrap();
        let w = Tensor::from_f16_vec(Shape::from_dims(&[3]), &[f16::ONE; 3]).unwrap();
        assert!(matches!(
            rmsnorm(&x, &w, 1e-6),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn rmsnorm_rejects_non_f16() {
        let x = Tensor::from_f32_vec(Shape::from_dims(&[1, 2]), &[0.0, 0.0]).unwrap();
        let w = Tensor::from_f16_vec(Shape::from_dims(&[2]), &[f16::ONE, f16::ONE]).unwrap();
        assert!(matches!(
            rmsnorm(&x, &w, 1e-6),
            Err(TensorError::DTypeMismatch { .. })
        ));
    }

    // ---------- softmax ----------

    #[test]
    fn softmax_matches_reference() {
        let mut rng = Lcg::new(0xC0FFEE);
        let (rows, k) = (5usize, 7usize);
        let mut xd = vec![f16::ZERO; rows * k];
        rng.fill_f16(&mut xd);
        let x = Tensor::from_f16_vec(Shape::from_dims(&[rows, k]), &xd).unwrap();

        let out = softmax_lastdim(&x, None).unwrap();
        // rows sum to 1 (f16 rounding loosens this a little)
        for r in 0..rows {
            let sum: f32 = out.as_f16().unwrap()[r * k..(r + 1) * k]
                .iter()
                .map(|v| v.to_f32())
                .sum();
            assert!((sum - 1.0).abs() < 1e-2, "row {r} sums to {sum}");
        }
        assert_close(
            out.as_f16().unwrap(),
            &ref_softmax(&xd, rows, k, None),
            ATOL,
            RTOL,
        );
    }

    #[test]
    fn softmax_mask_keeps_true_positions() {
        let mut rng = Lcg::new(0xD00D);
        let (rows, k) = (3usize, 6usize);
        let mut xd = vec![f16::ZERO; rows * k];
        rng.fill_f16(&mut xd);
        let x = Tensor::from_f16_vec(Shape::from_dims(&[rows, k]), &xd).unwrap();
        let mask = causal_mask(rows, k);

        let out = softmax_lastdim(&x, Some(&mask)).unwrap();
        let ov = out.as_f16().unwrap();
        for r in 0..rows {
            for j in 0..k {
                if !mask[r * k + j] {
                    assert_eq!(ov[r * k + j], f16::ZERO, "masked r={r} j={j}");
                }
            }
            let sum: f32 = ov[r * k..(r + 1) * k].iter().map(|v| v.to_f32()).sum();
            assert!((sum - 1.0).abs() < 1e-2);
        }
        assert_close(ov, &ref_softmax(&xd, rows, k, Some(&mask)), ATOL, RTOL);
    }

    #[test]
    fn softmax_fully_masked_row_yields_zeros() {
        let xd = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ];
        let x = Tensor::from_f16_vec(Shape::from_dims(&[2, 3]), &xd).unwrap();
        let mask = [
            true, true, true, //
            false, false, false,
        ];
        let out = softmax_lastdim(&x, Some(&mask)).unwrap();
        let ov = out.as_f16().unwrap();
        let e: Vec<f64> = (1..=6).map(|v| (v as f64).exp()).collect();
        let sum: f64 = e[0..3].iter().sum();
        assert_close(&ov[0..3], &[e[0] / sum, e[1] / sum, e[2] / sum], ATOL, RTOL);
        assert!(ov[3..6].iter().all(|&v| v == f16::ZERO));
    }

    #[test]
    fn softmax_is_stable_for_large_magnitudes() {
        // exp(1000..1002) would overflow f32 without max subtraction; the
        // f64 raw reference computes it without subtraction.
        let x = Tensor::from_f16_vec(
            Shape::from_dims(&[1, 3]),
            &[
                f16::from_f32(1000.0),
                f16::from_f32(1001.0),
                f16::from_f32(1002.0),
            ],
        )
        .unwrap();
        let out = softmax_lastdim(&x, None).unwrap();
        assert_close(
            out.as_f16().unwrap(),
            &ref_softmax_raw(
                &[
                    f16::from_f32(1000.0),
                    f16::from_f32(1001.0),
                    f16::from_f32(1002.0),
                ],
                1,
                3,
            ),
            ATOL,
            RTOL,
        );
    }

    #[test]
    fn softmax_mask_length_mismatch_is_an_error() {
        let x = Tensor::from_f16_vec(Shape::from_dims(&[2, 3]), &[f16::ONE; 6]).unwrap();
        assert!(matches!(
            softmax_lastdim(&x, Some(&[true; 4])),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn causal_and_sliding_window_mask_patterns() {
        assert_eq!(
            causal_mask(3, 3),
            vec![
                true, false, false, //
                true, true, false, //
                true, true, true,
            ]
        );
        assert_eq!(
            sliding_window_mask(4, 4, 2),
            vec![
                true, false, false, false, //
                true, true, false, false, //
                false, true, true, false, //
                false, false, true, true,
            ]
        );
        assert!(sliding_window_mask(2, 2, 0).iter().all(|v| !v));
    }

    // ---------- rope ----------

    #[test]
    fn rope_matches_reference_with_gqa_shapes() {
        let mut rng = Lcg::new(0x50CE);
        let (seq, q_heads, kv_heads, head_dim) = (5usize, 16usize, 8usize, 16usize);
        let theta = 1e6f64;
        let positions: Vec<u32> = [0, 1, 2, 5, 123_456].to_vec();

        let mut qd = vec![f16::ZERO; seq * q_heads * head_dim];
        rng.fill_f16(&mut qd);
        let mut kd = vec![f16::ZERO; seq * kv_heads * head_dim];
        rng.fill_f16(&mut kd);
        let q = Tensor::from_f16_vec(Shape::from_dims(&[seq, q_heads, head_dim]), &qd).unwrap();
        let k = Tensor::from_f16_vec(Shape::from_dims(&[seq, kv_heads, head_dim]), &kd).unwrap();

        let (q_out, k_out) = apply_rope(&q, &k, &positions, theta).unwrap();
        assert_eq!(q_out.shape().dims(), &[seq, q_heads, head_dim]);
        assert_eq!(k_out.shape().dims(), &[seq, kv_heads, head_dim]);
        assert_close(
            q_out.as_f16().unwrap(),
            &ref_rope(&qd, q_heads, head_dim, &positions, theta),
            ATOL,
            RTOL,
        );
        assert_close(
            k_out.as_f16().unwrap(),
            &ref_rope(&kd, kv_heads, head_dim, &positions, theta),
            ATOL,
            RTOL,
        );
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let mut rng = Lcg::new(0x1DEA0);
        let (seq, heads, head_dim) = (3usize, 4usize, 8usize);
        let mut xd = vec![f16::ZERO; seq * heads * head_dim];
        rng.fill_f16(&mut xd);
        let t = Tensor::from_f16_vec(Shape::from_dims(&[seq, heads, head_dim]), &xd).unwrap();
        let (q_out, k_out) = apply_rope(&t, &t, &[0, 0, 0], 1e6).unwrap();
        assert_eq!(q_out.as_f16().unwrap(), &xd[..]);
        assert_eq!(k_out.as_f16().unwrap(), &xd[..]);
    }

    #[test]
    fn rope_matches_analytic_rotation() {
        // head_dim 4, theta 10000: inv_freq = [1, 10000^-0.5]. For
        // x = [1, 0, 0, 0] at position 7: out[0] = cos(7), out[2] = sin(7).
        let x = Tensor::from_f16_vec(
            Shape::from_dims(&[1, 1, 4]),
            &[f16::ONE, f16::ZERO, f16::ZERO, f16::ZERO],
        )
        .unwrap();
        let (q_out, _) = apply_rope(&x, &x, &[7], 10000.0).unwrap();
        let ov = q_out.as_f16().unwrap();
        assert!((ov[0].to_f32() - 7.0f64.cos() as f32).abs() < ATOL as f32);
        assert!((ov[2].to_f32() - 7.0f64.sin() as f32).abs() < ATOL as f32);
        assert_eq!(ov[1], f16::ZERO);
        assert_eq!(ov[3], f16::ZERO);
    }

    #[test]
    fn rope_validates_shapes() {
        let good =
            Tensor::from_f16_vec(Shape::from_dims(&[2, 4, 4]), &[f16::ONE; 2 * 4 * 4]).unwrap();

        // odd head_dim
        let odd = Tensor::from_f16_vec(Shape::from_dims(&[2, 4, 3]), &[f16::ONE; 24]).unwrap();
        assert!(matches!(
            apply_rope(&odd, &odd, &[0, 0], 1e6),
            Err(TensorError::InvalidShape { .. })
        ));

        // positions length mismatch
        assert!(matches!(
            apply_rope(&good, &good, &[0], 1e6),
            Err(TensorError::ShapeMismatch { .. })
        ));

        // q/k seq mismatch
        let short_k = Tensor::from_f16_vec(Shape::from_dims(&[1, 4, 4]), &[f16::ONE; 16]).unwrap();
        assert!(matches!(
            apply_rope(&good, &short_k, &[0, 0], 1e6),
            Err(TensorError::ShapeMismatch { .. })
        ));

        // rank mismatch
        let rank2 = Tensor::from_f16_vec(Shape::from_dims(&[4, 4]), &[f16::ONE; 16]).unwrap();
        assert!(matches!(
            apply_rope(&rank2, &rank2, &[0], 1e6),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    // ---------- swiglu ----------

    #[test]
    fn swiglu_matches_reference() {
        let mut rng = Lcg::new(0x5A1C0);
        let dims = [4usize, 8usize];
        let mut gd = vec![f16::ZERO; 32];
        rng.fill_f16(&mut gd);
        let mut ud = vec![f16::ZERO; 32];
        rng.fill_f16(&mut ud);
        let g = Tensor::from_f16_vec(Shape::from_dims(&dims), &gd).unwrap();
        let u = Tensor::from_f16_vec(Shape::from_dims(&dims), &ud).unwrap();

        let out = swiglu(&g, &u).unwrap();
        assert_close(out.as_f16().unwrap(), &ref_swiglu(&gd, &ud), ATOL, RTOL);
    }

    #[test]
    fn swiglu_analytic_value_and_extremes() {
        // silu(1) * 2 = 2 / (1 + e^-1)
        let expected = 2.0 / (1.0 + (-1.0f64).exp());
        let g = Tensor::from_f16_vec(
            Shape::from_dims(&[3]),
            &[
                f16::from_f32(1.0),
                f16::from_f32(-30.0),
                f16::from_f32(30.0),
            ],
        )
        .unwrap();
        let u = Tensor::from_f16_vec(
            Shape::from_dims(&[3]),
            &[f16::from_f32(2.0), f16::ONE, f16::ONE],
        )
        .unwrap();
        let out = swiglu(&g, &u).unwrap();
        let ov = out.as_f16().unwrap();
        assert!((ov[0].to_f32() as f64 - expected).abs() < ATOL);
        assert!(ov[1].to_f32().abs() < 1e-6, "silu(-30) ~ 0, got {}", ov[1]);
        assert!((f64::from(ov[2].to_f32()) - 30.0).abs() < ATOL);
    }

    #[test]
    fn swiglu_shape_mismatch_is_an_error() {
        let g = Tensor::from_f16_vec(Shape::from_dims(&[2, 4]), &[f16::ONE; 8]).unwrap();
        let u = Tensor::from_f16_vec(Shape::from_dims(&[8]), &[f16::ONE; 8]).unwrap();
        assert!(matches!(
            swiglu(&g, &u),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }

    // ---------- matmul ----------

    #[test]
    fn matmul_identity() {
        let mut eye = vec![f16::ZERO; 9];
        for (i, e) in eye.iter_mut().enumerate() {
            *e = if i % 4 == 0 { f16::ONE } else { f16::ZERO };
        }
        let a = Tensor::from_f16_vec(Shape::from_dims(&[3, 3]), &eye).unwrap();
        let mut rng = Lcg::new(0x1DEA01);
        let mut bd = vec![f16::ZERO; 9];
        rng.fill_f16(&mut bd);
        let b = Tensor::from_f16_vec(Shape::from_dims(&[3, 3]), &bd).unwrap();

        let out = matmul_f16_f32acc(&a, &b).unwrap();
        assert_eq!(out.shape().dims(), &[3, 3]);
        assert_close(
            out.as_f16().unwrap(),
            &ref_matmul(&eye, &bd, 3, 3, 3),
            ATOL,
            RTOL,
        );
    }

    #[test]
    fn matmul_hand_2x2() {
        let a = Tensor::from_f16_vec(
            Shape::from_dims(&[2, 2]),
            &[
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
        )
        .unwrap();
        let b = Tensor::from_f16_vec(
            Shape::from_dims(&[2, 2]),
            &[
                f16::from_f32(5.0),
                f16::from_f32(6.0),
                f16::from_f32(7.0),
                f16::from_f32(8.0),
            ],
        )
        .unwrap();
        let out = matmul_f16_f32acc(&a, &b).unwrap();
        let ov = out.as_f16().unwrap();
        assert_eq!(ov[0].to_f32(), 19.0);
        assert_eq!(ov[1].to_f32(), 22.0);
        assert_eq!(ov[2].to_f32(), 43.0);
        assert_eq!(ov[3].to_f32(), 50.0);
    }

    #[test]
    fn matmul_matches_reference_and_exercises_blocks() {
        // m = 70 spans multiple MC=32 row blocks (with a 6-row remainder),
        // k = 300 spans KC=256 (with a 44 remainder), n = 45 < NC=128.
        let mut rng = Lcg::new(0xABCDEF);
        let (m, k, n) = (70usize, 300usize, 45usize);
        let mut ad = vec![f16::ZERO; m * k];
        rng.fill_f16(&mut ad);
        let mut bd = vec![f16::ZERO; k * n];
        rng.fill_f16(&mut bd);
        let a = Tensor::from_f16_vec(Shape::from_dims(&[m, k]), &ad).unwrap();
        let b = Tensor::from_f16_vec(Shape::from_dims(&[k, n]), &bd).unwrap();

        let out = matmul_f16_f32acc(&a, &b).unwrap();
        assert_eq!(out.shape().dims(), &[m, n]);
        assert_close(
            out.as_f16().unwrap(),
            &ref_matmul(&ad, &bd, m, k, n),
            ATOL,
            RTOL,
        );
    }

    #[test]
    fn matmul_1x1_and_inner_product() {
        // [1,3] x [3,1]
        let a = Tensor::from_f16_vec(
            Shape::from_dims(&[1, 3]),
            &[f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
        )
        .unwrap();
        let b = Tensor::from_f16_vec(
            Shape::from_dims(&[3, 1]),
            &[f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
        )
        .unwrap();
        let out = matmul_f16_f32acc(&a, &b).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1]);
        assert_eq!(out.as_f16().unwrap()[0].to_f32(), 32.0);
    }

    #[test]
    fn matmul_k_mismatch_is_an_error() {
        let a = Tensor::from_f16_vec(Shape::from_dims(&[2, 3]), &[f16::ONE; 6]).unwrap();
        let b = Tensor::from_f16_vec(Shape::from_dims(&[4, 2]), &[f16::ONE; 8]).unwrap();
        assert!(matches!(
            matmul_f16_f32acc(&a, &b),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }

    // ---------- decode attention ----------

    #[test]
    fn decode_attention_matches_reference() {
        let mut rng = Lcg::new(0xDECADE);
        let (q_heads, kv_heads, head_dim, kv_len) = (4usize, 2usize, 8usize, 7usize);
        let mut qd = vec![f16::ZERO; q_heads * head_dim];
        rng.fill_f16(&mut qd);
        let mut kd = vec![f16::ZERO; kv_len * kv_heads * head_dim];
        rng.fill_f16(&mut kd);
        let mut vd = vec![f16::ZERO; kv_len * kv_heads * head_dim];
        rng.fill_f16(&mut vd);
        let q = Tensor::from_f16_vec(Shape::from_dims(&[q_heads, head_dim]), &qd).unwrap();

        let out = gqa_decode_attention(&q, &kd, &vd, kv_heads, head_dim, None, None).unwrap();
        assert_eq!(out.shape().dims(), &[q_heads, head_dim]);
        let reference = ref_decode_attention(
            &qd,
            &kd,
            &vd,
            q_heads,
            kv_heads,
            head_dim,
            kv_len,
            f64::from(attention_scale(head_dim)),
            None,
        );
        assert_close(out.as_f16().unwrap(), &reference, ATOL, RTOL);
    }

    #[test]
    fn decode_attention_window_keeps_last_rows() {
        let mut rng = Lcg::new(0x61A5);
        let (q_heads, kv_heads, head_dim, kv_len) = (4usize, 2usize, 8usize, 9usize);
        let mut qd = vec![f16::ZERO; q_heads * head_dim];
        rng.fill_f16(&mut qd);
        let mut kd = vec![f16::ZERO; kv_len * kv_heads * head_dim];
        rng.fill_f16(&mut kd);
        let mut vd = vec![f16::ZERO; kv_len * kv_heads * head_dim];
        rng.fill_f16(&mut vd);
        let q = Tensor::from_f16_vec(Shape::from_dims(&[q_heads, head_dim]), &qd).unwrap();

        let out = gqa_decode_attention(&q, &kd, &vd, kv_heads, head_dim, None, Some(3)).unwrap();
        let reference = ref_decode_attention(
            &qd,
            &kd,
            &vd,
            q_heads,
            kv_heads,
            head_dim,
            kv_len,
            f64::from(attention_scale(head_dim)),
            Some(3),
        );
        assert_close(out.as_f16().unwrap(), &reference, ATOL, RTOL);

        // window larger than the cache clamps to the full cache
        let full = gqa_decode_attention(&q, &kd, &vd, kv_heads, head_dim, None, Some(100)).unwrap();
        let all = gqa_decode_attention(&q, &kd, &vd, kv_heads, head_dim, None, None).unwrap();
        assert_eq!(full.as_f16().unwrap(), all.as_f16().unwrap());
    }

    #[test]
    fn decode_attention_mha_and_edge_cases() {
        let mut rng = Lcg::new(0x31415);
        // MHA: q_heads == kv_heads
        let (heads, head_dim, kv_len) = (3usize, 4usize, 5usize);
        let mut qd = vec![f16::ZERO; heads * head_dim];
        rng.fill_f16(&mut qd);
        let mut kd = vec![f16::ZERO; kv_len * heads * head_dim];
        rng.fill_f16(&mut kd);
        let mut vd = vec![f16::ZERO; kv_len * heads * head_dim];
        rng.fill_f16(&mut vd);
        let q = Tensor::from_f16_vec(Shape::from_dims(&[heads, head_dim]), &qd).unwrap();
        let out = gqa_decode_attention(&q, &kd, &vd, heads, head_dim, None, None).unwrap();
        let reference = ref_decode_attention(
            &qd,
            &kd,
            &vd,
            heads,
            heads,
            head_dim,
            kv_len,
            f64::from(attention_scale(head_dim)),
            None,
        );
        assert_close(out.as_f16().unwrap(), &reference, ATOL, RTOL);

        // empty cache -> zeros; window 0 -> zeros
        let empty: Vec<f16> = Vec::new();
        let out = gqa_decode_attention(&q, &empty, &empty, heads, head_dim, None, None).unwrap();
        assert!(out.as_f16().unwrap().iter().all(|&v| v == f16::ZERO));
        let out = gqa_decode_attention(&q, &kd, &vd, heads, head_dim, None, Some(0)).unwrap();
        assert!(out.as_f16().unwrap().iter().all(|&v| v == f16::ZERO));

        // q_heads not divisible by kv_heads
        assert!(matches!(
            gqa_decode_attention(&q, &kd, &vd, 2, head_dim, None, None),
            Err(TensorError::ShapeMismatch { .. })
        ));

        // head_dim mismatch
        assert!(matches!(
            gqa_decode_attention(&q, &kd, &vd, heads, 3, None, None),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn attention_scale_value() {
        assert!((attention_scale(64) - 0.125).abs() < 1e-6);
        assert!((attention_scale(16) - 0.25).abs() < 1e-6);
    }

    // ---------- prefill attention ----------

    #[test]
    fn prefill_attention_matches_reference() {
        let mut rng = Lcg::new(0x2B1B);
        let (q_len, k_len, q_heads, kv_heads, head_dim) = (5usize, 5usize, 4usize, 2usize, 8usize);
        let mut qd = vec![f16::ZERO; q_len * q_heads * head_dim];
        rng.fill_f16(&mut qd);
        let mut kd = vec![f16::ZERO; k_len * kv_heads * head_dim];
        rng.fill_f16(&mut kd);
        let mut vd = vec![f16::ZERO; k_len * kv_heads * head_dim];
        rng.fill_f16(&mut vd);
        let q = Tensor::from_f16_vec(Shape::from_dims(&[q_len, q_heads, head_dim]), &qd).unwrap();
        let k = Tensor::from_f16_vec(Shape::from_dims(&[k_len, kv_heads, head_dim]), &kd).unwrap();
        let v = Tensor::from_f16_vec(Shape::from_dims(&[k_len, kv_heads, head_dim]), &vd).unwrap();

        let out = causal_prefill_attention(&q, &k, &v, None).unwrap();
        assert_eq!(out.shape().dims(), &[q_len, q_heads, head_dim]);
        assert_close(
            out.as_f16().unwrap(),
            &ref_prefill_attention(
                &qd,
                &kd,
                &vd,
                q_len,
                q_heads,
                kv_heads,
                head_dim,
                k_len,
                f64::from(attention_scale(head_dim)),
            ),
            ATOL,
            RTOL,
        );
    }

    #[test]
    fn prefill_attention_q_shorter_than_k_uses_offset_causality() {
        let mut rng = Lcg::new(0x0FF1CE);
        let (q_len, k_len, q_heads, kv_heads, head_dim) = (3usize, 6usize, 2usize, 1usize, 4usize);
        let mut qd = vec![f16::ZERO; q_len * q_heads * head_dim];
        rng.fill_f16(&mut qd);
        let mut kd = vec![f16::ZERO; k_len * kv_heads * head_dim];
        rng.fill_f16(&mut kd);
        let mut vd = vec![f16::ZERO; k_len * kv_heads * head_dim];
        rng.fill_f16(&mut vd);
        let q = Tensor::from_f16_vec(Shape::from_dims(&[q_len, q_heads, head_dim]), &qd).unwrap();
        let k = Tensor::from_f16_vec(Shape::from_dims(&[k_len, kv_heads, head_dim]), &kd).unwrap();
        let v = Tensor::from_f16_vec(Shape::from_dims(&[k_len, kv_heads, head_dim]), &vd).unwrap();

        let out = causal_prefill_attention(&q, &k, &v, Some(0.25)).unwrap();
        assert_close(
            out.as_f16().unwrap(),
            &ref_prefill_attention(
                &qd, &kd, &vd, q_len, q_heads, kv_heads, head_dim, k_len, 0.25,
            ),
            ATOL,
            RTOL,
        );

        // q_len > k_len is rejected (heads divisible so the length rule is
        // what fires)
        let big_q = Tensor::from_f16_vec(Shape::from_dims(&[6, 2, 4]), &[f16::ONE; 48]).unwrap();
        let small_k = Tensor::from_f16_vec(Shape::from_dims(&[3, 1, 4]), &[f16::ONE; 12]).unwrap();
        let small_v = small_k.clone();
        assert!(matches!(
            causal_prefill_attention(&big_q, &small_k, &small_v, None),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn prefill_last_row_consistent_with_decode_attention() {
        // Reference-vs-reference: the last prefill row must equal decode
        // attention against a cache holding exactly those keys/values.
        let mut rng = Lcg::new(0x1717);
        let (seq, q_heads, kv_heads, head_dim) = (6usize, 4usize, 2usize, 8usize);
        let mut qd = vec![f16::ZERO; seq * q_heads * head_dim];
        rng.fill_f16(&mut qd);
        let mut kd = vec![f16::ZERO; seq * kv_heads * head_dim];
        rng.fill_f16(&mut kd);
        let mut vd = vec![f16::ZERO; seq * kv_heads * head_dim];
        rng.fill_f16(&mut vd);
        let q = Tensor::from_f16_vec(Shape::from_dims(&[seq, q_heads, head_dim]), &qd).unwrap();
        let k = Tensor::from_f16_vec(Shape::from_dims(&[seq, kv_heads, head_dim]), &kd).unwrap();
        let v = Tensor::from_f16_vec(Shape::from_dims(&[seq, kv_heads, head_dim]), &vd).unwrap();

        let prefill = causal_prefill_attention(&q, &k, &v, None).unwrap();
        let last_q = Tensor::from_f16_vec(
            Shape::from_dims(&[q_heads, head_dim]),
            &qd[(seq - 1) * q_heads * head_dim..],
        )
        .unwrap();
        let decode =
            gqa_decode_attention(&last_q, &kd, &vd, kv_heads, head_dim, None, None).unwrap();
        assert_close(
            &prefill.as_f16().unwrap()[(seq - 1) * q_heads * head_dim..],
            &decode
                .as_f16()
                .unwrap()
                .iter()
                .map(|v| f64::from(v.to_f32()))
                .collect::<Vec<_>>(),
            ATOL,
            RTOL,
        );
    }
}
