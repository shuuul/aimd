//! MLX affine quantization: unpack + dequantize (WS-02).
//!
//! # Verified MLX affine packing layout (mlx 0.32.2)
//!
//! This layout was reverse-engineered from the installed MLX sources
//! (`mlx/nn/layers/quantized.py`, the Metal kernel
//! `mlx/include/mlx/backend/metal/kernels/quantized.h`,
//! `affine_dequantize`) and verified empirically against `mx.quantize` /
//! `mx.dequantize` on random tensors AND on real checkpoint tensors. It is
//! locked bit-exactly by the golden fixtures exported by
//! `scripts/export_quant_reference.py` into `tests/golden/quant/`.
//!
//! A quantized matrix `W` with logical shape `(rows, cols)` (row-major, the
//! last axis is the quantization axis; `cols` must be divisible by
//! `group_size`) is stored in a safetensors checkpoint as a tensor triple.
//! MLX NN saves a quantized module `<m>` as:
//!
//! - `<m>.weight`: the packed payload, dtype `U32`, shape
//!   `(rows, cols * bits / 32)` — i.e. `bits` quantized elements per logical
//!   weight element packed into little-endian `uint32` words.
//! - `<m>.scales`: dtype f16 or bf16 (always the dtype of the original
//!   matrix), shape `(rows, cols / group_size)`.
//! - `<m>.biases`: same dtype and shape as `scales`.
//!
//! (Verified in the cached `mlx-community` checkpoints, e.g. `Qwen3-ASR`
//! `model.embed_tokens.{weight,scales,biases}` with bf16 scales.)
//!
//! ## Payload bit packing
//!
//! Within one payload row (one logical matrix row), quantized element `j`
//! (`0 <= j < cols`) is stored in raw byte `(j * bits) / 8` at bit offset
//! `(j * bits) % 8` counting from the least-significant bit, as an unsigned
//! integer. Concretely:
//!
//! - `bits = 4`: two elements per byte — element `2k` occupies the **low**
//!   nibble of byte `k`, element `2k + 1` the **high** nibble (LSB-first
//!   nibble order; each `uint32` word packs 8 elements).
//! - `bits = 8`: one element per byte, in order (each `uint32` word packs 4
//!   elements little-endian).
//!
//! Equivalently: the row payload is the little-endian byte serialization of
//! `cols * bits / 32` `uint32` words, where word `w` holds elements
//! `[8w .. 8w+7]` for `bits = 4` (nibble `i` from the LSB = element `8w + i`)
//! and elements `[4w .. 4w+3]` for `bits = 8` (byte `i` from the LSB =
//! element `4w + i`).
//!
//! ## Group layout
//!
//! Group `g` of row `r` covers logical columns `[g * group_size,
//! (g + 1) * group_size)` and uses the scalar scale/bias pair
//! `scales[r][g]` / `biases[r][g]`. Scales and biases are stored row-major.
//!
//! ## Dequantization arithmetic (bit-exact)
//!
//! ```text
//! out[r][c] = cast(scales_f32[r][c / group_size] * q[r][c]
//!                  + biases_f32[r][c / group_size])
//! ```
//!
//! where `q[r][c]` is the unpacked unsigned integer (`0 .. 2^bits - 1`), the
//! multiply-add is performed in **f32** (the f16/bf16 scale and bias are
//! first widened exactly to f32; the product `scale * q` is exact in f32 for
//! both dtypes, so a fused or unfused add is equivalent), and the result is
//! cast to f16/bf16 with round-to-nearest-even. This reproduces
//! `mx.dequantize` output **bit-exactly** for f16 and bf16, verified on
//! random tensors and on real checkpoint slices for `bits = 4` and `8`
//! (output dtype always follows the scales/biases dtype).
//!
//! ## Stored scale/bias sign convention
//!
//! The formula above is sign-agnostic. The installed MLX (0.32.2) does not
//! always store the textbook affine parameters `s = (max - min) / (2^bits -
//! 1)` (positive) with `bias = min`: per group it may instead store a
//! **negative** scale with the group **max** as bias (then `q = 0` encodes
//! the max element and `q = 2^bits - 1` the min). Both encodings satisfy
//! `w ~ scales * q + biases`; real checkpoints show mixed signs across
//! groups, and the formula reproduces `mx.dequantize` bit-exactly either
//! way. Dequantization must therefore never assume a scale sign or reorder
//! `q`.

/// Invalid input to the affine reference implementation.
#[derive(Debug, thiserror::Error)]
pub enum QuantError {
    #[error("unsupported quantization: {0}")]
    UnsupportedQuant(String),
    #[error("invalid tensor (shape/layout/offset): {0}")]
    InvalidTensor(String),
}

/// Element dtype of the scales/biases (and therefore of the dequantized
/// output). MLX affine quantization keeps the original matrix dtype, which is
/// f16 or bf16 for all `mlx-community` checkpoints used by aimd.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleDtype {
    /// IEEE half precision (`F16` in safetensors).
    F16,
    /// bfloat16 (`BF16` in safetensors).
    Bf16,
}

impl ScaleDtype {
    /// Widen a stored 16-bit pattern to f32 exactly.
    pub fn to_f32(self, bits: u16) -> f32 {
        match self {
            ScaleDtype::F16 => half::f16::from_bits(bits).to_f32(),
            ScaleDtype::Bf16 => half::bf16::from_bits(bits).to_f32(),
        }
    }

    /// Cast an f32 value to the stored dtype (round-to-nearest-even) and
    /// return its 16-bit pattern.
    fn round_to_bits(self, value: f32) -> u16 {
        match self {
            ScaleDtype::F16 => half::f16::from_f32(value).to_bits(),
            ScaleDtype::Bf16 => half::bf16::from_f32(value).to_bits(),
        }
    }
}

/// A quantized weight tensor in MLX affine format (payload triple copied out
/// of a checkpoint).
///
/// See the [module documentation](self) for the exact packing layout.
#[derive(Clone, Debug)]
pub struct QuantizedTensor {
    /// Quantization bit width (`4` or `8`).
    pub bits: u32,
    /// Quantization group size along the last (columns) axis (64 for every
    /// checkpoint aimd consumes).
    pub group_size: u32,
    /// Logical (dequantized) shape, row-major: `(rows, cols)`.
    pub shape: Vec<usize>,
    /// Dtype of `scales`/`biases` (and of the dequantized output).
    pub scale_dtype: ScaleDtype,
    /// Packed payload bytes: little-endian `uint32` words, row-major,
    /// `rows * cols * bits / 8` bytes total.
    pub payload: Vec<u8>,
    /// Scales as f16/bf16 bit patterns, row-major `(rows, cols / group_size)`.
    pub scales: Vec<u16>,
    /// Biases as f16/bf16 bit patterns, same layout as [`Self::scales`].
    pub biases: Vec<u16>,
}

/// A dequantized tensor: f16 or bf16 bit patterns in row-major order.
#[derive(Clone, Debug)]
pub struct DequantizedTensor {
    /// Element dtype (always the [`QuantizedTensor::scale_dtype`]).
    pub dtype: ScaleDtype,
    /// Logical shape `(rows, cols)`.
    pub shape: Vec<usize>,
    /// f16/bf16 bit patterns, `rows * cols` entries, row-major.
    pub data: Vec<u16>,
}

/// Dequantize an MLX affine-quantized tensor to f16/bf16 values, row-major.
///
/// MUST be bit-exact against the `scripts/export_quant_reference.py`
/// fixtures under `tests/golden/quant/` (the tests enforce this). Errors are
/// typed; malformed input never panics.
pub fn dequantize_affine(q: &QuantizedTensor) -> Result<DequantizedTensor, QuantError> {
    let bits = match q.bits {
        4 => 4usize,
        8 => 8usize,
        other => {
            return Err(QuantError::UnsupportedQuant(format!(
                "bits must be 4 or 8, got {other}"
            )));
        }
    };
    let mask = u16::try_from((1u32 << bits) - 1).expect("bits <= 8 fits u16");
    if q.group_size == 0 {
        return Err(QuantError::InvalidTensor(
            "group_size must be non-zero".to_string(),
        ));
    }
    let [rows, cols] = q.shape.as_slice() else {
        return Err(QuantError::InvalidTensor(format!(
            "quantized tensor must be 2-D, got shape {:?}",
            q.shape
        )));
    };
    let (rows, cols) = (*rows, *cols);
    let group_size = q.group_size as usize;
    if cols == 0 || cols % group_size != 0 {
        return Err(QuantError::InvalidTensor(format!(
            "cols {cols} must be non-zero and divisible by group_size {}",
            q.group_size
        )));
    }
    let groups = cols / group_size;

    let expected_payload = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(bits))
        .and_then(|n| n.checked_div(8))
        .ok_or_else(|| QuantError::InvalidTensor("shape overflows usize".to_string()))?;
    if q.payload.len() != expected_payload {
        return Err(QuantError::InvalidTensor(format!(
            "payload length {} does not match shape {rows}x{cols} at {bits} bits (expected {expected_payload})",
            q.payload.len()
        )));
    }
    let expected_groups = rows
        .checked_mul(groups)
        .ok_or_else(|| QuantError::InvalidTensor("shape overflows usize".to_string()))?;
    if q.scales.len() != expected_groups || q.biases.len() != expected_groups {
        return Err(QuantError::InvalidTensor(format!(
            "scales/biases length ({}/{}) does not match {rows} rows x {groups} groups (expected {expected_groups})",
            q.scales.len(),
            q.biases.len()
        )));
    }

    let bytes_per_row = cols * bits / 8;
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let payload_row = &q.payload[r * bytes_per_row..(r + 1) * bytes_per_row];
        let scales_row = &q.scales[r * groups..(r + 1) * groups];
        let biases_row = &q.biases[r * groups..(r + 1) * groups];
        for g in 0..groups {
            let scale = q.scale_dtype.to_f32(scales_row[g]);
            let bias = q.scale_dtype.to_f32(biases_row[g]);
            let base_col = g * group_size;
            for k in 0..group_size {
                let j = base_col + k;
                let byte_index = j * bits / 8;
                let shift = (j * bits) % 8;
                let quant = (u16::from(payload_row[byte_index]) >> shift) & mask;
                let value = scale * f32::from(quant) + bias;
                out.push(q.scale_dtype.round_to_bits(value));
            }
        }
    }

    Ok(DequantizedTensor {
        dtype: q.scale_dtype,
        shape: vec![rows, cols],
        data: out,
    })
}
