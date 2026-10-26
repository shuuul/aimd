//! Official dense `PaddlePaddle/PaddleOCR-VL-1.6` checkpoint -> MLX modules,
//! with optional load-time affine quantization.
//!
//! Names follow the upstream checkpoint; the mlx-vlm `Model.sanitize`
//! mapping is applied structurally:
//!
//! - `visual.vision_model.{embeddings,post_layernorm}` -> `visual.*`,
//!   `visual.vision_model.encoder.layers` -> `visual.layers`,
//!   `mlp_AR` -> `visual.projector`, `model.*` / `lm_head` ->
//!   `language_model.*` (the sanitized names drive the quantization
//!   predicate);
//! - vision `q_proj`/`k_proj`/`v_proj` weights and biases are concatenated
//!   along the output axis into one `qkv` linear (`[q, k, v]`);
//! - the patch-embedding conv weight `[O, C, kh, kw]` is transposed to MLX's
//!   `[O, kh, kw, C]`;
//! - `packing_position_embedding` and `vision_model.head` are dropped.
//!
//! Quantization (`quant_bits = Some(4 | 6 | 8)`) reproduces
//! `mlx_vlm.convert` (`nn.quantize(model, 64, bits, class_predicate)`):
//! every `nn.Linear` / `nn.Embedding` whose input dimension is a multiple
//! of the group size (64) and whose path names no vision/projector
//! component of mlx-vlm's skip list (`vision_model`, `vision_tower`,
//! `aligner`, `multi_modal_projector`, ... — the sanitized `visual.*` and
//! `visual.projector.*` paths match none of them) is quantized with
//! `mx.quantize(w, 64, bits)` (affine). For this checkpoint that is the
//! decoder (`embed_tokens`, all attention/MLP projections, `lm_head`), the
//! vision `qkv` / `out_proj` / `fc1`, the vision `position_embedding`, and
//! the projector `linear_1` / `linear_2`; only the 27 vision `fc2` (4304
//! inputs) stay dense. The result is bit-exact with
//! `mlx-community/PaddleOCR-VL-1.6-4bit` (checked by the checkpoint tier).

use aimd_mlx::{MlxArray, MlxError};
use aimd_weights::WeightStore;

use crate::models::unlimited_ocr::mlx_weights::{LoadError, QWeights, dense};

/// `mlx_vlm.convert` default quantization group size.
pub const QUANT_GROUP_SIZE: usize = 64;

/// Precisions accepted by [`super::pipeline::PaddleOcrVlPipeline::load`].
pub fn validate_quant_bits(bits: Option<u32>) -> Result<(), LoadError> {
    match bits {
        None | Some(4 | 6 | 8) => Ok(()),
        Some(other) => Err(LoadError::Layout(format!(
            "unsupported PaddleOCR-VL quantization: {other} bits (expected 4, 6 or 8)"
        ))),
    }
}

/// `nn.Linear` or its quantized replacement.
pub(crate) enum Lin {
    /// `addmm(bias, x, W.T)` / `x @ W.T`.
    Dense {
        wt: MlxArray,
        bias: Option<MlxArray>,
    },
    /// `quantized_matmul(x, w, transpose=True)` then `+ bias`.
    Quant(QWeights),
}

impl Lin {
    pub(crate) fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        match self {
            Lin::Dense { wt, bias: Some(b) } => b.addmm(x, wt, 1.0, 1.0),
            Lin::Dense { wt, bias: None } => x.matmul(wt, None),
            Lin::Quant(q) => q.linear(x),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_quantized(&self) -> bool {
        matches!(self, Lin::Quant(_))
    }

    /// The packed `(weight, scales, biases)` of a quantized module.
    #[cfg(test)]
    pub(crate) fn packed(&self) -> Option<[&MlxArray; 3]> {
        match self {
            Lin::Quant(q) => Some(q.packed()),
            Lin::Dense { .. } => None,
        }
    }
}

/// `nn.Embedding` or `nn.QuantizedEmbedding`.
pub(crate) enum Emb {
    Dense(MlxArray),
    Quant(QWeights),
}

impl Emb {
    /// Rows for `ids` (any-shape `u32`): `[*ids, dim]`.
    pub(crate) fn embed(&self, ids: &MlxArray) -> Result<MlxArray, MlxError> {
        match self {
            Emb::Dense(w) => w.take(ids, 0),
            Emb::Quant(q) => q.embed(ids),
        }
    }

    #[cfg(test)]
    pub(crate) fn packed(&self) -> Option<[&MlxArray; 3]> {
        match self {
            Emb::Quant(q) => Some(q.packed()),
            Emb::Dense(_) => None,
        }
    }
}

/// Reads dense modules from the official checkpoint and quantizes the
/// ones the convert predicate selects.
pub(crate) struct Loader<'a> {
    store: &'a WeightStore,
    bits: Option<u32>,
}

impl<'a> Loader<'a> {
    pub(crate) fn new(store: &'a WeightStore, bits: Option<u32>) -> Result<Self, LoadError> {
        validate_quant_bits(bits)?;
        Ok(Self { store, bits })
    }

    /// One dense tensor, verbatim.
    pub(crate) fn tensor(
        &self,
        name: &str,
        shape: Option<&[usize]>,
    ) -> Result<MlxArray, LoadError> {
        dense(self.store, name, shape)
    }

    /// `mx.quantize(w, 64, bits)` of a dense `[out, in]` weight, submitted
    /// (`async_eval`) so the dense copy is freed once the GPU is done while
    /// the host goes on reading the next tensor.
    fn quantize(
        &self,
        w: &MlxArray,
        bias: Option<MlxArray>,
        bits: u32,
    ) -> Result<QWeights, LoadError> {
        let shape = w.shape()?;
        let (qw, scales, biases) = w.quantize(QUANT_GROUP_SIZE as i32, bits as i32)?;
        MlxArray::async_eval(&[&qw, &scales, &biases])?;
        Ok(QWeights {
            w: qw,
            scales,
            biases,
            bias,
            group_size: QUANT_GROUP_SIZE as i32,
            bits: bits as i32,
            shape,
        })
    }

    /// Whether the convert predicate quantizes a module with `in_dim`
    /// inputs (the sanitized path never names a skipped component).
    fn quantizes(&self, in_dim: usize) -> Option<u32> {
        self.bits
            .filter(|_| in_dim.is_multiple_of(QUANT_GROUP_SIZE))
    }

    /// A linear from a dense `[out, in]` weight (+ optional bias).
    pub(crate) fn linear_from(
        &self,
        w: MlxArray,
        bias: Option<MlxArray>,
    ) -> Result<Lin, LoadError> {
        let shape = w.shape()?;
        if shape.len() != 2 {
            return Err(LoadError::Layout(format!("linear weight rank {shape:?}")));
        }
        if let Some(bits) = self.quantizes(shape[1]) {
            return Ok(Lin::Quant(self.quantize(&w, bias, bits)?));
        }
        Ok(Lin::Dense {
            wt: w.transpose(&[1, 0])?,
            bias,
        })
    }

    /// `module.weight` (+ `module.bias` when `bias`) as a [`Lin`].
    pub(crate) fn linear(
        &self,
        module: &str,
        bias: bool,
        shape: &[usize; 2],
    ) -> Result<Lin, LoadError> {
        let w = self.tensor(&format!("{module}.weight"), Some(shape))?;
        let b = if bias {
            Some(self.tensor(&format!("{module}.bias"), Some(&shape[..1]))?)
        } else {
            None
        };
        self.linear_from(w, b)
    }

    /// `module.weight` as an [`Emb`].
    pub(crate) fn embedding(&self, module: &str, shape: &[usize; 2]) -> Result<Emb, LoadError> {
        let w = self.tensor(&format!("{module}.weight"), Some(shape))?;
        if let Some(bits) = self.quantizes(shape[1]) {
            return Ok(Emb::Quant(self.quantize(&w, None, bits)?));
        }
        Ok(Emb::Dense(w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paddleocr_vl_weights_accept_only_supported_bits() {
        for ok in [None, Some(4), Some(6), Some(8)] {
            assert!(validate_quant_bits(ok).is_ok(), "{ok:?}");
        }
        for bad in [Some(2), Some(3), Some(5), Some(16)] {
            assert!(validate_quant_bits(bad).is_err(), "{bad:?}");
        }
    }
}
