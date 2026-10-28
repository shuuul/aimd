//! The deformable decoder layer (`PPDocLayoutV3DecoderLayer`): self
//! attention with query position embeddings, multi-scale deformable cross
//! attention (bilinear `grid_sample` per level, `aimd_mlx`), ReLU FFN, all
//! post-norm.

use aimd_mlx::{Activation, MlxArray, MlxError};
use aimd_weights::WeightStore;

use super::config::{D_MODEL, FFN_DIM, NUM_HEADS, NUM_LEVELS, NUM_POINTS};
use super::layers::{Ffn, LayerNorm, SelfAttention, scalar};
use crate::models::pp_ocrv6::layers::{Linear, LoadError};

/// `PPDocLayoutV3MultiscaleDeformableAttention` with 4-coordinate
/// reference boxes.
struct DeformableAttention {
    offsets: Linear,
    weights: Linear,
    value: Linear,
    output: Linear,
}

impl DeformableAttention {
    fn load(store: &WeightStore, p: &str) -> Result<Self, LoadError> {
        let n = NUM_HEADS * NUM_LEVELS * NUM_POINTS;
        Ok(Self {
            offsets: Linear::load(store, &format!("{p}.sampling_offsets"), n * 2, D_MODEL)?,
            weights: Linear::load(store, &format!("{p}.attention_weights"), n, D_MODEL)?,
            value: Linear::load(store, &format!("{p}.value_proj"), D_MODEL, D_MODEL)?,
            output: Linear::load(store, &format!("{p}.output_proj"), D_MODEL, D_MODEL)?,
        })
    }

    /// `query` `[Q, 256]` (position embedding already added), `refs`
    /// `[Q, 4]` sigmoid cxcywh, `memory` `[S, 256]` flattened levels of
    /// `shapes`.
    fn forward(
        &self,
        query: &MlxArray,
        refs: &MlxArray,
        memory: &MlxArray,
        shapes: &[(usize, usize)],
    ) -> Result<MlxArray, MlxError> {
        let q = query.shape()?[0];
        let hd = D_MODEL / NUM_HEADS;
        let lp = NUM_LEVELS * NUM_POINTS;
        let value = self.value.forward(memory)?;
        let off = self
            .offsets
            .forward(query)?
            .reshape(&[q, NUM_HEADS, NUM_LEVELS, NUM_POINTS, 2])?;
        let aw = self
            .weights
            .forward(query)?
            .reshape(&[q, NUM_HEADS, lp])?
            .softmax(-1)?;
        let xy = refs
            .slice(&[0, 0], &[q, 2], &[1, 1])?
            .reshape(&[q, 1, 1, 1, 2])?;
        let wh = refs
            .slice(&[0, 2], &[q, 4], &[1, 1])?
            .reshape(&[q, 1, 1, 1, 2])?;
        // ref_xy + offsets / n_points * ref_wh * 0.5, then 2 * loc - 1.
        let loc = off
            .divide(&scalar(NUM_POINTS as f32)?)?
            .multiply(&wh)?
            .multiply(&scalar(0.5)?)?;
        let loc = xy.add(&loc)?;
        let grid = loc.multiply(&scalar(2.0)?)?.subtract(&scalar(1.0)?)?;
        let mut sampled = Vec::with_capacity(NUM_LEVELS);
        let mut start = 0;
        for (l, &(h, w)) in shapes.iter().enumerate() {
            let v = value
                .slice(&[start, 0], &[start + h * w, D_MODEL], &[1, 1])?
                .reshape(&[h, w, NUM_HEADS, hd])?
                .transpose(&[2, 0, 1, 3])?;
            let g = grid
                .slice(
                    &[0, 0, l, 0, 0],
                    &[q, NUM_HEADS, l + 1, NUM_POINTS, 2],
                    &[1, 1, 1, 1, 1],
                )?
                .reshape(&[q, NUM_HEADS, NUM_POINTS, 2])?
                .transpose(&[1, 0, 2, 3])?;
            sampled.push(v.grid_sample(&g)?);
            start += h * w;
        }
        let refs_s: Vec<&MlxArray> = sampled.iter().collect();
        // [heads, Q, L*P, hd] weighted by [heads, Q, L*P, 1], summed.
        let s = MlxArray::concatenate(&refs_s, 2)?;
        let a = aw.transpose(&[1, 0, 2])?.reshape(&[NUM_HEADS, q, lp, 1])?;
        let o = s
            .multiply(&a)?
            .sum_axis(2, false)?
            .transpose(&[1, 0, 2])?
            .reshape(&[q, D_MODEL])?;
        self.output.forward(&o)
    }
}

/// One decoder layer.
pub(crate) struct DecoderLayer {
    self_attn: SelfAttention,
    self_norm: LayerNorm,
    cross: DeformableAttention,
    cross_norm: LayerNorm,
    ffn: Ffn,
    ffn_norm: LayerNorm,
}

impl DecoderLayer {
    pub(crate) fn load(store: &WeightStore, p: &str) -> Result<Self, LoadError> {
        Ok(Self {
            self_attn: SelfAttention::load(store, &format!("{p}.self_attn"))?,
            self_norm: LayerNorm::load(store, &format!("{p}.self_attn_layer_norm"))?,
            cross: DeformableAttention::load(store, &format!("{p}.encoder_attn"))?,
            cross_norm: LayerNorm::load(store, &format!("{p}.encoder_attn_layer_norm"))?,
            ffn: Ffn::load(store, p, FFN_DIM, Activation::Relu)?,
            ffn_norm: LayerNorm::load(store, &format!("{p}.final_layer_norm"))?,
        })
    }

    /// `h`, `pos` `[Q, 256]`; `refs` `[Q, 4]` (sigmoid space).
    pub(crate) fn forward(
        &self,
        h: &MlxArray,
        pos: &MlxArray,
        refs: &MlxArray,
        memory: &MlxArray,
        shapes: &[(usize, usize)],
    ) -> Result<MlxArray, MlxError> {
        let a = self.self_attn.forward(h, pos)?;
        let h = self.self_norm.forward(&h.add(&a)?)?;
        let c = self.cross.forward(&h.add(pos)?, refs, memory, shapes)?;
        let h = self.cross_norm.forward(&h.add(&c)?)?;
        let f = self.ffn.forward(&h)?;
        self.ffn_norm.forward(&h.add(&f)?)
    }
}
