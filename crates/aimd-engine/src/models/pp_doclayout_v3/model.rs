//! The full `PPDocLayoutV3ForObjectDetection` forward pass: backbone,
//! hybrid encoder, decoder input projections, two-stage query selection
//! (valid anchors, `enc_output`, top-300 by max class logit), the
//! mask-enhanced initial boxes (`mask_to_box_coordinate` of the encoder
//! masks, then `inverse_sigmoid`), the six refinement layers (`bbox_embed`
//! and `class_embed` are tied to `enc_bbox_head` / `enc_score_head`), and
//! the final class logits, boxes and global-pointer order logits.

use aimd_mlx::{BinaryOp, MlxArray, MlxDtype, MlxError};
use aimd_weights::WeightStore;

use super::backbone::Backbone;
use super::config::{
    D_MODEL, DECODER_LAYERS, NUM_LABELS, NUM_PROTOTYPES, NUM_QUERIES, POINTER_HEAD,
};
use super::decoder::DecoderLayer;
use super::encoder::Encoder;
use super::layers::{LayerNorm, Mlp, SEQ_BN, conv, scalar};
use crate::models::pp_ocrv6::layers::{Conv, Linear, LoadError};

/// Raw detector outputs for one image (last decoder layer).
#[derive(Clone, Debug)]
pub struct RawOutputs {
    /// `[300, 25]` class logits.
    pub logits: Vec<f32>,
    /// `[300, 4]` normalized cxcywh boxes.
    pub boxes: Vec<f32>,
    /// `[300, 300]` reading-order logits (lower triangle = -1e4).
    pub order: Vec<f32>,
}

/// Test hooks: replace the selected queries or the initial boxes with the
/// reference's, so decoder parity is measured without top-k swaps or mask
/// sign flips cascading from earlier stages.
#[derive(Default)]
pub(crate) struct Overrides {
    pub topk: Option<Vec<usize>>,
    pub init_ref: Option<Vec<f32>>,
}

/// Intermediate tensors for the stage-parity tests (only tests read them).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct Trace {
    pub stages: Vec<MlxArray>,
    pub pan: Vec<MlxArray>,
    pub mask_feat: MlxArray,
    pub topk: Vec<usize>,
    pub init_ref: Vec<f32>,
    /// Per-layer refined boxes, `[300, 4]` each.
    pub refs: Vec<Vec<f32>>,
    /// Per-layer decoder outputs, `[300, 256]` each.
    pub hidden: Vec<MlxArray>,
}

/// `generate_anchors` validity: every anchor coordinate in (0.01, 0.99)
/// (f32), per flattened position of `shapes`.
fn valid_anchor_mask(shapes: &[(usize, usize)]) -> Vec<f32> {
    let (lo, hi) = (0.01f32, 0.99f32);
    let mut out = Vec::new();
    for (l, &(h, w)) in shapes.iter().enumerate() {
        let wh = 0.05f32 * 2f32.powi(l as i32);
        for i in 0..h {
            for j in 0..w {
                let x = (j as f32 + 0.5) / w as f32;
                let y = (i as f32 + 0.5) / h as f32;
                let ok = [x, y, wh, wh].iter().all(|&v| v > lo && v < hi);
                out.push(if ok { 1.0 } else { 0.0 });
            }
        }
    }
    out
}

/// `torch.topk(v, k).indices` (descending; ties by lower index).
fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

/// `inverse_sigmoid` (eps 1e-5) in f32.
fn inverse_sigmoid(v: f32) -> f32 {
    let x = v.clamp(0.0, 1.0);
    (x.max(1e-5) / (1.0 - x).max(1e-5)).ln()
}

/// `mask_to_box_coordinate` from per-query column / row occupancy
/// (`[Q, W]`, `[Q, H]`, 1.0 where any mask pixel is set), then
/// `inverse_sigmoid`: the initial (pre-sigmoid) reference boxes.
fn mask_boxes(xocc: &[f32], yocc: &[f32], q: usize, w: usize, h: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(q * 4);
    for i in 0..q {
        let span = |occ: &[f32]| {
            let first = occ.iter().position(|&v| v > 0.5)?;
            let last = occ.iter().rposition(|&v| v > 0.5)?;
            Some((first as f32, last as f32 + 1.0))
        };
        let xs = span(&xocc[i * w..(i + 1) * w]);
        let ys = span(&yocc[i * h..(i + 1) * h]);
        let [x0, y0, x1, y1] = match (xs, ys) {
            (Some((x0, x1)), Some((y0, y1))) => [x0, y0, x1, y1],
            _ => [0.0; 4],
        };
        let (x0, x1) = (x0 / w as f32, x1 / w as f32);
        let (y0, y1) = (y0 / h as f32, y1 / h as f32);
        let b = [(x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0];
        out.extend(b.map(inverse_sigmoid));
    }
    out
}

/// The PP-DocLayoutV3 detector.
pub struct PpDocLayoutV3Model {
    backbone: Backbone,
    encoder: Encoder,
    decoder_input_proj: Vec<Conv>,
    enc_output: Linear,
    enc_norm: LayerNorm,
    score_head: Linear,
    bbox_head: Mlp,
    query_pos: Mlp,
    mask_query: Mlp,
    decoder_norm: LayerNorm,
    layers: Vec<DecoderLayer>,
    order_head: Linear,
    pointer: Linear,
}

impl PpDocLayoutV3Model {
    pub fn load(store: &WeightStore) -> Result<Self, LoadError> {
        let lin = |n: &str, o: usize, i: usize| Linear::load(store, n, o, i);
        Ok(Self {
            backbone: Backbone::load(store)?,
            encoder: Encoder::load(store)?,
            decoder_input_proj: (0..3)
                .map(|l| {
                    conv(
                        store,
                        &format!("model.decoder_input_proj.{l}"),
                        SEQ_BN,
                        1,
                        1,
                        1,
                    )
                })
                .collect::<Result<_, _>>()?,
            enc_output: lin("model.enc_output.0", D_MODEL, D_MODEL)?,
            enc_norm: LayerNorm::load(store, "model.enc_output.1")?,
            score_head: lin("model.enc_score_head", NUM_LABELS, D_MODEL)?,
            bbox_head: Mlp::load(
                store,
                "model.enc_bbox_head",
                &[D_MODEL, D_MODEL, D_MODEL, 4],
            )?,
            query_pos: Mlp::load(
                store,
                "model.decoder.query_pos_head",
                &[4, 2 * D_MODEL, D_MODEL],
            )?,
            mask_query: Mlp::load(
                store,
                "model.mask_query_head",
                &[D_MODEL, D_MODEL, D_MODEL, NUM_PROTOTYPES],
            )?,
            decoder_norm: LayerNorm::load(store, "model.decoder_norm")?,
            layers: (0..DECODER_LAYERS)
                .map(|i| DecoderLayer::load(store, &format!("model.decoder.layers.{i}")))
                .collect::<Result<_, _>>()?,
            // Only the last layer's order logits reach the output.
            order_head: lin(
                &format!("model.decoder_order_head.{}", DECODER_LAYERS - 1),
                D_MODEL,
                D_MODEL,
            )?,
            pointer: lin(
                "model.decoder_global_pointer.dense",
                2 * POINTER_HEAD,
                D_MODEL,
            )?,
        })
    }

    /// Forward pass over `[800, 800, 3]` BGR pixels in [0, 1].
    pub fn forward(&self, pixels: &[f32], size: usize) -> Result<RawOutputs, MlxError> {
        Ok(self.forward_impl(pixels, size, &Overrides::default())?.0)
    }

    pub(crate) fn forward_impl(
        &self,
        pixels: &[f32],
        size: usize,
        ov: &Overrides,
    ) -> Result<(RawOutputs, Trace), MlxError> {
        let x = MlxArray::from_f32(&[1, size, size, 3], pixels)?;
        let stages = self.backbone.forward(&x)?;
        let enc = self.encoder.forward(&stages)?;

        // Flattened decoder memory.
        let mut shapes = Vec::new();
        let mut flats = Vec::new();
        for (c, p) in self.decoder_input_proj.iter().zip(&enc.pan) {
            let s = c.forward(p)?;
            let d = s.shape()?;
            shapes.push((d[1], d[2]));
            flats.push(s.reshape(&[d[1] * d[2], D_MODEL])?);
        }
        let flat_refs: Vec<&MlxArray> = flats.iter().collect();
        let memory = MlxArray::concatenate(&flat_refs, 0)?;
        let s_len: usize = shapes.iter().map(|(h, w)| h * w).sum();

        // Query selection.
        let valid = MlxArray::from_f32(&[s_len, 1], &valid_anchor_mask(&shapes))?;
        let out_mem = self
            .enc_norm
            .forward(&self.enc_output.forward(&valid.multiply(&memory)?)?)?;
        let topk = match &ov.topk {
            Some(t) => t.clone(),
            None => {
                let best = self
                    .score_head
                    .forward(&out_mem)?
                    .max_axes(&[1], false)?
                    .to_f32_vec()?;
                top_k(&best, NUM_QUERIES)
            }
        };
        let q = topk.len();
        let idx: Vec<u32> = topk.iter().map(|&i| i as u32).collect();
        let target = out_mem.take(&MlxArray::from_u32(&[q], &idx)?, 0)?;

        // Mask-enhanced initial boxes.
        let init_ref = match &ov.init_ref {
            Some(r) => r.clone(),
            None => {
                let mq = self
                    .mask_query
                    .forward(&self.decoder_norm.forward(&target)?)?;
                let md = enc.mask_feat.shape()?;
                let (mh, mw) = (md[1], md[2]);
                let mf = enc.mask_feat.reshape(&[mh * mw, NUM_PROTOTYPES])?;
                let on = mq
                    .matmul(&mf.transpose(&[1, 0])?, None)?
                    .binary(&scalar(0.0)?, BinaryOp::Greater)?
                    .astype(MlxDtype::F32)?
                    .reshape(&[q, mh, mw])?;
                let xocc = on.max_axes(&[1], false)?;
                let yocc = on.max_axes(&[2], false)?;
                MlxArray::eval_many(&[&xocc, &yocc])?;
                mask_boxes(&xocc.to_f32_vec()?, &yocc.to_f32_vec()?, q, mw, mh)
            }
        };

        // Decoder.
        let mut refs = MlxArray::from_f32(&[q, 4], &init_ref)?.sigmoid()?;
        let mut h = target;
        let mut trace_refs = Vec::with_capacity(self.layers.len());
        let mut hidden = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let pos = self.query_pos.forward(&refs)?;
            h = layer.forward(&h, &pos, &refs, &memory, &shapes)?;
            let inv = {
                let x = refs.clip(0.0, 1.0)?;
                let x1 = x.clip(1e-5, 1.0)?;
                let x2 = scalar(1.0)?.subtract(&x)?.clip(1e-5, 1.0)?;
                x1.divide(&x2)?.unary(aimd_mlx::UnaryOp::Log)?
            };
            refs = self.bbox_head.forward(&h)?.add(&inv)?.sigmoid()?;
            refs.eval()?;
            trace_refs.push(refs.to_f32_vec()?);
            hidden.push(h.try_clone()?);
        }

        let out_q = self.decoder_norm.forward(&h)?;
        let logits = self.score_head.forward(&out_q)?;
        let qk = self
            .pointer
            .forward(&self.order_head.forward(&out_q)?)?
            .reshape(&[q, 2, POINTER_HEAD])?;
        let qs = qk
            .slice(&[0, 0, 0], &[q, 1, POINTER_HEAD], &[1, 1, 1])?
            .reshape(&[q, POINTER_HEAD])?;
        let ks = qk
            .slice(&[0, 1, 0], &[q, 2, POINTER_HEAD], &[1, 1, 1])?
            .reshape(&[q, POINTER_HEAD])?;
        let order = qs
            .matmul(&ks.transpose(&[1, 0])?, None)?
            .divide(&scalar((POINTER_HEAD as f32).sqrt())?)?;
        MlxArray::eval_many(&[&logits, &order])?;
        let mut order = order.to_f32_vec()?;
        for i in 0..q {
            for j in 0..=i {
                order[i * q + j] = -1e4;
            }
        }
        let raw = RawOutputs {
            logits: logits.to_f32_vec()?,
            boxes: trace_refs.last().cloned().unwrap_or_default(),
            order,
        };
        let trace = Trace {
            stages,
            pan: enc.pan,
            mask_feat: enc.mask_feat,
            topk,
            init_ref,
            refs: trace_refs,
            hidden,
        };
        Ok((raw, trace))
    }
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn anchor_validity_drops_borders_and_large_levels() {
        let m = valid_anchor_mask(&[(100, 100), (50, 50), (25, 25)]);
        assert_eq!(m.len(), 13125);
        // Level 0: (0.5 / 100) = 0.005 is not > 0.01; (1.5 / 100) is.
        assert_eq!(m[0], 0.0);
        assert_eq!(m[100 + 1], 1.0);
        // Level 2: wh = 0.2 is valid; x = 24.5 / 25 = 0.98 is valid.
        assert_eq!(m[12500 + 25 + 24], 1.0);
    }

    #[test]
    fn empty_and_full_masks_map_to_reference_boxes() {
        let (w, h) = (4, 2);
        let xocc = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let yocc = [0.0, 0.0, 1.0, 1.0];
        let r = mask_boxes(&xocc, &yocc, 2, w, h);
        let empty = inverse_sigmoid(0.0);
        assert_eq!(&r[..4], &[empty; 4]);
        // x in [1, 3) of 4, y in [0, 2) of 2.
        let want = [0.5f32, 0.5, 0.5, 1.0].map(inverse_sigmoid);
        assert_eq!(&r[4..], &want);
    }
}
