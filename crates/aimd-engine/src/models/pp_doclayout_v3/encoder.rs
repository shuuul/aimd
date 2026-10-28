//! Hybrid encoder (`PPDocLayoutV3HybridEncoder`): input projections, the
//! AIFI transformer layer on the stride-32 level, the top-down FPN and
//! bottom-up PAN with CSP-RepVGG blocks, and the mask-feature branch
//! (`MaskFeatFPN` + x4 lateral + prototype head).

use aimd_mlx::{Activation, MlxArray, MlxError};
use aimd_weights::WeightStore;

use super::config::{D_MODEL, FFN_DIM, NUM_PROTOTYPES};
use super::layers::{
    CN_BN, Ffn, HG_BN, LayerNorm, SEQ_BN, SelfAttention, conv, conv_silu, upsample_bilinear2x,
    upsample_nearest2x,
};
use crate::models::pp_ocrv6::layers::{Conv, ConvKind, LoadError};

const PREFIX: &str = "model.encoder";

/// `build_2d_sinusoidal_position_embedding(h, w, 256, 10000)`: computed in
/// f64 and cast to f32, `[sin_h | cos_h | sin_w | cos_w]` per position,
/// row-major positions.
pub fn sine_position_embedding(h: usize, w: usize) -> Vec<f32> {
    let pos_dim = D_MODEL / 4;
    let omega: Vec<f64> = (0..pos_dim)
        .map(|k| 1.0 / 10000f64.powf(k as f64 / pos_dim as f64))
        .collect();
    let mut out = Vec::with_capacity(h * w * D_MODEL);
    for y in 0..h {
        for x in 0..w {
            let (yf, xf) = (y as f64, x as f64);
            out.extend(omega.iter().map(|o| (yf * o).sin() as f32));
            out.extend(omega.iter().map(|o| (yf * o).cos() as f32));
            out.extend(omega.iter().map(|o| (xf * o).sin() as f32));
            out.extend(omega.iter().map(|o| (xf * o).cos() as f32));
        }
    }
    out
}

/// The AIFI post-norm transformer layer (GELU FFN).
struct Aifi {
    attn: SelfAttention,
    attn_norm: LayerNorm,
    ffn: Ffn,
    ffn_norm: LayerNorm,
}

impl Aifi {
    fn load(store: &WeightStore, p: &str) -> Result<Self, LoadError> {
        Ok(Self {
            attn: SelfAttention::load(store, &format!("{p}.self_attn"))?,
            attn_norm: LayerNorm::load(store, &format!("{p}.self_attn_layer_norm"))?,
            ffn: Ffn::load(store, p, FFN_DIM, Activation::Gelu)?,
            ffn_norm: LayerNorm::load(store, &format!("{p}.final_layer_norm"))?,
        })
    }

    /// `[1, H, W, 256]` -> same.
    fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let s = x.shape()?;
        let (h, w) = (s[1], s[2]);
        let tokens = x.reshape(&[h * w, D_MODEL])?;
        let pos = MlxArray::from_f32(&[h * w, D_MODEL], &sine_position_embedding(h, w))?;
        let a = self.attn.forward(&tokens, &pos)?;
        let t = self.attn_norm.forward(&tokens.add(&a)?)?;
        let f = self.ffn.forward(&t)?;
        let t = self.ffn_norm.forward(&t.add(&f)?)?;
        t.reshape(&[1, h, w, D_MODEL])
    }
}

/// `PPDocLayoutV3RepVggBlock`: `silu(bn(conv3x3(x)) + bn(conv1x1(x)))`.
struct RepVgg {
    c3: Conv,
    c1: Conv,
}

impl RepVgg {
    fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        self.c3.forward(x)?.add(&self.c1.forward(x)?)?.silu()
    }
}

/// `PPDocLayoutV3CSPRepLayer` (3 RepVGG blocks; `conv3` is the identity).
struct CspRep {
    conv1: Conv,
    conv2: Conv,
    blocks: Vec<RepVgg>,
}

impl CspRep {
    fn load(store: &WeightStore, p: &str) -> Result<Self, LoadError> {
        let blocks = (0..3)
            .map(|i| {
                let b = format!("{p}.bottlenecks.{i}");
                Ok(RepVgg {
                    c3: conv(store, &format!("{b}.conv1"), CN_BN, 3, 1, 1)?,
                    c1: conv(store, &format!("{b}.conv2"), CN_BN, 1, 1, 1)?,
                })
            })
            .collect::<Result<_, LoadError>>()?;
        Ok(Self {
            conv1: conv(store, &format!("{p}.conv1"), CN_BN, 1, 1, 1)?,
            conv2: conv(store, &format!("{p}.conv2"), CN_BN, 1, 1, 1)?,
            blocks,
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let mut h1 = conv_silu(&self.conv1, x)?;
        for b in &self.blocks {
            h1 = b.forward(&h1)?;
        }
        h1.add(&conv_silu(&self.conv2, x)?)
    }
}

/// Encoder outputs.
pub(crate) struct EncoderOut {
    /// PAN levels at strides 8 / 16 / 32, `[1, H, W, 256]`.
    pub pan: Vec<MlxArray>,
    /// `[1, 200, 200, 32]` mask prototypes.
    pub mask_feat: MlxArray,
}

/// The hybrid encoder plus its input projections.
pub(crate) struct Encoder {
    input_proj: Vec<Conv>,
    aifi: Aifi,
    lateral: Vec<Conv>,
    fpn: Vec<CspRep>,
    downsample: Vec<Conv>,
    pan: Vec<CspRep>,
    /// `scale_heads[i]` convs (one per 2x step, at least one).
    scale_heads: Vec<Vec<Conv>>,
    mask_out: Conv,
    mask_lateral: Conv,
    mask_base: Conv,
    mask_proto: Conv,
}

impl Encoder {
    pub(crate) fn load(store: &WeightStore) -> Result<Self, LoadError> {
        let p = |n: &str| format!("{PREFIX}.{n}");
        let input_proj = (0..3)
            .map(|l| {
                conv(
                    store,
                    &format!("model.encoder_input_proj.{l}"),
                    SEQ_BN,
                    1,
                    1,
                    1,
                )
            })
            .collect::<Result<_, _>>()?;
        let two = |f: &dyn Fn(usize) -> Result<Conv, LoadError>| -> Result<Vec<Conv>, LoadError> {
            (0..2).map(f).collect()
        };
        let lateral = two(&|i| conv(store, &p(&format!("lateral_convs.{i}")), CN_BN, 1, 1, 1))?;
        let downsample =
            two(&|i| conv(store, &p(&format!("downsample_convs.{i}")), CN_BN, 3, 2, 1))?;
        let fpn = (0..2)
            .map(|i| CspRep::load(store, &p(&format!("fpn_blocks.{i}"))))
            .collect::<Result<_, _>>()?;
        let pan = (0..2)
            .map(|i| CspRep::load(store, &p(&format!("pan_blocks.{i}"))))
            .collect::<Result<_, _>>()?;
        // Strides 8 / 16 / 32 over base stride 8: 1, 1 and 2 convs; layer
        // indices skip the interleaved upsample modules.
        let scale_heads = [vec![0], vec![0], vec![0, 2]]
            .iter()
            .enumerate()
            .map(|(i, layers)| {
                layers
                    .iter()
                    .map(|k| {
                        conv(
                            store,
                            &p(&format!("mask_feature_head.scale_heads.{i}.layers.{k}")),
                            HG_BN,
                            3,
                            1,
                            1,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            input_proj,
            aifi: Aifi::load(store, &p("encoder.0.layers.0"))?,
            lateral,
            fpn,
            downsample,
            pan,
            scale_heads,
            mask_out: conv(store, &p("mask_feature_head.output_conv"), HG_BN, 3, 1, 1)?,
            mask_lateral: conv(store, &p("encoder_mask_lateral"), HG_BN, 3, 1, 1)?,
            mask_base: conv(store, &p("encoder_mask_output.base_conv"), HG_BN, 3, 1, 1)?,
            mask_proto: Conv::load(
                store,
                &p("encoder_mask_output.conv"),
                ConvKind::Plain { bias: true },
                [1, 1],
                [0, 0],
                1,
            )?,
        })
    }

    /// Backbone stages (strides 4..32) -> PAN levels + mask prototypes.
    pub(crate) fn forward(&self, stages: &[MlxArray]) -> Result<EncoderOut, MlxError> {
        let mut feats: Vec<MlxArray> = self
            .input_proj
            .iter()
            .zip(&stages[1..])
            .map(|(c, x)| c.forward(x))
            .collect::<Result<_, _>>()?;
        feats[2] = self.aifi.forward(&feats[2])?;

        // Top-down FPN.
        let mut fpn = vec![feats[2].try_clone()?];
        for idx in 0..2 {
            let top = conv_silu(&self.lateral[idx], fpn.last().unwrap())?;
            *fpn.last_mut().unwrap() = top.try_clone()?;
            let up = upsample_nearest2x(&top)?;
            let fused = MlxArray::concatenate(&[&up, &feats[1 - idx]], 3)?;
            fpn.push(self.fpn[idx].forward(&fused)?);
        }
        fpn.reverse();

        // Bottom-up PAN.
        let mut pan = vec![fpn[0].try_clone()?];
        for idx in 0..2 {
            let down = conv_silu(&self.downsample[idx], pan.last().unwrap())?;
            let fused = MlxArray::concatenate(&[&down, &fpn[idx + 1]], 3)?;
            pan.push(self.pan[idx].forward(&fused)?);
        }

        // Mask features.
        let mut out: Option<MlxArray> = None;
        for (i, (head, x)) in self.scale_heads.iter().zip(&pan).enumerate() {
            let mut h = x.try_clone()?;
            for c in head {
                h = conv_silu(c, &h)?;
                if i > 0 {
                    h = upsample_bilinear2x(&h)?;
                }
            }
            out = Some(match out {
                None => h,
                Some(o) => o.add(&h)?,
            });
        }
        let m = conv_silu(&self.mask_out, &out.expect("three scale heads"))?;
        let m = upsample_bilinear2x(&m)?.add(&conv_silu(&self.mask_lateral, &stages[0])?)?;
        let m = conv_silu(&self.mask_base, &m)?;
        let mask_feat = self.mask_proto.forward(&m)?;
        debug_assert_eq!(mask_feat.shape()?[3], NUM_PROTOTYPES);
        Ok(EncoderOut { pan, mask_feat })
    }
}
