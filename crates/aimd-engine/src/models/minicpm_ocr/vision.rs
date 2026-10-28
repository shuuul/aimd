//! MiniCPM-OCR vision tower (SigLIP NaViT, 4x mode) and merger, op for op
//! with mlx-vlm `minicpmv4_6` `get_vision_embedding` on one crop:
//!
//! - Packed patch embedding: `[L, 14 * 14 * 3] @ W^T` (the conv weight read
//!   as `(py, px, c)` rows) then a separate `+ bias`, bf16.
//! - Position embedding: rows `bucket_h * 70 + bucket_w` of the 70 x 70
//!   table ([`super::preprocess::position_ids`]), added, cast to bf16.
//! - 27 pre-LN blocks (`LayerNorm(eps 1e-6)`, fused `qkv`, SDPA scale
//!   `72^-0.5`, no mask, `fc1 -> gelu_tanh -> fc2`), `post_layernorm`.
//!   4x mode never runs the window `vit_merger` (its weights are not read).
//! - Merger (`merger_times = 1`): 2 x 2 merge -> `LayerNorm(4608, 1e-6)`
//!   -> `linear_1` -> exact (erf) GELU -> `linear_2` -> `[L / 4, 1024]`
//!   bf16. transformers uses `nn.GELU()` here; mlx-vlm's
//!   `GELU(approx="precise")` is the tanh form (spec 009 decisions).
//!
//! Vision and merger stay dense at every precision (mlx-vlm's convert
//! skips `vision_tower` and the predicate skips `merger`).

use aimd_mlx::{MlxArray, MlxDtype, MlxError, SdpaMask};
use half::bf16;

use super::config::VisionConfig;
use super::preprocess::{MERGE, POS_SIDE, position_ids};
use crate::models::paddleocr_vl::weights::{Lin, Loader};
use crate::models::unlimited_ocr::mlx_weights::LoadError;

/// Vision-graph failures.
#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error(transparent)]
    Mlx(#[from] MlxError),
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error("vision layout: {0}")]
    Layout(String),
}

struct VisionLayer {
    ln1_w: MlxArray,
    ln1_b: MlxArray,
    ln2_w: MlxArray,
    ln2_b: MlxArray,
    qkv: Lin,
    out_proj: Lin,
    fc1: Lin,
    fc2: Lin,
}

/// Vision tower + merger.
pub struct MiniCpmVision {
    cfg: VisionConfig,
    /// `[patch * patch * C, hidden]` (transposed packed conv weight).
    patch_wt: MlxArray,
    patch_b: MlxArray,
    pos_table: MlxArray,
    layers: Vec<VisionLayer>,
    post_w: MlxArray,
    post_b: MlxArray,
    pre_w: MlxArray,
    pre_b: MlxArray,
    linear_1: Lin,
    linear_2: Lin,
    out_dim: usize,
}

/// Captured intermediate tensors (`stage name`, value) for parity tests.
pub type Taps = Vec<(String, MlxArray)>;

impl MiniCpmVision {
    /// Loads `model.vision_tower.*` and `model.merger.mlp.0.*` (dense).
    pub(crate) fn load(
        ld: &Loader<'_>,
        cfg: &VisionConfig,
        out_dim: usize,
    ) -> Result<Self, VisionError> {
        let (d, p, c, i) = (
            cfg.hidden_size,
            cfg.patch_size,
            cfg.num_channels,
            cfg.intermediate_size,
        );
        if !d.is_multiple_of(cfg.num_attention_heads) || cfg.image_size / p != POS_SIDE {
            return Err(VisionError::Layout(format!(
                "unsupported vision config (hidden {d}, heads {}, image {} / patch {p})",
                cfg.num_attention_heads, cfg.image_size
            )));
        }
        let root = "model.vision_tower";
        // [O, C, kh, kw] -> [O, kh, kw, C] -> [O, kh * kw * C] -> transposed.
        let patch_wt = ld
            .tensor(
                &format!("{root}.embeddings.patch_embedding.weight"),
                Some(&[d, c, p, p]),
            )?
            .transpose(&[0, 2, 3, 1])?
            .reshape(&[d, p * p * c])?
            .transpose(&[1, 0])?;
        let patch_b = ld.tensor(
            &format!("{root}.embeddings.patch_embedding.bias"),
            Some(&[d]),
        )?;
        let pos_table = ld.tensor(
            &format!("{root}.embeddings.position_embedding.weight"),
            Some(&[POS_SIDE * POS_SIDE, d]),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let pre = format!("{root}.encoder.layers.{l}");
            let t = |n: &str, s: &[usize]| ld.tensor(&format!("{pre}.{n}"), Some(s));
            let qkv_w = MlxArray::concatenate(
                &[
                    &t("self_attn.q_proj.weight", &[d, d])?,
                    &t("self_attn.k_proj.weight", &[d, d])?,
                    &t("self_attn.v_proj.weight", &[d, d])?,
                ],
                0,
            )?;
            let qkv_b = MlxArray::concatenate(
                &[
                    &t("self_attn.q_proj.bias", &[d])?,
                    &t("self_attn.k_proj.bias", &[d])?,
                    &t("self_attn.v_proj.bias", &[d])?,
                ],
                0,
            )?;
            qkv_w.eval()?;
            qkv_b.eval()?;
            layers.push(VisionLayer {
                ln1_w: t("layer_norm1.weight", &[d])?,
                ln1_b: t("layer_norm1.bias", &[d])?,
                ln2_w: t("layer_norm2.weight", &[d])?,
                ln2_b: t("layer_norm2.bias", &[d])?,
                qkv: ld.linear_from(qkv_w, Some(qkv_b))?,
                out_proj: ld.linear(&format!("{pre}.self_attn.out_proj"), true, &[d, d])?,
                fc1: ld.linear(&format!("{pre}.mlp.fc1"), true, &[i, d])?,
                fc2: ld.linear(&format!("{pre}.mlp.fc2"), true, &[d, i])?,
            });
        }
        let merged = d * MERGE * MERGE;
        let m = "model.merger.mlp.0";
        Ok(Self {
            cfg: cfg.clone(),
            patch_wt,
            patch_b,
            pos_table,
            layers,
            post_w: ld.tensor(&format!("{root}.post_layernorm.weight"), Some(&[d]))?,
            post_b: ld.tensor(&format!("{root}.post_layernorm.bias"), Some(&[d]))?,
            pre_w: ld.tensor(&format!("{m}.pre_norm.weight"), Some(&[merged]))?,
            pre_b: ld.tensor(&format!("{m}.pre_norm.bias"), Some(&[merged]))?,
            linear_1: ld.linear(&format!("{m}.linear_1"), true, &[merged, merged])?,
            linear_2: ld.linear(&format!("{m}.linear_2"), true, &[out_dim, merged])?,
            out_dim,
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    fn layer(&self, l: &VisionLayer, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let eps = self.cfg.layer_norm_eps;
        let (n, d) = (x.dim(0)?, self.cfg.hidden_size);
        let heads = self.cfg.num_attention_heads;
        let hd = d / heads;
        let xn = layer_norm(x, &l.ln1_w, &l.ln1_b, eps)?;
        let qkv = l.qkv.forward(&xn)?.reshape(&[1, n, 3, heads, hd])?;
        let part = |i: usize| -> Result<MlxArray, MlxError> {
            qkv.slice(
                &[0, 0, i, 0, 0],
                &[1, n, i + 1, heads, hd],
                &[1, 1, 1, 1, 1],
            )?
            .reshape(&[1, n, heads, hd])?
            .transpose(&[0, 2, 1, 3])
        };
        let scale = (hd as f64).powf(-0.5) as f32;
        let o = part(0)?
            .scaled_dot_product_attention(&part(1)?, &part(2)?, scale, SdpaMask::None)?
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[n, d])?;
        let h = x.add(&l.out_proj.forward(&o)?)?;
        let m = l.fc2.forward(
            &l.fc1
                .forward(&layer_norm(&h, &l.ln2_w, &l.ln2_b, eps)?)?
                .astype(MlxDtype::F32)?
                .gelu_tanh()?
                .astype(MlxDtype::Bf16)?,
        )?;
        h.add(&m)
    }

    fn merger(&self, x: &MlxArray, gh: usize, gw: usize) -> Result<MlxArray, MlxError> {
        let d = self.cfg.hidden_size;
        let (hb, wb) = (gh / MERGE, gw / MERGE);
        let x = x
            .reshape(&[hb, MERGE, wb, MERGE, d])?
            .transpose(&[0, 2, 1, 3, 4])?
            .reshape(&[hb * wb, MERGE * MERGE * d])?;
        let x = layer_norm(&x, &self.pre_w, &self.pre_b, 1e-6)?;
        let h = self.linear_1.forward(&x)?;
        let h = h.astype(MlxDtype::F32)?.gelu()?.astype(MlxDtype::Bf16)?;
        self.linear_2.forward(&h)?.astype(MlxDtype::Bf16)
    }

    /// Merged features `[(gh / 2) * (gw / 2), 1024]` of one crop from its
    /// `[gh * gw, patch * patch * 3]` bf16 patch rows; with `taps`, records
    /// `patch_embed`, `embeddings`, `layer_{i}`, `post_layernorm`, `merger`.
    pub fn forward(
        &self,
        patches: &[bf16],
        gh: usize,
        gw: usize,
        mut taps: Option<&mut Taps>,
    ) -> Result<MlxArray, VisionError> {
        let p = self.cfg.patch_size;
        let row = p * p * self.cfg.num_channels;
        if gh == 0 || gw == 0 || !gh.is_multiple_of(MERGE) || !gw.is_multiple_of(MERGE) {
            return Err(VisionError::Layout(format!("crop grid {gh}x{gw}")));
        }
        if patches.len() != gh * gw * row {
            return Err(VisionError::Layout(format!(
                "{} patch values for a {gh}x{gw} grid",
                patches.len()
            )));
        }
        let mut tap = |name: &str, a: &MlxArray| -> Result<(), MlxError> {
            if let Some(t) = taps.as_deref_mut() {
                t.push((name.to_string(), a.try_clone()?));
            }
            Ok(())
        };
        let n = gh * gw;
        let x = MlxArray::from_bf16(&[n, row], patches)?;
        // Conv2d adds its bias in the GEMM epilogue: one rounding.
        let patch = self.patch_b.addmm(&x, &self.patch_wt, 1.0, 1.0)?;
        tap("patch_embed", &patch)?;
        let ids = position_ids(gh, gw, POS_SIDE);
        let pos = self.pos_table.take(&MlxArray::from_u32(&[n], &ids)?, 0)?;
        let mut x = patch.add(&pos)?.astype(MlxDtype::Bf16)?;
        tap("embeddings", &x)?;
        for (i, l) in self.layers.iter().enumerate() {
            x = self.layer(l, &x)?;
            tap(&format!("layer_{i}"), &x)?;
        }
        let x = layer_norm(&x, &self.post_w, &self.post_b, self.cfg.layer_norm_eps)?;
        tap("post_layernorm", &x)?;
        let out = self.merger(&x, gh, gw)?;
        tap("merger", &out)?;
        if out.dim(1)? != self.out_dim {
            return Err(VisionError::Layout("merger width".into()));
        }
        Ok(out)
    }
}

/// torch `F.layer_norm` on bf16: statistics, normalization and the affine
/// step in f32, one rounding (mlx's kernel rounds the normalized value to
/// the input dtype before the affine step).
fn layer_norm(x: &MlxArray, w: &MlxArray, b: &MlxArray, eps: f32) -> Result<MlxArray, MlxError> {
    x.astype(MlxDtype::F32)?
        .layer_norm(Some(w), Some(b), eps)?
        .astype(MlxDtype::Bf16)
}
