//! PaddleOCR-VL vision tower (SigLIP-so400m-style NaViT encoder with 2-D
//! rotary attention) and the `mlp_AR` projector, op for op with mlx-vlm
//! `paddleocr_vl/vision.py` on a single image (the generic packed path:
//! 2-D activations `[N, 1152]`, one `cu_seqlens` segment, no mask).
//!
//! - Patch embed: NHWC `conv2d(stride 14) + bias`, then `astype(bf16)`.
//! - Position embedding: the 27 x 27 learned table bilinearly resized to
//!   the patch grid (`mlx_vlm.models.interpolate.bilinear_interpolate`,
//!   `align_corners=False`): sampling positions on device, floor/clip on
//!   the host, f32 corner weights in Python's evaluation order, `bf16`
//!   result.
//! - Rotary: `inv_freq = 1 / 10000^(arange(0, 36, 2) / 36)`, per patch
//!   `[freqs[row], freqs[col]]` -> `cos`/`sin` tiled to 72; applied in f32
//!   (`bf16 x f32`) and cast back to bf16.
//! - 27 pre-LN blocks (`LayerNorm(eps 1e-6)`, fused `qkv`, SDPA scale
//!   `72^-0.5`, `fc1 -> gelu_tanh -> fc2`), `post_layernorm`.
//! - Projector: `LayerNorm -> 2x2 merge [(h/2)(w/2), 4608] -> linear_1 ->
//!   gelu(erf) -> linear_2 -> [n, 1024]`.

use aimd_mlx::{MlxArray, MlxDtype, MlxError, SdpaMask, UnaryOp};

use super::config::VisionConfig;
use super::weights::{Lin, Loader};
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

/// Output width of the projector (`context_dim`, the decoder hidden size).
pub const PROJECTOR_OUT: usize = 1024;

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

/// Vision tower + projector.
pub struct PaddleVision {
    cfg: VisionConfig,
    patch_w: MlxArray,
    patch_b: MlxArray,
    /// `[num_positions, hidden]` bf16 (dequantized once when quantized).
    pos_table: MlxArray,
    pos_side: usize,
    layers: Vec<VisionLayer>,
    post_w: MlxArray,
    post_b: MlxArray,
    pre_w: MlxArray,
    pre_b: MlxArray,
    linear_1: Lin,
    linear_2: Lin,
}

/// Captured intermediate tensors (`stage name`, value) for parity tests.
pub type Taps = Vec<(String, MlxArray)>;

fn scalar(v: f32) -> Result<MlxArray, MlxError> {
    MlxArray::scalar_f32(v)
}

impl PaddleVision {
    pub(crate) fn load(ld: &Loader<'_>, cfg: &VisionConfig) -> Result<Self, VisionError> {
        let d = cfg.hidden_size;
        let p = cfg.patch_size;
        let c = cfg.num_channels;
        let i = cfg.intermediate_size;
        if !d.is_multiple_of(cfg.num_attention_heads) {
            return Err(VisionError::Layout(format!(
                "hidden {d} not divisible by {} heads",
                cfg.num_attention_heads
            )));
        }
        let root = "visual.vision_model";
        // Conv weight [O, C, kh, kw] -> MLX [O, kh, kw, C] (vision sanitize).
        let patch_w = ld
            .tensor(
                &format!("{root}.embeddings.patch_embedding.weight"),
                Some(&[d, c, p, p]),
            )?
            .transpose(&[0, 2, 3, 1])?;
        let patch_b = ld.tensor(
            &format!("{root}.embeddings.patch_embedding.bias"),
            Some(&[d]),
        )?;
        let side = cfg.image_size / p;
        let n_pos = side * side;
        let pos_emb = ld.embedding(
            &format!("{root}.embeddings.position_embedding"),
            &[n_pos, d],
        )?;
        let ids: Vec<u32> = (0..n_pos as u32).collect();
        let pos_table = pos_emb.embed(&MlxArray::from_u32(&[n_pos], &ids)?)?;
        pos_table.eval()?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let pre = format!("{root}.encoder.layers.{l}");
            let t = |n: &str, s: &[usize]| ld.tensor(&format!("{pre}.{n}"), Some(s));
            // sanitize: qkv = concat([q, k, v], axis=0) for weight and bias.
            let qw = t("self_attn.q_proj.weight", &[d, d])?;
            let kw = t("self_attn.k_proj.weight", &[d, d])?;
            let vw = t("self_attn.v_proj.weight", &[d, d])?;
            let qb = t("self_attn.q_proj.bias", &[d])?;
            let kb = t("self_attn.k_proj.bias", &[d])?;
            let vb = t("self_attn.v_proj.bias", &[d])?;
            let qkv_w = MlxArray::concatenate(&[&qw, &kw, &vw], 0)?;
            let qkv_b = MlxArray::concatenate(&[&qb, &kb, &vb], 0)?;
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
        let m = cfg.spatial_merge_size;
        let merged = d * m * m;
        Ok(Self {
            cfg: cfg.clone(),
            patch_w,
            patch_b,
            pos_table,
            pos_side: side,
            layers,
            post_w: ld.tensor(&format!("{root}.post_layernorm.weight"), Some(&[d]))?,
            post_b: ld.tensor(&format!("{root}.post_layernorm.bias"), Some(&[d]))?,
            pre_w: ld.tensor("mlp_AR.pre_norm.weight", Some(&[d]))?,
            pre_b: ld.tensor("mlp_AR.pre_norm.bias", Some(&[d]))?,
            linear_1: ld.linear("mlp_AR.linear_1", true, &[merged, merged])?,
            linear_2: ld.linear("mlp_AR.linear_2", true, &[PROJECTOR_OUT, merged])?,
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    /// The (dequantized) `[num_positions, hidden]` position table.
    #[cfg(test)]
    pub(crate) fn position_table(&self) -> &MlxArray {
        &self.pos_table
    }

    #[cfg(test)]
    pub(crate) fn layer_linear(&self, layer: usize, name: &str) -> &Lin {
        let l = &self.layers[layer];
        match name {
            "qkv" => &l.qkv,
            "out_proj" => &l.out_proj,
            "fc1" => &l.fc1,
            "fc2" => &l.fc2,
            other => panic!("unknown vision linear {other}"),
        }
    }

    #[cfg(test)]
    pub(crate) fn projector_linear(&self, name: &str) -> &Lin {
        match name {
            "linear_1" => &self.linear_1,
            "linear_2" => &self.linear_2,
            other => panic!("unknown projector linear {other}"),
        }
    }

    /// Patch embedding of `[N, C, p, p]` bf16 pixels -> `[N, hidden]`.
    fn patch_embed(&self, pixels: &MlxArray) -> Result<MlxArray, VisionError> {
        let n = pixels.dim(0)?;
        let p = self.cfg.patch_size;
        let x = pixels.astype(MlxDtype::Bf16)?.transpose(&[0, 2, 3, 1])?;
        let y = x
            .conv2d(&self.patch_w, [p, p], [0, 0])?
            .add(&self.patch_b)?
            .astype(MlxDtype::Bf16)?;
        Ok(y.reshape(&[n, self.cfg.hidden_size])?)
    }

    /// `bilinear_interpolate` sampling positions for `n` outputs over an
    /// input side of `side` (`(arange(n) + 0.5) * side / n - 0.5`, f32 on
    /// device; `[0.0]` when `n == 1`).
    fn sample_positions(side: usize, n: usize) -> Result<Vec<f32>, MlxError> {
        if n == 1 {
            return Ok(vec![0.0]);
        }
        let idx: Vec<f32> = (0..n).map(|v| v as f32).collect();
        MlxArray::from_f32(&[n], &idx)?
            .add(&scalar(0.5)?)?
            .multiply(&scalar(side as f32)?)?
            .divide(&scalar(n as f32)?)?
            .subtract(&scalar(0.5)?)?
            .to_f32_vec()
    }

    /// The interpolated position embedding for a `gh x gw` grid ->
    /// `[gh * gw, hidden]` bf16.
    fn position_embedding(&self, gh: usize, gw: usize) -> Result<MlxArray, VisionError> {
        let side = self.pos_side;
        let d = self.cfg.hidden_size;
        let rows = Self::sample_positions(side, gh)?;
        let cols = Self::sample_positions(side, gw)?;
        let split = |pos: &[f32]| -> (Vec<u32>, Vec<u32>, Vec<f32>) {
            let hi = side as i64 - 1;
            let mut lo_i = Vec::with_capacity(pos.len());
            let mut hi_i = Vec::with_capacity(pos.len());
            let mut wt = Vec::with_capacity(pos.len());
            for &p in pos {
                let f = p.floor() as i64;
                let fl = f.clamp(0, hi);
                let ce = (f + 1).clamp(0, hi);
                lo_i.push(fl as u32);
                hi_i.push(ce as u32);
                wt.push(p - fl as f32);
            }
            (lo_i, hi_i, wt)
        };
        let (rf, rc, rw) = split(&rows);
        let (cf, cc, cw) = split(&cols);
        let gather = |ri: &[u32], ci: &[u32]| -> Result<MlxArray, MlxError> {
            let mut flat = Vec::with_capacity(gh * gw);
            for &r in ri {
                for &c in ci {
                    flat.push(r * side as u32 + c);
                }
            }
            self.pos_table
                .take(&MlxArray::from_u32(&[gh * gw], &flat)?, 0)?
                .reshape(&[gh, gw, d])
        };
        // Corner weights in Python's evaluation order, f32 [gh, gw, 1].
        let weight = |f: &dyn Fn(f32, f32) -> f32| -> Result<MlxArray, MlxError> {
            let mut w = Vec::with_capacity(gh * gw);
            for &r in &rw {
                for &c in &cw {
                    w.push(f(r, c));
                }
            }
            MlxArray::from_f32(&[gh, gw, 1], &w)
        };
        let tl = gather(&rf, &cf)?;
        let tr = gather(&rf, &cc)?;
        let bl = gather(&rc, &cf)?;
        let br = gather(&rc, &cc)?;
        let w_tl = weight(&|r, c| (1.0 - r) * (1.0 - c))?;
        let w_tr = weight(&|r, c| (1.0 - r) * c)?;
        let w_bl = weight(&|r, c| r * (1.0 - c))?;
        let w_br = weight(&|r, c| r * c)?;
        let out = w_tl
            .multiply(&tl)?
            .add(&w_tr.multiply(&tr)?)?
            .add(&w_bl.multiply(&bl)?)?
            .add(&w_br.multiply(&br)?)?
            .astype(MlxDtype::Bf16)?;
        Ok(out.reshape(&[gh * gw, d])?)
    }

    /// Rotary `(cos, sin)` as f32 `[1, N, 1, head_dim]` (`rot_pos_emb` +
    /// `apply_rotary_pos_emb_vision`'s tiling).
    fn rotary(&self, gh: usize, gw: usize) -> Result<(MlxArray, MlxArray), MlxError> {
        let head_dim = self.cfg.hidden_size / self.cfg.num_attention_heads;
        let dim = head_dim / 2;
        let half = dim.div_ceil(2);
        let steps: Vec<f32> = (0..half).map(|i| (2 * i) as f32).collect();
        let inv_freq =
            scalar(1.0)?
                .divide(&scalar(10000.0)?.power(
                    &MlxArray::from_f32(&[half], &steps)?.divide(&scalar(dim as f32)?)?,
                )?)?;
        let max_grid = gh.max(gw);
        let seq: Vec<f32> = (0..max_grid).map(|v| v as f32).collect();
        let freqs =
            MlxArray::from_f32(&[max_grid, 1], &seq)?.multiply(&inv_freq.reshape(&[1, half])?)?;
        let n = gh * gw;
        let hid: Vec<u32> = (0..n as u32).map(|i| i / gw as u32).collect();
        let wid: Vec<u32> = (0..n as u32).map(|i| i % gw as u32).collect();
        let fh = freqs.take(&MlxArray::from_u32(&[n], &hid)?, 0)?;
        let fw = freqs.take(&MlxArray::from_u32(&[n], &wid)?, 0)?;
        let f = MlxArray::concatenate(&[&fh, &fw], 1)?;
        let tile = |a: MlxArray| -> Result<MlxArray, MlxError> {
            MlxArray::concatenate(&[&a, &a], 1)?.reshape(&[1, n, 1, 4 * half])
        };
        Ok((tile(f.unary(UnaryOp::Cos)?)?, tile(f.unary(UnaryOp::Sin)?)?))
    }

    fn attention(
        &self,
        l: &VisionLayer,
        x: &MlxArray,
        cos: &MlxArray,
        sin: &MlxArray,
    ) -> Result<MlxArray, MlxError> {
        let n = x.dim(0)?;
        let heads = self.cfg.num_attention_heads;
        let hd = self.cfg.hidden_size / heads;
        let qkv = l.qkv.forward(x)?.reshape(&[1, n, 3, heads, hd])?;
        let part = |i: usize| -> Result<MlxArray, MlxError> {
            qkv.slice(
                &[0, 0, i, 0, 0],
                &[1, n, i + 1, heads, hd],
                &[1, 1, 1, 1, 1],
            )?
            .reshape(&[1, n, heads, hd])
        };
        let rope = |t: MlxArray| -> Result<MlxArray, MlxError> {
            let h = hd / 2;
            let x1 = t.slice(&[0, 0, 0, 0], &[1, n, heads, h], &[1, 1, 1, 1])?;
            let x2 = t.slice(&[0, 0, 0, h], &[1, n, heads, hd], &[1, 1, 1, 1])?;
            let rot = MlxArray::concatenate(&[&x2.unary(UnaryOp::Negative)?, &x1], 3)?;
            t.multiply(cos)?
                .add(&rot.multiply(sin)?)?
                .astype(MlxDtype::Bf16)
        };
        let q = rope(part(0)?)?.transpose(&[0, 2, 1, 3])?;
        let k = rope(part(1)?)?.transpose(&[0, 2, 1, 3])?;
        let v = part(2)?.transpose(&[0, 2, 1, 3])?;
        let scale = (hd as f64).powf(-0.5) as f32;
        let o = q
            .scaled_dot_product_attention(&k, &v, scale, SdpaMask::None)?
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[n, self.cfg.hidden_size])?;
        l.out_proj.forward(&o)
    }

    fn layer(
        &self,
        l: &VisionLayer,
        x: &MlxArray,
        cos: &MlxArray,
        sin: &MlxArray,
    ) -> Result<MlxArray, MlxError> {
        let eps = self.cfg.layer_norm_eps;
        let h = x.add(&self.attention(
            l,
            &x.layer_norm(Some(&l.ln1_w), Some(&l.ln1_b), eps)?,
            cos,
            sin,
        )?)?;
        let m = l.fc2.forward(
            &l.fc1
                .forward(&h.layer_norm(Some(&l.ln2_w), Some(&l.ln2_b), eps)?)?
                .gelu_tanh()?,
        )?;
        h.add(&m)
    }

    fn projector(&self, x: &MlxArray, gh: usize, gw: usize) -> Result<MlxArray, MlxError> {
        let d = self.cfg.hidden_size;
        let m = self.cfg.spatial_merge_size;
        let (hb, wb) = (gh / m, gw / m);
        let x = x
            .layer_norm(Some(&self.pre_w), Some(&self.pre_b), 1e-6)?
            .reshape(&[hb, m, wb, m, d])?
            .transpose(&[0, 2, 1, 3, 4])?
            .reshape(&[hb * wb, m * m * d])?;
        self.linear_2.forward(&self.linear_1.forward(&x)?.gelu()?)
    }

    /// Image features `[(gh / m) * (gw / m), 1024]` for `[N, C, p, p]`
    /// pixels and `grid_thw = [1, gh, gw]`; with `taps`, records
    /// `patch_embed`, `embeddings`, `layer_{i}`, `post_layernorm`,
    /// `projector`.
    pub fn forward(
        &self,
        pixels: &MlxArray,
        grid_thw: [usize; 3],
        mut taps: Option<&mut Taps>,
    ) -> Result<MlxArray, VisionError> {
        let [t, gh, gw] = grid_thw;
        let m = self.cfg.spatial_merge_size;
        if t != 1 || gh % m != 0 || gw % m != 0 {
            return Err(VisionError::Layout(format!(
                "unsupported grid_thw {grid_thw:?} (single image, merge {m})"
            )));
        }
        let p = self.cfg.patch_size;
        let want = [gh * gw, self.cfg.num_channels, p, p];
        if pixels.shape()? != want {
            return Err(VisionError::Layout(format!(
                "pixels {:?}, expected {want:?}",
                pixels.shape()?
            )));
        }
        let mut tap = |name: &str, a: &MlxArray| -> Result<(), MlxError> {
            if let Some(t) = taps.as_deref_mut() {
                t.push((name.to_string(), a.try_clone()?));
            }
            Ok(())
        };
        let patch = self.patch_embed(pixels)?;
        tap("patch_embed", &patch)?;
        let mut x = patch.add(&self.position_embedding(gh, gw)?)?;
        tap("embeddings", &x)?;
        let (cos, sin) = self.rotary(gh, gw)?;
        for (i, l) in self.layers.iter().enumerate() {
            x = self.layer(l, &x, &cos, &sin)?;
            tap(&format!("layer_{i}"), &x)?;
        }
        let x = x.layer_norm(
            Some(&self.post_w),
            Some(&self.post_b),
            self.cfg.layer_norm_eps,
        )?;
        tap("post_layernorm", &x)?;
        let out = self.projector(&x, gh, gw)?;
        tap("projector", &out)?;
        Ok(out)
    }
}
