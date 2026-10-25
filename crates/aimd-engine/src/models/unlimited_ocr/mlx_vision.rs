//! Unlimited-OCR vision encoder on MLX: SAM ViT-B + CLIP-L towers, the
//! 4-bit linear projector, and the gundam token shaping.
//!
//! # Op-for-op port of the mlx_vlm graph
//!
//! The porting source of truth is mlx_vlm 0.7.0
//! (`.venv/.../mlx_vlm/models/deepseekocr/{sam.py, vision.py,
//! deepseekocr.py}` + `unlimited_ocr/unlimitedocr.py`); every stage mirrors
//! one Python call with the same dtypes (bf16 activations, bf16 weights,
//! fast LayerNorm/SDPA kernels, exact-erf GELU), so Metal sees the same
//! kernel sequence:
//!
//! - SAM: `patch_embed` conv 16x16/s16 + bias, `+ pos_embed` (raw `[1, 64,
//!   64, 768]` for the 1024 view; bicubic-antialias resample to the 40-grid
//!   for 640 crops), 12 pre-LN blocks (window 14 with zero padding, global
//!   blocks 2/5/8/11) whose attention adds the decomposed rel-pos bias
//!   (`einsum("bhwc,hkc->bhwk")` + `einsum("bhwc,wkc->bhwk")`, expressed as
//!   batched matmuls over the same operands) as an additive SDPA mask,
//!   then the neck (1x1 conv, LN, 3x3 conv, LN) and the `net_2`/`net_3`
//!   stride-2 convs.
//! - CLIP: `flatten(sam_out)` patch embeds after a broadcast class token,
//!   `+ position_embedding` (raw 257 rows, or the quirky `Upsample(cubic)`
//!   interpolation for crops), `pre_layrnorm` (eps 1e-5), 24 pre-LN layers
//!   (fused qkv, 16 heads, unmasked SDPA, GELU MLP).
//! - `concat(clip[:, 1:], sam.flatten(1, 2))` -> quantized projector (+bias);
//!   crops are batched in one tower call (`B = n_crops`), the global view
//!   runs with `B = 1`; local rows get one `image_newline` per grid row,
//!   then the global grid with its newline column, then `view_separator`.
//!
//! Host-side tables (Rust, bit-compatible with the Python kernels and
//! locked by the golden tests): the rel-pos gathers (+ the 127 -> 79 linear
//! resample for 40-grid global blocks), the SAM pos-embed bicubic resample
//! and the CLIP pos-embed interpolation. They are computed once per grid
//! size, cached, and uploaded as bf16 (exactly the reference's
//! `.astype(bf16)` values).

use std::path::Path;
use std::sync::Mutex;

use aimd_mlx::{MlxArray, MlxDtype, MlxError, SdpaMask};
use aimd_weights::{WeightStore, WeightsError, load_checkpoint};
use half::bf16;

use super::config::{UnlimitedOcrConfig, parse_config};
use super::mlx_weights::{Linear, LoadError, QWeights, dense, host_f32};
use super::preprocess::{
    GUNDAM_BASE_PX, GUNDAM_CROP_PX, GundamPreprocess, PreprocessError, clip_pos_embed_interp,
    get_rel_pos_table, pixels_nhwc_bf16, preprocess_gundam, sam_pos_embed_interp,
};

/// Errors surfaced by the vision encoder.
#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error(transparent)]
    Mlx(#[from] MlxError),
    #[error(transparent)]
    Weights(#[from] WeightsError),
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error(transparent)]
    Preprocess(#[from] PreprocessError),
    #[error("io: {0}")]
    Io(String),
    #[error("config: {0}")]
    Config(String),
    #[error("image decode: {0}")]
    Image(String),
    #[error("layout: {0}")]
    Layout(String),
}

/// Stage tags for the golden-parity hooks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StageTag {
    SamPatchEmbed,
    SamPosAdded,
    SamBlock(usize),
    SamFinal,
    ClipPreNorm,
    ClipLayer(usize),
    ClipFinal,
    ProjectorGlobal,
    ProjectorLocal,
}

/// Collected stage tensors (lazy MLX arrays, channel-last).
pub(crate) type Stages = Vec<(StageTag, MlxArray)>;

fn push_stage(sink: &mut Option<&mut Stages>, tag: StageTag, x: &MlxArray) -> Result<(), MlxError> {
    if let Some(stages) = sink {
        stages.push((tag, x.try_clone()?));
    }
    Ok(())
}

fn bf16_array(shape: &[usize], data: &[f32]) -> Result<MlxArray, MlxError> {
    let v: Vec<bf16> = data.iter().map(|&x| bf16::from_f32(x)).collect();
    MlxArray::from_bf16(shape, &v)
}

/// `nn.LayerNorm` (fast kernel).
struct LayerNorm {
    w: MlxArray,
    b: MlxArray,
    eps: f32,
}

impl LayerNorm {
    fn load(store: &WeightStore, module: &str, eps: f32) -> Result<Self, LoadError> {
        Ok(Self {
            w: dense(store, &format!("{module}.weight"), None)?,
            b: dense(store, &format!("{module}.bias"), None)?,
            eps,
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        x.layer_norm(Some(&self.w), Some(&self.b), self.eps)
    }
}

/// `nn.Conv2d` (MLX `[O, KH, KW, I]` weight): `conv2d(x, w)` then `+ bias`.
struct Conv {
    w: MlxArray,
    b: Option<MlxArray>,
    stride: usize,
    pad: usize,
}

impl Conv {
    fn load(
        store: &WeightStore,
        module: &str,
        bias: bool,
        stride: usize,
        pad: usize,
    ) -> Result<Self, LoadError> {
        let w = dense(store, &format!("{module}.weight"), None)?;
        if w.ndim()? != 4 {
            return Err(LoadError::Layout(format!(
                "{module}.weight must be rank-4 OHWI"
            )));
        }
        let b = if bias {
            Some(dense(store, &format!("{module}.bias"), None)?)
        } else {
            None
        };
        Ok(Self { w, b, stride, pad })
    }

    fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let y = x.conv2d(&self.w, [self.stride, self.stride], [self.pad, self.pad])?;
        match &self.b {
            Some(b) => y.add(b),
            None => Ok(y),
        }
    }
}

/// Per-grid-size cache of uploaded host tables.
struct TableCache {
    entries: Mutex<Vec<(usize, Vec<MlxArray>)>>,
}

impl TableCache {
    fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }

    fn get_or_build(
        &self,
        key: usize,
        build: impl FnOnce() -> Result<Vec<MlxArray>, VisionError>,
    ) -> Result<Vec<MlxArray>, VisionError> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, arrays)) = entries.iter().find(|(k, _)| *k == key) {
            return arrays
                .iter()
                .map(|a| a.try_clone().map_err(VisionError::from))
                .collect();
        }
        let arrays = build()?;
        let out = arrays
            .iter()
            .map(|a| a.try_clone().map_err(VisionError::from))
            .collect::<Result<Vec<_>, _>>()?;
        entries.push((key, arrays));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// SAM
// ---------------------------------------------------------------------------

struct SamBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    proj: Linear,
    lin1: Linear,
    lin2: Linear,
    /// 0 = global attention.
    window: usize,
    heads: usize,
    /// Host rel-pos rows `[rel_rows, head_dim]` (bf16-exact f32).
    rel_pos_h: Vec<f32>,
    rel_pos_w: Vec<f32>,
    rel_rows: usize,
    head_dim: usize,
    /// `(Rh, Rw)` tables `[q, head_dim, q]` keyed by `q`.
    tables: TableCache,
}

impl SamBlock {
    fn load(
        store: &WeightStore,
        i: usize,
        window: usize,
        heads: usize,
    ) -> Result<Self, VisionError> {
        let p = format!("sam_model.blocks.{i}");
        let (rel_pos_h, sh) = host_f32(store, &format!("{p}.attn.rel_pos_h"))?;
        let (rel_pos_w, sw) = host_f32(store, &format!("{p}.attn.rel_pos_w"))?;
        if sh.len() != 2 || sh != sw {
            return Err(VisionError::Layout(format!(
                "{p}: rel_pos shapes {sh:?} / {sw:?}"
            )));
        }
        let block = Self {
            norm1: LayerNorm::load(store, &format!("{p}.norm1"), 1e-6)?,
            norm2: LayerNorm::load(store, &format!("{p}.norm2"), 1e-6)?,
            qkv: Linear::load(store, &format!("{p}.attn.qkv"), true)?,
            proj: Linear::load(store, &format!("{p}.attn.proj"), true)?,
            lin1: Linear::load(store, &format!("{p}.mlp.lin1"), true)?,
            lin2: Linear::load(store, &format!("{p}.mlp.lin2"), true)?,
            window,
            heads,
            rel_rows: sh[0],
            head_dim: sh[1],
            rel_pos_h,
            rel_pos_w,
            tables: TableCache::new(),
        };
        Ok(block)
    }

    /// `get_rel_pos` tables for a `q x q` query grid, laid out `[q, c, k]`
    /// (the einsum's per-row `[c, k]` right operand).
    fn rel_tables(&self, q: usize) -> Result<(MlxArray, MlxArray), VisionError> {
        let c = self.head_dim;
        let mut t = self.tables.get_or_build(q, || {
            let mut out = Vec::with_capacity(2);
            for rel in [&self.rel_pos_h, &self.rel_pos_w] {
                let qkc = get_rel_pos_table(rel, self.rel_rows, c, q);
                let mut qck = vec![bf16::ZERO; q * c * q];
                for i in 0..q {
                    for k in 0..q {
                        for ch in 0..c {
                            qck[(i * c + ch) * q + k] = bf16::from_f32(qkc[(i * q + k) * c + ch]);
                        }
                    }
                }
                out.push(MlxArray::from_bf16(&[q, c, q], &qck)?);
            }
            Ok(out)
        })?;
        let rw = t.pop().expect("two tables");
        let rh = t.pop().expect("two tables");
        Ok((rh, rw))
    }

    /// `Attention.__call__` over `[B, H, W, C]` (H == W).
    fn attention(&self, x: &MlxArray) -> Result<MlxArray, VisionError> {
        let s = x.shape()?;
        let (b, h, w, c) = (s[0], s[1], s[2], s[3]);
        if h != w {
            return Err(VisionError::Layout(format!(
                "SAM attention grid {h}x{w} not square"
            )));
        }
        let heads = self.heads;
        let hd = c / heads;
        let l = h * w;
        let bh = b * heads;
        let qkv = self
            .qkv
            .forward(x)?
            .reshape(&[b, l, 3, heads, hd])?
            .transpose(&[2, 0, 3, 1, 4])?
            .reshape(&[3, bh, l, hd])?;
        let part = |i: usize| -> Result<MlxArray, MlxError> {
            qkv.slice(&[i, 0, 0, 0], &[i + 1, bh, l, hd], &[1, 1, 1, 1])?
                .reshape(&[bh, l, hd])
        };
        let (q, k, v) = (part(0)?, part(1)?, part(2)?);

        // add_decomposed_rel_pos: r_q = q.reshape(B*heads, h, w, c).
        let (rh, rw) = self.rel_tables(h)?;
        let r_q = q.reshape(&[bh, h, w, hd])?;
        // einsum("bhwc,hkc->bhwk"): per query row h, [(b, w), c] @ [c, k].
        let rel_h = r_q
            .transpose(&[1, 0, 2, 3])?
            .reshape(&[h, bh * w, hd])?
            .matmul(&rh, None)?
            .reshape(&[h, bh, w, h])?
            .transpose(&[1, 0, 2, 3])?;
        // einsum("bhwc,wkc->bhwk"): per query column w, [(b, h), c] @ [c, k].
        let rel_w = r_q
            .transpose(&[2, 0, 1, 3])?
            .reshape(&[w, bh * h, hd])?
            .matmul(&rw, None)?
            .reshape(&[w, bh, h, w])?
            .transpose(&[1, 2, 0, 3])?;
        let bias = rel_h
            .reshape(&[b, heads, l, h, 1])?
            .add(&rel_w.reshape(&[b, heads, l, 1, w])?)?
            .reshape(&[b, heads, l, h * w])?;

        let q = q.reshape(&[b, heads, l, hd])?;
        let k = k.reshape(&[b, heads, l, hd])?;
        let v = v.reshape(&[b, heads, l, hd])?;
        let scale = (hd as f32).powf(-0.5);
        let o = q.scaled_dot_product_attention(&k, &v, scale, SdpaMask::Array(&bias))?;
        let o = o
            .reshape(&[b, heads, h, w, hd])?
            .transpose(&[0, 2, 3, 1, 4])?
            .reshape(&[b, h, w, c])?;
        Ok(self.proj.forward(&o)?)
    }

    fn forward(&self, x: &MlxArray) -> Result<MlxArray, VisionError> {
        let shortcut = x;
        let mut y = self.norm1.forward(x)?;
        if self.window > 0 {
            let s = y.shape()?;
            let (h, w) = (s[1], s[2]);
            let (win, hp, wp) = window_partition(&y, self.window)?;
            let att = self.attention(&win)?;
            y = window_unpartition(&att, self.window, (hp, wp), (h, w))?;
        } else {
            y = self.attention(&y)?;
        }
        let x = shortcut.add(&y)?;
        let m = self
            .lin2
            .forward(&self.lin1.forward(&self.norm2.forward(&x)?)?.gelu()?)?;
        Ok(x.add(&m)?)
    }
}

/// `window_partition`: zero-pad to a multiple of `ws`, then
/// `[B*nW, ws, ws, C]`. Returns `(windows, Hp, Wp)`.
pub(crate) fn window_partition(
    x: &MlxArray,
    ws: usize,
) -> Result<(MlxArray, usize, usize), MlxError> {
    let s = x.shape()?;
    let (b, h, w, c) = (s[0], s[1], s[2], s[3]);
    let pad_h = (ws - h % ws) % ws;
    let pad_w = (ws - w % ws) % ws;
    let padded = if pad_h > 0 || pad_w > 0 {
        x.pad(&[(0, 0), (0, pad_h), (0, pad_w), (0, 0)])?
    } else {
        x.try_clone()?
    };
    let (hp, wp) = (h + pad_h, w + pad_w);
    let win = padded
        .reshape(&[b, hp / ws, ws, wp / ws, ws, c])?
        .transpose(&[0, 1, 3, 2, 4, 5])?
        .reshape(&[b * (hp / ws) * (wp / ws), ws, ws, c])?;
    Ok((win, hp, wp))
}

/// `window_unpartition`: inverse of [`window_partition`] + crop to `(H, W)`.
pub(crate) fn window_unpartition(
    win: &MlxArray,
    ws: usize,
    (hp, wp): (usize, usize),
    (h, w): (usize, usize),
) -> Result<MlxArray, MlxError> {
    let s = win.shape()?;
    let c = s[3];
    let b = s[0] / (hp * wp / ws / ws);
    let x = win
        .reshape(&[b, hp / ws, wp / ws, ws, ws, c])?
        .transpose(&[0, 1, 3, 2, 4, 5])?
        .reshape(&[b, hp, wp, c])?;
    if hp > h || wp > w {
        x.slice(&[0, 0, 0, 0], &[b, h, w, c], &[1, 1, 1, 1])
    } else {
        Ok(x)
    }
}

struct Sam {
    patch_embed: Conv,
    pos_embed: MlxArray,
    /// Host `[64*64, 768]` pos embed for the crop-grid resample.
    pos_host: Vec<f32>,
    pos_grid: usize,
    embed_dim: usize,
    blocks: Vec<SamBlock>,
    neck0: Conv,
    neck1: LayerNorm,
    neck2: Conv,
    neck3: LayerNorm,
    net_2: Conv,
    net_3: Conv,
    pos_tables: TableCache,
}

impl Sam {
    fn load(store: &WeightStore, cfg: &UnlimitedOcrConfig) -> Result<Self, VisionError> {
        let sam = cfg.vision_config.sam_vit_b();
        let patch_embed = Conv::load(store, "sam_model.patch_embed.proj", true, 16, 0)?;
        let pos_embed = dense(store, "sam_model.pos_embed", None)?;
        let ps = pos_embed.shape()?;
        if ps.len() != 4 || ps[1] != ps[2] || ps[3] != sam.width {
            return Err(VisionError::Layout(format!("sam pos_embed shape {ps:?}")));
        }
        let (pos_host, _) = host_f32(store, "sam_model.pos_embed")?;
        let heads = sam.heads;
        let mut blocks = Vec::with_capacity(sam.layers);
        for i in 0..sam.layers {
            let window = if sam.global_attn_indexes.contains(&i) {
                0
            } else {
                14
            };
            blocks.push(SamBlock::load(store, i, window, heads)?);
        }
        let s = Self {
            patch_embed,
            pos_embed,
            pos_host,
            pos_grid: ps[1],
            embed_dim: ps[3],
            blocks,
            neck0: Conv::load(store, "sam_model.neck.0", false, 1, 0)?,
            neck1: LayerNorm::load(store, "sam_model.neck.1", 1e-6)?,
            neck2: Conv::load(store, "sam_model.neck.2", false, 1, 1)?,
            neck3: LayerNorm::load(store, "sam_model.neck.3", 1e-6)?,
            net_2: Conv::load(store, "sam_model.net_2", false, 2, 1)?,
            net_3: Conv::load(store, "sam_model.net_3", false, 2, 1)?,
            pos_tables: TableCache::new(),
        };
        // Warm the 1024-view tables (window q = 14, global q = 64).
        for blk in &s.blocks {
            blk.rel_tables(if blk.window > 0 {
                blk.window
            } else {
                s.pos_grid
            })?;
        }
        Ok(s)
    }

    /// `get_abs_pos_sam(pos_embed, grid)`.
    fn pos_for(&self, grid: usize) -> Result<MlxArray, VisionError> {
        if grid == self.pos_grid {
            return Ok(self.pos_embed.try_clone()?);
        }
        let c = self.embed_dim;
        let mut t = self.pos_tables.get_or_build(grid, || {
            let data = sam_pos_embed_interp(&self.pos_host, self.pos_grid, grid, c);
            Ok(vec![bf16_array(&[1, grid, grid, c], &data)?])
        })?;
        Ok(t.pop().expect("one table"))
    }

    fn forward(
        &self,
        x: &MlxArray,
        stages: &mut Option<&mut Stages>,
    ) -> Result<MlxArray, VisionError> {
        let mut x = self.patch_embed.forward(x)?;
        push_stage(stages, StageTag::SamPatchEmbed, &x)?;
        let grid = x.shape()?[1];
        x = x.add(&self.pos_for(grid)?)?;
        push_stage(stages, StageTag::SamPosAdded, &x)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            x = blk.forward(&x)?;
            push_stage(stages, StageTag::SamBlock(i), &x)?;
        }
        x = self.neck0.forward(&x)?;
        x = self.neck1.forward(&x)?;
        x = self.neck2.forward(&x)?;
        x = self.neck3.forward(&x)?;
        x = self.net_2.forward(&x)?;
        x = self.net_3.forward(&x)?;
        push_stage(stages, StageTag::SamFinal, &x)?;
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// CLIP
// ---------------------------------------------------------------------------

struct ClipLayer {
    ln1: LayerNorm,
    ln2: LayerNorm,
    qkv: Linear,
    out: Linear,
    fc1: Linear,
    fc2: Linear,
}

struct Clip {
    class_embedding: MlxArray,
    position_embedding: MlxArray,
    pos_host: Vec<f32>,
    pre_ln: LayerNorm,
    layers: Vec<ClipLayer>,
    hidden: usize,
    heads: usize,
    pos_tables: TableCache,
}

impl Clip {
    fn load(store: &WeightStore) -> Result<Self, VisionError> {
        let p = "vision_model";
        let class_embedding = dense(store, &format!("{p}.embeddings.class_embedding"), None)?;
        let hidden = class_embedding.shape()?[0];
        let position_embedding = dense(
            store,
            &format!("{p}.embeddings.position_embedding.weight"),
            None,
        )?;
        let (pos_host, pshape) =
            host_f32(store, &format!("{p}.embeddings.position_embedding.weight"))?;
        if pshape.len() != 2 || pshape[1] != hidden {
            return Err(VisionError::Layout(format!(
                "clip position_embedding {pshape:?}"
            )));
        }
        let rows = pshape[0];
        let position_embedding = position_embedding.reshape(&[1, rows, hidden])?;
        let mut layers = Vec::new();
        let mut i = 0;
        while store
            .tensor_info(&format!("{p}.transformer.layers.{i}.layer_norm1.weight"))
            .is_some()
        {
            let l = format!("{p}.transformer.layers.{i}");
            layers.push(ClipLayer {
                ln1: LayerNorm::load(store, &format!("{l}.layer_norm1"), 1e-6)?,
                ln2: LayerNorm::load(store, &format!("{l}.layer_norm2"), 1e-6)?,
                qkv: Linear::load(store, &format!("{l}.self_attn.qkv_proj"), true)?,
                out: Linear::load(store, &format!("{l}.self_attn.out_proj"), true)?,
                fc1: Linear::load(store, &format!("{l}.mlp.fc1"), true)?,
                fc2: Linear::load(store, &format!("{l}.mlp.fc2"), true)?,
            });
            i += 1;
        }
        if layers.len() != 24 {
            return Err(VisionError::Layout(format!(
                "expected 24 CLIP layers, found {}",
                layers.len()
            )));
        }
        Ok(Self {
            class_embedding,
            position_embedding,
            pos_host,
            pre_ln: LayerNorm::load(store, &format!("{p}.pre_layrnorm"), 1e-5)?,
            layers,
            hidden,
            heads: 16,
            pos_tables: TableCache::new(),
        })
    }

    /// `VisionEmbeddings._get_abs_pos` for a `tokens = 1 + side^2` sequence.
    fn pos_for(&self, tokens: usize) -> Result<MlxArray, VisionError> {
        let rows = self.pos_host.len() / self.hidden;
        if tokens.isqrt() == (rows - 1).isqrt() {
            return Ok(self.position_embedding.try_clone()?);
        }
        let d = self.hidden;
        let mut t = self.pos_tables.get_or_build(tokens, || {
            let data = clip_pos_embed_interp(&self.pos_host, d, tokens);
            let n = data.len() / d;
            Ok(vec![bf16_array(&[1, n, d], &data)?])
        })?;
        Ok(t.pop().expect("one table"))
    }

    /// `VisionModel.__call__(x, patch_embeds=sam_out)`.
    fn forward(
        &self,
        sam_out: &MlxArray,
        stages: &mut Option<&mut Stages>,
    ) -> Result<MlxArray, VisionError> {
        let s = sam_out.shape()?;
        let (b, l, d) = (s[0], s[1] * s[2], s[3]);
        if d != self.hidden {
            return Err(VisionError::Layout(format!(
                "SAM out channels {d} != CLIP hidden {}",
                self.hidden
            )));
        }
        let patch = sam_out.reshape(&[b, l, d])?;
        let cls = self
            .class_embedding
            .broadcast_to(&[b, 1, d])?
            .astype(MlxDtype::Bf16)?;
        let emb = MlxArray::concatenate(&[&cls, &patch], 1)?;
        let pos = self.pos_for(l + 1)?.astype(MlxDtype::Bf16)?;
        let mut x = self.pre_ln.forward(&emb.add(&pos)?)?;
        push_stage(stages, StageTag::ClipPreNorm, &x)?;
        let heads = self.heads;
        let hd = d / heads;
        let scale = (hd as f32).powf(-0.5);
        let n = l + 1;
        for (i, layer) in self.layers.iter().enumerate() {
            let y = layer.ln1.forward(&x)?;
            let qkv = layer.qkv.forward(&y)?;
            let split = |j: usize| -> Result<MlxArray, MlxError> {
                qkv.slice(&[0, 0, j * d], &[b, n, (j + 1) * d], &[1, 1, 1])?
                    .reshape(&[b, n, heads, hd])?
                    .transpose(&[0, 2, 1, 3])
            };
            let (q, k, v) = (split(0)?, split(1)?, split(2)?);
            let o = q
                .scaled_dot_product_attention(&k, &v, scale, SdpaMask::None)?
                .transpose(&[0, 2, 1, 3])?
                .reshape(&[b, n, d])?;
            x = x.add(&layer.out.forward(&o)?)?;
            let m = layer
                .fc2
                .forward(&layer.fc1.forward(&layer.ln2.forward(&x)?)?.gelu()?)?;
            x = x.add(&m)?;
            push_stage(stages, StageTag::ClipLayer(i), &x)?;
        }
        push_stage(stages, StageTag::ClipFinal, &x)?;
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Public encoder
// ---------------------------------------------------------------------------

/// Unlimited-OCR vision path on MLX: SAM + CLIP towers, the 4-bit
/// projector, and gundam token shaping, producing the `[n_image_tokens,
/// 1280]` bf16 rows that replace the `<image>` prompt embeddings.
pub struct OcrVision {
    sam: Sam,
    clip: Clip,
    projector: QWeights,
    image_newline: MlxArray,
    view_separator: MlxArray,
    feature_dim: usize,
    base_px: usize,
    crop_px: usize,
}

impl std::fmt::Debug for OcrVision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OcrVision")
            .field("feature_dim", &self.feature_dim)
            .field("base_px", &self.base_px)
            .field("crop_px", &self.crop_px)
            .finish_non_exhaustive()
    }
}

impl OcrVision {
    /// Loads from an HF snapshot directory (`config.json` + safetensors).
    pub fn load(checkpoint_dir: &Path) -> Result<Self, VisionError> {
        let store = load_checkpoint(checkpoint_dir)?;
        let raw = std::fs::read(checkpoint_dir.join("config.json"))
            .map_err(|e| VisionError::Io(format!("reading config.json: {e}")))?;
        let cfg = parse_config(&raw).map_err(|e| VisionError::Config(e.to_string()))?;
        Self::from_store(&store, &cfg)
    }

    /// Builds from an already-open [`WeightStore`] + parsed config.
    pub fn from_store(store: &WeightStore, cfg: &UnlimitedOcrConfig) -> Result<Self, VisionError> {
        if cfg.tile_tag != "2D" {
            return Err(VisionError::Config(format!(
                "only tile_tag == \"2D\" is supported, got {:?}",
                cfg.tile_tag
            )));
        }
        let q = cfg
            .quantization
            .as_ref()
            .ok_or_else(|| VisionError::Config("checkpoint is not quantized".into()))?;
        let sam = Sam::load(store, cfg)?;
        let clip = Clip::load(store)?;
        let projector = QWeights::load(store, "projector.layers", q.group_size, q.bits)?;
        let feature_dim = cfg.projector_config.n_embed;
        let sam_ch = *cfg
            .vision_config
            .sam_vit_b()
            .downsample_channels
            .last()
            .ok_or_else(|| VisionError::Config("empty downsample_channels".into()))?;
        if projector.shape != [feature_dim, cfg.projector_config.input_dim]
            || cfg.projector_config.input_dim != sam_ch + clip.hidden
        {
            return Err(VisionError::Layout(format!(
                "projector {:?} vs input_dim {} (SAM {sam_ch} + CLIP {}) -> {feature_dim}",
                projector.shape, cfg.projector_config.input_dim, clip.hidden
            )));
        }
        let base_px = sam.pos_grid * 16;
        if base_px != GUNDAM_BASE_PX {
            return Err(VisionError::Layout(format!(
                "SAM base view {base_px} != {GUNDAM_BASE_PX}"
            )));
        }
        Ok(Self {
            sam,
            clip,
            projector,
            image_newline: dense(store, "image_newline", Some(&[feature_dim]))?,
            view_separator: dense(store, "view_separator", Some(&[feature_dim]))?,
            feature_dim,
            base_px,
            crop_px: GUNDAM_CROP_PX,
        })
    }

    /// Projector output width (decoder hidden size).
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Decode PNG/JPEG bytes, gundam-preprocess, and run the encoder.
    pub fn image_features(&self, image_bytes: &[u8]) -> Result<MlxArray, VisionError> {
        let decoded =
            image::load_from_memory(image_bytes).map_err(|e| VisionError::Image(e.to_string()))?;
        let rgb = decoded.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let prep = preprocess_gundam(rgb.as_raw(), w, h)?;
        self.features_from_preprocessed(&prep)
    }

    /// Towers + projector + token shaping over a preprocessed gundam input:
    /// lazy `[n_tokens, feature_dim]` bf16 features.
    pub fn features_from_preprocessed(
        &self,
        prep: &GundamPreprocess,
    ) -> Result<MlxArray, VisionError> {
        self.features_stages(prep, &mut None)
    }

    /// One tower pass over a `[B, px, px, 3]` bf16 NHWC batch -> projector
    /// rows `[B, L, feature_dim]`.
    pub(crate) fn tower(
        &self,
        pixels: &MlxArray,
        stages: &mut Option<&mut Stages>,
        tag: StageTag,
    ) -> Result<MlxArray, VisionError> {
        let sam_out = self.sam.forward(pixels, stages)?;
        let clip_out = self.clip.forward(&sam_out, stages)?;
        let s = clip_out.shape()?;
        let (b, n, d) = (s[0], s[1], s[2]);
        let clip_tokens = clip_out.slice(&[0, 1, 0], &[b, n, d], &[1, 1, 1])?;
        let ss = sam_out.shape()?;
        let sam_flat = sam_out.reshape(&[ss[0], ss[1] * ss[2], ss[3]])?;
        let feats = MlxArray::concatenate(&[&clip_tokens, &sam_flat], 2)?;
        let y = self.projector.linear(&feats)?;
        push_stage(stages, tag, &y)?;
        Ok(y)
    }

    fn pixels(&self, images: &[&[u8]], px: usize) -> Result<MlxArray, VisionError> {
        let mut data: Vec<bf16> = Vec::with_capacity(images.len() * px * px * 3);
        for img in images {
            if img.len() != px * px * 3 {
                return Err(VisionError::Layout(format!(
                    "view has {} bytes, expected {px}x{px}x3",
                    img.len()
                )));
            }
            data.extend(pixels_nhwc_bf16(img));
        }
        Ok(MlxArray::from_bf16(&[images.len(), px, px, 3], &data)?)
    }

    /// Full forward with optional stage collection.
    pub(crate) fn features_stages(
        &self,
        prep: &GundamPreprocess,
        stages: &mut Option<&mut Stages>,
    ) -> Result<MlxArray, VisionError> {
        let dim = self.feature_dim;
        let mut parts: Vec<MlxArray> = Vec::with_capacity(3);
        if !prep.crops.is_empty() {
            let (wc, hc) = prep.crop_ratio;
            if wc * hc != prep.crops.len() {
                return Err(VisionError::Layout(format!(
                    "crop_ratio {wc}x{hc} vs {} crops",
                    prep.crops.len()
                )));
            }
            let views: Vec<&[u8]> = prep.crops.iter().map(|c| c.as_slice()).collect();
            let px = self.pixels(&views, self.crop_px)?;
            let local = self.tower(&px, stages, StageTag::ProjectorLocal)?;
            let l = local.shape()?[1];
            let side = l.isqrt();
            if side * side != l {
                return Err(VisionError::Layout(format!(
                    "crop token count {l} not square"
                )));
            }
            let grid = local
                .reshape(&[hc, wc, side, side, dim])?
                .transpose(&[0, 2, 1, 3, 4])?
                .reshape(&[hc * side, wc * side, dim])?;
            let nl =
                self.image_newline
                    .reshape(&[1, 1, dim])?
                    .broadcast_to(&[hc * side, 1, dim])?;
            parts.push(
                MlxArray::concatenate(&[&grid, &nl], 1)?
                    .reshape(&[hc * side * (wc * side + 1), dim])?,
            );
        }
        let px = self.pixels(&[prep.global.as_slice()], self.base_px)?;
        let global = self.tower(&px, stages, StageTag::ProjectorGlobal)?;
        let l = global.shape()?[1];
        let side = l.isqrt();
        if side * side != l {
            return Err(VisionError::Layout(format!(
                "global token count {l} not square"
            )));
        }
        let grid = global.reshape(&[side, side, dim])?;
        let nl = self
            .image_newline
            .reshape(&[1, 1, dim])?
            .broadcast_to(&[side, 1, dim])?;
        parts.push(MlxArray::concatenate(&[&grid, &nl], 1)?.reshape(&[side * (side + 1), dim])?);
        parts.push(self.view_separator.reshape(&[1, dim])?);
        let refs: Vec<&MlxArray> = parts.iter().collect();
        Ok(MlxArray::concatenate(&refs, 0)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::unlimited_ocr::test_support::{
        Parity, compare_rows, mlx_rows_f32, ocr_vision_golden as golden, shared_pipeline,
    };

    /// Tolerances (see the module docs of `test_support`). Against the
    /// mlx 0.32.2 reference every vision stage is bit-exact on brew mlx
    /// 0.32.4; the bounds leave headroom for kernel-order drift across MLX
    /// releases while still catching wiring bugs (O(1) rel RMS). SAM
    /// stages and CLIP through layer 11:
    const STAGE: Parity = Parity {
        rel_rms: 0.02,
        max_frac: 0.02,
    };
    /// CLIP final and projector outputs (sampled rows).
    const HEAD: Parity = Parity {
        rel_rms: 0.03,
        max_frac: 0.05,
    };
    /// Final `[n_tokens, 1280]` features.
    const FINAL: Parity = Parity {
        rel_rms: 0.03,
        max_frac: 0.08,
    };
    /// Pure-f32 host interpolation tables vs the MLX kernels (~1 ulp).
    const ATOL_INTERP: f64 = 1e-6;

    // ------------------------------------------------------------------
    // Hermetic: window partition round-trip (no checkpoint)
    // ------------------------------------------------------------------

    #[test]
    fn vision_window_partition_roundtrip() {
        let (b, h, w, c) = (2usize, 17usize, 17usize, 3usize);
        let data: Vec<f32> = (0..b * h * w * c).map(|i| i as f32).collect();
        let x = MlxArray::from_f32(&[b, h, w, c], &data).unwrap();
        let (win, hp, wp) = window_partition(&x, 14).unwrap();
        assert_eq!((hp, wp), (28, 28));
        assert_eq!(win.shape().unwrap(), vec![b * 4, 14, 14, c]);
        // Window index = b * 4 + row_tile * 2 + col_tile. Window (b=1,
        // row-tile 1, col-tile 0) = 6, pixel (2, 3) == x[1, 16, 3].
        let wv = win.to_f32_vec().unwrap();
        let widx = (6 * 14 * 14 + 2 * 14 + 3) * c;
        assert_eq!(wv[widx], data[((h + 16) * w + 3) * c]);
        // Padded region is zero: window (b=0, 1, 1) = 3, pixel (13, 13).
        let pad_idx = (3 * 14 * 14 + 13 * 14 + 13) * c;
        assert_eq!(wv[pad_idx], 0.0);
        let back = window_unpartition(&win, 14, (hp, wp), (h, w)).unwrap();
        assert_eq!(back.to_f32_vec().unwrap(), data);
    }

    // ------------------------------------------------------------------
    // Golden: interpolation tables + tower stages + final features
    // ------------------------------------------------------------------

    #[test]
    fn vision_interp_tables_golden() {
        let Some(p) = shared_pipeline() else {
            return;
        };
        let vision = p.vision();
        let m = golden::manifest();
        // SAM pos embed 64 -> 40 bicubic (antialias).
        let got = sam_pos_embed_interp(&vision.sam.pos_host, 64, 40, 768);
        let e = golden::dump_entry(&m, "aux.sam_pos_embed_interp_40");
        let rows = golden::dump_rows(e);
        let want = golden::read_f32(e["file"].as_str().unwrap());
        let mut max = 0f64;
        for (ri, &r) in rows.iter().enumerate() {
            for c in 0..768 {
                max = max.max((got[r * 768 + c] - want[ri * 768 + c]).abs() as f64);
            }
        }
        eprintln!("vision_interp: pos embed 64->40 max abs {max:.3e}");
        assert!(max <= ATOL_INTERP, "pos embed interp: {max:.3e}");
        // rel_pos 127 -> 79 linear interp + gather (block 2 is global).
        let blk = &vision.sam.blocks[2];
        let got = get_rel_pos_table(&blk.rel_pos_h, blk.rel_rows, 64, 40);
        let want = golden::read_f32("aux_rel_pos_h_interp_40.f32");
        assert_eq!(got.len(), want.len());
        let max = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0f64, f64::max);
        eprintln!("vision_interp: rel pos 127->79 max abs {max:.3e}");
        assert!(max <= ATOL_INTERP, "rel pos interp: {max:.3e}");
    }

    fn stage_bound(stage: &str) -> Parity {
        if stage.ends_with("clip.final") || stage.ends_with("projector") {
            HEAD
        } else {
            STAGE
        }
    }

    fn stage_tag(stage: &str) -> StageTag {
        let last = stage.rsplit('.').next().unwrap();
        match last {
            "patch_embed" => StageTag::SamPatchEmbed,
            "pos_added" => StageTag::SamPosAdded,
            "pre_layernorm" => StageTag::ClipPreNorm,
            "projector" if stage.contains("global") => StageTag::ProjectorGlobal,
            "projector" => StageTag::ProjectorLocal,
            "final" if stage.contains(".sam.") => StageTag::SamFinal,
            "final" => StageTag::ClipFinal,
            s if s.starts_with("block") => StageTag::SamBlock(s["block".len()..].parse().unwrap()),
            s if s.starts_with("layer") => StageTag::ClipLayer(s["layer".len()..].parse().unwrap()),
            other => panic!("unknown stage {other}"),
        }
    }

    /// Every dumped tower stage of the crop image (global 1024 pass: SAM
    /// stages + projector; crop 640 batch: SAM + CLIP stages + projector)
    /// against the mlx_vlm Metal reference, one full encoder run.
    #[test]
    fn vision_tower_stages_golden() {
        let Some(p) = shared_pipeline() else {
            return;
        };
        let vision = p.vision();
        let m = golden::manifest();
        let (rgb, w, h) = golden::source_image("source_crop_1000x500.png");
        let prep = preprocess_gundam(&rgb, w, h).unwrap();
        let mut stages: Stages = Vec::new();
        // Crops first, then global: tag the passes by order.
        let crop_px = vision
            .pixels(
                &prep.crops.iter().map(|c| c.as_slice()).collect::<Vec<_>>(),
                640,
            )
            .unwrap();
        let mut sink = Some(&mut stages);
        vision
            .tower(&crop_px, &mut sink, StageTag::ProjectorLocal)
            .unwrap();
        let n_crop_stages = stages.len();
        let global_px = vision.pixels(&[prep.global.as_slice()], 1024).unwrap();
        let mut sink = Some(&mut stages);
        vision
            .tower(&global_px, &mut sink, StageTag::ProjectorGlobal)
            .unwrap();
        let (crop_stages, global_stages) = stages.split_at(n_crop_stages);

        let mut fails = Vec::new();
        let mut checked = 0;
        for e in m["dumps"].as_array().unwrap() {
            let stage = e["stage"].as_str().unwrap();
            let pass = if stage.starts_with("crop_global.") || stage == "crop.global.projector" {
                global_stages
            } else if stage.starts_with("crop_crops.") || stage == "crop.crops.projector" {
                crop_stages
            } else {
                continue;
            };
            let tag = stage_tag(stage);
            let arr = &pass
                .iter()
                .find(|(t, _)| *t == tag)
                .unwrap_or_else(|| panic!("{stage} not captured"))
                .1;
            let (got, cols) = mlx_rows_f32(arr);
            let src = &e["source_shape"];
            if !src.is_null() {
                assert_eq!(
                    (got.len() / cols, cols),
                    (
                        src[0].as_u64().unwrap() as usize,
                        src[1].as_u64().unwrap() as usize
                    ),
                    "{stage} shape"
                );
            }
            compare_rows(
                stage,
                &got,
                cols,
                &golden::dump_rows(e),
                &golden::read_f32(e["file"].as_str().unwrap()),
                stage_bound(stage),
                &mut fails,
            );
            checked += 1;
        }
        assert!(checked >= 22, "only {checked} stage dumps compared");
        assert!(
            fails.is_empty(),
            "stage parity failures:\n{}",
            fails.join("\n")
        );
    }

    /// Final features of both golden images (all dumped rows).
    #[test]
    fn vision_final_features_golden() {
        let Some(p) = shared_pipeline() else {
            return;
        };
        let vision = p.vision();
        let m = golden::manifest();
        let mut fails = Vec::new();
        for (tag, src) in [
            ("small", "source_small_500x400.png"),
            ("crop", "source_crop_1000x500.png"),
        ] {
            let (rgb, w, h) = golden::source_image(src);
            let prep = preprocess_gundam(&rgb, w, h).unwrap();
            let feats = vision.features_from_preprocessed(&prep).unwrap();
            let (got, cols) = mlx_rows_f32(&feats);
            let n_tokens = got.len() / cols;
            assert_eq!(
                n_tokens,
                m["images"][tag]["n_tokens"].as_u64().unwrap() as usize,
                "{tag} tokens"
            );
            let e = golden::dump_entry(&m, &format!("{tag}.final_features"));
            let want = golden::read_bf16_f32(e["file"].as_str().unwrap());
            let rows: Vec<usize> = match e["rows"].as_array() {
                Some(r) => r.iter().map(|v| v.as_u64().unwrap() as usize).collect(),
                None => (0..n_tokens).collect(),
            };
            compare_rows(
                &format!("{tag}.final_features"),
                &got,
                cols,
                &rows,
                &want,
                FINAL,
                &mut fails,
            );
        }
        assert!(
            fails.is_empty(),
            "final feature parity failures:\n{}",
            fails.join("\n")
        );
    }
}
