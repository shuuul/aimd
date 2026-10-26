//! PP-OCRv6 medium CTC recognizer: PPLCNetV4 backbone + AvgPool(3, 2) ->
//! `EncoderWithLightSVTR` -> CTC linear head + softmax, op-for-op with
//! mlx4ocr `models/rec/{backbone,encoder,head,model}.py`.
//!
//! The HF safetensors export stores the encoder's two 1x1 convs swapped:
//! `head.encoder.conv_block.1` is the reference `conv_reduce` and
//! `conv_block.0` is `skip_conv` (mlx4ocr repairs this by patching the
//! tensors from the official Paddle checkpoint; swapping the two blocks
//! reproduces that patch exactly, without the pickle download).

use aimd_mlx::{MlxArray, MlxError, SdpaMask};
use aimd_weights::WeightStore;

use super::config::RecConfig;
use super::layers::{
    Conv, ConvKind, LcNetBlock, Linear, LoadError, Stem, host_f32, load_stages, tensor,
};

const BN: ConvKind<'static> = ConvKind::Bn {
    conv: "convolution",
    norm: "normalization",
    bias: false,
};

/// LightSVTR attention heads.
const NUM_HEADS: usize = 8;

struct LayerNorm {
    w: MlxArray,
    b: MlxArray,
    eps: f32,
}

impl LayerNorm {
    fn load(store: &WeightStore, prefix: &str, dim: usize, eps: f32) -> Result<Self, LoadError> {
        Ok(Self {
            w: tensor(store, &format!("{prefix}.weight"), &[dim])?,
            b: tensor(store, &format!("{prefix}.bias"), &[dim])?,
            eps,
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        x.layer_norm(Some(&self.w), Some(&self.b), self.eps)
    }
}

/// `SVTRBlock`: `x + attn(norm1(x))`, then `x + mlp(norm2(x))`, with
/// `nn.MultiHeadAttention` built from the fused `qkv` projection.
struct SvtrBlock {
    norm1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    dim: usize,
}

impl SvtrBlock {
    fn load(
        store: &WeightStore,
        prefix: &str,
        dim: usize,
        hidden: usize,
    ) -> Result<Self, LoadError> {
        let (w, ws) = host_f32(store, &format!("{prefix}.self_attn.qkv.weight"))?;
        let (b, bs) = host_f32(store, &format!("{prefix}.self_attn.qkv.bias"))?;
        if ws != [3 * dim, dim] || bs != [3 * dim] {
            return Err(LoadError::Layout(format!(
                "{prefix}.self_attn.qkv: shapes {ws:?} / {bs:?}"
            )));
        }
        let part = |i: usize| {
            Linear::from_parts(
                &w[i * dim * dim..(i + 1) * dim * dim],
                &b[i * dim..(i + 1) * dim],
                dim,
                dim,
            )
        };
        Ok(Self {
            norm1: LayerNorm::load(store, &format!("{prefix}.layer_norm1"), dim, 1e-5)?,
            q: part(0)?,
            k: part(1)?,
            v: part(2)?,
            out: Linear::load(store, &format!("{prefix}.self_attn.projection"), dim, dim)?,
            norm2: LayerNorm::load(store, &format!("{prefix}.layer_norm2"), dim, 1e-5)?,
            fc1: Linear::load(store, &format!("{prefix}.mlp.fc1"), hidden, dim)?,
            fc2: Linear::load(store, &format!("{prefix}.mlp.fc2"), dim, hidden)?,
            dim,
        })
    }

    fn attention(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let s = x.shape()?;
        let (b, l) = (s[0], s[1]);
        let hd = self.dim / NUM_HEADS;
        let heads = |p: &Linear| {
            p.forward(x)?
                .reshape(&[b, l, NUM_HEADS, hd])?
                .transpose(&[0, 2, 1, 3])
        };
        let (q, k, v) = (heads(&self.q)?, heads(&self.k)?, heads(&self.v)?);
        let scale = (1.0f64 / hd as f64).sqrt() as f32;
        let o = q.scaled_dot_product_attention(&k, &v, scale, SdpaMask::None)?;
        self.out
            .forward(&o.transpose(&[0, 2, 1, 3])?.reshape(&[b, l, self.dim])?)
    }

    fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let x = x.add(&self.attention(&self.norm1.forward(x)?)?)?;
        let h = self
            .fc2
            .forward(&self.fc1.forward(&self.norm2.forward(&x)?)?.silu()?)?;
        x.add(&h)
    }
}

/// `EncoderWithLightSVTR`.
struct LightSvtr {
    conv_reduce: Conv,
    skip_conv: Conv,
    local_conv: Conv,
    blocks: Vec<SvtrBlock>,
    norm: LayerNorm,
}

impl LightSvtr {
    fn load(store: &WeightStore, config: &RecConfig) -> Result<Self, LoadError> {
        let dim = config.hidden_size;
        let hidden = (dim as f64 * config.mlp_ratio) as usize;
        let k = config.local_kernel;
        let p = "head.encoder";
        Ok(Self {
            // swapped in the HF export (see the module docs)
            conv_reduce: Conv::load(store, &format!("{p}.conv_block.1"), BN, [1, 1], [0, 0], 1)?,
            skip_conv: Conv::load(store, &format!("{p}.conv_block.0"), BN, [1, 1], [0, 0], 1)?,
            local_conv: Conv::load(
                store,
                &format!("{p}.conv_block.2"),
                BN,
                [1, 1],
                [0, k / 2],
                dim,
            )?,
            blocks: (0..config.depth)
                .map(|i| SvtrBlock::load(store, &format!("{p}.svtr_block.{i}"), dim, hidden))
                .collect::<Result<_, _>>()?,
            norm: LayerNorm::load(store, &format!("{p}.norm"), dim, 1e-6)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let skip = self.skip_conv.forward(x)?.silu()?;
        let z = self.conv_reduce.forward(x)?.silu()?;
        let z = z.add(&self.local_conv.forward(&z)?.silu()?)?;
        let s = z.shape()?;
        let (b, h, w, c) = (s[0], s[1], s[2], s[3]);
        let mut z = z.reshape(&[b, h * w, c])?;
        for block in &self.blocks {
            z = block.forward(&z)?;
        }
        self.norm.forward(&z)?.reshape(&[b, h, w, c])?.add(&skip)
    }
}

/// The recognition network.
pub struct RecModel {
    stem: Stem,
    stages: Vec<Vec<LcNetBlock>>,
    encoder: LightSvtr,
    head: Linear,
}

impl RecModel {
    /// Loads (and BN-fuses) the medium recognizer from its checkpoint.
    pub fn load(store: &WeightStore, config: &RecConfig) -> Result<Self, LoadError> {
        Ok(Self {
            stem: Stem::load(store, "model.backbone.encoder.convolution")?,
            stages: load_stages(store, "model.backbone.encoder", &config.stages)?,
            encoder: LightSvtr::load(store, config)?,
            head: Linear::load(store, "head.head", config.num_classes, config.hidden_size)?,
        })
    }

    /// `AvgPool2d((3, 2), stride (3, 2))` over NHWC (floor mode).
    fn avg_pool(x: &MlxArray) -> Result<MlxArray, MlxError> {
        let s = x.shape()?;
        let (b, h, w, c) = (s[0], s[1], s[2], s[3]);
        let (ho, wo) = (h / 3, w / 2);
        let x = if ho * 3 != h || wo * 2 != w {
            x.slice(&[0, 0, 0, 0], &[b, ho * 3, wo * 2, c], &[1, 1, 1, 1])?
        } else {
            x.try_clone()?
        };
        x.reshape(&[b, ho, 3, wo, 2, c])?
            .transpose(&[0, 1, 3, 2, 4, 5])?
            .mean_axes(&[-3, -2], false)
    }

    /// Softmax probabilities `[B, T, num_classes]` for a normalized
    /// `[B, 48, W, 3]` batch.
    pub fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let mut x = self.stem.forward(x)?;
        for stage in &self.stages {
            for block in stage {
                x = block.forward(&x)?;
            }
        }
        let x = Self::avg_pool(&x)?;
        let x = self.encoder.forward(&x)?;
        let s = x.shape()?;
        if s[1] != 1 {
            return Err(MlxError::Op(format!(
                "expected pooled height 1 before Im2Seq, got {}",
                s[1]
            )));
        }
        let seq = x.reshape(&[s[0], s[2], s[3]])?;
        self.head.forward(&seq)?.softmax(-1)
    }
}
