//! Shared PP-DocLayoutV3 building blocks on top of the PP-OCRv6 weight
//! import (`Conv` with folded batch norms, `Linear`): MLP heads, layer
//! norms, multi-head self-attention with position embeddings on q/k, and
//! the NHWC nearest / bilinear 2x upsamplers.

use aimd_mlx::{Activation, MlxArray, MlxError, SdpaMask};
use aimd_weights::WeightStore;

use super::config::{D_MODEL, EPS, NUM_HEADS};
use crate::models::pp_ocrv6::layers::{Conv, ConvKind, Linear, LoadError, tensor};

/// `HGNetV2ConvLayer` / `PPDocLayoutV3ConvLayer` parameter names.
pub(crate) const HG_BN: ConvKind<'static> = ConvKind::Bn {
    conv: "convolution",
    norm: "normalization",
    bias: false,
};
/// `PPDocLayoutV3ConvNormLayer` parameter names.
pub(crate) const CN_BN: ConvKind<'static> = ConvKind::Bn {
    conv: "conv",
    norm: "norm",
    bias: false,
};
/// `nn.Sequential(Conv2d, BatchNorm2d)` input projections.
pub(crate) const SEQ_BN: ConvKind<'static> = ConvKind::Bn {
    conv: "0",
    norm: "1",
    bias: false,
};

/// A `k x k` conv with "same" padding `k / 2`.
pub(crate) fn conv(
    store: &WeightStore,
    prefix: &str,
    kind: ConvKind<'_>,
    k: usize,
    stride: usize,
    groups: usize,
) -> Result<Conv, LoadError> {
    Conv::load(
        store,
        prefix,
        kind,
        [stride, stride],
        [k / 2, k / 2],
        groups,
    )
}

/// `conv -> SiLU`.
pub(crate) fn conv_silu(c: &Conv, x: &MlxArray) -> Result<MlxArray, MlxError> {
    c.forward(x)?.silu()
}

pub(crate) fn scalar(v: f32) -> Result<MlxArray, MlxError> {
    MlxArray::scalar_f32(v)
}

/// `nn.LayerNorm(256)`.
pub(crate) struct LayerNorm {
    w: MlxArray,
    b: MlxArray,
}

impl LayerNorm {
    pub(crate) fn load(store: &WeightStore, prefix: &str) -> Result<Self, LoadError> {
        Ok(Self {
            w: tensor(store, &format!("{prefix}.weight"), &[D_MODEL])?,
            b: tensor(store, &format!("{prefix}.bias"), &[D_MODEL])?,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        x.layer_norm(Some(&self.w), Some(&self.b), EPS)
    }
}

/// `PPDocLayoutV3MLPPredictionHead`: Linear layers with ReLU between.
pub(crate) struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    /// `dims` = `[in, hidden.., out]`.
    pub(crate) fn load(
        store: &WeightStore,
        prefix: &str,
        dims: &[usize],
    ) -> Result<Self, LoadError> {
        let layers = dims
            .windows(2)
            .enumerate()
            .map(|(i, d)| Linear::load(store, &format!("{prefix}.layers.{i}"), d[1], d[0]))
            .collect::<Result<_, _>>()?;
        Ok(Self { layers })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let mut x = self.layers[0].forward(x)?;
        for layer in &self.layers[1..] {
            x = layer.forward(&x.relu()?)?;
        }
        Ok(x)
    }
}

/// `PPDocLayoutV3SelfAttention` (8 heads; `pos` added to the q/k input
/// only), checkpoint names `{q,k,v,out}_proj`.
pub(crate) struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

impl SelfAttention {
    pub(crate) fn load(store: &WeightStore, prefix: &str) -> Result<Self, LoadError> {
        let lin = |n: &str| Linear::load(store, &format!("{prefix}.{n}"), D_MODEL, D_MODEL);
        Ok(Self {
            q: lin("q_proj")?,
            k: lin("k_proj")?,
            v: lin("v_proj")?,
            o: lin("out_proj")?,
        })
    }

    /// `x`, `pos`: `[L, 256]` -> `[L, 256]`.
    pub(crate) fn forward(&self, x: &MlxArray, pos: &MlxArray) -> Result<MlxArray, MlxError> {
        let l = x.shape()?[0];
        let hd = D_MODEL / NUM_HEADS;
        let heads = |t: MlxArray| t.reshape(&[1, l, NUM_HEADS, hd])?.transpose(&[0, 2, 1, 3]);
        let qk_in = x.add(pos)?;
        let q = heads(self.q.forward(&qk_in)?)?;
        let k = heads(self.k.forward(&qk_in)?)?;
        let v = heads(self.v.forward(x)?)?;
        let scale = (hd as f32).powf(-0.5);
        let a = q.scaled_dot_product_attention(&k, &v, scale, SdpaMask::None)?;
        let a = a.transpose(&[0, 2, 1, 3])?.reshape(&[l, D_MODEL])?;
        self.o.forward(&a)
    }
}

/// `fc2(act(fc1(x)))`.
pub(crate) struct Ffn {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
}

impl Ffn {
    pub(crate) fn load(
        store: &WeightStore,
        prefix: &str,
        hidden: usize,
        act: Activation,
    ) -> Result<Self, LoadError> {
        Ok(Self {
            fc1: Linear::load(store, &format!("{prefix}.fc1"), hidden, D_MODEL)?,
            fc2: Linear::load(store, &format!("{prefix}.fc2"), D_MODEL, hidden)?,
            act,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let h = self.fc1.forward(x)?;
        let h = match self.act {
            Activation::Gelu => h.gelu()?,
            Activation::Relu => h.relu()?,
            _ => h,
        };
        self.fc2.forward(&h)
    }
}

/// `F.interpolate(scale_factor=2, mode="nearest")` over NHWC.
pub(crate) fn upsample_nearest2x(x: &MlxArray) -> Result<MlxArray, MlxError> {
    let d = x.shape()?;
    let (n, h, w, c) = (d[0], d[1], d[2], d[3]);
    x.reshape(&[n, h, 1, w, 1, c])?
        .broadcast_to(&[n, h, 2, w, 2, c])?
        .reshape(&[n, 2 * h, 2 * w, c])
}

/// Bilinear 2x along `axis` (1 = H, 2 = W) of an NHWC array with
/// `align_corners=False`: even outputs `0.25 * x[k-1] + 0.75 * x[k]`, odd
/// outputs `0.75 * x[k] + 0.25 * x[k+1]`, indices clamped at the borders
/// (torch's `upsample_bilinear2d` source-index rule for an exact 2x).
fn bilinear2x_axis(x: &MlxArray, axis: usize) -> Result<MlxArray, MlxError> {
    let d = x.shape()?;
    let n = d[axis];
    let part = |a: usize, b: usize| {
        let mut start = vec![0; 4];
        let mut stop = d.clone();
        start[axis] = a;
        stop[axis] = b;
        x.slice(&start, &stop, &[1, 1, 1, 1])
    };
    let prev = MlxArray::concatenate(&[&part(0, 1)?, &part(0, n - 1)?], axis)?;
    let next = MlxArray::concatenate(&[&part(1, n)?, &part(n - 1, n)?], axis)?;
    let (q, tq) = (scalar(0.25)?, scalar(0.75)?);
    let even = prev.multiply(&q)?.add(&x.multiply(&tq)?)?;
    let odd = x.multiply(&tq)?.add(&next.multiply(&q)?)?;
    // Interleave along `axis`: stack on a new axis right after it.
    let mut expanded = d.clone();
    expanded.insert(axis + 1, 1);
    let even = even.reshape(&expanded)?;
    let odd = odd.reshape(&expanded)?;
    let mut out = d.clone();
    out[axis] = 2 * n;
    MlxArray::concatenate(&[&even, &odd], axis + 1)?.reshape(&out)
}

/// `F.interpolate(scale_factor=2, mode="bilinear", align_corners=False)`
/// over NHWC (W pass, then H pass, as torch's separable kernel weighs).
pub(crate) fn upsample_bilinear2x(x: &MlxArray) -> Result<MlxArray, MlxError> {
    bilinear2x_axis(&bilinear2x_axis(x, 2)?, 1)
}
