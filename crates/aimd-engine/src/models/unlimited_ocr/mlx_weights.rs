//! Checkpoint -> MLX array import helpers shared by [`super::mlx_vision`]
//! and [`super::mlx_decoder`] (the Unlimited-OCR MLX graph).
//!
//! Every tensor is uploaded verbatim from the safetensors mmap
//! ([`MlxArray::from_bytes`] is a bit-exact byte copy): dense weights keep
//! their checkpoint dtype (bf16), and quantized modules keep the packed
//! `u32` payload plus bf16 scales/biases, fed directly to
//! [`MlxArray::quantized_matmul`] / [`MlxArray::gather_qmm`] exactly like
//! `nn.QuantizedLinear` / `QuantizedSwitchLinear` do — no eager
//! dequantization.

use aimd_mlx::{MlxArray, MlxDtype, MlxError};
use aimd_weights::{WeightStore, WeightsError};

/// Weight-import failures.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error(transparent)]
    Mlx(#[from] MlxError),
    #[error(transparent)]
    Weights(#[from] WeightsError),
    #[error("checkpoint layout: {0}")]
    Layout(String),
}

/// `(dtype, shape)` of a checkpoint tensor.
fn info(store: &WeightStore, name: &str) -> Result<(MlxDtype, Vec<usize>), LoadError> {
    let info = store
        .tensor_info(name)
        .ok_or_else(|| LoadError::Layout(format!("missing checkpoint tensor {name}")))?;
    let dtype = match format!("{:?}", info.dtype).as_str() {
        "F16" => MlxDtype::F16,
        "BF16" => MlxDtype::Bf16,
        "F32" => MlxDtype::F32,
        "U32" => MlxDtype::U32,
        other => {
            return Err(LoadError::Layout(format!(
                "{name}: unsupported dtype {other}"
            )));
        }
    };
    Ok((dtype, info.shape.clone()))
}

/// Whether the checkpoint holds `name`.
pub(crate) fn has(store: &WeightStore, name: &str) -> bool {
    store.tensor_info(name).is_some()
}

/// Uploads one dense tensor verbatim, checking its shape when `expected` is
/// given.
pub(crate) fn dense(
    store: &WeightStore,
    name: &str,
    expected: Option<&[usize]>,
) -> Result<MlxArray, LoadError> {
    let (dtype, shape) = info(store, name)?;
    if let Some(want) = expected
        && shape.as_slice() != want
    {
        return Err(LoadError::Layout(format!(
            "{name}: shape {shape:?}, expected {want:?}"
        )));
    }
    Ok(MlxArray::from_bytes(
        store.tensor_bytes(name)?,
        dtype,
        &shape,
    )?)
}

/// Host f32 copy of a bf16/f16/f32 tensor (for the CPU-side interpolation
/// tables) plus its shape.
pub(crate) fn host_f32(
    store: &WeightStore,
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>), LoadError> {
    let (dtype, shape) = info(store, name)?;
    let bytes = store.tensor_bytes(name)?;
    let data = match dtype {
        MlxDtype::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| half::bf16::from_bits(u16::from_le_bytes(*p)).to_f32())
            .collect(),
        MlxDtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| half::f16::from_bits(u16::from_le_bytes(*p)).to_f32())
            .collect(),
        MlxDtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| f32::from_le_bytes(*p))
            .collect(),
        _ => {
            return Err(LoadError::Layout(format!(
                "{name}: expected a float tensor"
            )));
        }
    };
    Ok((data, shape))
}

/// `nn.Linear`: `y = addmm(bias, x, W.T)` (or `x @ W.T` without bias). The
/// transpose is a lazy strided view, as in Python.
pub(crate) struct Linear {
    wt: MlxArray,
    bias: Option<MlxArray>,
}

impl Linear {
    pub(crate) fn load(store: &WeightStore, module: &str, bias: bool) -> Result<Self, LoadError> {
        let w = dense(store, &format!("{module}.weight"), None)?;
        if w.ndim()? != 2 {
            return Err(LoadError::Layout(format!("{module}.weight must be rank-2")));
        }
        let bias = if bias {
            Some(dense(store, &format!("{module}.bias"), None)?)
        } else {
            None
        };
        Ok(Self {
            wt: w.transpose(&[1, 0])?,
            bias,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        match &self.bias {
            Some(b) => b.addmm(x, &self.wt, 1.0, 1.0),
            None => x.matmul(&self.wt, None),
        }
    }
}

/// An affine-quantized module (`nn.QuantizedLinear` /
/// `nn.QuantizedEmbedding` / `QuantizedSwitchLinear`): packed `u32` payload
/// `[..., out, in*bits/32]`, scales/biases `[..., out, in/group_size]`, and
/// an optional float `bias`.
pub(crate) struct QWeights {
    pub(crate) w: MlxArray,
    pub(crate) scales: MlxArray,
    pub(crate) biases: MlxArray,
    pub(crate) bias: Option<MlxArray>,
    pub(crate) group_size: i32,
    pub(crate) bits: i32,
    /// Logical (dequantized) `[..., out, in]` shape.
    pub(crate) shape: Vec<usize>,
}

impl QWeights {
    pub(crate) fn load(
        store: &WeightStore,
        module: &str,
        group_size: u32,
        bits: u32,
    ) -> Result<Self, LoadError> {
        let (wdt, wshape) = info(store, &format!("{module}.weight"))?;
        if wdt != MlxDtype::U32 {
            return Err(LoadError::Layout(format!(
                "{module}.weight: payload must be U32"
            )));
        }
        let (sdt, sshape) = info(store, &format!("{module}.scales"))?;
        let (bdt, bshape) = info(store, &format!("{module}.biases"))?;
        if sdt != bdt || bshape != sshape || !matches!(sdt, MlxDtype::Bf16 | MlxDtype::F16) {
            return Err(LoadError::Layout(format!(
                "{module}: scales/biases must share an f16/bf16 dtype and shape"
            )));
        }
        let rank = wshape.len();
        if rank < 2 || sshape.len() != rank || wshape[..rank - 1] != sshape[..rank - 1] {
            return Err(LoadError::Layout(format!(
                "{module}: payload {wshape:?} vs scales {sshape:?}"
            )));
        }
        let cols = wshape[rank - 1] * 32 / bits as usize;
        if sshape[rank - 1] * group_size as usize != cols {
            return Err(LoadError::Layout(format!(
                "{module}: {} groups x {group_size} != {cols} input columns",
                sshape[rank - 1]
            )));
        }
        let w = MlxArray::from_bytes(
            store.tensor_bytes(&format!("{module}.weight"))?,
            MlxDtype::U32,
            &wshape,
        )?;
        let scales = MlxArray::from_bytes(
            store.tensor_bytes(&format!("{module}.scales"))?,
            sdt,
            &sshape,
        )?;
        let biases = MlxArray::from_bytes(
            store.tensor_bytes(&format!("{module}.biases"))?,
            bdt,
            &bshape,
        )?;
        let bias_name = format!("{module}.bias");
        let bias = if has(store, &bias_name) {
            Some(dense(store, &bias_name, None)?)
        } else {
            None
        };
        let mut shape = wshape.clone();
        shape[rank - 1] = cols;
        Ok(Self {
            w,
            scales,
            biases,
            bias,
            group_size: group_size as i32,
            bits: bits as i32,
            shape,
        })
    }

    /// `nn.QuantizedLinear.__call__`: `quantized_matmul(x, w, transpose)`
    /// then `+ bias`.
    pub(crate) fn linear(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let y = x.quantized_matmul(
            &self.w,
            &self.scales,
            Some(&self.biases),
            true,
            Some(self.group_size),
            Some(self.bits),
            None,
        )?;
        match &self.bias {
            Some(b) => y.add(b),
            None => Ok(y),
        }
    }

    /// `QuantizedSwitchLinear.__call__`: `gather_qmm(x, w, rhs_indices=idx,
    /// transpose=True, sorted_indices=sorted)`.
    pub(crate) fn gather(
        &self,
        x: &MlxArray,
        rhs_indices: &MlxArray,
        sorted: bool,
    ) -> Result<MlxArray, MlxError> {
        x.gather_qmm(
            &self.w,
            &self.scales,
            Some(&self.biases),
            None,
            Some(rhs_indices),
            true,
            Some(self.group_size),
            Some(self.bits),
            sorted,
        )
    }

    /// `nn.QuantizedEmbedding.__call__`: gather payload/scales/biases rows,
    /// then dequantize. `ids` is any-shape `u32`; returns `[*ids, dim]`.
    pub(crate) fn embed(&self, ids: &MlxArray) -> Result<MlxArray, MlxError> {
        let w = self.w.take(ids, 0)?;
        let s = self.scales.take(ids, 0)?;
        let b = self.biases.take(ids, 0)?;
        w.dequantize(&s, Some(&b), Some(self.group_size), Some(self.bits))
    }
}
