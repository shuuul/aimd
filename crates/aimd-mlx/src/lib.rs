//! MLX C++ shim: the compute backend (Metal, MLX CPU device fallback) (WS-11a).
//!
//! # Contract (WS-11a)
//!
//! A thin C++ shim (`shim/shim.cpp`, compiled by `build.rs` with `cc`)
//! exposes a C ABI over `mlx::core`; this crate wraps it in safe RAII types.
//! The link target is brew `mlx` 0.32.4 (bottled; same 0.32 line as the
//! installed Python wheel) discovered via `brew --prefix mlx`, overridable
//! with `MLX_PREFIX`.
//!
//! Implemented surface (minimal — the ASR decoder graph is the consumer):
//! - array lifecycle: [`MlxArray::from_bytes`] (+ typed `from_f16`/`from_bf16`/
//!   `from_f32`/`from_u32`/`from_u8` constructors), bit-exact readback via
//!   `to_f16_bytes`/`to_bf16_bytes`/`to_f32_bytes`/`to_u32_bytes` (+ typed
//!   `to_*_vec` accessors), [`MlxArray::eval`], RAII [`MlxArray::free` on drop].
//! - ops: [`MlxArray::matmul`], [`MlxArray::quantized_matmul`] (w/scales/
//!   biases with a transpose flag and group_size/bits — bf16 scales/biases
//!   are accepted natively: mlx promotes mixed activation/scale dtypes),
//!   `transpose` (axes or reverse), `reshape`, elementwise
//!   add/multiply/divide/sqrt/power, [`MlxArray::softmax`], [`MlxArray::silu`]
//!   (composed `x * sigmoid(x)` — mlx 0.32.4 has no C++ `silu`),
//!   [`MlxArray::rms_norm`] (native `mlx::core::fast::rms_norm`).
//! - WS-11b additions (the full-MLX ASR graph): [`MlxArray::conv2d`],
//!   [`MlxArray::scaled_dot_product_attention`] (none/causal/array mask, GQA
//!   handled by mlx), [`MlxArray::layer_norm`] (native fused biased LN),
//!   [`MlxArray::gelu`] (exact-erf, composed in `nn.gelu`'s order),
//!   [`MlxArray::slice`], [`MlxArray::concatenate`], [`MlxArray::take`]
//!   (embedding gathers), [`MlxArray::dequantize`] (quantized-embedding
//!   rows), [`MlxArray::rope`] (native `mx.fast.rope` behind `nn.RoPE`),
//!   [`MlxArray::argmax`] (GPU greedy sampling), [`MlxArray::scalar_f32`].
//! - a last-error string slot: every failed call records the C++ exception
//!   message; the wrapper surfaces it as [`MlxError::Op`].
//!
//! Parity expectations: MLX runs fp16/bf16 on Metal (or the MLX CPU device,
//! see [`device_is_cpu`]) — results match an f32-accumulate reference within
//! fp16 tolerance, NOT bit-exactly. Parity tests reuse the committed golden
//! quant fixtures (`aimd-weights/tests/golden/quant/`): shim `quantized_matmul` vs
//! `aimd-weights::dequantize_affine` + a naive f32-accumulate matmul.
//!
//! # Device selection
//!
//! The compute device is picked once per process: Metal by default, the MLX
//! CPU device when `AIMD_MLX_DEVICE=cpu` or when no GPU is available
//! (virtualized CI runners). `AIMD_MLX_DEVICE=gpu` forces the GPU.
//!
//! Import/export of f16/bf16/u32 data is bit-exact: construction uses mlx's
//! raw-pointer array constructor (a byte copy, never a numeric conversion)
//! and readback memcpies the raw buffer.
//!
//! # Threading
//!
//! [`MlxArray`] is `Send`/`Sync` (decided in WS-11b: mlx arrays are
//! immutable refcounted COW C++ values, and the shim pins ONE process-global
//! `new_thread_unsafe_stream` as every calling thread's default — the only
//! stream mlx allows across threads; see `shim/shim.cpp` — so graphs built
//! and evaluated from any thread run on the same serialized stream). The
//! last-error slot is process-global; read it immediately after a failing
//! call.

use std::ffi::CStr;

mod dash;

pub use dash::{DashDecodeCtx, DashLayerConsts, DashMode, DashParams, DashPromptState, DashRoute};

/// Errors surfaced by the MLX backend.
#[derive(Debug, thiserror::Error)]
pub enum MlxError {
    #[error("mlx unavailable: {0}")]
    Unavailable(String),
    #[error("mlx op failed: {0}")]
    Op(String),
}

/// Element dtypes exposed by the shim.
///
/// Codes match the C ABI in `shim/shim.cpp` (0 = f16, 1 = bf16, 2 = f32,
/// 3 = u32, 4 = u8, 5 = i32, 6 = bool).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlxDtype {
    /// IEEE half precision (compute dtype).
    F16,
    /// bfloat16 (checkpoint scale/bias and MLX-runtime activation dtype).
    Bf16,
    /// Single precision.
    F32,
    /// Unsigned 32-bit (quantized payload words).
    U32,
    /// Unsigned 8-bit.
    U8,
    /// Signed 32-bit (positions, page coordinates).
    I32,
    /// Boolean (one byte per element).
    Bool,
}

impl MlxDtype {
    /// Wire size in bytes per element.
    pub fn size_bytes(self) -> usize {
        match self {
            MlxDtype::F16 | MlxDtype::Bf16 => 2,
            MlxDtype::F32 | MlxDtype::U32 | MlxDtype::I32 => 4,
            MlxDtype::U8 | MlxDtype::Bool => 1,
        }
    }

    fn code(self) -> i32 {
        match self {
            MlxDtype::F16 => 0,
            MlxDtype::Bf16 => 1,
            MlxDtype::F32 => 2,
            MlxDtype::U32 => 3,
            MlxDtype::U8 => 4,
            MlxDtype::I32 => 5,
            MlxDtype::Bool => 6,
        }
    }

    fn from_code(code: i32) -> Result<Self, MlxError> {
        match code {
            0 => Ok(MlxDtype::F16),
            1 => Ok(MlxDtype::Bf16),
            2 => Ok(MlxDtype::F32),
            3 => Ok(MlxDtype::U32),
            4 => Ok(MlxDtype::U8),
            5 => Ok(MlxDtype::I32),
            6 => Ok(MlxDtype::Bool),
            other => Err(MlxError::Op(format!(
                "mlx returned unknown dtype code {other}"
            ))),
        }
    }
}

/// Activation for [`MlxArray::bias_act`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Identity = 0,
    /// `maximum(x, 0)`.
    Relu = 1,
    /// `clip(maximum(x + 3, 0), 0, 6) / 6`.
    HardSigmoid = 2,
    /// Exact-erf GELU ([`MlxArray::gelu`]).
    Gelu = 3,
}

/// Attention mask mode for [`MlxArray::scaled_dot_product_attention`].
#[derive(Clone, Copy, Debug)]
pub enum SdpaMask<'a> {
    /// No mask (full attention — single-query decode steps).
    None,
    /// mlx `"causal"` mode: query `i` attends keys `0..=i` (aligned to the
    /// query block's end — correct when the block is the sequence suffix).
    Causal,
    /// Additive mask array broadcast against `[batch, heads, q_len, kv_len]`
    /// (the audio tower's block-diagonal window mask).
    Array(&'a MlxArray),
}

/// C-ABI declarations for `shim/shim.cpp` (compiled and linked by `build.rs`).
mod ffi {
    use std::ffi::c_char;

    /// Opaque handle around a heap-allocated `mlx::core::array`.
    #[repr(C)]
    pub struct AimdMlxArray {
        _private: [u8; 0],
    }

    unsafe extern "C" {
        pub fn mlx_last_error() -> *const c_char;
        pub fn mlx_array_new(
            bytes: *const u8,
            dtype: i32,
            ndim: i32,
            shape: *const i64,
        ) -> *mut AimdMlxArray;
        pub fn mlx_array_free(array_handle: *mut AimdMlxArray);
        pub fn mlx_array_eval(array_handle: *mut AimdMlxArray) -> i32;
        pub fn mlx_array_dtype(array_handle: *mut AimdMlxArray) -> i32;
        pub fn mlx_array_ndim(array_handle: *mut AimdMlxArray) -> i64;
        pub fn mlx_array_dim(array_handle: *mut AimdMlxArray, dim: i32) -> i64;
        pub fn mlx_array_size(array_handle: *mut AimdMlxArray) -> i64;
        pub fn mlx_array_to_f16_bytes(
            array_handle: *mut AimdMlxArray,
            out: *mut u8,
            cap: i64,
        ) -> i32;
        pub fn mlx_array_to_bf16_bytes(
            array_handle: *mut AimdMlxArray,
            out: *mut u8,
            cap: i64,
        ) -> i32;
        pub fn mlx_array_to_f32_bytes(
            array_handle: *mut AimdMlxArray,
            out: *mut u8,
            cap: i64,
        ) -> i32;
        pub fn mlx_array_to_u32_bytes(
            array_handle: *mut AimdMlxArray,
            out: *mut u8,
            cap: i64,
        ) -> i32;
        pub fn mlx_array_astype(array_handle: *mut AimdMlxArray, dtype: i32) -> *mut AimdMlxArray;
        pub fn mlx_matmul(
            a: *mut AimdMlxArray,
            b: *mut AimdMlxArray,
            out_dtype: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_quantized_matmul(
            x: *mut AimdMlxArray,
            w: *mut AimdMlxArray,
            scales: *mut AimdMlxArray,
            biases: *mut AimdMlxArray,
            transpose_w: u8,
            group_size: i32,
            bits: i32,
            out_dtype: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_transpose(
            array_handle: *mut AimdMlxArray,
            axes: *const i32,
            naxes: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_reshape(
            array_handle: *mut AimdMlxArray,
            shape: *const i64,
            ndim: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_add(a: *mut AimdMlxArray, b: *mut AimdMlxArray) -> *mut AimdMlxArray;
        pub fn mlx_multiply(a: *mut AimdMlxArray, b: *mut AimdMlxArray) -> *mut AimdMlxArray;
        pub fn mlx_divide(a: *mut AimdMlxArray, b: *mut AimdMlxArray) -> *mut AimdMlxArray;
        pub fn mlx_sqrt(a: *mut AimdMlxArray) -> *mut AimdMlxArray;
        pub fn mlx_power(a: *mut AimdMlxArray, b: *mut AimdMlxArray) -> *mut AimdMlxArray;
        pub fn mlx_softmax(array_handle: *mut AimdMlxArray, axis: i32) -> *mut AimdMlxArray;
        pub fn mlx_silu(array_handle: *mut AimdMlxArray) -> *mut AimdMlxArray;
        pub fn mlx_swiglu(gate: *mut AimdMlxArray, up: *mut AimdMlxArray) -> *mut AimdMlxArray;
        pub fn mlx_rms_norm(
            x: *mut AimdMlxArray,
            weight: *mut AimdMlxArray,
            eps: f32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_conv2d(
            input: *mut AimdMlxArray,
            weight: *mut AimdMlxArray,
            stride_h: i32,
            stride_w: i32,
            pad_h: i32,
            pad_w: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_sdpa(
            q: *mut AimdMlxArray,
            k: *mut AimdMlxArray,
            v: *mut AimdMlxArray,
            scale: f32,
            mask: *mut AimdMlxArray,
            mask_mode: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_layer_norm(
            x: *mut AimdMlxArray,
            weight: *mut AimdMlxArray,
            bias: *mut AimdMlxArray,
            eps: f32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_gelu(array_handle: *mut AimdMlxArray) -> *mut AimdMlxArray;
        pub fn mlx_slice(
            array_handle: *mut AimdMlxArray,
            start: *const i64,
            stop: *const i64,
            strides: *const i64,
            ndim: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_concatenate(
            arrays: *mut *mut AimdMlxArray,
            n: i32,
            axis: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_take(
            a: *mut AimdMlxArray,
            indices: *mut AimdMlxArray,
            axis: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_argmax(a: *mut AimdMlxArray, axis: i32, keepdims: i32) -> *mut AimdMlxArray;
        pub fn mlx_dequantize(
            w: *mut AimdMlxArray,
            scales: *mut AimdMlxArray,
            biases: *mut AimdMlxArray,
            group_size: i32,
            bits: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_rope(
            x: *mut AimdMlxArray,
            dims: i32,
            base: f64,
            scale: f32,
            offset: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_addmm(
            c: *mut AimdMlxArray,
            a: *mut AimdMlxArray,
            b: *mut AimdMlxArray,
            alpha: f32,
            beta: f32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_subtract(a: *mut AimdMlxArray, b: *mut AimdMlxArray) -> *mut AimdMlxArray;
        pub fn mlx_floor_divide(a: *mut AimdMlxArray, b: *mut AimdMlxArray) -> *mut AimdMlxArray;
        pub fn mlx_softmax_precise(
            a: *mut AimdMlxArray,
            axis: i32,
            precise: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_logsumexp(a: *mut AimdMlxArray, axis: i32, keepdims: i32) -> *mut AimdMlxArray;
        pub fn mlx_sum_axis(a: *mut AimdMlxArray, axis: i32, keepdims: i32) -> *mut AimdMlxArray;
        pub fn mlx_argsort(a: *mut AimdMlxArray, axis: i32) -> *mut AimdMlxArray;
        pub fn mlx_argpartition(a: *mut AimdMlxArray, kth: i32, axis: i32) -> *mut AimdMlxArray;
        pub fn mlx_take_along_axis(
            a: *mut AimdMlxArray,
            indices: *mut AimdMlxArray,
            axis: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_put_along_axis(
            a: *mut AimdMlxArray,
            indices: *mut AimdMlxArray,
            values: *mut AimdMlxArray,
            axis: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_broadcast_to(
            a: *mut AimdMlxArray,
            shape: *const i64,
            ndim: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_pad(
            a: *mut AimdMlxArray,
            low: *const i64,
            high: *const i64,
            ndim: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_slice_update(
            src: *mut AimdMlxArray,
            update: *mut AimdMlxArray,
            start: *const i64,
            stop: *const i64,
            strides: *const i64,
            ndim: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_zeros(shape: *const i64, ndim: i32, dtype: i32) -> *mut AimdMlxArray;
        pub fn mlx_gather_qmm(
            x: *mut AimdMlxArray,
            w: *mut AimdMlxArray,
            scales: *mut AimdMlxArray,
            biases: *mut AimdMlxArray,
            lhs_indices: *mut AimdMlxArray,
            rhs_indices: *mut AimdMlxArray,
            transpose_w: u8,
            group_size: i32,
            bits: i32,
            sorted_indices: u8,
        ) -> *mut AimdMlxArray;
        pub fn mlx_moe_swiglu_4bit(
            x: *mut AimdMlxArray,
            inds: *mut AimdMlxArray,
            scores: *mut AimdMlxArray,
            gate_w: *mut AimdMlxArray,
            gate_s: *mut AimdMlxArray,
            gate_b: *mut AimdMlxArray,
            up_w: *mut AimdMlxArray,
            up_s: *mut AimdMlxArray,
            up_b: *mut AimdMlxArray,
            down_w: *mut AimdMlxArray,
            down_s: *mut AimdMlxArray,
            down_b: *mut AimdMlxArray,
            group_size: i32,
            addends: *mut *mut AimdMlxArray,
            n_add: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_moe_router(
            x: *mut AimdMlxArray,
            gate_w: *mut AimdMlxArray,
            k: i32,
            out_inds: *mut *mut AimdMlxArray,
            out_scores: *mut *mut AimdMlxArray,
        ) -> i32;
        pub fn mlx_sdpa_decode(
            q: *mut AimdMlxArray,
            k: *mut AimdMlxArray,
            v: *mut AimdMlxArray,
            scale: f32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_async_eval(arrays: *mut *mut AimdMlxArray, n: i32) -> i32;
        pub fn mlx_eval_many(arrays: *mut *mut AimdMlxArray, n: i32) -> i32;
        pub fn mlx_device_is_cpu() -> i32;
        pub fn mlx_metal_kernels_available() -> i32;
        pub fn mlx_clear_cache() -> i32;
        pub fn mlx_memory_stat(which: i32) -> i64;
        pub fn mlx_array_clone(a: *mut AimdMlxArray) -> *mut AimdMlxArray;
        pub fn mlx_conv2d_ex(
            input: *mut AimdMlxArray,
            weight: *mut AimdMlxArray,
            stride_h: i32,
            stride_w: i32,
            pad_h: i32,
            pad_w: i32,
            dil_h: i32,
            dil_w: i32,
            groups: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_conv_transpose2d(
            input: *mut AimdMlxArray,
            weight: *mut AimdMlxArray,
            stride_h: i32,
            stride_w: i32,
            pad_h: i32,
            pad_w: i32,
            groups: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_unary(a: *mut AimdMlxArray, op: i32) -> *mut AimdMlxArray;
        pub fn mlx_binary(a: *mut AimdMlxArray, b: *mut AimdMlxArray, op: i32)
        -> *mut AimdMlxArray;
        pub fn mlx_reduce(
            a: *mut AimdMlxArray,
            axes: *const i32,
            n_axes: i32,
            keepdims: i32,
            op: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_where(
            cond: *mut AimdMlxArray,
            x: *mut AimdMlxArray,
            y: *mut AimdMlxArray,
        ) -> *mut AimdMlxArray;
        pub fn mlx_clip(a: *mut AimdMlxArray, lo: f32, hi: f32) -> *mut AimdMlxArray;
        pub fn mlx_bias_act(
            y: *mut AimdMlxArray,
            bias: *mut AimdMlxArray,
            residual: *mut AimdMlxArray,
            act: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_quantize(
            w: *mut AimdMlxArray,
            group_size: i32,
            bits: i32,
            out_w: *mut *mut AimdMlxArray,
            out_scales: *mut *mut AimdMlxArray,
            out_biases: *mut *mut AimdMlxArray,
        ) -> i32;
    }
}

/// Message for the most recent failed shim call, or an empty string.
fn last_error() -> String {
    // SAFETY: `mlx_last_error` returns a valid NUL-terminated pointer into a
    // static slot (or NULL before the first failure).
    let ptr = unsafe { ffi::mlx_last_error() };
    if ptr.is_null() {
        String::new()
    } else {
        // SAFETY: same as above; the pointer is NUL-terminated.
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

fn op_failed(op: &str) -> MlxError {
    let message = last_error();
    if message.is_empty() {
        MlxError::Op(format!("{op} failed"))
    } else {
        MlxError::Op(format!("{op}: {message}"))
    }
}

/// RAII handle around an mlx array. Freed on drop.
///
/// # Threading
///
/// `MlxArray` is `Send`/`Sync` (the WS-11b stream-affinity decision): the
/// underlying `mlx::core::array` is an immutable, refcounted COW C++ value
/// — the shim never mutates one in place, all compute is enqueued on mlx's
/// default stream (serialized by mlx), and array destruction from any
/// thread is reference-counted. Explicit `eval`/readback discipline (WS-11a)
/// keeps results deterministic regardless of which thread drives the graph.
/// The last-error slot is process-global and mutex-protected; read it
/// immediately after a failing call.
pub struct MlxArray {
    ptr: std::ptr::NonNull<ffi::AimdMlxArray>,
}

// SAFETY: see the type's threading docs — immutable COW arrays, shim-serialized
// default-stream compute, refcounted destruction; no thread-local C++ state is
// reachable through the handle.
unsafe impl Send for MlxArray {}
unsafe impl Sync for MlxArray {}

// SAFETY notes for every FFI call below: `self.ptr` is a valid handle created
// by `mlx_array_new` and not yet freed (RAII invariant); the shim never
// panics across the boundary (all C++ exceptions are caught) and reports
// failure via NULL / -1 plus the last-error slot.

impl Drop for MlxArray {
    fn drop(&mut self) {
        // SAFETY: valid, not-yet-freed handle per the RAII invariant.
        unsafe { ffi::mlx_array_free(self.ptr.as_ptr()) };
    }
}

impl std::fmt::Debug for MlxArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let describe = || Ok::<(MlxDtype, Vec<usize>), MlxError>((self.dtype()?, self.shape()?));
        match describe() {
            Ok((dtype, shape)) => f
                .debug_struct("MlxArray")
                .field("dtype", &dtype)
                .field("shape", &shape)
                .finish(),
            Err(_) => f.write_str("MlxArray(<metadata unavailable>)"),
        }
    }
}

impl MlxArray {
    fn from_handle(ptr: *mut ffi::AimdMlxArray, op: &str) -> Result<Self, MlxError> {
        match std::ptr::NonNull::new(ptr) {
            Some(ptr) => Ok(Self { ptr }),
            None => Err(op_failed(op)),
        }
    }

    /// Creates an array from raw little-endian bytes (row-major), which must
    /// be exactly `dtype.size_bytes() * prod(shape)` long. The bytes are
    /// copied into mlx-managed storage bit-exactly.
    pub fn from_bytes(bytes: &[u8], dtype: MlxDtype, shape: &[usize]) -> Result<Self, MlxError> {
        let expected = shape
            .iter()
            .try_fold::<usize, _, _>(dtype.size_bytes(), |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| MlxError::Op("array shape overflows usize".to_string()))?;
        if bytes.len() != expected {
            return Err(MlxError::Op(format!(
                "byte length {} does not match shape {shape:?} x {} (expected {expected})",
                bytes.len(),
                dtype.size_bytes()
            )));
        }
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let ptr = unsafe {
            ffi::mlx_array_new(
                bytes.as_ptr(),
                dtype.code(),
                shape_i64.len() as i32,
                shape_i64.as_ptr(),
            )
        };
        Self::from_handle(ptr, "mlx_array_new")
    }

    /// Packs `f16` elements into an mlx array.
    pub fn from_f16(shape: &[usize], data: &[half::f16]) -> Result<Self, MlxError> {
        let bytes = pack(data, half::f16::to_le_bytes);
        Self::from_bytes(&bytes, MlxDtype::F16, shape)
    }

    /// Packs `bf16` elements into an mlx array.
    pub fn from_bf16(shape: &[usize], data: &[half::bf16]) -> Result<Self, MlxError> {
        let bytes = pack(data, half::bf16::to_le_bytes);
        Self::from_bytes(&bytes, MlxDtype::Bf16, shape)
    }

    /// Packs `f32` elements into an mlx array.
    pub fn from_f32(shape: &[usize], data: &[f32]) -> Result<Self, MlxError> {
        let bytes = pack(data, f32::to_le_bytes);
        Self::from_bytes(&bytes, MlxDtype::F32, shape)
    }

    /// Packs `u32` elements (e.g. quantized payload words) into an mlx array.
    pub fn from_u32(shape: &[usize], data: &[u32]) -> Result<Self, MlxError> {
        let bytes = pack(data, u32::to_le_bytes);
        Self::from_bytes(&bytes, MlxDtype::U32, shape)
    }

    /// Copies `u8` elements into an mlx array.
    pub fn from_u8(shape: &[usize], data: &[u8]) -> Result<Self, MlxError> {
        Self::from_bytes(data, MlxDtype::U8, shape)
    }

    /// Packs `i32` elements into an mlx array.
    pub fn from_i32(shape: &[usize], data: &[i32]) -> Result<Self, MlxError> {
        let bytes = pack(data, i32::to_le_bytes);
        Self::from_bytes(&bytes, MlxDtype::I32, shape)
    }

    /// Packs `bool` elements into an mlx array.
    pub fn from_bool(shape: &[usize], data: &[bool]) -> Result<Self, MlxError> {
        let bytes: Vec<u8> = data.iter().map(|&b| u8::from(b)).collect();
        Self::from_bytes(&bytes, MlxDtype::Bool, shape)
    }

    /// Builds a quantized weight triple the way checkpoints store it: the
    /// packed `u32` payload plus `scales`/`biases` as 16-bit patterns of the
    /// given dtype, all row-major.
    pub fn quantized_weight(
        payload: &[u8],
        packed_shape: &[usize],
        scales: &[u16],
        biases: &[u16],
        scale_shape: &[usize],
        scale_dtype: MlxDtype,
    ) -> Result<(Self, Self, Self), MlxError> {
        let w = Self::from_bytes(payload, MlxDtype::U32, packed_shape)?;
        match scale_dtype {
            MlxDtype::Bf16 => {
                let scales =
                    Self::from_bf16(scale_shape, &unpack_bits(scales, half::bf16::from_bits))?;
                let biases =
                    Self::from_bf16(scale_shape, &unpack_bits(biases, half::bf16::from_bits))?;
                Ok((w, scales, biases))
            }
            MlxDtype::F16 => {
                let scales =
                    Self::from_f16(scale_shape, &unpack_bits(scales, half::f16::from_bits))?;
                let biases =
                    Self::from_f16(scale_shape, &unpack_bits(biases, half::f16::from_bits))?;
                Ok((w, scales, biases))
            }
            other => Err(MlxError::Op(format!(
                "unsupported scale dtype {other:?}; checkpoints use f16 or bf16"
            ))),
        }
    }

    /// Evaluates the array's computation graph.
    pub fn eval(&self) -> Result<(), MlxError> {
        let status = unsafe { ffi::mlx_array_eval(self.ptr.as_ptr()) };
        if status == 0 {
            Ok(())
        } else {
            Err(op_failed("mlx_array_eval"))
        }
    }

    /// Element dtype.
    pub fn dtype(&self) -> Result<MlxDtype, MlxError> {
        let code = unsafe { ffi::mlx_array_dtype(self.ptr.as_ptr()) };
        MlxDtype::from_code(code)
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> Result<usize, MlxError> {
        let ndim = unsafe { ffi::mlx_array_ndim(self.ptr.as_ptr()) };
        usize::try_from(ndim).map_err(|_| op_failed("mlx_array_ndim"))
    }

    /// Size of dimension `dim`.
    pub fn dim(&self, dim: usize) -> Result<usize, MlxError> {
        let size = unsafe { ffi::mlx_array_dim(self.ptr.as_ptr(), dim as i32) };
        usize::try_from(size).map_err(|_| op_failed("mlx_array_dim"))
    }

    /// Shape as `usize` dims.
    pub fn shape(&self) -> Result<Vec<usize>, MlxError> {
        let ndim = self.ndim()?;
        (0..ndim).map(|d| self.dim(d)).collect()
    }

    /// Total element count.
    pub fn size(&self) -> Result<usize, MlxError> {
        let size = unsafe { ffi::mlx_array_size(self.ptr.as_ptr()) };
        usize::try_from(size).map_err(|_| op_failed("mlx_array_size"))
    }

    fn copy_out(
        &self,
        op: &'static str,
        size_bytes: usize,
        f: unsafe extern "C" fn(*mut ffi::AimdMlxArray, *mut u8, i64) -> i32,
    ) -> Result<Vec<u8>, MlxError> {
        let mut out = vec![0u8; size_bytes];
        let status = unsafe { f(self.ptr.as_ptr(), out.as_mut_ptr(), out.len() as i64) };
        if status == 0 {
            Ok(out)
        } else {
            Err(op_failed(op))
        }
    }

    /// Eval + bit-exact `f16` byte readback (little-endian). Errors when the
    /// array is not `f16`.
    pub fn to_f16_bytes(&self) -> Result<Vec<u8>, MlxError> {
        let size = self.size()?;
        self.copy_out(
            "mlx_array_to_f16_bytes",
            size * 2,
            ffi::mlx_array_to_f16_bytes,
        )
    }

    /// Eval + bit-exact `bf16` byte readback. Errors when the array is not
    /// `bf16`.
    pub fn to_bf16_bytes(&self) -> Result<Vec<u8>, MlxError> {
        let size = self.size()?;
        self.copy_out(
            "mlx_array_to_bf16_bytes",
            size * 2,
            ffi::mlx_array_to_bf16_bytes,
        )
    }

    /// Eval + bit-exact `f32` byte readback. Errors when the array is not
    /// `f32`.
    pub fn to_f32_bytes(&self) -> Result<Vec<u8>, MlxError> {
        let size = self.size()?;
        self.copy_out(
            "mlx_array_to_f32_bytes",
            size * 4,
            ffi::mlx_array_to_f32_bytes,
        )
    }

    /// Eval + bit-exact `u32` byte readback. Errors when the array is not
    /// `u32`.
    pub fn to_u32_bytes(&self) -> Result<Vec<u8>, MlxError> {
        let size = self.size()?;
        self.copy_out(
            "mlx_array_to_u32_bytes",
            size * 4,
            ffi::mlx_array_to_u32_bytes,
        )
    }

    /// Eval + typed `f16` readback.
    pub fn to_f16_vec(&self) -> Result<Vec<half::f16>, MlxError> {
        let bytes = self.to_f16_bytes()?;
        Ok(unpack(&bytes, half::f16::from_le_bytes))
    }

    /// Eval + typed `bf16` readback.
    pub fn to_bf16_vec(&self) -> Result<Vec<half::bf16>, MlxError> {
        let bytes = self.to_bf16_bytes()?;
        Ok(unpack(&bytes, half::bf16::from_le_bytes))
    }

    /// Eval + typed `f32` readback.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, MlxError> {
        let bytes = self.to_f32_bytes()?;
        Ok(unpack(&bytes, f32::from_le_bytes))
    }

    /// Eval + typed `u32` readback.
    pub fn to_u32_vec(&self) -> Result<Vec<u32>, MlxError> {
        let bytes = self.to_u32_bytes()?;
        Ok(unpack(&bytes, u32::from_le_bytes))
    }

    /// Casts the array to `dtype` (`mlx::core::astype`).
    pub fn astype(&self, dtype: MlxDtype) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_array_astype(self.ptr.as_ptr(), dtype.code()) };
        Self::from_handle(ptr, "mlx_array_astype")
    }

    /// Matrix multiply `self @ rhs`; `out_dtype: None` keeps mlx's promoted
    /// result dtype.
    pub fn matmul(&self, rhs: &Self, out_dtype: Option<MlxDtype>) -> Result<Self, MlxError> {
        let ptr =
            unsafe { ffi::mlx_matmul(self.ptr.as_ptr(), rhs.ptr.as_ptr(), out_code(out_dtype)) };
        Self::from_handle(ptr, "mlx_matmul")
    }

    /// Quantized matmul (affine mode): `self [.., cols]` against the packed
    /// `w` payload `(rows, cols*bits/32)`; with `transpose_w` (checkpoint
    /// layout) the result is `self @ deq(w).T` of shape `[.., rows]`.
    /// `group_size`/`bits` of `None` use mlx defaults; mixed activation /
    /// scale dtypes are promoted by mlx (f16 x bf16 -> f32), so checkpoint
    /// bf16 scales/biases are accepted natively.
    #[allow(clippy::too_many_arguments)]
    pub fn quantized_matmul(
        &self,
        w: &Self,
        scales: &Self,
        biases: Option<&Self>,
        transpose_w: bool,
        group_size: Option<i32>,
        bits: Option<i32>,
        out_dtype: Option<MlxDtype>,
    ) -> Result<Self, MlxError> {
        let biases_ptr = biases.map_or(std::ptr::null_mut(), |b| b.ptr.as_ptr());
        let ptr = unsafe {
            ffi::mlx_quantized_matmul(
                self.ptr.as_ptr(),
                w.ptr.as_ptr(),
                scales.ptr.as_ptr(),
                biases_ptr,
                u8::from(transpose_w),
                group_size.unwrap_or(-1),
                bits.unwrap_or(-1),
                out_code(out_dtype),
            )
        };
        Self::from_handle(ptr, "mlx_quantized_matmul")
    }

    /// Permute dimensions; an empty `axes` slice uses mlx's reverse-order
    /// transpose.
    pub fn transpose(&self, axes: &[usize]) -> Result<Self, MlxError> {
        let axes_i32: Vec<i32> = axes.iter().map(|&a| a as i32).collect();
        let ptr = unsafe {
            ffi::mlx_transpose(self.ptr.as_ptr(), axes_i32.as_ptr(), axes_i32.len() as i32)
        };
        Self::from_handle(ptr, "mlx_transpose")
    }

    /// Reshape to `shape` (element count must match).
    pub fn reshape(&self, shape: &[usize]) -> Result<Self, MlxError> {
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let ptr = unsafe {
            ffi::mlx_reshape(
                self.ptr.as_ptr(),
                shape_i64.as_ptr(),
                shape_i64.len() as i32,
            )
        };
        Self::from_handle(ptr, "mlx_reshape")
    }

    /// Element-wise add.
    pub fn add(&self, rhs: &Self) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_add(self.ptr.as_ptr(), rhs.ptr.as_ptr()) };
        Self::from_handle(ptr, "mlx_add")
    }

    /// Element-wise multiply.
    pub fn multiply(&self, rhs: &Self) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_multiply(self.ptr.as_ptr(), rhs.ptr.as_ptr()) };
        Self::from_handle(ptr, "mlx_multiply")
    }

    /// Element-wise divide.
    pub fn divide(&self, rhs: &Self) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_divide(self.ptr.as_ptr(), rhs.ptr.as_ptr()) };
        Self::from_handle(ptr, "mlx_divide")
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_sqrt(self.ptr.as_ptr()) };
        Self::from_handle(ptr, "mlx_sqrt")
    }

    /// Element-wise power (`self ** rhs`).
    pub fn power(&self, rhs: &Self) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_power(self.ptr.as_ptr(), rhs.ptr.as_ptr()) };
        Self::from_handle(ptr, "mlx_power")
    }

    /// Softmax over `axis` (negative indices count from the end).
    pub fn softmax(&self, axis: i32) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_softmax(self.ptr.as_ptr(), axis) };
        Self::from_handle(ptr, "mlx_softmax")
    }

    /// SiLU (`x * sigmoid(x)`), composed from `sigmoid` + `multiply`.
    pub fn silu(&self) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_silu(self.ptr.as_ptr()) };
        Self::from_handle(ptr, "mlx_silu")
    }

    /// `silu(self) * up` as one compiled (fused, shapeless) kernel: mlx-lm
    /// `swiglu`.
    pub fn swiglu(&self, up: &Self) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_swiglu(self.ptr.as_ptr(), up.ptr.as_ptr()) };
        Self::from_handle(ptr, "mlx_swiglu")
    }

    /// Fused RMS norm over the last axis; `weight: None` skips the affine
    /// weight.
    pub fn rms_norm(&self, weight: Option<&Self>, eps: f32) -> Result<Self, MlxError> {
        let weight_ptr = weight.map_or(std::ptr::null_mut(), |w| w.ptr.as_ptr());
        let ptr = unsafe { ffi::mlx_rms_norm(self.ptr.as_ptr(), weight_ptr, eps) };
        Self::from_handle(ptr, "mlx_rms_norm")
    }

    /// Zero-dimensional (scalar) `f32` array; broadcasts against any shape,
    /// mirroring Python weak-scalar promotion.
    pub fn scalar_f32(value: f32) -> Result<Self, MlxError> {
        Self::from_bytes(&value.to_le_bytes(), MlxDtype::F32, &[])
    }

    /// 2-D convolution over an NHWC input (`[N, H, W, C]`) with a
    /// `[O, KH, KW, C]` weight; `stride`/`padding` per spatial axis. mlx
    /// promotes mixed input/weight dtypes (f32 x bf16 -> f32).
    pub fn conv2d(
        &self,
        weight: &Self,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Self, MlxError> {
        let ptr = unsafe {
            ffi::mlx_conv2d(
                self.ptr.as_ptr(),
                weight.ptr.as_ptr(),
                stride[0] as i32,
                stride[1] as i32,
                padding[0] as i32,
                padding[1] as i32,
            )
        };
        Self::from_handle(ptr, "mlx_conv2d")
    }

    /// `conv2d` with per-axis dilation and channel `groups` (depthwise when
    /// `groups` equals the input channels); weight `[O, KH, KW, C / groups]`.
    pub fn conv2d_ex(
        &self,
        weight: &Self,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<Self, MlxError> {
        let ptr = unsafe {
            ffi::mlx_conv2d_ex(
                self.ptr.as_ptr(),
                weight.ptr.as_ptr(),
                stride[0] as i32,
                stride[1] as i32,
                padding[0] as i32,
                padding[1] as i32,
                dilation[0] as i32,
                dilation[1] as i32,
                groups as i32,
            )
        };
        Self::from_handle(ptr, "mlx_conv2d_ex")
    }

    /// Transposed 2-D convolution over NHWC; weight `[O, KH, KW, C / groups]`
    /// (mlx `conv_transpose2d`).
    pub fn conv_transpose2d(
        &self,
        weight: &Self,
        stride: [usize; 2],
        padding: [usize; 2],
        groups: usize,
    ) -> Result<Self, MlxError> {
        let ptr = unsafe {
            ffi::mlx_conv_transpose2d(
                self.ptr.as_ptr(),
                weight.ptr.as_ptr(),
                stride[0] as i32,
                stride[1] as i32,
                padding[0] as i32,
                padding[1] as i32,
                groups as i32,
            )
        };
        Self::from_handle(ptr, "mlx_conv_transpose2d")
    }

    /// Element-wise unary op.
    pub fn unary(&self, op: UnaryOp) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_unary(self.ptr.as_ptr(), op as i32) };
        Self::from_handle(ptr, "mlx_unary")
    }

    /// `sigmoid(self)`.
    pub fn sigmoid(&self) -> Result<Self, MlxError> {
        self.unary(UnaryOp::Sigmoid)
    }

    /// `max(self, 0)`.
    pub fn relu(&self) -> Result<Self, MlxError> {
        self.unary(UnaryOp::Relu)
    }

    /// Element-wise binary op; comparisons yield `bool` arrays for [`Self::where_`].
    pub fn binary(&self, rhs: &Self, op: BinaryOp) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_binary(self.ptr.as_ptr(), rhs.ptr.as_ptr(), op as i32) };
        Self::from_handle(ptr, "mlx_binary")
    }

    /// Element-wise maximum.
    pub fn maximum(&self, rhs: &Self) -> Result<Self, MlxError> {
        self.binary(rhs, BinaryOp::Maximum)
    }

    /// Element-wise minimum.
    pub fn minimum(&self, rhs: &Self) -> Result<Self, MlxError> {
        self.binary(rhs, BinaryOp::Minimum)
    }

    /// Reduction over `axes` (negative indices count from the end).
    pub fn reduce(&self, axes: &[i32], keepdims: bool, op: ReduceOp) -> Result<Self, MlxError> {
        let ptr = unsafe {
            ffi::mlx_reduce(
                self.ptr.as_ptr(),
                axes.as_ptr(),
                axes.len() as i32,
                i32::from(keepdims),
                op as i32,
            )
        };
        Self::from_handle(ptr, "mlx_reduce")
    }

    /// Mean over `axes`.
    pub fn mean_axes(&self, axes: &[i32], keepdims: bool) -> Result<Self, MlxError> {
        self.reduce(axes, keepdims, ReduceOp::Mean)
    }

    /// Max over `axes`.
    pub fn max_axes(&self, axes: &[i32], keepdims: bool) -> Result<Self, MlxError> {
        self.reduce(axes, keepdims, ReduceOp::Max)
    }

    /// `where(self, x, y)` with `self` a boolean condition.
    pub fn where_(&self, x: &Self, y: &Self) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_where(self.ptr.as_ptr(), x.ptr.as_ptr(), y.ptr.as_ptr()) };
        Self::from_handle(ptr, "mlx_where")
    }

    /// `clip(self, lo, hi)` with scalar bounds in the input dtype.
    pub fn clip(&self, lo: f32, hi: f32) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_clip(self.ptr.as_ptr(), lo, hi) };
        Self::from_handle(ptr, "mlx_clip")
    }

    /// `act(self + bias) + residual` as one compiled elementwise kernel
    /// (bias and residual optional), matching the unfused ops value for
    /// value.
    pub fn bias_act(
        &self,
        bias: Option<&Self>,
        act: Activation,
        residual: Option<&Self>,
    ) -> Result<Self, MlxError> {
        let raw = |a: Option<&Self>| a.map_or(std::ptr::null_mut(), |a| a.ptr.as_ptr());
        let ptr =
            unsafe { ffi::mlx_bias_act(self.ptr.as_ptr(), raw(bias), raw(residual), act as i32) };
        Self::from_handle(ptr, "mlx_bias_act")
    }

    /// Affine `quantize` along the last axis -> `(packed, scales, biases)`,
    /// the `mlx.nn.quantize` layout consumed by [`Self::quantized_matmul`].
    pub fn quantize(&self, group_size: i32, bits: i32) -> Result<(Self, Self, Self), MlxError> {
        let mut w = std::ptr::null_mut();
        let mut scales = std::ptr::null_mut();
        let mut biases = std::ptr::null_mut();
        let status = unsafe {
            ffi::mlx_quantize(
                self.ptr.as_ptr(),
                group_size,
                bits,
                &raw mut w,
                &raw mut scales,
                &raw mut biases,
            )
        };
        if status != 0 {
            return Err(op_failed("mlx_quantize"));
        }
        Ok((
            Self::from_handle(w, "mlx_quantize")?,
            Self::from_handle(scales, "mlx_quantize")?,
            Self::from_handle(biases, "mlx_quantize")?,
        ))
    }

    /// Fused scaled dot product attention `softmax(Q @ K.T * scale) @ V`
    /// over head-major `[batch, heads, seq, head_dim]` operands; mlx handles
    /// GQA (query heads != key/value heads) internally.
    pub fn scaled_dot_product_attention(
        &self,
        keys: &Self,
        values: &Self,
        scale: f32,
        mask: SdpaMask<'_>,
    ) -> Result<Self, MlxError> {
        let (mask_ptr, mask_mode) = match mask {
            SdpaMask::None => (std::ptr::null_mut(), 0),
            SdpaMask::Causal => (std::ptr::null_mut(), 1),
            SdpaMask::Array(m) => (m.ptr.as_ptr(), 2),
        };
        let ptr = unsafe {
            ffi::mlx_sdpa(
                self.ptr.as_ptr(),
                keys.ptr.as_ptr(),
                values.ptr.as_ptr(),
                scale,
                mask_ptr,
                mask_mode,
            )
        };
        Self::from_handle(ptr, "mlx_sdpa")
    }

    /// Fused (biased) layer norm over the last axis; `weight`/`bias` are
    /// skipped when `None`.
    pub fn layer_norm(
        &self,
        weight: Option<&Self>,
        bias: Option<&Self>,
        eps: f32,
    ) -> Result<Self, MlxError> {
        let weight_ptr = weight.map_or(std::ptr::null_mut(), |w| w.ptr.as_ptr());
        let bias_ptr = bias.map_or(std::ptr::null_mut(), |b| b.ptr.as_ptr());
        let ptr = unsafe { ffi::mlx_layer_norm(self.ptr.as_ptr(), weight_ptr, bias_ptr, eps) };
        Self::from_handle(ptr, "mlx_layer_norm")
    }

    /// Exact-erf GELU, composed in `mlx.nn.gelu`'s evaluation order
    /// (`x * (1 + erf(x / sqrt(2))) / 2` with input-dtype scalars) and
    /// compiled into one kernel like `mlx.nn.gelu`.
    pub fn gelu(&self) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_gelu(self.ptr.as_ptr()) };
        Self::from_handle(ptr, "mlx_gelu")
    }

    /// Static slice with per-axis `start`/`stop`/`strides` (all three must
    /// have one entry per axis).
    pub fn slice(
        &self,
        start: &[usize],
        stop: &[usize],
        strides: &[usize],
    ) -> Result<Self, MlxError> {
        if start.len() != stop.len() || start.len() != strides.len() {
            return Err(MlxError::Op(format!(
                "slice index lengths differ: start {}, stop {}, strides {}",
                start.len(),
                stop.len(),
                strides.len()
            )));
        }
        let start_i: Vec<i64> = start.iter().map(|&v| v as i64).collect();
        let stop_i: Vec<i64> = stop.iter().map(|&v| v as i64).collect();
        let strides_i: Vec<i64> = strides.iter().map(|&v| v as i64).collect();
        let ptr = unsafe {
            ffi::mlx_slice(
                self.ptr.as_ptr(),
                start_i.as_ptr(),
                stop_i.as_ptr(),
                strides_i.as_ptr(),
                start_i.len() as i32,
            )
        };
        Self::from_handle(ptr, "mlx_slice")
    }

    /// Concatenates `arrays` along `axis` (shapes must agree elsewhere).
    pub fn concatenate(arrays: &[&Self], axis: usize) -> Result<Self, MlxError> {
        if arrays.is_empty() {
            return Err(MlxError::Op("concatenate of zero arrays".to_string()));
        }
        let mut handles: Vec<*mut ffi::AimdMlxArray> = Vec::with_capacity(arrays.len());
        for array in arrays {
            handles.push(array.ptr.as_ptr());
        }
        let ptr = unsafe {
            ffi::mlx_concatenate(handles.as_mut_ptr(), handles.len() as i32, axis as i32)
        };
        Self::from_handle(ptr, "mlx_concatenate")
    }

    /// Gathers entries along `axis` at integer `indices` (an integral array,
    /// u32 from [`MlxArray::from_u32`]).
    pub fn take(&self, indices: &Self, axis: usize) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_take(self.ptr.as_ptr(), indices.ptr.as_ptr(), axis as i32) };
        Self::from_handle(ptr, "mlx_take")
    }

    /// Argmax along `axis`: a `uint32` index array with first-index
    /// tie-breaking (numpy convention). `keepdims` keeps the reduced axis
    /// as size 1.
    pub fn argmax(&self, axis: i32, keepdims: bool) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_argmax(self.ptr.as_ptr(), axis, i32::from(keepdims)) };
        Self::from_handle(ptr, "mlx_argmax")
    }

    /// Dequantizes an affine triple (`mx.dequantize`): `self` is the packed
    /// `u32` payload; the result dtype is the scales/biases dtype (mixed
    /// `group_size`/`bits` of `None` use mlx defaults).
    pub fn dequantize(
        &self,
        scales: &Self,
        biases: Option<&Self>,
        group_size: Option<i32>,
        bits: Option<i32>,
    ) -> Result<Self, MlxError> {
        let biases_ptr = biases.map_or(std::ptr::null_mut(), |b| b.ptr.as_ptr());
        let ptr = unsafe {
            ffi::mlx_dequantize(
                self.ptr.as_ptr(),
                scales.ptr.as_ptr(),
                biases_ptr,
                group_size.unwrap_or(-1),
                bits.unwrap_or(-1),
            )
        };
        Self::from_handle(ptr, "mlx_dequantize")
    }

    /// NeoX rotary position embedding over the last axis (`mx.fast.rope`,
    /// non-traditional): rotates row `j` with angles from absolute position
    /// `offset + j`. `dims` rotary dimensions of the last axis, `base` the
    /// theta, `scale` the position scale (1.0 for Qwen3).
    pub fn rope(&self, dims: usize, base: f64, scale: f32, offset: i32) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_rope(self.ptr.as_ptr(), dims as i32, base, scale, offset) };
        Self::from_handle(ptr, "mlx_rope")
    }

    // ---------------------------------------------------------------------
    // WS-07 (Unlimited-OCR) additions
    // ---------------------------------------------------------------------

    /// `beta * self + alpha * (a @ b)` (`mx.addmm`; `nn.Linear` with bias
    /// is `addmm(bias, x, W.T)`).
    pub fn addmm(&self, a: &Self, b: &Self, alpha: f32, beta: f32) -> Result<Self, MlxError> {
        let ptr = unsafe {
            ffi::mlx_addmm(
                self.ptr.as_ptr(),
                a.ptr.as_ptr(),
                b.ptr.as_ptr(),
                alpha,
                beta,
            )
        };
        Self::from_handle(ptr, "mlx_addmm")
    }

    /// Element-wise subtract.
    pub fn subtract(&self, rhs: &Self) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_subtract(self.ptr.as_ptr(), rhs.ptr.as_ptr()) };
        Self::from_handle(ptr, "mlx_subtract")
    }

    /// Element-wise floor division (integral inputs stay integral).
    pub fn floor_divide(&self, rhs: &Self) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_floor_divide(self.ptr.as_ptr(), rhs.ptr.as_ptr()) };
        Self::from_handle(ptr, "mlx_floor_divide")
    }

    /// Softmax over `axis` with mlx's `precise` flag (`mx.softmax(...,
    /// precise=True)` accumulates half inputs in f32).
    pub fn softmax_precise(&self, axis: i32, precise: bool) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_softmax_precise(self.ptr.as_ptr(), axis, i32::from(precise)) };
        Self::from_handle(ptr, "mlx_softmax_precise")
    }

    /// `logsumexp` over `axis`.
    pub fn logsumexp(&self, axis: i32, keepdims: bool) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_logsumexp(self.ptr.as_ptr(), axis, i32::from(keepdims)) };
        Self::from_handle(ptr, "mlx_logsumexp")
    }

    /// Sum over `axis`.
    pub fn sum_axis(&self, axis: i32, keepdims: bool) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_sum_axis(self.ptr.as_ptr(), axis, i32::from(keepdims)) };
        Self::from_handle(ptr, "mlx_sum_axis")
    }

    /// Stable argsort along `axis` (`uint32` indices).
    pub fn argsort(&self, axis: i32) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_argsort(self.ptr.as_ptr(), axis) };
        Self::from_handle(ptr, "mlx_argsort")
    }

    /// `mx.argpartition(self, kth, axis)` (`uint32` indices; negative
    /// `kth` counts from the end).
    pub fn argpartition(&self, kth: i32, axis: i32) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_argpartition(self.ptr.as_ptr(), kth, axis) };
        Self::from_handle(ptr, "mlx_argpartition")
    }

    /// `mx.take_along_axis`.
    pub fn take_along_axis(&self, indices: &Self, axis: i32) -> Result<Self, MlxError> {
        let ptr =
            unsafe { ffi::mlx_take_along_axis(self.ptr.as_ptr(), indices.ptr.as_ptr(), axis) };
        Self::from_handle(ptr, "mlx_take_along_axis")
    }

    /// `mx.put_along_axis` (`values` broadcasts against `indices`).
    pub fn put_along_axis(
        &self,
        indices: &Self,
        values: &Self,
        axis: i32,
    ) -> Result<Self, MlxError> {
        let ptr = unsafe {
            ffi::mlx_put_along_axis(
                self.ptr.as_ptr(),
                indices.ptr.as_ptr(),
                values.ptr.as_ptr(),
                axis,
            )
        };
        Self::from_handle(ptr, "mlx_put_along_axis")
    }

    /// Broadcast to `shape`.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self, MlxError> {
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let ptr = unsafe {
            ffi::mlx_broadcast_to(
                self.ptr.as_ptr(),
                shape_i64.as_ptr(),
                shape_i64.len() as i32,
            )
        };
        Self::from_handle(ptr, "mlx_broadcast_to")
    }

    /// Constant-zero padding with per-axis `(low, high)` widths (`mx.pad`).
    pub fn pad(&self, widths: &[(usize, usize)]) -> Result<Self, MlxError> {
        let low: Vec<i64> = widths.iter().map(|w| w.0 as i64).collect();
        let high: Vec<i64> = widths.iter().map(|w| w.1 as i64).collect();
        let ptr = unsafe {
            ffi::mlx_pad(
                self.ptr.as_ptr(),
                low.as_ptr(),
                high.as_ptr(),
                low.len() as i32,
            )
        };
        Self::from_handle(ptr, "mlx_pad")
    }

    /// Functional slice assignment `self[start:stop:strides] = update`
    /// (mlx reuses the buffer in place when `self` is otherwise unreferenced).
    pub fn slice_update(
        &self,
        update: &Self,
        start: &[usize],
        stop: &[usize],
        strides: &[usize],
    ) -> Result<Self, MlxError> {
        if start.len() != stop.len() || start.len() != strides.len() {
            return Err(MlxError::Op(
                "slice_update index lengths differ".to_string(),
            ));
        }
        let start_i: Vec<i64> = start.iter().map(|&v| v as i64).collect();
        let stop_i: Vec<i64> = stop.iter().map(|&v| v as i64).collect();
        let strides_i: Vec<i64> = strides.iter().map(|&v| v as i64).collect();
        let ptr = unsafe {
            ffi::mlx_slice_update(
                self.ptr.as_ptr(),
                update.ptr.as_ptr(),
                start_i.as_ptr(),
                stop_i.as_ptr(),
                strides_i.as_ptr(),
                start_i.len() as i32,
            )
        };
        Self::from_handle(ptr, "mlx_slice_update")
    }

    /// Zero-filled array.
    pub fn zeros(shape: &[usize], dtype: MlxDtype) -> Result<Self, MlxError> {
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let ptr =
            unsafe { ffi::mlx_zeros(shape_i64.as_ptr(), shape_i64.len() as i32, dtype.code()) };
        Self::from_handle(ptr, "mlx_zeros")
    }

    /// `mx.gather_qmm` (affine): quantized matmul whose weight matrix is
    /// gathered per batch element by `rhs_indices` (the
    /// `QuantizedSwitchLinear` primitive of mlx MoE layers).
    #[allow(clippy::too_many_arguments)]
    pub fn gather_qmm(
        &self,
        w: &Self,
        scales: &Self,
        biases: Option<&Self>,
        lhs_indices: Option<&Self>,
        rhs_indices: Option<&Self>,
        transpose_w: bool,
        group_size: Option<i32>,
        bits: Option<i32>,
        sorted_indices: bool,
    ) -> Result<Self, MlxError> {
        let opt = |a: Option<&Self>| a.map_or(std::ptr::null_mut(), |a| a.ptr.as_ptr());
        let ptr = unsafe {
            ffi::mlx_gather_qmm(
                self.ptr.as_ptr(),
                w.ptr.as_ptr(),
                scales.ptr.as_ptr(),
                opt(biases),
                opt(lhs_indices),
                opt(rhs_indices),
                u8::from(transpose_w),
                group_size.unwrap_or(-1),
                bits.unwrap_or(-1),
                u8::from(sorted_indices),
            )
        };
        Self::from_handle(ptr, "mlx_gather_qmm")
    }

    /// Fused single-token MoE experts over affine 4-bit expert stacks:
    /// `sum_k scores[k] * down[e](silu(gate[e](x)) * up[e](x))`, `e =
    /// inds[k]`, in two custom Metal kernels. `self` is `[D]`, `inds` `[K]`
    /// u32, `scores` `[K]`; each projection is `(weight, scales, biases)`
    /// with `[E, out, in/8]` u32 weights. Returns `[D]`. Tracks the
    /// `gather_qmm` + SwiGLU + weighted-sum graph to bf16 rounding; up to two
    /// `[D]` `addends` are then added in order (each rounds like `add`).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_swiglu_4bit(
        &self,
        inds: &Self,
        scores: &Self,
        gate: [&Self; 3],
        up: [&Self; 3],
        down: [&Self; 3],
        group_size: i32,
        addends: &[&Self],
    ) -> Result<Self, MlxError> {
        let mut adds: Vec<*mut ffi::AimdMlxArray> =
            addends.iter().map(|a| a.ptr.as_ptr()).collect();
        let ptr = unsafe {
            ffi::mlx_moe_swiglu_4bit(
                self.ptr.as_ptr(),
                inds.ptr.as_ptr(),
                scores.ptr.as_ptr(),
                gate[0].ptr.as_ptr(),
                gate[1].ptr.as_ptr(),
                gate[2].ptr.as_ptr(),
                up[0].ptr.as_ptr(),
                up[1].ptr.as_ptr(),
                up[2].ptr.as_ptr(),
                down[0].ptr.as_ptr(),
                down[1].ptr.as_ptr(),
                down[2].ptr.as_ptr(),
                group_size,
                adds.as_mut_ptr(),
                adds.len() as i32,
            )
        };
        Self::from_handle(ptr, "mlx_moe_swiglu_4bit")
    }

    /// Single-token decode attention (`self` is `[1, H, 1, D]`, keys/values
    /// `[1, Hkv, N, D]`, no mask): mlx's 2-pass long-KV kernel without the
    /// 4-key prefetch, bit-identical to [`Self::scaled_dot_product_attention`]
    /// (which it falls back to wherever mlx would not take that kernel).
    pub fn sdpa_decode(&self, keys: &Self, values: &Self, scale: f32) -> Result<Self, MlxError> {
        let ptr = unsafe {
            ffi::mlx_sdpa_decode(
                self.ptr.as_ptr(),
                keys.ptr.as_ptr(),
                values.ptr.as_ptr(),
                scale,
            )
        };
        Self::from_handle(ptr, "mlx_sdpa_decode")
    }

    /// Fused single-token MoE router (`MoEGate`, softmax + greedy top-k):
    /// `self` is `[D]`, `gate_w` the row-major `[E, D]` gate weight. Returns
    /// `(inds [k] u32, scores [k])`, bit-identical to `x @ gate_w.T`,
    /// `softmax(precise)`, `argpartition` and `take_along_axis`.
    pub fn moe_router(&self, gate_w: &Self, k: i32) -> Result<(Self, Self), MlxError> {
        let mut inds = std::ptr::null_mut();
        let mut scores = std::ptr::null_mut();
        let status = unsafe {
            ffi::mlx_moe_router(
                self.ptr.as_ptr(),
                gate_w.ptr.as_ptr(),
                k,
                &mut inds,
                &mut scores,
            )
        };
        if status != 0 {
            return Err(op_failed("mlx_moe_router"));
        }
        Ok((
            Self::from_handle(inds, "mlx_moe_router")?,
            Self::from_handle(scores, "mlx_moe_router")?,
        ))
    }

    /// Schedules `arrays` for evaluation without blocking (`mx.async_eval`).
    pub fn async_eval(arrays: &[&Self]) -> Result<(), MlxError> {
        Self::eval_list(arrays, ffi::mlx_async_eval, "mlx_async_eval")
    }

    /// Evaluates `arrays` together (`mx.eval([...])`).
    pub fn eval_many(arrays: &[&Self]) -> Result<(), MlxError> {
        Self::eval_list(arrays, ffi::mlx_eval_many, "mlx_eval_many")
    }

    fn eval_list(
        arrays: &[&Self],
        f: unsafe extern "C" fn(*mut *mut ffi::AimdMlxArray, i32) -> i32,
        op: &str,
    ) -> Result<(), MlxError> {
        if arrays.is_empty() {
            return Ok(());
        }
        let mut handles: Vec<*mut ffi::AimdMlxArray> =
            arrays.iter().map(|a| a.ptr.as_ptr()).collect();
        let status = unsafe { f(handles.as_mut_ptr(), handles.len() as i32) };
        if status == 0 {
            Ok(())
        } else {
            Err(op_failed(op))
        }
    }

    /// Scalar (0-d) array of one `bf16` value.
    pub fn scalar_bf16(value: f32) -> Result<Self, MlxError> {
        Self::from_bf16(&[], &[half::bf16::from_f32(value)])
    }

    /// Shallow copy: a second refcounted handle to the same (immutable)
    /// array, like Python `y = x`. No data is copied.
    pub fn try_clone(&self) -> Result<Self, MlxError> {
        let ptr = unsafe { ffi::mlx_array_clone(self.ptr.as_ptr()) };
        Self::from_handle(ptr, "mlx_array_clone")
    }

    /// Scalar (0-d) `u32` array.
    pub fn scalar_u32(value: u32) -> Result<Self, MlxError> {
        Self::from_u32(&[], &[value])
    }
}

/// Whether the shim computes on the MLX CPU device instead of Metal.
///
/// The device is chosen once per process, before the first array is
/// evaluated: `AIMD_MLX_DEVICE=cpu` forces the CPU device, `gpu` forces the
/// GPU, and anything else (unset) uses Metal when a GPU device is available
/// and falls back to the CPU device otherwise (e.g. virtualized CI runners).
pub fn device_is_cpu() -> Result<bool, MlxError> {
    match unsafe { ffi::mlx_device_is_cpu() } {
        1 => Ok(true),
        0 => Ok(false),
        _ => Err(op_failed("mlx_device_is_cpu")),
    }
}

/// Whether the custom Metal kernels (fused MoE, decode SDPA) can run: the
/// GPU device is selected and it is Metal. False on the CPU device and on
/// the mlx CUDA backend, where the engine keeps plain mlx ops.
pub fn metal_kernels_available() -> Result<bool, MlxError> {
    match unsafe { ffi::mlx_metal_kernels_available() } {
        1 => Ok(true),
        0 => Ok(false),
        _ => Err(op_failed("mlx_metal_kernels_available")),
    }
}

/// mlx allocator counters, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStats {
    /// Buffers held by live arrays.
    pub active: usize,
    /// Highest `active` since the process started or [`reset_peak_memory`].
    pub peak: usize,
    /// Freed buffers kept for reuse.
    pub cache: usize,
    /// The cap on `cache` the shim sets at device selection.
    pub cache_limit: usize,
}

/// Reads mlx's allocator counters (`mx.get_active_memory` and friends).
pub fn memory_stats() -> MemoryStats {
    let stat = |which| unsafe { ffi::mlx_memory_stat(which) }.max(0) as usize;
    MemoryStats {
        active: stat(0),
        peak: stat(1),
        cache: stat(2),
        cache_limit: stat(3),
    }
}

/// Resets [`MemoryStats::peak`] to the current active memory.
pub fn reset_peak_memory() {
    unsafe { ffi::mlx_memory_stat(4) };
}

/// Releases mlx's cached Metal buffers (`mx.clear_cache`).
pub fn clear_cache() -> Result<(), MlxError> {
    let status = unsafe { ffi::mlx_clear_cache() };
    if status == 0 {
        Ok(())
    } else {
        Err(op_failed("mlx_clear_cache"))
    }
}

/// Element-wise unary ops behind [`MlxArray::unary`] (codes match the shim).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum UnaryOp {
    Sigmoid = 0,
    Tanh = 1,
    Exp = 2,
    Sin = 3,
    Cos = 4,
    Negative = 5,
    Abs = 6,
    Log = 7,
    Rsqrt = 8,
    Erf = 9,
    Square = 10,
    Relu = 11,
}

/// Element-wise binary ops behind [`MlxArray::binary`] (codes match the shim).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum BinaryOp {
    Maximum = 0,
    Minimum = 1,
    Greater = 2,
    Less = 3,
    Equal = 4,
}

/// Reductions behind [`MlxArray::reduce`] (codes match the shim).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ReduceOp {
    Mean = 0,
    Max = 1,
    Min = 2,
    Sum = 3,
}

/// `None` maps to the shim's "keep result dtype" sentinel (-1).
fn out_code(out_dtype: Option<MlxDtype>) -> i32 {
    out_dtype.map_or(-1, MlxDtype::code)
}

/// Packs elements to little-endian wire bytes.
fn pack<const N: usize, E: Copy>(data: &[E], encode: fn(E) -> [u8; N]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * N);
    for &elem in data {
        bytes.extend_from_slice(&encode(elem));
    }
    bytes
}

/// Decodes little-endian wire bytes into typed elements.
fn unpack<const N: usize, E: Copy>(bytes: &[u8], decode: fn([u8; N]) -> E) -> Vec<E> {
    let (chunks, rest) = bytes.as_chunks::<N>();
    assert!(rest.is_empty(), "byte length must be a multiple of {N}");
    chunks.iter().map(|&chunk| decode(chunk)).collect()
}

/// Reinterprets 16-bit patterns as typed half elements (bit-exact).
fn unpack_bits<E: Copy>(patterns: &[u16], decode: fn(u16) -> E) -> Vec<E> {
    patterns.iter().map(|&bits| decode(bits)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_dtype_codes_match_shim_abi() {
        assert_eq!(MlxDtype::F16.code(), 0);
        assert_eq!(MlxDtype::Bf16.code(), 1);
        assert_eq!(MlxDtype::F32.code(), 2);
        assert_eq!(MlxDtype::U32.code(), 3);
        assert_eq!(MlxDtype::U8.code(), 4);
        for (code, dtype) in [
            (0, MlxDtype::F16),
            (1, MlxDtype::Bf16),
            (2, MlxDtype::F32),
            (3, MlxDtype::U32),
            (4, MlxDtype::U8),
        ] {
            assert_eq!(MlxDtype::from_code(code).unwrap(), dtype);
        }
        assert!(MlxDtype::from_code(99).is_err());
        assert_eq!(MlxDtype::F16.size_bytes(), 2);
        assert_eq!(MlxDtype::U8.size_bytes(), 1);
    }
}
