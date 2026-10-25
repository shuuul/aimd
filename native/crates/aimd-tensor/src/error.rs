//! Error type shared by tensor construction, typed views, CPU ops, and the
//! KV cache.

use crate::dtype::DType;

/// Errors surfaced by [`crate::Tensor`] construction, typed accessors, the
/// CPU ops, and the KV cache. All variants are value-level (bad dtype,
/// shape, geometry, capacity); buffer-length/element-count misuse of
/// constructors is reported here too rather than panicking.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TensorError {
    /// A tensor (or op input) had a different dtype than required.
    #[error("dtype mismatch: expected {expected}, got {actual}")]
    DTypeMismatch { expected: DType, actual: DType },

    /// A tensor had the wrong rank or dimensions for an op.
    #[error("shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    /// An element/buffer count did not match the requested shape.
    #[error("element count mismatch: expected {expected} elements, got {actual}")]
    ElementCount { expected: usize, actual: usize },

    /// A dimension is invalid for the requested operation (e.g. odd
    /// `head_dim` for RoPE, zero heads).
    #[error("invalid shape {shape:?}: {reason}")]
    InvalidShape { shape: Vec<usize>, reason: String },

    /// The backing byte buffer is not aligned for a typed view. Construction
    /// paths always produce fresh allocations aligned for every supported
    /// dtype, so this only fires for hand-crafted buffers on unusual
    /// allocators.
    #[error("buffer not aligned enough for {dtype} views")]
    UnalignedBuffer { dtype: DType },

    /// Typed views decode little-endian buffers; big-endian targets are
    /// unsupported.
    #[error("typed views require a little-endian target")]
    BigEndianTarget,

    /// The KV cache ran out of preallocated slots.
    #[error("kv cache full: capacity {capacity} tokens, append of {requested} more")]
    CacheFull { capacity: usize, requested: usize },

    /// Layer index outside `[0, num_layers)`.
    #[error("kv cache layer index {index} out of range for {count} layers")]
    LayerIndexOutOfBounds { index: usize, count: usize },

    /// Appended keys/values (or attention inputs) do not match the cache
    /// geometry.
    #[error("kv cache geometry mismatch: cache is {cache}, input is {input}")]
    GeometryMismatch { cache: String, input: String },
}
