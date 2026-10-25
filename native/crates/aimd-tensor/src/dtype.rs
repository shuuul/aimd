//! Dtype enumeration shared across CPU and future GPU backends.

use std::fmt;

/// Element types used by the engine. Weights and activations are `F16`
/// (compute) or quantized (see `aimd-weights`); `BF16` appears only as a
/// dequantization target for bf16 checkpoints.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DType {
    F16,
    BF16,
    F32,
    I32,
}

impl DType {
    /// Fixed wire size in bytes per element. Buffers are little-endian
    /// (see [`crate::tensor`] for the layout contract).
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F16 | DType::BF16 => 2,
            DType::F32 | DType::I32 => 4,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            DType::I32 => "i32",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::DType;

    #[test]
    fn size_bytes_matches_wire_format() {
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::I32.size_bytes(), 4);
    }
}
