//! Owned CPU tensor over an immutable byte buffer.
//!
//! Layout contract: little-endian, contiguous, row-major (outermost
//! dimension first, C order). There are no strides — every [`Tensor`] is
//! dense, and ops that reshape produce fresh tensors. Mutation never happens
//! in place; ops return new tensors over freshly packed buffers.
//!
//! Typed accessors ([`Tensor::as_f16`] and friends) borrow the buffer
//! directly as element slices instead of copying. This uses
//! `slice::align_to`, which is fully safe: if the backing allocation is
//! somehow not aligned for the element type, the accessor returns
//! [`TensorError::UnalignedBuffer`] rather than misinterpreting bytes. All
//! element types used here (`f16`, `bf16`, `f32`, `i32`) accept every bit
//! pattern, so the reinterpretation is always value-valid.

use crate::dtype::DType;
use crate::error::TensorError;
use crate::shape::Shape;
use half::{bf16, f16};
use std::sync::Arc;

/// Immutable tensor: dtype + shape + contiguous row-major little-endian
/// byte buffer. Compute paths borrow typed slices via the typed accessors
/// instead of copying.
#[derive(Clone, Debug)]
pub struct Tensor {
    dtype: DType,
    shape: Shape,
    buf: Arc<Vec<u8>>,
}

impl Tensor {
    /// Builds a tensor from raw little-endian bytes.
    ///
    /// Returns [`TensorError::ElementCount`] when `buf.len()` does not equal
    /// `shape.elem_count() * dtype.size_bytes()`.
    pub fn from_bytes(dtype: DType, shape: Shape, buf: Vec<u8>) -> Result<Self, TensorError> {
        let expected = shape.elem_count().checked_mul(dtype.size_bytes());
        let Some(expected) = expected else {
            return Err(TensorError::ElementCount {
                expected: usize::MAX,
                actual: buf.len(),
            });
        };
        if buf.len() != expected {
            return Err(TensorError::ElementCount {
                expected,
                actual: buf.len(),
            });
        }
        Ok(Self {
            dtype,
            shape,
            buf: Arc::new(buf),
        })
    }

    /// Packs `f16` elements (little-endian on the wire) into a new tensor.
    pub fn from_f16_vec(shape: Shape, data: &[f16]) -> Result<Self, TensorError> {
        Self::from_le_elems(DType::F16, shape, data, f16::to_le_bytes)
    }

    /// Packs `bf16` elements (little-endian on the wire) into a new tensor.
    pub fn from_bf16_vec(shape: Shape, data: &[bf16]) -> Result<Self, TensorError> {
        Self::from_le_elems(DType::BF16, shape, data, bf16::to_le_bytes)
    }

    /// Packs `f32` elements (little-endian on the wire) into a new tensor.
    pub fn from_f32_vec(shape: Shape, data: &[f32]) -> Result<Self, TensorError> {
        Self::from_le_elems(DType::F32, shape, data, f32::to_le_bytes)
    }

    /// Packs `i32` elements (little-endian on the wire) into a new tensor.
    pub fn from_i32_vec(shape: Shape, data: &[i32]) -> Result<Self, TensorError> {
        Self::from_le_elems(DType::I32, shape, data, i32::to_le_bytes)
    }

    fn from_le_elems<const N: usize, E: Copy>(
        dtype: DType,
        shape: Shape,
        data: &[E],
        encode: fn(E) -> [u8; N],
    ) -> Result<Self, TensorError> {
        let expected = shape.elem_count();
        if data.len() != expected {
            return Err(TensorError::ElementCount {
                expected,
                actual: data.len(),
            });
        }
        let mut buf = Vec::with_capacity(expected * N);
        for &elem in data {
            buf.extend_from_slice(&encode(elem));
        }
        Ok(Self {
            dtype,
            shape,
            buf: Arc::new(buf),
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    /// Borrows the buffer as `f16` elements (little-endian decode).
    pub fn as_f16(&self) -> Result<&[f16], TensorError> {
        self.typed_view(DType::F16)
    }

    /// Borrows the buffer as `bf16` elements (little-endian decode).
    pub fn as_bf16(&self) -> Result<&[bf16], TensorError> {
        self.typed_view(DType::BF16)
    }

    /// Borrows the buffer as `f32` elements (little-endian decode).
    pub fn as_f32(&self) -> Result<&[f32], TensorError> {
        self.typed_view(DType::F32)
    }

    /// Borrows the buffer as `i32` elements (little-endian decode).
    pub fn as_i32(&self) -> Result<&[i32], TensorError> {
        self.typed_view(DType::I32)
    }

    fn typed_view<T>(&self, expected: DType) -> Result<&[T], TensorError> {
        if self.dtype != expected {
            return Err(TensorError::DTypeMismatch {
                expected,
                actual: self.dtype,
            });
        }
        if !cfg!(target_endian = "little") {
            return Err(TensorError::BigEndianTarget);
        }
        // SAFETY: `align_to` reinterprets the middle subslice as `T`; its
        // safety contract requires every value in that subslice to be a
        // valid `T`. All element types exposed here (`f16`, `bf16`, `f32`,
        // `i32`) accept every bit pattern, and the buffer is fully
        // initialized, so the cast is sound for any content. The middle
        // subslice is the largest suitably aligned part; if the whole buffer
        // is not aligned we report an error instead of reading it.
        let (prefix, middle, suffix) = unsafe { self.buf.as_slice().align_to::<T>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err(TensorError::UnalignedBuffer { dtype: self.dtype });
        }
        debug_assert_eq!(middle.len(), self.elem_count());
        Ok(middle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_and_layout() {
        let data: Vec<f16> = [1.0f32, -2.0, 0.5]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let t = Tensor::from_f16_vec(Shape::from_dims(&[3]), &data).unwrap();
        assert_eq!(t.dtype(), DType::F16);
        assert_eq!(t.elem_count(), 3);
        // f16 1.0 == 0x3C00 -> little-endian bytes [0x00, 0x3C]
        assert_eq!(&t.as_bytes()[..2], &[0x00, 0x3C]);
        let view = t.as_f16().unwrap();
        assert_eq!(view.len(), 3);
        assert_eq!(view[0].to_f32(), 1.0);
        assert_eq!(view[1].to_f32(), -2.0);
        assert_eq!(view[2].to_f32(), 0.5);
    }

    #[test]
    fn f32_layout_and_roundtrip() {
        let t = Tensor::from_f32_vec(Shape::from_dims(&[2, 1]), &[-2.0, 3.5]).unwrap();
        // f32 -2.0 == 0xC0000000 -> LE [0x00, 0x00, 0x00, 0xC0]
        assert_eq!(&t.as_bytes()[..4], &[0x00, 0x00, 0x00, 0xC0]);
        let view = t.as_f32().unwrap();
        assert_eq!(view, &[-2.0, 3.5]);
    }

    #[test]
    fn i32_and_bf16_roundtrips() {
        let t = Tensor::from_i32_vec(Shape::from_dims(&[2]), &[1, -1]).unwrap();
        assert_eq!(&t.as_bytes()[..4], &[1, 0, 0, 0]);
        assert_eq!(t.as_i32().unwrap(), &[1, -1]);

        let b = Tensor::from_bf16_vec(Shape::from_dims(&[1]), &[bf16::from_f32(1.0)]).unwrap();
        // bf16 1.0 == 0x3F80 -> LE [0x80, 0x3F]
        assert_eq!(&b.as_bytes()[..2], &[0x80, 0x3F]);
        assert_eq!(b.as_bf16().unwrap()[0].to_f32(), 1.0);
    }

    #[test]
    fn from_bytes_matches_element_count() {
        let t =
            Tensor::from_bytes(DType::F16, Shape::from_dims(&[2]), vec![0, 0x3C, 0, 0x3C]).unwrap();
        assert_eq!(t.as_f16().unwrap(), &[f16::ONE, f16::ONE]);

        let err = Tensor::from_bytes(DType::F16, Shape::from_dims(&[2]), vec![0; 3]).unwrap_err();
        assert_eq!(
            err,
            TensorError::ElementCount {
                expected: 4,
                actual: 3
            }
        );
    }

    #[test]
    fn typed_vec_length_mismatch_is_an_error() {
        let err = Tensor::from_f16_vec(Shape::from_dims(&[4]), &[f16::ONE]).unwrap_err();
        assert_eq!(
            err,
            TensorError::ElementCount {
                expected: 4,
                actual: 1
            }
        );
    }

    #[test]
    fn accessor_reports_dtype_mismatch() {
        let t = Tensor::from_f32_vec(Shape::from_dims(&[2]), &[0.0, 0.0]).unwrap();
        let err = t.as_f16().unwrap_err();
        assert_eq!(
            err,
            TensorError::DTypeMismatch {
                expected: DType::F16,
                actual: DType::F32
            }
        );
        assert!(t.as_f32().is_ok());
    }

    #[test]
    fn clone_shares_the_buffer() {
        let t = Tensor::from_f16_vec(Shape::from_dims(&[2]), &[f16::ONE, f16::ONE]).unwrap();
        let u = t.clone();
        assert_eq!(t.as_bytes().as_ptr(), u.as_bytes().as_ptr());
        assert_eq!(u.as_f16().unwrap().len(), 2);
    }
}
