//! Shape and stride arithmetic. Row-major, contiguous by default.

/// Tensor shape. `dims` is row-major (outermost first).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn from_dims(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn elem_count(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Size of the last dimension, or `None` for rank-0 shapes.
    pub fn last_dim(&self) -> Option<usize> {
        self.dims.last().copied()
    }

    /// Product of all dimensions except the last: the "row" count for
    /// last-dim ops. Rank-1 shapes yield 1; rank-0 yields `None`.
    pub fn leading_rows(&self) -> Option<usize> {
        self.dims
            .split_last()
            .map(|(_, leading)| leading.iter().product())
    }
}

#[cfg(test)]
mod tests {
    use super::Shape;

    #[test]
    fn last_dim_and_leading_rows() {
        let s = Shape::from_dims(&[2, 3, 4]);
        assert_eq!(s.last_dim(), Some(4));
        assert_eq!(s.leading_rows(), Some(6));
        assert_eq!(s.elem_count(), 24);

        let v = Shape::from_dims(&[5]);
        assert_eq!(v.last_dim(), Some(5));
        assert_eq!(v.leading_rows(), Some(1));

        let scalar = Shape::from_dims(&[]);
        assert_eq!(scalar.last_dim(), None);
        assert_eq!(scalar.leading_rows(), None);
    }
}
