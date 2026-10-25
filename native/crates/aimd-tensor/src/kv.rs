//! Per-layer KV cache: preallocated contiguous `F16` buffers with append
//! and sliding-window views.
//!
//! The engine allocates one [`KvCache`] up front (max_tokens sized), appends
//! each token's rotated keys/values per layer, and runs attention against
//! the borrowed views. Buffers stay dense row-major `[token, kv_heads,
//! head_dim]`; the sliding window is a view (a row offset), never a copy or
//! a compaction.
//!
//! Decode flow per layer: `append` the current token's rotated k/v, then
//! call [`KvLayerCache::decode_attention`] (or feed `keys_view()` /
//! `values_view()` to [`crate::ops::gqa_decode_attention`] directly).

use crate::dtype::DType;
use crate::error::TensorError;
use crate::ops;
use crate::tensor::Tensor;
use half::f16;

/// One layer's keys/values: `max_tokens` preallocated rows of
/// `[num_kv_heads, head_dim]` `F16` plus the live row count. `head_dim` is
/// shared by keys and values (both models use square heads today; a
/// separate `v_head_dim` can be added when a model needs one).
#[derive(Debug)]
pub struct KvLayerCache {
    num_kv_heads: usize,
    head_dim: usize,
    max_tokens: usize,
    len: usize,
    keys: Vec<f16>,
    values: Vec<f16>,
}

impl KvLayerCache {
    /// Preallocates `max_tokens` rows. Memory is
    /// `2 * max_tokens * num_kv_heads * head_dim * 2` bytes (k + v, f16).
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        max_tokens: usize,
    ) -> Result<Self, TensorError> {
        if num_kv_heads == 0 {
            return Err(TensorError::InvalidShape {
                shape: vec![num_kv_heads, head_dim],
                reason: "num_kv_heads must be nonzero".to_string(),
            });
        }
        if head_dim == 0 {
            return Err(TensorError::InvalidShape {
                shape: vec![num_kv_heads, head_dim],
                reason: "head_dim must be nonzero".to_string(),
            });
        }
        let row = num_kv_heads * head_dim;
        Ok(Self {
            num_kv_heads,
            head_dim,
            max_tokens,
            len: 0,
            keys: vec![f16::ZERO; max_tokens * row],
            values: vec![f16::ZERO; max_tokens * row],
        })
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Live token count.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Slots still available before [`TensorError::CacheFull`].
    pub fn remaining(&self) -> usize {
        self.max_tokens - self.len
    }

    fn row_len(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Appends `new_tokens` rows; keys/values are `F16
    /// [new_tokens, num_kv_heads, head_dim]` (a single-token append is
    /// `[1, num_kv_heads, head_dim]`). Errors with
    /// [`TensorError::CacheFull`] when the preallocation would overflow; the
    /// cache is left untouched in that case.
    pub fn append(&mut self, keys: &Tensor, values: &Tensor) -> Result<(), TensorError> {
        for t in [keys, values] {
            if t.dtype() != DType::F16 {
                return Err(TensorError::DTypeMismatch {
                    expected: DType::F16,
                    actual: t.dtype(),
                });
            }
        }
        let kd = keys.shape().dims();
        if kd.len() != 3 {
            return Err(TensorError::InvalidShape {
                shape: kd.to_vec(),
                reason: "keys must be [tokens, num_kv_heads, head_dim]".to_string(),
            });
        }
        if values.shape().dims() != kd {
            return Err(TensorError::ShapeMismatch {
                expected: format!("{kd:?}"),
                actual: format!("{:?}", values.shape().dims()),
            });
        }
        let (new_tokens, heads, head_dim) = (kd[0], kd[1], kd[2]);
        if heads != self.num_kv_heads || head_dim != self.head_dim {
            return Err(TensorError::GeometryMismatch {
                cache: format!("[_, {}, {}]", self.num_kv_heads, self.head_dim),
                input: format!("[_, {heads}, {head_dim}]"),
            });
        }
        if new_tokens > self.remaining() {
            return Err(TensorError::CacheFull {
                capacity: self.max_tokens,
                requested: new_tokens,
            });
        }
        let ks = keys.as_f16()?;
        let vs = values.as_f16()?;
        let start = self.len * self.row_len();
        self.keys[start..start + ks.len()].copy_from_slice(ks);
        self.values[start..start + vs.len()].copy_from_slice(vs);
        self.len += new_tokens;
        Ok(())
    }

    /// Resets the live length to zero; the allocation is reused.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Live key rows, layout `[len, num_kv_heads, head_dim]` flattened —
    /// exactly the layout [`crate::ops::gqa_decode_attention`] expects.
    pub fn keys_view(&self) -> &[f16] {
        &self.keys[..self.len * self.row_len()]
    }

    /// Live value rows, same layout as [`KvLayerCache::keys_view`].
    pub fn values_view(&self) -> &[f16] {
        &self.values[..self.len * self.row_len()]
    }

    /// Sliding window over the most recent rows. Returns the absolute
    /// position of the first visible token plus the visible slices; never
    /// copies. `window == 0` (or an empty cache) yields empty slices with
    /// `start == len`.
    pub fn window(&self, window: usize) -> KvWindow<'_> {
        let visible = window.min(self.len);
        let start = self.len - visible;
        let row = self.row_len();
        KvWindow {
            start,
            len: visible,
            keys: &self.keys[start * row..self.len * row],
            values: &self.values[start * row..self.len * row],
        }
    }

    /// Convenience: decode attention for the current token against this
    /// layer. `window == Some(w)` keeps only the last `w` rows; `scale ==
    /// None` uses `1 / sqrt(head_dim)`. See
    /// [`crate::ops::gqa_decode_attention`].
    pub fn decode_attention(
        &self,
        q: &Tensor,
        scale: Option<f32>,
        window: Option<usize>,
    ) -> Result<Tensor, TensorError> {
        ops::gqa_decode_attention(
            q,
            self.keys_view(),
            self.values_view(),
            self.num_kv_heads,
            self.head_dim,
            scale,
            window,
        )
    }
}

/// Borrowed sliding-window view over a live [`KvLayerCache`].
#[derive(Debug)]
pub struct KvWindow<'a> {
    /// Absolute token position of the first visible row.
    pub start: usize,
    /// Number of visible rows.
    pub len: usize,
    /// Visible key rows `[len, num_kv_heads, head_dim]` flattened.
    pub keys: &'a [f16],
    /// Visible value rows, same layout as `keys`.
    pub values: &'a [f16],
}

/// All layers' KV caches for one generation.
#[derive(Debug)]
pub struct KvCache {
    layers: Vec<KvLayerCache>,
}

impl KvCache {
    /// Preallocates `num_layers` per-layer caches.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_tokens: usize,
    ) -> Result<Self, TensorError> {
        let layers = (0..num_layers)
            .map(|_| KvLayerCache::new(num_kv_heads, head_dim, max_tokens))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { layers })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, index: usize) -> Result<&KvLayerCache, TensorError> {
        self.layers
            .get(index)
            .ok_or(TensorError::LayerIndexOutOfBounds {
                index,
                count: self.layers.len(),
            })
    }

    pub fn layer_mut(&mut self, index: usize) -> Result<&mut KvLayerCache, TensorError> {
        let count = self.layers.len();
        self.layers
            .get_mut(index)
            .ok_or(TensorError::LayerIndexOutOfBounds { index, count })
    }

    /// Resets every layer (start of a new generation).
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Shape;
    use crate::testutil::Lcg;

    fn token_tensor(rng: &mut Lcg, tokens: usize, kv_heads: usize, head_dim: usize) -> Tensor {
        let mut data = vec![f16::ZERO; tokens * kv_heads * head_dim];
        rng.fill_f16(&mut data);
        Tensor::from_f16_vec(Shape::from_dims(&[tokens, kv_heads, head_dim]), &data).unwrap()
    }

    #[test]
    fn append_grows_views_contiguously() {
        let mut rng = Lcg::new(0xCAFE);
        let mut c = KvLayerCache::new(2, 3, 8).unwrap();
        assert!(c.is_empty());
        assert_eq!(c.remaining(), 8);

        let k1 = token_tensor(&mut rng, 2, 2, 3);
        let v1 = token_tensor(&mut rng, 2, 2, 3);
        c.append(&k1, &v1).unwrap();
        assert_eq!(c.len(), 2);
        let k1d = k1.as_f16().unwrap();
        assert_eq!(c.keys_view(), k1d);

        let k2 = token_tensor(&mut rng, 1, 2, 3);
        let v2 = token_tensor(&mut rng, 1, 2, 3);
        c.append(&k2, &v2).unwrap();
        assert_eq!(c.len(), 3);
        assert_eq!(c.keys_view()[..12], k1d[..12]);
        assert_eq!(c.keys_view()[12..], k2.as_f16().unwrap()[..]);
        assert_eq!(c.values_view()[12..], v2.as_f16().unwrap()[..]);
        assert_eq!(c.remaining(), 5);
    }

    #[test]
    fn append_overflow_and_geometry_errors() {
        let mut rng = Lcg::new(0xBEEF);
        let mut c = KvLayerCache::new(2, 3, 3).unwrap();
        let k = token_tensor(&mut rng, 2, 2, 3);
        let v = token_tensor(&mut rng, 2, 2, 3);
        c.append(&k, &v).unwrap();

        // cache full: requested 2, remaining 1
        let err = c.append(&k, &v).unwrap_err();
        assert_eq!(
            err,
            TensorError::CacheFull {
                capacity: 3,
                requested: 2
            }
        );
        assert_eq!(c.len(), 2, "failed append must leave the cache untouched");

        // geometry mismatch
        let bad = token_tensor(&mut rng, 1, 3, 3);
        assert!(matches!(
            c.append(&bad, &bad),
            Err(TensorError::GeometryMismatch { .. })
        ));

        // rank mismatch
        let rank2 = Tensor::from_f16_vec(Shape::from_dims(&[2, 6]), &[f16::ZERO; 12]).unwrap();
        assert!(matches!(
            c.append(&rank2, &rank2),
            Err(TensorError::InvalidShape { .. })
        ));

        // dtype mismatch
        let f32t = Tensor::from_f32_vec(Shape::from_dims(&[1, 2, 3]), &[0.0; 6]).unwrap();
        assert!(matches!(
            c.append(&f32t, &f32t),
            Err(TensorError::DTypeMismatch { .. })
        ));
    }

    #[test]
    fn window_is_a_view_over_the_last_rows() {
        let mut rng = Lcg::new(0x610C);
        let mut c = KvLayerCache::new(2, 3, 8).unwrap();
        for _ in 0..4 {
            let k = token_tensor(&mut rng, 1, 2, 3);
            let v = token_tensor(&mut rng, 1, 2, 3);
            c.append(&k, &v).unwrap();
        }
        assert_eq!(c.len(), 4);

        let w = c.window(2);
        assert_eq!(w.start, 2);
        assert_eq!(w.len, 2);
        assert_eq!(w.keys, &c.keys_view()[12..24]);

        let full = c.window(100);
        assert_eq!(full.start, 0);
        assert_eq!(full.len, 4);
        assert_eq!(full.keys, c.keys_view());

        let none = c.window(0);
        assert_eq!(none.start, 4);
        assert_eq!(none.len, 0);
        assert!(none.keys.is_empty() && none.values.is_empty());

        let empty_cache = KvLayerCache::new(1, 1, 4).unwrap();
        let w = empty_cache.window(3);
        assert_eq!(w.start, 0);
        assert_eq!(w.len, 0);
    }

    #[test]
    fn reset_reuses_the_allocation() {
        let mut rng = Lcg::new(0x2E2E);
        let mut c = KvLayerCache::new(2, 3, 8).unwrap();
        let k = token_tensor(&mut rng, 2, 2, 3);
        let v = token_tensor(&mut rng, 2, 2, 3);
        c.append(&k, &v).unwrap();
        let base = c.keys_view().as_ptr();
        c.reset();
        assert!(c.is_empty());
        assert_eq!(c.keys_view(), &[] as &[f16]);
        c.append(&k, &v).unwrap();
        assert_eq!(c.keys_view().as_ptr(), base);
        assert_eq!(c.keys_view(), k.as_f16().unwrap());
    }

    #[test]
    fn decode_attention_wrapper_matches_direct_ops_call() {
        let mut rng = Lcg::new(0x91A7C);
        let (kv_heads, head_dim, max_tokens) = (2usize, 4usize, 16usize);
        let mut c = KvLayerCache::new(kv_heads, head_dim, max_tokens).unwrap();
        for _ in 0..5 {
            let k = token_tensor(&mut rng, 1, kv_heads, head_dim);
            let v = token_tensor(&mut rng, 1, kv_heads, head_dim);
            c.append(&k, &v).unwrap();
        }
        let mut qd = vec![f16::ZERO; 4 * head_dim];
        rng.fill_f16(&mut qd);
        let q = Tensor::from_f16_vec(Shape::from_dims(&[4, head_dim]), &qd).unwrap();

        let via_wrapper = c.decode_attention(&q, None, None).unwrap();
        let direct = ops::gqa_decode_attention(
            &q,
            c.keys_view(),
            c.values_view(),
            kv_heads,
            head_dim,
            None,
            None,
        )
        .unwrap();
        assert_eq!(via_wrapper.as_f16().unwrap(), direct.as_f16().unwrap());

        let via_window = c.decode_attention(&q, None, Some(2)).unwrap();
        let direct_window = ops::gqa_decode_attention(
            &q,
            c.keys_view(),
            c.values_view(),
            kv_heads,
            head_dim,
            None,
            Some(2),
        )
        .unwrap();
        assert_eq!(
            via_window.as_f16().unwrap(),
            direct_window.as_f16().unwrap()
        );
    }

    #[test]
    fn multi_layer_cache_is_independent() {
        let mut rng = Lcg::new(0x1A7E5);
        let mut cache = KvCache::new(3, 2, 4, 6).unwrap();
        assert_eq!(cache.num_layers(), 3);

        let k = token_tensor(&mut rng, 2, 2, 4);
        let v = token_tensor(&mut rng, 2, 2, 4);
        cache.layer_mut(1).unwrap().append(&k, &v).unwrap();

        assert_eq!(cache.layer(0).unwrap().len(), 0);
        assert_eq!(cache.layer(1).unwrap().len(), 2);
        assert_eq!(cache.layer(2).unwrap().len(), 0);

        assert!(matches!(
            cache.layer(3),
            Err(TensorError::LayerIndexOutOfBounds { index: 3, count: 3 })
        ));
        assert!(matches!(
            cache.layer_mut(9),
            Err(TensorError::LayerIndexOutOfBounds { .. })
        ));

        cache.reset();
        assert!(cache.layer(1).unwrap().is_empty());
    }

    #[test]
    fn layer_cache_rejects_zero_geometry() {
        assert!(matches!(
            KvLayerCache::new(0, 4, 8),
            Err(TensorError::InvalidShape { .. })
        ));
        assert!(matches!(
            KvLayerCache::new(2, 0, 8),
            Err(TensorError::InvalidShape { .. })
        ));
    }
}
