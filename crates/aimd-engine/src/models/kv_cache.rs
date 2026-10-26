//! Python-exact per-layer KV caches shared by the decoders: mlx-lm/mlx_vlm
//! `KVCache` (step-256 preallocation + `slice_update`) and the Unlimited-OCR
//! `RingSlidingKVCache`.
//!
//! [`KvCache::update_and_fetch`] grows by `ceil(seq / 256) * 256` zero rows
//! (trimming to `offset` first when `offset % 256 != 0`) and slice-assigns
//! `[prev, offset)`; the ring mode fixes `prefill_length` at the first
//! single-token update, appends until `prefill_length + window` rows, then
//! overwrites ring slots while the absolute RoPE offset keeps increasing.

use aimd_mlx::{MlxArray, MlxError};

/// `KVCache.step`.
const KV_STEP: usize = 256;

/// One layer's KV cache: `KVCache`, or `RingSlidingKVCache(window)` when
/// `window` is set.
pub(crate) struct KvCache {
    keys: Option<MlxArray>,
    values: Option<MlxArray>,
    offset: usize,
    window: Option<usize>,
    prefill_length: Option<usize>,
    ring_pos: usize,
}

impl KvCache {
    pub(crate) fn new(window: Option<usize>) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            window,
            prefill_length: None,
            ring_pos: 0,
        }
    }

    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    #[cfg(test)]
    pub(crate) fn prefill_length(&self) -> Option<usize> {
        self.prefill_length
    }

    /// Backing key buffer (all allocated rows), for golden checks.
    #[cfg(test)]
    pub(crate) fn keys_buffer(&self) -> Option<&MlxArray> {
        self.keys.as_ref()
    }

    /// `KVCache.update_and_fetch`.
    fn base_update(
        &mut self,
        k: &MlxArray,
        v: &MlxArray,
    ) -> Result<(MlxArray, MlxArray), MlxError> {
        let ks = k.shape()?;
        let vs = v.shape()?;
        let (b, h, seq, kd) = (ks[0], ks[1], ks[2], ks[3]);
        let vd = vs[3];
        let prev = self.offset;
        let capacity = match &self.keys {
            Some(keys) => keys.shape()?[2],
            None => 0,
        };
        if self.keys.is_none() || prev + seq > capacity {
            let n_steps = (KV_STEP + seq - 1) / KV_STEP;
            let dtype = k.dtype()?;
            let new_k = MlxArray::zeros(&[b, h, n_steps * KV_STEP, kd], dtype)?;
            let new_v = MlxArray::zeros(&[b, h, n_steps * KV_STEP, vd], v.dtype()?)?;
            match (self.keys.take(), self.values.take()) {
                (Some(mut keys), Some(mut values)) => {
                    if !prev.is_multiple_of(KV_STEP) {
                        keys = keys.slice(&[0, 0, 0, 0], &[b, h, prev, kd], &[1, 1, 1, 1])?;
                        values = values.slice(&[0, 0, 0, 0], &[b, h, prev, vd], &[1, 1, 1, 1])?;
                    }
                    self.keys = Some(MlxArray::concatenate(&[&keys, &new_k], 2)?);
                    self.values = Some(MlxArray::concatenate(&[&values, &new_v], 2)?);
                }
                _ => {
                    self.keys = Some(new_k);
                    self.values = Some(new_v);
                }
            }
        }
        self.offset += seq;
        let end = self.offset;
        let keys = self.keys.take().expect("allocated");
        let values = self.values.take().expect("allocated");
        let keys = keys.slice_update(k, &[0, 0, prev, 0], &[b, h, end, kd], &[1, 1, 1, 1])?;
        let values = values.slice_update(v, &[0, 0, prev, 0], &[b, h, end, vd], &[1, 1, 1, 1])?;
        let out_k = keys.slice(&[0, 0, 0, 0], &[b, h, end, kd], &[1, 1, 1, 1])?;
        let out_v = values.slice(&[0, 0, 0, 0], &[b, h, end, vd], &[1, 1, 1, 1])?;
        self.keys = Some(keys);
        self.values = Some(values);
        Ok((out_k, out_v))
    }

    /// `update_and_fetch` (dispatching on the ring mode).
    pub(crate) fn update_and_fetch(
        &mut self,
        k: &MlxArray,
        v: &MlxArray,
    ) -> Result<(MlxArray, MlxArray), MlxError> {
        let Some(window) = self.window else {
            return self.base_update(k, v);
        };
        let ks = k.shape()?;
        let (b, h, seq, kd) = (ks[0], ks[1], ks[2], ks[3]);
        let vd = v.shape()?[3];
        let prefill = match self.prefill_length {
            Some(p) => p,
            None => {
                if seq > 1 {
                    return self.base_update(k, v);
                }
                self.prefill_length = Some(self.offset);
                self.offset
            }
        };
        if self.keys.is_none() || self.offset < prefill + window {
            let out = self.base_update(k, v)?;
            if self.offset >= prefill + window {
                self.ring_pos = 0;
            }
            return Ok(out);
        }
        let mut keys = self.keys.take().expect("allocated");
        let mut values = self.values.take().expect("allocated");
        for idx in 0..seq {
            let slot = prefill + self.ring_pos;
            let kr = k.slice(&[0, 0, idx, 0], &[b, h, idx + 1, kd], &[1, 1, 1, 1])?;
            let vr = v.slice(&[0, 0, idx, 0], &[b, h, idx + 1, vd], &[1, 1, 1, 1])?;
            keys =
                keys.slice_update(&kr, &[0, 0, slot, 0], &[b, h, slot + 1, kd], &[1, 1, 1, 1])?;
            values =
                values.slice_update(&vr, &[0, 0, slot, 0], &[b, h, slot + 1, vd], &[1, 1, 1, 1])?;
            self.ring_pos = (self.ring_pos + 1) % window;
        }
        self.offset += seq;
        let end = prefill + window;
        let out_k = keys.slice(&[0, 0, 0, 0], &[b, h, end, kd], &[1, 1, 1, 1])?;
        let out_v = values.slice(&[0, 0, 0, 0], &[b, h, end, vd], &[1, 1, 1, 1])?;
        self.keys = Some(keys);
        self.values = Some(values);
        Ok((out_k, out_v))
    }

    /// `cache.state` arrays (what mlx_vlm evaluates after each prefill
    /// chunk).
    pub(crate) fn state(&self) -> Result<Vec<MlxArray>, MlxError> {
        let (Some(k), Some(v)) = (&self.keys, &self.values) else {
            return Ok(Vec::new());
        };
        let ks = k.shape()?;
        let vd = v.shape()?[3];
        let end = match (self.window, self.prefill_length) {
            (Some(w), Some(p)) => self.offset.min(p + w),
            _ => self.offset,
        };
        Ok(vec![
            k.slice(&[0, 0, 0, 0], &[ks[0], ks[1], end, ks[3]], &[1, 1, 1, 1])?,
            v.slice(&[0, 0, 0, 0], &[ks[0], ks[1], end, vd], &[1, 1, 1, 1])?,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ring/plain cache bookkeeping on tiny arrays vs the Python
    /// `RingSlidingKVCache` trace (no checkpoint).
    #[test]
    fn decoder_ring_cache_semantics() {
        let row = |v: f32, n: usize| {
            MlxArray::from_f32(&[1, 1, n, 2], &vec![v; n * 2])
                .unwrap()
                .astype(aimd_mlx::MlxDtype::Bf16)
                .unwrap()
        };
        let col0 = |a: &MlxArray| -> Vec<f32> {
            let v = a
                .astype(aimd_mlx::MlxDtype::F32)
                .unwrap()
                .to_f32_vec()
                .unwrap();
            v.chunks(2).map(|c| c[0]).collect()
        };
        let mut c = KvCache::new(Some(2));
        // Multi-row prefill: plain append, prefill_length unset.
        let (k, _) = c.update_and_fetch(&row(1.0, 3), &row(1.0, 3)).unwrap();
        assert_eq!((c.offset(), c.prefill_length()), (3, None));
        assert_eq!(col0(&k), vec![1.0; 3]);
        // First single-token step fixes prefill_length = 3; appends until
        // offset == prefill + window.
        let (k, _) = c.update_and_fetch(&row(2.0, 1), &row(2.0, 1)).unwrap();
        assert_eq!((c.offset(), c.prefill_length()), (4, Some(3)));
        assert_eq!(col0(&k), vec![1.0, 1.0, 1.0, 2.0]);
        let (k, _) = c.update_and_fetch(&row(3.0, 1), &row(3.0, 1)).unwrap();
        assert_eq!(col0(&k), vec![1.0, 1.0, 1.0, 2.0, 3.0]);
        // Ring phase: overwrite slots 3, 4, 3, ... ; returned length fixed.
        let (k, _) = c.update_and_fetch(&row(4.0, 1), &row(4.0, 1)).unwrap();
        assert_eq!(c.offset(), 6);
        assert_eq!(col0(&k), vec![1.0, 1.0, 1.0, 4.0, 3.0]);
        let (k, _) = c.update_and_fetch(&row(5.0, 1), &row(5.0, 1)).unwrap();
        assert_eq!(col0(&k), vec![1.0, 1.0, 1.0, 4.0, 5.0]);
        let (k, _) = c.update_and_fetch(&row(6.0, 1), &row(6.0, 1)).unwrap();
        assert_eq!(col0(&k), vec![1.0, 1.0, 1.0, 6.0, 5.0]);
        assert_eq!(c.offset(), 8);
        // Step-256 growth: the backing buffer holds one 256-row block.
        assert_eq!(c.keys_buffer().unwrap().shape().unwrap()[2], 256);
        // Plain cache: growth past 256 trims to offset first.
        let mut p = KvCache::new(None);
        p.update_and_fetch(&row(1.0, 255), &row(1.0, 255)).unwrap();
        let (k, _) = p.update_and_fetch(&row(2.0, 3), &row(2.0, 3)).unwrap();
        assert_eq!(p.offset(), 258);
        assert_eq!(k.shape().unwrap()[2], 258);
        assert_eq!(p.keys_buffer().unwrap().shape().unwrap()[2], 255 + 256);
    }
}
