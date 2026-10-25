//! Deterministic helpers shared by the unit tests: an LCG for random data
//! (no external `rand` dependency) and f16-vs-reference tolerance asserts.

use crate::f16;

/// Small deterministic LCG (Numerical Recipes constants). The same seed
/// produces the same sequence on every platform and every run, so test data
/// is reproducible without an external dependency.
pub(crate) struct Lcg(u64);

impl Lcg {
    /// Seeds the generator; seed mixing makes small seeds diverge quickly.
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    /// Next raw value, taking the high 32 bits of the state.
    pub(crate) fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    /// Uniform sample in `[-1, 1)`.
    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    /// Uniform f16 sample in `[-1, 1)`.
    pub(crate) fn next_f16(&mut self) -> f16 {
        f16::from_f32(self.next_f32())
    }

    /// Fills `out` with uniform f16 samples in `[-1, 1)`.
    pub(crate) fn fill_f16(&mut self, out: &mut [f16]) {
        for v in out.iter_mut() {
            *v = self.next_f16();
        }
    }
}

/// Asserts `actual` (f16) matches `reference` (f64) elementwise within
/// `atol + rtol * |reference|`. Panics with the first offending index.
pub(crate) fn assert_close(actual: &[f16], reference: &[f64], atol: f64, rtol: f64) {
    assert_eq!(actual.len(), reference.len(), "length mismatch");
    for (i, (&a, &r)) in actual.iter().zip(reference).enumerate() {
        let af = f64::from(a.to_f32());
        let tol = atol + rtol * r.abs();
        assert!(
            (af - r).abs() <= tol,
            "element {i}: actual {af} vs reference {r} (tolerance {tol})"
        );
    }
}
