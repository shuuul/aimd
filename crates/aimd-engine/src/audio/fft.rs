//! Hand-written FFT (no external DSP crate).
//!
//! Two strategies, selected at plan time:
//!
//! - **Iterative radix-2** (decimation in time) for power-of-two sizes.
//! - **Four-step Cooley-Tukey** for composite sizes: with `n = n1 * n2`
//!   (balanced split), the input is viewed as an `n2 x n1` matrix
//!   `M[m2][m1] = x[n1*m2 + m1]`, transposed, processed as `n1` small DFTs
//!   of length `n2`, twiddled by `W_n^{m1*k2}`, then `n2` small DFTs of
//!   length `n1`, yielding `X[n2*k1 + k2]` in natural order.
//!   `n_fft = 400 = 2^4 * 5^2` from the Qwen3-ASR extractor is **not** a
//!   power of two, so zero-padding to 512 is not an option (it would move
//!   the frequency grid); the four-step decomposition handles it with
//!   ~16k complex multiply-adds per frame instead of the naive 160k.
//! - **Naive DFT** fallback for sizes that factor poorly (correctness
//!   path only; not used by the Qwen3-ASR configuration).
//!
//! All arithmetic is f64; the mel front-end casts to f32 where the torch
//! reference does (see [`super::features`]).

use std::f64::consts::TAU;

/// Precomputed tables for one FFT size.
#[derive(Debug, Clone)]
pub(crate) struct FftPlan {
    n: usize,
    kind: PlanKind,
}

#[derive(Debug, Clone)]
enum PlanKind {
    Radix2 {
        /// Bit-reversal permutation of `0..n`.
        rev: Vec<usize>,
        /// `e^{-2*pi*i*k/n}` for `k in 0..n/2`.
        twiddles: Vec<(f64, f64)>,
    },
    FourStep {
        n1: usize,
        n2: usize,
        small1: SmallDft,
        small2: SmallDft,
        /// `e^{-2*pi*i*m1*k2/n}` for `m1 in 0..n1`, `k2 in 0..n2` (row-major).
        twiddle: Vec<(f64, f64)>,
    },
    Naive {
        /// `e^{-2*pi*i*j*k/n}` table, row-major `[j][k]`.
        twiddles: Vec<(f64, f64)>,
    },
}

/// Twiddle tables plus O(m^2) butterflies for a small DFT.
#[derive(Debug, Clone)]
struct SmallDft {
    m: usize,
    /// `e^{-2*pi*i*j*k/m}` for `j, k in 0..m` (row-major).
    twiddles: Vec<(f64, f64)>,
}

impl SmallDft {
    fn new(m: usize) -> Self {
        let mut twiddles = Vec::with_capacity(m * m);
        for j in 0..m {
            for k in 0..m {
                twiddles.push(cis(-(TAU * (j * k) as f64) / m as f64));
            }
        }
        Self { m, twiddles }
    }

    /// In-place naive DFT of a contiguous slice of length `m`.
    fn dft(&self, re: &mut [f64], im: &mut [f64]) {
        debug_assert_eq!(re.len(), self.m);
        let m = self.m;
        let mut out_re = vec![0.0; m];
        let mut out_im = vec![0.0; m];
        for k in 0..m {
            let mut sr = 0.0;
            let mut si = 0.0;
            for j in 0..m {
                let (w_re, w_im) = self.twiddles[j * m + k];
                sr += w_re * re[j] - w_im * im[j];
                si += w_re * im[j] + w_im * re[j];
            }
            out_re[k] = sr;
            out_im[k] = si;
        }
        re.copy_from_slice(&out_re);
        im.copy_from_slice(&out_im);
    }
}

/// `e^{i*angle}` as `(cos, sin)`.
fn cis(angle: f64) -> (f64, f64) {
    (angle.cos(), angle.sin())
}

impl FftPlan {
    /// Build tables for transform length `n` (`n >= 2`).
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "FFT length must be >= 2");
        if n.is_power_of_two() {
            let rev = (0..n)
                .map(|i| {
                    // Bit-reverse an n-bit index.
                    let bits = n.trailing_zeros();
                    i.reverse_bits() >> (usize::BITS - bits)
                })
                .collect();
            let twiddles = (0..n / 2)
                .map(|k| cis(-TAU * k as f64 / n as f64))
                .collect();
            return Self {
                n,
                kind: PlanKind::Radix2 { rev, twiddles },
            };
        }

        // Most balanced split: the largest divisor of n that is <= sqrt(n).
        // Total butterfly work is n * (n1 + n2), minimized when n1 ~ n2
        // (400 -> 20 x 20).
        let mut n1 = 1;
        let mut d = 2;
        while d * d <= n {
            if n.is_multiple_of(d) {
                n1 = d;
            }
            d += 1;
        }
        let n2 = n / n1;
        if n1 > 1 && n2 <= 64 {
            let small1 = SmallDft::new(n1);
            let small2 = SmallDft::new(n2);
            let twiddle = (0..n1)
                .flat_map(|m1| (0..n2).map(move |k2| cis(-TAU * (m1 * k2) as f64 / n as f64)))
                .collect();
            return Self {
                n,
                kind: PlanKind::FourStep {
                    n1,
                    n2,
                    small1,
                    small2,
                    twiddle,
                },
            };
        }

        // Poorly factored size: correctness-first naive fallback.
        let twiddles = (0..n)
            .flat_map(|j| (0..n).map(move |k| cis(-TAU * (j * k) as f64 / n as f64)))
            .collect();
        Self {
            n,
            kind: PlanKind::Naive { twiddles },
        }
    }

    /// In-place forward FFT. `re`/`im` must have length `n`.
    pub fn fft_in_place(&self, re: &mut [f64], im: &mut [f64], scratch: &mut FftScratch) {
        debug_assert_eq!(re.len(), self.n);
        debug_assert_eq!(im.len(), self.n);
        match &self.kind {
            PlanKind::Radix2 { rev, twiddles } => {
                for (i, &j) in rev.iter().enumerate() {
                    if j > i {
                        re.swap(i, j);
                        im.swap(i, j);
                    }
                }
                let mut len = 2;
                while len <= self.n {
                    let step = self.n / len;
                    let half = len / 2;
                    for start in (0..self.n).step_by(len) {
                        // Twiddle k of this stage is e^{-2*pi*i*k/len} = twiddles[k * step].
                        for k in 0..half {
                            let (w_re, w_im) = twiddles[k * step];
                            let j = start + k;
                            let xr = re[j + half];
                            let xi = im[j + half];
                            let tr = w_re * xr - w_im * xi;
                            let ti = w_re * xi + w_im * xr;
                            re[j + half] = re[j] - tr;
                            im[j + half] = im[j] - ti;
                            re[j] += tr;
                            im[j] += ti;
                        }
                    }
                    len *= 2;
                }
            }
            PlanKind::FourStep {
                n1,
                n2,
                small1,
                small2,
                twiddle,
            } => {
                // Input as an n2 x n1 matrix M[m2][m1] = x[n1*m2 + m1].
                // 1) Transpose to T[m1][m2] = M[m2][m1] (n1 rows x n2 cols).
                scratch.resize(self.n);
                for m2 in 0..*n2 {
                    for m1 in 0..*n1 {
                        scratch.re[m1 * n2 + m2] = re[m2 * n1 + m1];
                        scratch.im[m1 * n2 + m2] = im[m2 * n1 + m1];
                    }
                }
                // 2) Row DFTs of length n2 -> Z[m1][k2].
                for m1 in 0..*n1 {
                    small2.dft(
                        &mut scratch.re[m1 * n2..(m1 + 1) * n2],
                        &mut scratch.im[m1 * n2..(m1 + 1) * n2],
                    );
                }
                // 3) Elementwise twiddle W_n^{m1*k2}.
                for (w, (re_slot, im_slot)) in twiddle
                    .iter()
                    .zip(scratch.re.iter_mut().zip(scratch.im.iter_mut()))
                {
                    let (w_re, w_im) = *w;
                    let (xr, xi) = (*re_slot, *im_slot);
                    *re_slot = w_re * xr - w_im * xi;
                    *im_slot = w_re * xi + w_im * xr;
                }
                // 4) Column DFTs of length n1 over m1: R[k1][k2] =
                //    X[n2*k1 + k2], i.e. natural-order output.
                for k2 in 0..*n2 {
                    for (m1, slot) in scratch.col_re.iter_mut().enumerate().take(*n1) {
                        *slot = scratch.re[m1 * n2 + k2];
                    }
                    for (m1, slot) in scratch.col_im.iter_mut().enumerate().take(*n1) {
                        *slot = scratch.im[m1 * n2 + k2];
                    }
                    small1.dft(&mut scratch.col_re[..*n1], &mut scratch.col_im[..*n1]);
                    for k1 in 0..*n1 {
                        scratch.re[k1 * n2 + k2] = scratch.col_re[k1];
                        scratch.im[k1 * n2 + k2] = scratch.col_im[k1];
                    }
                }
                re.copy_from_slice(&scratch.re);
                im.copy_from_slice(&scratch.im);
            }
            PlanKind::Naive { twiddles } => {
                scratch.resize(self.n);
                for k in 0..self.n {
                    let mut sr = 0.0;
                    let mut si = 0.0;
                    for (j, (&xr, &xi)) in re.iter().zip(im.iter()).enumerate() {
                        let (w_re, w_im) = twiddles[j * self.n + k];
                        sr += w_re * xr - w_im * xi;
                        si += w_re * xi + w_im * xr;
                    }
                    scratch.re[k] = sr;
                    scratch.im[k] = si;
                }
                re.copy_from_slice(&scratch.re);
                im.copy_from_slice(&scratch.im);
            }
        }
    }
}

/// Reusable scratch buffers for [`FftPlan::fft_in_place`].
#[derive(Debug, Default, Clone)]
pub(crate) struct FftScratch {
    re: Vec<f64>,
    im: Vec<f64>,
    col_re: Vec<f64>,
    col_im: Vec<f64>,
}

impl FftScratch {
    fn resize(&mut self, n: usize) {
        self.re.clear();
        self.re.resize(n, 0.0);
        self.im.clear();
        self.im.resize(n, 0.0);
        // Column gather buffers: n is an upper bound for the factor n1.
        self.col_re.clear();
        self.col_re.resize(n, 0.0);
        self.col_im.clear();
        self.col_im.resize(n, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::{FftPlan, FftScratch};

    /// O(n^2) reference DFT.
    fn naive_dft(re: &[f64], im: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = re.len();
        let (mut out_re, mut out_im) = (vec![0.0; n], vec![0.0; n]);
        for k in 0..n {
            let (mut sr, mut si) = (0.0, 0.0);
            for (j, (&xr, &xi)) in re.iter().zip(im.iter()).enumerate() {
                let angle = -std::f64::consts::TAU * ((j * k) % n) as f64 / n as f64;
                let (c, s) = (angle.cos(), angle.sin());
                sr += xr * c - xi * s;
                si += xr * s + xi * c;
            }
            out_re[k] = sr;
            out_im[k] = si;
        }
        (out_re, out_im)
    }

    fn deterministic_signal(n: usize) -> (Vec<f64>, Vec<f64>) {
        (0..n)
            .map(|i| {
                let t = i as f64;
                (
                    (0.7 * t).sin() + 0.3 * (1.3 * t).cos() + 0.05 * (t * t % 17.0).sin(),
                    0.4 * (0.9 * t).cos() - 0.2 * (2.1 * t).sin(),
                )
            })
            .unzip()
    }

    #[test]
    fn audio_fft_matches_naive_dft() {
        // Powers of two (radix-2 path), composites incl. n_fft = 400
        // (four-step path), and awkward sizes (naive fallback).
        for n in [2, 3, 4, 5, 6, 8, 12, 16, 20, 32, 64, 100, 128, 400, 512] {
            let (re, im) = deterministic_signal(n);
            let (want_re, want_im) = naive_dft(&re, &im);
            let plan = FftPlan::new(n);
            let mut scratch = FftScratch::default();
            let (mut got_re, mut got_im) = (re.clone(), im.clone());
            plan.fft_in_place(&mut got_re, &mut got_im, &mut scratch);

            let scale = n as f64; // |X| can reach n
            let worst = got_re
                .iter()
                .zip(got_im.iter())
                .zip(want_re.iter().zip(want_im.iter()))
                .map(|((gr, gi), (wr, wi))| (gr - wr).abs().max((gi - wi).abs()) / scale)
                .fold(0.0f64, f64::max);
            assert!(worst < 1e-12, "n={n}: worst normalized diff {worst:.3e}");
        }
    }

    #[test]
    fn audio_fft_energy_is_preserved() {
        // Parseval sanity on the production size: sum |x|^2 = sum |X|^2 / n.
        let n = 400;
        let (re, im) = deterministic_signal(n);
        let time_energy: f64 = re.iter().zip(im.iter()).map(|(r, i)| r * r + i * i).sum();
        let plan = FftPlan::new(n);
        let mut scratch = FftScratch::default();
        let (mut fre, mut fim) = (re.clone(), im.clone());
        plan.fft_in_place(&mut fre, &mut fim, &mut scratch);
        let freq_energy: f64 = fre.iter().zip(fim.iter()).map(|(r, i)| r * r + i * i).sum();
        assert!((freq_energy / n as f64 - time_energy).abs() / time_energy < 1e-12);
    }
}
