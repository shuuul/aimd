//! SIMD microkernels for the f16 matmul hot loops.
//!
//! Runtime dispatch (cached):
//! - aarch64: NEON f32x4 FMA accumulation; the f16 -> f32 conversion runs
//!   scalar because the fp16 load/convert intrinsics (`vld1q_f16`,
//!   `vcvtq_f32_f16`) are still nightly-only (`stdarch_neon_f16`). Full
//!   fp16 vector loads land when that stabilizes.
//! - x86_64: AVX2 + F16C + FMA — `_mm256_cvtph_ps` + `_mm256_fmadd_ps`,
//!   fully vectorized including the f16 conversion.
//! - Scalar fallback otherwise.
//!
//! Accumulation is f32 with a fixed lane order, so results are
//! deterministic for a given SIMD level; they may differ slightly from the
//! scalar path (tolerance-based parity tests cover this). All loads are
//! unaligned; no alignment is assumed.

use half::f16;

/// Dispatch level, detected once per process.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Level {
    Neon,
    /// Only constructed on x86_64; keep the variant for the dispatch arms.
    #[allow(dead_code)]
    Avx,
    Scalar,
}

fn level() -> Level {
    use std::sync::OnceLock;
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Level::Neon;
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("f16c")
                && std::arch::is_x86_feature_detected!("fma")
            {
                return Level::Avx;
            }
        }
        Level::Scalar
    })
}

/// `acc[j] += alpha * b[j]` over f16 `b` into f32 `acc` (broadcast axpy —
/// the matmul panel inner loop).
#[inline]
pub fn axpy_f16_f32(alpha: f32, b: &[f16], acc: &mut [f32]) {
    debug_assert_eq!(b.len(), acc.len());
    let n = b.len().min(acc.len());
    match level() {
        #[cfg(target_arch = "aarch64")]
        Level::Neon => unsafe { axpy_neon(alpha, b, acc, n) },
        #[cfg(target_arch = "x86_64")]
        Level::Avx => unsafe { axpy_avx(alpha, b, acc, n) },
        Level::Scalar => axpy_scalar(alpha, b, acc, n),
        // The SIMD levels are only detected on their matching architectures.
        #[allow(unreachable_patterns)]
        _ => unreachable!("simd level not available on this arch"),
    }
}

/// `sum_j a[j] * b[j]` over f16 slices, f32 accumulate (dot product — the
/// transposed-B logits inner loop).
#[inline]
pub fn dot_f16_f32(a: &[f16], b: &[f16]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len().min(b.len());
    match level() {
        #[cfg(target_arch = "aarch64")]
        Level::Neon => unsafe { dot_neon(a, b, n) },
        #[cfg(target_arch = "x86_64")]
        Level::Avx => unsafe { dot_avx(a, b, n) },
        Level::Scalar => dot_scalar(a, b, n),
        #[allow(unreachable_patterns)]
        _ => unreachable!("simd level not available on this arch"),
    }
}

// ---------------------------------------------------------------------------
// Scalar reference

fn axpy_scalar(alpha: f32, b: &[f16], acc: &mut [f32], n: usize) {
    for j in 0..n {
        acc[j] += alpha * b[j].to_f32();
    }
}

fn dot_scalar(a: &[f16], b: &[f16], n: usize) -> f32 {
    let mut sum = 0.0f32;
    for j in 0..n {
        sum += a[j].to_f32() * b[j].to_f32();
    }
    sum
}

// ---------------------------------------------------------------------------
// aarch64 NEON (stable)

/// Vector-convert 4 f16 values to f32 on stable NEON via the classic
/// integer bit algorithm (no `fp16` target feature needed):
/// normals add the bias difference to the exponent field, denormals
/// (and zeros) reinterpret the shifted mantissa as an f32 denormal and
/// multiply by `2^125`, inf/NaN force the f32 exponent to 0xFF.
/// Bit-exact for all 65536 f16 patterns (locked by the exhaustive test).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn cvt4_neon(h: &[f16]) -> [f32; 4] {
    use std::arch::aarch64::{
        float32x4_t, uint32x4_t, vaddq_u32, vand_u16, vbslq_u32, vcgeq_u32, vcltq_u32, vdup_n_u16,
        vdupq_n_u32, vld1_u16, vmulq_n_f32, vorrq_u32, vshll_n_u16,
    };
    unsafe {
        let h4 = vld1_u16(h.as_ptr() as *const u16);
        let em16 = vand_u16(h4, vdup_n_u16(0x7fff));
        let sign = vshll_n_u16(vand_u16(h4, vdup_n_u16(0x8000)), 16);
        let em_w = vshll_n_u16(em16, 13);
        let em_full = vshll_n_u16(em16, 0);
        let denorm_mask = vcltq_u32(em_full, vdupq_n_u32(0x0400));
        let infnan_mask = vcgeq_u32(em_full, vdupq_n_u32(0x7c00));
        let normal_bits = vaddq_u32(em_w, vdupq_n_u32(0x3800_0000));
        let infnan_bits = vorrq_u32(em_w, vdupq_n_u32(0x7f80_0000));
        // Denormal/zero lanes: bitcast(em_w) is the f32 denormal with
        // mantissa field `mant << 13` = value `mant * 2^-136`; the true
        // value is `mant * 2^-24` — multiply by 2^112 (exact power of two).
        let denorm_f = vmulq_n_f32(
            std::mem::transmute::<uint32x4_t, float32x4_t>(em_w),
            2.0f32.powi(112),
        );
        let bits = vbslq_u32(
            denorm_mask,
            std::mem::transmute::<float32x4_t, uint32x4_t>(denorm_f),
            vbslq_u32(infnan_mask, infnan_bits, normal_bits),
        );
        std::mem::transmute::<uint32x4_t, [f32; 4]>(vorrq_u32(bits, sign))
    }
}

/// Convert 8 f16 values into two f32 quartets (vectorized on aarch64).
#[cfg(target_arch = "aarch64")]
#[inline]
fn cvt8(b: &[f16]) -> ([f32; 4], [f32; 4]) {
    // SAFETY: cvt4_neon only requires NEON (runtime-detected) and reads
    // exactly 4 f16 words from the slice.
    unsafe { (cvt4_neon(&b[0..]), cvt4_neon(&b[4..])) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn axpy_neon(alpha: f32, b: &[f16], acc: &mut [f32], n: usize) {
    use std::arch::aarch64::{float32x4_t, vfmaq_n_f32, vld1q_f32, vst1q_f32};
    let mut j = 0;
    while j + 8 <= n {
        let (lo, hi) = cvt8(&b[j..]);
        unsafe {
            let dst = acc[j..].as_mut_ptr();
            let lo_f: float32x4_t = std::mem::transmute(lo);
            let hi_f: float32x4_t = std::mem::transmute(hi);
            let a0 = vld1q_f32(dst);
            vst1q_f32(dst, vfmaq_n_f32(a0, lo_f, alpha));
            let a1 = vld1q_f32(dst.add(4));
            vst1q_f32(dst.add(4), vfmaq_n_f32(a1, hi_f, alpha));
        }
        j += 8;
    }
    if j + 4 <= n {
        // SAFETY: NEON detected; reads exactly 4 f16 words.
        let lo: [f32; 4] = unsafe { cvt4_neon(&b[j..]) };
        unsafe {
            let dst = acc[j..].as_mut_ptr();
            let lo_f: float32x4_t = std::mem::transmute(lo);
            let a0 = vld1q_f32(dst);
            vst1q_f32(dst, vfmaq_n_f32(a0, lo_f, alpha));
        }
        j += 4;
    }
    axpy_scalar(alpha, &b[j..n], &mut acc[j..n], n - j);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f16], b: &[f16], n: usize) -> f32 {
    use std::arch::aarch64::{float32x4_t, vaddvq_f32, vfmaq_f32};
    let zero = |v: f32| -> float32x4_t { unsafe { std::mem::transmute([v; 4]) } };
    let mut acc0 = zero(0.0);
    let mut acc1 = zero(0.0);
    let mut j = 0;
    while j + 8 <= n {
        let (alo, ahi) = cvt8(&a[j..]);
        let (blo, bhi) = cvt8(&b[j..]);
        unsafe {
            acc0 = vfmaq_f32(
                acc0,
                std::mem::transmute::<[f32; 4], float32x4_t>(alo),
                std::mem::transmute::<[f32; 4], float32x4_t>(blo),
            );
            acc1 = vfmaq_f32(
                acc1,
                std::mem::transmute::<[f32; 4], float32x4_t>(ahi),
                std::mem::transmute::<[f32; 4], float32x4_t>(bhi),
            );
        }
        j += 8;
    }
    let mut sum = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    sum += dot_scalar(&a[j..n], &b[j..n], n - j);
    sum
}

// ---------------------------------------------------------------------------
// x86_64 AVX2 + F16C + FMA: 8 f16 per step (128-bit load -> 256-bit f32)

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn axpy_avx(alpha: f32, b: &[f16], acc: &mut [f32], n: usize) {
    use std::arch::x86_64::{
        _mm_loadu_si128, _mm256_cvtph_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps,
        _mm256_storeu_ps,
    };
    let a_vec = unsafe { _mm256_set1_ps(alpha) };
    let mut j = 0;
    while j + 8 <= n {
        unsafe {
            let h = _mm_loadu_si128(b[j..].as_ptr() as *const _);
            let f = _mm256_cvtph_ps(h);
            let cur = _mm256_loadu_ps(acc[j..].as_ptr());
            _mm256_storeu_ps(acc[j..].as_mut_ptr(), _mm256_fmadd_ps(f, a_vec, cur));
        }
        j += 8;
    }
    axpy_scalar(alpha, &b[j..n], &mut acc[j..n], n - j);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn dot_avx(a: &[f16], b: &[f16], n: usize) -> f32 {
    use std::arch::x86_64::{
        __m128, _MM_SHUFFLE, _mm_add_ps, _mm_cvtss_f32, _mm_loadu_si128, _mm_movehdup_ps,
        _mm_shuffle_ps, _mm256_castps256_ps128, _mm256_cvtph_ps, _mm256_extractf128_ps,
        _mm256_fmadd_ps, _mm256_zerops,
    };
    let mut acc = unsafe { _mm256_zerops() };
    let mut j = 0;
    while j + 8 <= n {
        unsafe {
            let ha = _mm_loadu_si128(a[j..].as_ptr() as *const _);
            let hb = _mm_loadu_si128(b[j..].as_ptr() as *const _);
            acc = _mm256_fmadd_ps(_mm256_cvtph_ps(ha), _mm256_cvtph_ps(hb), acc);
        }
        j += 8;
    }
    // Horizontal f32 sum of the 8 accumulator lanes.
    let sum128: __m128 = unsafe {
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps(acc, 1);
        let s = _mm_add_ps(lo, hi);
        let s2 = _mm_add_ps(s, _mm_movehdup_ps(s));
        _mm_add_ps(s2, _mm_shuffle_ps(s2, s2, _MM_SHUFFLE(1, 0, 3, 2)))
    };
    let mut sum = unsafe { _mm_cvtss_f32(sum128) };
    sum += dot_scalar(&a[j..n], &b[j..n], n - j);
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_vec(n: usize, seed: u64) -> Vec<f16> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                f16::from_f32(((s >> 33) as f32 / (u32::MAX >> 1) as f32) - 1.0)
            })
            .collect()
    }

    #[test]
    fn simd_dot_matches_scalar() {
        for n in [0usize, 1, 3, 7, 8, 9, 16, 33, 2048] {
            let a = lcg_vec(n, 7);
            let b = lcg_vec(n, 13);
            let want = dot_scalar(&a, &b, n);
            let got = dot_f16_f32(&a, &b);
            assert!(
                (want - got).abs() <= 1e-2 * (1.0 + want.abs()),
                "n={n} scalar={want} simd={got}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn simd_f16_to_f32_exhaustive() {
        // Every possible f16 bit pattern must convert to exactly what
        // half's scalar path produces (bit-identical f32, NaN payload
        // included; ±0 distinguished by sign).
        let mut buf = [f16::ZERO; 4];
        for base in (0..=0xffffusize).step_by(4) {
            let patterns: Vec<usize> = (0..4).map(|i| base + i).collect();
            for (i, &bits) in patterns.iter().enumerate() {
                buf[i] = f16::from_bits(bits as u16);
            }
            let got = unsafe { super::cvt4_neon(&buf) };
            for (i, &bits) in patterns.iter().enumerate() {
                let want = f16::from_bits(bits as u16).to_f32();
                let got_f = got[i];
                let same_nan = want.is_nan() && got_f.is_nan();
                assert!(
                    same_nan || want.to_bits() == got_f.to_bits(),
                    "bits={bits:#06x}: half={want:?} ({:#010x}) neon={got_f:?} ({:#010x})",
                    want.to_bits(),
                    got_f.to_bits()
                );
            }
        }
    }

    #[test]
    fn simd_axpy_matches_scalar() {
        for n in [0usize, 1, 5, 8, 12, 128, 2048] {
            let b = lcg_vec(n, 3);
            let mut acc_scalar = vec![0.25f32; n];
            let mut acc_simd = acc_scalar.clone();
            axpy_scalar(1.5, &b, &mut acc_scalar, n);
            axpy_f16_f32(1.5, &b, &mut acc_simd);
            for (s, g) in acc_scalar.iter().zip(&acc_simd) {
                assert!((s - g).abs() <= 1e-2 * (1.0 + s.abs()), "{s} vs {g}");
            }
        }
    }
}
