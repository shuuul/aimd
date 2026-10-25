//! Per-symbol unit tests for the WS-11b shim extensions (`mlx_conv2d`,
//! `mlx_sdpa`, `mlx_layer_norm`, `mlx_gelu`, `mlx_slice`, `mlx_concatenate`,
//! `mlx_take`, `mlx_dequantize`, `mlx_rope`, scalar arrays).
//!
//! Small hand-checked arrays; references are computed in `f64` in Rust.
//! MLX runs on Metal, so comparisons carry small fp tolerances.

use aimd_mlx::{MlxArray, MlxDtype, SdpaMask};
use aimd_weights::{QuantizedTensor, ScaleDtype, dequantize_affine};
use half::bf16;

/// xorshift64* deterministic rng (no rand dev-dependency).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((x >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

#[test]
fn conv2d_matches_naive_reference_and_promotes() {
    // f32 input (the tower mel), bf16 weight (the checkpoint) — mlx must
    // promote to f32 output, like mlx-audio's nn.Conv2d path.
    let (h, w, c, o, k) = (5usize, 7usize, 2usize, 3usize, 3usize);
    let mut rng = Rng::new(0xC0);
    let input: Vec<f32> = (0..h * w * c).map(|_| rng.next_f32()).collect();
    let weight: Vec<bf16> = (0..o * k * k * c)
        .map(|_| bf16::from_f32(rng.next_f32()))
        .collect();
    let bias_free_input = MlxArray::from_f32(&[1, h, w, c], &input).unwrap();
    let weight_arr = MlxArray::from_bf16(&[o, k, k, c], &weight).unwrap();

    // Tower geometry: 3x3, stride 2, padding 1 on both axes.
    let out = bias_free_input.conv2d(&weight_arr, [2, 2], [1, 1]).unwrap();
    let (h_out, w_out) = ((h - 1) / 2 + 1, (w - 1) / 2 + 1);
    assert_eq!(out.shape().unwrap(), vec![1, h_out, w_out, o]);
    assert_eq!(out.dtype().unwrap(), MlxDtype::F32);
    let got = out.to_f32_vec().unwrap();

    // Naive NHWC reference: out[i, j, ch] = sum weight[ch, di, dj, cc] * x[2i+di-1, 2j+dj-1, cc].
    let x_at = |i: i64, j: i64, cc: usize| -> f32 {
        if i < 0 || j < 0 || i as usize >= h || j as usize >= w {
            0.0
        } else {
            input[((i as usize) * w + j as usize) * c + cc]
        }
    };
    let mut worst = 0f32;
    for i in 0..h_out {
        for j in 0..w_out {
            for ch in 0..o {
                let mut acc = 0f32;
                for di in 0..k {
                    for dj in 0..k {
                        for cc in 0..c {
                            acc += x_at(
                                2 * i as i64 + di as i64 - 1,
                                2 * j as i64 + dj as i64 - 1,
                                cc,
                            ) * weight[((ch * k + di) * k + dj) * c + cc].to_f32();
                        }
                    }
                }
                worst = worst.max((got[(i * w_out + j) * o + ch] - acc).abs());
            }
        }
    }
    assert!(worst < 1e-5, "conv2d worst abs diff {worst}");
}

#[test]
fn sdpa_matches_naive_softmax_with_masks_and_gqa() {
    let (batch, q_heads, kv_heads, seq, head_dim) = (1usize, 2usize, 2usize, 4usize, 4usize);
    let mut rng = Rng::new(0x5D);
    let q: Vec<f32> = (0..batch * q_heads * seq * head_dim)
        .map(|_| rng.next_f32())
        .collect();
    let k: Vec<f32> = (0..batch * kv_heads * seq * head_dim)
        .map(|_| rng.next_f32())
        .collect();
    let v: Vec<f32> = k.clone();
    let scale = 1.0f32;

    let naive = |mask: Option<&[Vec<f32>]>, gqa_kv_heads: usize| -> Vec<f32> {
        let mut out = vec![0f32; batch * q_heads * seq * head_dim];
        for b in 0..batch {
            for h in 0..q_heads {
                let kv_h = if gqa_kv_heads == q_heads {
                    h
                } else {
                    h * gqa_kv_heads / q_heads
                };
                for i in 0..seq {
                    let mut scores = vec![0f64; seq];
                    for (j, score) in scores.iter_mut().enumerate() {
                        let mut dot = 0f64;
                        for d in 0..head_dim {
                            dot += q[b * q_heads * seq * head_dim
                                + h * seq * head_dim
                                + i * head_dim
                                + d] as f64
                                * k[b * kv_heads * seq * head_dim
                                    + kv_h * seq * head_dim
                                    + j * head_dim
                                    + d] as f64;
                        }
                        *score = dot * scale as f64 + mask.map_or(0.0, |m| m[i][j] as f64);
                    }
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let mut sum = 0f64;
                    for s in &mut scores {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    for d in 0..head_dim {
                        let mut acc = 0f64;
                        for j in 0..seq {
                            acc += (scores[j] / sum)
                                * v[b * kv_heads * seq * head_dim
                                    + kv_h * seq * head_dim
                                    + j * head_dim
                                    + d] as f64;
                        }
                        out[b * q_heads * seq * head_dim + h * seq * head_dim + i * head_dim + d] =
                            acc as f32;
                    }
                }
            }
        }
        out
    };

    let q_arr = MlxArray::from_f32(&[batch, q_heads, seq, head_dim], &q).unwrap();
    let k_arr = MlxArray::from_f32(&[batch, kv_heads, seq, head_dim], &k).unwrap();
    let v_arr = MlxArray::from_f32(&[batch, kv_heads, seq, head_dim], &v).unwrap();

    // No mask.
    let out = q_arr
        .scaled_dot_product_attention(&k_arr, &v_arr, scale, SdpaMask::None)
        .unwrap();
    let got = out.to_f32_vec().unwrap();
    for (a, r) in got.iter().zip(naive(None, kv_heads)) {
        assert!((a - r).abs() < 1e-5, "{a} vs {r}");
    }

    // Additive block mask (rows 0..2 and 2..4 attend only within blocks).
    let mut mask = vec![vec![-1e9f32; seq]; seq];
    for (i, row) in mask.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            if i / 2 == j / 2 {
                *cell = 0.0;
            }
        }
    }
    let mask_flat: Vec<f32> = mask.iter().flatten().copied().collect();
    let mask_arr = MlxArray::from_f32(&[seq, seq], &mask_flat).unwrap();
    let out = q_arr
        .scaled_dot_product_attention(&k_arr, &v_arr, scale, SdpaMask::Array(&mask_arr))
        .unwrap();
    let got = out.to_f32_vec().unwrap();
    for (a, r) in got.iter().zip(naive(Some(&mask), kv_heads)) {
        assert!((a - r).abs() < 1e-5, "{a} vs {r}");
    }
}

#[test]
fn sdpa_causal_mode_masks_future_keys() {
    // With causal masking, changing FUTURE keys must not change earlier
    // query rows (a cheap semantic lock without a numeric reference).
    let (batch, heads, seq, head_dim) = (1usize, 1usize, 4usize, 2usize);
    let q_arr = MlxArray::from_f32(
        &[batch, heads, seq, head_dim],
        &[1.0, 0.0, 0.5, 0.5, -0.3, 0.8, 0.1, -0.9],
    )
    .unwrap();
    let k1 = MlxArray::from_f32(
        &[batch, heads, seq, head_dim],
        &[0.2, 0.4, -0.6, 0.1, 0.9, -0.2, 0.3, 0.7],
    )
    .unwrap();
    let k2 = MlxArray::from_f32(
        &[batch, heads, seq, head_dim],
        &[0.2, 0.4, -0.6, 0.1, 0.9, -0.2, 9.9, -9.2], // only the LAST key perturbed
    )
    .unwrap();
    let v = MlxArray::from_f32(
        &[batch, heads, seq, head_dim],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();
    let out1 = q_arr
        .scaled_dot_product_attention(&k1, &v, 1.0, SdpaMask::Causal)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let out2 = q_arr
        .scaled_dot_product_attention(&k2, &v, 1.0, SdpaMask::Causal)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    // Rows 0..2 depend only on keys 0..=i, so they must be identical; the
    // last row sees the perturbed keys and must differ.
    assert_eq!(&out1[..6], &out2[..6]);
    assert_ne!(out1[6..], out2[6..]);
}

#[test]
fn layer_norm_matches_hand_values_and_promotes() {
    // Biased LayerNorm over the last axis with bf16 affine on f32 input
    // (the audio tower's actual dtype combination).
    let x = [3.0f32, 4.0, -1.0, 1.0];
    let w = [bf16::from_f32(2.0), bf16::from_f32(0.5)];
    let b = [bf16::from_f32(1.0), bf16::from_f32(-1.0)];
    let x_arr = MlxArray::from_f32(&[2, 2], &x).unwrap();
    let w_arr = MlxArray::from_bf16(&[2], &w).unwrap();
    let b_arr = MlxArray::from_bf16(&[2], &b).unwrap();
    let out = x_arr.layer_norm(Some(&w_arr), Some(&b_arr), 1e-5).unwrap();
    assert_eq!(out.dtype().unwrap(), MlxDtype::F32);
    let got = out.to_f32_vec().unwrap();
    for row in 0..2 {
        let r = &x[row * 2..row * 2 + 2];
        let mean = (r[0] + r[1]) / 2.0;
        let var = ((r[0] - mean) * (r[0] - mean) + (r[1] - mean) * (r[1] - mean)) / 2.0;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for j in 0..2 {
            let expected = (r[j] - mean) * inv * w[j].to_f32() + b[j].to_f32();
            assert!(
                (got[row * 2 + j] - expected).abs() < 1e-5,
                "[{row},{j}]: {} vs {expected}",
                got[row * 2 + j]
            );
        }
    }
}

#[test]
fn gelu_matches_erf_reference_and_keeps_dtype() {
    let values = [0.0f32, 1.0, -1.0, 0.5, 2.5, -3.0];
    let x = MlxArray::from_f32(&[values.len()], &values).unwrap();
    let out = x.gelu().unwrap();
    assert_eq!(out.dtype().unwrap(), MlxDtype::F32);
    let got = out.to_f32_vec().unwrap();
    for (&v, &g) in values.iter().zip(&got) {
        let expected = v as f64 * (1.0 + erf(v as f64 / std::f64::consts::SQRT_2)) / 2.0;
        assert!(
            (g - expected as f32).abs() < 1e-6,
            "gelu({v}): {g} vs {expected}"
        );
    }

    // bf16 input stays bf16 (scalar constants use the input dtype).
    let x16 = MlxArray::from_bf16(&[2], &[bf16::from_f32(1.0), bf16::from_f32(-2.0)]).unwrap();
    let out16 = x16.gelu().unwrap();
    assert_eq!(out16.dtype().unwrap(), MlxDtype::Bf16);
    let g16 = out16.to_bf16_vec().unwrap();
    let expected = (1.0f64 * (1.0 + erf(1.0f64 / std::f64::consts::SQRT_2)) / 2.0) as f32;
    assert!((g16[0].to_f32() - expected).abs() < 1e-2);
}

/// erf via the standard library is not available on f32/f64 in no_std-less
/// profiles here; `libm`-free approximation is unnecessary — use the
/// identity through the f64 libm that std does provide on macOS.
fn erf(x: f64) -> f64 {
    // std::f64::erf is unstable; use a high-accuracy rational/series mix
    // good to ~1e-12 over the test range.
    if x == 0.0 {
        return 0.0;
    }
    if x < 0.0 {
        return -erf(-x);
    }
    if x < 3.0 {
        let x2 = x * x;
        let mut term = x;
        let mut sum = x;
        let mut n = 1.0f64;
        loop {
            term *= -x2 / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() <= 1e-17 * sum.abs() || n > 100.0 {
                break;
            }
            n += 1.0;
        }
        return sum * (2.0 / std::f64::consts::PI.sqrt());
    }
    let z = x;
    let mut tail = 0f64;
    for k in (1..=64).rev() {
        tail = (k as f64 / 2.0) / (z + tail);
    }
    1.0 - (-x * x).exp() / (std::f64::consts::PI.sqrt() * (z + tail))
}

#[test]
fn slice_selects_rows_and_materializes_noncontiguous_views() {
    // Contiguous slice: rows 1..3 of a [3, 2] array.
    let x = MlxArray::from_f32(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rows = x.slice(&[1, 0], &[3, 2], &[1, 1]).unwrap();
    assert_eq!(rows.shape().unwrap(), vec![2, 2]);
    assert_eq!(rows.to_f32_vec().unwrap(), vec![3.0, 4.0, 5.0, 6.0]);

    // Non-contiguous: transpose first (lazy strided view), then slice —
    // readback must materialize (the WS-11a strided-readback SIGSEGV).
    // x = [[1,2],[3,4],[5,6]]; transpose -> [[1,3,5],[2,4,6]]; the slice is
    // column 1 of the transpose: [3, 4].
    let t = x.transpose(&[]).unwrap(); // [2, 3]
    let col = t.slice(&[0, 1], &[2, 2], &[1, 1]).unwrap();
    assert_eq!(col.to_f32_vec().unwrap(), vec![3.0, 4.0]);

    // Last row of a rank-3 array (the decoder logits path).
    let y = MlxArray::from_f32(&[2, 1, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let last = y.slice(&[1, 0, 0], &[2, 1, 2], &[1, 1, 1]).unwrap();
    assert_eq!(last.to_f32_vec().unwrap(), vec![3.0, 4.0]);
}

#[test]
fn concatenate_appends_along_axis() {
    let a = MlxArray::from_f32(&[1, 2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = MlxArray::from_f32(&[1, 1, 2], &[5.0, 6.0]).unwrap();
    let out = MlxArray::concatenate(&[&a, &b], 1).unwrap();
    assert_eq!(out.shape().unwrap(), vec![1, 3, 2]);
    assert_eq!(
        out.to_f32_vec().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn take_gathers_rows_along_axis() {
    let x = MlxArray::from_f32(&[4, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let idx = MlxArray::from_u32(&[3], &[2, 0, 3]).unwrap();
    let out = x.take(&idx, 0).unwrap();
    assert_eq!(out.shape().unwrap(), vec![3, 2]);
    assert_eq!(
        out.to_f32_vec().unwrap(),
        vec![5.0, 6.0, 1.0, 2.0, 7.0, 8.0]
    );
}

#[test]
fn dequantize_matches_affine_reference() {
    // 4-bit, group 32 (a group size the bundled mlx.metallib ships kernels
    // for — group sizes outside {32, 64, 128} fail to load a Metal kernel),
    // one row of 64 values = two groups, packed into eight u32 words. Nibble
    // packing (mlx affine): low nibbles first — value v at position p sits
    // in nibble p. Values 0..15 repeated: words alternate
    // 0x76543210 / 0xFEDCBA98.
    let payload: Vec<u8> = [
        0x7654_3210u32,
        0xFEDC_BA98,
        0x7654_3210,
        0xFEDC_BA98,
        0x7654_3210,
        0xFEDC_BA98,
        0x7654_3210,
        0xFEDC_BA98,
    ]
    .iter()
    .flat_map(|w| w.to_le_bytes())
    .collect();
    let half = |v: f32| half::f16::from_f32(v).to_bits();
    let q = QuantizedTensor {
        bits: 4,
        group_size: 32,
        shape: vec![1, 64],
        scale_dtype: ScaleDtype::F16,
        payload,
        scales: vec![half(0.5), half(2.0)],
        biases: vec![half(0.25), half(-1.0)],
    };
    let reference = dequantize_affine(&q).unwrap();

    let w = MlxArray::from_bytes(&q.payload, MlxDtype::U32, &[1, 8]).unwrap();
    let scales = MlxArray::from_f16(
        &[1, 2],
        &q.scales
            .iter()
            .map(|&b| half::f16::from_bits(b))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let biases = MlxArray::from_f16(
        &[1, 2],
        &q.biases
            .iter()
            .map(|&b| half::f16::from_bits(b))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let out = w
        .dequantize(&scales, Some(&biases), Some(32), Some(4))
        .unwrap();
    assert_eq!(out.shape().unwrap(), vec![1, 64]);
    assert_eq!(out.dtype().unwrap(), MlxDtype::F16);
    let got = out.to_f16_vec().unwrap();
    for (p, (a, &r)) in got.iter().zip(&reference.data).enumerate() {
        let r = half::f16::from_bits(r);
        assert!(
            (a.to_f32() - r.to_f32()).abs() < 1e-6,
            "[{p}]: {} vs {r}",
            a.to_f32()
        );
    }
}

#[test]
fn rope_matches_neox_reference() {
    // NeoX rotate-half: x = [first half, second half]; for position p and
    // inv_i = base^(-2i/d): out_i = x_i cos(p inv_i) - x_{i+h} sin(p inv_i),
    // out_{i+h} = x_{i+h} cos(p inv_i) + x_i sin(p inv_i).
    let (heads, seq, head_dim, offset, base) = (1usize, 3usize, 4usize, 2i32, 1e6f64);
    let mut rng = Rng::new(0x4E);
    let x: Vec<f32> = (0..heads * seq * head_dim)
        .map(|_| rng.next_f32())
        .collect();
    let arr = MlxArray::from_f32(&[1, heads, seq, head_dim], &x).unwrap();
    let out = arr.rope(head_dim, base, 1.0, offset).unwrap();
    let got = out.to_f32_vec().unwrap();

    let half = head_dim / 2;
    for s in 0..seq {
        let p = (offset as usize + s) as f64;
        for h in 0..heads {
            let base_idx = h * seq * head_dim + s * head_dim;
            for i in 0..half {
                let inv = base.powf(-2.0 * i as f64 / head_dim as f64);
                let angle = p * inv;
                let (cos, sin) = (angle.cos(), angle.sin());
                let (xi, xh) = (x[base_idx + i] as f64, x[base_idx + half + i] as f64);
                assert!(
                    (got[base_idx + i] as f64 - (xi * cos - xh * sin)).abs() < 1e-5,
                    "first half [{s},{i}]"
                );
                assert!(
                    (got[base_idx + half + i] as f64 - (xh * cos + xi * sin)).abs() < 1e-5,
                    "second half [{s},{i}]"
                );
            }
        }
    }
}

#[test]
fn argmax_returns_first_max_index() {
    let x = MlxArray::from_f32(&[2, 4], &[1.0, 3.0, 3.0, -2.0, -1.0, 0.5, 0.5, 2.0]).unwrap();
    let out = x.argmax(-1, false).unwrap();
    assert_eq!(out.dtype().unwrap(), MlxDtype::U32);
    assert_eq!(out.to_u32_vec().unwrap(), vec![1, 3]);
    // keepdims keeps the axis as size 1.
    let kept = x.argmax(-1, true).unwrap();
    assert_eq!(kept.shape().unwrap(), vec![2, 1]);
}

#[test]
fn scalar_f32_broadcasts_over_any_shape() {
    let x = MlxArray::from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let two = MlxArray::scalar_f32(2.5).unwrap();
    let out = x.multiply(&two).unwrap();
    assert_eq!(out.shape().unwrap(), vec![2, 2]);
    assert_eq!(out.to_f32_vec().unwrap(), vec![2.5, 5.0, 7.5, 10.0]);
}
