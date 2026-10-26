//! Unit tests for the spec 004 shim additions used by the PP-OCRv6 conv nets
//! and the PaddleOCR-VL graph: `conv2d_ex` (groups, dilation),
//! `conv_transpose2d`, the `unary` / `binary` / `reduce` families, `where_`,
//! `clip`, `gelu_tanh`, and `quantize`.
//!
//! References are computed in f64 on the host.

use aimd_mlx::{BinaryOp, MlxArray, UnaryOp};

fn f32_arr(shape: &[usize], data: &[f32]) -> MlxArray {
    MlxArray::from_f32(shape, data).unwrap()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * scale)
        .collect()
}

#[test]
fn depthwise_dilated_conv_matches_host() {
    // NHWC [1, 6, 7, 2], depthwise 3x3 kernel, dilation 2, padding 2.
    let (h, w, c, k, d, p) = (6usize, 7usize, 2usize, 3usize, 2usize, 2usize);
    let x = ramp(h * w * c, 0.25);
    let wt = ramp(c * k * k, 0.5);
    let out = f32_arr(&[1, h, w, c], &x)
        .conv2d_ex(&f32_arr(&[c, k, k, 1], &wt), [1, 1], [p, p], [d, d], c)
        .unwrap();
    assert_eq!(out.shape().unwrap(), vec![1, h, w, c]);
    let got = out.to_f32_vec().unwrap();
    let mut want = vec![0.0f32; h * w * c];
    for oy in 0..h {
        for ox in 0..w {
            for ch in 0..c {
                let mut acc = 0.0f64;
                for ky in 0..k {
                    for kx in 0..k {
                        let iy = (oy + ky * d) as isize - p as isize;
                        let ix = (ox + kx * d) as isize - p as isize;
                        if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                            continue;
                        }
                        let xv = x[(iy as usize * w + ix as usize) * c + ch];
                        acc += f64::from(xv) * f64::from(wt[(ch * k + ky) * k + kx]);
                    }
                }
                want[(oy * w + ox) * c + ch] = acc as f32;
            }
        }
    }
    assert!(max_abs(&got, &want) < 1e-4);
}

#[test]
fn conv_transpose_stride2_matches_host() {
    // kernel 2, stride 2: every input pixel scatters into a disjoint 2x2 block.
    let (h, w, ci, co) = (2usize, 3usize, 2usize, 3usize);
    let x = ramp(h * w * ci, 0.5);
    let wt = ramp(co * 2 * 2 * ci, 0.25);
    let out = f32_arr(&[1, h, w, ci], &x)
        .conv_transpose2d(&f32_arr(&[co, 2, 2, ci], &wt), [2, 2], [0, 0], 1)
        .unwrap();
    assert_eq!(out.shape().unwrap(), vec![1, 2 * h, 2 * w, co]);
    let got = out.to_f32_vec().unwrap();
    let mut want = vec![0.0f32; 4 * h * w * co];
    for y in 0..h {
        for xx in 0..w {
            for ky in 0..2 {
                for kx in 0..2 {
                    for o in 0..co {
                        let mut acc = 0.0f64;
                        for i in 0..ci {
                            // PyTorch semantics: out[2y + ky] += x[y] * w[ky].
                            let wv = wt[((o * 2 + ky) * 2 + kx) * ci + i];
                            acc += f64::from(x[(y * w + xx) * ci + i]) * f64::from(wv);
                        }
                        want[((2 * y + ky) * 2 * w + 2 * xx + kx) * co + o] = acc as f32;
                    }
                }
            }
        }
    }
    assert!(max_abs(&got, &want) < 1e-4, "{got:?} vs {want:?}");
}

#[test]
fn unary_binary_reduce_where_clip() {
    let v = [-2.0f32, -0.5, 0.0, 0.75, 3.0, 9.0];
    let a = f32_arr(&[2, 3], &v);
    let check = |op: UnaryOp, f: fn(f64) -> f64| {
        let got = a.unary(op).unwrap().to_f32_vec().unwrap();
        let want: Vec<f32> = v.iter().map(|&x| f(f64::from(x)) as f32).collect();
        assert!(max_abs(&got, &want) < 1e-5, "{op:?}: {got:?} vs {want:?}");
    };
    check(UnaryOp::Sigmoid, |x| 1.0 / (1.0 + (-x).exp()));
    check(UnaryOp::Tanh, f64::tanh);
    check(UnaryOp::Sin, f64::sin);
    check(UnaryOp::Cos, f64::cos);
    check(UnaryOp::Negative, |x| -x);
    check(UnaryOp::Abs, f64::abs);
    check(UnaryOp::Square, |x| x * x);
    check(UnaryOp::Relu, |x| x.max(0.0));

    let b = f32_arr(&[3], &[0.0, 1.0, 2.0]);
    let mx = a.maximum(&b).unwrap().to_f32_vec().unwrap();
    assert_eq!(mx, vec![0.0, 1.0, 2.0, 0.75, 3.0, 9.0]);
    let mn = a.minimum(&b).unwrap().to_f32_vec().unwrap();
    assert_eq!(mn, vec![-2.0, -0.5, 0.0, 0.0, 1.0, 2.0]);
    let cond = a.binary(&b, BinaryOp::Greater).unwrap();
    let picked = cond.where_(&a, &b.broadcast_to(&[2, 3]).unwrap()).unwrap();
    assert_eq!(picked.to_f32_vec().unwrap(), mx);

    assert_eq!(
        a.clip(0.0, 1.0).unwrap().to_f32_vec().unwrap(),
        vec![0.0, 0.0, 0.0, 0.75, 1.0, 1.0]
    );
    let mean = a.mean_axes(&[1], false).unwrap().to_f32_vec().unwrap();
    assert!(max_abs(&mean, &[-2.5 / 3.0, 12.75 / 3.0]) < 1e-6);
    let max = a.max_axes(&[0, 1], true).unwrap();
    assert_eq!(max.shape().unwrap(), vec![1, 1]);
    assert_eq!(max.to_f32_vec().unwrap(), vec![9.0]);
}

#[test]
fn gelu_tanh_matches_formula() {
    let v = [-3.0f32, -1.0, -0.1, 0.0, 0.4, 2.5];
    let got = f32_arr(&[6], &v).gelu_tanh().unwrap().to_f32_vec().unwrap();
    let want: Vec<f32> = v
        .iter()
        .map(|&x| {
            let x = f64::from(x);
            let inner = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3));
            (0.5 * x * (1.0 + inner.tanh())) as f32
        })
        .collect();
    assert!(max_abs(&got, &want) < 1e-5);
}

#[test]
fn quantize_round_trips_through_dequantize() {
    let (rows, cols) = (4usize, 128usize);
    let w: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 37 % 101) as f32 / 101.0 - 0.5) * 0.2)
        .collect();
    let arr = f32_arr(&[rows, cols], &w);
    for bits in [4, 8] {
        let (q, scales, biases) = arr.quantize(64, bits).unwrap();
        assert_eq!(q.shape().unwrap(), vec![rows, cols * bits as usize / 32]);
        assert_eq!(scales.shape().unwrap(), vec![rows, cols / 64]);
        let back = q
            .dequantize(&scales, Some(&biases), Some(64), Some(bits))
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let step = 0.2 / f32::from(u8::try_from((1u32 << bits) - 1).unwrap());
        assert!(max_abs(&back, &w) <= step, "bits {bits}");
    }
}
