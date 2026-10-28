//! Unit tests for the spec 004 shim additions used by the PP-OCRv6 conv nets:
//! `conv2d_ex` (groups, dilation), `conv_transpose2d`, the `unary` / `binary`
//! / `reduce` families, `where_`, `clip`, and `quantize`, plus the fused
//! `bias_act` epilogue.
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

fn lcg(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
        })
        .collect()
}

#[test]
fn bias_act_is_bit_identical_to_unfused_graph() {
    use aimd_mlx::Activation;
    let s = |v: f32| MlxArray::scalar_f32(v).unwrap();
    let shape = [2, 5, 7, 24];
    let n: usize = shape.iter().product();
    // lcg is in [-4, 4); spread it so hard sigmoid hits both clamps.
    let y = f32_arr(
        &shape,
        &lcg(n, 4).iter().map(|v| v * 8.0).collect::<Vec<_>>(),
    );
    let b = f32_arr(&[24], &lcg(24, 5));
    let r = f32_arr(&shape, &lcg(n, 6));
    let unfused = |x: &MlxArray, act: Activation| -> MlxArray {
        match act {
            Activation::Identity => unreachable!(),
            Activation::Relu => x.maximum(&s(0.0)).unwrap(),
            Activation::HardSigmoid => x
                .add(&s(3.0))
                .unwrap()
                .maximum(&s(0.0))
                .unwrap()
                .clip(0.0, 6.0)
                .unwrap()
                .divide(&s(6.0))
                .unwrap(),
            Activation::Gelu => x.gelu().unwrap(),
        }
    };
    for act in [
        Activation::Identity,
        Activation::Relu,
        Activation::HardSigmoid,
        Activation::Gelu,
    ] {
        for (bias, res) in [
            (None, None),
            (Some(&b), None),
            (None, Some(&r)),
            (Some(&b), Some(&r)),
        ] {
            let pre = bias.map_or_else(|| y.add(&s(0.0)).unwrap(), |b| y.add(b).unwrap());
            let mut want = if act == Activation::Identity {
                pre
            } else {
                unfused(&pre, act)
            };
            if let Some(r) = res {
                want = r.add(&want).unwrap();
            }
            let got = y.bias_act(bias, act, res).unwrap();
            let (g, w) = (got.to_f32_vec().unwrap(), want.to_f32_vec().unwrap());
            let same = g.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(
                same,
                "{act:?} bias={} res={}",
                bias.is_some(),
                res.is_some()
            );
        }
    }
}
