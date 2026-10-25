//! Unit tests for the WS-07 shim additions used by the Unlimited-OCR MLX
//! graph: `addmm`, `subtract`, `floor_divide`, `softmax_precise`,
//! `logsumexp`, `sum_axis`, `argsort`, `argpartition`, `take_along_axis`,
//! `put_along_axis`, `broadcast_to`, `pad`, `slice_update`, `zeros`,
//! `gather_qmm`, `async_eval` / `eval_many`.
//!
//! References are computed in f64 on the host (or, for `gather_qmm`, via
//! the already-verified `quantized_matmul` on the selected expert slice).

use aimd_mlx::{MlxArray, MlxDtype};
use half::bf16;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

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

#[test]
fn addmm_matches_bias_plus_matmul() {
    let (m, k, n) = (3usize, 5usize, 4usize);
    let mut rng = Rng::new(1);
    let a: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.next_f32()).collect();
    let c: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let out = f32_arr(&[n], &c)
        .addmm(&f32_arr(&[m, k], &a), &f32_arr(&[k, n], &b), 1.0, 1.0)
        .unwrap();
    assert_eq!(out.shape().unwrap(), vec![m, n]);
    let got = out.to_f32_vec().unwrap();
    let mut want = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = c[j] as f64;
            for t in 0..k {
                acc += a[i * k + t] as f64 * b[t * n + j] as f64;
            }
            want[i * n + j] = acc as f32;
        }
    }
    assert!(max_abs(&got, &want) < 1e-5);
}

#[test]
fn subtract_and_floor_divide() {
    let a = f32_arr(&[3], &[5.0, 1.5, -2.0]);
    let b = f32_arr(&[3], &[1.0, 0.5, 1.0]);
    assert_eq!(
        a.subtract(&b).unwrap().to_f32_vec().unwrap(),
        vec![4.0, 1.0, -3.0]
    );
    let idx = MlxArray::from_u32(&[5], &[0, 5, 6, 11, 12]).unwrap();
    let q = idx.floor_divide(&MlxArray::scalar_u32(6).unwrap()).unwrap();
    assert_eq!(q.dtype().unwrap(), MlxDtype::U32);
    assert_eq!(q.to_u32_vec().unwrap(), vec![0, 0, 1, 1, 2]);
}

#[test]
fn softmax_precise_logsumexp_and_sum() {
    let x: Vec<f32> = vec![0.5, -1.0, 2.0, 0.25, 3.0, -0.5];
    let arr = f32_arr(&[2, 3], &x);
    let sm = arr.softmax_precise(-1, true).unwrap().to_f32_vec().unwrap();
    let lse = arr.logsumexp(-1, true).unwrap();
    assert_eq!(lse.shape().unwrap(), vec![2, 1]);
    let lse_v = lse.to_f32_vec().unwrap();
    let sums = arr.sum_axis(-1, false).unwrap().to_f32_vec().unwrap();
    for r in 0..2 {
        let row = &x[r * 3..r * 3 + 3];
        let m = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
        let z: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
        for c in 0..3 {
            let want = ((row[c] as f64 - m).exp() / z) as f32;
            assert!((sm[r * 3 + c] - want).abs() < 1e-6);
        }
        assert!((lse_v[r] as f64 - (m + z.ln())).abs() < 1e-5);
        assert!((sums[r] - row.iter().sum::<f32>()).abs() < 1e-6);
    }
    // bf16 input keeps the bf16 dtype under precise softmax.
    let xb: Vec<bf16> = x.iter().map(|&v| bf16::from_f32(v)).collect();
    let smb = MlxArray::from_bf16(&[2, 3], &xb)
        .unwrap()
        .softmax_precise(-1, true)
        .unwrap();
    assert_eq!(smb.dtype().unwrap(), MlxDtype::Bf16);
}

#[test]
fn argsort_argpartition_take_along_axis() {
    let x: Vec<f32> = vec![0.3, 0.9, 0.1, 0.7, 0.5, 0.2, 0.8, 0.4];
    let arr = f32_arr(&[1, 8], &x);
    let order = arr.argsort(-1).unwrap().to_u32_vec().unwrap();
    assert_eq!(order, vec![2, 5, 0, 7, 4, 3, 6, 1]);
    // Stable on ties.
    let ties = MlxArray::from_u32(&[6], &[3, 1, 3, 1, 0, 3]).unwrap();
    assert_eq!(
        ties.argsort(0).unwrap().to_u32_vec().unwrap(),
        vec![4, 1, 3, 0, 2, 5]
    );
    // Top-3 via argpartition(kth=-3)[..., -3:].
    let part = arr.argpartition(-3, -1).unwrap();
    let top = part.slice(&[0, 5], &[1, 8], &[1, 1]).unwrap();
    let mut top_v = top.to_u32_vec().unwrap();
    top_v.sort();
    assert_eq!(top_v, vec![1, 3, 6]);
    let vals = arr.take_along_axis(&top, -1).unwrap().to_f32_vec().unwrap();
    for (i, &v) in top.to_u32_vec().unwrap().iter().zip(&vals) {
        assert_eq!(x[*i as usize], v);
    }
}

#[test]
fn put_along_axis_writes_neg_inf() {
    let x: Vec<bf16> = (0..6).map(|v| bf16::from_f32(v as f32)).collect();
    let arr = MlxArray::from_bf16(&[1, 6], &x).unwrap();
    let idx = MlxArray::from_u32(&[1, 2], &[1, 4]).unwrap();
    let neg = MlxArray::scalar_bf16(f32::NEG_INFINITY).unwrap();
    let out = arr.put_along_axis(&idx, &neg, -1).unwrap();
    let got: Vec<f32> = out
        .to_bf16_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect();
    assert_eq!(got[0], 0.0);
    assert_eq!(got[1], f32::NEG_INFINITY);
    assert_eq!(got[2], 2.0);
    assert_eq!(got[4], f32::NEG_INFINITY);
    assert_eq!(got[5], 5.0);
}

#[test]
fn broadcast_pad_zeros_and_slice_update() {
    let v = f32_arr(&[3], &[1.0, 2.0, 3.0]);
    let b = v.broadcast_to(&[2, 1, 3]).unwrap();
    assert_eq!(b.to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

    let x = f32_arr(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let p = x.pad(&[(0, 1), (1, 0)]).unwrap();
    assert_eq!(p.shape().unwrap(), vec![3, 3]);
    assert_eq!(
        p.to_f32_vec().unwrap(),
        vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0]
    );

    let z = MlxArray::zeros(&[2, 4], MlxDtype::Bf16).unwrap();
    assert_eq!(z.dtype().unwrap(), MlxDtype::Bf16);
    let upd: Vec<bf16> = [7.0f32, 8.0].iter().map(|&v| bf16::from_f32(v)).collect();
    let upd = MlxArray::from_bf16(&[1, 2], &upd).unwrap();
    let out = z.slice_update(&upd, &[1, 1], &[2, 3], &[1, 1]).unwrap();
    let got: Vec<f32> = out
        .to_bf16_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect();
    assert_eq!(got, vec![0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 0.0]);
}

#[test]
fn gather_qmm_matches_per_expert_quantized_matmul() {
    // 4 experts, out 8, in 128 (2 groups of 64), 4-bit affine.
    let (experts, out_dim, in_dim, gs, bits) = (4usize, 8usize, 128usize, 64usize, 4usize);
    let words = in_dim * bits / 32;
    let groups = in_dim / gs;
    let mut rng = Rng::new(7);
    let payload: Vec<u32> = (0..experts * out_dim * words)
        .map(|_| rng.next_u64() as u32)
        .collect();
    let scales: Vec<bf16> = (0..experts * out_dim * groups)
        .map(|_| bf16::from_f32(0.01 + 0.02 * rng.next_f32().abs()))
        .collect();
    let biases: Vec<bf16> = (0..experts * out_dim * groups)
        .map(|_| bf16::from_f32(-0.08 * rng.next_f32().abs()))
        .collect();
    let w = MlxArray::from_u32(&[experts, out_dim, words], &payload).unwrap();
    let s = MlxArray::from_bf16(&[experts, out_dim, groups], &scales).unwrap();
    let bz = MlxArray::from_bf16(&[experts, out_dim, groups], &biases).unwrap();

    // Two tokens x top-2 experts, SwitchGLU layout: x [T, 1, 1, in],
    // indices [T, 2] -> out [T, 2, 1, out].
    let tokens = 2usize;
    let xs: Vec<bf16> = (0..tokens * in_dim)
        .map(|_| bf16::from_f32(rng.next_f32()))
        .collect();
    let x = MlxArray::from_bf16(&[tokens, 1, 1, in_dim], &xs).unwrap();
    let inds = [2u32, 0, 3, 2];
    let idx = MlxArray::from_u32(&[tokens, 2], &inds).unwrap();
    let out = x
        .gather_qmm(
            &w,
            &s,
            Some(&bz),
            None,
            Some(&idx),
            true,
            Some(gs as i32),
            Some(bits as i32),
            false,
        )
        .unwrap();
    assert_eq!(out.shape().unwrap(), vec![tokens, 2, 1, out_dim]);
    let got: Vec<f32> = out
        .to_bf16_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect();

    for t in 0..tokens {
        let xt = MlxArray::from_bf16(&[1, in_dim], &xs[t * in_dim..(t + 1) * in_dim]).unwrap();
        for k in 0..2 {
            let e = inds[t * 2 + k] as usize;
            let we = w
                .slice(&[e, 0, 0], &[e + 1, out_dim, words], &[1, 1, 1])
                .unwrap();
            let we = we.reshape(&[out_dim, words]).unwrap();
            let se = s
                .slice(&[e, 0, 0], &[e + 1, out_dim, groups], &[1, 1, 1])
                .unwrap()
                .reshape(&[out_dim, groups])
                .unwrap();
            let be = bz
                .slice(&[e, 0, 0], &[e + 1, out_dim, groups], &[1, 1, 1])
                .unwrap()
                .reshape(&[out_dim, groups])
                .unwrap();
            let want = xt
                .quantized_matmul(
                    &we,
                    &se,
                    Some(&be),
                    true,
                    Some(gs as i32),
                    Some(bits as i32),
                    None,
                )
                .unwrap();
            let want: Vec<f32> = want
                .to_bf16_vec()
                .unwrap()
                .iter()
                .map(|v| v.to_f32())
                .collect();
            let off = (t * 2 + k) * out_dim;
            let d = max_abs(&got[off..off + out_dim], &want);
            let mag = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
            assert!(
                d <= mag * 1e-2 + 1e-3,
                "token {t} expert {e}: diff {d} (mag {mag})"
            );
        }
    }

    // sorted_indices path (flattened, sorted rhs indices) gives the same rows.
    let x_flat = MlxArray::from_bf16(
        &[4, 1, in_dim],
        &[&xs[..in_dim], &xs[..in_dim], &xs[in_dim..], &xs[in_dim..]].concat(),
    )
    .unwrap();
    let sorted = MlxArray::from_u32(&[4], &[0, 2, 2, 3]).unwrap();
    let out_sorted = x_flat
        .gather_qmm(
            &w,
            &s,
            Some(&bz),
            None,
            Some(&sorted),
            true,
            Some(gs as i32),
            Some(bits as i32),
            true,
        )
        .unwrap();
    assert_eq!(out_sorted.shape().unwrap(), vec![4, 1, out_dim]);
    let gs_rows: Vec<f32> = out_sorted
        .to_bf16_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect();
    // rows: (t0,e0), (t0,e2), (t1,e2), (t1,e3) <- got indices: t0:[2,0], t1:[3,2]
    let pick = |t: usize, k: usize| &got[(t * 2 + k) * out_dim..(t * 2 + k + 1) * out_dim];
    for (row, (t, k)) in [(0usize, 1usize), (0, 0), (1, 1), (1, 0)]
        .iter()
        .enumerate()
    {
        let d = max_abs(&gs_rows[row * out_dim..(row + 1) * out_dim], pick(*t, *k));
        assert!(d < 1e-2, "sorted row {row}: {d}");
    }
}

#[test]
fn async_eval_and_eval_many() {
    let a = f32_arr(&[2], &[1.0, 2.0]);
    let b = a.add(&a).unwrap();
    let c = b.multiply(&a).unwrap();
    MlxArray::async_eval(&[&b, &c]).unwrap();
    MlxArray::eval_many(&[&b, &c]).unwrap();
    assert_eq!(c.to_f32_vec().unwrap(), vec![2.0, 8.0]);
    let d = c.try_clone().unwrap();
    drop(c);
    assert_eq!(d.to_f32_vec().unwrap(), vec![2.0, 8.0]);
    aimd_mlx::clear_cache().unwrap();
}
