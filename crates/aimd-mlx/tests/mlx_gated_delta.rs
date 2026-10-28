//! Qwen3.5 decoder ops (spec 009): `gated_delta_gates`, `gated_delta`
//! (Metal kernel or stock-op loop), and `conv1d`, against
//! f64 host references.

use aimd_mlx::{MlxArray, MlxDtype};
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

fn bf16_round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| bf16::from_f32(x).to_f32()).collect()
}

fn bf16_arr(shape: &[usize], data: &[f32]) -> MlxArray {
    let d: Vec<bf16> = data.iter().map(|&x| bf16::from_f32(x)).collect();
    MlxArray::from_bf16(shape, &d).unwrap()
}

#[test]
fn gates_match_host_formula() {
    let h = 16;
    let mut rng = Rng::new(7);
    let a = bf16_round(&(0..2 * h).map(|_| rng.next_f32() * 4.0).collect::<Vec<_>>());
    let b = bf16_round(&(0..2 * h).map(|_| rng.next_f32() * 4.0).collect::<Vec<_>>());
    let a_log = bf16_round(&(0..h).map(|_| rng.next_f32() * 2.0).collect::<Vec<_>>());
    let dt = bf16_round(&(0..h).map(|_| rng.next_f32()).collect::<Vec<_>>());
    let (g, beta) = bf16_arr(&[1, 2, h], &a)
        .gated_delta_gates(
            &bf16_arr(&[1, 2, h], &b),
            &bf16_arr(&[h], &a_log),
            &bf16_arr(&[h], &dt),
        )
        .unwrap();
    assert_eq!(g.dtype().unwrap(), MlxDtype::F32);
    assert_eq!(beta.dtype().unwrap(), MlxDtype::Bf16);
    let g = g.to_f32_vec().unwrap();
    let beta = beta.astype(MlxDtype::F32).unwrap().to_f32_vec().unwrap();
    for i in 0..2 * h {
        let x = bf16::from_f32(a[i] + dt[i % h]).to_f32() as f64;
        let sp = x.exp().ln_1p();
        let want = (-(a_log[i % h] as f64).exp() * sp).exp();
        // softplus runs in bf16, so allow its rounding.
        assert!(
            (g[i] as f64 - want).abs() <= 1e-2 * want.max(1e-3),
            "g[{i}] {} vs {want}",
            g[i]
        );
        let want_b = 1.0 / (1.0 + (-(b[i] as f64)).exp());
        assert!((beta[i] as f64 - want_b).abs() < 1e-2, "beta[{i}]");
    }
}

/// Host recurrence in f64: state `[H, Dv, Dk]`.
#[allow(clippy::too_many_arguments)]
fn host_delta(
    t: usize,
    h: usize,
    dk: usize,
    dv: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f64],
) -> Vec<f64> {
    let mut y = vec![0f64; t * h * dv];
    for ti in 0..t {
        for hi in 0..h {
            let qk = (ti * h + hi) * dk;
            let gi = g[ti * h + hi] as f64;
            let bi = beta[ti * h + hi] as f64;
            for vi in 0..dv {
                let row = &mut state[(hi * dv + vi) * dk..(hi * dv + vi + 1) * dk];
                let mut kv = 0f64;
                for j in 0..dk {
                    row[j] *= gi;
                    kv += row[j] * k[qk + j] as f64;
                }
                let delta = (v[(ti * h + hi) * dv + vi] as f64 - kv) * bi;
                let mut out = 0f64;
                for j in 0..dk {
                    row[j] += k[qk + j] as f64 * delta;
                    out += row[j] * q[qk + j] as f64;
                }
                y[(ti * h + hi) * dv + vi] = out;
            }
        }
    }
    y
}

#[test]
fn gated_delta_matches_host_recurrence_across_calls() {
    // Prefill of 5 steps, then a 1-step decode carrying the state: the
    // second call must continue the same recurrence.
    let (h, dk, dv) = (2usize, 64usize, 32usize);
    let mut rng = Rng::new(42);
    let mut state = vec![0f64; h * dv * dk];
    let mut st = MlxArray::zeros(&[1, h, dv, dk], MlxDtype::F32).unwrap();
    for t in [5usize, 1] {
        let scale = (dk as f32).powf(-0.5);
        let q = bf16_round(
            &(0..t * h * dk)
                .map(|_| rng.next_f32() * scale)
                .collect::<Vec<_>>(),
        );
        let k = bf16_round(
            &(0..t * h * dk)
                .map(|_| rng.next_f32() * scale)
                .collect::<Vec<_>>(),
        );
        let v = bf16_round(&(0..t * h * dv).map(|_| rng.next_f32()).collect::<Vec<_>>());
        let g: Vec<f32> = (0..t * h).map(|_| 0.75 + 0.2 * rng.next_f32()).collect();
        let beta = bf16_round(
            &(0..t * h)
                .map(|_| 0.5 + 0.4 * rng.next_f32())
                .collect::<Vec<_>>(),
        );
        let (y, ns) = bf16_arr(&[1, t, h, dk], &q)
            .gated_delta(
                &bf16_arr(&[1, t, h, dk], &k),
                &bf16_arr(&[1, t, h, dv], &v),
                &MlxArray::from_f32(&[1, t, h], &g).unwrap(),
                &bf16_arr(&[1, t, h], &beta),
                &st,
            )
            .unwrap();
        assert_eq!(y.shape().unwrap(), vec![1, t, h, dv]);
        assert_eq!(y.dtype().unwrap(), MlxDtype::Bf16);
        let want = host_delta(t, h, dk, dv, &q, &k, &v, &g, &beta, &mut state);
        let got = y.astype(MlxDtype::F32).unwrap().to_f32_vec().unwrap();
        for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
            assert!(
                (a as f64 - b).abs() <= 1e-2 * b.abs().max(0.05),
                "y[{i}] {a} vs {b}"
            );
        }
        let got_state = ns.to_f32_vec().unwrap();
        for (i, (&a, &b)) in got_state.iter().zip(&state).enumerate() {
            assert!(
                (a as f64 - b).abs() <= 1e-4 * b.abs().max(1.0),
                "state[{i}]"
            );
        }
        st = ns;
    }
}

#[test]
fn depthwise_conv1d_matches_host() {
    let (l, c, kk) = (6usize, 5usize, 4usize);
    let mut rng = Rng::new(3);
    let x: Vec<f32> = (0..l * c).map(|_| rng.next_f32()).collect();
    let w: Vec<f32> = (0..c * kk).map(|_| rng.next_f32()).collect();
    let out = MlxArray::from_f32(&[1, l, c], &x)
        .unwrap()
        .conv1d(&MlxArray::from_f32(&[c, kk, 1], &w).unwrap(), 1, 0, 1, c)
        .unwrap();
    assert_eq!(out.shape().unwrap(), vec![1, l - kk + 1, c]);
    let got = out.to_f32_vec().unwrap();
    for t in 0..l - kk + 1 {
        for ch in 0..c {
            let want: f32 = (0..kk).map(|j| x[(t + j) * c + ch] * w[ch * kk + j]).sum();
            assert!((got[t * c + ch] - want).abs() < 1e-5);
        }
    }
}
