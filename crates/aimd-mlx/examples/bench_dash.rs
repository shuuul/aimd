//! Unlimited-OCR-2 DASH decode-step latency at the decoder's shapes
//! (10 heads, head/index dim 128, block capacity 64, a 2048-row cache).
//!
//! Usage: cargo run --release -p aimd-mlx --example bench_dash
//!
//! Each call feeds its output back as the next query, so the calls run one
//! after another: the reported time is the latency a decode step pays per
//! layer (GPU time plus Metal encode). `sdpa_decode` over the same number of
//! contiguous keys is the dense baseline; `dash_decode_prologue` is the
//! fused rope + router step before it.

use std::time::Instant;

use aimd_mlx::{DashLayerConsts, DashMode, DashParams, MlxArray, MlxDtype};

const HQ: usize = 10;
const D: usize = 128;
const DI: usize = 128;
const CAP: usize = 64;
const S: usize = 2048;
const POS: usize = 1800;
const N: usize = 400;

const P: DashParams = DashParams {
    block_size: 32,
    local_blocks: 4,
    topm: 16,
    topk: 12,
    sigma: 1.0,
};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * amp).collect()
    }
}

fn arr(rng: &mut Rng, shape: &[usize], amp: f32, dtype: MlxDtype) -> MlxArray {
    let n = shape.iter().product();
    MlxArray::from_f32(shape, &rng.vec(n, amp))
        .unwrap()
        .astype(dtype)
        .unwrap()
}

fn time(name: &str, f: impl Fn(&MlxArray) -> MlxArray, q0: &MlxArray) {
    for round in 0..3 {
        let t = Instant::now();
        let mut q = q0.astype(MlxDtype::Bf16).unwrap();
        for _ in 0..N {
            q = f(&q);
        }
        q.eval().unwrap();
        if round == 2 {
            let us = t.elapsed().as_secs_f64() * 1e6 / N as f64;
            println!("{name:<28} {us:>7.1} us/call");
        }
    }
}

fn main() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let bf = MlxDtype::Bf16;
    let keys = arr(&mut rng, &[1, HQ, S, D], 1.0, bf);
    let values = arr(&mut rng, &[1, HQ, S, D], 1.0, bf);
    let idx_q = arr(&mut rng, &[1, DI], 1.0, MlxDtype::F32);
    let sums = arr(&mut rng, &[CAP, DI], 3.0, MlxDtype::F32);
    let cand = MlxArray::from_bool(&[1, CAP], &[true; CAP]).unwrap();
    let pos = MlxArray::from_i32(&[], &[POS as i32]).unwrap();
    let q0 = arr(&mut rng, &[1, HQ, 1, D], 1.0, bf);
    for a in [&keys, &values, &idx_q, &sums, &cand, &pos, &q0] {
        a.eval().unwrap();
    }
    let c = DashLayerConsts {
        idx_scale: 2.0 / (DI as f32).sqrt(),
        floor: 1e-2,
        scale: 1.0 / (D as f32).sqrt(),
    };
    for (name, mode) in [
        ("dash_decode auto", DashMode::Auto),
        ("dash_decode gather", DashMode::Gather),
    ] {
        time(
            name,
            |q| {
                q.dash_decode(&keys, &values, &idx_q, &sums, &cand, &pos, &P, &c, mode)
                    .unwrap()
            },
            &q0,
        );
    }
    // Router + rope prologue: the next query is the roped q.
    const H: usize = 1280;
    let x = arr(&mut rng, &[1, 1, H], 1.0, bf);
    let k = arr(&mut rng, &[1, HQ, 1, D], 1.0, bf);
    let (qw, kw) = (
        arr(&mut rng, &[DI, H], 0.05, bf),
        arr(&mut rng, &[DI, H], 0.05, bf),
    );
    let nw = arr(&mut rng, &[DI], 0.1, bf);
    let cos = arr(&mut rng, &[1, 1, 1, D], 1.0, bf);
    let sin = arr(&mut rng, &[1, 1, 1, D], 1.0, bf);
    time(
        "dash_decode_prologue",
        |q| {
            x.dash_decode_prologue(q, &k, &qw, &kw, &nw, &nw, &cos, &sin, 1e-6)
                .unwrap()
                .0
        },
        &q0,
    );
    let n = (P.topk * P.block_size + P.local_blocks * P.block_size) as usize;
    let k512 = keys
        .slice(&[0, 0, 0, 0], &[1, HQ, n, D], &[1, 1, 1, 1])
        .unwrap();
    let v512 = values
        .slice(&[0, 0, 0, 0], &[1, HQ, n, D], &[1, 1, 1, 1])
        .unwrap();
    let (k512, v512) = (k512.astype(bf).unwrap(), v512.astype(bf).unwrap());
    k512.eval().unwrap();
    v512.eval().unwrap();
    time(
        "sdpa_decode 512 keys",
        |q| q.sdpa_decode(&k512, &v512, c.scale).unwrap(),
        &q0,
    );
    let kp = keys
        .slice(&[0, 0, 0, 0], &[1, HQ, POS + 1, D], &[1, 1, 1, 1])
        .unwrap();
    let vp = values
        .slice(&[0, 0, 0, 0], &[1, HQ, POS + 1, D], &[1, 1, 1, 1])
        .unwrap();
    time(
        "sdpa_decode 1801 keys (v1)",
        |q| q.sdpa_decode(&kp, &vp, c.scale).unwrap(),
        &q0,
    );
}
