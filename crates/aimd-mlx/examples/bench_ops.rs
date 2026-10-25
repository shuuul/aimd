//! Per-kernel decode-step microbenchmarks at the Qwen3-ASR-1.7B and
//! Unlimited-OCR shapes (WS-08 profiling).
//!
//! Usage: cargo run --release -p aimd-mlx --example bench_ops
//!
//! Each op runs as a dependent chain (every output feeds the next input), so
//! the kernels execute one after another. The reported time is
//! `eval(chain) / n`: GPU time plus Metal encode per kernel, the same cost a
//! decode step pays. Quantized weights are random 4-bit (group 64, bf16
//! scales); kernel time does not depend on the values. `GB/s` counts the
//! weight/KV bytes the kernel must read, against the measured copy roofline.

use std::time::Instant;

use aimd_mlx::{MlxArray, MlxDtype, SdpaMask};
use half::bf16;

const GROUP: usize = 64;
const BITS: usize = 4;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

/// Random 4-bit affine triple with logical shape `[experts * rows, cols]`.
struct Quant {
    w: MlxArray,
    s: MlxArray,
    b: MlxArray,
    bytes: usize,
}

fn quant(rng: &mut Rng, experts: usize, rows: usize, cols: usize) -> Quant {
    let r = experts * rows;
    let words = cols * BITS / 32;
    let groups = cols / GROUP;
    let scale = bf16::from_f32(0.002).to_bits();
    let bias = bf16::from_f32(-0.01).to_bits();
    let (w, s, b) = MlxArray::quantized_weight(
        &rng.bytes(r * words * 4),
        &[r, words],
        &vec![scale; r * groups],
        &vec![bias; r * groups],
        &[r, groups],
        MlxDtype::Bf16,
    )
    .unwrap();
    let (w, s, b) = if experts > 1 {
        (
            w.reshape(&[experts, rows, words]).unwrap(),
            s.reshape(&[experts, rows, groups]).unwrap(),
            b.reshape(&[experts, rows, groups]).unwrap(),
        )
    } else {
        (w, s, b)
    };
    MlxArray::eval_many(&[&w, &s, &b]).unwrap();
    Quant {
        w,
        s,
        b,
        bytes: r * words * 4 + 2 * r * groups * 2,
    }
}

fn qmm(x: &MlxArray, q: &Quant) -> MlxArray {
    x.quantized_matmul(
        &q.w,
        &q.s,
        Some(&q.b),
        true,
        Some(GROUP as i32),
        Some(BITS as i32),
        None,
    )
    .unwrap()
}

fn bf16_array(rng: &mut Rng, shape: &[usize]) -> MlxArray {
    let n: usize = shape.iter().product();
    let v: Vec<bf16> = (0..n)
        .map(|_| bf16::from_f32((rng.next() % 2000) as f32 / 1000.0 - 1.0))
        .collect();
    let a = MlxArray::from_bf16(shape, &v).unwrap();
    a.eval().unwrap();
    a
}

/// Best-of-5 `eval` time per link of an `n`-link dependent chain, in µs.
/// `step(i, x)` gets the link index so it can rotate through a weight pool.
fn chain(n: usize, x0: &MlxArray, step: impl Fn(usize, &MlxArray) -> MlxArray) -> f64 {
    let mut best = f64::MAX;
    for round in 0..6 {
        let mut x = x0.try_clone().unwrap();
        for i in 0..n {
            x = step(i, &x);
        }
        let t = Instant::now();
        x.eval().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        if round > 0 {
            best = best.min(us);
        }
    }
    aimd_mlx::clear_cache().unwrap();
    best
}

/// Distinct copies so a chain's working set (~96 MB) overflows the system
/// level cache, like a real decode step walking every layer once.
const POOL_BYTES: usize = 96 << 20;

fn pool_len(bytes_each: usize) -> usize {
    POOL_BYTES.div_ceil(bytes_each).clamp(1, 64)
}

fn quant_pool(rng: &mut Rng, experts: usize, rows: usize, cols: usize) -> Vec<Quant> {
    let first = quant(rng, experts, rows, cols);
    let n = pool_len(first.bytes);
    let mut v = vec![first];
    for _ in 1..n {
        v.push(quant(rng, experts, rows, cols));
    }
    v
}

fn bf16_pool(rng: &mut Rng, shape: &[usize]) -> Vec<MlxArray> {
    let bytes = shape.iter().product::<usize>() * 2;
    (0..pool_len(bytes))
        .map(|_| bf16_array(rng, shape))
        .collect()
}

fn row(name: &str, us: f64, bytes: usize, roof: f64) {
    if bytes == 0 {
        println!("{name:<46} {us:>8.1} µs");
    } else {
        let gbs = bytes as f64 / us / 1e3;
        println!(
            "{name:<46} {us:>8.1} µs  {:>7.2} MB  {gbs:>6.1} GB/s  {:>4.0}% roof",
            bytes as f64 / 1e6,
            100.0 * gbs / roof
        );
    }
}

fn main() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    println!(
        "device: {}",
        if aimd_mlx::device_is_cpu().unwrap() {
            "cpu"
        } else {
            "gpu"
        }
    );

    // Read roofline: full reduction over 256 MiB of bf16.
    let n = 128 << 20;
    let a = bf16_array(&mut rng, &[n]);
    let mut best = f64::MAX;
    for _ in 0..6 {
        let c = a.sum_axis(0, false).unwrap();
        let t = Instant::now();
        c.eval().unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    let roof = (n * 2) as f64 / best / 1e9;
    println!("read roofline (bf16 sum over 256 MiB): {roof:.1} GB/s");
    drop(a);
    aimd_mlx::clear_cache().unwrap();

    let x = bf16_array(&mut rng, &[1, 2048]);
    row(
        "dispatch floor: add [1,2048]",
        chain(2000, &x, |_, x| x.add(x).unwrap()),
        0,
        roof,
    );

    println!("\n-- Qwen3-ASR-1.7B decode step (28 layers) --");
    let h = 2048;
    let x = bf16_array(&mut rng, &[1, h]);
    let sq = quant_pool(&mut rng, 1, h, h);
    row(
        "qmv 2048->2048 (q, o)",
        chain(400, &x, |i, x| qmm(x, &sq[i % sq.len()])),
        sq[0].bytes,
        roof,
    );
    drop(sq);
    let kv = quant_pool(&mut rng, 1, 1024, h);
    let kv_back = quant_pool(&mut rng, 1, h, 1024);
    let us = chain(300, &x, |i, x| {
        qmm(&qmm(x, &kv[i % kv.len()]), &kv_back[i % kv_back.len()])
    });
    row(
        "qmv 2048->1024 (k, v; from a pair / 2)",
        us / 2.0,
        kv[0].bytes,
        roof,
    );
    drop((kv, kv_back));
    let fused = quant_pool(&mut rng, 1, 4096, h);
    row(
        "qmv 2048->4096 (fused q|k|v) + same add tail",
        chain(300, &x, |i, x| {
            let y = qmm(x, &fused[i % fused.len()]);
            let q = y.slice(&[0, 0], &[1, h], &[1, 1]).unwrap();
            let kv = y.slice(&[0, h], &[1, 2 * h], &[1, 1]).unwrap();
            q.add(&kv).unwrap()
        }),
        fused[0].bytes,
        roof,
    );
    drop(fused);
    // The decoder issues q, k and v from the same input; MLX dispatches the
    // three independent kernels concurrently.
    let qs = quant_pool(&mut rng, 1, h, h);
    let ks = quant_pool(&mut rng, 1, 1024, h);
    let vs = quant_pool(&mut rng, 1, 1024, h);
    row(
        "qmv q + k + v from one input (concurrent)",
        chain(300, &x, |i, x| {
            let q = qmm(x, &qs[i % qs.len()]);
            let k = qmm(x, &ks[i % ks.len()]);
            let v = qmm(x, &vs[i % vs.len()]);
            q.add(&MlxArray::concatenate(&[&k, &v], 1).unwrap())
                .unwrap()
        }),
        qs[0].bytes + 2 * ks[0].bytes,
        roof,
    );
    drop((qs, ks, vs));
    let up = quant_pool(&mut rng, 1, 6144, h);
    let down = quant_pool(&mut rng, 1, h, 6144);
    let us = chain(200, &x, |i, x| {
        qmm(&qmm(x, &up[i % up.len()]), &down[i % down.len()])
    });
    row(
        "qmv 2048->6144 + 6144->2048 (up|gate, down)",
        us,
        up[0].bytes + down[0].bytes,
        roof,
    );
    drop((up, down));
    let lm = quant(&mut rng, 1, 151_936, h);
    row(
        "qmv 2048->151936 (tied lm_head)",
        chain(40, &x, |_, x| {
            qmm(x, &lm).slice(&[0, 0], &[1, h], &[1, 1]).unwrap()
        }),
        lm.bytes,
        roof,
    );
    drop(lm);
    let x3 = bf16_array(&mut rng, &[1, 1, h]);
    let wn = bf16_array(&mut rng, &[h]);
    row(
        "rms_norm [1,1,2048]",
        chain(1000, &x3, |_, x| x.rms_norm(Some(&wn), 1e-6).unwrap()),
        0,
        roof,
    );
    let qh = bf16_array(&mut rng, &[1, 16, 1, 128]);
    let qn = bf16_array(&mut rng, &[128]);
    row(
        "rms_norm per-head [1,16,1,128]",
        chain(1000, &qh, |_, x| x.rms_norm(Some(&qn), 1e-6).unwrap()),
        0,
        roof,
    );
    row(
        "rope [1,16,1,128]",
        chain(1000, &qh, |_, x| x.rope(128, 1e6, 1.0, 900).unwrap()),
        0,
        roof,
    );
    for len in [256usize, 1024] {
        let ks = bf16_pool(&mut rng, &[1, 8, len, 128]);
        let vs = bf16_pool(&mut rng, &[1, 8, len, 128]);
        let us = chain(500, &qh, |i, q| {
            q.scaled_dot_product_attention(
                &ks[i % ks.len()],
                &vs[i % vs.len()],
                0.088,
                SdpaMask::None,
            )
            .unwrap()
        });
        row(
            &format!("sdpa GQA 16q/8kv, kv_len {len}"),
            us,
            2 * 8 * len * 128 * 2,
            roof,
        );
    }
    let buf = bf16_array(&mut rng, &[1, 8, 1024, 128]);
    let kr = bf16_array(&mut rng, &[1, 8, 1, 128]);
    row(
        "kv slice_update row into [1,8,1024,128]",
        chain(500, &buf, |_, b| {
            b.slice_update(&kr, &[0, 0, 900, 0], &[1, 8, 901, 128], &[1, 1, 1, 1])
                .unwrap()
        }),
        0,
        roof,
    );
    let g = bf16_array(&mut rng, &[1, 6144]);
    row(
        "swiglu [1,6144] (compiled)",
        chain(1000, &g, |_, x| x.swiglu(x).unwrap()),
        0,
        roof,
    );

    println!("\n-- Unlimited-OCR decode step (12 layers: 1 dense + 11 MoE) --");
    let d = 1280;
    let x = bf16_array(&mut rng, &[1, d]);
    let att = quant_pool(&mut rng, 1, d, d);
    row(
        "qmv 1280->1280 (q, k, v, o)",
        chain(400, &x, |i, x| qmm(x, &att[i % att.len()])),
        att[0].bytes,
        roof,
    );
    drop(att);
    let fused = quant_pool(&mut rng, 1, 3 * d, d);
    row(
        "qmv 1280->3840 (fused q|k|v)",
        chain(300, &x, |i, x| {
            qmm(x, &fused[i % fused.len()])
                .slice(&[0, 0], &[1, d], &[1, 1])
                .unwrap()
        }),
        fused[0].bytes,
        roof,
    );
    drop(fused);
    let su = quant_pool(&mut rng, 1, 1792, d);
    let sd = quant_pool(&mut rng, 1, d, 1792);
    row(
        "qmv 1280->1792 + 1792->1280 (shared up|gate, down)",
        chain(300, &x, |i, x| {
            qmm(&qmm(x, &su[i % su.len()]), &sd[i % sd.len()])
        }),
        su[0].bytes + sd[0].bytes,
        roof,
    );
    drop((su, sd));
    let (experts, inter, topk) = (64usize, 896usize, 6usize);
    let e_up = quant_pool(&mut rng, experts, inter, d);
    let e_down = quant_pool(&mut rng, experts, d, inter);
    let picks: Vec<u32> = (0..topk as u32).map(|i| i * 9 + 3).collect();
    let idx = MlxArray::from_u32(&[1, 1, topk], &picks).unwrap();
    let x5 = bf16_array(&mut rng, &[1, 1, 1, 1, d]);
    let gq = |x: &MlxArray, q: &Quant| {
        x.gather_qmm(
            &q.w,
            &q.s,
            Some(&q.b),
            None,
            Some(&idx),
            true,
            Some(GROUP as i32),
            Some(BITS as i32),
            false,
        )
        .unwrap()
    };
    let per_expert = (e_up[0].bytes + e_down[0].bytes) / experts;
    row(
        "gather_qmm up + down, 6 of 64 experts (+sum)",
        chain(200, &x5, |i, x| {
            gq(&gq(x, &e_up[i % e_up.len()]), &e_down[i % e_down.len()])
                .sum_axis(2, true)
                .unwrap()
        }),
        per_expert * topk,
        roof,
    );
    let e_gate = quant_pool(&mut rng, experts, inter, d);
    let scores = bf16_array(&mut rng, &[1, 1, topk, 1]);
    row(
        "MoE experts unfused: gate|up + swiglu + down + score sum",
        chain(200, &x5, |i, x| {
            let (g, u) = (&e_gate[i % e_gate.len()], &e_up[i % e_up.len()]);
            let h = gq(x, g).swiglu(&gq(x, u)).unwrap();
            gq(&h, &e_down[i % e_down.len()])
                .reshape(&[1, 1, topk, d])
                .unwrap()
                .multiply(&scores)
                .unwrap()
                .sum_axis(-2, false)
                .unwrap()
                .reshape(&[1, 1, 1, 1, d])
                .unwrap()
        }),
        per_expert * topk * 3 / 2,
        roof,
    );
    if std::env::var("AIMD_MOE_STAGE").is_ok() {
        row(
            "  unfused stage: gate|up + swiglu (slice back to x)",
            chain(200, &x5, |i, x| {
                let (g, u) = (&e_gate[i % e_gate.len()], &e_up[i % e_up.len()]);
                gq(x, g)
                    .swiglu(&gq(x, u))
                    .unwrap()
                    .reshape(&[topk * inter])
                    .unwrap()
                    .slice(&[0], &[d], &[1])
                    .unwrap()
                    .reshape(&[1, 1, 1, 1, d])
                    .unwrap()
            }),
            per_expert * topk,
            roof,
        );
        let h0 = bf16_array(&mut rng, &[1, 1, topk, 1, inter]);
        row(
            "  unfused stage: down + score sum (fixed h)",
            chain(200, &x5, |i, x| {
                gq(&h0, &e_down[i % e_down.len()])
                    .reshape(&[1, 1, topk, d])
                    .unwrap()
                    .multiply(&scores)
                    .unwrap()
                    .sum_axis(-2, false)
                    .unwrap()
                    .reshape(&[1, 1, 1, 1, d])
                    .unwrap()
                    .add(x)
                    .unwrap()
            }),
            per_expert * topk / 2,
            roof,
        );
    }
    let inds1 = MlxArray::from_u32(&[topk], &picks).unwrap();
    let scores1 = scores.reshape(&[topk]).unwrap();
    let x1 = bf16_array(&mut rng, &[d]);
    row(
        "MoE experts fused kernel (2 dispatches)",
        chain(200, &x1, |i, x| {
            let (g, u, dn) = (
                &e_gate[i % e_gate.len()],
                &e_up[i % e_up.len()],
                &e_down[i % e_down.len()],
            );
            x.moe_swiglu_4bit(
                &inds1,
                &scores1,
                [&g.w, &g.s, &g.b],
                [&u.w, &u.s, &u.b],
                [&dn.w, &dn.s, &dn.b],
                GROUP as i32,
            )
            .unwrap()
        }),
        per_expert * topk * 3 / 2,
        roof,
    );
    drop((e_up, e_down, e_gate));
    let wgate = bf16_array(&mut rng, &[d, experts]);
    let x3 = bf16_array(&mut rng, &[1, 1, d]);
    let gate_logits = |x: &MlxArray| x.matmul(&wgate, None).unwrap();
    row(
        "router: gate matmul [1,1280]x[1280,64] (+add)",
        chain(500, &x3, |_, x| {
            x.add(&gate_logits(x).sum_axis(-1, true).unwrap()).unwrap()
        }),
        0,
        roof,
    );
    let wgate_t = bf16_array(&mut rng, &[experts, d])
        .transpose(&[1, 0])
        .unwrap();
    row(
        "router: gate matmul via W_g.T view (decoder layout)",
        chain(500, &x3, |_, x| {
            x.add(
                &x.matmul(&wgate_t, None)
                    .unwrap()
                    .sum_axis(-1, true)
                    .unwrap(),
            )
            .unwrap()
        }),
        0,
        roof,
    );
    row(
        "  of which sum + add alone [1,1,1280]",
        chain(1000, &x3, |_, x| {
            x.add(&x.sum_axis(-1, true).unwrap()).unwrap()
        }),
        0,
        roof,
    );
    let scores0 = bf16_array(&mut rng, &[1, 1, experts]);
    row(
        "router: softmax(precise) [1,1,64]",
        chain(1000, &scores0, |_, s| s.softmax_precise(-1, true).unwrap()),
        0,
        roof,
    );
    row(
        "router: argpartition top-6 + take_along (+add)",
        chain(500, &scores0, |_, s| {
            let top = s
                .argpartition(-(topk as i32), -1)
                .unwrap()
                .slice(&[0, 0, experts - topk], &[1, 1, experts], &[1, 1, 1])
                .unwrap();
            s.add(
                &s.take_along_axis(&top, -1)
                    .unwrap()
                    .sum_axis(-1, true)
                    .unwrap(),
            )
            .unwrap()
        }),
        0,
        roof,
    );
    row(
        "  of which sum + add alone",
        chain(1000, &scores0, |_, s| {
            s.add(&s.sum_axis(-1, true).unwrap()).unwrap()
        }),
        0,
        roof,
    );
    let lm = quant(&mut rng, 1, 129_280, d);
    row(
        "qmv 1280->129280 (lm_head)",
        chain(40, &x, |_, x| {
            qmm(x, &lm).slice(&[0, 0], &[1, d], &[1, 1]).unwrap()
        }),
        lm.bytes,
        roof,
    );
    drop(lm);
    let qo = bf16_array(&mut rng, &[1, 10, 1, 128]);
    let len = 2327 + 128;
    let ks = bf16_pool(&mut rng, &[1, 10, len, 128]);
    let vs = bf16_pool(&mut rng, &[1, 10, len, 128]);
    row(
        "sdpa MHA 10 heads, kv_len 2455 (prefill + ring)",
        chain(300, &qo, |i, q| {
            q.scaled_dot_product_attention(
                &ks[i % ks.len()],
                &vs[i % vs.len()],
                0.088,
                SdpaMask::None,
            )
            .unwrap()
        }),
        2 * 10 * len * 128 * 2,
        roof,
    );
}
