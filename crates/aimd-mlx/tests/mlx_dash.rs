//! Unlimited-OCR-2 DASH ops (spec 008): the sparse prefill/decode paths
//! against the dense-mask oracle, and the router helpers against host f64
//! references. On Metal the custom kernels (`DashMode::Auto`) are checked
//! too; elsewhere `Auto` runs the mlx-op path.

use aimd_mlx::{DashLayerConsts, DashMode, DashParams, DashPromptState, MlxArray, MlxDtype};

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
    fn vec(&mut self, n: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * amp).collect()
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

const P: DashParams = DashParams {
    block_size: 32,
    local_blocks: 4,
    topm: 16,
    topk: 12,
    sigma: 1.0,
};

const HQ: usize = 4;
const HK: usize = 2;
const D: usize = 64;
const DI: usize = 32;

struct Case {
    keys: MlxArray,
    values: MlxArray,
    summaries: Vec<f32>,
    n_sum: usize,
    rng: Rng,
}

/// Cache buffers `[1, HK, s_alloc, D]` (positions `>= total` hold junk the
/// kernels must never read into the result) and block summaries of every
/// completed block.
fn case(total: usize, s_alloc: usize, seed: u64) -> Case {
    let mut rng = Rng::new(seed);
    let keys = f32_arr(&[1, HK, s_alloc, D], &rng.vec(HK * s_alloc * D, 1.0));
    let values = f32_arr(&[1, HK, s_alloc, D], &rng.vec(HK * s_alloc * D, 1.0));
    let n_sum = total / P.block_size as usize;
    let summaries = rng.vec(n_sum * DI, 3.0);
    Case {
        keys,
        values,
        summaries,
        n_sum,
        rng,
    }
}

fn consts() -> DashLayerConsts {
    DashLayerConsts {
        idx_scale: 2.0 / (DI as f32).sqrt(),
        floor: 1e-2,
        scale: 1.0 / (D as f32).sqrt(),
    }
}

fn metal() -> bool {
    aimd_mlx::metal_kernels_available().unwrap()
}

#[test]
fn prefill_sparse_paths_match_dense_oracle() {
    let (offset, len, s_alloc) = (300usize, 500usize, 1024usize);
    let total = offset + len;
    let nblk = total.div_ceil(P.block_size as usize);
    let mut c = case(total, s_alloc, 7);
    let q = f32_arr(&[1, HQ, len, D], &c.rng.vec(HQ * len * D, 1.0));
    let idx_q = f32_arr(&[len, DI], &c.rng.vec(len * DI, 1.0));
    let sums = f32_arr(&[c.n_sum, DI], &c.summaries);
    let cand: Vec<bool> = (0..len * nblk)
        .map(|_| c.rng.next_u64().is_multiple_of(2))
        .collect();
    let cand = MlxArray::from_bool(&[len, nblk], &cand).unwrap();
    let k = consts();
    let run = |q: &MlxArray, keys: &MlxArray, values: &MlxArray, mode| {
        q.dash_prefill(
            keys,
            values,
            &idx_q,
            Some(&sums),
            Some(&cand),
            offset as i32,
            &P,
            &k,
            mode,
            true,
        )
        .unwrap()
    };
    let (dense, dr) = run(&q, &c.keys, &c.values, DashMode::Dense);
    let dense = dense.to_f32_vec().unwrap();
    let dr = dr.unwrap();
    let valid = dr
        .valid
        .astype(MlxDtype::U32)
        .unwrap()
        .to_u32_vec()
        .unwrap();
    assert!(valid.contains(&1), "no remote block was selected");
    assert_eq!(dr.topi.shape().unwrap(), vec![len, P.topk as usize]);

    let (gather, gr) = run(&q, &c.keys, &c.values, DashMode::Gather);
    let gr = gr.unwrap();
    assert_eq!(gr.topi.to_u32_vec().unwrap(), dr.topi.to_u32_vec().unwrap());
    let err = max_abs(&gather.to_f32_vec().unwrap(), &dense);
    assert!(err < 1e-5, "gather vs dense: {err}");

    if metal() {
        let (auto, _) = run(&q, &c.keys, &c.values, DashMode::Auto);
        let err = max_abs(&auto.to_f32_vec().unwrap(), &dense);
        assert!(err < 1e-4, "kernel vs dense (f32): {err}");
        // bf16: the kernel accumulates in f32, sdpa in its own order.
        let cast = |a: &MlxArray| a.astype(MlxDtype::Bf16).unwrap();
        let (qb, kb, vb) = (cast(&q), cast(&c.keys), cast(&c.values));
        let to32 = |a: MlxArray| a.astype(MlxDtype::F32).unwrap().to_f32_vec().unwrap();
        let dense_b = to32(run(&qb, &kb, &vb, DashMode::Dense).0);
        let auto_b = to32(run(&qb, &kb, &vb, DashMode::Auto).0);
        let err = max_abs(&auto_b, &dense_b);
        assert!(err < 3e-2, "kernel vs dense (bf16): {err}");
    }
}

#[test]
fn prefill_without_summaries_is_local_attention() {
    let (len, s_alloc) = (100usize, 128usize);
    let mut c = case(len, s_alloc, 11);
    let q = f32_arr(&[1, HQ, len, D], &c.rng.vec(HQ * len * D, 1.0));
    let idx_q = f32_arr(&[len, DI], &c.rng.vec(len * DI, 1.0));
    let k = consts();
    let run = |mode| {
        q.dash_prefill(
            &c.keys, &c.values, &idx_q, None, None, 0, &P, &k, mode, false,
        )
        .unwrap()
        .0
        .to_f32_vec()
        .unwrap()
    };
    let dense = run(DashMode::Dense);
    assert!(max_abs(&run(DashMode::Gather), &dense) < 1e-5);
    assert!(max_abs(&run(DashMode::Auto), &dense) < 1e-4);
}

#[test]
fn decode_matches_single_query_prefill() {
    let s_alloc = 1280usize;
    let cap = 64usize;
    for (pos, seed) in [(1099usize, 3u64), (1056, 5), (40, 9)] {
        let total = pos + 1;
        let nblk = total.div_ceil(P.block_size as usize);
        let mut c = case(total, s_alloc, seed);
        let q = f32_arr(&[1, HQ, 1, D], &c.rng.vec(HQ * D, 1.0));
        let idx_q = f32_arr(&[1, DI], &c.rng.vec(DI, 1.0));
        let cand: Vec<bool> = (0..cap)
            .map(|_| !c.rng.next_u64().is_multiple_of(3))
            .collect();
        let mut padded = c.summaries.clone();
        padded.resize(cap * DI, 0.0);
        let k = consts();
        let sums = (c.n_sum > 0).then(|| f32_arr(&[c.n_sum, DI], &c.summaries));
        let want = q
            .dash_prefill(
                &c.keys,
                &c.values,
                &idx_q,
                sums.as_ref(),
                Some(&MlxArray::from_bool(&[1, nblk], &cand[..nblk]).unwrap()),
                pos as i32,
                &P,
                &k,
                DashMode::Dense,
                false,
            )
            .unwrap()
            .0
            .to_f32_vec()
            .unwrap();
        let sums = f32_arr(&[cap, DI], &padded);
        let cand = MlxArray::from_bool(&[1, cap], &cand).unwrap();
        let pos_a = MlxArray::from_i32(&[], &[pos as i32]).unwrap();
        for mode in [DashMode::Gather, DashMode::Auto] {
            let got = q
                .dash_decode(
                    &c.keys, &c.values, &idx_q, &sums, &cand, &pos_a, &P, &k, mode,
                )
                .unwrap()
                .to_f32_vec()
                .unwrap();
            let err = max_abs(&got, &want);
            assert!(err < 1e-4, "pos {pos} {mode:?}: {err}");
        }
    }
}

fn rotate_half_ref(x: &[f64]) -> Vec<f64> {
    let h = x.len() / 2;
    (0..x.len())
        .map(|i| if i < h { -x[i + h] } else { x[i - h] })
        .collect()
}

#[test]
fn rope_tables_and_rotation() {
    let tok = [0i32, 5, 17];
    let page = [0i32, 1, 3];
    let inv_tok = [1.0f32, 0.3, 0.05];
    let inv_page = [1.0f32];
    let (cos, sin) = MlxArray::dash_rope(
        &MlxArray::from_i32(&[3], &tok).unwrap(),
        &MlxArray::from_i32(&[3], &page).unwrap(),
        &f32_arr(&[3], &inv_tok),
        &f32_arr(&[1], &inv_page),
        2,
        MlxDtype::F32,
    )
    .unwrap();
    assert_eq!(cos.shape().unwrap(), vec![1, 1, 3, 8]);
    let (cv, sv) = (cos.to_f32_vec().unwrap(), sin.to_f32_vec().unwrap());
    let mut x = Rng::new(2);
    let xs = x.vec(3 * 8, 1.0);
    let rot = f32_arr(&[1, 1, 3, 8], &xs)
        .dash_apply_rope(&cos, &sin, false)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    for l in 0..3 {
        let mut ang: Vec<f64> = inv_tok.iter().map(|&f| tok[l] as f64 * f as f64).collect();
        ang.push((page[l] % 2) as f64 * inv_page[0] as f64);
        let emb: Vec<f64> = ang.iter().chain(ang.iter()).copied().collect();
        let row: Vec<f64> = xs[l * 8..(l + 1) * 8].iter().map(|&v| v as f64).collect();
        let rh = rotate_half_ref(&row);
        for i in 0..8 {
            assert!((cv[l * 8 + i] as f64 - emb[i].cos()).abs() < 1e-5);
            assert!((sv[l * 8 + i] as f64 - emb[i].sin()).abs() < 1e-5);
            let want = row[i] * emb[i].cos() + rh[i] * emb[i].sin();
            assert!((rot[l * 8 + i] as f64 - want).abs() < 1e-5);
        }
    }
}

#[test]
fn index_qk_and_summaries_match_host() {
    let (l, h, d) = (3usize, 16usize, 8usize);
    let mut r = Rng::new(4);
    let x = r.vec(l * h, 1.0);
    let qw = r.vec(d * h, 1.0);
    let kw = r.vec(d * h, 1.0);
    let qn = r.vec(d, 0.5);
    let kn = r.vec(d, 0.5);
    let cos_v = r.vec(l * d, 1.0);
    let sin_v = r.vec(l * d, 1.0);
    let eps = 1e-6f32;
    let (cos, sin) = (
        f32_arr(&[1, 1, l, d], &cos_v),
        f32_arr(&[1, 1, l, d], &sin_v),
    );
    for decode in [false, true] {
        let (q, k) = f32_arr(&[1, l, h], &x)
            .dash_index_qk(
                &f32_arr(&[d, h], &qw),
                &f32_arr(&[d, h], &kw),
                &f32_arr(&[d], &qn),
                &f32_arr(&[d], &kn),
                &cos,
                &sin,
                eps,
                decode,
            )
            .unwrap();
        for (got, w, nw) in [(q, &qw, &qn), (k, &kw, &kn)] {
            let got = got.to_f32_vec().unwrap();
            for i in 0..l {
                let y: Vec<f64> = (0..d)
                    .map(|o| {
                        (0..h)
                            .map(|j| x[i * h + j] as f64 * w[o * h + j] as f64)
                            .sum()
                    })
                    .collect();
                let ms = y.iter().map(|v| v * v).sum::<f64>() / d as f64;
                let n: Vec<f64> = (0..d)
                    .map(|o| y[o] / (ms + eps as f64).sqrt() * (1.0 + nw[o] as f64))
                    .collect();
                let rh = rotate_half_ref(&n);
                for o in 0..d {
                    let want = n[o] * cos_v[i * d + o] as f64 + rh[o] * sin_v[i * d + o] as f64;
                    let g = got[i * d + o] as f64;
                    assert!(
                        (g - want).abs() < 1e-4 * (1.0 + want.abs()),
                        "{g} vs {want}"
                    );
                }
            }
        }
    }

    let (n, bs) = (3usize, 32usize);
    let blocks = r.vec(n * bs * d, 2.0);
    let cls = r.vec(d, 2.0);
    let got = MlxArray::dash_summarize(&f32_arr(&[n, bs, d], &blocks), &f32_arr(&[d], &cls))
        .unwrap()
        .to_f32_vec()
        .unwrap();
    for b in 0..n {
        let row = |t: usize| &blocks[(b * bs + t) * d..(b * bs + t + 1) * d];
        let aw: Vec<f64> = (0..bs)
            .map(|t| {
                row(t)
                    .iter()
                    .zip(&cls)
                    .map(|(a, c)| *a as f64 * *c as f64)
                    .sum::<f64>()
                    / (d as f64).sqrt()
            })
            .collect();
        let m = aw.iter().copied().fold(f64::MIN, f64::max);
        let z: f64 = aw.iter().map(|a| (a - m).exp()).sum();
        for o in 0..d {
            let want: f64 = (0..bs)
                .map(|t| (aw[t] - m).exp() / z * row(t)[o] as f64)
                .sum();
            assert!((got[b * d + o] as f64 - want).abs() < 1e-5);
        }
    }
}

#[test]
fn decode_prologue_matches_separate_ops() {
    let (h, d) = (256usize, 128usize);
    let mut r = Rng::new(6);
    let bf = |shape: &[usize], amp: f32, r: &mut Rng| {
        let n = shape.iter().product();
        f32_arr(shape, &r.vec(n, amp))
            .astype(MlxDtype::Bf16)
            .unwrap()
    };
    let x = bf(&[1, 1, h], 1.0, &mut r);
    let q = bf(&[1, HQ, 1, d], 1.0, &mut r);
    let k = bf(&[1, HK, 1, d], 1.0, &mut r);
    let (qw, kw) = (bf(&[d, h], 0.1, &mut r), bf(&[d, h], 0.1, &mut r));
    let (qn, kn) = (bf(&[d], 0.5, &mut r), bf(&[d], 0.5, &mut r));
    let (cos, sin) = (
        bf(&[1, 1, 1, d], 1.0, &mut r),
        bf(&[1, 1, 1, d], 1.0, &mut r),
    );
    let eps = 1e-6f32;
    let (gq, gk, giq, gik) = x
        .dash_decode_prologue(&q, &k, &qw, &kw, &qn, &kn, &cos, &sin, eps)
        .unwrap();
    let (wiq, wik) = x
        .dash_index_qk(&qw, &kw, &qn, &kn, &cos, &sin, eps, true)
        .unwrap();
    let f = |a: &MlxArray| a.astype(MlxDtype::F32).unwrap().to_f32_vec().unwrap();
    for (got, x) in [(&gq, &q), (&gk, &k)] {
        let want = x.dash_apply_rope(&cos, &sin, true).unwrap();
        assert_eq!(got.shape().unwrap(), want.shape().unwrap());
        assert_eq!(max_abs(&f(got), &f(&want)), 0.0);
    }
    // The router matvec may sum in another order: one bf16 rounding apart.
    for (got, want) in [(&giq, &wiq), (&gik, &wik)] {
        assert_eq!(got.shape().unwrap(), vec![1, d]);
        let (g, w) = (f(got), f(want));
        let scale = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            max_abs(&g, &w) <= 2e-2 * scale,
            "{} vs scale {scale}",
            max_abs(&g, &w)
        );
    }
}

#[test]
fn decode_ctx_advances_pages_in_graph() {
    let pf_cnt = MlxArray::from_i32(&[3], &[10, 7, 4]).unwrap();
    // Prompt blocks: [page 0, page 0..1, page 1, text].
    let blk_min = MlxArray::from_i32(&[4], &[0, 0, 1, -1]).unwrap();
    let blk_max = MlxArray::from_i32(&[4], &[0, 1, 1, -1]).unwrap();
    let inv_tok = f32_arr(&[3], &[1.0, 0.1, 0.01]);
    let inv_page = f32_arr(&[1], &[1.0]);
    let prompt = DashPromptState {
        pf_cnt: &pf_cnt,
        blk_min: &blk_min,
        blk_max: &blk_max,
        inv_tok: &inv_tok,
        inv_page: &inv_page,
        page_token_id: 99,
        n_pages: 3,
        prompt_pages: 2,
        page_mod: 2,
        image_only: true,
    };
    let scalar = |v: i32| MlxArray::from_i32(&[], &[v]).unwrap();
    let read = |a: &MlxArray| a.astype(MlxDtype::U32).unwrap().to_u32_vec().unwrap();
    // Answer starts: 2 prompt markers, answer page 0, tok 3.
    let (mut markers, mut page, mut tok) = (scalar(2), scalar(0), scalar(3));
    // A text token advances tok only; no answer page yet -> all candidates.
    let ctx = MlxArray::dash_decode_ctx(
        &MlxArray::from_u32(&[1], &[5]).unwrap(),
        &markers,
        &page,
        &tok,
        &prompt,
        8,
        MlxDtype::F32,
    )
    .unwrap();
    assert_eq!(read(&ctx.tok), vec![4]);
    assert_eq!(read(&ctx.cand), vec![1; 8]);
    (markers, page, tok) = (ctx.markers, ctx.page, ctx.tok);
    // `<PAGE>`: answer page 0 -> candidates are blocks touching page 0.
    let ctx = MlxArray::dash_decode_ctx(
        &MlxArray::from_u32(&[1], &[99]).unwrap(),
        &markers,
        &page,
        &tok,
        &prompt,
        8,
        MlxDtype::F32,
    )
    .unwrap();
    assert_eq!(read(&ctx.markers), vec![3]);
    assert_eq!(read(&ctx.page), vec![1]);
    assert_eq!(read(&ctx.tok), vec![7]);
    assert_eq!(read(&ctx.cand), vec![1, 1, 0, 0, 0, 0, 0, 0]);
    let cos = ctx.cos.to_f32_vec().unwrap();
    assert_eq!(cos.len(), 8);
    assert!((cos[0] - 7f32.cos()).abs() < 1e-6);
    assert!((cos[3] - 1f32.cos()).abs() < 1e-6);
}
