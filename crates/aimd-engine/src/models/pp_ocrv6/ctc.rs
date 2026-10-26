//! CTC greedy decoding (mlx4ocr `postprocess/ctc.py`) plus the two numpy
//! routines whose exact behaviour shows up in the output: `np.argsort`
//! (the default introsort, which is not stable and decides the recognition
//! batch composition) and the `float32` pairwise `np.mean` used for the line
//! score.

/// numpy `npy_get_msb`.
fn msb(mut n: usize) -> i32 {
    let mut depth = 0;
    n >>= 1;
    while n > 0 {
        depth += 1;
        n >>= 1;
    }
    depth
}

/// numpy `aheapsort_` (1-based heap over `tosort`).
fn aheapsort(v: &[f64], tosort: &mut [usize]) {
    let n0 = tosort.len();
    // a[k] == tosort[k - 1]
    let less = |a: usize, b: usize| v[a] < v[b];
    let mut n = n0;
    let mut l = n >> 1;
    while l > 0 {
        let tmp = tosort[l - 1];
        let (mut i, mut j) = (l, l << 1);
        while j <= n {
            if j < n && less(tosort[j - 1], tosort[j]) {
                j += 1;
            }
            if less(tmp, tosort[j - 1]) {
                tosort[i - 1] = tosort[j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[i - 1] = tmp;
        l -= 1;
    }
    while n > 1 {
        let tmp = tosort[n - 1];
        tosort[n - 1] = tosort[0];
        n -= 1;
        let (mut i, mut j) = (1usize, 2usize);
        while j <= n {
            if j < n && less(tosort[j - 1], tosort[j]) {
                j += 1;
            }
            if less(tmp, tosort[j - 1]) {
                tosort[i - 1] = tosort[j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[i - 1] = tmp;
    }
}

/// `np.argsort(v)` for a 1-D `float64` array (numpy's scalar
/// `aquicksort_<double>`: median-of-3 introsort, insertion sort below 16
/// elements, heapsort past `2 * msb(n)` partition levels).
pub fn np_argsort(v: &[f64]) -> Vec<usize> {
    const SMALL_QUICKSORT: usize = 15;
    let num = v.len();
    let mut t: Vec<usize> = (0..num).collect();
    if num < 2 {
        return t;
    }
    let less = |a: usize, b: usize| v[a] < v[b];
    let mut stack: Vec<(usize, usize, i32)> = Vec::new();
    let (mut pl, mut pr) = (0usize, num - 1);
    let mut cdepth = msb(num) * 2;
    loop {
        if cdepth < 0 {
            aheapsort(v, &mut t[pl..=pr]);
        } else {
            while pr - pl > SMALL_QUICKSORT {
                let pm = pl + ((pr - pl) >> 1);
                if less(t[pm], t[pl]) {
                    t.swap(pm, pl);
                }
                if less(t[pr], t[pm]) {
                    t.swap(pr, pm);
                }
                if less(t[pm], t[pl]) {
                    t.swap(pm, pl);
                }
                let vp = t[pm];
                let mut pi = pl;
                let mut pj = pr - 1;
                t.swap(pm, pj);
                loop {
                    loop {
                        pi += 1;
                        if !less(t[pi], vp) {
                            break;
                        }
                    }
                    loop {
                        pj -= 1;
                        if !less(vp, t[pj]) {
                            break;
                        }
                    }
                    if pi >= pj {
                        break;
                    }
                    t.swap(pi, pj);
                }
                let pk = pr - 1;
                t.swap(pi, pk);
                cdepth -= 1;
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, cdepth));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, cdepth));
                    pl = pi + 1;
                }
            }
            for pi in pl + 1..=pr {
                let vi = t[pi];
                let mut pj = pi;
                while pj > pl && less(vi, t[pj - 1]) {
                    t[pj] = t[pj - 1];
                    pj -= 1;
                }
                t[pj] = vi;
            }
        }
        match stack.pop() {
            Some((l, r, d)) => {
                pl = l;
                pr = r;
                cdepth = d;
            }
            None => break,
        }
    }
    t
}

/// numpy `FLOAT_pairwise_sum` (8 accumulators, blocks of 128).
fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut res = 0f32;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
    }
}

/// `float(np.mean(a))` for a non-empty `float32` array.
pub fn np_mean_f32(a: &[f32]) -> f32 {
    (0f32 + pairwise_sum_f32(a)) / a.len() as f32
}

/// One decoded line: text and mean max-probability of the kept steps.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Decoded {
    pub text: String,
    pub score: f32,
}

/// Greedy CTC decode of one sequence from per-step `argmax` indices and
/// `max` probabilities (blank = 0, repeats collapsed).
pub fn ctc_decode(indices: &[u32], probs: &[f32], characters: &[String]) -> Decoded {
    let mut text = String::new();
    let mut kept = Vec::new();
    for (t, &idx) in indices.iter().enumerate() {
        if idx == 0 || (t > 0 && indices[t - 1] == idx) {
            continue;
        }
        if let Some(ch) = characters.get(idx as usize) {
            text.push_str(ch);
        }
        kept.push(probs[t]);
    }
    let score = if kept.is_empty() {
        0.0
    } else {
        np_mean_f32(&kept)
    };
    Decoded { text, score }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argsort_sorts_and_matches_small_cases() {
        let v = [3.0, 1.0, 2.0, 1.0, 0.5];
        let idx = np_argsort(&v);
        // insertion sort is stable for n <= 16
        assert_eq!(idx, vec![4, 1, 3, 2, 0]);
        let big: Vec<f64> = (0..200).map(|i| ((i * 7919) % 101) as f64 / 7.0).collect();
        let idx = np_argsort(&big);
        for w in idx.windows(2) {
            assert!(big[w[0]] <= big[w[1]]);
        }
        let mut seen = idx.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..200).collect::<Vec<_>>());
    }

    #[test]
    fn ctc_collapses_repeats_and_blanks() {
        let chars: Vec<String> = ["blank", "a", "b"].iter().map(|s| s.to_string()).collect();
        let d = ctc_decode(
            &[1, 1, 0, 1, 2, 2, 0],
            &[0.5, 0.9, 0.1, 0.75, 1.0, 0.2, 0.3],
            &chars,
        );
        assert_eq!(d.text, "aab");
        assert_eq!(d.score, (0.5f32 + 0.75 + 1.0) / 3.0);
        let e = ctc_decode(&[0, 0], &[0.9, 0.9], &chars);
        assert_eq!(
            e,
            Decoded {
                text: String::new(),
                score: 0.0
            }
        );
    }
}
