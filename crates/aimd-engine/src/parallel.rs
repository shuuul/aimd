//! Scoped-thread helpers for host-side pre/postprocessing (spec 007).
//!
//! Every helper computes each element exactly as a single thread would;
//! threads only change which core does the work.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Worker threads for host work: the available cores.
pub fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Splits `out` into bands of whole rows (`row_len` elements each, at
/// least `min_rows` rows per band) and runs `f(first_row, band)` on each
/// band in parallel.
pub fn for_row_bands<T: Send>(
    out: &mut [T],
    row_len: usize,
    min_rows: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    if row_len == 0 || out.is_empty() {
        return;
    }
    let rows = out.len() / row_len;
    let per = rows.div_ceil(threads()).max(min_rows.max(1));
    if per >= rows {
        f(0, out);
        return;
    }
    std::thread::scope(|scope| {
        for (i, band) in out.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            scope.spawn(move || f(i * per, band));
        }
    });
}

/// `items.iter().map(f).collect()` on up to [`threads`] threads. Items are
/// handed out one at a time, so uneven items balance; results keep the
/// input order.
pub fn map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let n = threads().min(items.len());
    if n <= 1 {
        return items.iter().map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let mut parts: Vec<Vec<(usize, R)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(i) else {
                            return done;
                        };
                        done.push((i, f(item)));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    });
    let mut out: Vec<Option<R>> = (0..items.len()).map(|_| None).collect();
    for (i, r) in parts.iter_mut().flat_map(std::mem::take) {
        out[i] = Some(r);
    }
    out.into_iter()
        .map(|r| r.expect("every item mapped"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_keeps_order_and_bands_cover_every_row() {
        let items: Vec<u64> = (0..1000).collect();
        assert_eq!(
            map(&items, |&x| x * x),
            items.iter().map(|&x| x * x).collect::<Vec<_>>()
        );
        assert!(map(&[] as &[u64], |&x| x).is_empty());

        let mut rows = vec![0usize; 7 * 333];
        for_row_bands(&mut rows, 7, 1, |first, band| {
            for (r, row) in band.chunks_mut(7).enumerate() {
                row.fill(first + r);
            }
        });
        for (r, row) in rows.chunks(7).enumerate() {
            assert!(row.iter().all(|&v| v == r));
        }
    }
}
