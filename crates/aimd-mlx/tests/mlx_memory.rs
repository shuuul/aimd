//! The shim caps mlx's buffer cache, so shape-varying calls cannot grow it
//! without bound (PP-OCRv6 recognition batches once reached 25 GB).
use aimd_mlx::{MlxArray, MlxDtype};

#[test]
fn buffer_cache_stays_under_the_cap_for_varying_shapes() {
    let limit = aimd_mlx::memory_stats().cache_limit;
    assert_eq!(limit, 1 << 30);
    // 24 distinct sizes of 64-210 MB (about 3.3 GB freed in all): without
    // the cap every one stays cached because none fits another.
    for i in 0..24 {
        let rows = 1000 + 100 * i;
        let x = MlxArray::zeros(&[rows, 1 << 14], MlxDtype::F32).unwrap();
        let y = x.add(&x).unwrap();
        y.eval().unwrap();
        drop((x, y));
        // mlx trims the cache on the next allocation, so it may exceed the
        // cap by the buffers this step just freed.
        let step = 2 * rows * (1 << 14) * 4;
        let cache = aimd_mlx::memory_stats().cache;
        assert!(
            cache <= limit + step,
            "step {i}: cache {cache} > cap {limit} + {step}"
        );
    }
    aimd_mlx::clear_cache().unwrap();
    assert_eq!(aimd_mlx::memory_stats().cache, 0);
}
