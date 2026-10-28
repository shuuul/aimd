//! `MlxArray::grid_sample` (spec 009 WS-04) against a host port of torch's
//! CPU `grid_sampler_2d` (bilinear, zeros padding, `align_corners=False`),
//! computed in f64. Covers in-bounds, edge-straddling and fully outside
//! sample points over several batches and channels.

use aimd_mlx::MlxArray;

fn host_grid_sample(
    x: &[f32],
    [n, h, w, c]: [usize; 4],
    grid: &[f32],
    [hg, wg]: [usize; 2],
) -> Vec<f32> {
    let mut out = vec![0f32; n * hg * wg * c];
    for b in 0..n {
        for gy in 0..hg {
            for gx in 0..wg {
                let g = ((b * hg + gy) * wg + gx) * 2;
                let ix = ((f64::from(grid[g]) + 1.0) * w as f64 - 1.0) / 2.0;
                let iy = ((f64::from(grid[g + 1]) + 1.0) * h as f64 - 1.0) / 2.0;
                let (x0, y0) = (ix.floor(), iy.floor());
                let corners = [
                    (x0, y0, (x0 + 1.0 - ix) * (y0 + 1.0 - iy)),
                    (x0 + 1.0, y0, (ix - x0) * (y0 + 1.0 - iy)),
                    (x0, y0 + 1.0, (x0 + 1.0 - ix) * (iy - y0)),
                    (x0 + 1.0, y0 + 1.0, (ix - x0) * (iy - y0)),
                ];
                for ch in 0..c {
                    let mut acc = 0f64;
                    for &(cx, cy, wt) in &corners {
                        if cx < 0.0 || cy < 0.0 || cx >= w as f64 || cy >= h as f64 {
                            continue;
                        }
                        let v = x[((b * h + cy as usize) * w + cx as usize) * c + ch];
                        acc += wt * f64::from(v);
                    }
                    out[((b * hg + gy) * wg + gx) * c + ch] = acc as f32;
                }
            }
        }
    }
    out
}

#[test]
fn grid_sample_matches_torch_bilinear_zeros() {
    let (n, h, w, c) = (3usize, 5usize, 7usize, 4usize);
    let (hg, wg) = (6usize, 9usize);
    let x: Vec<f32> = (0..n * h * w * c)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.03)
        .collect();
    // Normalized coordinates spanning [-1.4, 1.4]: interior points, points
    // whose corners straddle the border, and points fully outside.
    let grid: Vec<f32> = (0..n * hg * wg * 2)
        .map(|i| ((i * 53 % 97) as f32 / 96.0) * 2.8 - 1.4)
        .collect();
    let mut grid = grid;
    // Exact pixel centers and the image corners.
    grid[0] = -1.0 + 1.0 / w as f32;
    grid[1] = -1.0 + 1.0 / h as f32;
    grid[2] = -1.0;
    grid[3] = -1.0;
    grid[4] = 1.0;
    grid[5] = 1.0;
    let got = MlxArray::from_f32(&[n, h, w, c], &x)
        .unwrap()
        .grid_sample(&MlxArray::from_f32(&[n, hg, wg, 2], &grid).unwrap())
        .unwrap();
    assert_eq!(got.shape().unwrap(), vec![n, hg, wg, c]);
    let got = got.to_f32_vec().unwrap();
    let want = host_grid_sample(&x, [n, h, w, c], &grid, [hg, wg]);
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(err < 1e-5, "max abs error {err}");
    // The exact pixel center reproduces the pixel; far-outside points are 0.
    for ch in 0..c {
        assert!((got[ch] - x[ch]).abs() < 1e-6);
    }
    assert!(want.contains(&0.0));
}

#[test]
fn grid_sample_rejects_bad_shapes() {
    let x = MlxArray::from_f32(&[1, 2, 2, 1], &[1.0; 4]).unwrap();
    let bad = MlxArray::from_f32(&[1, 2, 3], &[0.0; 6]).unwrap();
    assert!(x.grid_sample(&bad).is_err());
}
