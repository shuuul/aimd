//! Host-side image operations, bit-exact with the OpenCV 4.13 calls in
//! mlx4ocr: `cv2.resize` (`INTER_LINEAR`, u8), `cv2.getPerspectiveTransform`
//! with `cv2.warpPerspective` (`INTER_CUBIC`, `BORDER_REPLICATE`) and
//! `np.rot90`.
//!
//! Images are interleaved 3-channel `u8` in BGR order (`[H, W, 3]`), the
//! layout mlx4ocr feeds every stage. The fixed-point arithmetic follows
//! `modules/imgproc/src/resize.cpp` / `imgwarp.cpp`; the floating-point
//! steps that clang contracts into FMAs (perspective LU solve, the per-pixel
//! projective map) use [`f64::mul_add`] in the same places.

/// An interleaved 3-channel `u8` image (BGR, `[height, width, 3]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgrImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl BgrImage {
    /// A zero (black) image.
    pub fn zeros(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0; width * height * 3],
        }
    }

    /// Decodes an encoded image the way mlx4ocr does
    /// (`PIL.Image.open(..).convert("RGB")`, then reversed to BGR).
    pub fn decode(bytes: &[u8]) -> Result<Self, image::ImageError> {
        let rgb = image::load_from_memory(bytes)?.to_rgb8();
        let (width, height) = (rgb.width() as usize, rgb.height() as usize);
        let mut data = rgb.into_raw();
        for px in data.as_chunks_mut::<3>().0 {
            px.swap(0, 2);
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Pixel `(x, y)` channel `c`.
    #[inline]
    pub fn at(&self, x: usize, y: usize, c: usize) -> u8 {
        self.data[(y * self.width + x) * 3 + c]
    }
}

/// OpenCV `cvRound(float)`: round half to even, saturated to `i16` range
/// (`saturate_cast<short>`).
#[inline]
fn sat_short_round(v: f32) -> i32 {
    (v.round_ties_even() as i32).clamp(-32768, 32767)
}

/// Linear-interpolation taps for one axis: `(source index, alpha0, alpha1)`
/// per destination index (`INTER_RESIZE_COEF_SCALE = 2048`).
fn linear_taps(src_n: usize, dst_n: usize) -> Vec<(i64, i32, i32)> {
    let inv_scale = dst_n as f64 / src_n as f64;
    let scale = 1.0 / inv_scale;
    (0..dst_n)
        .map(|d| {
            let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let s = f.floor();
            let f = f - s;
            (
                s as i64,
                sat_short_round((1.0f32 - f) * 2048.0),
                sat_short_round(f * 2048.0),
            )
        })
        .collect()
}

/// `cv2.resize(img, (w, h))` with the default `INTER_LINEAR` on `u8`.
pub fn resize_linear(img: &BgrImage, w: usize, h: usize) -> BgrImage {
    let (sw, sh) = (img.width, img.height);
    let mut out = BgrImage::zeros(w, h);
    if w == 0 || h == 0 {
        return out;
    }
    // OpenCV switches an exact 2x downscale to the INTER_AREA fast path.
    let sx = 1.0 / (w as f64 / sw as f64);
    let sy = 1.0 / (h as f64 / sh as f64);
    if (sx - 2.0).abs() < f64::EPSILON && (sy - 2.0).abs() < f64::EPSILON {
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let s = img.at(2 * x, 2 * y, c) as u32
                        + img.at(2 * x + 1, 2 * y, c) as u32
                        + img.at(2 * x, 2 * y + 1, c) as u32
                        + img.at(2 * x + 1, 2 * y + 1, c) as u32;
                    out.data[(y * w + x) * 3 + c] = ((s + 2) >> 2) as u8;
                }
            }
        }
        return out;
    }

    // Horizontal taps with OpenCV's border handling (`xmax` columns past the
    // right edge copy the last pixel at full weight).
    let mut xtaps = Vec::with_capacity(w);
    let mut xmax = w;
    for (d, (s, a0, a1)) in linear_taps(sw, w).into_iter().enumerate() {
        let (mut s, mut a0, mut a1) = (s, a0, a1);
        if s < 0 {
            (s, a0, a1) = (0, 2048, 0);
        }
        if s + 1 >= sw as i64 {
            xmax = xmax.min(d);
            if s >= sw as i64 - 1 {
                (s, a0, a1) = (sw as i64 - 1, 2048, 0);
            }
        }
        xtaps.push((s as usize, a0, a1));
    }
    let hrow = |y: usize, buf: &mut [i32]| {
        let row = &img.data[y * sw * 3..(y + 1) * sw * 3];
        for (d, &(s, a0, a1)) in xtaps.iter().enumerate() {
            let s1 = (s + 1).min(sw - 1);
            for c in 0..3 {
                buf[d * 3 + c] = if d >= xmax {
                    row[s * 3 + c] as i32 * 2048
                } else {
                    row[s * 3 + c] as i32 * a0 + row[s1 * 3 + c] as i32 * a1
                };
            }
        }
    };

    let ytaps = linear_taps(sh, h);
    // Horizontally filtered rows for the two source rows of the current
    // output row; rows shared with the previous output row are reused.
    let mut cache: [(usize, Vec<i32>); 2] =
        [(usize::MAX, vec![0; w * 3]), (usize::MAX, vec![0; w * 3])];
    let fetch = |y: usize, cache: &[(usize, Vec<i32>); 2]| -> Vec<i32> {
        match cache.iter().find(|(r, _)| *r == y) {
            Some((_, row)) => row.clone(),
            None => {
                let mut buf = vec![0i32; w * 3];
                hrow(y, &mut buf);
                buf
            }
        }
    };
    for (d, &(s, b0, b1)) in ytaps.iter().enumerate() {
        let r0 = s.clamp(0, sh as i64 - 1) as usize;
        let r1 = (s + 1).clamp(0, sh as i64 - 1) as usize;
        if cache[0].0 != r0 || cache[1].0 != r1 {
            let row0 = fetch(r0, &cache);
            let row1 = fetch(r1, &cache);
            cache = [(r0, row0), (r1, row1)];
        }
        let (h0, h1) = (&cache[0].1, &cache[1].1);
        let dst = &mut out.data[d * w * 3..(d + 1) * w * 3];
        for i in 0..w * 3 {
            let v = ((b0 * (h0[i] >> 4)) >> 16) + ((b1 * (h1[i] >> 4)) >> 16) + 2;
            dst[i] = (v >> 2) as u8;
        }
    }
    out
}

/// `cv2.getPerspectiveTransform(src, dst)` (`DECOMP_LU`): the 3x3 map as
/// row-major `f64`, `M[8] = 1`.
pub fn perspective_transform(src: &[[f32; 2]; 4], dst: &[[f32; 2]; 4]) -> Option<[f64; 9]> {
    let mut a = [[0f64; 8]; 8];
    let mut b = [0f64; 8];
    for i in 0..4 {
        let (sx, sy) = (src[i][0], src[i][1]);
        let (dx, dy) = (dst[i][0], dst[i][1]);
        a[i][0] = sx as f64;
        a[i + 4][3] = sx as f64;
        a[i][1] = sy as f64;
        a[i + 4][4] = sy as f64;
        a[i][2] = 1.0;
        a[i + 4][5] = 1.0;
        a[i][6] = (-sx * dx) as f64;
        a[i][7] = (-sy * dx) as f64;
        a[i + 4][6] = (-sx * dy) as f64;
        a[i + 4][7] = (-sy * dy) as f64;
        b[i] = dx as f64;
        b[i + 4] = dy as f64;
    }
    // OpenCV `LUImpl` (partial pivoting, FMA-contracted updates).
    let eps = f64::EPSILON * 100.0;
    for i in 0..8 {
        let mut k = i;
        for j in i + 1..8 {
            if a[j][i].abs() > a[k][i].abs() {
                k = j;
            }
        }
        if a[k][i].abs() < eps {
            return None;
        }
        if k != i {
            a.swap(i, k);
            b.swap(i, k);
        }
        let d = -1.0 / a[i][i];
        for j in i + 1..8 {
            let alpha = a[j][i] * d;
            let ai = a[i];
            for (v, &p) in a[j].iter_mut().zip(&ai).skip(i + 1) {
                *v = alpha.mul_add(p, *v);
            }
            b[j] = alpha.mul_add(b[i], b[j]);
        }
    }
    for i in (0..8).rev() {
        let mut s = b[i];
        for c in i + 1..8 {
            s = (-a[i][c]).mul_add(b[c], s);
        }
        b[i] = s / a[i][i];
    }
    Some([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], 1.0])
}

/// `cv::invert` of a 3x3 matrix (`DECOMP_LU` small-matrix cofactor path).
pub fn invert3(m: &[f64; 9]) -> Option<[f64; 9]> {
    let g = |y: usize, x: usize| m[y * 3 + x];
    let det = g(0, 0) * (g(1, 1) * g(2, 2) - g(1, 2) * g(2, 1))
        - g(0, 1) * (g(1, 0) * g(2, 2) - g(1, 2) * g(2, 0))
        + g(0, 2) * (g(1, 0) * g(2, 1) - g(1, 1) * g(2, 0));
    if det == 0.0 {
        return None;
    }
    let d = 1.0 / det;
    Some([
        (g(1, 1) * g(2, 2) - g(1, 2) * g(2, 1)) * d,
        (g(0, 2) * g(2, 1) - g(0, 1) * g(2, 2)) * d,
        (g(0, 1) * g(1, 2) - g(0, 2) * g(1, 1)) * d,
        (g(1, 2) * g(2, 0) - g(1, 0) * g(2, 2)) * d,
        (g(0, 0) * g(2, 2) - g(0, 2) * g(2, 0)) * d,
        (g(0, 2) * g(1, 0) - g(0, 0) * g(1, 2)) * d,
        (g(1, 0) * g(2, 1) - g(1, 1) * g(2, 0)) * d,
        (g(0, 1) * g(2, 0) - g(0, 0) * g(2, 1)) * d,
        (g(0, 0) * g(1, 1) - g(0, 1) * g(1, 0)) * d,
    ])
}

/// OpenCV's 1-D bicubic weights (`A = -0.75`) at `x` in f32.
fn cubic_coeffs(x: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let c0 = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
    let c1 = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let c2 = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    let c3 = 1.0 - c0 - c1 - c2;
    [c0, c1, c2, c3]
}

/// `initInterTab2D(INTER_CUBIC)`: 32x32 sub-pixel positions x 16 taps in
/// `INTER_REMAP_COEF_SCALE = 32768` fixed point, with OpenCV's sum fix-up.
fn bicubic_tab() -> Vec<[i32; 16]> {
    let t1: Vec<[f32; 4]> = (0..32)
        .map(|i| cubic_coeffs(i as f32 * (1.0f32 / 32.0)))
        .collect();
    let mut tab = vec![[0i32; 16]; 1024];
    for i in 0..32 {
        for j in 0..32 {
            let mut it = [0i32; 16];
            for k1 in 0..4 {
                for k2 in 0..4 {
                    let v = t1[i][k1] * t1[j][k2];
                    it[k1 * 4 + k2] = (v * 32768.0).round_ties_even() as i32;
                }
            }
            let isum: i32 = it.iter().sum();
            if isum != 32768 {
                let diff = isum - 32768;
                let (mut mk1, mut mk2, mut big1, mut big2) = (2, 2, 2, 2);
                for k1 in 2..4 {
                    for k2 in 2..4 {
                        if it[k1 * 4 + k2] < it[mk1 * 4 + mk2] {
                            (mk1, mk2) = (k1, k2);
                        } else if it[k1 * 4 + k2] > it[big1 * 4 + big2] {
                            (big1, big2) = (k1, k2);
                        }
                    }
                }
                if diff < 0 {
                    it[big1 * 4 + big2] -= diff;
                } else {
                    it[mk1 * 4 + mk2] -= diff;
                }
            }
            tab[i * 32 + j] = it;
        }
    }
    tab
}

fn bicubic_table() -> &'static [[i32; 16]] {
    static TAB: std::sync::OnceLock<Vec<[i32; 16]>> = std::sync::OnceLock::new();
    TAB.get_or_init(bicubic_tab)
}

/// `cv2.warpPerspective(img, M, (w, h), flags=INTER_CUBIC,
/// borderMode=BORDER_REPLICATE)` where `inv` is the already inverted map
/// (destination -> source), as `warpPerspective` uses internally.
pub fn warp_perspective_cubic(img: &BgrImage, inv: &[f64; 9], w: usize, h: usize) -> BgrImage {
    let tab = bicubic_table();
    let mut out = BgrImage::zeros(w, h);
    if w == 0 || h == 0 {
        return out;
    }
    let (sw, sh) = (img.width as i64, img.height as i64);
    let m = inv;
    let mut bh0 = 16.min(h);
    let bw0 = (1024 / bh0).min(w);
    bh0 = (1024 / bw0).min(h);
    let clamp_i = |v: f64| v.clamp(i32::MIN as f64, i32::MAX as f64);
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            let bw = bw0.min(w - x);
            let bh = bh0.min(h - y);
            for y1 in 0..bh {
                let yy = (y + y1) as f64;
                let xf = x as f64;
                let x0 = m[0] * xf + m[1] * yy + m[2];
                let ys0 = m[3] * xf + m[4] * yy + m[5];
                let w0 = m[6] * xf + m[7] * yy + m[8];
                for x1 in 0..bw {
                    let x1f = x1 as f64;
                    let mut ww = m[6].mul_add(x1f, w0);
                    ww = if ww != 0.0 { 32.0 / ww } else { 0.0 };
                    let fx = clamp_i(m[0].mul_add(x1f, x0) * ww);
                    let fy = clamp_i(m[3].mul_add(x1f, ys0) * ww);
                    let xi = fx.round_ties_even() as i64;
                    let yi = fy.round_ties_even() as i64;
                    let sx = (xi >> 5) - 1;
                    let sy = (yi >> 5) - 1;
                    let wt = &tab[((yi & 31) * 32 + (xi & 31)) as usize];
                    let xs: [usize; 4] =
                        std::array::from_fn(|i| (sx + i as i64).clamp(0, sw - 1) as usize);
                    let ys: [usize; 4] =
                        std::array::from_fn(|i| (sy + i as i64).clamp(0, sh - 1) as usize);
                    let o = ((y + y1) * w + x + x1) * 3;
                    for c in 0..3 {
                        let mut s: i64 = 0;
                        for (k1, &py) in ys.iter().enumerate() {
                            for (k2, &px) in xs.iter().enumerate() {
                                s += img.at(px, py, c) as i64 * wt[k1 * 4 + k2] as i64;
                            }
                        }
                        out.data[o + c] = ((s + (1 << 14)) >> 15).clamp(0, 255) as u8;
                    }
                }
            }
            x += bw0;
        }
        y += bh0;
    }
    out
}

/// `np.rot90(img)` (one counter-clockwise quarter turn): output is
/// `[W, H]` with `out[i][j] = img[j][W - 1 - i]`.
pub fn rot90(img: &BgrImage) -> BgrImage {
    let (w, h) = (img.width, img.height);
    let mut out = BgrImage::zeros(h, w);
    for i in 0..w {
        for j in 0..h {
            for c in 0..3 {
                out.data[(i * h + j) * 3 + c] = img.at(w - 1 - i, j, c);
            }
        }
    }
    out
}

/// mlx4ocr `get_rotate_crop_image`: rectify the quad `points` to a
/// `crop_w x crop_h` patch (`int(max(edge lengths))`), rotating tall crops
/// (`h / w >= 1.5`) a quarter turn counter-clockwise.
pub fn rotate_crop(img: &BgrImage, points: &[[f32; 2]; 4]) -> BgrImage {
    let dist = |a: [f32; 2], b: [f32; 2]| {
        let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
        (dx * dx + dy * dy).sqrt()
    };
    let crop_w = dist(points[0], points[1]).max(dist(points[2], points[3])) as usize;
    let crop_h = dist(points[0], points[3]).max(dist(points[1], points[2])) as usize;
    let (cw, ch) = (crop_w as f32, crop_h as f32);
    let std = [[0.0, 0.0], [cw, 0.0], [cw, ch], [0.0, ch]];
    let crop = match perspective_transform(points, &std).and_then(|m| invert3(&m)) {
        Some(inv) => warp_perspective_cubic(img, &inv, crop_w, crop_h),
        None => BgrImage::zeros(crop_w, crop_h),
    };
    if crop.width > 0 && crop.height as f32 / crop.width as f32 >= 1.5 {
        rot90(&crop)
    } else {
        crop
    }
}
