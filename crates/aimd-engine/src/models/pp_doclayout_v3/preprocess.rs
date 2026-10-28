//! SDK preprocessing: `cv2.cvtColor(RGB2BGR)`, `cv2.resize((800, 800),
//! interpolation=cv2.INTER_CUBIC)`, then `x.astype(f32) * (1 / 255)` per
//! channel. The model sees **BGR** channels in [0, 1].
//!
//! [`resize_cubic_u8`] is a port of OpenCV 4.13's generic `INTER_CUBIC`
//! path for 8-bit images (`resize.cpp`: `interpolateCubic` with A = -0.75,
//! 11-bit fixed-point taps, integer horizontal pass with clamped columns,
//! the `VResizeCubicVec_32s8u` float vertical pass with clamped rows), so
//! the resized pixels match `cv2.resize` bit for bit (checked against the
//! golden `*_resized.png`).

use super::config::INPUT_SIZE;

/// `INTER_RESIZE_COEF_SCALE` (11 fractional bits).
const COEF_SCALE: f32 = 2048.0;

/// OpenCV `interpolateCubic` (A = -0.75), in f32.
fn cubic_coeffs(x: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let c0 = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
    let c1 = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let c2 = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    let c3 = 1.0 - c0 - c1 - c2;
    [c0, c1, c2, c3]
}

/// `saturate_cast<short>(c * INTER_RESIZE_COEF_SCALE)` (`cvRound`: ties to
/// even).
fn fixed_taps(c: [f32; 4]) -> [i32; 4] {
    c.map(|v| (v * COEF_SCALE).round_ties_even().clamp(-32768.0, 32767.0) as i32)
}

/// Per output index: the source index `floor(f)` and the fixed-point taps
/// for `f - floor(f)`, with `f = (float)((d + 0.5) * scale - 0.5)` and
/// `scale = 1 / (dst / src)` in double.
fn axis_taps(src: usize, dst: usize) -> Vec<(i64, [i32; 4])> {
    let inv_scale = dst as f64 / src as f64;
    let scale = 1.0 / inv_scale;
    (0..dst)
        .map(|d| {
            let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let s = f.floor();
            (s as i64, fixed_taps(cubic_coeffs(f - s)))
        })
        .collect()
}

/// `cv2.resize(src, (dw, dh), interpolation=cv2.INTER_CUBIC)` for an
/// interleaved 8-bit image with `cn` channels.
pub fn resize_cubic_u8(
    src: &[u8],
    sw: usize,
    sh: usize,
    cn: usize,
    dw: usize,
    dh: usize,
) -> Vec<u8> {
    assert_eq!(src.len(), sw * sh * cn, "source buffer size");
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let xt = axis_taps(sw, dw);
    let yt = axis_taps(sh, dh);
    let clamp = |v: i64, n: usize| v.clamp(0, n as i64 - 1) as usize;
    // Horizontal pass of one source row into `dw * cn` integers.
    let hresize = |row: usize| -> Vec<i32> {
        let s = &src[row * sw * cn..(row + 1) * sw * cn];
        let mut out = vec![0i32; dw * cn];
        for (dx, &(sx, a)) in xt.iter().enumerate() {
            let cols = [0i64, 1, 2, 3].map(|j| clamp(sx - 1 + j, sw));
            for k in 0..cn {
                out[dx * cn + k] = (0..4).map(|j| i32::from(s[cols[j] * cn + k]) * a[j]).sum();
            }
        }
        out
    };
    let rows: Vec<Vec<i32>> = (0..sh).map(hresize).collect();
    let scale = 1.0f32 / (COEF_SCALE * COEF_SCALE);
    let mut dst = vec![0u8; dw * dh * cn];
    for (dy, &(sy, beta)) in yt.iter().enumerate() {
        let r = [0i64, 1, 2, 3].map(|k| &rows[clamp(sy - 1 + k, sh)]);
        let b = beta.map(|v| v as f32 * scale);
        let out = &mut dst[dy * dw * cn..(dy + 1) * dw * cn];
        for (x, o) in out.iter_mut().enumerate() {
            // v_muladd(S0, b0, v_muladd(S1, b1, v_muladd(S2, b2, S3 * b3)))
            let mut acc = r[3][x] as f32 * b[3];
            acc = (r[2][x] as f32).mul_add(b[2], acc);
            acc = (r[1][x] as f32).mul_add(b[1], acc);
            acc = (r[0][x] as f32).mul_add(b[0], acc);
            *o = acc.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
    dst
}

/// The SDK's model input for an RGB page: `[800, 800, 3]` BGR f32 in
/// [0, 1] (NHWC without the batch axis), plus the resized BGR bytes.
pub fn preprocess(rgb: &image::RgbImage) -> (Vec<f32>, Vec<u8>) {
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let bgr: Vec<u8> = rgb
        .as_raw()
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|p| [p[2], p[1], p[0]])
        .collect();
    let resized = resize_cubic_u8(&bgr, w, h, 3, INPUT_SIZE, INPUT_SIZE);
    let inv = (1.0f64 / 255.0) as f32;
    let pixels = resized.iter().map(|&v| f32::from(v) * inv).collect();
    (pixels, resized)
}
