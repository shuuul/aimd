//! DB text-detection post-processing (mlx4ocr `postprocess/db.py`), with
//! the OpenCV / pyclipper / shapely primitives it calls ported from source:
//!
//! - [`find_contours`]: `cv2.findContours(RETR_LIST, CHAIN_APPROX_SIMPLE)`
//!   (OpenCV 4.13 `contours_new.cpp`, Suzuki border following), including
//!   hole borders and OpenCV's output order (reverse discovery order).
//! - [`min_area_rect`] / [`box_points`]: `cv2.minAreaRect` (Sklansky
//!   convex hull + rotating calipers, `convhull.cpp` / `rotcalipers.cpp`)
//!   and `cv2.boxPoints`, with clang's FMA contraction mirrored.
//! - [`box_score_fast`]: `cv2.fillPoly` (`drawing.cpp` edge collection plus
//!   the 8-connected outline) and masked `cv2.mean`.
//! - [`unclip`]: shapely area/length (GEOS) and pyclipper
//!   `PyclipperOffset(JT_ROUND, ET_CLOSEDPOLYGON)` (Clipper 6.4.2
//!   `DoOffset`); the final Clipper union of a single convex offset ring
//!   reduces to dropping duplicate and collinear vertices.

/// DB post-process knobs (`inference.yml` `PostProcess`).
#[derive(Clone, Copy, Debug)]
pub struct DbParams {
    pub thresh: f32,
    pub box_thresh: f64,
    pub max_candidates: usize,
    pub unclip_ratio: f64,
}

impl Default for DbParams {
    fn default() -> Self {
        Self {
            thresh: 0.2,
            box_thresh: 0.45,
            max_candidates: 3000,
            unclip_ratio: 1.4,
        }
    }
}

// ---------------------------------------------------------------------------
// findContours (RETR_LIST, CHAIN_APPROX_SIMPLE)
// ---------------------------------------------------------------------------

const CHAIN_DELTAS: [(i32, i32); 8] = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];
const MASK8_NEW: i8 = 2;
const MASK8_RIGHT_NEW: i8 = (0x80u8 | 0x02) as i8;

/// Traces one border starting at `start` (bordered-image coordinates) and
/// returns its CHAIN_APPROX_SIMPLE points in source coordinates
/// (`icvFetchContourEx<schar>` with `nbd = MASK8_NEW`).
fn fetch_contour(img: &mut [i8], step: i64, start: (i64, i64), is_hole: bool) -> Vec<[i32; 2]> {
    let delta = |s: i32| -> i64 {
        let (dx, dy) = CHAIN_DELTAS[(s % 8) as usize];
        dx as i64 + dy as i64 * step
    };
    let mut pts = Vec::new();
    let mut pt = (start.0 as i32 - 1, start.1 as i32 - 1);
    let i0 = start.1 * step + start.0;
    let s_end0 = if is_hole { 0 } else { 4 };
    let mut s = s_end0;
    let mut i1;
    loop {
        s = (s - 1) & 7;
        i1 = i0 + delta(s);
        if img[i1 as usize] != 0 || s == s_end0 {
            break;
        }
    }
    if s == s_end0 {
        img[i0 as usize] = MASK8_RIGHT_NEW;
        pts.push([pt.0, pt.1]);
        return pts;
    }
    let mut i3 = i0;
    let mut prev_s = s ^ 4;
    loop {
        let s_end = s;
        s = s.min(15);
        let mut i4 = i3;
        while s < 15 {
            s += 1;
            i4 = i3 + delta(s);
            if img[i4 as usize] != 0 {
                break;
            }
        }
        s &= 7;
        if ((s - 1) as u32) < (s_end as u32) {
            img[i3 as usize] = MASK8_RIGHT_NEW;
        } else if img[i3 as usize] == 1 {
            img[i3 as usize] = MASK8_NEW;
        }
        if s != prev_s {
            pts.push([pt.0, pt.1]);
        }
        prev_s = s;
        pt.0 += CHAIN_DELTAS[s as usize].0;
        pt.1 += CHAIN_DELTAS[s as usize].1;
        if i4 == i0 && i3 == i1 {
            break;
        }
        i3 = i4;
        s = (s + 4) & 7;
    }
    pts
}

/// `cv2.findContours(bitmap, RETR_LIST, CHAIN_APPROX_SIMPLE)` on a `w x h`
/// row-major mask (non-zero = foreground). Contours come back in OpenCV's
/// order (last discovered first) as `[x, y]` integer points.
pub fn find_contours(mask: &[bool], w: usize, h: usize) -> Vec<Vec<[i32; 2]>> {
    let (bw, bh) = (w + 2, h + 2);
    let mut img = vec![0i8; bw * bh];
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] {
                img[(y + 1) * bw + x + 1] = 1;
            }
        }
    }
    let step = bw as i64;
    let (width, height) = (bw as i64 - 1, bh as i64 - 1);
    let mut found = Vec::new();
    let (mut x, mut y) = (1i64, 1i64);
    let at = |img: &[i8], x: i64, y: i64| img[(y * step + x) as usize];
    'scan: loop {
        let mut prev = at(&img, x - 1, y);
        while y < height {
            while x < width {
                let mut p = at(&img, x, y);
                while x < width {
                    p = at(&img, x, y);
                    if p != prev {
                        break;
                    }
                    x += 1;
                }
                if x >= width {
                    break;
                }
                let is_hole = if prev == 0 && p == 1 {
                    false
                } else if p != 0 || prev < 1 {
                    prev = p;
                    x += 1;
                    continue;
                } else {
                    true
                };
                let start = (x - i64::from(is_hole), y);
                found.push(fetch_contour(&mut img, step, start, is_hole));
                x += 1;
                continue 'scan;
            }
            y += 1;
            x = 1;
            prev = 0;
        }
        break;
    }
    found.reverse();
    found
}

// ---------------------------------------------------------------------------
// convexHull + minAreaRect + boxPoints
// ---------------------------------------------------------------------------

#[inline]
fn cv_sign(v: f64) -> i32 {
    i32::from(v > 0.0) - i32::from(v < 0.0)
}

/// Point set for the hull: integer contours (`CV_32S`) or float boxes
/// (`CV_32F`), which take different Sklansky arithmetic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Depth {
    Int,
    Float,
}

/// `Sklansky_` over the sorted point array `p` (values are exact f32 / i32
/// held in f64).
fn sklansky(
    p: &[[f64; 2]],
    start: i64,
    end: i64,
    stack: &mut [i64],
    nsign: i32,
    sign2: i32,
    depth: Depth,
) -> usize {
    let incr = if end > start { 1 } else { -1 };
    let (mut pprev, mut pcur) = (start, start + incr);
    let mut pnext = pcur + incr;
    let mut stacksize = 3usize;
    let pt = |i: i64| p[i as usize];
    if start == end || (pt(start)[0] == pt(end)[0] && pt(start)[1] == pt(end)[1]) {
        stack[0] = start;
        return 1;
    }
    stack[0] = pprev;
    stack[1] = pcur;
    stack[2] = pnext;
    let end = end + incr;
    let sub = |a: f64, b: f64| -> f64 {
        match depth {
            Depth::Int => a - b,
            Depth::Float => (a as f32 - b as f32) as f64,
        }
    };
    while pnext != end {
        let cury = pt(pcur)[1];
        let nexty = pt(pnext)[1];
        let by = sub(nexty, cury);
        if cv_sign(by) != nsign {
            let mut a = [sub(pt(pcur)[0], pt(pprev)[0]), sub(cury, pt(pprev)[1])];
            let mut b = [sub(pt(pnext)[0], pt(pcur)[0]), by];
            if depth == Depth::Float {
                for v in [&mut a, &mut b] {
                    let nv = (v[0] * v[0] + v[1] * v[1]).sqrt();
                    let alpha = if nv != 0.0 { 1.0 / nv } else { 0.0 };
                    v[0] = (v[0] * alpha) as f32 as f64;
                    v[1] = (v[1] * alpha) as f32 as f64;
                }
            }
            let convexity = a[1] * b[0] - a[0] * b[1];
            if cv_sign(convexity) == sign2 && (a[0] != 0.0 || a[1] != 0.0) {
                pprev = pcur;
                pcur = pnext;
                pnext += incr;
                stack[stacksize] = pnext;
                stacksize += 1;
            } else if pprev == start {
                pcur = pnext;
                stack[1] = pcur;
                pnext += incr;
                stack[2] = pnext;
            } else {
                stack[stacksize - 2] = pnext;
                pcur = pprev;
                pprev = stack[stacksize - 4];
                stacksize -= 1;
            }
        } else {
            pnext += incr;
            stack[stacksize - 1] = pnext;
        }
    }
    stacksize - 1
}

/// `cv::convexHull(points, clockwise=false, returnPoints=true)`, returning
/// the hull vertices converted to f32.
fn convex_hull(points: &[[f64; 2]], depth: Depth) -> Vec<[f32; 2]> {
    let total = points.len();
    if total == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..total).collect();
    order.sort_by(|&a, &b| {
        let (pa, pb) = (points[a], points[b]);
        pa[0]
            .partial_cmp(&pb[0])
            .unwrap()
            .then(pa[1].partial_cmp(&pb[1]).unwrap())
            .then(a.cmp(&b))
    });
    let p: Vec<[f64; 2]> = order.iter().map(|&i| points[i]).collect();
    let (mut miny, mut maxy) = (0usize, 0usize);
    for i in 1..total {
        let y = p[i][1];
        if p[miny][1] > y {
            miny = i;
        }
        if p[maxy][1] < y {
            maxy = i;
        }
    }
    let mut hull: Vec<i64> = Vec::with_capacity(total);
    let mut stack = vec![0i64; total + 2];
    if p[0] == p[total - 1] {
        hull.push(0);
    } else {
        let tl_count = sklansky(&p, 0, maxy as i64, &mut stack, -1, 1, depth);
        let mut tr = vec![0i64; total + 2];
        let tr_count = sklansky(&p, total as i64 - 1, maxy as i64, &mut tr, -1, -1, depth);
        let tl: Vec<i64> = stack[..tl_count].to_vec();
        let tr: Vec<i64> = tr[..tr_count].to_vec();
        // counter-clockwise: swap the upper stacks
        let (tl, tr) = (tr, tl);
        for &v in tl.iter().take(tl.len().saturating_sub(1)) {
            hull.push(v);
        }
        for i in (1..tr.len()).rev() {
            hull.push(tr[i]);
        }
        let stop_idx = if tr.len() > 2 {
            tr[1]
        } else if tl.len() > 2 {
            tl[tl.len() - 2]
        } else {
            -1
        };

        let mut bl_buf = vec![0i64; total + 2];
        let mut bl_count = sklansky(&p, 0, miny as i64, &mut bl_buf, 1, -1, depth);
        let mut br_buf = vec![0i64; total + 2];
        let mut br_count = sklansky(&p, total as i64 - 1, miny as i64, &mut br_buf, 1, 1, depth);
        let (bl, br) = (&bl_buf, &br_buf);
        if stop_idx >= 0 {
            let check_idx = if bl_count > 2 {
                bl[1]
            } else if bl_count + br_count > 2 {
                br[2 - bl_count]
            } else {
                -1
            };
            if check_idx == stop_idx
                || (check_idx >= 0 && p[check_idx as usize] == p[stop_idx as usize])
            {
                bl_count = bl_count.min(2);
                br_count = br_count.min(2);
            }
        }
        for &v in bl.iter().take(bl_count.saturating_sub(1)) {
            hull.push(v);
        }
        for i in (1..br_count).rev() {
            hull.push(br[i]);
        }
    }
    // Map back to original indices and apply OpenCV's cyclic shift towards
    // a monotonic index sequence.
    let mut idx: Vec<i64> = hull.iter().map(|&h| order[h as usize] as i64).collect();
    let nout = idx.len();
    if nout >= 3 {
        let (mut min_idx, mut max_idx, mut lt) = (0usize, 0usize, 0usize);
        for i in 1..nout {
            let v = idx[i];
            lt += usize::from(idx[i - 1] < v);
            if lt > 1 && lt + 2 <= i {
                break;
            }
            if v < idx[min_idx] {
                min_idx = i;
            }
            if v > idx[max_idx] {
                max_idx = i;
            }
        }
        let mmdist = min_idx.abs_diff(max_idx);
        if (mmdist == 1 || mmdist == nout - 1) && (lt <= 1 || lt + 2 >= nout) {
            let ascending = (max_idx + 1) % nout == min_idx;
            let i0 = if ascending { min_idx } else { max_idx };
            if i0 > 0 {
                let mut shifted = vec![0i64; nout];
                let mut j = i0;
                let mut i = 0;
                while i < nout {
                    let curr = idx[j];
                    shifted[i] = curr;
                    let next_j = if j + 1 < nout { j + 1 } else { 0 };
                    let next = idx[next_j];
                    if i < nout - 1 && (ascending != (curr < next)) {
                        break;
                    }
                    j = next_j;
                    i += 1;
                }
                if i == nout {
                    idx = shifted;
                }
            }
        }
    }
    idx.iter()
        .map(|&i| {
            let q = points[i as usize];
            [q[0] as f32, q[1] as f32]
        })
        .collect()
}

#[inline]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

/// OpenCV `rotatingCalipers(.., CALIPERS_MINAREARECT)` on a CCW hull.
fn rotating_calipers(points: &[[f32; 2]]) -> [[f32; 2]; 3] {
    let n = points.len();
    let mut vect = vec![[0f32; 2]; n];
    let mut inv_len = vec![0f32; n];
    let (mut left, mut bottom, mut right, mut top) = (0usize, 0usize, 0usize, 0usize);
    let mut pt0 = points[0];
    let (mut left_x, mut right_x) = (pt0[0], pt0[0]);
    let (mut top_y, mut bottom_y) = (pt0[1], pt0[1]);
    for i in 0..n {
        if pt0[0] < left_x {
            left_x = pt0[0];
            left = i;
        }
        if pt0[0] > right_x {
            right_x = pt0[0];
            right = i;
        }
        if pt0[1] > top_y {
            top_y = pt0[1];
            top = i;
        }
        if pt0[1] < bottom_y {
            bottom_y = pt0[1];
            bottom = i;
        }
        let pt = points[if i + 1 < n { i + 1 } else { 0 }];
        let dx = (pt[0] - pt0[0]) as f64;
        let dy = (pt[1] - pt0[1]) as f64;
        vect[i] = [dx as f32, dy as f32];
        inv_len[i] = (1.0 / (dx * dx + dy * dy).sqrt()) as f32;
        pt0 = pt;
    }
    let mut seq = [bottom, right, top, left];
    let mut minarea = f32::MAX;
    let mut buf_i = [0usize; 2];
    let mut buf = [0f32; 4]; // base_a, width, base_b, height
    let first_is_right = |v1: [f32; 2], v2: [f32; 2]| {
        let t = [v1[1], -v1[0]];
        fma(t[0], v2[0], t[1] * v2[1]) < 0.0
    };
    for _ in 0..n {
        let v1 = vect[seq[1]];
        let v2 = vect[seq[2]];
        let v3 = vect[seq[3]];
        let rot = [
            vect[seq[0]],
            [v1[1], -v1[0]],
            [-v2[0], -v2[1]],
            [-v3[1], v3[0]],
        ];
        let mut main = 0usize;
        for i in 1..4 {
            if first_is_right(rot[i], rot[main]) {
                main = i;
            }
        }
        let pindex = seq[main];
        let lead_x = vect[pindex][0] * inv_len[pindex];
        let lead_y = vect[pindex][1] * inv_len[pindex];
        let (base_a, base_b) = match main {
            0 => (lead_x, lead_y),
            1 => (lead_y, -lead_x),
            2 => (-lead_x, -lead_y),
            _ => (-lead_y, lead_x),
        };
        seq[main] += 1;
        if seq[main] == n {
            seq[main] = 0;
        }
        let dx = points[seq[1]][0] - points[seq[3]][0];
        let dy = points[seq[1]][1] - points[seq[3]][1];
        let width = fma(dx, base_a, dy * base_b);
        let dx = points[seq[2]][0] - points[seq[0]][0];
        let dy = points[seq[2]][1] - points[seq[0]][1];
        let height = fma(-dx, base_b, dy * base_a);
        let area = width * height;
        if area <= minarea {
            minarea = area;
            buf_i = [seq[3], seq[0]];
            buf = [base_a, width, base_b, height];
        }
    }
    let (a1, b1) = (buf[0], buf[2]);
    let (a2, b2) = (-buf[2], buf[0]);
    let p0 = points[buf_i[0]];
    let p5 = points[buf_i[1]];
    let c1 = fma(a1, p0[0], p0[1] * b1);
    let c2 = fma(a2, p5[0], p5[1] * b2);
    let idet = 1.0 / fma(a1, b2, -(a2 * b1));
    let px = fma(c1, b2, -(c2 * b1)) * idet;
    let py = fma(a1, c2, -(a2 * c1)) * idet;
    [
        [px, py],
        [a1 * buf[1], b1 * buf[1]],
        [a2 * buf[3], b2 * buf[3]],
    ]
}

/// `cv::RotatedRect` as returned by `cv2.minAreaRect`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotatedRect {
    pub center: [f32; 2],
    pub size: [f32; 2],
    pub angle: f32,
}

fn min_area_rect_impl(points: &[[f64; 2]], depth: Depth) -> RotatedRect {
    let hull = convex_hull(points, depth);
    let n = hull.len();
    let mut angle = -std::f64::consts::FRAC_PI_2;
    let mut r = RotatedRect {
        center: [0.0, 0.0],
        size: [0.0, 0.0],
        angle: 0.0,
    };
    if n > 2 {
        let out = rotating_calipers(&hull);
        r.center = [
            out[0][0] + (out[1][0] + out[2][0]) * 0.5,
            out[0][1] + (out[1][1] + out[2][1]) * 0.5,
        ];
        r.size = [
            ((out[2][0] as f64) * out[2][0] as f64 + (out[2][1] as f64) * out[2][1] as f64).sqrt()
                as f32,
            ((out[1][0] as f64) * out[1][0] as f64 + (out[1][1] as f64) * out[1][1] as f64).sqrt()
                as f32,
        ];
        if out[1][0] == 0.0 && out[1][1] > 0.0 {
            r.size.swap(0, 1);
        } else {
            angle = -(out[1][0] as f64).atan2(out[1][1] as f64);
        }
    } else if n == 2 {
        r.center = [
            (hull[0][0] + hull[1][0]) * 0.5,
            (hull[0][1] + hull[1][1]) * 0.5,
        ];
        let dx = (hull[0][0] - hull[1][0]) as f64;
        let dy = (hull[0][1] - hull[1][1]) as f64;
        r.size = [0.0, (dx * dx + dy * dy).sqrt() as f32];
        if dx == 0.0 {
            r.size.swap(0, 1);
        } else if dy < 0.0 {
            angle = dy.atan2(dx);
            r.size.swap(0, 1);
        } else if dy > 0.0 {
            angle = -dx.atan2(dy);
        }
    } else if n == 1 {
        r.center = hull[0];
    }
    r.angle = (angle * 180.0 / std::f64::consts::PI) as f32;
    r
}

/// `cv2.minAreaRect` of an integer point set (a contour).
pub fn min_area_rect_i32(points: &[[i32; 2]]) -> RotatedRect {
    let p: Vec<[f64; 2]> = points.iter().map(|q| [q[0] as f64, q[1] as f64]).collect();
    min_area_rect_impl(&p, Depth::Int)
}

/// `cv2.minAreaRect` of a float32 point set.
pub fn min_area_rect_f32(points: &[[f32; 2]]) -> RotatedRect {
    let p: Vec<[f64; 2]> = points.iter().map(|q| [q[0] as f64, q[1] as f64]).collect();
    min_area_rect_impl(&p, Depth::Float)
}

/// `cv2.boxPoints` (`RotatedRect::points`).
pub fn box_points(r: &RotatedRect) -> [[f32; 2]; 4] {
    let a_rad = r.angle as f64 * std::f64::consts::PI / 180.0;
    let b = a_rad.cos() as f32 * 0.5;
    let a = a_rad.sin() as f32 * 0.5;
    let [cx, cy] = r.center;
    let [w, h] = r.size;
    let p0 = [fma(-b, w, fma(-a, h, cx)), fma(-a, w, fma(b, h, cy))];
    let p1 = [fma(-b, w, fma(a, h, cx)), fma(-a, w, fma(-b, h, cy))];
    let p2 = [2.0 * cx - p0[0], 2.0 * cy - p0[1]];
    let p3 = [2.0 * cx - p1[0], 2.0 * cy - p1[1]];
    [p0, p1, p2, p3]
}

/// mlx4ocr `_mini_boxes`: box corners ordered TL, TR, BR, BL plus the short
/// side of the rectangle.
pub fn mini_boxes(r: &RotatedRect) -> ([[f32; 2]; 4], f32) {
    let mut pts = box_points(r).to_vec();
    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
    let (i1, i4) = if pts[1][1] > pts[0][1] {
        (0, 1)
    } else {
        (1, 0)
    };
    let (i2, i3) = if pts[3][1] > pts[2][1] {
        (2, 3)
    } else {
        (3, 2)
    };
    (
        [pts[i1], pts[i2], pts[i3], pts[i4]],
        r.size[0].min(r.size[1]),
    )
}

// ---------------------------------------------------------------------------
// fillPoly + mean (box score)
// ---------------------------------------------------------------------------

/// `clipLine(Size2l, Point2l&, Point2l&)`.
fn clip_line(w: i64, h: i64, p1: &mut [i64; 2], p2: &mut [i64; 2]) -> bool {
    let (right, bottom) = (w - 1, h - 1);
    if w <= 0 || h <= 0 {
        return false;
    }
    let code = |p: &[i64; 2]| {
        i32::from(p[0] < 0)
            + i32::from(p[0] > right) * 2
            + i32::from(p[1] < 0) * 4
            + i32::from(p[1] > bottom) * 8
    };
    let mut c1 = code(p1);
    let mut c2 = code(p2);
    if (c1 & c2) == 0 && (c1 | c2) != 0 {
        if c1 & 12 != 0 {
            let a = if c1 < 8 { 0 } else { bottom };
            p1[0] += ((a - p1[1]) as f64 * (p2[0] - p1[0]) as f64 / (p2[1] - p1[1]) as f64) as i64;
            p1[1] = a;
            c1 = i32::from(p1[0] < 0) + i32::from(p1[0] > right) * 2;
        }
        if c2 & 12 != 0 {
            let a = if c2 < 8 { 0 } else { bottom };
            p2[0] += ((a - p2[1]) as f64 * (p2[0] - p1[0]) as f64 / (p2[1] - p1[1]) as f64) as i64;
            p2[1] = a;
            c2 = i32::from(p2[0] < 0) + i32::from(p2[0] > right) * 2;
        }
        if (c1 & c2) == 0 && (c1 | c2) != 0 {
            if c1 != 0 {
                let a = if c1 == 1 { 0 } else { right };
                p1[1] +=
                    ((a - p1[0]) as f64 * (p2[1] - p1[1]) as f64 / (p2[0] - p1[0]) as f64) as i64;
                p1[0] = a;
                c1 = 0;
            }
            if c2 != 0 {
                let a = if c2 == 1 { 0 } else { right };
                p2[1] +=
                    ((a - p2[0]) as f64 * (p2[1] - p1[1]) as f64 / (p2[0] - p1[0]) as f64) as i64;
                p2[0] = a;
                c2 = 0;
            }
        }
    }
    (c1 | c2) == 0
}

/// A `u8` mask for [`fill_poly`].
struct Mask {
    w: usize,
    h: usize,
    data: Vec<u8>,
}

/// OpenCV `Line(img, pt1, pt2, color, 8)` (8-connected `LineIterator`,
/// left-to-right).
fn draw_line(m: &mut Mask, a: [i64; 2], b: [i64; 2]) {
    let (w, h) = (m.w as i64, m.h as i64);
    let (mut p1, mut p2) = (a, b);
    if (p1[0] as u64) >= w as u64
        || (p2[0] as u64) >= w as u64
        || (p1[1] as u64) >= h as u64
        || (p2[1] as u64) >= h as u64
    {
        // Point -> Point2l -> Point round trip in `clipLine(Size, Point&, Point&)`.
        if !clip_line(w, h, &mut p1, &mut p2) {
            return;
        }
        p1 = [p1[0] as i32 as i64, p1[1] as i32 as i64];
        p2 = [p2[0] as i32 as i64, p2[1] as i32 as i64];
    }
    let (mut delta_x, mut delta_y) = (1i64, 1i64);
    let mut dx = p2[0] - p1[0];
    let mut dy = p2[1] - p1[1];
    if dx < 0 {
        dx = -dx;
        dy = -dy;
        p1 = p2;
    }
    if dy < 0 {
        dy = -dy;
        delta_y = -1;
    }
    let vert = dy > dx;
    if vert {
        std::mem::swap(&mut dx, &mut dy);
        std::mem::swap(&mut delta_x, &mut delta_y);
    }
    let mut err = dx - (dy + dy);
    let plus_delta = dx + dx;
    let minus_delta = -(dy + dy);
    let (mut minus_shift, mut plus_shift) = (delta_x, 0i64);
    let (mut minus_step, mut plus_step) = (0i64, delta_y);
    let count = dx + 1;
    if vert {
        std::mem::swap(&mut plus_step, &mut plus_shift);
        std::mem::swap(&mut minus_step, &mut minus_shift);
    }
    let mut p = p1;
    for _ in 0..count {
        m.data[(p[1] * w + p[0]) as usize] = 1;
        let mask = if err < 0 { -1i64 } else { 0 };
        err += minus_delta + (plus_delta & mask);
        p[0] += minus_shift + (plus_shift & mask);
        p[1] += minus_step + (plus_step & mask);
    }
}

#[derive(Clone, Copy)]
struct PolyEdge {
    y0: i64,
    y1: i64,
    x: i64,
    dx: i64,
}

/// `cv2.fillPoly(mask, [pts], 1)` for one integer polygon (`LINE_8`,
/// `shift = 0`).
fn fill_poly(m: &mut Mask, v: &[[i64; 2]]) {
    const XY_SHIFT: i64 = 16;
    const XY_ONE: i64 = 1 << XY_SHIFT;
    let (w, h) = (m.w as i64, m.h as i64);
    let count = v.len();
    let mut edges: Vec<PolyEdge> = Vec::with_capacity(count + 1);
    let mut pt0 = [v[count - 1][0] << XY_SHIFT, v[count - 1][1]];
    for &q in v {
        let pt1 = [q[0] << XY_SHIFT, q[1]];
        let (mut pt0c, mut pt1c) = (pt0, pt1);
        let mut t0 = [(pt0[0] + (XY_ONE >> 1)) >> XY_SHIFT, pt0[1]];
        let mut t1 = [(pt1[0] + (XY_ONE >> 1)) >> XY_SHIFT, pt1[1]];
        draw_line(m, t0, t1);
        if (t0[0] as u64) >= w as u64
            || (t1[0] as u64) >= w as u64
            || (t0[1] as u64) >= h as u64
            || (t1[1] as u64) >= h as u64
        {
            clip_line(w, h, &mut t0, &mut t1);
            if t0[1] != t1[1] {
                pt0c[1] = t0[1];
                pt1c[1] = t1[1];
            }
        }
        pt0c[0] = t0[0] << XY_SHIFT;
        pt1c[0] = t1[0] << XY_SHIFT;
        if pt0[1] != pt1[1] {
            let dx = (pt1c[0] - pt0c[0]) / (pt1c[1] - pt0c[1]);
            let e = if pt0[1] < pt1[1] {
                PolyEdge {
                    y0: pt0[1],
                    y1: pt1[1],
                    x: pt0c[0] + (pt0[1] - pt0c[1]) * dx,
                    dx,
                }
            } else {
                PolyEdge {
                    y0: pt1[1],
                    y1: pt0[1],
                    x: pt1c[0] + (pt1[1] - pt1c[1]) * dx,
                    dx,
                }
            };
            edges.push(e);
        }
        pt0 = pt1;
    }
    fill_edge_collection(m, edges);
}

/// OpenCV `FillEdgeCollection` (active-edge scanline fill).
fn fill_edge_collection(m: &mut Mask, mut edges: Vec<PolyEdge>) {
    const XY_SHIFT: i64 = 16;
    const NIL: usize = usize::MAX;
    let (w, h) = (m.w as i64, m.h as i64);
    let total = edges.len();
    if total < 2 {
        return;
    }
    let (mut y_min, mut y_max) = (i64::MAX, i64::MIN);
    let (mut x_min, mut x_max) = (i64::MAX, -1i64);
    for e in &edges {
        let x1 = e.x + (e.y1 - e.y0) * e.dx;
        y_min = y_min.min(e.y0);
        y_max = y_max.max(e.y1);
        x_min = x_min.min(e.x).min(x1);
        x_max = x_max.max(e.x).max(x1);
    }
    if y_max < 0 || y_min >= h || x_max < 0 || x_min >= (w << XY_SHIFT) {
        return;
    }
    edges.sort_by(|a, b| a.y0.cmp(&b.y0).then(a.x.cmp(&b.x)).then(a.dx.cmp(&b.dx)));
    // node 0 = list head `tmp`, 1..=total = edges, total + 1 = sentinel.
    let mut nodes: Vec<PolyEdge> = Vec::with_capacity(total + 2);
    nodes.push(PolyEdge {
        y0: i64::from(i32::MAX),
        y1: 0,
        x: 0,
        dx: 0,
    });
    nodes.extend(edges);
    nodes.push(PolyEdge {
        y0: i64::from(i32::MAX),
        y1: 0,
        x: 0,
        dx: 0,
    });
    let mut next = vec![NIL; total + 2];
    let mut i = 0usize;
    let mut e = 1usize;
    let y_max = y_max.min(h);
    let delta = (1i64 << XY_SHIFT) - 1;
    let mut y = nodes[e].y0;
    while y < y_max {
        let mut draw = false;
        let clipline = y < 0;
        let mut prelast = 0usize;
        let mut last = next[0];
        while last != NIL || nodes[e].y0 == y {
            if last != NIL && nodes[last].y1 == y {
                next[prelast] = next[last];
                last = next[last];
                continue;
            }
            let keep_prelast = prelast;
            if last != NIL && (nodes[e].y0 > y || nodes[last].x < nodes[e].x) {
                prelast = last;
                last = next[last];
            } else if i < total {
                next[prelast] = e;
                next[e] = last;
                prelast = e;
                i += 1;
                e = i + 1;
            } else {
                break;
            }
            if draw {
                if !clipline {
                    let (kx, px) = (nodes[keep_prelast].x, nodes[prelast].x);
                    let (mut x1, mut x2) = if kx > px {
                        ((px + delta) >> XY_SHIFT, kx >> XY_SHIFT)
                    } else {
                        ((kx + delta) >> XY_SHIFT, px >> XY_SHIFT)
                    };
                    if x1 < w && x2 >= 0 {
                        x1 = x1.max(0);
                        x2 = x2.min(w - 1);
                        let row = (y * w) as usize;
                        for x in x1..=x2 {
                            m.data[row + x as usize] = 1;
                        }
                    }
                }
                nodes[keep_prelast].x += nodes[keep_prelast].dx;
                nodes[prelast].x += nodes[prelast].dx;
            }
            draw = !draw;
        }
        // bubble sort of the active list by x
        let mut keep_prelast = NIL;
        loop {
            let mut prelast = 0usize;
            let mut last = next[0];
            let mut last_exchange = NIL;
            while last != keep_prelast && next[last] != NIL {
                let te = next[last];
                if nodes[last].x > nodes[te].x {
                    next[prelast] = te;
                    next[last] = next[te];
                    next[te] = last;
                    prelast = te;
                    last_exchange = prelast;
                } else {
                    prelast = last;
                    last = te;
                }
            }
            if last_exchange == NIL {
                break;
            }
            keep_prelast = last_exchange;
            if !(keep_prelast != next[0] && keep_prelast != 0) {
                break;
            }
        }
        y += 1;
    }
}

/// mlx4ocr `_box_score_fast`: mean probability inside the (int-truncated)
/// box over its clipped bounding rectangle.
pub fn box_score_fast(pred: &[f32], w: usize, h: usize, quad: &[[f32; 2]; 4]) -> f64 {
    let clampi = |v: f32, hi: usize| (v.clamp(0.0, (hi - 1) as f32)) as i64;
    let xs = quad.map(|p| p[0]);
    let ys = quad.map(|p| p[1]);
    let fmin = |a: [f32; 4]| a.iter().copied().fold(f32::INFINITY, f32::min);
    let fmax = |a: [f32; 4]| a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let xmin = clampi(fmin(xs).floor(), w);
    let xmax = clampi(fmax(xs).ceil(), w);
    let ymin = clampi(fmin(ys).floor(), h);
    let ymax = clampi(fmax(ys).ceil(), h);
    let (mw, mh) = ((xmax - xmin + 1) as usize, (ymax - ymin + 1) as usize);
    let mut mask = Mask {
        w: mw,
        h: mh,
        data: vec![0; mw * mh],
    };
    let pts: Vec<[i64; 2]> = quad
        .iter()
        .map(|p| {
            [
                (p[0] - xmin as f32) as i32 as i64,
                (p[1] - ymin as f32) as i32 as i64,
            ]
        })
        .collect();
    fill_poly(&mut mask, &pts);
    let mut s = 0f64;
    let mut n = 0usize;
    for y in 0..mh {
        for x in 0..mw {
            if mask.data[y * mw + x] != 0 {
                s += pred[(ymin as usize + y) * w + xmin as usize + x] as f64;
                n += 1;
            }
        }
    }
    s * if n != 0 { 1.0 / n as f64 } else { 0.0 }
}

// ---------------------------------------------------------------------------
// unclip (shapely + pyclipper)
// ---------------------------------------------------------------------------

/// shapely `Polygon(quad).area` / `.length` (GEOS shoelace / segment sum).
fn geos_area_length(quad: &[[f32; 2]; 4]) -> (f64, f64) {
    let ring: Vec<[f64; 2]> = quad
        .iter()
        .chain(std::iter::once(&quad[0]))
        .map(|p| [p[0] as f64, p[1] as f64])
        .collect();
    let n = ring.len();
    let x0 = ring[0][0];
    let mut p1 = ring[0];
    let mut p2 = [ring[1][0] - x0, ring[1][1]];
    let mut sum = 0.0;
    for i in 1..n - 1 {
        let p0y = p1[1];
        p1 = p2;
        p2 = [ring[i + 1][0] - x0, ring[i + 1][1]];
        sum += p1[0] * (p0y - p2[1]);
    }
    let area = (sum / 2.0).abs();
    let mut len = 0.0;
    for i in 1..n {
        let dx = ring[i][0] - ring[i - 1][0];
        let dy = ring[i][1] - ring[i - 1][1];
        len += (dx * dx + dy * dy).sqrt();
    }
    (area, len)
}

/// Clipper `Round`.
fn clipper_round(v: f64) -> i64 {
    if v < 0.0 {
        (v - 0.5) as i64
    } else {
        (v + 0.5) as i64
    }
}

fn unit_normal(a: [i64; 2], b: [i64; 2]) -> [f64; 2] {
    if a == b {
        return [0.0, 0.0];
    }
    let mut dx = (b[0] - a[0]) as f64;
    let mut dy = (b[1] - a[1]) as f64;
    let f = 1.0 / (dx * dx + dy * dy).sqrt();
    dx *= f;
    dy *= f;
    [dy, -dx]
}

/// Clipper `ClipperOffset` with `JT_ROUND` / `ET_CLOSEDPOLYGON` and the
/// default arc tolerance (0.25), for one closed path.
fn clipper_offset(path: &[[i64; 2]], delta: f64) -> Option<Vec<[i64; 2]>> {
    const PI: f64 = std::f64::consts::PI;
    let mut hi = path.len() as i64 - 1;
    while hi > 0 && path[0] == path[hi as usize] {
        hi -= 1;
    }
    let mut cont = vec![path[0]];
    for &p in path.iter().take(hi as usize + 1).skip(1) {
        if *cont.last().unwrap() != p {
            cont.push(p);
        }
    }
    if cont.len() < 3 {
        return None;
    }
    // FixOrientations (Clipper `Area` >= 0 is the wanted orientation).
    let mut a = 0.0;
    let mut j = cont.len() - 1;
    for i in 0..cont.len() {
        a += (cont[j][0] as f64 + cont[i][0] as f64) * (cont[j][1] as f64 - cont[i][1] as f64);
        j = i;
    }
    if -a * 0.5 < 0.0 {
        cont.reverse();
    }
    if delta.abs() < 1e-20 {
        return Some(cont);
    }
    let mut y = 0.25;
    if y > delta.abs() * 0.25 {
        y = delta.abs() * 0.25;
    }
    let mut steps = PI / (1.0 - y / delta.abs()).acos();
    if steps > delta.abs() * PI {
        steps = delta.abs() * PI;
    }
    let (sn, cs) = ((2.0 * PI / steps).sin(), (2.0 * PI / steps).cos());
    let sn = if delta < 0.0 { -sn } else { sn };
    let steps_per_rad = steps / (2.0 * PI);
    let n = cont.len();
    let mut normals: Vec<[f64; 2]> = (0..n - 1)
        .map(|j| unit_normal(cont[j], cont[j + 1]))
        .collect();
    normals.push(unit_normal(cont[n - 1], cont[0]));
    let mut out = Vec::new();
    let mut k = n - 1;
    let off = |p: [i64; 2], nrm: [f64; 2]| {
        [
            clipper_round(p[0] as f64 + nrm[0] * delta),
            clipper_round(p[1] as f64 + nrm[1] * delta),
        ]
    };
    for j in 0..n {
        let (nk, nj, src) = (normals[k], normals[j], cont[j]);
        let mut sina = nk[0] * nj[1] - nj[0] * nk[1];
        if (sina * delta).abs() < 1.0 {
            let cosa = nk[0] * nj[0] + nj[1] * nk[1];
            if cosa > 0.0 {
                out.push(off(src, nk));
                continue;
            }
        } else {
            sina = sina.clamp(-1.0, 1.0);
        }
        if sina * delta < 0.0 {
            out.push(off(src, nk));
            out.push(src);
            out.push(off(src, nj));
        } else {
            let ang = sina.atan2(nk[0] * nj[0] + nk[1] * nj[1]);
            let st = clipper_round(steps_per_rad * ang.abs()).max(1);
            let (mut x, mut yv) = (nk[0], nk[1]);
            for _ in 0..st {
                out.push(off(src, [x, yv]));
                let x2 = x;
                x = x * cs - sn * yv;
                yv = x2 * sn + yv * cs;
            }
            out.push(off(src, nj));
        }
        k = j;
    }
    Some(out)
}

/// Drops consecutive duplicates and collinear vertices (cyclically): what
/// Clipper's union clean-up leaves of a simple convex ring.
fn clean_ring(pts: Vec<[i64; 2]>) -> Vec<[i64; 2]> {
    let mut pts = pts;
    loop {
        if pts.len() <= 2 {
            return pts;
        }
        let n = pts.len();
        let mut res = Vec::with_capacity(n);
        let mut changed = false;
        for i in 0..n {
            let a = pts[(i + n - 1) % n];
            let b = pts[i];
            let c = pts[(i + 1) % n];
            if a == b {
                changed = true;
                continue;
            }
            let cr = (b[0] - a[0]) as i128 * (c[1] - b[1]) as i128
                - (b[1] - a[1]) as i128 * (c[0] - b[0]) as i128;
            if cr == 0 {
                changed = true;
                continue;
            }
            res.push(b);
        }
        pts = res;
        if !changed {
            return pts;
        }
    }
}

/// mlx4ocr `_unclip`: offset the box outward by `area * ratio / perimeter`.
/// `None` when the offset yields no single polygon.
pub fn unclip(quad: &[[f32; 2]; 4], ratio: f64) -> Option<Vec<[i64; 2]>> {
    let (area, length) = geos_area_length(quad);
    let distance = area * ratio / length;
    // pyclipper converts each coordinate with `int()` (truncation).
    let path: Vec<[i64; 2]> = quad
        .iter()
        .map(|p| [p[0] as f64 as i64, p[1] as f64 as i64])
        .collect();
    let ring = clean_ring(clipper_offset(&path, distance)?);
    if ring.len() < 3 {
        return None;
    }
    Some(ring)
}

// ---------------------------------------------------------------------------
// boxes_from_bitmap
// ---------------------------------------------------------------------------

/// One detected text quad in source-image coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub quad: [[f32; 2]; 4],
    pub score: f64,
}

/// mlx4ocr `db_postprocess` on a `[h, w]` probability map for a source
/// image of `src_w x src_h`.
pub fn db_postprocess(
    pred: &[f32],
    w: usize,
    h: usize,
    src_w: usize,
    src_h: usize,
    params: &DbParams,
) -> Vec<Detection> {
    let mask: Vec<bool> = pred.iter().map(|&v| v > params.thresh).collect();
    let contours = find_contours(&mask, w, h);
    let (dest_w, dest_h) = (src_w as f32, src_h as f32);
    let mut out = Vec::new();
    for contour in contours.iter().take(params.max_candidates) {
        let (quad, short) = mini_boxes(&min_area_rect_i32(contour));
        if short < 3.0 {
            continue;
        }
        let score = box_score_fast(pred, w, h, &quad);
        if params.box_thresh > score {
            continue;
        }
        let Some(ring) = unclip(&quad, params.unclip_ratio) else {
            continue;
        };
        let ring_f: Vec<[f32; 2]> = ring.iter().map(|p| [p[0] as f32, p[1] as f32]).collect();
        let (expanded, short) = mini_boxes(&min_area_rect_f32(&ring_f));
        if short < 5.0 {
            continue;
        }
        let quad = expanded.map(|p| {
            [
                (p[0] / w as f32 * dest_w)
                    .round_ties_even()
                    .clamp(0.0, dest_w),
                (p[1] / h as f32 * dest_h)
                    .round_ties_even()
                    .clamp(0.0, dest_h),
            ]
        });
        out.push(Detection { quad, score });
    }
    out
}

/// mlx4ocr `sorted_box_indices`: top-to-bottom by the first corner, then a
/// bubble pass that swaps neighbours on (nearly) the same line
/// (`|dy| < 10`) into left-to-right order.
pub fn reading_order(quads: &[[[f32; 2]; 4]]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..quads.len()).collect();
    order.sort_by(|&a, &b| {
        let (pa, pb) = (quads[a][0], quads[b][0]);
        pa[1]
            .partial_cmp(&pb[1])
            .unwrap()
            .then(pa[0].partial_cmp(&pb[0]).unwrap())
    });
    for i in 0..order.len().saturating_sub(1) {
        let mut j = i as i64;
        while j >= 0 {
            let (a, b) = (quads[order[j as usize]][0], quads[order[j as usize + 1]][0]);
            if (b[1] - a[1]).abs() < 10.0 && b[0] < a[0] {
                order.swap(j as usize, j as usize + 1);
                j -= 1;
            } else {
                break;
            }
        }
    }
    order
}
