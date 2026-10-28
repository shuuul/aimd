//! Bilinear grid sampling (spec 009 WS-04).
//!
//! PP-DocLayoutV3's multi-scale deformable attention samples each feature
//! level at fractional locations with torch `F.grid_sample` (bilinear, zeros
//! padding, `align_corners=False`). The shim builds it from stock gathers,
//! so it runs on every backend.

use crate::{MlxArray, MlxError};

mod sys {
    use crate::ffi::AimdMlxArray;

    unsafe extern "C" {
        pub fn mlx_grid_sample(x: *mut AimdMlxArray, grid: *mut AimdMlxArray) -> *mut AimdMlxArray;
    }
}

impl MlxArray {
    /// Bilinear sampling of channels-last `self` (`[N, H, W, C]`) at
    /// normalized `grid` locations (`[N, Hg, Wg, 2]`, `(x, y)` in [-1, 1]),
    /// returning `[N, Hg, Wg, C]`. Matches torch `F.grid_sample` with
    /// `mode="bilinear"`, `padding_mode="zeros"`, `align_corners=False`:
    /// corners outside the input contribute zero.
    pub fn grid_sample(&self, grid: &Self) -> Result<Self, MlxError> {
        // SAFETY: both handles are valid for the duration of the call; the
        // shim returns a new owned handle or NULL on error.
        let ptr = unsafe { sys::mlx_grid_sample(self.ptr.as_ptr(), grid.ptr.as_ptr()) };
        Self::from_handle(ptr, "mlx_grid_sample")
    }
}
