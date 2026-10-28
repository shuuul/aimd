//! Unlimited-OCR-2 DASH attention ops (spec 008).
//!
//! Safe wrappers over the shim's `mlx_dash_*` entry points: page2d RoPE, the
//! per-layer router, block summaries, and the block-sparse prefill / gathered
//! decode attention. On Metal the attention runs two custom kernels; other
//! devices run the same selection from mlx ops. No path builds a dense
//! `[L, S]` mask except [`DashMode::Dense`], which exists for tests.

use crate::{MlxArray, MlxDtype, MlxError, ffi, op_failed};

mod sys {
    use crate::ffi::AimdMlxArray;

    unsafe extern "C" {
        pub fn mlx_dash_rope(
            tok: *mut AimdMlxArray,
            page: *mut AimdMlxArray,
            inv_tok: *mut AimdMlxArray,
            inv_page: *mut AimdMlxArray,
            page_mod: i32,
            dtype: i32,
            out_cos: *mut *mut AimdMlxArray,
            out_sin: *mut *mut AimdMlxArray,
        ) -> i32;
        pub fn mlx_dash_apply_rope(
            x: *mut AimdMlxArray,
            cos: *mut AimdMlxArray,
            sin: *mut AimdMlxArray,
            decode: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_dash_index_qk(
            x: *mut AimdMlxArray,
            q_w: *mut AimdMlxArray,
            k_w: *mut AimdMlxArray,
            q_norm_w: *mut AimdMlxArray,
            k_norm_w: *mut AimdMlxArray,
            cos: *mut AimdMlxArray,
            sin: *mut AimdMlxArray,
            eps: f32,
            decode: i32,
            out_q: *mut *mut AimdMlxArray,
            out_k: *mut *mut AimdMlxArray,
        ) -> i32;
        pub fn mlx_dash_decode_prologue(
            x: *mut AimdMlxArray,
            q: *mut AimdMlxArray,
            k: *mut AimdMlxArray,
            q_w: *mut AimdMlxArray,
            k_w: *mut AimdMlxArray,
            q_norm_w: *mut AimdMlxArray,
            k_norm_w: *mut AimdMlxArray,
            cos: *mut AimdMlxArray,
            sin: *mut AimdMlxArray,
            eps: f32,
            out_q: *mut *mut AimdMlxArray,
            out_k: *mut *mut AimdMlxArray,
            out_iq: *mut *mut AimdMlxArray,
            out_ik: *mut *mut AimdMlxArray,
        ) -> i32;
        pub fn mlx_dash_summarize(
            blocks: *mut AimdMlxArray,
            head_cls: *mut AimdMlxArray,
        ) -> *mut AimdMlxArray;
        pub fn mlx_dash_prefill(
            q: *mut AimdMlxArray,
            keys: *mut AimdMlxArray,
            values: *mut AimdMlxArray,
            idx_q: *mut AimdMlxArray,
            summaries: *mut AimdMlxArray,
            cand: *mut AimdMlxArray,
            offset: i32,
            block_size: i32,
            local_blocks: i32,
            topm: i32,
            topk: i32,
            sigma: f32,
            idx_scale: f32,
            floor: f32,
            scale: f32,
            mode: i32,
            out: *mut *mut AimdMlxArray,
            out_topi: *mut *mut AimdMlxArray,
            out_bias: *mut *mut AimdMlxArray,
            out_valid: *mut *mut AimdMlxArray,
        ) -> i32;
        pub fn mlx_dash_decode(
            q: *mut AimdMlxArray,
            keys: *mut AimdMlxArray,
            values: *mut AimdMlxArray,
            idx_q: *mut AimdMlxArray,
            summaries: *mut AimdMlxArray,
            cand: *mut AimdMlxArray,
            pos: *mut AimdMlxArray,
            block_size: i32,
            local_blocks: i32,
            topm: i32,
            topk: i32,
            sigma: f32,
            idx_scale: f32,
            floor: f32,
            scale: f32,
            mode: i32,
        ) -> *mut AimdMlxArray;
        pub fn mlx_dash_decode_ctx(
            token: *mut AimdMlxArray,
            markers: *mut AimdMlxArray,
            page: *mut AimdMlxArray,
            tok: *mut AimdMlxArray,
            pf_cnt: *mut AimdMlxArray,
            blk_min: *mut AimdMlxArray,
            blk_max: *mut AimdMlxArray,
            inv_tok: *mut AimdMlxArray,
            inv_page: *mut AimdMlxArray,
            page_token_id: i32,
            n_pages: i32,
            prompt_pages: i32,
            page_mod: i32,
            image_only: i32,
            cap: i32,
            dtype: i32,
            out_markers: *mut *mut AimdMlxArray,
            out_page: *mut *mut AimdMlxArray,
            out_tok: *mut *mut AimdMlxArray,
            out_cos: *mut *mut AimdMlxArray,
            out_sin: *mut *mut AimdMlxArray,
            out_cand: *mut *mut AimdMlxArray,
        ) -> i32;
    }
}

/// Block routing knobs shared by every layer (`DashParams`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DashParams {
    pub block_size: i32,
    pub local_blocks: i32,
    pub topm: i32,
    pub topk: i32,
    pub sigma: f32,
}

/// Per-layer router constants plus the attention scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DashLayerConsts {
    /// `dash_scale / sqrt(index_dim)`.
    pub idx_scale: f32,
    /// Prior floor of `log(max(prior, floor))`.
    pub floor: f32,
    /// Attention softmax scale (`head_dim^-0.5`).
    pub scale: f32,
}

/// Attention implementation of [`MlxArray::dash_prefill`] / [`MlxArray::dash_decode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DashMode {
    /// Custom Metal kernel where supported, else the mlx-op sparse path.
    Auto = 0,
    /// Always the mlx-op sparse path (tile-union gather / gathered sdpa).
    Gather = 1,
    /// Dense masked sdpa (prefill only): the test oracle.
    Dense = 2,
}

/// `route_select` output of a prefill call, each `[L, K]`.
#[derive(Debug)]
pub struct DashRoute {
    /// Selected remote blocks (u32).
    pub topi: MlxArray,
    /// Additive bias per selected block (f32).
    pub bias: MlxArray,
    /// Whether each selected block is a valid candidate (bool).
    pub valid: MlxArray,
}

/// Next decode-step state from [`MlxArray::dash_decode_ctx`].
#[derive(Debug)]
pub struct DashDecodeCtx {
    pub markers: MlxArray,
    pub page: MlxArray,
    pub tok: MlxArray,
    pub cos: MlxArray,
    pub sin: MlxArray,
    pub cand: MlxArray,
}

/// Prompt-derived inputs of [`MlxArray::dash_decode_ctx`].
#[derive(Debug)]
pub struct DashPromptState<'a> {
    /// `[n_pages]` int32 prefill tokens per page.
    pub pf_cnt: &'a MlxArray,
    /// `[nblk_prompt]` int32 decode page range of each prompt block.
    pub blk_min: &'a MlxArray,
    pub blk_max: &'a MlxArray,
    pub inv_tok: &'a MlxArray,
    pub inv_page: &'a MlxArray,
    pub page_token_id: i32,
    pub n_pages: i32,
    pub prompt_pages: i32,
    pub page_mod: i32,
    pub image_only: bool,
}

fn opt_ptr(a: Option<&MlxArray>) -> *mut ffi::AimdMlxArray {
    a.map_or(std::ptr::null_mut(), |a| a.ptr.as_ptr())
}

fn status(code: i32, op: &str) -> Result<(), MlxError> {
    if code == 0 {
        Ok(())
    } else {
        Err(op_failed(op))
    }
}

impl MlxArray {
    /// page2d RoPE tables: `tok`/`page` int32 `[L]` and the f32 inverse
    /// frequencies -> `(cos, sin)` `[1, 1, L, D]` in `dtype`. `page_mod > 0`
    /// folds the page coordinate.
    pub fn dash_rope(
        tok: &Self,
        page: &Self,
        inv_tok: &Self,
        inv_page: &Self,
        page_mod: i32,
        dtype: MlxDtype,
    ) -> Result<(Self, Self), MlxError> {
        let (mut cos, mut sin) = (std::ptr::null_mut(), std::ptr::null_mut());
        let code = unsafe {
            sys::mlx_dash_rope(
                tok.ptr.as_ptr(),
                page.ptr.as_ptr(),
                inv_tok.ptr.as_ptr(),
                inv_page.ptr.as_ptr(),
                page_mod,
                dtype.code(),
                &mut cos,
                &mut sin,
            )
        };
        status(code, "mlx_dash_rope")?;
        Ok((
            Self::from_handle(cos, "mlx_dash_rope")?,
            Self::from_handle(sin, "mlx_dash_rope")?,
        ))
    }

    /// `x * cos + rotate_half(x) * sin`; `decode` runs the compiled trace.
    pub fn dash_apply_rope(&self, cos: &Self, sin: &Self, decode: bool) -> Result<Self, MlxError> {
        let ptr = unsafe {
            sys::mlx_dash_apply_rope(
                self.ptr.as_ptr(),
                cos.ptr.as_ptr(),
                sin.ptr.as_ptr(),
                i32::from(decode),
            )
        };
        Self::from_handle(ptr, "mlx_dash_apply_rope")
    }

    /// Router index query/key for hidden states `self` `[1, L, H]`: dense
    /// `q_w`/`k_w` `[d, H]`, Gemma norm weights `[d]`. Returns f32 `[L, d]`.
    #[allow(clippy::too_many_arguments)]
    pub fn dash_index_qk(
        &self,
        q_w: &Self,
        k_w: &Self,
        q_norm_w: &Self,
        k_norm_w: &Self,
        cos: &Self,
        sin: &Self,
        eps: f32,
        decode: bool,
    ) -> Result<(Self, Self), MlxError> {
        let (mut q, mut k) = (std::ptr::null_mut(), std::ptr::null_mut());
        let code = unsafe {
            sys::mlx_dash_index_qk(
                self.ptr.as_ptr(),
                q_w.ptr.as_ptr(),
                k_w.ptr.as_ptr(),
                q_norm_w.ptr.as_ptr(),
                k_norm_w.ptr.as_ptr(),
                cos.ptr.as_ptr(),
                sin.ptr.as_ptr(),
                eps,
                i32::from(decode),
                &mut q,
                &mut k,
            )
        };
        status(code, "mlx_dash_index_qk")?;
        Ok((
            Self::from_handle(q, "mlx_dash_index_qk")?,
            Self::from_handle(k, "mlx_dash_index_qk")?,
        ))
    }

    /// Decode prologue for hidden state `self` `[1, 1, H]`: the roped
    /// attention `q`/`k` and the router index q/k of [`Self::dash_index_qk`],
    /// in one Metal launch when it applies. Returns `(q, k, idx_q, idx_k)`.
    #[allow(clippy::too_many_arguments)]
    pub fn dash_decode_prologue(
        &self,
        q: &Self,
        k: &Self,
        q_w: &Self,
        k_w: &Self,
        q_norm_w: &Self,
        k_norm_w: &Self,
        cos: &Self,
        sin: &Self,
        eps: f32,
    ) -> Result<(Self, Self, Self, Self), MlxError> {
        let mut out = [std::ptr::null_mut(); 4];
        let [oq, ok, oiq, oik] = &mut out;
        let code = unsafe {
            sys::mlx_dash_decode_prologue(
                self.ptr.as_ptr(),
                q.ptr.as_ptr(),
                k.ptr.as_ptr(),
                q_w.ptr.as_ptr(),
                k_w.ptr.as_ptr(),
                q_norm_w.ptr.as_ptr(),
                k_norm_w.ptr.as_ptr(),
                cos.ptr.as_ptr(),
                sin.ptr.as_ptr(),
                eps,
                oq,
                ok,
                oiq,
                oik,
            )
        };
        status(code, "mlx_dash_decode_prologue")?;
        let [q, k, iq, ik] = out.map(|p| Self::from_handle(p, "mlx_dash_decode_prologue"));
        Ok((q?, k?, iq?, ik?))
    }

    /// Block summaries: f32 index keys `[n, bs, d]` -> `[n, d]`.
    pub fn dash_summarize(blocks: &Self, head_cls: &Self) -> Result<Self, MlxError> {
        let ptr = unsafe { sys::mlx_dash_summarize(blocks.ptr.as_ptr(), head_cls.ptr.as_ptr()) };
        Self::from_handle(ptr, "mlx_dash_summarize")
    }

    /// DASH prefill attention of queries `self` `[1, HQ, L, D]` at positions
    /// `[offset, offset + L)` over the full cache buffers `keys`/`values`
    /// `[1, HK, S_alloc, D]`. `summaries` `[n, d]` (None before a block
    /// completes), `cand` `[L, nblk]` same-page candidates (None =
    /// unconstrained). With `want_route` the routing is returned too.
    #[allow(clippy::too_many_arguments)]
    pub fn dash_prefill(
        &self,
        keys: &Self,
        values: &Self,
        idx_q: &Self,
        summaries: Option<&Self>,
        cand: Option<&Self>,
        offset: i32,
        p: &DashParams,
        c: &DashLayerConsts,
        mode: DashMode,
        want_route: bool,
    ) -> Result<(Self, Option<DashRoute>), MlxError> {
        let mut out = std::ptr::null_mut();
        let (mut topi, mut bias, mut valid) = (
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        let (pt, pb, pv) = if want_route {
            (
                &mut topi as *mut _,
                &mut bias as *mut _,
                &mut valid as *mut _,
            )
        } else {
            (
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        let code = unsafe {
            sys::mlx_dash_prefill(
                self.ptr.as_ptr(),
                keys.ptr.as_ptr(),
                values.ptr.as_ptr(),
                idx_q.ptr.as_ptr(),
                opt_ptr(summaries),
                opt_ptr(cand),
                offset,
                p.block_size,
                p.local_blocks,
                p.topm,
                p.topk,
                p.sigma,
                c.idx_scale,
                c.floor,
                c.scale,
                mode as i32,
                &mut out,
                pt,
                pb,
                pv,
            )
        };
        status(code, "mlx_dash_prefill")?;
        let out = Self::from_handle(out, "mlx_dash_prefill")?;
        let route = if want_route {
            Some(DashRoute {
                topi: Self::from_handle(topi, "mlx_dash_prefill")?,
                bias: Self::from_handle(bias, "mlx_dash_prefill")?,
                valid: Self::from_handle(valid, "mlx_dash_prefill")?,
            })
        } else {
            None
        };
        Ok((out, route))
    }

    /// One DASH decode step of query `self` `[1, HQ, 1, D]` at position `pos`
    /// (int32 scalar): routing over `summaries` `[cap, d]` (zero-padded to
    /// the block capacity) with candidates `cand` `[1, cap]`, then attention
    /// over the selected blocks and the local window of the cache buffers.
    /// [`DashMode::Dense`] is treated as [`DashMode::Gather`].
    #[allow(clippy::too_many_arguments)]
    pub fn dash_decode(
        &self,
        keys: &Self,
        values: &Self,
        idx_q: &Self,
        summaries: &Self,
        cand: &Self,
        pos: &Self,
        p: &DashParams,
        c: &DashLayerConsts,
        mode: DashMode,
    ) -> Result<Self, MlxError> {
        let ptr = unsafe {
            sys::mlx_dash_decode(
                self.ptr.as_ptr(),
                keys.ptr.as_ptr(),
                values.ptr.as_ptr(),
                idx_q.ptr.as_ptr(),
                summaries.ptr.as_ptr(),
                cand.ptr.as_ptr(),
                pos.ptr.as_ptr(),
                p.block_size,
                p.local_blocks,
                p.topm,
                p.topk,
                p.sigma,
                c.idx_scale,
                c.floor,
                c.scale,
                mode as i32,
            )
        };
        Self::from_handle(ptr, "mlx_dash_decode")
    }

    /// Next decode-step page/RoPE/candidate state from the previous token
    /// `token` and the int32 scalars `markers`/`page`/`tok`, all in-graph.
    pub fn dash_decode_ctx(
        token: &Self,
        markers: &Self,
        page: &Self,
        tok: &Self,
        prompt: &DashPromptState<'_>,
        cap: i32,
        dtype: MlxDtype,
    ) -> Result<DashDecodeCtx, MlxError> {
        let mut outs = [std::ptr::null_mut(); 6];
        let [o0, o1, o2, o3, o4, o5] = &mut outs;
        let code = unsafe {
            sys::mlx_dash_decode_ctx(
                token.ptr.as_ptr(),
                markers.ptr.as_ptr(),
                page.ptr.as_ptr(),
                tok.ptr.as_ptr(),
                prompt.pf_cnt.ptr.as_ptr(),
                prompt.blk_min.ptr.as_ptr(),
                prompt.blk_max.ptr.as_ptr(),
                prompt.inv_tok.ptr.as_ptr(),
                prompt.inv_page.ptr.as_ptr(),
                prompt.page_token_id,
                prompt.n_pages,
                prompt.prompt_pages,
                prompt.page_mod,
                i32::from(prompt.image_only),
                cap,
                dtype.code(),
                o0,
                o1,
                o2,
                o3,
                o4,
                o5,
            )
        };
        status(code, "mlx_dash_decode_ctx")?;
        let op = "mlx_dash_decode_ctx";
        Ok(DashDecodeCtx {
            markers: Self::from_handle(outs[0], op)?,
            page: Self::from_handle(outs[1], op)?,
            tok: Self::from_handle(outs[2], op)?,
            cos: Self::from_handle(outs[3], op)?,
            sin: Self::from_handle(outs[4], op)?,
            cand: Self::from_handle(outs[5], op)?,
        })
    }
}
