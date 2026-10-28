//! Unlimited-OCR-2 DASH decoder state (spec 008).
//!
//! Port of mlx-vlm `models/unlimited_ocr/dash.py` (written for aimd and
//! contributed upstream): `DashParams.from_config`, page2d RoPE frequencies,
//! `PromptPages` (host page bookkeeping from the prompt ids), the per-layer
//! router weights and block-summary cache, and `build_context` (per-forward
//! RoPE tables, routing candidates and decode position). The attention math
//! itself is in the shim (`aimd_mlx::MlxArray::dash_*`).

use aimd_mlx::{DashLayerConsts, DashParams, DashPromptState, MlxArray, MlxDtype, MlxError};
use aimd_weights::WeightStore;
use serde::Deserialize;

use super::mlx_weights::{LoadError, dense};

/// `_DEFAULT_TOPM`: remote blocks ranked before the top-k cut.
const DEFAULT_TOPM: i32 = 16;
/// Decode routing buffers are padded to a multiple of this many blocks, so
/// the decode shapes (and compiled traces) change only every 64 blocks.
const BLOCK_CAPACITY: usize = 64;

pub(crate) fn block_capacity(nblk: usize) -> usize {
    nblk.div_ceil(BLOCK_CAPACITY) * BLOCK_CAPACITY
}

/// `sparse_attention_config` keys read at inference (defaults as in
/// `DashParams.from_config`).
#[derive(Clone, Debug, Deserialize)]
pub struct SparseAttentionConfig {
    #[serde(default)]
    pub training_stage: String,
    #[serde(default)]
    pub use_sparse_attention: bool,
    pub sparse_index_dim: Option<usize>,
    pub sparse_attention_freq: Option<Vec<u32>>,
    #[serde(default = "two")]
    pub dash_scaling_factor: f32,
    #[serde(default = "floor_default")]
    pub dash_prior_floor: f32,
    pub dash_scale_per_layer: Option<Vec<f32>>,
    pub dash_floor_per_layer: Option<Vec<f32>>,
    #[serde(default = "rope_1d")]
    pub sparse_main_rope_mode: String,
    #[serde(default = "stage0_default")]
    pub dash_stage0: String,
    #[serde(default = "block_default")]
    pub sparse_block_size: usize,
    #[serde(default = "one")]
    pub sparse_local_block: usize,
    pub dash_topk_blocks: Option<usize>,
    #[serde(default = "one_f")]
    pub dash_sigma: f32,
    #[serde(default = "yes")]
    pub dash_image_only: bool,
    #[serde(default = "image_token_default")]
    pub image_token_id: u32,
    pub page_token_id: Option<u32>,
    #[serde(default = "per_page_default")]
    pub per_page_image_tokens: usize,
    #[serde(default = "tok_theta_default")]
    pub sparse_rope_tok_theta: f32,
    #[serde(default = "page_theta_default")]
    pub sparse_rope_page_theta: f32,
    #[serde(default = "page_pairs_default")]
    pub sparse_rope_page_pairs: usize,
    #[serde(default = "truncate")]
    pub sparse_main_rope_tok_freq_mode: String,
    pub sparse_rope_page_mod: Option<i32>,
}

fn two() -> f32 {
    2.0
}
fn floor_default() -> f32 {
    1e-2
}
fn rope_1d() -> String {
    "1d".into()
}
fn stage0_default() -> String {
    "indexer".into()
}
fn block_default() -> usize {
    128
}
fn one() -> usize {
    1
}
fn one_f() -> f32 {
    1.0
}
fn yes() -> bool {
    true
}
fn image_token_default() -> u32 {
    128_815
}
fn per_page_default() -> usize {
    273
}
fn tok_theta_default() -> f32 {
    10_000.0
}
fn page_theta_default() -> f32 {
    256.0
}
fn page_pairs_default() -> usize {
    8
}
fn truncate() -> String {
    "truncate".into()
}

impl SparseAttentionConfig {
    /// `is_dash_config`.
    pub fn is_dash(&self) -> bool {
        self.training_stage == "dash" && self.use_sparse_attention
    }
}

/// Resolved DASH inference parameters (`DashParams`) plus the page2d RoPE
/// frequencies.
#[derive(Debug)]
pub(crate) struct DashSpec {
    pub(crate) params: DashParams,
    pub(crate) index_dim: usize,
    /// Per layer: `(dash_scale / sqrt(index_dim), prior floor)`.
    pub(crate) layer: Vec<(f32, f32)>,
    pub(crate) image_only: bool,
    pub(crate) image_token_id: u32,
    pub(crate) page_token_id: Option<u32>,
    pub(crate) per_page_image_tokens: usize,
    pub(crate) page_mod: i32,
    pub(crate) inv_tok: MlxArray,
    pub(crate) inv_page: MlxArray,
}

impl DashSpec {
    pub(crate) fn from_config(
        sp: &SparseAttentionConfig,
        num_layers: usize,
        head_dim: usize,
    ) -> Result<Self, String> {
        let index_dim = sp.sparse_index_dim.unwrap_or(head_dim);
        let freq = sp
            .sparse_attention_freq
            .clone()
            .unwrap_or_else(|| vec![1; num_layers]);
        let per_layer = |values: &Option<Vec<f32>>, slot: usize, default: f32| {
            values
                .as_ref()
                .and_then(|v| v.get(slot).copied())
                .unwrap_or(default)
        };
        let mut layer = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            // Per-layer arrays are indexed by the sparse-layer ordinal.
            let slot = freq.iter().take(l).filter(|&&f| f != 0).count();
            let scale = per_layer(&sp.dash_scale_per_layer, slot, sp.dash_scaling_factor);
            let floor = per_layer(&sp.dash_floor_per_layer, slot, sp.dash_prior_floor);
            layer.push((scale / (index_dim as f32).sqrt(), floor));
        }
        if sp.dash_stage0.to_lowercase() != "head_cls" {
            return Err(format!(
                "dash_stage0={:?} is not supported (only 'head_cls' checkpoints are)",
                sp.dash_stage0
            ));
        }
        if sp.sparse_main_rope_mode.to_lowercase() != "page2d" {
            return Err(format!(
                "sparse_main_rope_mode={:?} is not supported (page2d only)",
                sp.sparse_main_rope_mode
            ));
        }
        let params = DashParams {
            block_size: sp.sparse_block_size as i32,
            local_blocks: sp.sparse_local_block.max(1) as i32,
            topm: DEFAULT_TOPM,
            topk: sp.dash_topk_blocks.filter(|&k| k > 0).unwrap_or(12) as i32,
            sigma: sp.dash_sigma,
        };
        let (inv_tok, inv_page) = page2d_inv_freq(sp, head_dim);
        Ok(Self {
            params,
            index_dim,
            layer,
            image_only: sp.dash_image_only,
            image_token_id: sp.image_token_id,
            page_token_id: sp.page_token_id,
            per_page_image_tokens: sp.per_page_image_tokens,
            page_mod: sp.sparse_rope_page_mod.unwrap_or(0),
            inv_tok: MlxArray::from_f32(&[inv_tok.len()], &inv_tok).map_err(|e| e.to_string())?,
            inv_page: MlxArray::from_f32(&[inv_page.len()], &inv_page)
                .map_err(|e| e.to_string())?,
        })
    }

    pub(crate) fn consts(&self, layer: usize, head_dim: usize) -> DashLayerConsts {
        let (idx_scale, floor) = self.layer[layer];
        DashLayerConsts {
            idx_scale,
            floor,
            scale: (head_dim as f32).powf(-0.5),
        }
    }
}

/// `page2d_inv_freq` (f32 like the numpy reference).
fn page2d_inv_freq(sp: &SparseAttentionConfig, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let pairs = sp.sparse_rope_page_pairs;
    let n_tok = half - pairs;
    let resample = sp.sparse_main_rope_tok_freq_mode.to_lowercase() != "truncate";
    let inv_tok = (0..n_tok)
        .map(|i| {
            let expo = if resample {
                i as f32 / n_tok as f32
            } else {
                i as f32 * (2.0f32 / dim as f32)
            };
            1.0f32 / sp.sparse_rope_tok_theta.powf(expo)
        })
        .collect();
    let inv_page = (0..pairs)
        .map(|i| 1.0f32 / sp.sparse_rope_page_theta.powf(i as f32 / pairs as f32))
        .collect();
    (inv_tok, inv_page)
}

/// Dense per-layer router weights (`self_attn.dash_router`).
pub(crate) struct DashRouter {
    q_w: MlxArray,
    k_w: MlxArray,
    q_norm: MlxArray,
    k_norm: MlxArray,
    head_cls: MlxArray,
}

impl DashRouter {
    pub(crate) fn load(
        store: &WeightStore,
        prefix: &str,
        hidden: usize,
        index_dim: usize,
    ) -> Result<Self, LoadError> {
        let w =
            |name: &str, shape: &[usize]| dense(store, &format!("{prefix}.{name}"), Some(shape));
        Ok(Self {
            q_w: w("q_proj.weight", &[index_dim, hidden])?,
            k_w: w("k_proj.weight", &[index_dim, hidden])?,
            q_norm: w("q_norm.weight", &[index_dim])?,
            k_norm: w("k_norm.weight", &[index_dim])?,
            head_cls: w("head_cls", &[index_dim])?,
        })
    }

    /// Index query/key `[L, d]` f32 for hidden states `x` `[1, L, H]`.
    pub(crate) fn index_qk(
        &self,
        x: &MlxArray,
        step: &DashStep,
        eps: f32,
    ) -> Result<(MlxArray, MlxArray), MlxError> {
        x.dash_index_qk(
            &self.q_w,
            &self.k_w,
            &self.q_norm,
            &self.k_norm,
            &step.cos,
            &step.sin,
            eps,
            step.decode(),
        )
    }

    /// Decode step: roped attention `q`/`k` plus [`Self::index_qk`] in one
    /// launch. Returns `(q, k, idx_q, idx_k)`.
    pub(crate) fn decode_prologue(
        &self,
        x: &MlxArray,
        q: &MlxArray,
        k: &MlxArray,
        step: &DashStep,
        eps: f32,
    ) -> Result<(MlxArray, MlxArray, MlxArray, MlxArray), MlxError> {
        x.dash_decode_prologue(
            q,
            k,
            &self.q_w,
            &self.k_w,
            &self.q_norm,
            &self.k_norm,
            &step.cos,
            &step.sin,
            eps,
        )
    }

    pub(crate) fn head_cls(&self) -> &MlxArray {
        &self.head_cls
    }
}

/// Per-layer router state: pending index keys of the open block and one
/// summary key per completed block (`DashKVCache` minus the KV).
#[derive(Default)]
pub(crate) struct DashIndexCache {
    pending: Option<MlxArray>,
    summaries: Option<MlxArray>,
    num: usize,
}

impl DashIndexCache {
    /// `add_index_keys`: append `[L, d]` f32 keys, summarizing every block
    /// they complete.
    pub(crate) fn add_index_keys(
        &mut self,
        idx_k: &MlxArray,
        head_cls: &MlxArray,
        block_size: usize,
    ) -> Result<(), MlxError> {
        let k = match self.pending.take() {
            Some(p) => MlxArray::concatenate(&[&p, idx_k], 0)?,
            None => idx_k.try_clone()?,
        };
        let s = k.shape()?;
        let (rows, d) = (s[0], s[1]);
        let n_full = rows / block_size;
        if n_full > 0 {
            let blocks = k
                .slice(&[0, 0], &[n_full * block_size, d], &[1, 1])?
                .reshape(&[n_full, block_size, d])?;
            self.append(&MlxArray::dash_summarize(&blocks, head_cls)?)?;
        }
        if rows > n_full * block_size {
            self.pending = Some(k.slice(&[n_full * block_size, 0], &[rows, d], &[1, 1])?);
        }
        Ok(())
    }

    fn append(&mut self, s: &MlxArray) -> Result<(), MlxError> {
        let sh = s.shape()?;
        let (n, d) = (sh[0], sh[1]);
        let need = self.num + n;
        let cap = match &self.summaries {
            Some(buf) => buf.shape()?[0],
            None => 0,
        };
        if need > cap {
            self.grow(block_capacity(need), d)?;
        }
        let buf = self.summaries.take().expect("grown");
        self.summaries = Some(buf.slice_update(s, &[self.num, 0], &[need, d], &[1, 1])?);
        self.num = need;
        Ok(())
    }

    fn grow(&mut self, capacity: usize, d: usize) -> Result<(), MlxError> {
        let mut grown = MlxArray::zeros(&[capacity, d], MlxDtype::F32)?;
        if let (Some(old), true) = (&self.summaries, self.num > 0) {
            let keep = old.slice(&[0, 0], &[self.num, d], &[1, 1])?;
            grown = grown.slice_update(&keep, &[0, 0], &[self.num, d], &[1, 1])?;
        }
        self.summaries = Some(grown);
        Ok(())
    }

    /// Summaries zero-padded to `capacity` rows (decode routing).
    pub(crate) fn summary_buffer(
        &mut self,
        capacity: usize,
        d: usize,
    ) -> Result<MlxArray, MlxError> {
        let have = match &self.summaries {
            Some(buf) => buf.shape()?[0],
            None => 0,
        };
        if have < capacity {
            self.grow(capacity, d)?;
        }
        self.summaries
            .as_ref()
            .expect("grown")
            .slice(&[0, 0], &[capacity, d], &[1, 1])
    }

    /// Summaries of the completed blocks (None before the first).
    pub(crate) fn block_summaries(&self) -> Result<Option<MlxArray>, MlxError> {
        match (&self.summaries, self.num) {
            (Some(buf), n) if n > 0 => {
                let d = buf.shape()?[1];
                Ok(Some(buf.slice(&[0, 0], &[n, d], &[1, 1])?))
            }
            _ => Ok(None),
        }
    }

    /// Arrays to evaluate with the KV state after a prefill chunk.
    pub(crate) fn state(&self) -> Vec<&MlxArray> {
        self.summaries.iter().chain(self.pending.iter()).collect()
    }
}

/// `PromptPages`: page bookkeeping derived from the prompt token ids
/// (prefill query/image pages, page2d coordinates, decode candidate state).
#[derive(Debug)]
pub(crate) struct PromptPages {
    pub(crate) length: usize,
    page_aware: bool,
    blk_min: Vec<i64>,
    blk_max: Vec<i64>,
    query_pages: Vec<i64>,
    dec_blk_min: Vec<i32>,
    dec_blk_max: Vec<i32>,
    dec_prompt_pages: i32,
    marker_count: i32,
    /// page2d `(tok, page)` per prompt position.
    coords: Option<(Vec<i32>, Vec<i32>)>,
    n_pages: i32,
    pf_cnt: Vec<i32>,
}

/// `(min, max)` page of each `bs`-row block of `pages` (-1 = no image),
/// padded with -1.
fn block_page_range(pages: &[i64], bs: usize) -> (Vec<i64>, Vec<i64>) {
    let nblk = pages.len().div_ceil(bs);
    let mut lo = vec![-1i64; nblk];
    let mut hi = vec![-1i64; nblk];
    for b in 0..nblk {
        let blk = &pages[b * bs..((b + 1) * bs).min(pages.len())];
        let max = blk.iter().copied().max().unwrap_or(-1).max(-1);
        hi[b] = max;
        if max >= 0 {
            lo[b] = blk.iter().copied().filter(|&p| p >= 0).min().unwrap_or(-1);
        }
    }
    (lo, hi)
}

impl PromptPages {
    pub(crate) fn new(ids: &[u32], spec: &DashSpec) -> Self {
        let s = ids.len();
        let bs = spec.params.block_size as usize;
        let img: Vec<bool> = ids.iter().map(|&t| t == spec.image_token_id).collect();
        let mk: Vec<bool> = ids.iter().map(|&t| Some(t) == spec.page_token_id).collect();
        let mut cum = Vec::with_capacity(s);
        let mut running = Vec::with_capacity(s);
        let (mut c, mut r) = (0i64, -1i64);
        for i in 0..s {
            c += i64::from(mk[i]);
            r += i64::from(img[i]);
            cum.push(c);
            running.push(r);
        }
        let last_img = img.iter().rposition(|&b| b);
        let prompt_pages = last_img.map_or(0, |i| cum[i]);
        let marker_mode = prompt_pages > 0;
        let per_page = spec.per_page_image_tokens.max(1) as i64;
        let in_prefill = |i: usize| last_img.is_some_and(|l| i <= l);

        // Prefill candidate pages (query -> same-page image blocks).
        let ipi: Vec<i64> = (0..s)
            .map(|i| match (img[i], marker_mode) {
                (false, _) => -1,
                (true, true) => cum[i] - 1,
                (true, false) => running[i].div_euclid(per_page),
            })
            .collect();
        let query_pages: Vec<i64> = (0..s)
            .map(|i| {
                let raw = cum[i] - 1;
                let q = if in_prefill(i) {
                    raw
                } else {
                    raw - if marker_mode { prompt_pages } else { 0 }
                };
                if q < 0 { -1 } else { q }
            })
            .collect();
        let page_aware = img.iter().any(|&b| b) && query_pages.iter().any(|&q| q >= 0);
        let (blk_min, blk_max) = block_page_range(&ipi, bs);

        // Decode candidate state (pages of the response restart at 0).
        let dec_page: Vec<i64> = (0..s)
            .map(|i| {
                if !img[i] {
                    -1
                } else if cum[i] > 0 {
                    cum[i] - 1
                } else {
                    running[i].div_euclid(per_page)
                }
            })
            .collect();
        let (dec_min, dec_max) = block_page_range(&dec_page, bs);

        // page2d coordinates.
        let mut coords = None;
        let mut n_pages = 0;
        let mut pf_cnt = Vec::new();
        if img.iter().any(|&b| b) && mk.iter().any(|&b| b) && marker_mode {
            let page: Vec<i64> = (0..s)
                .map(|i| {
                    let p = if in_prefill(i) {
                        cum[i]
                    } else {
                        cum[i] - prompt_pages
                    };
                    p.max(0)
                })
                .collect();
            let n_pg = page.iter().copied().max().unwrap_or(0) as usize + 1;
            let mut counts = vec![0i32; n_pg];
            let mut tok = Vec::with_capacity(s);
            for &p in &page {
                tok.push(counts[p as usize]);
                counts[p as usize] += 1;
            }
            pf_cnt = vec![0i32; n_pg];
            for (i, &p) in page.iter().enumerate() {
                if in_prefill(i) {
                    pf_cnt[p as usize] += 1;
                }
            }
            coords = Some((tok, page.iter().map(|&p| p as i32).collect()));
            n_pages = n_pg as i32;
        }
        Self {
            length: s,
            page_aware,
            blk_min,
            blk_max,
            query_pages,
            dec_blk_min: dec_min.iter().map(|&v| v as i32).collect(),
            dec_blk_max: dec_max.iter().map(|&v| v as i32).collect(),
            dec_prompt_pages: if last_img.is_some() {
                prompt_pages as i32
            } else {
                0
            },
            marker_count: cum.last().copied().unwrap_or(0) as i32,
            coords,
            n_pages,
            pf_cnt,
        }
    }

    /// `[L, nblk]` same-page image blocks of the queries at `positions`
    /// (None = unconstrained).
    fn prefill_candidates(
        &self,
        positions: std::ops::Range<usize>,
        nblk: usize,
    ) -> Option<Vec<bool>> {
        if !self.page_aware {
            return None;
        }
        let n0 = nblk.min(self.blk_max.len());
        let mut out = Vec::with_capacity(positions.len() * nblk);
        for pos in positions {
            let qp = self.query_pages[pos];
            let row: Vec<bool> = (0..nblk)
                .map(|b| {
                    b < n0
                        && qp >= 0
                        && self.blk_min[b] >= 0
                        && self.blk_min[b] <= qp
                        && self.blk_max[b] >= qp
                })
                .collect();
            let any = row.iter().any(|&v| v);
            out.extend(row.into_iter().map(|v| v || !any));
        }
        Some(out)
    }
}

/// Per-forward inputs shared by every layer (`DashContext`).
pub(crate) struct DashStep {
    pub(crate) cos: MlxArray,
    pub(crate) sin: MlxArray,
    pub(crate) offset: usize,
    /// Prefill `[L, nblk]` / decode `[1, capacity]` candidates.
    pub(crate) cand: Option<MlxArray>,
    /// Decode query position (int32 scalar); None for prefill.
    pub(crate) pos: Option<MlxArray>,
}

impl DashStep {
    pub(crate) fn decode(&self) -> bool {
        self.pos.is_some()
    }
}

/// Prompt metadata plus the in-graph decode state (`DashState`) and every
/// layer's router cache.
pub(crate) struct DashRun {
    prompt: PromptPages,
    markers: MlxArray,
    page: MlxArray,
    tok: MlxArray,
    pf_cnt: MlxArray,
    dec_blk_min: MlxArray,
    dec_blk_max: MlxArray,
    pub(crate) layers: Vec<DashIndexCache>,
}

impl DashRun {
    /// `DashState.start`. Prompts without `<PAGE>`-delimited images (no
    /// page2d coordinates) are rejected.
    pub(crate) fn start(
        prompt_ids: &[u32],
        spec: &DashSpec,
        layers: usize,
    ) -> Result<Self, String> {
        let prompt = PromptPages::new(prompt_ids, spec);
        let Some((tok, page)) = &prompt.coords else {
            return Err("Unlimited-OCR-2 prompts need <PAGE>-delimited images".into());
        };
        let scalar = |v: i32| MlxArray::from_i32(&[], &[v]).map_err(|e| e.to_string());
        let vec = |v: &[i32]| MlxArray::from_i32(&[v.len()], v).map_err(|e| e.to_string());
        Ok(Self {
            markers: scalar(prompt.marker_count)?,
            page: scalar(*page.last().expect("non-empty"))?,
            tok: scalar(*tok.last().expect("non-empty"))?,
            pf_cnt: vec(&prompt.pf_cnt)?,
            dec_blk_min: vec(&prompt.dec_blk_min)?,
            dec_blk_max: vec(&prompt.dec_blk_max)?,
            layers: (0..layers).map(|_| DashIndexCache::default()).collect(),
            prompt,
        })
    }

    /// `build_context`: prefill for positions inside the prompt, otherwise
    /// one decode step whose input token is `token` (in-graph; never read
    /// back).
    pub(crate) fn step(
        &mut self,
        spec: &DashSpec,
        offset: usize,
        len: usize,
        token: Option<&MlxArray>,
        dtype: MlxDtype,
    ) -> Result<DashStep, String> {
        let bs = spec.params.block_size as usize;
        let total = offset + len;
        let nblk = total.div_ceil(bs);
        let err = |e: MlxError| e.to_string();
        if total <= self.prompt.length {
            let (tok, page) = self.prompt.coords.as_ref().expect("checked at start");
            let tok = MlxArray::from_i32(&[len], &tok[offset..total]).map_err(err)?;
            let page = MlxArray::from_i32(&[len], &page[offset..total]).map_err(err)?;
            let (cos, sin) = MlxArray::dash_rope(
                &tok,
                &page,
                &spec.inv_tok,
                &spec.inv_page,
                spec.page_mod,
                dtype,
            )
            .map_err(err)?;
            let cand = match self.prompt.prefill_candidates(offset..total, nblk) {
                Some(c) => Some(MlxArray::from_bool(&[len, nblk], &c).map_err(err)?),
                None => None,
            };
            return Ok(DashStep {
                cos,
                sin,
                offset,
                cand,
                pos: None,
            });
        }
        let token = match token {
            Some(t) if len == 1 && offset >= self.prompt.length => t,
            _ => return Err("Unlimited-OCR-2 decodes one token at a time after the prompt".into()),
        };
        let cap = block_capacity(nblk);
        let prompt = DashPromptState {
            pf_cnt: &self.pf_cnt,
            blk_min: &self.dec_blk_min,
            blk_max: &self.dec_blk_max,
            inv_tok: &spec.inv_tok,
            inv_page: &spec.inv_page,
            page_token_id: spec.page_token_id.map_or(-1, |t| t as i32),
            n_pages: self.prompt.n_pages,
            prompt_pages: self.prompt.dec_prompt_pages,
            page_mod: spec.page_mod,
            image_only: spec.image_only,
        };
        let ctx = MlxArray::dash_decode_ctx(
            token,
            &self.markers,
            &self.page,
            &self.tok,
            &prompt,
            cap as i32,
            dtype,
        )
        .map_err(err)?;
        self.markers = ctx.markers;
        self.page = ctx.page;
        self.tok = ctx.tok;
        Ok(DashStep {
            cos: ctx.cos,
            sin: ctx.sin,
            offset,
            cand: Some(ctx.cand),
            pos: Some(MlxArray::from_i32(&[], &[offset as i32]).map_err(err)?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> DashSpec {
        let sp: SparseAttentionConfig = serde_json::from_value(serde_json::json!({
            "training_stage": "dash",
            "use_sparse_attention": true,
            "dash_stage0": "head_cls",
            "sparse_main_rope_mode": "page2d",
            "sparse_main_rope_tok_freq_mode": "resample",
            "sparse_block_size": 4,
            "sparse_local_block": 1,
            "sparse_index_dim": 8,
            "page_token_id": 9,
            "image_token_id": 7,
            "per_page_image_tokens": 3,
            "sparse_rope_page_pairs": 1,
            "sparse_rope_page_mod": 2,
            "dash_scale_per_layer": [128.0, 64.0],
            "dash_floor_per_layer": [1e-4, 1e-3],
        }))
        .unwrap();
        assert!(sp.is_dash());
        DashSpec::from_config(&sp, 2, 8).unwrap()
    }

    /// Two `<PAGE><image>x3` pages + text; expected values from mlx-vlm
    /// `PromptPages` on the same ids.
    #[test]
    fn prompt_pages_match_reference() {
        let s = spec();
        assert_eq!(s.layer[1], (64.0 / 8f32.sqrt(), 1e-3));
        assert_eq!(s.inv_tok.to_f32_vec().unwrap().len(), 3);
        let ids = [9, 7, 7, 7, 9, 7, 7, 7, 1, 2, 3];
        let p = PromptPages::new(&ids, &s);
        assert!(p.page_aware);
        assert_eq!(p.query_pages, vec![0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1]);
        assert_eq!(
            (p.dec_blk_min.clone(), p.dec_blk_max.clone()),
            (vec![0, 1, -1], vec![0, 1, -1])
        );
        assert_eq!(
            (p.blk_min.clone(), p.blk_max.clone()),
            (vec![0, 1, -1], vec![0, 1, -1])
        );
        let (tok, page) = p.coords.clone().unwrap();
        assert_eq!(page, vec![1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0]);
        assert_eq!(tok, vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]);
        assert_eq!(p.pf_cnt, vec![0, 4, 4]);
        assert_eq!((p.dec_prompt_pages, p.marker_count, p.n_pages), (2, 2, 3));
        let c = p.prefill_candidates(4..6, 2).unwrap();
        assert_eq!(c, vec![false, true, false, true]);
        assert_eq!(block_capacity(65), 128);
    }
}
