//! Unlimited-OCR DeepSeek-V2 MoE decoder on MLX, plus the Python-exact
//! KV caches (`KVCache` step-256 growth and `RingSlidingKVCache`).
//!
//! # Op-for-op port of the mlx_vlm graph
//!
//! Source of truth: mlx_vlm 0.7.0 `models/deepseekocr/language.py`
//! (`LlamaAttention`, `MoEGate`, `DeepseekV2MoE`, `DeepseekV2DecoderLayer`),
//! `models/switch_layers.py` (`SwitchGLU` + `_gather_sort`), `models/mlp.py`
//! (`DeepseekMLP`), `models/cache.py` (`KVCache`), and
//! `models/unlimited_ocr/language.py` (`RingSlidingKVCache`,
//! `DeepseekV2RingSlidingModel`):
//!
//! - 12 pre-RMSNorm layers (eps 1e-6). Attention is plain MHA (10 heads x
//!   128, 4-bit q/k/v/o), NeoX RoPE (base 1e4) at `offset = cache.offset`,
//!   SDPA scale `128^-0.5` with the `"causal"` mask for `N > 1` and no mask
//!   for single-token steps.
//! - Layer 0 is a dense SwiGLU MLP (intermediate 6848); layers 1..11 are
//!   MoE: bf16 gate logits `x @ W_g.T`, `softmax(precise=True)`, greedy
//!   top-6 via `argpartition` + `take_along_axis` (routed scaling 1.0),
//!   `SwitchGLU` over 4-bit experts through `gather_qmm` (sorted path when
//!   `indices.size >= 64`), weighted sum over the selected experts, plus
//!   the 2-expert shared MLP (intermediate 1792). Single-token steps on the
//!   GPU run the router (`MlxArray::moe_router`) and the routed experts
//!   (`MlxArray::moe_swiglu_4bit`) as custom kernels, bit-identical to that
//!   graph.
//! - Final RMSNorm and the untied 4-bit `lm_head`.
//! - Embeddings are `nn.QuantizedEmbedding` (row gather + dequantize).
//!
//! Unlimited-OCR-2 checkpoints (`sparse_attention_config.training_stage ==
//! "dash"`) keep these weights and swap the attention for DASH (mlx-vlm
//! `models/unlimited_ocr/dash.py`): page2d RoPE (`rotate_half`) on q/k, a
//! per-layer dense `dash_router`, a full KV cache, and block-sparse
//! attention (12 routed 32-token blocks + a 128-token local window) through
//! `MlxArray::dash_prefill` / `MlxArray::dash_decode`. Never dense.
//!
//! The caches live in [`crate::models::kv_cache`].

use std::path::Path;

use aimd_mlx::{DashLayerConsts, DashMode, MlxArray, MlxError, SdpaMask};
use aimd_weights::{WeightStore, WeightsError, load_checkpoint};

use crate::models::kv_cache::KvCache;

use super::config::{LanguageConfig, UnlimitedOcrConfig, parse_config};
use super::dash::{DashIndexCache, DashRouter, DashRun, DashSpec, DashStep};
use super::mlx_weights::{LoadError, QWeights, dense};

/// Errors surfaced by the decoder.
#[derive(Debug, thiserror::Error)]
pub enum DecoderError {
    #[error(transparent)]
    Mlx(#[from] MlxError),
    #[error(transparent)]
    Weights(#[from] WeightsError),
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error("io: {0}")]
    Io(String),
    #[error("config: {0}")]
    Config(String),
    #[error("layout: {0}")]
    Layout(String),
    #[error("dash: {0}")]
    Dash(String),
}

struct Attention {
    q: QWeights,
    k: QWeights,
    v: QWeights,
    o: QWeights,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_base: f64,
    /// Single-token steps use [`MlxArray::sdpa_decode`] (GPU device only).
    sdpa_decode: bool,
    /// Unlimited-OCR-2 DASH router and per-layer constants.
    dash: Option<(DashRouter, DashLayerConsts)>,
}

/// One layer's view of a DASH forward.
struct DashIo<'a> {
    spec: &'a DashSpec,
    step: &'a DashStep,
    index: &'a mut DashIndexCache,
    eps: f32,
}

impl Attention {
    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KvCache,
        dash: Option<DashIo<'_>>,
    ) -> Result<MlxArray, MlxError> {
        if let (Some(io), Some(layer)) = (dash, &self.dash) {
            return self.forward_dash(x, cache, io, layer);
        }
        let s = x.shape()?;
        let (b, l) = (s[0], s[1]);
        let hd = self.head_dim;
        let q = self
            .q
            .linear(x)?
            .reshape(&[b, l, self.heads, hd])?
            .transpose(&[0, 2, 1, 3])?;
        let k = self
            .k
            .linear(x)?
            .reshape(&[b, l, self.kv_heads, hd])?
            .transpose(&[0, 2, 1, 3])?;
        let v = self
            .v
            .linear(x)?
            .reshape(&[b, l, self.kv_heads, hd])?
            .transpose(&[0, 2, 1, 3])?;
        let offset = cache.offset() as i32;
        let q = q.rope(hd, self.rope_base, 1.0, offset)?;
        let k = k.rope(hd, self.rope_base, 1.0, offset)?;
        let (keys, values) = cache.update_and_fetch(&k, &v)?;
        let mask = if l > 1 {
            SdpaMask::Causal
        } else {
            SdpaMask::None
        };
        let scale = (hd as f32).powf(-0.5);
        let o = if self.sdpa_decode && l == 1 {
            q.sdpa_decode(&keys, &values, scale)?
        } else {
            q.scaled_dot_product_attention(&keys, &values, scale, mask)?
        };
        let o = o
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[b, l, self.heads * hd])?;
        self.o.linear(&o)
    }

    /// `DashAttention.__call__` (batch 1).
    fn forward_dash(
        &self,
        x: &MlxArray,
        cache: &mut KvCache,
        io: DashIo<'_>,
        (router, consts): &(DashRouter, DashLayerConsts),
    ) -> Result<MlxArray, MlxError> {
        let s = x.shape()?;
        let (b, l) = (s[0], s[1]);
        let hd = self.head_dim;
        let step = io.step;
        let decode = step.decode();
        let proj = |w: &QWeights, heads: usize| -> Result<MlxArray, MlxError> {
            w.linear(x)?
                .reshape(&[b, l, heads, hd])?
                .transpose(&[0, 2, 1, 3])
        };
        let q = proj(&self.q, self.heads)?;
        let k = proj(&self.k, self.kv_heads)?;
        let v = proj(&self.v, self.kv_heads)?;
        let (q, k, idx_q, idx_k) = if decode {
            router.decode_prologue(x, &q, &k, step, io.eps)?
        } else {
            let q = q.dash_apply_rope(&step.cos, &step.sin, false)?;
            let k = k.dash_apply_rope(&step.cos, &step.sin, false)?;
            let (idx_q, idx_k) = router.index_qk(x, step, io.eps)?;
            (q, k, idx_q, idx_k)
        };
        cache.update_and_fetch(&k, &v)?;
        let (keys, values) = cache.buffers().expect("updated");

        let p = &io.spec.params;
        io.index
            .add_index_keys(&idx_k, router.head_cls(), p.block_size as usize)?;
        let cand = step.cand.as_ref();
        let o = if let Some(pos) = &step.pos {
            let cand = cand.expect("decode candidates");
            let cap = cand.shape()?[1];
            let sums = io.index.summary_buffer(cap, io.spec.index_dim)?;
            q.dash_decode(
                keys,
                values,
                &idx_q,
                &sums,
                cand,
                pos,
                p,
                consts,
                DashMode::Auto,
            )?
        } else {
            let sums = io.index.block_summaries()?;
            let (o, route) = q.dash_prefill(
                keys,
                values,
                &idx_q,
                sums.as_ref(),
                cand,
                step.offset as i32,
                p,
                consts,
                DashMode::Auto,
                route_probe::active(),
            )?;
            route_probe::record(route);
            o
        };
        let o = o
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[b, l, self.heads * hd])?;
        self.o.linear(&o)
    }
}

/// Test hook: the prefill routing of each layer, in call order.
#[cfg(test)]
pub(crate) mod route_probe {
    use std::cell::RefCell;

    use aimd_mlx::DashRoute;

    thread_local! {
        static ROUTES: RefCell<Option<Vec<DashRoute>>> = const { RefCell::new(None) };
    }

    pub(crate) fn start() {
        ROUTES.with(|r| *r.borrow_mut() = Some(Vec::new()));
    }

    pub(crate) fn take() -> Vec<DashRoute> {
        ROUTES.with(|r| r.borrow_mut().take()).unwrap_or_default()
    }

    pub(super) fn active() -> bool {
        ROUTES.with(|r| r.borrow().is_some())
    }

    pub(super) fn record(route: Option<DashRoute>) {
        if let Some(route) = route {
            ROUTES.with(|r| r.borrow_mut().as_mut().map(|v| v.push(route)));
        }
    }
}

#[cfg(not(test))]
mod route_probe {
    pub(super) fn active() -> bool {
        false
    }

    pub(super) fn record(_: Option<aimd_mlx::DashRoute>) {}
}

/// `DeepseekMLP`: `down(swiglu(gate(x), up(x)))`.
struct DenseMlp {
    gate: QWeights,
    up: QWeights,
    down: QWeights,
    /// Single-token GPU steps run the MLP as one expert of
    /// [`MlxArray::moe_swiglu_4bit`]: `[1, out, in/..]` views of the
    /// gate/up/down weights, `inds = [0]` and `scores = [1]`.
    one: Option<OneExpert>,
}

struct OneExpert {
    w: [MlxArray; 9],
    inds: MlxArray,
    scores: MlxArray,
}

impl DenseMlp {
    fn load(
        store: &WeightStore,
        p: &str,
        gs: u32,
        bits: u32,
        fused: bool,
    ) -> Result<Self, LoadError> {
        let gate = QWeights::load(store, &format!("{p}.gate_proj"), gs, bits)?;
        let up = QWeights::load(store, &format!("{p}.up_proj"), gs, bits)?;
        let down = QWeights::load(store, &format!("{p}.down_proj"), gs, bits)?;
        let one = if fused {
            let view = |a: &MlxArray| -> Result<MlxArray, MlxError> {
                let s = a.shape()?;
                a.reshape(&[1, s[0], s[1]])
            };
            let [g, u, d] = [&gate, &up, &down].map(QWeights::packed);
            Some(OneExpert {
                w: [
                    view(g[0])?,
                    view(g[1])?,
                    view(g[2])?,
                    view(u[0])?,
                    view(u[1])?,
                    view(u[2])?,
                    view(d[0])?,
                    view(d[1])?,
                    view(d[2])?,
                ],
                inds: MlxArray::from_u32(&[1], &[0])?,
                scores: MlxArray::scalar_f32(1.0)?
                    .reshape(&[1])?
                    .astype(gate.scales.dtype()?)?,
            })
        } else {
            None
        };
        Ok(Self {
            gate,
            up,
            down,
            one,
        })
    }

    /// `down(swiglu(gate(x), up(x)))`, then each of `add` in order.
    fn forward(&self, x: &MlxArray, add: &[&MlxArray]) -> Result<MlxArray, MlxError> {
        if let Some(one) = &self.one {
            let s = x.shape()?;
            if s[0] * s[1] == 1 {
                let w = &one.w;
                return x
                    .reshape(&[s[2]])?
                    .moe_swiglu_4bit(
                        &one.inds,
                        &one.scores,
                        [&w[0], &w[1], &w[2]],
                        [&w[3], &w[4], &w[5]],
                        [&w[6], &w[7], &w[8]],
                        self.gate.group_size,
                        add,
                    )?
                    .reshape(&s);
            }
        }
        let g = self.gate.linear(x)?;
        let u = self.up.linear(x)?;
        let mut y = self.down.linear(&g.swiglu(&u)?)?;
        for a in add {
            y = y.add(a)?;
        }
        Ok(y)
    }
}

struct Moe {
    /// `gate.weight` (bf16 `[n_experts, hidden]`) and its `.T` view.
    gate_w: MlxArray,
    gate_wt: MlxArray,
    n_experts: usize,
    top_k: usize,
    sw_gate: QWeights,
    sw_up: QWeights,
    sw_down: QWeights,
    shared: DenseMlp,
    /// Single-token steps use the fused router and expert kernels
    /// ([`MlxArray::moe_router`], [`MlxArray::moe_swiglu_4bit`]),
    /// bit-identical to `route` + `switch_glu`: 4-bit experts on the GPU
    /// device only.
    fused: bool,
}

impl Moe {
    /// `MoEGate` (softmax, greedy): `(inds [B, L, k] u32, scores [B, L, k])`.
    fn route(&self, x: &MlxArray) -> Result<(MlxArray, MlxArray), MlxError> {
        let s = x.shape()?;
        let (b, l) = (s[0], s[1]);
        let gates = x.matmul(&self.gate_wt, None)?;
        let scores = gates.softmax_precise(-1, true)?;
        let kth = (self.n_experts - self.top_k) as i32;
        let inds = scores.argpartition(kth, -1)?.slice(
            &[0, 0, self.n_experts - self.top_k],
            &[b, l, self.n_experts],
            &[1, 1, 1],
        )?;
        let sel = scores.take_along_axis(&inds, -1)?;
        Ok((inds, sel))
    }

    /// `SwitchGLU.__call__(x, indices)` -> `[B, L, k, D]`.
    fn switch_glu(&self, x: &MlxArray, inds: &MlxArray) -> Result<MlxArray, MlxError> {
        let s = x.shape()?;
        let (b, l, d) = (s[0], s[1], s[2]);
        let k = self.top_k;
        let n = b * l * k;
        let x5 = x.reshape(&[b, l, 1, 1, d])?;
        let act = |xin: &MlxArray, idx: &MlxArray, sorted: bool| -> Result<MlxArray, MlxError> {
            let up = self.sw_up.gather(xin, idx, sorted)?;
            let gate = self.sw_gate.gather(xin, idx, sorted)?;
            self.sw_down.gather(&gate.swiglu(&up)?, idx, sorted)
        };
        let y = if n >= 64 {
            // _gather_sort / _scatter_unsort.
            let flat = inds.reshape(&[n])?;
            let order = flat.argsort(0)?;
            let inv = order.argsort(0)?;
            let rows = order.floor_divide(&MlxArray::scalar_u32(k as u32)?)?;
            let xs = x5.reshape(&[b * l, 1, d])?.take(&rows, 0)?;
            let idx = flat.take(&order, 0)?;
            act(&xs, &idx, true)?
                .take(&inv, 0)?
                .reshape(&[b, l, k, 1, d])?
        } else {
            act(&x5, inds, false)?
        };
        y.reshape(&[b, l, k, d])
    }

    /// MoE output plus the shared experts, then `residual`.
    fn forward(&self, x: &MlxArray, residual: &MlxArray) -> Result<MlxArray, MlxError> {
        let s = x.shape()?;
        let (b, l, d) = (s[0], s[1], s[2]);
        let shared = self.shared.forward(x, &[])?;
        if self.fused && b * l == 1 {
            let x1 = x.reshape(&[d])?;
            let (inds, scores) = x1.moe_router(&self.gate_w, self.top_k as i32)?;
            return x1
                .moe_swiglu_4bit(
                    &inds,
                    &scores,
                    self.sw_gate.packed(),
                    self.sw_up.packed(),
                    self.sw_down.packed(),
                    self.sw_gate.group_size,
                    &[&shared, residual],
                )?
                .reshape(&[b, l, d]);
        }
        let (inds, scores) = self.route(x)?;
        let y = self.switch_glu(x, &inds)?;
        let y = y
            .multiply(&scores.reshape(&[b, l, self.top_k, 1])?)?
            .sum_axis(-2, false)?;
        y.add(&shared)?.add(residual)
    }
}

enum Mlp {
    Dense(Box<DenseMlp>),
    Moe(Box<Moe>),
}

struct Layer {
    input_ln: MlxArray,
    post_ln: MlxArray,
    attn: Attention,
    mlp: Mlp,
}

/// The Unlimited-OCR language model on MLX.
pub struct OcrDecoder {
    cfg: LanguageConfig,
    embed: QWeights,
    layers: Vec<Layer>,
    norm: MlxArray,
    lm_head: QWeights,
    eps: f32,
    dash: Option<DashSpec>,
}

/// Generation cache: per-layer KV plus, for Unlimited-OCR-2, the DASH
/// prompt/decode state and router caches.
pub(crate) struct OcrCache {
    pub(crate) kv: Vec<KvCache>,
    pub(crate) dash: Option<DashRun>,
}

impl OcrCache {
    /// Arrays to evaluate after a prefill chunk (`cache.state` + router
    /// summaries).
    pub(crate) fn state(&self) -> Result<Vec<MlxArray>, MlxError> {
        let mut out = Vec::with_capacity(self.kv.len() * 3);
        for c in &self.kv {
            out.extend(c.state()?);
        }
        if let Some(run) = &self.dash {
            for l in &run.layers {
                for a in l.state() {
                    out.push(a.try_clone()?);
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn offset(&self) -> usize {
        self.kv.first().map_or(0, KvCache::offset)
    }
}

impl std::fmt::Debug for OcrDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OcrDecoder")
            .field("layers", &self.layers.len())
            .field("hidden", &self.cfg.hidden_size)
            .finish_non_exhaustive()
    }
}

impl OcrDecoder {
    pub fn load(checkpoint_dir: &Path) -> Result<Self, DecoderError> {
        let store = load_checkpoint(checkpoint_dir)?;
        let raw = std::fs::read(checkpoint_dir.join("config.json"))
            .map_err(|e| DecoderError::Io(format!("reading config.json: {e}")))?;
        let cfg = parse_config(&raw).map_err(|e| DecoderError::Config(e.to_string()))?;
        Self::from_store(&store, &cfg)
    }

    pub fn from_store(
        store: &WeightStore,
        full: &UnlimitedOcrConfig,
    ) -> Result<Self, DecoderError> {
        let cfg = full.language_config.clone();
        let q = full
            .quantization
            .as_ref()
            .ok_or_else(|| DecoderError::Config("checkpoint is not quantized".into()))?;
        let (gs, bits) = (q.group_size, q.bits);
        if cfg.topk_method != "greedy" {
            return Err(DecoderError::Config(format!(
                "unsupported topk_method {}",
                cfg.topk_method
            )));
        }
        if cfg.num_attention_heads == 0 || !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads)
        {
            return Err(DecoderError::Config(
                "hidden_size not divisible by heads".into(),
            ));
        }
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        // Custom Metal kernels need Metal; CPU and CUDA keep mlx ops.
        let gpu = aimd_mlx::metal_kernels_available().unwrap_or(false);
        let dash = match &full.sparse_attention_config {
            Some(sp) if sp.is_dash() => Some(
                DashSpec::from_config(sp, cfg.num_hidden_layers, head_dim)
                    .map_err(DecoderError::Config)?,
            ),
            _ => None,
        };
        let fused_moe = bits == 4 && gpu;
        let p = "language_model.model";
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let l = format!("{p}.layers.{i}");
            let attn = Attention {
                q: QWeights::load(store, &format!("{l}.self_attn.q_proj"), gs, bits)?,
                k: QWeights::load(store, &format!("{l}.self_attn.k_proj"), gs, bits)?,
                v: QWeights::load(store, &format!("{l}.self_attn.v_proj"), gs, bits)?,
                o: QWeights::load(store, &format!("{l}.self_attn.o_proj"), gs, bits)?,
                heads: cfg.num_attention_heads,
                kv_heads: cfg.num_key_value_heads,
                head_dim,
                rope_base: cfg.rope_theta,
                sdpa_decode: gpu,
                dash: match &dash {
                    Some(spec) => Some((
                        DashRouter::load(
                            store,
                            &format!("{l}.self_attn.dash_router"),
                            cfg.hidden_size,
                            spec.index_dim,
                        )?,
                        spec.consts(i, head_dim),
                    )),
                    None => None,
                },
            };
            let mlp = if i >= cfg.first_k_dense_replace && cfg.n_routed_experts > 0 {
                let gate_w = dense(
                    store,
                    &format!("{l}.mlp.gate.weight"),
                    Some(&[cfg.n_routed_experts, cfg.hidden_size]),
                )?;
                Mlp::Moe(Box::new(Moe {
                    gate_wt: gate_w.transpose(&[1, 0])?,
                    gate_w,
                    n_experts: cfg.n_routed_experts,
                    top_k: cfg.num_experts_per_tok,
                    sw_gate: QWeights::load(
                        store,
                        &format!("{l}.mlp.switch_mlp.gate_proj"),
                        gs,
                        bits,
                    )?,
                    sw_up: QWeights::load(store, &format!("{l}.mlp.switch_mlp.up_proj"), gs, bits)?,
                    sw_down: QWeights::load(
                        store,
                        &format!("{l}.mlp.switch_mlp.down_proj"),
                        gs,
                        bits,
                    )?,
                    shared: DenseMlp::load(
                        store,
                        &format!("{l}.mlp.shared_experts"),
                        gs,
                        bits,
                        fused_moe,
                    )?,
                    fused: fused_moe,
                }))
            } else {
                Mlp::Dense(Box::new(DenseMlp::load(
                    store,
                    &format!("{l}.mlp"),
                    gs,
                    bits,
                    fused_moe,
                )?))
            };
            layers.push(Layer {
                input_ln: dense(
                    store,
                    &format!("{l}.input_layernorm.weight"),
                    Some(&[cfg.hidden_size]),
                )?,
                post_ln: dense(
                    store,
                    &format!("{l}.post_attention_layernorm.weight"),
                    Some(&[cfg.hidden_size]),
                )?,
                attn,
                mlp,
            });
        }
        let embed = QWeights::load(store, &format!("{p}.embed_tokens"), gs, bits)?;
        let lm_head = QWeights::load(store, "language_model.lm_head", gs, bits)?;
        if embed.shape != [cfg.vocab_size, cfg.hidden_size]
            || lm_head.shape != [cfg.vocab_size, cfg.hidden_size]
        {
            return Err(DecoderError::Layout(format!(
                "embed {:?} / lm_head {:?} vs vocab {} x hidden {}",
                embed.shape, lm_head.shape, cfg.vocab_size, cfg.hidden_size
            )));
        }
        Ok(Self {
            eps: cfg.rms_norm_eps,
            norm: dense(store, &format!("{p}.norm.weight"), Some(&[cfg.hidden_size]))?,
            embed,
            layers,
            lm_head,
            cfg,
            dash,
        })
    }

    /// The `<PAGE>` marker id of an Unlimited-OCR-2 decoder.
    pub(crate) fn page_token_id(&self) -> Option<u32> {
        self.dash.as_ref().and_then(|d| d.page_token_id)
    }

    pub fn is_dash(&self) -> bool {
        self.dash.is_some()
    }

    pub fn config(&self) -> &LanguageConfig {
        &self.cfg
    }

    /// `make_cache()`: ring caches when the config (or `window`) sets a
    /// sliding window.
    pub(crate) fn make_cache(&self, window: Option<usize>) -> Vec<KvCache> {
        let w = window.or(self.cfg.window());
        (0..self.layers.len()).map(|_| KvCache::new(w)).collect()
    }

    /// `embed_tokens(ids)` for `[1, n]` ids -> `[1, n, hidden]` bf16.
    pub(crate) fn embed_ids(&self, ids: &MlxArray) -> Result<MlxArray, MlxError> {
        self.embed.embed(ids)
    }

    /// Validated host ids -> `[1, n]` u32 array.
    pub(crate) fn ids_array(&self, ids: &[u32]) -> Result<MlxArray, DecoderError> {
        if let Some(&bad) = ids.iter().find(|&&id| id as usize >= self.cfg.vocab_size) {
            return Err(DecoderError::Layout(format!(
                "token id {bad} outside vocab {}",
                self.cfg.vocab_size
            )));
        }
        Ok(MlxArray::from_u32(&[1, ids.len()], ids)?)
    }

    /// Generation cache for `prompt_ids`: ring KV caches (v1, `window`
    /// overriding the config) or full caches plus the DASH state (v2).
    pub(crate) fn new_cache(
        &self,
        window: Option<usize>,
        prompt_ids: &[u32],
    ) -> Result<OcrCache, DecoderError> {
        let Some(spec) = &self.dash else {
            // R-SWA keeps the whole prompt, as the reference's one-call
            // prefill does; the chunked prefill must not move that boundary.
            let mut kv = self.make_cache(window);
            for c in &mut kv {
                c.retain_prompt(prompt_ids.len());
            }
            return Ok(OcrCache { kv, dash: None });
        };
        let n = self.layers.len();
        Ok(OcrCache {
            kv: (0..n).map(|_| KvCache::new(None)).collect(),
            dash: Some(DashRun::start(prompt_ids, spec, n).map_err(DecoderError::Dash)?),
        })
    }

    /// One forward over `h` `[1, L, hidden]` at the cache offset. DASH
    /// decode steps (past the prompt) need the input `token` (`[1, 1]`,
    /// in-graph) for the page state.
    pub(crate) fn step(
        &self,
        h: &MlxArray,
        cache: &mut OcrCache,
        token: Option<&MlxArray>,
    ) -> Result<MlxArray, DecoderError> {
        let offset = cache.offset();
        let (Some(spec), Some(run)) = (&self.dash, cache.dash.as_mut()) else {
            return Ok(self.forward(h, &mut cache.kv)?);
        };
        let len = h.shape()?[1];
        let step = run
            .step(spec, offset, len, token, h.dtype()?)
            .map_err(DecoderError::Dash)?;
        let ios = run.layers.iter_mut().map(|index| {
            Some(DashIo {
                spec,
                step: &step,
                index,
                eps: self.eps,
            })
        });
        Ok(self.run_layers(h, &mut cache.kv, ios)?)
    }

    /// `DeepseekV2RingSlidingModel.__call__(inputs_embeds=h, cache)`:
    /// layers + final norm -> `[B, L, hidden]`.
    pub(crate) fn forward(
        &self,
        h: &MlxArray,
        caches: &mut [KvCache],
    ) -> Result<MlxArray, MlxError> {
        self.run_layers(h, caches, std::iter::repeat_with(|| None))
    }

    fn run_layers<'a>(
        &self,
        h: &MlxArray,
        caches: &mut [KvCache],
        dash: impl Iterator<Item = Option<DashIo<'a>>>,
    ) -> Result<MlxArray, MlxError> {
        let mut h = h.try_clone()?;
        for ((layer, cache), io) in self.layers.iter().zip(caches.iter_mut()).zip(dash) {
            let r = layer
                .attn
                .forward(&h.rms_norm(Some(&layer.input_ln), self.eps)?, cache, io)?;
            let hh = h.add(&r)?;
            let xn = hh.rms_norm(Some(&layer.post_ln), self.eps)?;
            h = match &layer.mlp {
                Mlp::Dense(m) => m.forward(&xn, &[&hh])?,
                Mlp::Moe(m) => m.forward(&xn, &hh)?,
            };
        }
        h.rms_norm(Some(&self.norm), self.eps)
    }

    /// Untied quantized `lm_head`.
    pub(crate) fn logits(&self, hidden: &MlxArray) -> Result<MlxArray, MlxError> {
        self.lm_head.linear(hidden)
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::unlimited_ocr::generate::{
        OCR_IMAGE_TOKEN_ID, OcrGenerationSettings, greedy_ocr_token_stream, splice_image_features,
    };
    use crate::models::unlimited_ocr::test_support::{
        Parity, compare_rows, mlx_rows_f32, ocr_decoder_golden as golden,
        ocr2_decoder_golden as golden2, shared_pipeline, shared_pipeline2,
    };
    use std::collections::BTreeSet;

    /// Logits rows vs the bf16 reference. Prefill is bit-exact; decode
    /// steps drift by bf16 ulps because brew mlx 0.32.4's small-M
    /// `quantized_matmul` kernel rounds differently from the reference's
    /// mlx 0.32.2 (measured: ring logits rel rms 1.3e-2, max 1.1% of max
    /// |ref|; greedy tokens still identical).
    const LOGITS: Parity = Parity {
        rel_rms: 0.03,
        max_frac: 0.05,
    };
    /// Prefill logits bound. `LOGITS` holds Metal mlx rounding; mlx's CUDA
    /// kernels round the 4-bit prefill differently (RTX 5090: rel rms
    /// 3.8e-2, max 3.1% of max |ref|), so off Metal this is a noise bound
    /// and the forced greedy stream must still match exactly.
    fn prefill_logits_bound() -> Parity {
        if aimd_mlx::metal_kernels_available().unwrap_or(false) {
            LOGITS
        } else {
            Parity {
                rel_rms: 0.06,
                max_frac: 0.08,
            }
        }
    }
    /// Layer-0 key rows (one RMSNorm + q-proj + RoPE from the embeddings).
    const KEYS: Parity = Parity {
        rel_rms: 0.01,
        max_frac: 0.01,
    };

    fn argmax(row: &[f32]) -> u32 {
        let mut best = 0usize;
        for (i, &v) in row.iter().enumerate() {
            if v > row[best] {
                best = i;
            }
        }
        best as u32
    }

    struct Golden {
        m: serde_json::Value,
        prompt: Vec<u32>,
        embeds: MlxArray,
    }

    fn golden_inputs(dec: &OcrDecoder) -> Golden {
        let m = golden::manifest();
        let prompt = golden::u32s(&m["prompt_ids"]);
        let feats = golden::image_features(&m);
        let embeds = splice_image_features(dec, &prompt, &feats, OCR_IMAGE_TOKEN_ID).unwrap();
        Golden { m, prompt, embeds }
    }

    /// Full-prompt prefill (unchunked, like the exporter) over ring caches.
    fn prefill(dec: &OcrDecoder, g: &Golden, window: usize) -> (Vec<KvCache>, Vec<f32>, usize) {
        let mut caches = dec.make_cache(Some(window));
        let h = dec.forward(&g.embeds, &mut caches).unwrap();
        let (logits, vocab) = mlx_rows_f32(&dec.logits(&h).unwrap());
        (caches, logits, vocab)
    }

    fn step(dec: &OcrDecoder, caches: &mut [KvCache], tok: u32) -> Vec<f32> {
        let emb = dec.embed_ids(&dec.ids_array(&[tok]).unwrap()).unwrap();
        let h = dec.forward(&emb, caches).unwrap();
        mlx_rows_f32(&dec.logits(&h).unwrap()).0
    }

    /// Prefill logits rows + the window-128 forced greedy stream (16
    /// tokens, EOS at step 15) against mlx-vlm.
    #[test]
    fn decoder_prefill_and_forced_stream_golden() {
        let Some(p) = shared_pipeline() else {
            return;
        };
        let dec = p.decoder();
        let g = golden_inputs(dec);
        let (mut caches, logits, vocab) = prefill(dec, &g, 128);
        let kept = golden::usizes(&g.m["logits_kept_positions"]);
        let want = golden::read_bf16_f32("logits.bf16");
        let mut fails = Vec::new();
        compare_rows(
            "prefill logits",
            &logits,
            vocab,
            &kept,
            &want,
            prefill_logits_bound(),
            &mut fails,
        );
        assert!(fails.is_empty(), "{}", fails.join("\n"));

        let want_tokens = golden::u32s(&g.m["greedy_tokens"]);
        let p_len = g.prompt.len();
        let mut got = vec![argmax(&logits[(p_len - 1) * vocab..p_len * vocab])];
        for &tok in &want_tokens[..want_tokens.len() - 1] {
            got.push(argmax(&step(dec, &mut caches, tok)));
        }
        assert_eq!(got, want_tokens, "forced greedy stream");
    }

    /// Window-4 ring run: forced stream, logits rows of the first ring
    /// writes, and layer-0 key rows after decode step 4.
    #[test]
    fn decoder_ring_window_golden() {
        let Some(p) = shared_pipeline() else {
            return;
        };
        let dec = p.decoder();
        let g = golden_inputs(dec);
        let ring = &g.m["ring"];
        let window = ring["window"].as_u64().unwrap() as usize;
        let forced = golden::u32s(&ring["forced_tokens"]);
        let kept = golden::usizes(&ring["kept_steps"]);
        let dump = &ring["cache_dump"];
        let (after, from, to) = (
            dump["after_step"].as_u64().unwrap() as usize,
            dump["from_row"].as_u64().unwrap() as usize,
            dump["to_row"].as_u64().unwrap() as usize,
        );
        let (mut caches, logits, vocab) = prefill(dec, &g, window);
        let p_len = g.prompt.len();
        let mut got = vec![argmax(&logits[(p_len - 1) * vocab..p_len * vocab])];
        let mut rows = Vec::new();
        let mut fails = Vec::new();
        for (k, &tok) in forced.iter().enumerate().take(forced.len() - 1) {
            let row = step(dec, &mut caches, tok);
            got.push(argmax(&row));
            if kept.contains(&k) {
                rows.extend(row);
            }
            if k == after {
                assert_eq!(caches[0].prefill_length(), Some(p_len));
                let keys = caches[0].keys_buffer().unwrap();
                let s = keys.shape().unwrap();
                let sl = keys
                    .slice(&[0, 0, from, 0], &[1, s[1], to, s[3]], &[1, 1, 1, 1])
                    .unwrap()
                    .transpose(&[0, 2, 1, 3])
                    .unwrap();
                let (k_rows, _) = mlx_rows_f32(&sl);
                let cols = s[1] * s[3];
                let want = golden::read_f32("ring_cache_keys.f32");
                let idx: Vec<usize> = (0..to - from).collect();
                compare_rows(
                    "ring layer-0 keys",
                    &k_rows,
                    cols,
                    &idx,
                    &want,
                    KEYS,
                    &mut fails,
                );
            }
        }
        let idx: Vec<usize> = (0..kept.len()).collect();
        let want = golden::read_bf16_f32("ring_logits.bf16");
        compare_rows("ring logits", &rows, vocab, &idx, &want, LOGITS, &mut fails);
        assert!(fails.is_empty(), "{}", fails.join("\n"));
        assert_eq!(got, forced, "ring forced stream");
    }

    /// The production loop (chunked prefill, guard, pipelined decode)
    /// stops on EOS at the golden step and drops it.
    #[test]
    fn decoder_token_stream_stops_on_eos() {
        let Some(p) = shared_pipeline() else {
            return;
        };
        let dec = p.decoder();
        let g = golden_inputs(dec);
        let want = golden::u32s(&g.m["greedy_tokens"]);
        let eos_step = g.m["first_eos_step"].as_u64().unwrap() as usize;
        let eos: BTreeSet<u32> = golden::u32s(&g.m["eos_ids"]).into_iter().collect();
        let settings = OcrGenerationSettings {
            max_new_tokens: 64,
            ..OcrGenerationSettings::default()
        };
        let (stream, _) =
            greedy_ocr_token_stream(dec, &g.prompt, &g.embeds, &eos, settings).unwrap();
        assert_eq!(
            stream.finish_reason,
            crate::models::unlimited_ocr::generate::FinishReason::Stop
        );
        assert_eq!(stream.tokens, want[..eos_step]);
        // max_new_tokens caps sampled tokens (Length).
        let settings = OcrGenerationSettings {
            max_new_tokens: 3,
            ..settings
        };
        let (stream, _) =
            greedy_ocr_token_stream(dec, &g.prompt, &g.embeds, &eos, settings).unwrap();
        assert_eq!(
            stream.finish_reason,
            crate::models::unlimited_ocr::generate::FinishReason::Length
        );
        assert_eq!(stream.tokens, want[..3]);
    }

    /// A greedy step whose reference top-2 logits are this close may flip
    /// (bf16 ties: the stream has exact ties).
    const TIE_GAP: f64 = 0.5;

    fn u32_rows(a: &MlxArray) -> Vec<u32> {
        a.astype(aimd_mlx::MlxDtype::U32)
            .unwrap()
            .to_u32_vec()
            .unwrap()
    }

    /// Unlimited-OCR-2 (DASH) against the mlx-vlm port on a two-page
    /// base request: prefill logits rows, prefill routing on sampled
    /// layers, the forced greedy stream (both `<PAGE>` markers, EOS) and
    /// the decode logits after each `<PAGE>`.
    #[test]
    fn decoder2_dash_golden() {
        let Some(p) = shared_pipeline2() else {
            return;
        };
        let dec = p.decoder();
        assert!(dec.is_dash());
        let m = golden2::manifest();
        let prompt = golden::u32s(&m["prompt_ids"]);
        let feats = golden2::image_features(&m);
        let image_token = m["image_token_id"].as_u64().unwrap() as u32;
        let embeds = splice_image_features(dec, &prompt, &feats, image_token).unwrap();
        let mut cache = dec.new_cache(None, &prompt).unwrap();
        route_probe::start();
        let h = dec.step(&embeds, &mut cache, None).unwrap();
        let routes = route_probe::take();
        let (logits, vocab) = mlx_rows_f32(&dec.logits(&h).unwrap());
        let mut fails = Vec::new();
        let kept = golden::usizes(&m["logits_kept_positions"]);
        let want = golden2::read_bf16_f32("logits.bf16");
        compare_rows(
            "prefill logits",
            &logits,
            vocab,
            &kept,
            &want,
            LOGITS,
            &mut fails,
        );

        assert_eq!(routes.len(), dec.layers.len(), "one route per layer");
        let rows = golden::usizes(&m["route_rows"]);
        for probe in m["routes"].as_array().unwrap() {
            let layer = probe["layer"].as_u64().unwrap() as usize;
            let r = &routes[layer];
            let k = r.topi.shape().unwrap()[1];
            let topi = u32_rows(&r.topi);
            let valid = u32_rows(&r.valid);
            let bias = r.bias.to_f32_vec().unwrap();
            for (i, &q) in rows.iter().enumerate() {
                let wt = golden::u32s(&probe["topi"][i]);
                let wv: Vec<bool> = probe["valid"][i]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_bool().unwrap())
                    .collect();
                let mut want_sel: Vec<(u32, f64)> = (0..k)
                    .filter(|&j| wv[j])
                    .map(|j| (wt[j], probe["bias"][i][j].as_f64().unwrap()))
                    .collect();
                let mut got_sel: Vec<(u32, f64)> = (0..k)
                    .filter(|&j| valid[q * k + j] != 0)
                    .map(|j| (topi[q * k + j], bias[q * k + j] as f64))
                    .collect();
                want_sel.sort_by_key(|x| x.0);
                got_sel.sort_by_key(|x| x.0);
                let same_blocks = want_sel.iter().map(|x| x.0).eq(got_sel.iter().map(|x| x.0));
                // Layer 0 sees the spliced embeddings directly (bias error
                // ~5e-7); deeper layers inherit bf16 hidden-state drift,
                // which the log-prior amplifies in the entmax tail
                // (measured: 0.38 on a -5.1 bias, a ~e^-5 weight).
                let bias_excess = want_sel
                    .iter()
                    .zip(&got_sel)
                    .map(|(a, b)| (a.1 - b.1).abs() - (0.05 + 0.1 * a.1.abs()))
                    .fold(f64::NEG_INFINITY, f64::max);
                eprintln!(
                    "layer {layer} row {q}: blocks equal {same_blocks}, bias excess {bias_excess:.3e}"
                );
                if !same_blocks || bias_excess > 0.0 {
                    fails.push(format!(
                        "layer {layer} row {q} routing: want {want_sel:?} got {got_sel:?}"
                    ));
                }
            }
        }

        let want_tokens = golden::u32s(&m["greedy_tokens"]);
        let gaps: Vec<f64> = m["top2_gaps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g.as_f64().unwrap())
            .collect();
        let decode_kept = golden::usizes(&m["decode_kept_steps"]);
        let p_len = prompt.len();
        let mut row = logits[(p_len - 1) * vocab..p_len * vocab].to_vec();
        let mut decode_rows = Vec::new();
        let mut flips = Vec::new();
        for (k, &tok) in want_tokens.iter().enumerate() {
            let got = argmax(&row);
            if got != tok {
                if gaps[k] < TIE_GAP {
                    flips.push(k);
                } else {
                    fails.push(format!(
                        "greedy step {k}: want {tok} got {got} (gap {})",
                        gaps[k]
                    ));
                    break;
                }
            }
            if k + 1 == want_tokens.len() {
                break;
            }
            let ids = dec.ids_array(&[tok]).unwrap().reshape(&[1, 1]).unwrap();
            let emb = dec.embed_ids(&ids).unwrap();
            let h = dec.step(&emb, &mut cache, Some(&ids)).unwrap();
            row = mlx_rows_f32(&dec.logits(&h).unwrap()).0;
            if decode_kept.contains(&k) {
                decode_rows.extend_from_slice(&row);
            }
        }
        eprintln!(
            "greedy stream: {} tokens, tie flips at {flips:?}",
            want_tokens.len()
        );
        let idx: Vec<usize> = (0..decode_kept.len()).collect();
        let want = golden2::read_bf16_f32("decode_logits.bf16");
        compare_rows(
            "decode logits",
            &decode_rows,
            vocab,
            &idx,
            &want,
            LOGITS,
            &mut fails,
        );
        assert!(fails.is_empty(), "{}", fails.join("\n"));

        // The production loop (chunked prefill, pipelined in-graph DASH
        // decode) agrees with the reference up to the first bf16 tie.
        let eos: BTreeSet<u32> = [m["eos_id"].as_u64().unwrap() as u32].into();
        let settings = OcrGenerationSettings {
            max_new_tokens: 64,
            ..OcrGenerationSettings::default()
        };
        let (stream, _) = greedy_ocr_token_stream(dec, &prompt, &embeds, &eos, settings).unwrap();
        let first_tie = flips.first().copied().unwrap_or(64).min(64);
        assert_eq!(stream.tokens[..first_tie], want_tokens[..first_tie]);
    }
}
