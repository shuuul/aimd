//! Greedy generation for MiniCPM-OCR, structured like mlx-vlm
//! `generate.ar.generate_step` at temperature 0:
//!
//! - chunked prefill (2048) over `prompt[:-1]`, evaluating only the layer
//!   caches (KV for full attention, conv + recurrent state for DeltaNet),
//!   then `mx.clear_cache()`;
//! - the last prompt token as the first step; positions are plain
//!   sequential (MiniCPM's `get_rope_index` has no image offsets);
//! - `argmax` over f32 logits (transformers casts before its processors);
//! - `async_eval` pipelining and `clear_cache` every 256 tokens;
//! - EOS stops generation and is not part of the returned stream.
//!
//! The SDK decodes with `no_repeat_ngram_size = 100`; this port applies it
//! as [`SlidingWindowNoRepeatNgramProcessor`] over the generated stream
//! (window = the whole stream). A step can only be banned when the last
//! `n - 2` known tokens already occurred earlier, so pipelining continues
//! until that host-side check says a ban is possible.

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use aimd_mlx::{BinaryOp, MlxArray, MlxDtype, MlxError};

use super::decoder::{Capture, DecoderError, LayerCache, MiniCpmDecoder, Rope, validity_mask};
use super::dspark::{DSparkDrafter, DraftBatch};
pub use crate::models::paddleocr_vl::generate::{FinishReason, GreedyStream};
use crate::models::unlimited_ocr::generate::SlidingWindowNoRepeatNgramProcessor;

/// `generate_step(prefill_step_size=2048)`.
pub const PREFILL_STEP_SIZE: usize = 2048;

/// The SDK's full-page budget (`max_new_tokens` of the default task).
pub const DEFAULT_MAX_TOKENS: usize = 8192;

/// The SDK's `no_repeat_ngram_size`.
pub const NO_REPEAT_NGRAM_SIZE: usize = 100;

/// Generation failures.
#[derive(Debug, thiserror::Error)]
pub enum GenerateError {
    #[error(transparent)]
    Mlx(#[from] MlxError),
    #[error(transparent)]
    Decoder(#[from] DecoderError),
    #[error("empty prompt")]
    EmptyPrompt,
    #[error("generate layout: {0}")]
    Layout(String),
}

/// Whether the step after `stream` (one more, still unknown token) can
/// have a banned token: some earlier window must already match the last
/// `n - 2` tokens.
fn may_ban(guard: &SlidingWindowNoRepeatNgramProcessor, stream: &[u32]) -> bool {
    let n = guard.ngram_size();
    if n < 3 {
        return stream.len() + 1 >= n;
    }
    let len = stream.len() + 1;
    if len < n {
        return false;
    }
    let start = len.saturating_sub(guard.window());
    let end = len + 1 - n;
    let tail = &stream[stream.len() - (n - 2)..];
    (start..end).any(|i| &stream[i..i + n - 2] == tail)
}

fn sample(
    logits: &MlxArray,
    banned: &[u32],
    capture: Option<&mut Vec<MlxArray>>,
) -> Result<MlxArray, MlxError> {
    let logits = if banned.is_empty() {
        logits.try_clone()?
    } else {
        let idx = MlxArray::from_u32(&[1, banned.len()], banned)?;
        let vals = MlxArray::scalar_bf16(f32::NEG_INFINITY)?
            .astype(logits.dtype()?)?
            .broadcast_to(&[1, banned.len()])?;
        logits.put_along_axis(&idx, &vals, 1)?
    };
    // transformers casts the logits to f32 before the processors and the
    // argmax.
    let logits = logits.astype(MlxDtype::F32)?;
    let y = logits.argmax(-1, false)?;
    if let Some(c) = capture {
        c.push(logits.subtract(&logits.logsumexp(-1, true)?)?);
    }
    Ok(y)
}

/// Runs greedy decoding over prompt embeddings `embeds` (`[1, P, hidden]`,
/// image features spliced). `ngram` enables the no-repeat guard. With
/// `capture`, the `[1, vocab]` logprobs of every sampled step are pushed
/// (first step = the last prompt token).
pub fn greedy_stream(
    decoder: &MiniCpmDecoder,
    embeds: &MlxArray,
    eos_ids: &BTreeSet<u32>,
    max_tokens: usize,
    ngram: Option<usize>,
    mut capture: Option<&mut Vec<MlxArray>>,
) -> Result<GreedyStream, GenerateError> {
    let s = embeds.shape()?;
    let d = decoder.hidden_size();
    if s.len() != 3 || s[0] != 1 || s[2] != d {
        return Err(GenerateError::Layout(format!("embeds {s:?}")));
    }
    let p = s[1];
    if p == 0 {
        return Err(GenerateError::EmptyPrompt);
    }
    let guard = ngram_guard(ngram)?;
    let vocab = decoder.config().vocab_size;
    let t0 = Instant::now();
    let mut caches = decoder.make_cache();

    let mut start = 0usize;
    while p - start > 1 {
        let n = (p - start - 1).min(PREFILL_STEP_SIZE);
        let chunk = embeds.slice(&[0, start, 0], &[1, start + n, d], &[1, 1, 1])?;
        decoder.forward(&chunk, &decoder.rope(start, n)?, &mut caches, None)?;
        let mut state = Vec::new();
        for c in &caches {
            state.extend(c.state()?);
        }
        let refs: Vec<&MlxArray> = state.iter().collect();
        MlxArray::eval_many(&refs)?;
        start += n;
        aimd_mlx::clear_cache()?;
    }

    // Step `k` (0 = the last prompt token) sits at position `p - 1 + k`.
    let table: Rope = decoder.rope(p - 1, max_tokens.max(1))?;
    let mut k = 0usize;
    let mut submit = |emb: &MlxArray,
                      caches: &mut [LayerCache],
                      banned: &[u32],
                      capture: Option<&mut Vec<MlxArray>>|
     -> Result<MlxArray, GenerateError> {
        let h = decoder.forward(emb, &table.row(k)?, caches, None)?;
        k += 1;
        let logits = decoder.logits(&h)?.reshape(&[1, vocab])?;
        let y = sample(&logits, banned, capture)?;
        MlxArray::async_eval(&[&y])?;
        Ok(y)
    };

    let last = embeds.slice(&[0, p - 1, 0], &[1, p, d], &[1, 1, 1])?;
    let mut y = submit(&last, &mut caches, &[], capture.as_deref_mut())?;

    let mut tokens: Vec<u32> = Vec::with_capacity(max_tokens.min(1024));
    let mut finish_reason = FinishReason::Length;
    let mut prefill = None;
    if max_tokens > 0 {
        loop {
            // Submit step k+1 before reading token k unless the guard may
            // need token k to build its ban list.
            let pipelined = tokens.len() + 1 < max_tokens
                && !guard.as_ref().is_some_and(|g| may_ban(g, &tokens));
            let next = if pipelined {
                let emb = decoder.embed_ids(&y.reshape(&[1, 1])?)?;
                let mut lp = capture.is_some().then(Vec::new);
                Some((submit(&emb, &mut caches, &[], lp.as_mut())?, lp))
            } else {
                None
            };
            let tok = y.to_u32_vec()?[0];
            if prefill.is_none() {
                prefill = Some(t0.elapsed());
            }
            if eos_ids.contains(&tok) {
                finish_reason = FinishReason::Stop;
                break;
            }
            tokens.push(tok);
            if tokens.len() >= max_tokens {
                break;
            }
            let (next_y, lp) = match next {
                Some(n) => n,
                None => {
                    let banned = guard
                        .as_ref()
                        .map(|g| g.banned(&tokens, vocab))
                        .unwrap_or_default();
                    let emb = decoder.embed_ids(&y.reshape(&[1, 1])?)?;
                    let mut lp = capture.is_some().then(Vec::new);
                    (submit(&emb, &mut caches, &banned, lp.as_mut())?, lp)
                }
            };
            if let (Some(c), Some(l)) = (capture.as_deref_mut(), lp) {
                c.extend(l);
            }
            y = next_y;
            if tokens.len().is_multiple_of(256) {
                aimd_mlx::clear_cache()?;
            }
        }
    }
    let total = t0.elapsed();
    let prefill = prefill.unwrap_or(total);
    Ok(GreedyStream {
        tokens,
        finish_reason,
        prefill,
        decode: total.saturating_sub(prefill),
    })
}

/// Acceptance counters of one speculative run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpecStats {
    /// Verify forwards.
    pub rounds: usize,
    /// Draft tokens proposed / accepted.
    pub drafted: usize,
    pub accepted: usize,
}

/// Concatenated captured layer outputs `[1, L, n_layers * hidden]`.
fn features(capture: Capture<'_>) -> Result<MlxArray, MlxError> {
    let refs: Vec<&MlxArray> = capture.hidden.iter().collect();
    MlxArray::concatenate(&refs, 2)
}

/// The greedy target token for row `row` of `logits` (`[rows, vocab]`)
/// after `stream`: the row's argmax unless the no-repeat guard bans it,
/// then the argmax of the masked row (what [`greedy_stream`] samples).
fn target_token(
    logits: &MlxArray,
    row: usize,
    argmax: u32,
    guard: Option<&SlidingWindowNoRepeatNgramProcessor>,
    stream: &[u32],
    vocab: usize,
) -> Result<u32, MlxError> {
    let Some(g) = guard.filter(|g| may_ban(g, stream)) else {
        return Ok(argmax);
    };
    let banned = g.banned(stream, vocab);
    if !banned.contains(&argmax) {
        return Ok(argmax);
    }
    let row = logits.slice(&[row, 0], &[row + 1, vocab], &[1, 1])?;
    Ok(sample(&row, &banned, None)?.to_u32_vec()?[0])
}

/// Greedy decoding with DSpark speculative decoding (vLLM `dspark`,
/// `num_speculative_tokens = block_size`, no adaptive verification).
///
/// Each round drafts `N` tokens after the anchor, runs the target over
/// `[anchor, drafts]` once (capturing the drafter's feature layers and the
/// DeltaNet recurrence inputs), keeps the longest draft prefix that matches
/// the target's own greedy choices plus the target's next token, and rolls
/// the target caches back to the kept positions. Every emitted token is the
/// target's greedy choice (with the no-repeat guard) given the emitted
/// prefix, so the drafter only changes speed; the batched verify forward
/// rounds differently from one-token steps, so ties can resolve
/// differently from [`greedy_stream`].
pub fn speculative_stream(
    decoder: &MiniCpmDecoder,
    drafter: &DSparkDrafter,
    embeds: &MlxArray,
    eos_ids: &BTreeSet<u32>,
    max_tokens: usize,
    ngram: Option<usize>,
) -> Result<(GreedyStream, SpecStats), GenerateError> {
    let s = embeds.shape()?;
    let d = decoder.hidden_size();
    if s.len() != 3 || s[0] != 1 || s[2] != d {
        return Err(GenerateError::Layout(format!("embeds {s:?}")));
    }
    let p = s[1];
    if p == 0 {
        return Err(GenerateError::EmptyPrompt);
    }
    let guard = ngram_guard(ngram)?;
    let layers = drafter.config().target_layer_ids();
    let vocab = decoder.config().vocab_size;
    let t0 = Instant::now();
    let mut caches = decoder.make_cache();
    let mut stats = SpecStats::default();

    // Prefill `prompt[:-1]` in chunks, then the last prompt token alone, as
    // `greedy_stream` does, keeping every position's target features.
    let mut context = Vec::new();
    let mut start = 0usize;
    while p - start > 1 {
        let m = (p - start - 1).min(PREFILL_STEP_SIZE);
        let chunk = embeds.slice(&[0, start, 0], &[1, start + m, d], &[1, 1, 1])?;
        let mut cap = Capture::new(layers, false);
        decoder.forward_capture(
            &chunk,
            &decoder.rope(start, m)?,
            None,
            &mut caches,
            None,
            Some(&mut cap),
        )?;
        let f = features(cap)?;
        let mut state = vec![f.try_clone()?];
        for c in &caches {
            state.extend(c.state()?);
        }
        let refs: Vec<&MlxArray> = state.iter().collect();
        MlxArray::eval_many(&refs)?;
        context.push(f);
        start += m;
        aimd_mlx::clear_cache()?;
    }
    let last = embeds.slice(&[0, p - 1, 0], &[1, p, d], &[1, 1, 1])?;
    let mut cap = Capture::new(layers, false);
    let h = decoder.forward_capture(
        &last,
        &decoder.rope(p - 1, 1)?,
        None,
        &mut caches,
        None,
        Some(&mut cap),
    )?;
    context.push(features(cap)?);
    let refs: Vec<&MlxArray> = context.iter().collect();
    let ctx = MlxArray::concatenate(&refs, 1)?;
    drop(context);
    let mut tokens: Vec<u32> = Vec::with_capacity(max_tokens.min(1024));
    let mut finish_reason = FinishReason::Length;
    let logits = decoder.logits(&h)?.reshape(&[1, vocab])?;
    let anchor = sample(&logits, &[], None)?.to_u32_vec()?[0];
    let prefill = t0.elapsed();

    if max_tokens > 0 {
        if eos_ids.contains(&anchor) {
            finish_reason = FinishReason::Stop;
        } else {
            tokens.push(anchor);
            if tokens.len() < max_tokens {
                finish_reason = speculate(
                    decoder,
                    drafter,
                    &mut caches,
                    ctx,
                    p,
                    &mut tokens,
                    max_tokens,
                    guard.as_ref(),
                    eos_ids,
                    &mut stats,
                )?;
            }
        }
    }
    let total = t0.elapsed();
    Ok((
        GreedyStream {
            tokens,
            finish_reason,
            prefill,
            decode: total.saturating_sub(prefill),
        },
        stats,
    ))
}

fn ngram_guard(
    ngram: Option<usize>,
) -> Result<Option<SlidingWindowNoRepeatNgramProcessor>, GenerateError> {
    ngram
        .map(|n| {
            SlidingWindowNoRepeatNgramProcessor::new(n, usize::MAX)
                .map_err(|e| GenerateError::Layout(e.to_string()))
        })
        .transpose()
}

/// DSpark rounds from the anchor `tokens.last()` (sampled, not yet fed) of
/// a prompt of `p` positions: `caches` hold every fed position (`p +
/// tokens.len() - 1`) and `ctx` (`[1, that many, features]`) their target
/// features, which a fresh drafter cache consumes in the first round.
#[allow(clippy::too_many_arguments)]
fn speculate(
    decoder: &MiniCpmDecoder,
    drafter: &DSparkDrafter,
    caches: &mut [LayerCache],
    mut ctx: MlxArray,
    p: usize,
    tokens: &mut Vec<u32>,
    max_tokens: usize,
    guard: Option<&SlidingWindowNoRepeatNgramProcessor>,
    eos_ids: &BTreeSet<u32>,
    stats: &mut SpecStats,
) -> Result<FinishReason, GenerateError> {
    let layers = drafter.config().target_layer_ids();
    let vocab = decoder.config().vocab_size;
    let n = drafter.config().block_size;
    let mut draft_cache = drafter.make_cache();
    let mut anchor = *tokens
        .last()
        .ok_or_else(|| GenerateError::Layout("speculate without an anchor".into()))?;
    loop {
        let drafts = drafter.draft(decoder, anchor, &ctx, &mut draft_cache)?;
        let ids = MlxArray::concatenate(&[&MlxArray::from_u32(&[1], &[anchor])?, &drafts], 0)?;
        let x = decoder.embed_ids(&ids.reshape(&[1, n + 1])?)?;
        let pos = p + tokens.len() - 1;
        let mut cap = Capture::new(layers, true);
        let h = decoder.forward_capture(
            &x,
            &decoder.rope(pos, n + 1)?,
            None,
            caches,
            None,
            Some(&mut cap),
        )?;
        let logits = decoder.logits(&h)?.reshape(&[n + 1, vocab])?;
        let argmax = logits.astype(MlxDtype::F32)?.argmax(-1, false)?;
        let feats = features(Capture {
            layers,
            hidden: std::mem::take(&mut cap.hidden),
            delta: None,
        })?;
        MlxArray::async_eval(&[&argmax, &drafts, &feats])?;
        let drafts = drafts.to_u32_vec()?;
        let argmax = argmax.to_u32_vec()?;
        stats.rounds += 1;
        stats.drafted += n;

        // Walk: accept drafts while they equal the target's choice.
        let mut kept = 1usize;
        for (j, &am) in argmax.iter().enumerate() {
            let t = target_token(&logits, j, am, guard, tokens, vocab)?;
            if eos_ids.contains(&t) {
                return Ok(FinishReason::Stop);
            }
            tokens.push(t);
            if tokens.len() >= max_tokens {
                return Ok(FinishReason::Length);
            }
            if j < n && t == drafts[j] {
                kept += 1;
                stats.accepted += 1;
                continue;
            }
            anchor = t;
            break;
        }
        decoder.rollback(caches, cap.delta.take().unwrap_or_default(), kept, n + 1)?;
        ctx = feats.slice(&[0, 0, 0], &[1, kept, feats.dim(2)?], &[1, 1, 1])?;
        if stats.rounds.is_multiple_of(16) {
            aimd_mlx::clear_cache()?;
        }
    }
}

/// One prompt of a [`batch_stream`]: spliced embeddings `[1, P, hidden]`
/// and its token budget.
pub struct BatchPrompt<'a> {
    pub embeds: &'a MlxArray,
    pub max_tokens: usize,
}

/// Additive bf16 mask `[B, 1, n, start + n]` for the prefill columns
/// `start..start + n` of a batch left-padded by `pads`: causal, and real
/// queries never see padding. A padding query sees the padding before it
/// (all-zero rows stay zero), so no mask row is empty.
fn prefill_mask(start: usize, n: usize, pads: &[usize]) -> Result<MlxArray, MlxError> {
    let lk = start + n;
    let key = MlxArray::from_i32(&[1, 1, 1, lk], &(0..lk as i32).collect::<Vec<_>>())?;
    let query = MlxArray::from_i32(
        &[1, 1, n, 1],
        &(start as i32..lk as i32).collect::<Vec<_>>(),
    )?;
    let pad: Vec<i32> = pads.iter().map(|&p| p as i32).collect();
    let pad = MlxArray::from_i32(&[pads.len(), 1, 1, 1], &pad)?;
    let zero = MlxArray::scalar_bf16(0.0)?;
    let neg = MlxArray::scalar_bf16(f32::NEG_INFINITY)?;
    let future = key.binary(&query, BinaryOp::Greater)?.where_(&neg, &zero)?;
    let pad_query = query.binary(&pad, BinaryOp::Less)?.where_(&zero, &neg)?;
    let hidden = key
        .binary(&pad, BinaryOp::Less)?
        .where_(&pad_query, &zero)?;
    future.add(&hidden)
}

/// Additive bf16 `[1, 1, n, old + n]`: query `j` of a block appended after
/// `old` columns does not see the block columns after it.
fn block_causal(old: usize, n: usize) -> Result<MlxArray, MlxError> {
    let lk = old + n;
    let key = MlxArray::from_i32(&[1, 1, 1, lk], &(0..lk as i32).collect::<Vec<_>>())?;
    let query = MlxArray::from_i32(&[1, 1, n, 1], &(old as i32..lk as i32).collect::<Vec<_>>())?;
    key.binary(&query, BinaryOp::Greater)?.where_(
        &MlxArray::scalar_bf16(f32::NEG_INFINITY)?,
        &MlxArray::scalar_bf16(0.0)?,
    )
}

/// A left-padded batch after the prefill of every row's `prompt[:-1]`.
/// Padding is all-zero embeddings, which stay zero through every layer
/// (no biases) and leave the DeltaNet state zero; masks keep real rows
/// from attending to it and RoPE positions start at each row's first real
/// column.
struct Prefilled {
    lens: Vec<usize>,
    pads: Vec<usize>,
    /// `[B, pmax, hidden]` padded embeddings.
    x: MlxArray,
    caches: Vec<LayerCache>,
    /// Captured target features of the prefilled columns, per chunk.
    features: Vec<MlxArray>,
}

fn prefill_batch(
    decoder: &MiniCpmDecoder,
    prompts: &[BatchPrompt<'_>],
    layers: Option<&[usize]>,
) -> Result<Prefilled, GenerateError> {
    let d = decoder.hidden_size();
    let b = prompts.len();
    let mut lens = Vec::with_capacity(b);
    for pr in prompts {
        let s = pr.embeds.shape()?;
        if s.len() != 3 || s[0] != 1 || s[2] != d {
            return Err(GenerateError::Layout(format!("embeds {s:?}")));
        }
        if s[1] == 0 {
            return Err(GenerateError::EmptyPrompt);
        }
        lens.push(s[1]);
    }
    let pmax = lens.iter().copied().max().unwrap_or(0);
    let pads: Vec<usize> = lens.iter().map(|&p| pmax - p).collect();
    let mut rows = Vec::with_capacity(b);
    for (pr, &pad) in prompts.iter().zip(&pads) {
        rows.push(if pad == 0 {
            pr.embeds.try_clone()?
        } else {
            let z = MlxArray::zeros(&[1, pad, d], pr.embeds.dtype()?)?;
            MlxArray::concatenate(&[&z, pr.embeds], 1)?
        });
    }
    let refs: Vec<&MlxArray> = rows.iter().collect();
    let x = MlxArray::concatenate(&refs, 0)?;
    drop(rows);

    aimd_mlx::clear_cache()?;
    let mut caches = decoder.make_cache();
    let mut chunks = Vec::new();
    // Keep the per-chunk token count near the single-prompt chunk.
    let step = (PREFILL_STEP_SIZE / b).max(256);
    let mut start = 0usize;
    while pmax - start > 1 {
        let n = (pmax - start - 1).min(step);
        let chunk = x.slice(&[0, start, 0], &[b, start + n, d], &[1, 1, 1])?;
        let starts: Vec<i64> = pads.iter().map(|&p| start as i64 - p as i64).collect();
        let mask = prefill_mask(start, n, &pads)?;
        let mut cap = layers.map(|l| Capture::new(l, false));
        decoder.forward_capture(
            &chunk,
            &decoder.rope_rows(&starts, n)?,
            Some(&mask),
            &mut caches,
            None,
            cap.as_mut(),
        )?;
        let mut state = Vec::new();
        if let Some(c) = cap {
            let f = features(c)?;
            state.push(f.try_clone()?);
            chunks.push(f);
        }
        for c in &caches {
            state.extend(c.state()?);
        }
        let refs: Vec<&MlxArray> = state.iter().collect();
        MlxArray::eval_many(&refs)?;
        start += n;
        aimd_mlx::clear_cache()?;
    }
    Ok(Prefilled {
        lens,
        pads,
        x,
        caches,
        features: chunks,
    })
}

/// Per-row bookkeeping of a batched run.
struct Rows {
    tokens: Vec<Vec<u32>>,
    finish: Vec<Option<FinishReason>>,
    ended: Vec<Duration>,
}

impl Rows {
    fn new(budgets: &[usize]) -> Self {
        let b = budgets.len();
        Self {
            tokens: vec![Vec::new(); b],
            finish: budgets
                .iter()
                .map(|&m| (m == 0).then_some(FinishReason::Length))
                .collect(),
            ended: vec![Duration::ZERO; b],
        }
    }

    /// Records sampled token `t` of row `i`; returns whether it continues.
    fn push(&mut self, i: usize, t: u32, eos_ids: &BTreeSet<u32>, budget: usize) -> bool {
        if eos_ids.contains(&t) {
            self.finish[i] = Some(FinishReason::Stop);
        } else {
            self.tokens[i].push(t);
            if self.tokens[i].len() >= budget {
                self.finish[i] = Some(FinishReason::Length);
            }
        }
        self.finish[i].is_none()
    }

    fn streams(self, prefill: Duration) -> Vec<GreedyStream> {
        self.tokens
            .into_iter()
            .zip(self.finish)
            .zip(self.ended)
            .map(|((tokens, finish), ended)| GreedyStream {
                tokens,
                finish_reason: finish.unwrap_or(FinishReason::Length),
                prefill,
                decode: ended.saturating_sub(prefill),
            })
            .collect()
    }
}

/// Greedy generation of several prompts as one left-padded batch
/// ([`prefill_batch`]): one decode step per token for every unfinished
/// row; rows leave the batch as they finish. Each row computes what
/// [`greedy_stream`] computes, up to the batched kernels' rounding.
fn greedy_batch(
    decoder: &MiniCpmDecoder,
    prompts: &[BatchPrompt<'_>],
    eos_ids: &BTreeSet<u32>,
    guard: Option<&SlidingWindowNoRepeatNgramProcessor>,
) -> Result<Vec<GreedyStream>, GenerateError> {
    let t0 = Instant::now();
    let Prefilled {
        lens,
        pads,
        x,
        mut caches,
        ..
    } = prefill_batch(decoder, prompts, None)?;
    let (b, d) = (lens.len(), decoder.hidden_size());
    let pmax = x.dim(1)?;
    let vocab = decoder.config().vocab_size;
    let budgets: Vec<usize> = prompts.iter().map(|p| p.max_tokens).collect();
    let max_t = budgets.iter().copied().max().unwrap_or(0).max(1);

    // Step `k` (0 = the last prompt column) sits at column `pmax - 1 + k`,
    // position `len - 1 + k` of each row.
    let starts: Vec<i64> = lens.iter().map(|&p| p as i64 - 1).collect();
    let mut table = decoder.rope_rows(&starts, max_t)?;
    let valid: Vec<Vec<bool>> = pads
        .iter()
        .map(|&p| (0..pmax + max_t).map(|c| c >= p).collect())
        .collect();
    let mut dmask = validity_mask(&valid, 0)?;
    drop(valid);
    let mut k = 0usize;
    let mut submit = |emb: &MlxArray,
                      caches: &mut [LayerCache],
                      table: &Rope,
                      dmask: &MlxArray,
                      banned: Option<&[Vec<u32>]>|
     -> Result<MlxArray, GenerateError> {
        let bc = emb.dim(0)?;
        let mask = dmask.slice(&[0, 0, 0, 0], &[bc, 1, 1, pmax + k], &[1, 1, 1, 1])?;
        let h = decoder.forward_capture(emb, &table.row(k)?, Some(&mask), caches, None, None)?;
        k += 1;
        let mut logits = decoder.logits(&h)?.reshape(&[bc, vocab])?;
        if let Some(banned) = banned.filter(|b| b.iter().any(|r| !r.is_empty())) {
            let mut parts = Vec::with_capacity(bc);
            for (r, ban) in banned.iter().enumerate() {
                let row = logits.slice(&[r, 0], &[r + 1, vocab], &[1, 1])?;
                parts.push(if ban.is_empty() {
                    row
                } else {
                    let idx = MlxArray::from_u32(&[1, ban.len()], ban)?;
                    let vals = MlxArray::scalar_bf16(f32::NEG_INFINITY)?
                        .astype(row.dtype()?)?
                        .broadcast_to(&[1, ban.len()])?;
                    row.put_along_axis(&idx, &vals, 1)?
                });
            }
            let refs: Vec<&MlxArray> = parts.iter().collect();
            logits = MlxArray::concatenate(&refs, 0)?;
        }
        let y = logits.astype(MlxDtype::F32)?.argmax(-1, false)?;
        MlxArray::async_eval(&[&y])?;
        Ok(y)
    };

    let last = x.slice(&[0, pmax - 1, 0], &[b, pmax, d], &[1, 1, 1])?;
    let mut y = submit(&last, &mut caches, &table, &dmask, None)?;
    drop(x);

    let mut rows = Rows::new(&budgets);
    let mut active: Vec<usize> = (0..b).collect();
    let mut prefill = None;
    let mut steps = 0usize;
    loop {
        // Submit step k+1 before reading token k unless a row's guard may
        // need token k to build its ban list or a row reaches its budget.
        let pipelined = active.iter().all(|&i| {
            rows.tokens[i].len() + 1 < budgets[i]
                && !guard.is_some_and(|g| may_ban(g, &rows.tokens[i]))
        });
        let next = if pipelined {
            let emb = decoder.embed_ids(&y.reshape(&[active.len(), 1])?)?;
            Some(submit(&emb, &mut caches, &table, &dmask, None)?)
        } else {
            None
        };
        let toks = y.to_u32_vec()?;
        prefill.get_or_insert_with(|| t0.elapsed());
        let mut keep = Vec::with_capacity(active.len());
        for (j, &i) in active.iter().enumerate() {
            if rows.finish[i].is_none() && rows.push(i, toks[j], eos_ids, budgets[i]) {
                keep.push(j);
            } else {
                rows.ended[i] = t0.elapsed();
            }
        }
        if keep.is_empty() {
            break;
        }
        let mut y_next = next;
        if keep.len() < active.len() {
            let idx: Vec<u32> = keep.iter().map(|&j| j as u32).collect();
            let idx = MlxArray::from_u32(&[idx.len()], &idx)?;
            for c in caches.iter_mut() {
                c.select_rows(&idx)?;
            }
            table = table.select_rows(&idx)?;
            dmask = dmask.take(&idx, 0)?;
            y = y.take(&idx, 0)?;
            y_next = y_next.map(|n| n.take(&idx, 0)).transpose()?;
            active = keep.iter().map(|&j| active[j]).collect();
        }
        y = match y_next {
            Some(n) => n,
            None => {
                let banned: Vec<Vec<u32>> = active
                    .iter()
                    .map(|&i| {
                        guard
                            .filter(|g| may_ban(g, &rows.tokens[i]))
                            .map(|g| g.banned(&rows.tokens[i], vocab))
                            .unwrap_or_default()
                    })
                    .collect();
                let emb = decoder.embed_ids(&y.reshape(&[active.len(), 1])?)?;
                submit(&emb, &mut caches, &table, &dmask, Some(&banned))?
            }
        };
        steps += 1;
        if steps.is_multiple_of(256) {
            aimd_mlx::clear_cache()?;
        }
    }
    Ok(rows.streams(prefill.unwrap_or_else(|| t0.elapsed())))
}

/// [`speculative_stream`] over several prompts as one left-padded batch
/// ([`prefill_batch`]). Every round drafts for all unfinished rows at once
/// ([`DSparkDrafter::draft_batch`]) and verifies `[anchor, drafts]` of
/// every row in one target forward. Rows accept different prefixes: the
/// caches keep the longest ([`MiniCpmDecoder::rollback_rows`]) and the
/// rejected positions of the other rows stay as masked holes in the target
/// and drafter caches; each row's RoPE positions count only its own
/// tokens. Emitted tokens are each row's target greedy choices, as in
/// [`speculative_stream`].
fn speculative_batch(
    decoder: &MiniCpmDecoder,
    drafter: &DSparkDrafter,
    prompts: &[BatchPrompt<'_>],
    eos_ids: &BTreeSet<u32>,
    guard: Option<&SlidingWindowNoRepeatNgramProcessor>,
) -> Result<Vec<(GreedyStream, SpecStats)>, GenerateError> {
    let t0 = Instant::now();
    let layers = drafter.config().target_layer_ids();
    let Prefilled {
        lens,
        pads,
        x,
        mut caches,
        features: mut chunks,
    } = prefill_batch(decoder, prompts, Some(layers))?;
    let (b, d) = (lens.len(), decoder.hidden_size());
    let pmax = x.dim(1)?;
    let vocab = decoder.config().vocab_size;
    let n = drafter.config().block_size;
    let budgets: Vec<usize> = prompts.iter().map(|p| p.max_tokens).collect();

    // The last prompt column, then the drafter context: each row's
    // features moved to the front (`fresh` = its prompt length).
    let last = x.slice(&[0, pmax - 1, 0], &[b, pmax, d], &[1, 1, 1])?;
    drop(x);
    let starts: Vec<i64> = lens.iter().map(|&p| p as i64 - 1).collect();
    let mut cap = Capture::new(layers, false);
    let h = decoder.forward_capture(
        &last,
        &decoder.rope_rows(&starts, 1)?,
        Some(&prefill_mask(pmax - 1, 1, &pads)?),
        &mut caches,
        None,
        Some(&mut cap),
    )?;
    chunks.push(features(cap)?);
    let refs: Vec<&MlxArray> = chunks.iter().collect();
    let all = MlxArray::concatenate(&refs, 1)?;
    drop(chunks);
    let f = all.dim(2)?;
    let mut rows_ctx = Vec::with_capacity(b);
    for (i, &pad) in pads.iter().enumerate() {
        let row = all.slice(&[i, pad, 0], &[i + 1, pmax, f], &[1, 1, 1])?;
        rows_ctx.push(if pad == 0 {
            row
        } else {
            MlxArray::concatenate(&[&row, &MlxArray::zeros(&[1, pad, f], row.dtype()?)?], 1)?
        });
    }
    let refs: Vec<&MlxArray> = rows_ctx.iter().collect();
    let mut ctx = MlxArray::concatenate(&refs, 0)?;
    drop(rows_ctx);
    drop(all);
    let anchors = decoder
        .logits(&h)?
        .reshape(&[b, vocab])?
        .astype(MlxDtype::F32)?
        .argmax(-1, false)?
        .to_u32_vec()?;
    let prefill = t0.elapsed();

    let mut rows = Rows::new(&budgets);
    let mut stats = vec![SpecStats::default(); b];
    let mut valid: Vec<Vec<bool>> = pads
        .iter()
        .map(|&p| (0..pmax).map(|c| c >= p).collect())
        .collect();
    let mut fresh = lens.clone();
    let mut draft = DraftBatch::new(drafter, b);
    let mut active: Vec<usize> = (0..b).collect();
    let mut keep: Vec<usize> = Vec::with_capacity(b);
    for (j, &i) in active.iter().enumerate() {
        if rows.finish[i].is_none() && rows.push(i, anchors[i], eos_ids, budgets[i]) {
            keep.push(j);
        } else {
            rows.ended[i] = t0.elapsed();
        }
    }
    let mut round = 0usize;
    loop {
        if keep.len() < active.len() {
            if keep.is_empty() {
                break;
            }
            let idx: Vec<u32> = keep.iter().map(|&j| j as u32).collect();
            let idx = MlxArray::from_u32(&[idx.len()], &idx)?;
            for c in caches.iter_mut() {
                c.select_rows(&idx)?;
            }
            draft.select_rows(&keep, &idx)?;
            ctx = ctx.take(&idx, 0)?;
            valid = keep
                .iter()
                .map(|&j| std::mem::take(&mut valid[j]))
                .collect();
            fresh = keep.iter().map(|&j| fresh[j]).collect();
            active = keep.iter().map(|&j| active[j]).collect();
        }
        let bc = active.len();
        let anc: Vec<u32> = active
            .iter()
            .map(|&i| *rows.tokens[i].last().expect("an anchor"))
            .collect();
        let drafts = drafter.draft_batch(decoder, &anc, &ctx, &fresh, &mut draft)?;
        let ids = MlxArray::concatenate(&[&MlxArray::from_u32(&[bc, 1], &anc)?, &drafts], 1)?;
        let xb = decoder.embed_ids(&ids)?;
        let pos: Vec<i64> = active
            .iter()
            .map(|&i| (lens[i] + rows.tokens[i].len() - 1) as i64)
            .collect();
        let old = valid[0].len();
        let mask = validity_mask(&valid, n + 1)?.add(&block_causal(old, n + 1)?)?;
        let mut cap = Capture::new(layers, true);
        let h = decoder.forward_capture(
            &xb,
            &decoder.rope_rows(&pos, n + 1)?,
            Some(&mask),
            &mut caches,
            None,
            Some(&mut cap),
        )?;
        let logits = decoder.logits(&h)?.reshape(&[bc * (n + 1), vocab])?;
        let argmax = logits.astype(MlxDtype::F32)?.argmax(-1, false)?;
        let feats = features(Capture {
            layers,
            hidden: std::mem::take(&mut cap.hidden),
            delta: None,
        })?;
        MlxArray::async_eval(&[&argmax, &drafts, &feats])?;
        let drafts = drafts.to_u32_vec()?;
        let argmax = argmax.to_u32_vec()?;
        round += 1;

        // Walk each row: accept drafts while they equal its target choice.
        let mut kept = vec![1usize; bc];
        keep.clear();
        for (j, &i) in active.iter().enumerate() {
            stats[i].rounds += 1;
            stats[i].drafted += n;
            let mut alive = true;
            for t in 0..=n {
                let r = j * (n + 1) + t;
                let tok = target_token(&logits, r, argmax[r], guard, &rows.tokens[i], vocab)?;
                if !rows.push(i, tok, eos_ids, budgets[i]) {
                    alive = false;
                    break;
                }
                if t < n && tok == drafts[j * n + t] {
                    kept[j] += 1;
                    stats[i].accepted += 1;
                    continue;
                }
                break;
            }
            if alive {
                keep.push(j);
            } else {
                rows.ended[i] = t0.elapsed();
            }
        }
        if keep.is_empty() {
            break;
        }
        decoder.rollback_rows(
            &mut caches,
            cap.delta.take().unwrap_or_default(),
            &kept,
            n + 1,
        )?;
        let kmax = kept.iter().copied().max().unwrap_or(1);
        for (v, &k) in valid.iter_mut().zip(&kept) {
            v.extend((0..kmax).map(|t| t < k));
        }
        ctx = feats.slice(&[0, 0, 0], &[bc, kmax, feats.dim(2)?], &[1, 1, 1])?;
        fresh = kept;
        if round.is_multiple_of(16) {
            aimd_mlx::clear_cache()?;
        }
    }
    Ok(rows.streams(prefill).into_iter().zip(stats).collect())
}

/// Generation of several prompts in one batch: [`speculative_batch`] with
/// a drafter, else [`greedy_batch`]. A single prompt runs
/// [`speculative_stream`] / [`greedy_stream`].
pub fn batch_stream(
    decoder: &MiniCpmDecoder,
    drafter: Option<&DSparkDrafter>,
    prompts: &[BatchPrompt<'_>],
    eos_ids: &BTreeSet<u32>,
    ngram: Option<usize>,
) -> Result<Vec<(GreedyStream, Option<SpecStats>)>, GenerateError> {
    let out = match (prompts, drafter) {
        ([], _) => Vec::new(),
        ([one], Some(dr)) => {
            let (s, st) =
                speculative_stream(decoder, dr, one.embeds, eos_ids, one.max_tokens, ngram)?;
            vec![(s, Some(st))]
        }
        ([one], None) => vec![(
            greedy_stream(decoder, one.embeds, eos_ids, one.max_tokens, ngram, None)?,
            None,
        )],
        (_, Some(dr)) => {
            speculative_batch(decoder, dr, prompts, eos_ids, ngram_guard(ngram)?.as_ref())?
                .into_iter()
                .map(|(s, st)| (s, Some(st)))
                .collect()
        }
        (_, None) => greedy_batch(decoder, prompts, eos_ids, ngram_guard(ngram)?.as_ref())?
            .into_iter()
            .map(|s| (s, None))
            .collect(),
    };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minicpm_ocr_may_ban_is_exact_for_the_next_step() {
        let g = SlidingWindowNoRepeatNgramProcessor::new(4, usize::MAX).unwrap();
        // Brute force: a ban is possible iff some next token gets banned.
        let streams: [&[u32]; 5] = [
            &[1, 2, 3],
            &[1, 2, 3, 1, 2],
            &[1, 2, 3, 4, 2, 3],
            &[5, 5, 5, 5],
            &[1, 2, 1, 3, 1],
        ];
        for s in streams {
            let any = (0..8u32).any(|t| {
                let mut next = s.to_vec();
                next.push(t);
                !g.banned(&next, 8).is_empty()
            });
            assert_eq!(may_ban(&g, s), any, "{s:?}");
        }
    }
}
