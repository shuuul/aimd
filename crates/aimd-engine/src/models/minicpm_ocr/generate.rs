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
use std::time::Instant;

use aimd_mlx::{MlxArray, MlxDtype, MlxError};

use super::decoder::{Capture, DecoderError, LayerCache, MiniCpmDecoder, Rope};
use super::dspark::DSparkDrafter;
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
    let guard = match ngram {
        Some(n) => Some(
            SlidingWindowNoRepeatNgramProcessor::new(n, usize::MAX)
                .map_err(|e| GenerateError::Layout(e.to_string()))?,
        ),
        None => None,
    };
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
    let guard = match ngram {
        Some(n) => Some(
            SlidingWindowNoRepeatNgramProcessor::new(n, usize::MAX)
                .map_err(|e| GenerateError::Layout(e.to_string()))?,
        ),
        None => None,
    };
    let layers = drafter.config().target_layer_ids();
    let vocab = decoder.config().vocab_size;
    let n = drafter.config().block_size;
    let t0 = Instant::now();
    let mut caches = decoder.make_cache();
    let mut draft_cache = drafter.make_cache();
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
        &mut caches,
        None,
        Some(&mut cap),
    )?;
    context.push(features(cap)?);
    let refs: Vec<&MlxArray> = context.iter().collect();
    let mut ctx = MlxArray::concatenate(&refs, 1)?;
    drop(context);
    let mut tokens: Vec<u32> = Vec::with_capacity(max_tokens.min(1024));
    let mut finish_reason = FinishReason::Length;
    let logits = decoder.logits(&h)?.reshape(&[1, vocab])?;
    let mut anchor = sample(&logits, &[], None)?.to_u32_vec()?[0];
    let prefill = t0.elapsed();

    let mut done = max_tokens == 0;
    if !done && eos_ids.contains(&anchor) {
        finish_reason = FinishReason::Stop;
        done = true;
    }
    if !done {
        tokens.push(anchor);
        done = tokens.len() >= max_tokens;
    }
    while !done {
        let drafts = drafter.draft(decoder, anchor, &ctx, &mut draft_cache)?;
        let ids = MlxArray::concatenate(&[&MlxArray::from_u32(&[1], &[anchor])?, &drafts], 0)?;
        let x = decoder.embed_ids(&ids.reshape(&[1, n + 1])?)?;
        let pos = p + tokens.len() - 1;
        let mut cap = Capture::new(layers, true);
        let h = decoder.forward_capture(
            &x,
            &decoder.rope(pos, n + 1)?,
            &mut caches,
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
            let t = target_token(&logits, j, am, guard.as_ref(), &tokens, vocab)?;
            if eos_ids.contains(&t) {
                finish_reason = FinishReason::Stop;
                done = true;
                break;
            }
            tokens.push(t);
            if tokens.len() >= max_tokens {
                done = true;
                break;
            }
            if j < n && t == drafts[j] {
                kept += 1;
                stats.accepted += 1;
                continue;
            }
            anchor = t;
            break;
        }
        if done {
            break;
        }
        decoder.rollback(
            &mut caches,
            cap.delta.take().unwrap_or_default(),
            kept,
            n + 1,
        )?;
        ctx = feats.slice(&[0, 0, 0], &[1, kept, feats.dim(2)?], &[1, 1, 1])?;
        if stats.rounds.is_multiple_of(16) {
            aimd_mlx::clear_cache()?;
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
