//! Greedy generation for PaddleOCR-VL, mirroring mlx-vlm
//! `generate.ar.generate_step` at temperature 0:
//!
//! - chunked prefill (2048) over `prompt[:-1]` with the cached
//!   `get_rope_index` positions, evaluating only the KV caches, then
//!   `mx.clear_cache()`;
//! - the last prompt token as the first `_step`, at position
//!   `cache_offset + rope_delta` on all three mRoPE axes (as every later
//!   step);
//! - `logprobs = logits - logsumexp(logits)` (bf16) and `argmax`;
//! - `async_eval` pipelining (the next forward is enqueued before the
//!   current token is read back) and `clear_cache` every 256 tokens;
//! - EOS stops generation and is not part of the returned stream.

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use aimd_mlx::{MlxArray, MlxError};

use super::decoder::{DecoderError, PaddleDecoder};
use super::preprocess::RopeIndex;

/// `generate_step(prefill_step_size=2048)`.
pub const PREFILL_STEP_SIZE: usize = 2048;

/// Default `max_tokens` for a page (recorded in the spec; mlx-vlm's CLI
/// default of 256 truncates dense pages).
pub const DEFAULT_MAX_TOKENS: usize = 4096;

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

/// Why generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// An EOS token was produced.
    Stop,
    /// `max_tokens` reached.
    Length,
}

/// Greedy result: generated ids (EOS excluded) and timings.
#[derive(Clone, Debug)]
pub struct GreedyStream {
    pub tokens: Vec<u32>,
    pub finish_reason: FinishReason,
    /// Prompt processing up to the first sampled token.
    pub prefill: Duration,
    /// Everything after the first token.
    pub decode: Duration,
}

fn sample(logits: &MlxArray, capture: Option<&mut Vec<MlxArray>>) -> Result<MlxArray, MlxError> {
    let logprobs = logits.subtract(&logits.logsumexp(-1, true)?)?;
    let y = logprobs.argmax(-1, false)?;
    if let Some(c) = capture {
        c.push(logprobs);
    }
    Ok(y)
}

/// Runs greedy decoding over prompt embeddings `embeds` (`[1, P, hidden]`,
/// image features already spliced) with the prompt's mRoPE index. With
/// `capture`, the bf16 `[1, vocab]` logprobs of every sampled step are
/// pushed (first step = the last prompt token).
pub fn greedy_stream(
    decoder: &PaddleDecoder,
    embeds: &MlxArray,
    rope: &RopeIndex,
    eos_ids: &BTreeSet<u32>,
    max_tokens: usize,
    mut capture: Option<&mut Vec<MlxArray>>,
) -> Result<GreedyStream, GenerateError> {
    let p = rope.positions[0].len();
    if p == 0 {
        return Err(GenerateError::EmptyPrompt);
    }
    let d = decoder.hidden_size();
    if embeds.shape()? != [1, p, d] {
        return Err(GenerateError::Layout(format!(
            "embeds {:?}, expected [1, {p}, {d}]",
            embeds.shape()?
        )));
    }
    let vocab = decoder.config().vocab_size;
    let t0 = Instant::now();
    let mut caches = decoder.make_cache();

    let mut start = 0usize;
    while p - start > 1 {
        let n = (p - start - 1).min(PREFILL_STEP_SIZE);
        let chunk = embeds.slice(&[0, start, 0], &[1, start + n, d], &[1, 1, 1])?;
        let pos = |axis: usize| &rope.positions[axis][start..start + n];
        let r = decoder.rope([pos(0), pos(1), pos(2)])?;
        decoder.forward(&chunk, &r, &mut caches)?;
        let mut state = Vec::with_capacity(caches.len() * 2);
        for c in &caches {
            state.extend(c.state()?);
        }
        let refs: Vec<&MlxArray> = state.iter().collect();
        MlxArray::eval_many(&refs)?;
        start += n;
        aimd_mlx::clear_cache()?;
    }

    // Step `k` (0 = the last prompt token) sits at `p - 1 + k + delta` on
    // all three axes: one table for the whole generation, sliced per step.
    let table = decoder.rope_sequential((p - 1) as i64 + rope.delta, max_tokens.max(1))?;
    let step = |emb: &MlxArray,
                caches: &mut [crate::models::kv_cache::KvCache]|
     -> Result<MlxArray, GenerateError> {
        let r = table.row(caches[0].offset() + 1 - p)?;
        let h = decoder.forward(emb, &r, caches)?;
        Ok(decoder.logits(&h)?.reshape(&[1, vocab])?)
    };

    let last = embeds.slice(&[0, p - 1, 0], &[1, p, d], &[1, 1, 1])?;
    let logits = step(&last, &mut caches)?;
    let mut y = sample(&logits, capture.as_deref_mut())?;
    MlxArray::async_eval(&[&y])?;

    let mut tokens: Vec<u32> = Vec::with_capacity(max_tokens.min(1024));
    let mut finish_reason = FinishReason::Length;
    let mut prefill = None;
    if max_tokens > 0 {
        loop {
            // mlx-lm `generate_step`: submit step k+1 (it depends on `y`
            // lazily) before reading token k, so the GPU never waits on the
            // host's graph encode.
            let next = if tokens.len() + 1 < max_tokens {
                let emb = decoder.embed_ids(&y.reshape(&[1, 1])?)?;
                let logits = step(&emb, &mut caches)?;
                let mut logprobs = capture.is_some().then(Vec::new);
                let next_y = sample(&logits, logprobs.as_mut())?;
                MlxArray::async_eval(&[&next_y])?;
                Some((next_y, logprobs))
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
            let Some((next_y, logprobs)) = next else {
                break;
            };
            if let (Some(c), Some(l)) = (capture.as_deref_mut(), logprobs) {
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
