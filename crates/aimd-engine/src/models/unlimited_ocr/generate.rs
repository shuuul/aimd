//! Unlimited-OCR greedy generation, the sliding-window no-repeat n-gram
//! guard, and the markdown normalization port (WS-07b).
//!
//! Ports of the Python behavior contracts:
//!
//! - **Greedy loop** mirrors mlx-vlm `generate_step` + `stream_generate`
//!   (`.venv/.../mlx_vlm/generate/ar.py` and `.../dispatch.py`) as driven by
//!   `src/aimd/plugins/ocr/models/mlx.py::_generate_unlimited_ocr_text`:
//!   temperature 0 (argmax), `max_tokens = 8192`, EOS = the config's
//!   `eos_token_id` (1, `<｜end▁of▁sentence｜>`). An EOS sample terminates
//!   generation and is dropped from the output (`finish = stop`); hitting
//!   the budget yields `finish = length`. The runtime always takes the
//!   chunked-prefill path, where the logits processors see the token stream
//!   `[last prompt token] ++ generated` — the leading prompt token is
//!   reproduced here (`guard_stream`); with `n = 35` it can only matter
//!   while the stream is shorter than 35 tokens, where the guard is a
//!   no-op anyway.
//! - [`SlidingWindowNoRepeatNgramProcessor`] is a line-by-line port of
//!   `SlidingWindowNoRepeatNgramProcessor` in
//!   `src/aimd/plugins/ocr/models/mlx.py` (itself a port of Baidu's
//!   Unlimited-OCR processor for mlx-vlm's `logits_processors` hook). It
//!   bans every token that would complete a repeated n-gram whose start
//!   lies in the last `window` tokens of the generated stream. Test
//!   expectations are locked against the Python processor (see the test
//!   module notes).
//! - [`normalize_unlimited_ocr_markdown`] is an exact port of
//!   `src/aimd/plugins/ocr/models/unlimited.py::normalize_unlimited_ocr_markdown`
//!   (regex-free): it strips the `<|det|>` layout protocol into blank-line
//!   separated blocks, drops `image` categories, and passes text without
//!   det markers through `str.strip()`-ed.

use std::collections::BTreeSet;
use std::fmt;

use std::time::Duration;
#[cfg(feature = "mlx")]
use std::time::Instant;

#[cfg(feature = "mlx")]
use aimd_mlx::{MlxArray, MlxDtype, MlxError};

pub use super::markdown::{
    LayoutBlock, UNLIMITED_OCR_GRID, normalize_unlimited_ocr_markdown, parse_unlimited_ocr_blocks,
};
#[cfg(feature = "mlx")]
use super::mlx_decoder::{DecoderError, OcrDecoder};
use crate::tokenizer::{Tokenizer, TokenizerError};

/// Generation budget (`UNLIMITED_OCR_MAX_TOKENS` in the Python adapter).
pub const UNLIMITED_OCR_MAX_TOKENS: usize = 8192;

/// N-gram size of the sliding-window guard (`UNLIMITED_OCR_NGRAM_SIZE`).
pub const UNLIMITED_OCR_NGRAM_SIZE: usize = 35;

/// Search window of the sliding-window guard (`UNLIMITED_OCR_NGRAM_WINDOW`).
pub const UNLIMITED_OCR_NGRAM_WINDOW: usize = 128;

/// Unlimited-OCR-2 guard window for multi-page requests (upstream
/// `infer.py`: 1024 for several pages, 128 for one).
pub const UNLIMITED_OCR2_MULTI_PAGE_NGRAM_WINDOW: usize = 1024;

/// Per-page token cap of the Unlimited-OCR-2 page-count guard (upstream
/// `infer(max_page_tokens=20000)`).
pub const UNLIMITED_OCR2_MAX_PAGE_TOKENS: usize = 20000;

/// The `<image>` token id (128815) replaced by vision features.
pub const OCR_IMAGE_TOKEN_ID: u32 = 128815;

/// Errors surfaced by the generation loop.
#[derive(Debug, thiserror::Error)]
pub enum GenerateError {
    #[cfg(feature = "mlx")]
    #[error(transparent)]
    Decoder(#[from] DecoderError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[cfg(feature = "mlx")]
    #[error(transparent)]
    Mlx(#[from] MlxError),
    #[error("prompt token ids are empty")]
    EmptyPrompt,
    #[error("invalid generation settings: {0}")]
    InvalidSettings(String),
    #[error("layout: {0}")]
    Layout(String),
}

/// Knobs for the greedy OCR generation loop.
#[derive(Clone, Copy, Debug)]
pub struct OcrGenerationSettings {
    /// Hard stop on sampled tokens (mlx-vlm `max_tokens`). Default 8192.
    pub max_new_tokens: usize,
    /// Sliding-window guard size `n` (0 disables the guard). Default 35.
    pub ngram_size: usize,
    /// Sliding-window guard search window. Default 128.
    pub ngram_window: usize,
    /// KV ring window override for tests (`None` = config `sliding_window`).
    pub kv_window: Option<usize>,
    /// Page-count guard: the number of prompt pages the output must mark
    /// with `<PAGE>` (0 disables it).
    pub expected_pages: usize,
    /// Page-count guard: tokens after which a page is closed (the next
    /// `<PAGE>`, or EOS after the last page).
    pub max_page_tokens: usize,
}

impl Default for OcrGenerationSettings {
    fn default() -> Self {
        Self {
            max_new_tokens: UNLIMITED_OCR_MAX_TOKENS,
            ngram_size: UNLIMITED_OCR_NGRAM_SIZE,
            ngram_window: UNLIMITED_OCR_NGRAM_WINDOW,
            kv_window: None,
            expected_pages: 0,
            max_page_tokens: UNLIMITED_OCR2_MAX_PAGE_TOKENS,
        }
    }
}

/// Why generation stopped (mlx `finish_reason`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// An EOS token was sampled (and dropped from the output).
    Stop,
    /// `max_new_tokens` non-EOS tokens were sampled.
    Length,
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        })
    }
}

/// Result of one greedy generation.
#[derive(Clone, Debug)]
pub struct OcrGeneration {
    /// Generated token ids (the EOS token excluded, matching mlx
    /// `stream_generate`, which breaks before detokenizing the EOS).
    pub tokens: Vec<u32>,
    /// Decoded text of [`OcrGeneration::tokens`] (special tokens kept —
    /// the `<|det|>` markers are special tokens the normalizer needs).
    pub text: String,
    pub finish_reason: FinishReason,
}

/// Resolves the EOS id set exactly like mlx-vlm `generate`:
/// `stopping_criteria.reset(model.config.eos_token_id)` — the checkpoint
/// config's `eos_token_id` (1). The tokenizer's own EOS special id is used
/// when available (they coincide), with the config default 1 as fallback.
pub fn resolve_ocr_eos_ids(tokenizer: &Tokenizer) -> BTreeSet<u32> {
    let mut ids = BTreeSet::new();
    if let Ok(id) = tokenizer.token_id("<｜end▁of▁sentence｜>") {
        ids.insert(id);
    }
    if ids.is_empty() {
        ids.insert(1);
    }
    ids
}

/// Block n-gram repetitions within a sliding window of the generated stream.
///
/// Port of `SlidingWindowNoRepeatNgramProcessor` in
/// `src/aimd/plugins/ocr/models/mlx.py` (Baidu Unlimited-OCR, wired as an
/// mlx-vlm `logits_processors` hook): given the token stream so far and the
/// next-step logits, every token that would complete an n-gram whose first
/// `n-1` tokens equal the stream's last `n-1` tokens AND whose start index
/// lies within the last `window` tokens is set to `-inf`.
#[derive(Clone, Debug)]
pub struct SlidingWindowNoRepeatNgramProcessor {
    ngram_size: usize,
    window: usize,
    whitelist: BTreeSet<u32>,
}

impl SlidingWindowNoRepeatNgramProcessor {
    /// Creates the processor (`whitelist_token_ids` is empty in aimd's
    /// adapter but ported for parity with the Python constructor).
    pub fn new(ngram_size: usize, window: usize) -> Result<Self, GenerateError> {
        Self::with_whitelist(ngram_size, window, &[])
    }

    /// [`Self::new`] with an explicit whitelist (removed from every ban set).
    pub fn with_whitelist(
        ngram_size: usize,
        window: usize,
        whitelist: &[u32],
    ) -> Result<Self, GenerateError> {
        if ngram_size < 1 {
            return Err(GenerateError::InvalidSettings(
                "ngram_size must be >= 1".to_string(),
            ));
        }
        if window < 1 {
            return Err(GenerateError::InvalidSettings(
                "window must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            ngram_size,
            window,
            whitelist: whitelist.iter().copied().collect(),
        })
    }

    pub fn ngram_size(&self) -> usize {
        self.ngram_size
    }

    pub fn window(&self) -> usize {
        self.window
    }

    /// Token ids the Python hook sets to `-inf` for `stream` (the token
    /// stream it receives: `[last prompt token] ++ generated`), restricted
    /// to `0 <= id < vocab`, ascending.
    pub fn banned(&self, stream: &[u32], vocab: usize) -> Vec<u32> {
        let n = self.ngram_size;
        if stream.len() < n {
            return Vec::new();
        }
        let search_start = stream.len().saturating_sub(self.window);
        let search_end = stream.len() + 1 - n;
        if search_end <= search_start {
            return Vec::new();
        }
        let prefix: &[u32] = if n > 1 {
            &stream[stream.len() - (n - 1)..]
        } else {
            &[]
        };
        let mut banned: BTreeSet<u32> = BTreeSet::new();
        for idx in search_start..search_end {
            // n <= stream.len() so idx + n never exceeds the stream.
            let ngram_prefix = &stream[idx..idx + n - 1];
            if n == 1 || ngram_prefix == prefix {
                banned.insert(stream[idx + n - 1]);
            }
        }
        for id in &self.whitelist {
            banned.remove(id);
        }
        banned
            .into_iter()
            .filter(|&id| (id as usize) < vocab)
            .collect()
    }

    /// Applies the ban list to a host f32 logits row in place (mlx writes
    /// `-inf` into the banned positions).
    pub fn apply(&self, stream: &[u32], logits: &mut [f32]) {
        for id in self.banned(stream, logits.len()) {
            logits[id as usize] = f32::NEG_INFINITY;
        }
    }
}

/// Unlimited-OCR-2 page-count guard, a port of upstream
/// `_uocr_page_count_guard` (vLLM `_PageCountGuard`) over the generated
/// tokens: while fewer than `expected` `<PAGE>` markers were generated EOS
/// is banned; past `expected` markers only EOS remains; a page longer than
/// `max_page_tokens` is closed (forced `<PAGE>`, or EOS on the last page).
#[derive(Clone, Debug)]
pub struct PageCountGuard {
    expected: usize,
    /// The marker's token ids: `[<PAGE>]` for Unlimited-OCR-2,
    /// [`UNLIMITED_OCR_V1_PAGE_MARKER`] (plain text) for v1.
    marker: Vec<u32>,
    eos_id: u32,
    max_page_tokens: usize,
    markers: usize,
    page_len: usize,
    /// The last `marker.len()` generated tokens.
    tail: Vec<u32>,
}

/// What [`PageCountGuard::constraint`] imposes on the next sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageConstraint {
    None,
    BanEos,
    Force(u32),
}

impl PageCountGuard {
    pub fn new(expected: usize, marker: Vec<u32>, eos_id: u32, max_page_tokens: usize) -> Self {
        assert!(!marker.is_empty(), "empty page marker");
        Self {
            expected,
            tail: Vec::with_capacity(marker.len()),
            marker,
            eos_id,
            max_page_tokens,
            markers: 0,
            page_len: 0,
        }
    }

    /// Records one generated token.
    pub fn push(&mut self, tok: u32) {
        if self.tail.len() == self.marker.len() {
            self.tail.remove(0);
        }
        self.tail.push(tok);
        if self.tail == self.marker {
            self.markers += 1;
            self.page_len = 0;
            self.tail.clear();
        } else {
            self.page_len += 1;
        }
    }

    /// Marker tokens already at the end of the stream (a forced marker in
    /// progress).
    fn partial(&self) -> usize {
        (1..self.marker.len())
            .rev()
            .find(|&j| self.tail.ends_with(&self.marker[..j]))
            .unwrap_or(0)
    }

    /// The constraint on the next token.
    pub fn constraint(&self) -> PageConstraint {
        if self.markers > self.expected {
            PageConstraint::Force(self.eos_id)
        } else if self.max_page_tokens > 0 && self.page_len >= self.max_page_tokens {
            PageConstraint::Force(if self.markers < self.expected {
                self.marker[self.partial()]
            } else {
                self.eos_id
            })
        } else if self.markers < self.expected {
            PageConstraint::BanEos
        } else {
            PageConstraint::None
        }
    }
}

/// v1 `<PAGE>` page separator: plain text (`<`, `PAGE`, `>`), not a
/// special token.
pub const UNLIMITED_OCR_V1_PAGE_MARKER: [u32; 3] = [100855, 16412, 32];

/// mlx-vlm `DEFAULT_PREFILL_STEP_SIZE`.
pub const PREFILL_STEP_SIZE: usize = 2048;

/// Wall-clock split of one generation (prefill = up to the first sampled
/// token's readback, decode = the rest).
#[derive(Clone, Copy, Debug, Default)]
pub struct GenerationTimings {
    pub prefill: Duration,
    pub decode: Duration,
}

/// Token-level result of one greedy generation.
#[derive(Clone, Debug)]
pub struct TokenStream {
    /// Generated token ids (the EOS token excluded, matching mlx
    /// `stream_generate`).
    pub tokens: Vec<u32>,
    pub finish_reason: FinishReason,
}

/// Splices `features` (`[n, hidden]` bf16) over the prompt's `<image>`
/// tokens: `embed_tokens(ids)` with `input_embeds[0, image_positions] =
/// features` (`get_input_embeddings`). The prompt must hold exactly `n`
/// image tokens, in one run (v1 gundam) or one run per page (v2
/// `<PAGE><image>...`); feature rows fill the runs in order.
#[cfg(feature = "mlx")]
pub fn splice_image_features(
    decoder: &OcrDecoder,
    prompt_ids: &[u32],
    features: &MlxArray,
    image_token_id: u32,
) -> Result<MlxArray, GenerateError> {
    if prompt_ids.is_empty() {
        return Err(GenerateError::EmptyPrompt);
    }
    let d = decoder.hidden_size();
    let fs = features.shape()?;
    if fs.len() != 2 || fs[1] != d {
        return Err(GenerateError::Layout(format!(
            "image features {fs:?}, expected [n, {d}]"
        )));
    }
    let n = fs[0];
    let count = prompt_ids.iter().filter(|&&t| t == image_token_id).count();
    let embeds = decoder.embed_ids(&decoder.ids_array(prompt_ids)?)?;
    if count != n {
        return Err(GenerateError::Layout(format!(
            "prompt has {count} <image> tokens for {n} feature rows"
        )));
    }
    if n == 0 {
        return Ok(embeds);
    }
    let feats = features.astype(MlxDtype::Bf16)?.reshape(&[1, n, d])?;
    let mut parts = Vec::new();
    let (mut text, mut used, mut i) = (0, 0, 0);
    while i < prompt_ids.len() {
        if prompt_ids[i] != image_token_id {
            i += 1;
            continue;
        }
        let run = prompt_ids[i..]
            .iter()
            .take_while(|&&t| t == image_token_id)
            .count();
        if i > text {
            parts.push(embeds.slice(&[0, text, 0], &[1, i, d], &[1, 1, 1])?);
        }
        parts.push(feats.slice(&[0, used, 0], &[1, used + run, d], &[1, 1, 1])?);
        used += run;
        i += run;
        text = i;
    }
    if text < prompt_ids.len() {
        parts.push(embeds.slice(&[0, text, 0], &[1, prompt_ids.len(), d], &[1, 1, 1])?);
    }
    let refs: Vec<&MlxArray> = parts.iter().collect();
    Ok(MlxArray::concatenate(&refs, 1)?)
}

/// Greedy OCR generation over spliced prompt embeddings (tokenizer
/// included: decodes the generated tokens, special tokens kept). EOS ids
/// resolve via [`resolve_ocr_eos_ids`].
#[cfg(feature = "mlx")]
pub fn greedy_ocr_generate(
    decoder: &OcrDecoder,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    embeds: &MlxArray,
    settings: OcrGenerationSettings,
) -> Result<(OcrGeneration, GenerationTimings), GenerateError> {
    let eos_ids = resolve_ocr_eos_ids(tokenizer);
    let (stream, timings) =
        greedy_ocr_token_stream(decoder, prompt_ids, embeds, &eos_ids, settings)?;
    let text = tokenizer.decode(&stream.tokens, false)?;
    Ok((
        OcrGeneration {
            tokens: stream.tokens,
            text,
            finish_reason: stream.finish_reason,
        },
        timings,
    ))
}

/// Guard -> `logits - logsumexp(logits)` -> argmax over a `[1, vocab]`
/// logits row, all on device (`_step` + the temperature-0 sampler).
#[cfg(feature = "mlx")]
fn sample(
    logits: &MlxArray,
    guard: Option<&SlidingWindowNoRepeatNgramProcessor>,
    pages: Option<&PageCountGuard>,
    stream: &[u32],
    vocab: usize,
) -> Result<MlxArray, MlxError> {
    let mut banned = guard.map(|g| g.banned(stream, vocab)).unwrap_or_default();
    let mut logits = logits.try_clone()?;
    // v1 has no page-count training: an early EOS opens the next page
    // (its logit moves to the marker's first token).
    if let (Some(g), Some(PageConstraint::BanEos)) = (pages, pages.map(PageCountGuard::constraint))
        && g.marker.len() > 1
        && g.partial() == 0
    {
        let pair = MlxArray::from_u32(&[1, 2], &[g.eos_id, g.marker[0]])?;
        let best = logits.take_along_axis(&pair, 1)?.max_axes(&[1], true)?;
        let idx = MlxArray::from_u32(&[1, 1], &[g.marker[0]])?;
        logits = logits.put_along_axis(&idx, &best, 1)?;
    }
    match pages.map(PageCountGuard::constraint) {
        Some(PageConstraint::Force(tok)) => return MlxArray::from_u32(&[1], &[tok]),
        Some(PageConstraint::BanEos) => {
            let eos = pages.expect("constraint").eos_id;
            if let Err(i) = banned.binary_search(&eos) {
                banned.insert(i, eos);
            }
        }
        _ => {}
    }
    let logits = if banned.is_empty() {
        logits
    } else {
        let idx = MlxArray::from_u32(&[1, banned.len()], &banned)?;
        let vals = MlxArray::scalar_bf16(f32::NEG_INFINITY)?
            .astype(logits.dtype()?)?
            .broadcast_to(&[1, banned.len()])?;
        logits.put_along_axis(&idx, &vals, 1)?
    };
    let logprobs = logits.subtract(&logits.logsumexp(-1, true)?)?;
    logprobs.argmax(-1, false)
}

/// [`greedy_ocr_generate`] without the tokenizer: explicit EOS set,
/// returning the token stream and timings. Mirrors mlx-vlm
/// `generate_step`: chunked prefill (2048) over `prompt[:-1]` evaluating
/// only the caches, the last prompt token as the first `_step` (guard
/// stream `[last prompt token]`), then single-token steps whose next
/// forward is enqueued before the current token is read back
/// (`async_eval` pipelining).
#[cfg(feature = "mlx")]
pub fn greedy_ocr_token_stream(
    decoder: &OcrDecoder,
    prompt_ids: &[u32],
    embeds: &MlxArray,
    eos_ids: &BTreeSet<u32>,
    settings: OcrGenerationSettings,
) -> Result<(TokenStream, GenerationTimings), GenerateError> {
    let p = prompt_ids.len();
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
    let guard = if settings.ngram_size >= 1 {
        Some(SlidingWindowNoRepeatNgramProcessor::new(
            settings.ngram_size,
            settings.ngram_window,
        )?)
    } else {
        None
    };
    let marker = match decoder.page_token_id() {
        Some(page_id) => vec![page_id],
        None => UNLIMITED_OCR_V1_PAGE_MARKER.to_vec(),
    };
    let mut pages = match eos_ids.first() {
        Some(&eos) if settings.expected_pages > 0 => Some(PageCountGuard::new(
            settings.expected_pages,
            marker,
            eos,
            settings.max_page_tokens,
        )),
        _ => None,
    };
    let t0 = Instant::now();
    let mut cache = decoder.new_cache(settings.kv_window, prompt_ids)?;

    // Chunked prefill over prompt[:-1]; only the cache state is evaluated.
    let mut start = 0usize;
    while p - start > 1 {
        let n = (p - start - 1).min(PREFILL_STEP_SIZE);
        let chunk = embeds.slice(&[0, start, 0], &[1, start + n, d], &[1, 1, 1])?;
        decoder.step(&chunk, &mut cache, None)?;
        let state = cache.state()?;
        let refs: Vec<&MlxArray> = state.iter().collect();
        MlxArray::eval_many(&refs)?;
        start += n;
        aimd_mlx::clear_cache()?;
    }

    let mut guard_stream: Vec<u32> = Vec::with_capacity(settings.max_new_tokens.min(8192) + 1);
    guard_stream.push(prompt_ids[p - 1]);
    let last = embeds.slice(&[0, p - 1, 0], &[1, p, d], &[1, 1, 1])?;
    let h = decoder.step(&last, &mut cache, None)?;
    let logits = decoder.logits(&h)?.reshape(&[1, vocab])?;
    let mut y = sample(
        &logits,
        guard.as_ref(),
        pages.as_ref(),
        &guard_stream,
        vocab,
    )?;
    MlxArray::async_eval(&[&y])?;

    let mut tokens: Vec<u32> = Vec::with_capacity(settings.max_new_tokens.min(1024));
    let mut finish_reason = FinishReason::Length;
    let mut prefill = None;
    if settings.max_new_tokens > 0 {
        loop {
            // Submit the next forward before reading `y` back (mlx-vlm
            // `generate_step`), so the GPU runs it while the host reads the
            // token; dropped on stop. Sampling waits for the token: the
            // n-gram guard needs it on the host.
            let next_logits = if tokens.len() + 1 < settings.max_new_tokens {
                let ids = y.reshape(&[1, 1])?;
                let emb = decoder.embed_ids(&ids)?;
                let h = decoder.step(&emb, &mut cache, Some(&ids))?;
                let logits = decoder.logits(&h)?.reshape(&[1, vocab])?;
                MlxArray::async_eval(&[&logits])?;
                Some(logits)
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
            let Some(logits) = next_logits else {
                break;
            };
            guard_stream.push(tok);
            if let Some(pages) = pages.as_mut() {
                pages.push(tok);
            }
            y = sample(
                &logits,
                guard.as_ref(),
                pages.as_ref(),
                &guard_stream,
                vocab,
            )?;
            MlxArray::async_eval(&[&y])?;
            if tokens.len().is_multiple_of(256) {
                aimd_mlx::clear_cache()?;
            }
        }
    }
    let total = t0.elapsed();
    let prefill = prefill.unwrap_or(total);
    Ok((
        TokenStream {
            tokens,
            finish_reason,
        },
        GenerationTimings {
            prefill,
            decode: total.saturating_sub(prefill),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The expected values below were produced by running the actual Python
    // implementation (SlidingWindowNoRepeatNgramProcessor in
    // src/aimd/plugins/ocr/models/mlx.py) on these exact inputs.

    fn pattern() -> Vec<u32> {
        (100..135).collect()
    }

    fn banned_ids(stream: &[u32], n: usize, w: usize, vocab: usize) -> Vec<u32> {
        banned_ids_with_whitelist(stream, n, w, vocab, &[])
    }

    fn banned_ids_with_whitelist(
        stream: &[u32],
        n: usize,
        w: usize,
        vocab: usize,
        whitelist: &[u32],
    ) -> Vec<u32> {
        let p = SlidingWindowNoRepeatNgramProcessor::with_whitelist(n, w, whitelist).unwrap();
        let mut logits: Vec<f32> = (0..vocab).map(|i| i as f32).collect();
        p.apply(stream, &mut logits);
        (0..vocab)
            .filter(|&i| logits[i] == f32::NEG_INFINITY)
            .map(|i| i as u32)
            .collect()
    }

    #[test]
    fn page_count_guard_counts_and_forces_multi_token_markers() {
        let m = UNLIMITED_OCR_V1_PAGE_MARKER;
        let mut g = PageCountGuard::new(2, m.to_vec(), 7, 3);
        for t in m {
            g.push(t);
        }
        g.push(5);
        assert_eq!(g.constraint(), PageConstraint::BanEos);
        // A partial marker is not a page; at the page limit the rest of the
        // marker is forced token by token.
        g.push(m[0]);
        g.push(m[1]);
        assert_eq!(g.constraint(), PageConstraint::Force(m[2]));
        g.push(m[2]);
        assert_eq!(g.constraint(), PageConstraint::None);
        for t in [5, 5, 5] {
            g.push(t);
        }
        assert_eq!(g.constraint(), PageConstraint::Force(7));
        for t in m {
            g.push(t);
        }
        assert_eq!(g.constraint(), PageConstraint::Force(7));
    }

    #[test]
    fn ocr_generate_ngram_guard_bans_repetition_completions() {
        let pat = pattern();
        // Two full 35-patterns (len 70): prefix = values 2..35 occurs at
        // idx 1 -> bans the completing 1. (Python: [1])
        let two = pat.iter().chain(pat.iter()).copied().collect::<Vec<_>>();
        assert_eq!(banned_ids(&two, 35, 128, 200), vec![100]);
        // Constant stream: 7 would complete a repeated n-gram. (Python: [7])
        assert_eq!(banned_ids(&[7u32; 40], 35, 128, 20), vec![7]);
        // Five patterns (len 175, healthy-looking but looping). (Python: [1])
        let five = pat.repeat(5);
        assert_eq!(banned_ids(&five, 35, 128, 200), vec![100]);
        // Repeat fully inside the window: two completions. (Python: [134, 999])
        let mut seq = pat.clone();
        seq.extend_from_slice(&pat[..34]);
        seq.push(999);
        seq.extend_from_slice(&pat[..34]);
        assert_eq!(banned_ids(&seq, 35, 128, 1000), vec![134, 999]);
    }

    #[test]
    fn ocr_generate_ngram_guard_leaves_healthy_streams_untouched() {
        let pat = pattern();
        // No repeated prefix anywhere: pat*3 + 60 fresh tokens. (Python: [])
        let mut seq = pat.repeat(3);
        seq.extend(200..260);
        assert!(banned_ids(&seq, 35, 128, 300).is_empty());
        // The matching occurrence would start at search_end (exclusive):
        // pat + 100 rising + pat[:-1]. (Python: [])
        let mut seq = pat.clone();
        seq.extend(200..300);
        seq.extend_from_slice(&pat[..34]);
        assert!(banned_ids(&seq, 35, 128, 400).is_empty());
        // Short streams are no-ops.
        assert!(banned_ids(&[1u32, 2, 3], 35, 128, 10).is_empty());
        assert!(banned_ids(&pat[..34], 35, 128, 10).is_empty());
    }

    #[test]
    fn ocr_generate_ngram_guard_window_boundary_semantics() {
        let pat = pattern();
        // The only prefix occurrence sits at idx 201 while search_end =
        // len - n + 1 = 201 (exclusive) — no ban for ANY window: widening
        // the window only moves search_start, never search_end.
        // (Python-verified: [] for w=128 and w=300.)
        let mut seq = vec![9u32; 200];
        seq.extend_from_slice(&pat);
        assert!(banned_ids(&seq, 35, 128, 400).is_empty());
        assert!(banned_ids(&seq, 35, 300, 400).is_empty());
        // A unique earlier occurrence of the final 34 tokens at idx 35:
        // outside the w=128 search range (search_start = 76) but inside
        // w=300 (search_start = 0), which bans the completing 7.
        // (Python-verified: [] and [7].)
        let mut seq = pat.clone();
        seq.extend(std::iter::repeat_n(5u32, 34));
        seq.push(7);
        seq.extend(std::iter::repeat_n(9u32, 100));
        seq.extend(std::iter::repeat_n(5u32, 34));
        assert!(banned_ids(&seq, 35, 128, 400).is_empty());
        assert_eq!(banned_ids(&seq, 35, 300, 400), vec![7]);
        // n = 1 bans every token seen in the window. (Python: [5, 9])
        assert_eq!(banned_ids(&[5u32, 5, 5, 9], 1, 128, 20), vec![5, 9]);
    }

    #[test]
    fn ocr_generate_ngram_guard_whitelist_and_vocab_bounds() {
        let pat = pattern();
        let mut seq = pat.clone();
        seq.extend_from_slice(&pat[..34]);
        seq.push(999);
        seq.extend_from_slice(&pat[..34]);
        // Whitelisted completions are un-banned. (Python: [134])
        assert_eq!(
            banned_ids_with_whitelist(&seq, 35, 128, 1000, &[999]),
            vec![134]
        );
        // Unrelated whitelist entries change nothing. (Python: [134, 999])
        assert_eq!(
            banned_ids_with_whitelist(&seq, 35, 128, 1000, &[998]),
            vec![134, 999]
        );
        // Banned ids outside [0, vocab) are filtered. (Python: [134])
        assert_eq!(banned_ids(&seq, 35, 128, 300), vec![134]);
        // Non-banned logits are untouched (spot check via one case).
        let p = SlidingWindowNoRepeatNgramProcessor::new(35, 128).unwrap();
        let mut logits: Vec<f32> = vec![1.0; 300];
        p.apply(&seq, &mut logits);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[299], 1.0);
        assert_eq!(logits[134], f32::NEG_INFINITY);
    }

    #[test]
    fn ocr_generate_ngram_guard_rejects_invalid_construction() {
        assert!(SlidingWindowNoRepeatNgramProcessor::new(0, 128).is_err());
        assert!(SlidingWindowNoRepeatNgramProcessor::new(35, 0).is_err());
    }
}
