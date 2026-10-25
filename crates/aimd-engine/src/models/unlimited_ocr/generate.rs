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

use std::time::{Duration, Instant};

use aimd_mlx::{MlxArray, MlxDtype, MlxError};

use super::mlx_decoder::{DecoderError, OcrDecoder};
use crate::tokenizer::{Tokenizer, TokenizerError};

/// Generation budget (`UNLIMITED_OCR_MAX_TOKENS` in the Python adapter).
pub const UNLIMITED_OCR_MAX_TOKENS: usize = 8192;

/// N-gram size of the sliding-window guard (`UNLIMITED_OCR_NGRAM_SIZE`).
pub const UNLIMITED_OCR_NGRAM_SIZE: usize = 35;

/// Search window of the sliding-window guard (`UNLIMITED_OCR_NGRAM_WINDOW`).
pub const UNLIMITED_OCR_NGRAM_WINDOW: usize = 128;

/// The `<image>` token id (128815) replaced by vision features.
pub const OCR_IMAGE_TOKEN_ID: u32 = 128815;

/// Errors surfaced by the generation loop.
#[derive(Debug, thiserror::Error)]
pub enum GenerateError {
    #[error(transparent)]
    Decoder(#[from] DecoderError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error(transparent)]
    Mlx(#[from] MlxError),
    #[error("prompt token ids are empty")]
    EmptyPrompt,
    #[error("invalid generation settings: {0}")]
    InvalidSettings(String),
    #[error("layout: {0}")]
    Layout(String),
}

/// Knobs for [`greedy_ocr_token_stream`].
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
}

impl Default for OcrGenerationSettings {
    fn default() -> Self {
        Self {
            max_new_tokens: UNLIMITED_OCR_MAX_TOKENS,
            ngram_size: UNLIMITED_OCR_NGRAM_SIZE,
            ngram_window: UNLIMITED_OCR_NGRAM_WINDOW,
            kv_window: None,
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

/// Splices `features` (`[n, hidden]` bf16) over the prompt's `<image>` run:
/// `embed_tokens(ids)` with `input_embeds[0, image_positions] = features`
/// (`get_input_embeddings`). The run must be one contiguous block of
/// exactly `n` image tokens (the gundam prompt layout).
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
    let first = prompt_ids.iter().position(|&t| t == image_token_id);
    let count = prompt_ids.iter().filter(|&&t| t == image_token_id).count();
    let embeds = decoder.embed_ids(&decoder.ids_array(prompt_ids)?)?;
    let Some(first) = first else {
        if n == 0 {
            return Ok(embeds);
        }
        return Err(GenerateError::Layout("prompt has no <image> tokens".into()));
    };
    if count != n
        || prompt_ids[first..first + n]
            .iter()
            .any(|&t| t != image_token_id)
    {
        return Err(GenerateError::Layout(format!(
            "prompt has {count} <image> tokens (contiguous from {first}) for {n} feature rows"
        )));
    }
    let p = prompt_ids.len();
    let head = embeds.slice(&[0, 0, 0], &[1, first, d], &[1, 1, 1])?;
    let tail = embeds.slice(&[0, first + n, 0], &[1, p, d], &[1, 1, 1])?;
    let feats = features.astype(MlxDtype::Bf16)?.reshape(&[1, n, d])?;
    Ok(MlxArray::concatenate(&[&head, &feats, &tail], 1)?)
}

/// Greedy OCR generation over spliced prompt embeddings (tokenizer
/// included: decodes the generated tokens, special tokens kept). EOS ids
/// resolve via [`resolve_ocr_eos_ids`].
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
fn sample(
    logits: &MlxArray,
    guard: Option<&SlidingWindowNoRepeatNgramProcessor>,
    stream: &[u32],
    vocab: usize,
) -> Result<MlxArray, MlxError> {
    let banned = guard.map(|g| g.banned(stream, vocab)).unwrap_or_default();
    let logits = if banned.is_empty() {
        logits.try_clone()?
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
    let t0 = Instant::now();
    let mut caches = decoder.make_cache(settings.kv_window);

    // Chunked prefill over prompt[:-1]; only the cache state is evaluated.
    let mut start = 0usize;
    while p - start > 1 {
        let n = (p - start - 1).min(PREFILL_STEP_SIZE);
        let chunk = embeds.slice(&[0, start, 0], &[1, start + n, d], &[1, 1, 1])?;
        decoder.forward(&chunk, &mut caches)?;
        let mut state = Vec::with_capacity(caches.len() * 2);
        for c in &caches {
            state.extend(c.state()?);
        }
        let refs: Vec<&MlxArray> = state.iter().collect();
        MlxArray::eval_many(&refs)?;
        start += n;
        aimd_mlx::clear_cache()?;
    }

    let mut guard_stream: Vec<u32> = Vec::with_capacity(settings.max_new_tokens.min(8192) + 1);
    guard_stream.push(prompt_ids[p - 1]);
    let last = embeds.slice(&[0, p - 1, 0], &[1, p, d], &[1, 1, 1])?;
    let h = decoder.forward(&last, &mut caches)?;
    let logits = decoder.logits(&h)?.reshape(&[1, vocab])?;
    let mut y = sample(&logits, guard.as_ref(), &guard_stream, vocab)?;
    MlxArray::async_eval(&[&y])?;

    let mut tokens: Vec<u32> = Vec::with_capacity(settings.max_new_tokens.min(1024));
    let mut finish_reason = FinishReason::Length;
    let mut prefill = None;
    if settings.max_new_tokens > 0 {
        loop {
            // Enqueue the next forward before reading `y` back (mlx-vlm
            // builds `_step(y[None])` first); dropped unevaluated on stop.
            let next_logits = if tokens.len() + 1 < settings.max_new_tokens {
                let emb = decoder.embed_ids(&y.reshape(&[1, 1])?)?;
                let h = decoder.forward(&emb, &mut caches)?;
                Some(decoder.logits(&h)?.reshape(&[1, vocab])?)
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
            y = sample(&logits, guard.as_ref(), &guard_stream, vocab)?;
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

/// Removes the Unlimited-OCR layout protocol while preserving content.
///
/// Line-by-line port of `normalize_unlimited_ocr_markdown` in
/// `src/aimd/plugins/ocr/models/unlimited.py` (including its
/// `_DETECTION_LINE` regex, implemented by hand): lines that match
/// `(?:<\|det\|>)?([^<\s|]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)` open a new
/// block (the `image` category is dropped entirely); every other non-empty
/// line appends to the current block; blocks are joined with blank lines.
/// Text without any det marker passes through `str.strip()`-ed.
pub fn normalize_unlimited_ocr_markdown(raw: &str) -> String {
    if !raw.contains("<|det|>") && !raw.contains("<|/det|>") {
        return raw.trim().to_string();
    }

    let mut blocks: Vec<Vec<String>> = Vec::new();
    let mut current: Option<Vec<String>> = None;
    for line in python_splitlines(raw) {
        let line = line.rstrip();
        if line.is_empty() {
            continue;
        }
        if let Some((category, content)) = parse_detection_line(&line) {
            let category = category.trim();
            if category == "image" {
                continue;
            }
            if let Some(prev) = current.take() {
                blocks.push(prev);
            }
            current = if content.is_empty() {
                Some(Vec::new())
            } else {
                Some(vec![content])
            };
            continue;
        }
        current.get_or_insert_with(Vec::new).push(line);
    }
    if let Some(last) = current {
        blocks.push(last);
    }
    let joined: Vec<String> = blocks.iter().map(|block| block.join("\n")).collect();
    joined.join("\n\n").trim().to_string()
}

/// Python `str.splitlines()` boundaries: `\n`, `\r\n`, `\r`, `\v`, `\f`,
/// `\x1c`, `\x1d`, `\x1e`, `\x85` (NEL), `\u{2028}`, `\u{2029}`. The
/// terminators are not kept.
fn python_splitlines(text: &str) -> Vec<String> {
    let is_break = |c: char| {
        matches!(
            c,
            '\n' | '\r'
                | '\u{b}'
                | '\u{c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        )
    };
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' && chars.peek() == Some(&'\n') {
            chars.next();
            lines.push(std::mem::take(&mut current));
        } else if is_break(c) {
            lines.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    lines.push(current);
    lines
}

/// `str.rstrip()` (Python strips unicode whitespace from the end).
trait PythonStr {
    fn rstrip(&self) -> String;
}

impl PythonStr for String {
    fn rstrip(&self) -> String {
        self.trim_end().to_string()
    }
}

/// Hand-rolled `(?:<\|det\|>)?([^<\s|]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)`
/// anchored at the line start, with the regex's single backtrack point (the
/// optional bracket group). Returns `(group 1, group 2)`.
fn parse_detection_line(line: &str) -> Option<(String, String)> {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let starts_with = |at: usize, needle: &str| -> bool {
        let nd: Vec<char> = needle.chars().collect();
        at + nd.len() <= n && chars[at..at + nd.len()] == nd[..]
    };
    let mut p = 0usize;
    if starts_with(p, "<|det|>") {
        p += "<|det|>".chars().count();
    }
    let group1_start = p;
    // Greedy `[^<\s|]+` scan (the run cannot contain <, |, or whitespace).
    let mut group1_end = group1_start;
    while group1_end < n
        && chars[group1_end] != '<'
        && chars[group1_end] != '|'
        && !chars[group1_end].is_whitespace()
    {
        group1_end += 1;
    }
    if group1_end == group1_start {
        return None;
    }

    // Continuation after group 1: optional `\s*\[[^\]]*\]` then `\s*<|/det|>`.
    // The regex engine backtracks group 1 from its longest match downwards,
    // and the backtrack matters: "title[1, 2]<|/det|>" only matches with
    // group 1 = "title" so the bracket group can consume "[1, 2]".
    let try_continue = |end: usize| -> Option<usize> {
        // Attempt 1: `\s*\[[^\]]*\]` followed by `\s*<|/det|>`.
        let mut q = end;
        while q < n && chars[q].is_whitespace() {
            q += 1;
        }
        if q < n && chars[q] == '[' {
            let mut r = q + 1;
            while r < n && chars[r] != ']' {
                r += 1;
            }
            if r < n {
                let mut s = r + 1;
                while s < n && chars[s].is_whitespace() {
                    s += 1;
                }
                if starts_with(s, "<|/det|>") {
                    return Some(s + "<|/det|>".chars().count());
                }
            }
        }
        // Attempt 2 (backtrack past the bracket group): `\s*<|/det|>`.
        let mut s = end;
        while s < n && chars[s].is_whitespace() {
            s += 1;
        }
        if starts_with(s, "<|/det|>") {
            Some(s + "<|/det|>".chars().count())
        } else {
            None
        }
    };

    for end in (group1_start + 1..=group1_end).rev() {
        if let Some(content_start) = try_continue(end) {
            let category: String = chars[group1_start..end].iter().collect();
            return Some((category, chars[content_start..].iter().collect()));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // The expected values below were produced by running the actual Python
    // implementations (SlidingWindowNoRepeatNgramProcessor in
    // src/aimd/plugins/ocr/models/mlx.py and normalize_unlimited_ocr_markdown
    // in src/aimd/plugins/ocr/models/unlimited.py) on these exact inputs.

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

    #[test]
    fn ocr_generate_markdown_normalizer_matches_python() {
        let cases: Vec<(&str, &str)> = vec![
            ("  Hello world\nsecond line  ", "Hello world\nsecond line"),
            (
                "<|det|>title[1, 2, 3, 4]<|/det|>Hello world\nsecond line\n\n<|det|>text[0, 0, 9, 9]<|/det|>More text",
                "Hello world\nsecond line\n\nMore text",
            ),
            (
                "<|det|>image[0, 0, 5, 5]  <|/det|>\n<|det|>text[1, 1, 8, 8]<|/det|>keep me",
                "keep me",
            ),
            (
                "header[x, y]<|/det|>the content\nnext line",
                "the content\nnext line",
            ),
            ("<|det|>title[1, 2]<|/det|>\nfiller line", "filler line"),
            (
                "<|det|>a[1]<|/det|>one\n\n\n<|det|>b[2]<|/det|>two",
                "one\n\ntwo",
            ),
            (
                "intro line\n<|det|>text[1]<|/det|>body",
                "intro line\n\nbody",
            ),
            (
                "<|det|>text[0,0,9,9]<|/det|>line1\nline2\n\nline3\n<|det|>text[1,1,8,8]<|/det|>B1\nB2",
                "line1\nline2\nline3\n\nB1\nB2",
            ),
            ("<|det|>image[0,0,1,1]<|/det|>", ""),
            ("text before\n<|det|>image[0,0,1,1]<|/det|>", "text before"),
            (
                // Category run stops before whitespace, so this line does
                // not match the det regex and passes through untouched.
                "<|det|> table  [1, 2, 3, 4]  <|/det|> cell text",
                "<|det|> table  [1, 2, 3, 4]  <|/det|> cell text",
            ),
            (
                // Bracket group consumed but the continuation fails and the
                // no-bracket fallback fails too: no match, line kept.
                "text [1,2] blah <|/det|>not matched",
                "text [1,2] blah <|/det|>not matched",
            ),
            (
                "<|det|>标题[1, 2, 3, 4]<|/det|>你好世界\n第二行",
                "你好世界\n第二行",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(
                normalize_unlimited_ocr_markdown(input),
                expected,
                "case input {input:?}"
            );
        }
    }

    #[test]
    fn ocr_generate_markdown_normalizer_splitlines_matches_python() {
        // No det markers: plain strip — the \r is NOT a split boundary on
        // this path (Python returns raw.strip()).
        assert_eq!(normalize_unlimited_ocr_markdown("a\rb\rc"), "a\rb\rc");
        // With det markers present, splitlines applies (lone \r splits).
        assert_eq!(
            normalize_unlimited_ocr_markdown("<|det|>t[1]<|/det|>x\ry"),
            "x\ny"
        );
        // No det markers: plain strip.
        assert_eq!(normalize_unlimited_ocr_markdown("   "), "");
        assert_eq!(normalize_unlimited_ocr_markdown(""), "");
    }
}
