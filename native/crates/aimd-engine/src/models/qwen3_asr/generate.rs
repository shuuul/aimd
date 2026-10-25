//! Greedy generation loop, repetition-loop detection, and output parsing
//! (WS-06b).
//!
//! Ports of the Python behavior contracts:
//!
//! - Greedy decoding mirrors mlx-audio `generate_step` + `stream_generate`
//!   (`.venv/.../mlx_audio/lm/generate.py` and
//!   `.../qwen3_asr/qwen3_asr.py`): prefill over the merged prompt embeds
//!   fills the KV cache, the first token is the argmax at the last prompt
//!   position, each next token is the argmax of the previous token's logits,
//!   positions continue sequentially, and an EOS token terminates generation
//!   WITHOUT being added to the output. `max_new_tokens` defaults to 4096
//!   (the transformers.py `max_new_tokens=4096` behavior contract).
//! - [`detect_repetition_loop`] is a regex-free port of
//!   `src/aimd/plugins/asr/audio_utils.py::detect_repetition_loop`, applied
//!   to the decoded text; the pipeline uses it to retry with the 8-bit
//!   checkpoint (any checkpoint dir loads through
//!   [`Qwen3Decoder::load`], so 4-bit/8-bit is just a different input).
//! - [`parse_qwen_output`] ports
//!   `src/aimd/plugins/asr/models/transformers.py::parse_qwen_output`
//!   (strip everything before the first `<asr_text>` marker, falling back to
//!   everything after the first `assistant\n`).

use std::collections::BTreeSet;
use std::fmt;

use aimd_tensor::{KvCache, Tensor, TensorError};

use super::decoder::{DecoderError, Qwen3Decoder};
use crate::tokenizer::{Tokenizer, TokenizerError};

/// Default generation budget (`max_new_tokens` in the Python backends).
pub const DEFAULT_MAX_NEW_TOKENS: usize = 4096;

/// Errors surfaced by the generation loop.
#[derive(Debug, thiserror::Error)]
pub enum GenerateError {
    #[error(transparent)]
    Decoder(#[from] DecoderError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error("tensor op: {0}")]
    Tensor(#[from] TensorError),
    #[error("prompt token ids are empty")]
    EmptyPrompt,
}

/// Knobs for [`greedy_generate`].
#[derive(Clone, Copy, Debug)]
pub struct GenerationSettings {
    /// Hard stop on generated (non-EOS) tokens. Default 4096.
    pub max_new_tokens: usize,
}

impl Default for GenerationSettings {
    fn default() -> Self {
        Self {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// An EOS token was produced (and dropped from the output).
    Eos,
    /// `max_new_tokens` non-EOS tokens were generated.
    MaxNewTokens,
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            FinishReason::Eos => "eos",
            FinishReason::MaxNewTokens => "max_new_tokens",
        })
    }
}

/// Result of one greedy generation.
#[derive(Clone, Debug)]
pub struct GenerationOutcome {
    /// Generated token ids (EOS excluded, matching mlx `stream_generate`).
    pub tokens: Vec<u32>,
    /// Decoded text of [`GenerationOutcome::tokens`] (special tokens skipped).
    pub text: String,
    pub finish_reason: FinishReason,
    /// True when the decoded text trips the repetition-loop detector
    /// (the pipeline retries with the 8-bit checkpoint).
    pub loop_detected: bool,
}

/// Resolves the EOS id set from the tokenizer, mirroring mlx-audio
/// `Qwen3ASRModel._eos_token_ids`: the `<|im_end|>` and `<|endoftext|>`
/// special ids, with the Qwen defaults `{151645, 151643}` as fallback when
/// the tokenizer knows neither (equal to the checkpoint
/// `generation_config.json` `eos_token_id` list).
pub fn resolve_eos_ids(tokenizer: &Tokenizer) -> BTreeSet<u32> {
    let mut ids = BTreeSet::new();
    for token in ["<|im_end|>", "<|endoftext|>"] {
        if let Ok(id) = tokenizer.token_id(token) {
            ids.insert(id);
        }
    }
    if ids.is_empty() {
        ids.insert(151645);
        ids.insert(151643);
    }
    ids
}

/// Token-level result of one greedy generation.
#[derive(Clone, Debug)]
pub struct TokenStream {
    /// Generated token ids (EOS excluded, matching mlx `stream_generate`).
    pub tokens: Vec<u32>,
    pub finish_reason: FinishReason,
}

/// Greedy transcription generation over merged prompt + audio embeds.
///
/// `prompt_ids` comes from `tokenizer::build_asr_prompt` (encoded), and
/// `audio_features` is the encoder output `[n_audio, hidden]` — every
/// `<|audio_pad|>` prompt position is replaced with a feature row (see
/// [`Qwen3Decoder::build_input_embeds`]). EOS ids resolve from the
/// tokenizer via [`resolve_eos_ids`].
pub fn greedy_generate(
    decoder: &Qwen3Decoder,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    audio_features: &Tensor,
    audio_token_id: u32,
    settings: GenerationSettings,
) -> Result<GenerationOutcome, GenerateError> {
    let eos_ids = resolve_eos_ids(tokenizer);
    let stream = greedy_token_stream(
        decoder,
        prompt_ids,
        audio_features,
        audio_token_id,
        &eos_ids,
        settings,
    )?;
    let text = tokenizer.decode(&stream.tokens, true)?;
    let loop_detected = detect_repetition_loop(&text);
    Ok(GenerationOutcome {
        tokens: stream.tokens,
        text,
        finish_reason: stream.finish_reason,
        loop_detected,
    })
}

/// [`greedy_generate`] without the tokenizer: explicit EOS set, returns the
/// token stream only (the pipeline retries with an 8-bit checkpoint based on
/// the decoded text; tests inject custom EOS sets or force full budgets).
pub fn greedy_token_stream(
    decoder: &Qwen3Decoder,
    prompt_ids: &[u32],
    audio_features: &Tensor,
    audio_token_id: u32,
    eos_ids: &BTreeSet<u32>,
    settings: GenerationSettings,
) -> Result<TokenStream, GenerateError> {
    if prompt_ids.is_empty() {
        return Err(GenerateError::EmptyPrompt);
    }
    let cfg = decoder.config();
    let mut cache = KvCache::new(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim,
        prompt_ids.len() + settings.max_new_tokens,
    )?;

    let embeds = decoder.build_input_embeds(prompt_ids, audio_features, audio_token_id)?;
    let hidden = decoder.prefill(&embeds, &mut cache)?;
    let prompt_len = prompt_ids.len();

    // First token: argmax at the last prompt position.
    let logits = decoder.logits(&last_row(&hidden, cfg.hidden_size)?)?;
    let mut y = argmax(logits.as_f32()?);

    let mut tokens = Vec::with_capacity(settings.max_new_tokens.min(1024));
    let mut finish_reason = FinishReason::MaxNewTokens;
    loop {
        if eos_ids.contains(&y) {
            finish_reason = FinishReason::Eos;
            break;
        }
        if tokens.len() >= settings.max_new_tokens {
            break;
        }
        tokens.push(y);
        let emb = decoder.embed_tokens(&[y])?;
        let position = (prompt_len + tokens.len() - 1) as u32;
        let step_hidden = decoder.step(&emb, position, &mut cache)?;
        let logits = decoder.logits(&step_hidden)?;
        y = argmax(logits.as_f32()?);
    }

    Ok(TokenStream {
        tokens,
        finish_reason,
    })
}

/// Last row of a rank-2 `F16 [seq, hidden]` tensor as `[1, hidden]`.
fn last_row(hidden: &Tensor, hidden_size: usize) -> Result<Tensor, DecoderError> {
    let dims = hidden.shape().dims();
    if dims.len() != 2 || dims[1] != hidden_size {
        return Err(DecoderError::Layout(format!(
            "hidden states must be rank-2 [seq, {hidden_size}], got {dims:?}"
        )));
    }
    let seq = dims[0];
    if seq == 0 {
        return Err(DecoderError::Layout("empty hidden states".to_string()));
    }
    let src = hidden.as_f16()?;
    Ok(Tensor::from_f16_vec(
        aimd_tensor::Shape::from_dims(&[1, hidden_size]),
        &src[(seq - 1) * hidden_size..seq * hidden_size],
    )?)
}

/// First-index argmax of an `f32` row (matches `np.argmax` tie behavior).
fn argmax(values: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Detects repetition loops in generated text.
///
/// Regex-free port of `src/aimd/plugins/asr/audio_utils.py
/// ::detect_repetition_loop`. Checks, in order:
///
/// 1. Any single non-whitespace character repeated 8+ times consecutively
///    (`([^\s])\1{7,}`).
/// 2. Any block of 2-10 non-whitespace characters repeated 4+ times
///    consecutively (`([^\s]{L})\1{3,}` for `L` in `2..=10`).
/// 3. Space-separated word sequences: a single word repeated 6+ times, or a
///    2-5 word phrase repeated 4+ times consecutively.
///
/// Whitespace runs collapse to single spaces first (matching
/// `re.sub(r"\s+", " ", text).strip()`); the detector operates on the decoded
/// text of a whole segment, like the Python pipeline.
pub fn detect_repetition_loop(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    // Collapse whitespace runs to single spaces and trim (re.sub + strip).
    let mut cleaned = String::with_capacity(text.len());
    let mut pending_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            pending_space = !cleaned.is_empty();
        } else {
            if pending_space {
                cleaned.push(' ');
                pending_space = false;
            }
            cleaned.push(ch);
        }
    }
    if cleaned.is_empty() {
        return false;
    }
    let chars: Vec<char> = cleaned.chars().collect();

    // 1. Single non-space character repeated 8+ times consecutively.
    let mut i = 0;
    while i < chars.len() {
        let mut run = 1;
        while i + run < chars.len() && chars[i + run] == chars[i] {
            run += 1;
        }
        if run >= 8 && !chars[i].is_whitespace() {
            return true;
        }
        i += run;
    }

    // 2. Block of L (2..=10) non-space characters repeated 4+ times.
    for length in 2..=10usize {
        if has_repeated_block(&chars, length) {
            return true;
        }
    }

    // 3. Space-separated word sequences.
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    if words.len() >= 5 {
        // Single word repeated 6+ times consecutively.
        let mut count = 1;
        for pair in words.windows(2) {
            if pair[0] == pair[1] {
                count += 1;
                if count >= 6 {
                    return true;
                }
            } else {
                count = 1;
            }
        }
        // Phrase of 2-5 words repeated 4+ times consecutively.
        for phrase_len in 2..=5usize {
            if words.len() >= phrase_len * 4
                && (0..=words.len() - phrase_len * 4).any(|i| {
                    let sub = &words[i..i + phrase_len];
                    words[i + phrase_len..i + 2 * phrase_len] == *sub
                        && words[i + 2 * phrase_len..i + 3 * phrase_len] == *sub
                        && words[i + 3 * phrase_len..i + 4 * phrase_len] == *sub
                })
            {
                return true;
            }
        }
    }
    false
}

/// `([^\s]{length})\1{3,}`: exists a start position whose `length` non-space
/// characters repeat four times consecutively.
fn has_repeated_block(chars: &[char], length: usize) -> bool {
    if chars.len() < length * 4 {
        return false;
    }
    for start in 0..=chars.len() - length * 4 {
        let block = &chars[start..start + length];
        if block.iter().any(|&c| c.is_whitespace()) {
            continue;
        }
        if (1..4).all(|k| chars[start + k * length..start + (k + 1) * length] == *block) {
            return true;
        }
    }
    false
}

/// Extracts transcription text from Qwen3-ASR generated output.
///
/// Port of `src/aimd/plugins/asr/models/transformers.py::parse_qwen_output`:
/// strip everything through the first `<asr_text>` marker; without a marker,
/// strip everything through the first `assistant\n`; otherwise return the
/// (stripped) input.
pub fn parse_qwen_output(output: &str) -> String {
    let text = output.trim();
    const MARKER: &str = "<asr_text>";
    if let Some(idx) = text.find(MARKER) {
        return text[idx + MARKER.len()..].trim().to_string();
    }
    const ASSISTANT: &str = "assistant\n";
    if let Some(idx) = text.find(ASSISTANT) {
        return text[idx + ASSISTANT.len()..].trim().to_string();
    }
    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;

    const ASR_HF_REPO: &str = "models--Qwen--Qwen3-ASR-1.7B-hf";

    fn hf_cache() -> PathBuf {
        PathBuf::from(
            std::env::var("AIMD_TEST_HF_CACHE")
                .unwrap_or_else(|_| format!("{}/.cache/huggingface/hub", env!("HOME"))),
        )
    }

    fn tokenizer_dir() -> Option<PathBuf> {
        let snapshots = hf_cache().join(ASR_HF_REPO).join("snapshots");
        let mut candidates: Vec<PathBuf> = std::fs::read_dir(snapshots)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|dir| dir.join("tokenizer.json").is_file())
            .collect();
        candidates.sort();
        candidates.into_iter().next()
    }

    #[test]
    fn generate_parse_qwen_output_matches_python_cases() {
        // Marker present: everything after the FIRST marker, stripped.
        assert_eq!(
            parse_qwen_output("language Chinese<asr_text>你好，世界"),
            "你好，世界"
        );
        assert_eq!(parse_qwen_output("<asr_text>text"), "text");
        assert_eq!(parse_qwen_output("junk<asr_text>  spaced  "), "spaced");
        // First marker wins; later markers stay in the text.
        assert_eq!(parse_qwen_output("a<asr_text>b<asr_text>c"), "b<asr_text>c");
        assert_eq!(parse_qwen_output("<asr_text>"), "");
        // assistant fallback (no marker).
        assert_eq!(parse_qwen_output("assistant\nhello there"), "hello there");
        assert_eq!(
            parse_qwen_output("<|im_start|>assistant\nthe text"),
            "the text"
        );
        // first assistant\n wins
        assert_eq!(
            parse_qwen_output("assistant\none\nassistant\ntwo"),
            "one\nassistant\ntwo"
        );
        // Plain passthrough.
        assert_eq!(parse_qwen_output("plain text"), "plain text");
        assert_eq!(parse_qwen_output("  padded  "), "padded");
        assert_eq!(parse_qwen_output(""), "");
        assert_eq!(parse_qwen_output("   \n\t "), "");
    }

    #[test]
    fn generate_repetition_detector_triggers_on_loops() {
        // 1. single character repeated 8+ times
        assert!(detect_repetition_loop("哈哈哈哈哈哈哈哈"));
        assert!(detect_repetition_loop("好的xxxxxxxx"));
        // boundary: 7 does not trigger, 8 does
        assert!(!detect_repetition_loop("aaaaaaa"));
        assert!(detect_repetition_loop("aaaaaaaa"));
        // 2. blocks of 2-10 non-space chars repeated 4+ times
        assert!(detect_repetition_loop("abcabcabcabc")); // len-3 block x4
        assert!(detect_repetition_loop("abababababab")); // len-2 block x6
        assert!(detect_repetition_loop("....完....完....完....完"));
        // 3. word/phrase repeats
        assert!(detect_repetition_loop("the the the the the the the"));
        assert!(detect_repetition_loop("a b c d a b c d a b c d a b c d"));
        assert!(detect_repetition_loop("我们 我们 我们 我们 我们 我们 走吧"));
        // whitespace collapses before matching
        assert!(detect_repetition_loop("the\nthe\nthe\nthe\nthe\nthe\nthe"));
        assert!(detect_repetition_loop("aaaaaaaaaaaa world"));
    }

    #[test]
    fn generate_repetition_detector_allows_healthy_text() {
        assert!(!detect_repetition_loop(""));
        assert!(!detect_repetition_loop("   \n  "));
        assert!(!detect_repetition_loop(
            "你好，世界！今天天气不错，我们出去走走吧。"
        ));
        assert!(!detect_repetition_loop(
            "Context engineering is the art of choosing what the model sees."
        ));
        // Below every threshold: char x7, block x3, word x5, phrase x3.
        assert!(!detect_repetition_loop("好的aaaaaaagood"));
        assert!(!detect_repetition_loop("abcabcabc"));
        assert!(!detect_repetition_loop("ok ok ok ok ok"));
        assert!(!detect_repetition_loop(
            "one two three one two three one two three"
        ));
        assert!(!detect_repetition_loop("abab abab abab abab"));
        // repeated substrings separated by other words do not count
        assert!(!detect_repetition_loop(
            "the cat sat the dog ran the bird flew"
        ));
    }

    #[test]
    fn generate_resolve_eos_ids_matches_qwen() {
        let Some(dir) = tokenizer_dir() else {
            eprintln!("skipping: {ASR_HF_REPO} not in HF cache");
            return;
        };
        let tok = Tokenizer::from_checkpoint(&dir).expect("load tokenizer");
        let ids = resolve_eos_ids(&tok);
        assert_eq!(
            ids.iter().copied().collect::<Vec<_>>(),
            vec![151643, 151645]
        );
    }

    /// Golden fixture bundle shared by the generate tests.
    struct Golden {
        prompt_ids: Vec<u32>,
        audio_token_id: u32,
        eos_ids: Vec<u32>,
        greedy_tokens: Vec<u32>,
        greedy_is_eos: Vec<bool>,
        first_eos_step: Option<usize>,
        greedy_top2_gap: Vec<f64>,
        features: Tensor,
    }

    fn load_golden() -> Option<Golden> {
        let manifest = golden_manifest()?;
        let parse_ids = |key: &str| -> Vec<u32> {
            manifest[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect()
        };
        let n_audio = manifest["n_audio"].as_u64().unwrap() as usize;
        let hidden = manifest["hidden"].as_u64().unwrap() as usize;
        let feats_raw = read_f32_le(&golden_dir().join("features.f32")).expect("features fixture");
        assert_eq!(feats_raw.len(), n_audio * hidden);
        Some(Golden {
            prompt_ids: parse_ids("prompt_ids"),
            audio_token_id: manifest["audio_token_id"].as_u64().unwrap() as u32,
            eos_ids: parse_ids("eos_ids"),
            greedy_tokens: parse_ids("greedy_tokens"),
            greedy_is_eos: manifest["greedy_is_eos"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect(),
            first_eos_step: manifest["first_eos_step"].as_u64().map(|v| v as usize),
            greedy_top2_gap: manifest["greedy_top2_gap"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect(),
            features: Tensor::from_f32_vec(
                aimd_tensor::Shape::from_dims(&[n_audio, hidden]),
                &feats_raw,
            )
            .unwrap(),
        })
    }

    /// End-to-end decoder parity over the forced 16-step greedy argmax
    /// stream (see the golden manifest notes: the production eos-aware loop
    /// emits EOS at step 0 on this synthetic input, so the fixture locks the
    /// raw argmax stream instead). Argmax picks must match MLX exactly.
    #[test]
    fn generate_forced_argmax_stream_matches_golden() {
        let Some(decoder) = shared_decoder() else {
            return;
        };
        let Some(g) = load_golden() else {
            eprintln!("skipping: golden decoder fixture absent");
            return;
        };
        let cfg = decoder.config();
        let embeds = decoder
            .build_input_embeds(&g.prompt_ids, &g.features, g.audio_token_id)
            .unwrap();
        let prompt_len = g.prompt_ids.len();
        let max_tokens = g.greedy_tokens.len();
        let mut cache = KvCache::new(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            prompt_len + max_tokens,
        )
        .unwrap();
        let hidden = decoder.prefill(&embeds, &mut cache).unwrap();
        let logits = decoder
            .logits(&last_row(&hidden, cfg.hidden_size).unwrap())
            .unwrap();
        let mut y = argmax(logits.as_f32().unwrap());

        let mut tokens: Vec<u32> = Vec::with_capacity(max_tokens);
        for step in 0..max_tokens {
            tokens.push(y);
            if tokens.len() >= max_tokens {
                break;
            }
            let emb = decoder.embed_tokens(&[y]).unwrap();
            let step_hidden = decoder
                .step(&emb, (prompt_len + step) as u32, &mut cache)
                .unwrap();
            let logits = decoder.logits(&step_hidden).unwrap();
            y = argmax(logits.as_f32().unwrap());
        }
        assert_eq!(
            tokens, g.greedy_tokens,
            "forced argmax stream must match MLX"
        );
        // EOS flags must line up with the golden stream too.
        for (i, &t) in tokens.iter().enumerate() {
            assert_eq!(
                g.eos_ids.contains(&t),
                g.greedy_is_eos[i],
                "eos flag mismatch at step {i}"
            );
        }
        // The minimum top-2 logit gap MLX saw before each pick: any argmax
        // flip here would mean a near-tie diverged under f16 drift.
        if let Some(min_gap) = g.greedy_top2_gap.iter().copied().reduce(f64::min) {
            eprintln!("golden min top-2 logit gap: {min_gap:.6}");
            assert!(
                min_gap > 0.1,
                "golden stream has a near-tie; parity inconclusive"
            );
        }
    }

    /// Production semantics on the golden input: the eos-aware loop must
    /// stop exactly where MLX did (EOS dropped from the token list).
    #[test]
    fn generate_stops_at_golden_eos() {
        let Some(decoder) = shared_decoder() else {
            return;
        };
        let Some(g) = load_golden() else { return };
        let eos: BTreeSet<u32> = g.eos_ids.iter().copied().collect();
        let stream = greedy_token_stream(
            decoder,
            &g.prompt_ids,
            &g.features,
            g.audio_token_id,
            &eos,
            GenerationSettings {
                max_new_tokens: 4096,
            },
        )
        .expect("greedy token stream");
        match g.first_eos_step {
            Some(0) => {
                assert!(
                    stream.tokens.is_empty(),
                    "EOS at step 0 must yield no tokens"
                );
                assert_eq!(stream.finish_reason, FinishReason::Eos);
            }
            Some(step) => {
                assert_eq!(stream.tokens.len(), step);
                assert_eq!(stream.finish_reason, FinishReason::Eos);
            }
            None => {
                assert_eq!(stream.tokens, g.greedy_tokens);
                assert_eq!(stream.finish_reason, FinishReason::MaxNewTokens);
            }
        }
    }

    /// The full `greedy_generate` wrapper: tokenizer decode + loop detector
    /// run over the production outcome.
    #[test]
    fn generate_greedy_outcome_decodes_text_and_detects_loops() {
        let Some(decoder) = shared_decoder() else {
            return;
        };
        let Some(g) = load_golden() else { return };
        let Some(tok_dir) = tokenizer_dir() else {
            eprintln!("skipping: {ASR_HF_REPO} not in HF cache");
            return;
        };
        let tok = Tokenizer::from_checkpoint(&tok_dir).expect("load tokenizer");
        let outcome = greedy_generate(
            decoder,
            &tok,
            &g.prompt_ids,
            &g.features,
            g.audio_token_id,
            GenerationSettings { max_new_tokens: 16 },
        )
        .expect("greedy generate");
        match g.first_eos_step {
            Some(0) => {
                assert!(outcome.tokens.is_empty());
                assert_eq!(outcome.text, "");
                assert_eq!(outcome.finish_reason, FinishReason::Eos);
                assert!(!outcome.loop_detected);
            }
            _ => {
                assert_eq!(outcome.finish_reason, FinishReason::MaxNewTokens);
                assert!(!outcome.text.is_empty());
                // The token stream must agree with the forced golden prefix.
                let eos: BTreeSet<u32> = g.eos_ids.iter().copied().collect();
                let prefix_len = outcome.tokens.len();
                assert_eq!(outcome.tokens, &g.greedy_tokens[..prefix_len]);
                assert!(
                    !outcome.tokens.iter().any(|t| eos.contains(t)),
                    "eos-aware outcome must not contain eos tokens"
                );
            }
        }
    }

    #[test]
    fn generate_stops_at_max_new_tokens() {
        let Some(decoder) = shared_decoder() else {
            return;
        };
        let Some(g) = load_golden() else { return };
        assert!(
            g.greedy_tokens.len() >= 3,
            "golden stream must have >= 3 tokens"
        );
        // Empty EOS set: the budget is the only stop condition.
        let eos: BTreeSet<u32> = BTreeSet::new();
        let stream = greedy_token_stream(
            decoder,
            &g.prompt_ids,
            &g.features,
            g.audio_token_id,
            &eos,
            GenerationSettings { max_new_tokens: 3 },
        )
        .expect("greedy token stream");
        assert_eq!(stream.tokens.len(), 3);
        assert_eq!(stream.tokens, &g.greedy_tokens[..3]);
        assert_eq!(stream.finish_reason, FinishReason::MaxNewTokens);
    }

    fn golden_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden/asr_decoder")
    }

    fn golden_manifest() -> Option<serde_json::Value> {
        let raw = std::fs::read(golden_dir().join("manifest.json")).ok()?;
        serde_json::from_slice(&raw).ok()
    }

    fn read_f32_le(path: &Path) -> Result<Vec<f32>, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| f32::from_le_bytes(*p))
            .collect())
    }

    fn shared_decoder() -> Option<&'static Qwen3Decoder> {
        static DECODER: OnceLock<Option<Qwen3Decoder>> = OnceLock::new();
        DECODER
            .get_or_init(|| {
                let repo = "models--mlx-community--Qwen3-ASR-1.7B-4bit";
                let snapshots = hf_cache().join(repo).join("snapshots");
                let dir = std::fs::read_dir(snapshots)
                    .ok()?
                    .flatten()
                    .map(|e| e.path())
                    .find(|dir| {
                        dir.join("config.json").is_file() && dir.join("model.safetensors").is_file()
                    });
                let Some(dir) = dir else {
                    eprintln!("skipping: {repo} not in HF cache");
                    return None;
                };
                match Qwen3Decoder::load(&dir) {
                    Ok(decoder) => Some(decoder),
                    Err(e) => {
                        eprintln!("skipping: decoder load failed: {e}");
                        None
                    }
                }
            })
            .as_ref()
    }
}
