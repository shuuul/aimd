//! Backend-agnostic generation helpers: settings/outcome types, EOS
//! resolution, repetition-loop detection, and output parsing (WS-06b).
//!
//! The greedy loop itself lives in [`super::pipeline`] (MLX); this module
//! holds the pieces the Python behavior contracts define independent of the
//! compute backend:
//!
//! - Greedy decoding semantics mirror mlx-audio `generate_step` +
//!   `stream_generate`: the first token is the argmax at the last prompt
//!   position, each next token is the argmax of the previous token's logits,
//!   and an EOS token terminates generation WITHOUT being added to the
//!   output. `max_new_tokens` defaults to 4096 (the transformers.py
//!   `max_new_tokens=4096` behavior contract).
//! - [`detect_repetition_loop`] is a regex-free port of
//!   `src/aimd/plugins/asr/audio_utils.py::detect_repetition_loop`, applied
//!   to the decoded text; the pipeline uses it to retry with the 8-bit
//!   checkpoint (4-bit/8-bit is just a different checkpoint directory).
//! - [`parse_qwen_output`] ports
//!   `src/aimd/plugins/asr/models/transformers.py::parse_qwen_output`
//!   (strip everything before the first `<asr_text>` marker, falling back to
//!   everything after the first `assistant\n`).

use std::collections::BTreeSet;
use std::fmt;

use crate::tokenizer::Tokenizer;

/// Default generation budget (`max_new_tokens` in the Python backends).
pub const DEFAULT_MAX_NEW_TOKENS: usize = 4096;

/// Knobs for the greedy generation loop of [`super::pipeline`].
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
    use std::path::PathBuf;

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
}
