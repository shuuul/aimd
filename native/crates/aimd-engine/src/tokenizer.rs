//! Tokenizer wrapper and chat-template builders (WS-04).
//!
//! # Contract
//!
//! - [`Tokenizer::from_checkpoint`] loads `tokenizer.json` from a snapshot
//!   directory via the `tokenizers` crate (HF reference implementation).
//! - [`build_asr_prompt`] reproduces the exact chat string rendered by the
//!   Qwen3-ASR processor (golden-locked by `native/golden/tokenizer/`):
//!
//!   ```text
//!   <|im_start|>system\n{context or empty}<|im_end|>\n
//!   <|im_start|>user\n<|audio_start|>{<|audio_pad|> × N}<|audio_end|><|im_end|>\n
//!   <|im_start|>assistant\n[language {Name}<asr_text>]
//!   ```
//!
//!   Notes verified against fixtures: the system turn is always present
//!   (empty body without context); the language continuation has no
//!   trailing newline; `N` (`audio_pad_count`) is computed by the audio
//!   pipeline (WS-05/WS-06) and passed in.
//! - [`build_ocr_prompt_ids`] reproduces the Unlimited-OCR id layout:
//!   `[bos] ++ encode(pre_image_text, no specials) ++ [image_pad × N] ++
//!   encode(post_image_text, no specials)` with a single contiguous
//!   `<image>` pad run for single-image gundam mode. `N`
//!   (`image_pad_count`) comes from the vision preprocessing (WS-07).
//! - Language aliases match the Python map (short codes and full names).

use std::path::{Path, PathBuf};

/// Errors for tokenizer loading, encoding, and template building.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("cannot load tokenizer from {path}: {reason}")]
    Load { path: PathBuf, reason: String },
    #[error("tokenizer encode failed: {0}")]
    Encode(String),
    #[error("tokenizer decode failed: {0}")]
    Decode(String),
    #[error("unknown special token {0:?} in this tokenizer")]
    UnknownToken(String),
}

/// Wraps a HF `tokenizers` tokenizer loaded from a checkpoint snapshot.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Load a tokenizer from a checkpoint snapshot directory.
    ///
    /// Prefers `tokenizer.json` (fast path, e.g. the `-hf` repos). MLX
    /// quantized repos (e.g. `mlx-community/Qwen3-ASR-1.7B-4bit`) ship the
    /// legacy `vocab.json` + `merges.txt` + `tokenizer_config.json` form
    /// instead — exactly what `transformers.AutoTokenizer` consumes — so we
    /// reconstruct the byte-level BPE from those, mirroring mlx-audio's
    /// `post_load_hook`.
    pub fn from_checkpoint(dir: &Path) -> Result<Self, TokenizerError> {
        let file = dir.join("tokenizer.json");
        if file.is_file() {
            let inner =
                tokenizers::Tokenizer::from_file(&file).map_err(|e| TokenizerError::Load {
                    path: file,
                    reason: e.to_string(),
                })?;
            return Ok(Self { inner });
        }
        Self::from_vocab_merges(dir)
    }

    /// Reconstruct a byte-level BPE tokenizer from `vocab.json` +
    /// `merges.txt` + special tokens from `tokenizer_config.json`
    /// (`added_tokens_decoder`). Equivalence with the `tokenizer.json`
    /// fast path is locked by the `tokenizer_vocab_merges_equivalence`
    /// test.
    fn from_vocab_merges(dir: &Path) -> Result<Self, TokenizerError> {
        let load_err = |reason: String| TokenizerError::Load {
            path: dir.to_path_buf(),
            reason,
        };
        let vocab_path = dir.join("vocab.json");
        let merges_path = dir.join("merges.txt");
        if !vocab_path.is_file() || !merges_path.is_file() {
            return Err(load_err(
                "neither tokenizer.json nor vocab.json+merges.txt present".to_string(),
            ));
        }
        let bpe = tokenizers::models::bpe::BPE::from_file(
            vocab_path
                .to_str()
                .ok_or_else(|| load_err("non-utf8 vocab path".to_string()))?,
            merges_path
                .to_str()
                .ok_or_else(|| load_err("non-utf8 merges path".to_string()))?,
        )
        .build()
        .map_err(|e| load_err(format!("bpe build: {e}")))?;
        let mut inner = tokenizers::Tokenizer::new(bpe);
        // Qwen byte-level tokenization: regex split + byte alphabet,
        // no prefix-space injection (verified by the equivalence test).
        let pre: tokenizers::PreTokenizerWrapper =
            tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, true).into();
        let dec: tokenizers::DecoderWrapper =
            tokenizers::decoders::byte_level::ByteLevel::default().into();
        inner.with_pre_tokenizer(Some(pre)).with_decoder(Some(dec));

        // Special tokens live in tokenizer_config.json `added_tokens_decoder`
        // (id -> {content, special}); register them so <|...|> markers in
        // chat strings encode to their fixed ids.
        let cfg_path = dir.join("tokenizer_config.json");
        if cfg_path.is_file() {
            let cfg_raw = std::fs::read(&cfg_path)
                .map_err(|e| load_err(format!("tokenizer_config.json: {e}")))?;
            let cfg: serde_json::Value = serde_json::from_slice(&cfg_raw)
                .map_err(|e| load_err(format!("tokenizer_config.json: {e}")))?;
            if let Some(map) = cfg.get("added_tokens_decoder").and_then(|v| v.as_object()) {
                let mut specials: Vec<tokenizers::AddedToken> = Vec::new();
                for entry in map.values() {
                    let content = entry.get("content").and_then(|c| c.as_str());
                    let special = entry
                        .get("special")
                        .and_then(|s| s.as_bool())
                        .unwrap_or(false);
                    if let Some(content) = content {
                        specials.push(tokenizers::AddedToken {
                            content: content.to_string(),
                            special,
                            single_word: false,
                            lstrip: false,
                            rstrip: false,
                            normalized: false,
                        });
                    }
                }
                inner.add_special_tokens(&specials);
            }
        }
        Ok(Self { inner })
    }

    /// Encode text containing inline special-token markers; the tokenizer's
    /// added-special handling must not inject extras (chat strings carry
    /// their own specials), matching the HF processor behavior.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode ids back to text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    /// Resolve a special token string to its id.
    pub fn token_id(&self, token: &str) -> Result<u32, TokenizerError> {
        self.inner
            .token_to_id(token)
            .ok_or_else(|| TokenizerError::UnknownToken(token.to_string()))
    }
}

/// Build the exact Qwen3-ASR transcription prompt string.
///
/// `context` injects a free-form biasing text into the system turn;
/// `language` appends the `language {Name}<asr_text>` continuation to the
/// assistant turn; `audio_pad_count` is the number of `<|audio_pad|>`
/// repeats between the audio markers.
pub fn build_asr_prompt(
    context: Option<&str>,
    language: Option<&str>,
    audio_pad_count: usize,
) -> String {
    let mut s = String::with_capacity(96 + audio_pad_count * 13);
    s.push_str("<|im_start|>system\n");
    if let Some(ctx) = context {
        s.push_str(ctx);
    }
    s.push_str("<|im_end|>\n<|im_start|>user\n<|audio_start|>");
    for _ in 0..audio_pad_count {
        s.push_str("<|audio_pad|>");
    }
    s.push_str("<|audio_end|><|im_end|>\n<|im_start|>assistant\n");
    if let Some(lang) = language {
        s.push_str("language ");
        s.push_str(lang);
        s.push_str("<asr_text>");
    }
    s
}

/// Rendered OCR user turn text for single-image gundam mode
/// (`<image>{user_text}`).
pub fn build_ocr_prompt_text(user_text: &str) -> String {
    format!("<image>{user_text}")
}

/// Build the Unlimited-OCR prompt token ids for single-image mode:
/// `[bos] ++ encode(pre) ++ [image_pad × image_pad_count] ++ encode(post)`.
///
/// `pre`/`post` are the text around the single `<image>` placeholder
/// (currently empty before, `user_text` after).
pub fn build_ocr_prompt_ids(
    tok: &Tokenizer,
    user_text: &str,
    image_pad_count: usize,
) -> Result<Vec<u32>, TokenizerError> {
    // DeepSeek-V3 tokenizer: id 0 is <｜begin▁of▁sentence｜>.
    let bos = tok
        .inner
        .token_to_id("<｜begin▁of▁sentence｜>")
        .unwrap_or(0);
    let image_pad = tok.token_id("<image>")?;
    let post = tok.encode(user_text)?;
    let mut ids = Vec::with_capacity(1 + image_pad_count + post.len());
    ids.push(bos);
    for _ in 0..image_pad_count {
        ids.push(image_pad);
    }
    ids.extend_from_slice(&post);
    Ok(ids)
}

/// Short language code -> canonical name map (mirrors the Python
/// `LANGUAGE_CODE_TO_NAME` map; full names pass through case-insensitively).
pub fn resolve_language(input: &str) -> Option<String> {
    const CODES: &[(&str, &str)] = &[
        ("zh", "Chinese"),
        ("en", "English"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("de", "German"),
        ("es", "Spanish"),
        ("fr", "French"),
        ("it", "Italian"),
        ("pt", "Portuguese"),
        ("ru", "Russian"),
        ("ar", "Arabic"),
        ("hi", "Hindi"),
        ("th", "Thai"),
        ("vi", "Vietnamese"),
        ("id", "Indonesian"),
        ("tr", "Turkish"),
        ("nl", "Dutch"),
        ("pl", "Polish"),
        ("sv", "Swedish"),
        ("da", "Danish"),
        ("fi", "Finnish"),
        ("ms", "Malay"),
        ("yue", "Cantonese"),
        ("cs", "Czech"),
        ("fil", "Filipino"),
        ("fa", "Persian"),
        ("el", "Greek"),
        ("hu", "Hungarian"),
        ("mk", "Macedonian"),
        ("ro", "Romanian"),
    ];
    let lower = input.trim().to_ascii_lowercase();
    for (code, name) in CODES {
        if lower == *code {
            return Some((*name).to_string());
        }
    }
    for (_, name) in CODES {
        if lower == name.to_ascii_lowercase() {
            return Some((*name).to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    fn golden_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden/tokenizer")
    }

    #[derive(Deserialize)]
    struct AsrCase {
        #[serde(default)]
        context: Option<String>,
        #[serde(default)]
        language: Option<String>,
        audio_pad_count: usize,
        prompt_text: String,
        prompt_ids: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct OcrCase {
        user_text: String,
        image_pad_count: usize,
        prompt_text: String,
        prompt_ids: Vec<u32>,
    }

    fn hf_cache() -> PathBuf {
        PathBuf::from(
            std::env::var("AIMD_TEST_HF_CACHE")
                .unwrap_or_else(|_| format!("{}/.cache/huggingface/hub", env!("HOME"))),
        )
    }

    /// First snapshot dir under a cached model repo containing tokenizer.json.
    fn checkpoint_dir(repo_dirname: &str) -> Option<PathBuf> {
        let snapshots = hf_cache().join(repo_dirname).join("snapshots");
        let entries = std::fs::read_dir(snapshots).ok()?;
        for entry in entries.flatten() {
            let dir = entry.path();
            if dir.join("tokenizer.json").is_file() {
                return Some(dir);
            }
        }
        None
    }

    const ASR_REPO: &str = "models--Qwen--Qwen3-ASR-1.7B-hf";
    const OCR_REPO: &str = "models--mlx-community--Unlimited-OCR-4bit";

    fn load_asr_case(name: &str) -> AsrCase {
        let path = golden_root().join("cases").join(format!("{name}.json"));
        let raw = std::fs::read(path).expect("golden fixture present");
        serde_json::from_slice(&raw).expect("fixture schema")
    }

    fn load_ocr_case(name: &str) -> OcrCase {
        let path = golden_root().join("cases").join(format!("{name}.json"));
        let raw = std::fs::read(path).expect("golden fixture present");
        serde_json::from_slice(&raw).expect("fixture schema")
    }

    #[test]
    fn tokenizer_asr_prompt_strings_match_golden() {
        for case in [
            "asr_no_context_no_language",
            "asr_no_context_language_chinese",
            "asr_context_only",
            "asr_context_language_chinese",
            "asr_cjk_context_only",
            "asr_cjk_context_language_japanese",
        ] {
            let c = load_asr_case(case);
            let built = build_asr_prompt(
                c.context.as_deref(),
                c.language.as_deref(),
                c.audio_pad_count,
            );
            assert_eq!(built, c.prompt_text, "string mismatch for {case}");
        }
    }

    #[test]
    fn tokenizer_ocr_prompt_text_matches_golden() {
        for case in [
            "ocr_document_parsing",
            "ocr_document_parsing_multicrop",
            "ocr_cjk_document_parsing",
        ] {
            let c = load_ocr_case(case);
            assert_eq!(build_ocr_prompt_text(&c.user_text), c.prompt_text, "{case}");
        }
    }

    #[test]
    fn tokenizer_language_map() {
        assert_eq!(resolve_language("zh").as_deref(), Some("Chinese"));
        assert_eq!(resolve_language("YUE").as_deref(), Some("Cantonese"));
        assert_eq!(resolve_language("japanese").as_deref(), Some("Japanese"));
        assert_eq!(resolve_language("klingon"), None);
    }

    #[test]
    fn tokenizer_asr_prompt_ids_match_golden() {
        let Some(dir) = checkpoint_dir(ASR_REPO) else {
            eprintln!("skipping: {ASR_REPO} not in HF cache");
            return;
        };
        let tok = Tokenizer::from_checkpoint(&dir).expect("load tokenizer");
        for case in [
            "asr_no_context_no_language",
            "asr_no_context_language_chinese",
            "asr_context_only",
            "asr_context_language_chinese",
            "asr_cjk_context_only",
            "asr_cjk_context_language_japanese",
        ] {
            let c = load_asr_case(case);
            let text = build_asr_prompt(
                c.context.as_deref(),
                c.language.as_deref(),
                c.audio_pad_count,
            );
            let ids = tok.encode(&text).expect("encode");
            assert_eq!(ids, c.prompt_ids, "id mismatch for {case}");
        }
    }

    #[test]
    fn tokenizer_ocr_prompt_ids_match_golden() {
        let Some(dir) = checkpoint_dir(OCR_REPO) else {
            eprintln!("skipping: {OCR_REPO} not in HF cache");
            return;
        };
        let tok = Tokenizer::from_checkpoint(&dir).expect("load tokenizer");
        for case in [
            "ocr_document_parsing",
            "ocr_document_parsing_multicrop",
            "ocr_cjk_document_parsing",
        ] {
            let c = load_ocr_case(case);
            let ids = build_ocr_prompt_ids(&tok, &c.user_text, c.image_pad_count).expect("build");
            assert_eq!(ids, c.prompt_ids, "id mismatch for {case}");
        }
    }

    #[test]
    fn tokenizer_vocab_merges_equivalence() {
        // The MLX 4-bit repo ships vocab.json+merges.txt; the -hf repo ships
        // tokenizer.json. Both must encode identically.
        let Some(hf_dir) = checkpoint_dir(ASR_REPO) else {
            eprintln!("skipping: {ASR_REPO} not in HF cache");
            return;
        };
        let Some(mlx_dir) = checkpoint_dir("models--mlx-community--Qwen3-ASR-1.7B-4bit") else {
            eprintln!("skipping: Qwen3-ASR-1.7B-4bit not in HF cache");
            return;
        };
        let json_tok = Tokenizer::from_checkpoint(&hf_dir).expect("tokenizer.json path");
        let vm_tok = Tokenizer::from_checkpoint(&mlx_dir).expect("vocab+merges path");
        for sample in [
            "<|im_start|>system\n你好，世界！这是上下文工程的一小步。<|im_end|>\n",
            "<|im_start|>assistant\nlanguage Chinese<asr_text>",
            "高频词汇： bluesky、ZCode；话题：上下文工程",
            "mixed 中英 ab旷🎉 emoji と日本語",
            "<|audio_start|><|audio_pad|><|audio_end|>",
        ] {
            let a = json_tok.encode(sample).expect("json encode");
            let b = vm_tok.encode(sample).expect("vocab+merges encode");
            assert_eq!(a, b, "encoding mismatch for {sample:?}");
            let back = vm_tok.decode(&b, false).expect("decode");
            assert_eq!(back, sample);
        }
    }

    #[test]
    fn tokenizer_roundtrip_cjk() {
        let Some(dir) = checkpoint_dir(ASR_REPO) else {
            eprintln!("skipping: {ASR_REPO} not in HF cache");
            return;
        };
        let tok = Tokenizer::from_checkpoint(&dir).expect("load tokenizer");
        for sample in [
            "你好，世界！这是上下文工程的一小步。",
            "bluesky、ZCode；话题：上下文工程",
            "mixed 中英 ab旷🎉 emoji と日本語",
        ] {
            let ids = tok.encode(sample).expect("encode");
            let back = tok.decode(&ids, false).expect("decode");
            assert_eq!(back, sample);
        }
    }
}
