//! Host-side forced-aligner processor: unit splitting, prompt string, and
//! timestamp repair.
//!
//! Port of the official `Qwen3ForceAlignProcessor` (qwen-asr 0.0.6), locked
//! by `tests/golden/aligner/{units,fix_timestamp}.json`:
//!
//! - Every language except Korean (Chinese and Cantonese included) uses
//!   `tokenize_space_lang`: split on whitespace, keep letters, numbers and
//!   `'` of each piece (`clean_token`), then split the cleaned piece so every
//!   CJK ideograph is its own unit and the runs between them stay whole.
//! - Korean uses soynlp `LTokenizer(scores=korean_dict)`: each
//!   whitespace-separated eojeol longer than two characters splits into its
//!   longest dictionary prefix (length >= 2) and the rest, or stays whole
//!   when no prefix is in the dictionary; both parts are then cleaned. `korean_dict.txt` holds the words of
//!   length >= 2 from `korean_dict_jieba.dict` (written by
//!   `scripts/export_aligner_reference.py`).
//! - `is_kept_char` keeps Unicode categories L* and N* plus `'`. Rust's
//!   `is_alphabetic() || is_numeric()` is that set plus `Other_Alphabetic`
//!   (dependent vowel signs of Indic/SE Asian scripts, circled letters),
//!   none of which occur in the aligner's languages.

use std::collections::HashSet;
use std::sync::OnceLock;

use super::pipeline::AlignedUnit;

const KOREAN_DICT: &str = include_str!("korean_dict.txt");

fn korean_dict() -> &'static HashSet<&'static str> {
    static DICT: OnceLock<HashSet<&'static str>> = OnceLock::new();
    DICT.get_or_init(|| KOREAN_DICT.lines().filter(|l| !l.is_empty()).collect())
}

/// `is_kept_char`: letters, numbers, and the apostrophe.
fn is_kept_char(ch: char) -> bool {
    ch == '\'' || ch.is_alphabetic() || ch.is_numeric()
}

fn clean_token(token: &str) -> String {
    token.chars().filter(|&c| is_kept_char(c)).collect()
}

/// `is_cjk_char`: CJK Unified Ideographs (+ extensions A-E) and the
/// compatibility block.
fn is_cjk_char(ch: char) -> bool {
    matches!(
        ch as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0xF900..=0xFAFF
    )
}

/// Python `str.split()` separators: `White_Space` plus U+001C..U+001F.
fn is_py_space(ch: char) -> bool {
    ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch)
}

fn py_split(text: &str) -> impl Iterator<Item = &str> {
    text.split(is_py_space).filter(|s| !s.is_empty())
}

/// `split_segment_with_chinese`: one unit per CJK ideograph, other runs kept.
fn split_segment_with_chinese(seg: &str, out: &mut Vec<String>) {
    let mut buf = String::new();
    for ch in seg.chars() {
        if is_cjk_char(ch) {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            out.push(ch.to_string());
        } else {
            buf.push(ch);
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
}

fn tokenize_space_lang(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for seg in py_split(text) {
        let cleaned = clean_token(seg);
        if !cleaned.is_empty() {
            split_segment_with_chinese(&cleaned, &mut tokens);
        }
    }
    tokens
}

/// soynlp `LTokenizer.tokenize` (tolerance 0, flatten) followed by
/// `clean_token`.
fn tokenize_korean(text: &str) -> Vec<String> {
    let dict = korean_dict();
    let mut tokens = Vec::new();
    for eojeol in py_split(text) {
        let chars: Vec<char> = eojeol.chars().collect();
        // Candidates e = 2..=len sorted descending by (score, len(l)): the
        // longest dictionary prefix wins, else the whole eojeol (e = len).
        let split = if chars.len() <= 2 {
            chars.len()
        } else {
            (2..=chars.len())
                .rev()
                .find(|&e| dict.contains(chars[..e].iter().collect::<String>().as_str()))
                .unwrap_or(chars.len())
        };
        for part in [&chars[..split], &chars[split..]] {
            let cleaned: String = part.iter().copied().filter(|&c| is_kept_char(c)).collect();
            if !cleaned.is_empty() {
                tokens.push(cleaned);
            }
        }
    }
    tokens
}

/// Split `text` into alignment units for `language` (a canonical name from
/// [`super::pipeline::ALIGNER_LANGUAGES`]; anything but `Korean` takes the
/// whitespace path).
pub fn split_units(text: &str, language: &str) -> Vec<String> {
    if language.eq_ignore_ascii_case("korean") {
        tokenize_korean(text)
    } else {
        tokenize_space_lang(text)
    }
}

/// The aligner input string: audio block, then every unit followed by two
/// `<timestamp>` slots (start, end).
pub fn build_aligner_prompt(units: &[String], audio_pad_count: usize) -> String {
    let mut s = String::with_capacity(
        40 + audio_pad_count * 13 + units.iter().map(|u| u.len() + 22).sum::<usize>(),
    );
    s.push_str("<|audio_start|>");
    for _ in 0..audio_pad_count {
        s.push_str("<|audio_pad|>");
    }
    s.push_str("<|audio_end|>");
    for unit in units {
        s.push_str(unit);
        s.push_str("<timestamp><timestamp>");
    }
    s
}

/// `fix_timestamp`: keep the longest non-decreasing subsequence (first
/// maximal end, first maximal predecessor), replace runs of <= 2 outliers by
/// the nearer kept neighbour (ties to the left) and longer runs by linear
/// interpolation between the kept neighbours, then truncate to integers.
pub fn fix_timestamps(data: &[f64]) -> Vec<i64> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let mut dp = vec![1usize; n];
    let mut parent: Vec<Option<usize>> = vec![None; n];
    for i in 1..n {
        for j in 0..i {
            if data[j] <= data[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = Some(j);
            }
        }
    }
    let max_len = *dp.iter().max().expect("non-empty");
    let mut idx = dp.iter().position(|&d| d == max_len);
    let mut is_normal = vec![false; n];
    while let Some(i) = idx {
        is_normal[i] = true;
        idx = parent[i];
    }

    let mut result = data.to_vec();
    let mut i = 0;
    while i < n {
        if is_normal[i] {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && !is_normal[j] {
            j += 1;
        }
        let left = (0..i).rev().find(|&k| is_normal[k]).map(|k| result[k]);
        let right = (j..n).find(|&k| is_normal[k]).map(|k| result[k]);
        let count = j - i;
        for (offset, slot) in result[i..j].iter_mut().enumerate() {
            let k = i + offset;
            *slot = match (left, right) {
                (None, Some(r)) => r,
                (Some(l), None) => l,
                (Some(l), Some(r)) if count <= 2 => {
                    // `k - (i - 1) <= j - k`, with i - 1 possibly -1.
                    if k + 1 - i <= j - k { l } else { r }
                }
                (Some(l), Some(r)) => l + (r - l) / (count + 1) as f64 * (k - i + 1) as f64,
                (None, None) => *slot,
            };
        }
        i = j;
    }
    result.into_iter().map(|v| v as i64).collect()
}

/// Turn the predicted time classes of the `2 * units.len()` timestamp slots
/// into units: `class * segment_ms` milliseconds, [`fix_timestamps`], then
/// seconds (the reference's `round(ms / 1000, 3)` is exact for integer ms).
pub fn decode_timestamps(units: &[String], classes: &[u32], segment_ms: f64) -> Vec<AlignedUnit> {
    debug_assert_eq!(classes.len(), units.len() * 2);
    let raw: Vec<f64> = classes.iter().map(|&c| f64::from(c) * segment_ms).collect();
    let fixed = fix_timestamps(&raw);
    units
        .iter()
        .enumerate()
        .map(|(i, text)| AlignedUnit {
            text: text.clone(),
            start: (fixed[2 * i] as f64 / 1000.0) as f32,
            end: (fixed[2 * i + 1] as f64 / 1000.0) as f32,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{golden_dir, read_json};
    use super::*;

    #[test]
    fn aligner_units_match_reference() {
        let cases = read_json(&golden_dir().join("units.json"));
        let cases = cases.as_array().expect("units.json is a list");
        assert!(cases.len() >= 20);
        for case in cases {
            let text = case["text"].as_str().unwrap();
            let language = case["language"].as_str().unwrap();
            let expected: Vec<String> = case["units"]
                .as_array()
                .unwrap()
                .iter()
                .map(|u| u.as_str().unwrap().to_string())
                .collect();
            assert_eq!(
                split_units(text, language),
                expected,
                "{language}: {text:?}"
            );
        }
    }

    #[test]
    fn aligner_fix_timestamp_matches_reference() {
        let cases = read_json(&golden_dir().join("fix_timestamp.json"));
        for case in cases.as_array().unwrap() {
            let raw: Vec<f64> = case["raw_ms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let expected: Vec<i64> = case["fixed_ms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            assert_eq!(fix_timestamps(&raw), expected, "raw {raw:?}");
        }
        assert!(fix_timestamps(&[]).is_empty());
    }

    #[test]
    fn aligner_prompt_layout() {
        let units = vec!["The".to_string(), "fox".to_string()];
        assert_eq!(
            build_aligner_prompt(&units, 2),
            "<|audio_start|><|audio_pad|><|audio_pad|><|audio_end|>\
             The<timestamp><timestamp>fox<timestamp><timestamp>"
        );
        assert_eq!(
            build_aligner_prompt(&[], 1),
            "<|audio_start|><|audio_pad|><|audio_end|>"
        );
    }

    #[test]
    fn aligner_decode_timestamps_scales_and_repairs() {
        let units = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        // Class 50 at slot 2 is an outlier: repaired to its left neighbour.
        let out = decode_timestamps(&units, &[0, 1, 50, 3, 4, 5], 80.0);
        let spans: Vec<(f32, f32)> = out.iter().map(|u| (u.start, u.end)).collect();
        assert_eq!(spans, vec![(0.0, 0.08), (0.08, 0.24), (0.32, 0.4)]);
    }
}
