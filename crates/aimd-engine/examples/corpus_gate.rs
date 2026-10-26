//! Golden-corpus accuracy gate (tier 3): runs the native Qwen3-ASR and
//! Unlimited-OCR pipelines over the labeled corpus built by
//! `uv run scripts/build_corpus.py`, scores them against the labels, and
//! compares with the mlx-audio / mlx-vlm baseline written by
//! `uv run scripts/corpus_baseline.py` when `baseline.json` is present.
//!
//! Usage: cargo run --release -p aimd-engine --example corpus_gate -- \
//!            [CORPUS_DIR] [--only asr|ocr] [--asr-repo ID] [--ocr-repo ID]
//!
//! Metrics: ASR WER (English, words) and CER (Chinese, characters) over the
//! whole set; OCR character similarity per document (1 - edit distance /
//! longer length, over alphanumeric/CJK characters, markup dropped) and the
//! per-page `finish=stop` rate. Gates (specs/archive/002-native-engine.md): ASR error <= baseline + 1%
//! absolute per language; OCR mean similarity >= baseline - 2%, stop rate
//! no lower than the baseline. Exits 1 when a gate fails.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use aimd_engine::models::qwen3_asr::pipeline::{Qwen3AsrPipeline, TranscriptionRequest};
use aimd_engine::models::unlimited_ocr::pipeline::{OcrRequest, UnlimitedOcrPipeline};
use serde_json::{Value, json};

fn hub_cache() -> PathBuf {
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    if let Some(dir) = env("HF_HUB_CACHE") {
        return PathBuf::from(dir);
    }
    if let Some(dir) = env("HF_HOME") {
        return PathBuf::from(dir).join("hub");
    }
    PathBuf::from(env("HOME").expect("HOME")).join(".cache/huggingface/hub")
}

fn snapshot(repo: &str) -> PathBuf {
    let snapshots = hub_cache()
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(&snapshots)
        .unwrap_or_else(|e| panic!("{repo} not in the HF cache ({}): {e}", snapshots.display()))
        .flatten()
        .map(|e| e.path())
        .find(|p| p.join("config.json").is_file())
        .unwrap_or_else(|| panic!("{repo}: no complete snapshot"))
}

/// Lowercased alphanumeric/CJK characters; everything else is a separator.
fn units(text: &str, words: bool) -> Vec<String> {
    let cleaned: String = text
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_lowercase().next().unwrap_or(c)
            } else {
                ' '
            }
        })
        .collect();
    if words {
        cleaned.split_whitespace().map(str::to_string).collect()
    } else {
        cleaned
            .chars()
            .filter(|c| !c.is_whitespace())
            .map(String::from)
            .collect()
    }
}

/// OCR text without HTML tags (tables come back as `<table>` markup).
fn strip_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_tag = false;
    for c in text.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => {
                in_tag = false;
                out.push(' ');
            }
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

fn edit_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, x) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(x != y);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn similarity(reference: &str, hypothesis: &str) -> f64 {
    let r = units(&strip_tags(reference), false);
    let h = units(&strip_tags(hypothesis), false);
    let longest = r.len().max(h.len());
    if longest == 0 {
        return 1.0;
    }
    1.0 - edit_distance(&r, &h) as f64 / longest as f64
}

fn run_asr(corpus: &Path, items: &[Value], repo: &str) -> Value {
    let pipe = Qwen3AsrPipeline::load(&snapshot(repo)).expect("ASR pipeline");
    let mut out = serde_json::Map::new();
    for item in items {
        let id = item["id"].as_str().unwrap();
        let audio = corpus.join(item["audio"].as_str().unwrap());
        let text = pipe
            .transcribe(
                &audio,
                TranscriptionRequest {
                    language: item["language"].as_str(),
                    context: None,
                },
            )
            .unwrap_or_else(|e| panic!("{id}: {e}"));
        out.insert(id.to_string(), Value::String(text));
    }
    Value::Object(out)
}

fn run_ocr(corpus: &Path, items: &[Value], repo: &str) -> Value {
    let pipe = UnlimitedOcrPipeline::load(&snapshot(repo)).expect("OCR pipeline");
    let mut out = serde_json::Map::new();
    for item in items {
        let id = item["id"].as_str().unwrap();
        let pages: Vec<Value> = item["pages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|page| {
                let path = corpus.join(page.as_str().unwrap());
                let result = pipe
                    .recognize_path(&path, OcrRequest::default())
                    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                json!({
                    "markdown": result.markdown,
                    "finish": result.finish_reason.to_string(),
                })
            })
            .collect();
        out.insert(id.to_string(), json!({ "pages": pages }));
    }
    Value::Object(out)
}

/// Per-language (edit units, reference units) over the set.
fn asr_errors(items: &[Value], hyps: &Value) -> BTreeMap<String, (usize, usize)> {
    let mut totals: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for item in items {
        let language = item["language"].as_str().unwrap();
        let words = language == "en";
        let reference = units(item["reference"].as_str().unwrap(), words);
        let hyp = units(
            hyps[item["id"].as_str().unwrap()].as_str().unwrap_or(""),
            words,
        );
        let entry = totals.entry(language.to_string()).or_default();
        entry.0 += edit_distance(&reference, &hyp);
        entry.1 += reference.len();
    }
    totals
}

/// (mean document similarity, stop pages, pages).
fn ocr_scores(items: &[Value], hyps: &Value) -> (f64, usize, usize, Vec<(String, f64)>) {
    let (mut stops, mut pages, mut per_doc) = (0, 0, Vec::new());
    for item in items {
        let id = item["id"].as_str().unwrap();
        let doc = hyps[id]["pages"].as_array().cloned().unwrap_or_default();
        pages += doc.len();
        stops += doc.iter().filter(|p| p["finish"] == "stop").count();
        let text: Vec<&str> = doc.iter().filter_map(|p| p["markdown"].as_str()).collect();
        per_doc.push((
            id.to_string(),
            similarity(item["reference"].as_str().unwrap(), &text.join("\n\n")),
        ));
    }
    let mean = per_doc.iter().map(|(_, s)| s).sum::<f64>() / per_doc.len().max(1) as f64;
    (mean, stops, pages, per_doc)
}

fn rate((edits, total): (usize, usize)) -> f64 {
    edits as f64 / total.max(1) as f64
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut flag = |name: &str| {
        let i = args.iter().position(|a| a == name)?;
        let value = args.get(i + 1).cloned().expect("flag value");
        args.drain(i..i + 2);
        Some(value)
    };
    let only = flag("--only");
    let asr_repo = flag("--asr-repo").unwrap_or("mlx-community/Qwen3-ASR-1.7B-4bit".into());
    let ocr_repo = flag("--ocr-repo").unwrap_or("mlx-community/Unlimited-OCR-4bit".into());
    let corpus = PathBuf::from(args.first().map_or("target/corpus", String::as_str));
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(corpus.join("manifest.json"))
            .expect("manifest.json (run `uv run scripts/build_corpus.py` first)"),
    )
    .expect("manifest JSON");
    let baseline: Option<Value> = std::fs::read_to_string(corpus.join("baseline.json"))
        .ok()
        .map(|s| serde_json::from_str(&s).expect("baseline JSON"));
    let run = |kind: &str| only.as_deref().is_none_or(|o| o == kind);
    let mut native = serde_json::Map::new();
    let mut failed = false;

    if run("asr") {
        let items = manifest["asr"].as_array().unwrap();
        let hyps = run_asr(&corpus, items, &asr_repo);
        println!("ASR {asr_repo}: {} clips", items.len());
        let ours = asr_errors(items, &hyps);
        let base = baseline.as_ref().map(|b| asr_errors(items, &b["asr"]));
        for (language, counts) in &ours {
            let metric = if language == "en" { "WER" } else { "CER" };
            let mut line = format!("  {language} {metric} native {:.2}%", 100.0 * rate(*counts));
            if let Some(base) = &base {
                let b = rate(base[language]);
                let pass = rate(*counts) <= b + 0.01;
                failed |= !pass;
                line += &format!(
                    "  baseline {:.2}%  gate <= +1.00%: {}",
                    100.0 * b,
                    if pass { "PASS" } else { "FAIL" }
                );
            }
            println!("{line}");
        }
        if let Some(b) = &baseline {
            let same = items
                .iter()
                .filter(|i| {
                    let id = i["id"].as_str().unwrap();
                    hyps[id] == b["asr"][id]
                })
                .count();
            println!("  identical to baseline: {same}/{}", items.len());
        }
        native.insert("asr".into(), hyps);
    }

    if run("ocr") {
        let items = manifest["ocr"].as_array().unwrap();
        let hyps = run_ocr(&corpus, items, &ocr_repo);
        let (mean, stops, pages, per_doc) = ocr_scores(items, &hyps);
        println!("OCR {ocr_repo}: {} documents, {pages} pages", items.len());
        let base = baseline.as_ref().map(|b| ocr_scores(items, &b["ocr"]));
        for (i, (id, sim)) in per_doc.iter().enumerate() {
            let b = base
                .as_ref()
                .map_or(String::new(), |b| format!("  baseline {:.4}", b.3[i].1));
            println!("  {id:<10} similarity {sim:.4}{b}");
        }
        let mut line = format!("  mean similarity {mean:.4}, finish=stop {stops}/{pages}");
        if let Some((bmean, bstops, bpages, _)) = base {
            let pass = mean >= bmean - 0.02 && stops * bpages >= bstops * pages;
            failed |= !pass;
            line += &format!(
                "  baseline {bmean:.4}, stop {bstops}/{bpages}  gate: {}",
                if pass { "PASS" } else { "FAIL" }
            );
        }
        println!("{line}");
        if let Some(b) = &baseline {
            let same = items
                .iter()
                .map(|i| {
                    let id = i["id"].as_str().unwrap();
                    let page = |v: &Value, k: usize| v[id]["pages"][k]["markdown"].clone();
                    (0..i["pages"].as_array().unwrap().len())
                        .filter(|&k| page(&hyps, k) == page(&b["ocr"], k))
                        .count()
                })
                .sum::<usize>();
            println!("  pages identical to baseline: {same}/{pages}");
        }
        native.insert("ocr".into(), hyps);
    }

    let path = corpus.join("native.json");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&Value::Object(native)).unwrap(),
    )
    .expect("write native.json");
    println!("wrote {}", path.display());
    if baseline.is_none() {
        println!("no baseline.json: run `uv run scripts/corpus_baseline.py` to gate");
    }
    if failed {
        std::process::exit(1);
    }
}
