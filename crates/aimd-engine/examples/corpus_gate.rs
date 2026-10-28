//! Golden-corpus accuracy gate (tier 3): runs the native Qwen3-ASR and
//! OCR (Unlimited-OCR, PaddleOCR-VL-1.6, or PP-OCRv6) pipelines over the labeled corpus built by
//! `uv run scripts/build_corpus.py`, scores them against the labels, and
//! compares with the mlx-audio / mlx-vlm baseline written by
//! `uv run scripts/corpus_baseline.py` (Unlimited-OCR) or
//! `uv run scripts/corpus_baseline_paddle.py` (mlx-vlm PaddleOCR-VL, mlx4ocr
//! PP-OCRv6) when `baseline.json` is present.
//!
//! Usage: cargo run --release -p aimd-engine --example corpus_gate -- \
//!            [CORPUS_DIR] [--only asr|ocr] [--asr-repo ID] [--ocr-repo ID]
//!            [--ocr-model unlimited-ocr|unlimited-ocr-2|paddleocr-vl|pp-ocrv6]
//!            [--ocr-bits 4|6|8]
//!
//! `unlimited-ocr-2` sends each document as one multi-page request (as aimd
//! sends a PDF window) and scores it against `ocr_unlimited_ocr2` from
//! `uv run scripts/corpus_baseline_ocr2.py`, whose raw output goes through the
//! same Markdown port; its stop rate counts requests, not pages.
//! `--ocr-bits` applies to paddleocr-vl (load-time quantization of the bf16
//! checkpoint). Each OCR model is scored against its own baseline key
//! (`ocr`, `ocr_paddleocr_vl[_4bit]`, `ocr_pp_ocrv6`). PP-OCRv6 has no
//! generation, so its pages count as `stop` unless empty.
//!
//! Metrics: ASR WER (English, words) and CER (Chinese, characters) over the
//! whole set; OCR character similarity per document (1 - edit distance /
//! longer length, over alphanumeric/CJK characters, markup dropped) and the
//! per-page `finish=stop` rate. Gates (specs/archive/002-native-engine.md): ASR error <= baseline + 1%
//! absolute per language; OCR mean similarity >= baseline - 2%, stop rate
//! no lower than the baseline. Exits 1 when a gate fails.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use aimd_engine::models::paddleocr_vl::generate::FinishReason as VlFinish;
use aimd_engine::models::paddleocr_vl::pipeline::{PaddleOcrVlPipeline, VlRequest};
use aimd_engine::models::pp_ocrv6::pipeline::{PipelineError as PpError, PpOcrV6Pipeline};
use aimd_engine::models::qwen3_asr::pipeline::{Qwen3AsrPipeline, TranscriptionRequest};
use aimd_engine::models::unlimited_ocr::markdown2::render_unlimited_ocr2;
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

/// OCR model under test.
enum Ocr {
    Unlimited(UnlimitedOcrPipeline),
    Unlimited2(UnlimitedOcrPipeline),
    PaddleVl(PaddleOcrVlPipeline),
    PpOcrV6(PpOcrV6Pipeline),
}

impl Ocr {
    /// (markdown, finish reason) for one page.
    fn page(&self, path: &Path) -> (String, String) {
        fn fail<T>(path: &Path, e: &dyn std::fmt::Display) -> T {
            panic!("{}: {e}", path.display())
        }
        match self {
            Self::Unlimited(p) | Self::Unlimited2(p) => {
                let r = p
                    .recognize_path(path, OcrRequest::default())
                    .unwrap_or_else(|e| fail(path, &e));
                (r.markdown, r.finish_reason.to_string())
            }
            Self::PaddleVl(p) => {
                let (r, stats) = p
                    .recognize_path_with_stats(path, VlRequest::default())
                    .unwrap_or_else(|e| fail(path, &e));
                let finish = match stats.finish_reason {
                    VlFinish::Stop => "stop",
                    VlFinish::Length => "length",
                };
                (r.markdown, finish.to_string())
            }
            Self::PpOcrV6(p) => match p.recognize_path(path) {
                Ok(r) => (r.markdown.trim_end().to_string(), "stop".to_string()),
                Err(PpError::EmptyContent) => (String::new(), "empty".to_string()),
                Err(e) => fail(path, &e),
            },
        }
    }
}

fn run_ocr(corpus: &Path, items: &[Value], ocr: &Ocr) -> Value {
    let mut out = serde_json::Map::new();
    for item in items {
        let id = item["id"].as_str().unwrap();
        if let Ocr::Unlimited2(p) = ocr {
            let bytes: Vec<Vec<u8>> = item["pages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|page| std::fs::read(corpus.join(page.as_str().unwrap())).expect("page"))
                .collect();
            let refs: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
            let r = p
                .recognize_pages(&refs, OcrRequest::default())
                .unwrap_or_else(|e| panic!("{id}: {e}"));
            let pages = json!([{ "markdown": r.markdown, "finish": r.finish_reason.to_string() }]);
            out.insert(id.to_string(), json!({ "pages": pages }));
            continue;
        }
        let pages: Vec<Value> = item["pages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|page| {
                let (markdown, finish) = ocr.page(&corpus.join(page.as_str().unwrap()));
                json!({ "markdown": markdown, "finish": finish })
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

/// `baseline` with `ocr_unlimited_ocr2` documents (`{raw, finish}`) turned
/// into the one-request page list `ocr_scores` reads.
fn raw_baseline(baseline: &Value, key: &str) -> Value {
    let mut b = baseline.clone();
    if key == "ocr_unlimited_ocr2" {
        for doc in b[key].as_object_mut().unwrap().values_mut() {
            let markdown = render_unlimited_ocr2(doc["raw"].as_str().unwrap_or("")).markdown;
            *doc = json!({ "pages": [{ "markdown": markdown, "finish": doc["finish"] }] });
        }
    }
    b
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
    let ocr_model = flag("--ocr-model").unwrap_or("unlimited-ocr".into());
    // `--ocr-repo` names the Unlimited-OCR-2 checkpoint too when it is one.
    let flag_ocr2_repo = if ocr_repo.contains("Unlimited-OCR-2") {
        ocr_repo.clone()
    } else {
        "mlx-community/Unlimited-OCR-2-4bit".to_string()
    };
    let ocr_bits: Option<u32> = flag("--ocr-bits").map(|b| b.parse().expect("--ocr-bits"));
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
        let (label, key, ocr) = match ocr_model.as_str() {
            "unlimited-ocr" => (
                ocr_repo.clone(),
                "ocr".to_string(),
                Ocr::Unlimited(UnlimitedOcrPipeline::load(&snapshot(&ocr_repo)).expect("OCR")),
            ),
            "unlimited-ocr-2" => {
                let repo = flag_ocr2_repo.clone();
                (
                    repo.clone(),
                    "ocr_unlimited_ocr2".to_string(),
                    Ocr::Unlimited2(UnlimitedOcrPipeline::load(&snapshot(&repo)).expect("OCR-2")),
                )
            }
            "paddleocr-vl" => {
                let repo = "PaddlePaddle/PaddleOCR-VL-1.6";
                let pipe = PaddleOcrVlPipeline::load(&snapshot(repo), ocr_bits).expect("VL");
                let suffix = ocr_bits.map_or(String::new(), |b| format!("_{b}bit"));
                let label = format!(
                    "{repo}{}",
                    ocr_bits.map_or(String::new(), |b| format!(":{b}bit"))
                );
                (
                    label,
                    format!("ocr_paddleocr_vl{suffix}"),
                    Ocr::PaddleVl(pipe),
                )
            }
            "pp-ocrv6" => {
                let det = snapshot("PaddlePaddle/PP-OCRv6_medium_det_safetensors");
                let rec = snapshot("PaddlePaddle/PP-OCRv6_medium_rec_safetensors");
                let pipe = PpOcrV6Pipeline::load(&det, &rec).expect("PP-OCRv6");
                (
                    "PaddlePaddle/PP-OCRv6_medium".into(),
                    "ocr_pp_ocrv6".into(),
                    Ocr::PpOcrV6(pipe),
                )
            }
            other => panic!("unknown --ocr-model {other}"),
        };
        let hyps = run_ocr(&corpus, items, &ocr);
        let (mean, stops, pages, per_doc) = ocr_scores(items, &hyps);
        println!("OCR {label}: {} documents, {pages} pages", items.len());
        let baseline = baseline
            .as_ref()
            .filter(|b| b.get(&key).is_some())
            .map(|b| raw_baseline(b, &key));
        let baseline = baseline.as_ref();
        if baseline.is_none() {
            println!("  no baseline key {key}");
        }
        let base = baseline.map(|b| ocr_scores(items, &b[&key]));
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
        if let Some(b) = baseline {
            let same = items
                .iter()
                .map(|i| {
                    let id = i["id"].as_str().unwrap();
                    let page = |v: &Value, k: usize| v[id]["pages"][k]["markdown"].clone();
                    (0..hyps[id]["pages"].as_array().map_or(0, Vec::len))
                        .filter(|&k| page(&hyps, k) == page(&b[&key], k))
                        .count()
                })
                .sum::<usize>();
            println!("  pages identical to baseline: {same}/{pages}");
        }
        native.insert(key, hyps);
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
