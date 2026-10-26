//! Qwen3-ForcedAligner parity and latency against the mlx-audio reference.
//!
//! ```bash
//! uv run scripts/export_aligner_reference.py --corpus --corpus-dir target/corpus
//! cargo run --release -p aimd-engine --example aligner_parity [target/aligner/reference.json]
//! ```
//!
//! Every clip's unit texts must match exactly. The single corpus clips (short
//! real speech) must also keep the mean absolute start/end error <= 80 ms.
//! The 60 s / 300 s concatenations (`*_60s`, `*_300s`) are latency probes and
//! report their error without gating it: the native tower matches the
//! reference to ~1e-5 in f32, which flips the bf16 rounding of a few merged
//! audio embeddings, and the bf16 decoder amplifies that over thousands of
//! tokens of repeated corpus sentences. The reference shows the same spread
//! against itself under 2e-6 tower noise (zh_300s 55-204 ms over 4 seeds).
//! Prints per-clip error and native vs reference latency (one warm-up run
//! first).

use std::path::PathBuf;
use std::time::Instant;

use aimd_engine::models::qwen3_aligner::pipeline::ForcedAlignerPipeline;
use serde_json::Value;

const MAE_LIMIT_S: f64 = 0.080;

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

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/aligner/reference.json"));
    let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let reference: Value = serde_json::from_slice(&raw).expect("reference json");
    let repo = reference["checkpoint"].as_str().expect("checkpoint");
    let clips = reference["clips"].as_array().expect("clips");

    let t0 = Instant::now();
    let pipeline = ForcedAlignerPipeline::load(&snapshot(repo)).expect("load aligner");
    println!("loaded {repo} in {:.2}s", t0.elapsed().as_secs_f64());

    let mut failures = Vec::new();
    let (mut abs_sum, mut abs_n, mut exact_clips) = (0.0f64, 0usize, 0usize);
    let (mut gated_sum, mut gated_n) = (0.0f64, 0usize);
    let mut warmed = false;
    for clip in clips {
        let id = clip["id"].as_str().unwrap();
        let wav = PathBuf::from(clip["audio"].as_str().unwrap());
        let text = clip["text"].as_str().unwrap();
        let language = clip["language"].as_str().unwrap();
        if !warmed {
            pipeline.align_path(&wav, text, language).expect("warm-up");
            warmed = true;
        }
        let t = Instant::now();
        let got = pipeline
            .align_path(&wav, text, language)
            .unwrap_or_else(|e| panic!("{id}: {e}"));
        let latency = t.elapsed().as_secs_f64();
        let want = clip["items"].as_array().unwrap();
        let texts_match = got.len() == want.len()
            && got
                .iter()
                .zip(want)
                .all(|(g, w)| g.text == w["text"].as_str().unwrap());
        if !texts_match {
            failures.push(format!("{id}: unit texts differ"));
            println!(
                "{id}: UNIT MISMATCH ({} vs {} units)",
                got.len(),
                want.len()
            );
            continue;
        }
        let mut clip_sum = 0.0f64;
        let mut clip_max = 0.0f64;
        for (g, w) in got.iter().zip(want) {
            for (a, b) in [(g.start, &w["start"]), (g.end, &w["end"])] {
                let err = (f64::from(a) - b.as_f64().unwrap()).abs();
                clip_sum += err;
                clip_max = clip_max.max(err);
            }
        }
        let n = 2 * got.len();
        let mae = if n == 0 { 0.0 } else { clip_sum / n as f64 };
        abs_sum += clip_sum;
        abs_n += n;
        if clip_max < 1e-6 {
            exact_clips += 1;
        }
        let gated = !id.contains('_');
        if gated && mae > MAE_LIMIT_S {
            failures.push(format!("{id}: MAE {:.1} ms", mae * 1000.0));
        }
        let ref_latency = clip["latency_s"].as_f64().unwrap_or(f64::NAN);
        println!(
            "{id:>10}: {:>5} units  mae {:>6.1} ms  max {:>6.1} ms  native {:.3}s  ref {:.3}s{}",
            got.len(),
            mae * 1000.0,
            clip_max * 1000.0,
            latency,
            ref_latency,
            if gated {
                ""
            } else {
                "  (latency probe, not gated)"
            },
        );
        if gated {
            gated_sum += clip_sum;
            gated_n += n;
        }
    }
    let mean = |sum: f64, n: usize| if n == 0 { 0.0 } else { sum / n as f64 * 1000.0 };
    println!(
        "all: {} clips, {exact_clips} exact, mae {:.2} ms over {abs_n} timestamps; \
         gated clips: mae {:.2} ms over {gated_n} timestamps",
        clips.len(),
        mean(abs_sum, abs_n),
        mean(gated_sum, gated_n),
    );
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("FAIL {f}");
        }
        std::process::exit(1);
    }
    println!("PASS");
}
