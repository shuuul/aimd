//! Unlimited-OCR native MLX benchmark: checkpoint load + per-image
//! recognition with stage timings (preprocess, vision, prefill, decode
//! tok/s), printing the normalized markdown of the first round.
//!
//! Usage: cargo run --release -p aimd-engine --example bench_ocr -- \
//!            [--rounds N] [--quiet] image.png [more...]
//!
//! Round 1 includes Metal kernel compilation / first-touch costs; later
//! rounds are steady state (compare against mlx-vlm's
//! `generate(..., verbose=True)` prompt/generation tok/s).

use std::path::PathBuf;
use std::time::Instant;

use aimd_engine::models::unlimited_ocr::pipeline::{OcrRequest, UnlimitedOcrPipeline};

fn checkpoint_dir() -> Option<PathBuf> {
    let cache = std::env::var("AIMD_TEST_HF_CACHE")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub")))
        .ok()?;
    let snapshots = cache
        .join("models--mlx-community--Unlimited-OCR-4bit")
        .join("snapshots");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file() && p.join("model.safetensors").is_file())
        .collect();
    dirs.sort();
    dirs.into_iter().next()
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut rounds = 1usize;
    if let Some(i) = args.iter().position(|a| a == "--rounds") {
        rounds = args
            .get(i + 1)
            .and_then(|v| v.parse().ok())
            .expect("--rounds N");
        args.drain(i..i + 2);
    }
    let quiet = if let Some(i) = args.iter().position(|a| a == "--quiet") {
        args.remove(i);
        true
    } else {
        false
    };
    if args.is_empty() {
        eprintln!("usage: bench_ocr [--rounds N] [--quiet] <image>...");
        std::process::exit(2);
    }
    let dir = checkpoint_dir().expect("Unlimited-OCR-4bit snapshot in HF cache");

    let t0 = Instant::now();
    let pipe = UnlimitedOcrPipeline::load(&dir).expect("pipeline");
    println!(
        "pipeline load: {:>6.2} s  (safetensors mmap -> MLX arrays, tokenizer)",
        t0.elapsed().as_secs_f32()
    );

    for image in &args {
        for round in 0..rounds {
            let t = Instant::now();
            match pipe.recognize_path(&PathBuf::from(image), OcrRequest::default()) {
                Ok(out) => {
                    let total = t.elapsed().as_secs_f64();
                    let tm = out.timings;
                    let prefill = tm.prefill.as_secs_f64();
                    let decode = tm.decode.as_secs_f64();
                    // The first sampled token closes the prefill window.
                    let decode_tok = out.tokens.len().saturating_sub(1);
                    println!(
                        "{image} round{}: total {total:.2} s | preprocess {:.3} s, vision {:.3} s, \
                         prefill {prefill:.3} s ({} tok, {:.0} tok/s), decode {decode:.3} s \
                         ({} tok, {:.1} tok/s) | crops {:?}, image tokens {}, finish={}",
                        round + 1,
                        tm.preprocess.as_secs_f64(),
                        tm.vision.as_secs_f64(),
                        out.prompt_tokens,
                        out.prompt_tokens as f64 / prefill.max(1e-9),
                        decode_tok,
                        decode_tok as f64 / decode.max(1e-9),
                        out.crop_ratio,
                        out.image_tokens,
                        out.finish_reason,
                    );
                    if round == 0 && !quiet {
                        println!(
                            "----- markdown -----\n{}\n--------------------",
                            out.markdown
                        );
                    }
                }
                Err(e) => println!("{image} round{}: <error: {e}>", round + 1),
            }
        }
    }
}
