//! PaddleOCR-VL-1.6 native MLX benchmark: checkpoint load (+ load-time
//! quantization) and per-image recognition with prefill / decode tok/s.
//!
//! Usage: cargo run --release -p aimd-engine --example bench_paddleocr_vl -- \
//!            [--bits 4|6|8] [--rounds N] [--quiet] page.png [more...]
//!
//! Without `--bits` the dense bf16 checkpoint runs. Round 1 includes Metal
//! kernel compilation / first-touch costs; later rounds are steady state
//! (compare against mlx-vlm's `generate(..., verbose=True)`).

use std::path::PathBuf;
use std::time::Instant;

use aimd_engine::models::paddleocr_vl::pipeline::{PaddleOcrVlPipeline, VlRequest};

fn checkpoint_dir() -> Option<PathBuf> {
    let cache = std::env::var("AIMD_TEST_HF_CACHE")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub")))
        .ok()?;
    let snapshots = cache
        .join("models--PaddlePaddle--PaddleOCR-VL-1.6")
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

fn take_value(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let i = args.iter().position(|a| a == flag)?;
    let v = args
        .get(i + 1)
        .cloned()
        .unwrap_or_else(|| panic!("{flag} needs a value"));
    args.drain(i..i + 2);
    Some(v)
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let rounds: usize =
        take_value(&mut args, "--rounds").map_or(1, |v| v.parse().expect("--rounds N"));
    let bits: Option<u32> =
        take_value(&mut args, "--bits").map(|v| v.parse().expect("--bits 4|6|8"));
    let quiet = if let Some(i) = args.iter().position(|a| a == "--quiet") {
        args.remove(i);
        true
    } else {
        false
    };
    if args.is_empty() {
        eprintln!("usage: bench_paddleocr_vl [--bits 4|6|8] [--rounds N] [--quiet] <image>...");
        std::process::exit(2);
    }
    let dir = checkpoint_dir().expect("PaddlePaddle/PaddleOCR-VL-1.6 snapshot in HF cache");

    let t0 = Instant::now();
    let pipe = PaddleOcrVlPipeline::load(&dir, bits).expect("pipeline");
    println!(
        "pipeline load ({}): {:.2} s",
        bits.map_or("bf16".to_string(), |b| format!("{b}-bit")),
        t0.elapsed().as_secs_f32()
    );

    for image in &args {
        for round in 0..rounds {
            let t = Instant::now();
            match pipe.recognize_path_with_stats(&PathBuf::from(image), VlRequest::default()) {
                Ok((out, st)) => {
                    let total = t.elapsed().as_secs_f64();
                    let prefill = st.prefill.as_secs_f64();
                    let decode = st.decode.as_secs_f64();
                    let decode_tok = out.tokens.saturating_sub(1);
                    println!(
                        "{image} round{}: total {total:.2} s | prefill {prefill:.3} s ({} tok, \
                         {:.0} tok/s; {} image tokens), decode {decode:.3} s ({} tok, {:.1} tok/s), \
                         finish={:?}",
                        round + 1,
                        st.prompt_tokens,
                        st.prompt_tokens as f64 / prefill.max(1e-9),
                        st.image_tokens,
                        decode_tok,
                        decode_tok as f64 / decode.max(1e-9),
                        st.finish_reason,
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
