//! Qwen3-ASR native MLX/Metal benchmark: startup (load) + warm transcription.
//!
//! Usage: cargo run --release -p aimd-engine --features mlx --example bench_asr_mlx -- /path/to.wav [more.wav...]

use std::path::PathBuf;
use std::time::Instant;

use aimd_engine::models::qwen3_asr::mlx_backend::MlxAsrPipeline;
use aimd_engine::models::qwen3_asr::pipeline::TranscriptionRequest;
use aimd_engine::tokenizer::Tokenizer;

fn checkpoint_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join("models--mlx-community--Qwen3-ASR-1.7B-4bit")
        .join("snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.join("config.json").is_file())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bench_asr_mlx <wav>...");
        std::process::exit(2);
    }
    let dir = checkpoint_dir().expect("Qwen3-ASR-1.7B-4bit snapshot in HF cache");

    let t0 = Instant::now();
    let tok = Tokenizer::from_checkpoint(&dir).expect("tokenizer");
    println!(
        "tokenizer load:        {:>8.2} s",
        t0.elapsed().as_secs_f32()
    );

    let t1 = Instant::now();
    let pipe = MlxAsrPipeline::load(&dir).expect("pipeline");
    let load_s = t1.elapsed().as_secs_f32();
    println!(
        "pipeline load (total): {:>8.2} s  (weights mmap + bit-exact MLX import, quantized matmul)",
        load_s
    );

    for wav in &args {
        let path = PathBuf::from(wav);
        let dur = wav_duration(&path).unwrap_or(0.0);
        for round in 0..3 {
            let t = Instant::now();
            let text = pipe
                .transcribe(
                    &path,
                    TranscriptionRequest {
                        language: None,
                        context: None,
                    },
                )
                .unwrap_or_else(|e| format!("<error: {e}>"));
            let elapsed = t.elapsed().as_secs_f64();
            let rtf = if dur > 0.0 { Some(dur / elapsed) } else { None };
            println!(
                "  {wav} round{}: {:>7.2} s  (audio {dur:.2}s, {}x realtime)  {text}",
                round + 1,
                elapsed,
                rtf.map(|r| format!("{r:.1}"))
                    .unwrap_or_else(|| "n/a".into()),
            );
        }
    }
    let _ = tok;
}

/// Duration from the wav header via our own parser (fmt/data sizes).
fn wav_duration(path: &std::path::Path) -> Option<f64> {
    let audio = aimd_engine::audio::load_wav(path).ok()?;
    Some(audio.samples.len() as f64 / audio.sample_rate as f64)
}
