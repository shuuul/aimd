//! Qwen3-ASR audio-tower (encoder segment) CPU benchmark — ANE spike WS-12b.
//!
//! Times the existing native CPU `AudioTower::forward` on the same mel
//! fixture the Core ML/ANE and MLX benchmarks use
//! (scripts/ane_spike/make_fixture.py -> mel_tm.f32, row-major
//! [frames, 128] time-major f32). Weights load from the mlx-community
//! 4-bit snapshot's unquantized bf16 `audio_tower.*` tensors, exactly like
//! examples/bench_asr.rs.
//!
//! Usage (from native/):
//!   CARGO_TARGET_DIR=$PWD/target-ane cargo run --release -p aimd-engine \
//!     --example bench_tower -- /path/to/mel_tm.f32

use std::path::PathBuf;
use std::time::Instant;

use aimd_engine::models::qwen3_asr::config::parse_config;
use aimd_engine::models::qwen3_asr::encoder::{AudioTower, frames_to_audio_pads};
use aimd_weights::load_checkpoint;

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
    if args.len() != 1 {
        eprintln!("usage: bench_tower <mel_tm.f32>");
        std::process::exit(2);
    }
    let dir = checkpoint_dir().expect("Qwen3-ASR-1.7B-4bit snapshot in HF cache");

    let t0 = Instant::now();
    let store = load_checkpoint(&dir).expect("checkpoint");
    let raw = std::fs::read(dir.join("config.json")).expect("config.json");
    let cfg = parse_config(&raw).expect("config parse");
    let n_mels = cfg.thinker_config.audio_config.num_mel_bins;
    let tower = AudioTower::from_weight_store(&cfg.thinker_config.audio_config, &store)
        .expect("audio tower");
    println!(
        "tower load (weights bf16 verbatim): {:>8.2} s",
        t0.elapsed().as_secs_f32()
    );

    // Fixture: raw little-endian f32, row-major [frames, n_mels] (time-major).
    let bytes = std::fs::read(&args[0]).expect("fixture file");
    assert_eq!(bytes.len() % 4, 0, "fixture size not f32-aligned");
    let mel: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let frames = mel.len() / n_mels;
    assert_eq!(
        mel.len(),
        frames * n_mels,
        "fixture length {} not a multiple of num_mel_bins {}",
        mel.len(),
        n_mels
    );
    let expected_pads = frames_to_audio_pads(frames);
    println!(
        "fixture: {} frames x {} mel bins -> {} audio pads (formula)",
        frames, n_mels, expected_pads
    );

    // Warmup (includes any lazy allocations).
    let out = tower.forward(&mel, frames).expect("warmup forward");
    assert_eq!(out.pads, expected_pads, "pads mismatch vs formula");
    println!("warmup done: pads = {}", out.pads);

    let rounds = 7;
    let mut timings = Vec::with_capacity(rounds);
    for round in 1..=rounds {
        let t = Instant::now();
        let _out = tower.forward(&mel, frames).expect("forward");
        let elapsed = t.elapsed().as_secs_f64();
        timings.push(elapsed);
        println!("  round{round}: {:>8.2} ms", elapsed * 1e3);
    }
    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = timings[timings.len() / 2];
    let mean: f64 = timings.iter().sum::<f64>() / timings.len() as f64;
    println!(
        "median: {:#.2} ms   mean: {:.2} ms   ({} rounds)",
        median * 1e3,
        mean * 1e3,
        rounds
    );

    // Deterministic checksum for cross-backend sanity (printed once).
    let checksum: f32 = out.features.iter().map(|v| v.abs()).sum();
    let max_abs = out.features.iter().cloned().fold(0.0f32, f32::max);
    let max_mag = out.features.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!(
        "output: pads={} dims={} sum_abs={:.6} max={:.6} max_abs={:.6}",
        out.pads,
        out.features.len() / out.pads,
        checksum,
        max_abs,
        max_mag
    );
}
