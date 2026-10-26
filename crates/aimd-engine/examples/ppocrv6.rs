//! Runs PP-OCRv6 medium on images and prints the lines plus per-page
//! latency (first call and the mean of three warm calls):
//!
//! ```text
//! cargo run --release -p aimd-engine --example ppocrv6 -- <det_dir> <rec_dir> <image>...
//! ```
use std::path::Path;
use std::time::Instant;

use aimd_engine::models::pp_ocrv6::pipeline::PpOcrV6Pipeline;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: ppocrv6 <det_dir> <rec_dir> <image>...");
        std::process::exit(2);
    }
    let t = Instant::now();
    let pipe = PpOcrV6Pipeline::load(Path::new(&args[0]), Path::new(&args[1])).expect("load");
    println!("load {:.3}s", t.elapsed().as_secs_f64());
    for img in &args[2..] {
        let mut times = Vec::new();
        let mut last = None;
        for _ in 0..4 {
            let t = Instant::now();
            let r = pipe.recognize_path(Path::new(img));
            times.push(t.elapsed().as_secs_f64());
            last = Some(r);
        }
        println!(
            "== {img} first {:.3}s warm {:.3}s",
            times[0],
            times[1..].iter().sum::<f64>() / 3.0
        );
        match last.expect("ran") {
            Ok(out) => {
                for l in &out.lines {
                    println!("{:?} {:.4} {:?}", l.text, l.score, l.quad);
                }
            }
            Err(e) => println!("error: {e}"),
        }
    }
}
