//! Checkpoint-tier (and `AIMD_E2E=1`) PaddleOCR-VL tests against the
//! mlx-vlm goldens in `tests/golden/paddleocr_vl/`. All real-weight access
//! goes through the shared fixtures in [`super::test_support`].

use aimd_mlx::{MlxArray, MlxDtype};

use super::generate::greedy_stream;
use super::pipeline::{VlRequest, VlTask};
use super::preprocess::{OCR_TASK, build_prompt_text, image_token_count};
use super::test_support::*;
use crate::models::unlimited_ocr::mlx_weights::dense;
use crate::models::unlimited_ocr::test_support::{Parity, compare_rows, mlx_rows_f32};
use crate::tokenizer::Tokenizer;

const PRECISIONS: [Option<u32>; 2] = [None, Some(4)];

/// Off-Metal bound for the encoder / projector / logprobs: kernel-order and
/// bf16 rounding-boundary noise only (wiring bugs are O(1) relative RMS).
const NOISE: Parity = Parity {
    rel_rms: 2e-2,
    max_frac: 5e-2,
};
/// The port issues mlx-vlm's ops in its order, so on Metal every stage and
/// every logprob is bit-identical to the goldens.
const EXACT: Parity = Parity {
    rel_rms: 0.0,
    max_frac: 0.0,
};

/// Exact on Metal (where the goldens were recorded); elementwise-exact
/// stages (patch embed, position embedding) are exact everywhere.
fn bound(elementwise: bool) -> Parity {
    if elementwise || aimd_mlx::metal_kernels_available().unwrap_or(false) {
        EXACT
    } else {
        NOISE
    }
}

/// One manifest dump's bytes (each dump names its golden file).
fn stage_bytes(e: &serde_json::Value) -> Vec<u8> {
    let off = e["offset"].as_u64().unwrap() as usize;
    let len = e["bytes"].as_u64().unwrap() as usize;
    read_bytes(e["file"].as_str().unwrap())[off..off + len].to_vec()
}

/// Tokenizer-only: the Rust prompt ids equal the processor's for both
/// fixtures (needs the snapshot, not the GPU).
#[test]
fn paddleocr_vl_prompt_ids_match_processor() {
    let Some(dir) = snapshot(DENSE_REPO) else {
        eprintln!("skipping: {DENSE_REPO} not in HF cache");
        return;
    };
    let tok = Tokenizer::from_checkpoint(&dir).unwrap();
    let g = preprocess_golden();
    for (name, fx) in g["fixtures"].as_object().unwrap() {
        let grid = usizes(&fx["grid_thw"]);
        let n = image_token_count([grid[0], grid[1], grid[2]], 2);
        let ids = tok.encode(&build_prompt_text(OCR_TASK, n)).unwrap();
        assert_eq!(ids, u32s(&fx["prompt_ids"]), "{name} prompt ids");
    }
    for (name, fx) in g["spotting"].as_object().unwrap() {
        let grid = usizes(&fx["grid_thw"]);
        let n = image_token_count([grid[0], grid[1], grid[2]], 2);
        let ids = tok
            .encode(&build_prompt_text(VlTask::Spotting.prompt(), n))
            .unwrap();
        assert_eq!(ids, u32s(&fx["prompt_ids"]), "{name} spotting prompt ids");
    }
}

/// Vision stages (patch embed .. projector) vs mlx-vlm for bf16 and
/// load-time 4-bit, on both fixtures.
#[test]
fn paddleocr_vl_vision_stages_match_mlx_vlm() {
    let m = manifest();
    let mut fails = Vec::new();
    for bits in PRECISIONS {
        let Some(p) = shared_pipeline(bits) else {
            return;
        };
        let prec = prec_key(bits);
        for fx in ["small_text", "page_text"] {
            let (rgb, w, h) = fixture_image(fx);
            let prompt = p.prepare(&rgb, w, h, VlTask::Ocr).unwrap();
            let pp = p.vision().config().patch_size;
            let pixels =
                MlxArray::from_bf16(&[prompt.image.n_patches(), 3, pp, pp], &prompt.image.pixels)
                    .unwrap();
            let mut taps = Vec::new();
            p.vision()
                .forward(&pixels, prompt.image.grid_thw, Some(&mut taps))
                .unwrap();
            let v = &m[prec]["fixtures"][fx]["vision"];
            for e in v["dumps"].as_array().unwrap() {
                let stage = e["stage"].as_str().unwrap();
                let got = &taps.iter().find(|(n, _)| n == stage).unwrap().1;
                assert_eq!(
                    got.shape().unwrap(),
                    usizes(&e["shape"]),
                    "{prec} {fx} {stage}"
                );
                let want = bf16_le(&stage_bytes(e));
                let (g, cols) = mlx_rows_f32(got);
                let bound = bound(matches!(stage, "patch_embed" | "embeddings"));
                compare_rows(
                    &format!("{prec} {fx} {stage}"),
                    &g,
                    cols,
                    &usizes(&e["rows"]),
                    &want,
                    bound,
                    &mut fails,
                );
            }
        }
    }
    assert!(
        fails.is_empty(),
        "vision parity failures:\n{}",
        fails.join("\n")
    );
}

/// Decoder: with mlx-vlm's projector output spliced in, the first greedy
/// steps reproduce its tokens and (strided) bf16 logprobs.
#[test]
fn paddleocr_vl_decoder_logprobs_match_mlx_vlm() {
    let m = manifest();
    let stride = m["logprob_stride"].as_u64().unwrap() as usize;
    let mut fails = Vec::new();
    for bits in PRECISIONS {
        let Some(p) = shared_pipeline(bits) else {
            return;
        };
        let prec = prec_key(bits);
        let (rgb, w, h) = fixture_image("small_text");
        let prompt = p.prepare(&rgb, w, h, VlTask::Ocr).unwrap();
        let v = &m[prec]["fixtures"]["small_text"]["vision"];
        let e = v["dumps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["stage"] == "projector")
            .unwrap();
        let feats =
            MlxArray::from_bytes(&stage_bytes(e), MlxDtype::Bf16, &usizes(&e["shape"])).unwrap();
        let embeds = p.splice(&prompt, &feats).unwrap();
        let lp = &m[prec]["fixtures"]["small_text"]["logprobs"];
        let steps = lp["steps"].as_array().unwrap();
        let mut cap = Vec::new();
        let stream = greedy_stream(
            p.decoder(),
            &embeds,
            &prompt.rope,
            p.eos_ids(),
            steps.len(),
            Some(&mut cap),
        )
        .unwrap();
        let want_tokens: Vec<u32> = steps
            .iter()
            .map(|s| s["token"].as_u64().unwrap() as u32)
            .collect();
        assert_eq!(stream.tokens, want_tokens, "{prec} greedy tokens");
        assert_eq!(cap.len(), steps.len());
        let ref_lp = f32_le(&read_bytes(lp["file"].as_str().unwrap()));
        let per = ref_lp.len() / steps.len();
        for (i, (s, got)) in steps.iter().zip(&cap).enumerate() {
            assert_eq!(
                got.dtype().unwrap(),
                MlxDtype::Bf16,
                "{prec} step {i} dtype"
            );
            let all = got.astype(MlxDtype::F32).unwrap().to_f32_vec().unwrap();
            let strided: Vec<f32> = all.iter().step_by(stride).copied().collect();
            assert_eq!(strided.len(), per);
            let top = u32s(&s["top_ids"]);
            let mut best = 0usize;
            for (j, &x) in all.iter().enumerate() {
                if x > all[best] {
                    best = j;
                }
            }
            assert_eq!(best as u32, top[0], "{prec} step {i} argmax");
            compare_rows(
                &format!("{prec} step {i} logprobs"),
                &strided,
                per,
                &[0],
                &ref_lp[i * per..(i + 1) * per],
                bound(false),
                &mut fails,
            );
        }
    }
    assert!(
        fails.is_empty(),
        "decoder parity failures:\n{}",
        fails.join("\n")
    );
}

fn same_bytes(tag: &str, a: &MlxArray, b: &MlxArray) -> bool {
    let (da, db) = (a.dtype().unwrap(), b.dtype().unwrap());
    let (sa, sb) = (a.shape().unwrap(), b.shape().unwrap());
    if da != db || sa != sb {
        eprintln!("{tag}: {da:?}{sa:?} vs {db:?}{sb:?}");
        return false;
    }
    let eq = match da {
        MlxDtype::U32 => a.to_u32_bytes().unwrap() == b.to_u32_bytes().unwrap(),
        MlxDtype::Bf16 => a.to_bf16_bytes().unwrap() == b.to_bf16_bytes().unwrap(),
        _ => a.to_f32_bytes().unwrap() == b.to_f32_bytes().unwrap(),
    };
    if !eq {
        eprintln!("{tag}: bytes differ");
    }
    eq
}

/// Load-time 4-bit quantization is bit-exact with
/// `mlx-community/PaddleOCR-VL-1.6-4bit` for all 212 quantized modules
/// (position embedding through its dequantized table); vision `fc2` stays
/// dense.
#[test]
fn paddleocr_vl_4bit_quantization_bit_exact_with_mlx_community() {
    let Some(p) = shared_pipeline(Some(4)) else {
        return;
    };
    let Some(qdir) = snapshot(Q4_REPO) else {
        eprintln!("skipping: {Q4_REPO} not in HF cache");
        return;
    };
    let store = aimd_weights::load_checkpoint(&qdir).unwrap();
    let mut checked = 0usize;
    let mut bad: Vec<String> = Vec::new();
    let mut check = |name: &str, packed: Option<[&MlxArray; 3]>| {
        let packed = packed.unwrap_or_else(|| panic!("{name} not quantized"));
        for (suffix, got) in ["weight", "scales", "biases"].iter().zip(packed) {
            let want = dense(&store, &format!("{name}.{suffix}"), None).unwrap();
            if !same_bytes(&format!("{name}.{suffix}"), got, &want) {
                bad.push(format!("{name}.{suffix}"));
            }
        }
        checked += 1;
    };
    let dec = p.decoder();
    check(
        "language_model.model.embed_tokens",
        dec.embedding().packed(),
    );
    check("language_model.lm_head", dec.lm_head().packed());
    for l in 0..dec.config().num_hidden_layers {
        for n in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            check(
                &format!("language_model.model.layers.{l}.self_attn.{n}"),
                dec.layer_linear(l, n).packed(),
            );
        }
        for n in ["gate_proj", "up_proj", "down_proj"] {
            check(
                &format!("language_model.model.layers.{l}.mlp.{n}"),
                dec.layer_linear(l, n).packed(),
            );
        }
    }
    let vis = p.vision();
    for l in 0..vis.config().num_hidden_layers {
        for (n, path) in [
            ("qkv", "self_attn.qkv"),
            ("out_proj", "self_attn.out_proj"),
            ("fc1", "mlp.fc1"),
        ] {
            check(
                &format!("visual.layers.{l}.{path}"),
                vis.layer_linear(l, n).packed(),
            );
        }
        assert!(
            !vis.layer_linear(l, "fc2").is_quantized(),
            "fc2 must stay dense"
        );
    }
    for n in ["linear_1", "linear_2"] {
        check(
            &format!("visual.projector.{n}"),
            vis.projector_linear(n).packed(),
        );
    }
    let pq = |s: &str| {
        dense(
            &store,
            &format!("visual.embeddings.position_embedding.{s}"),
            None,
        )
        .unwrap()
    };
    let table = pq("weight")
        .dequantize(&pq("scales"), Some(&pq("biases")), Some(64), Some(4))
        .unwrap();
    if !same_bytes("position_embedding", vis.position_table(), &table) {
        bad.push("visual.embeddings.position_embedding".into());
    }
    checked += 1;
    assert_eq!(checked, 212, "quantized module count");
    assert!(bad.is_empty(), "mismatched tensors: {bad:?}");
}

/// Tier 3: full greedy generation equals mlx-vlm (tokens and markdown) for
/// both fixtures in bf16 and 4-bit.
#[test]
fn paddleocr_vl_e2e_greedy_matches_mlx_vlm() {
    if !e2e_enabled() {
        eprintln!("skipping: set AIMD_E2E=1");
        return;
    }
    let m = manifest();
    for bits in PRECISIONS {
        let Some(p) = shared_pipeline(bits) else {
            return;
        };
        let prec = prec_key(bits);
        for fx in ["small_text", "page_text"] {
            let g = &m[prec]["fixtures"][fx]["greedy"];
            let mut want = u32s(&g["tokens"]);
            if want.last().is_some_and(|t| p.eos_ids().contains(t)) {
                want.pop();
            }
            let (rgb, w, h) = fixture_image(fx);
            let (stream, prompt) = p
                .generate_rgb8(&rgb, w, h, VlRequest::default(), None)
                .unwrap();
            let pre = stream.prefill.as_secs_f64();
            let dec = stream.decode.as_secs_f64().max(1e-9);
            eprintln!(
                "{prec} {fx}: prompt {} tok {:.0} tok/s, gen {} tok {:.1} tok/s, total {:.3} s",
                prompt.ids.len(),
                prompt.ids.len() as f64 / pre,
                stream.tokens.len(),
                stream.tokens.len().saturating_sub(1) as f64 / dec,
                pre + dec
            );
            assert_eq!(stream.tokens, want, "{prec} {fx} tokens");
            let out = p.recognize_rgb8(&rgb, w, h, VlRequest::default()).unwrap();
            assert_eq!(
                out.markdown,
                g["markdown"].as_str().unwrap(),
                "{prec} {fx} markdown"
            );
            assert_eq!(out.tokens, want.len());
            assert!(out.spots.is_empty());

            // `Spotting:` (README preprocessing): tokens, plain text and
            // parsed quads equal the reference run.
            let s = &m[prec]["fixtures"][fx]["spotting"];
            let mut want = u32s(&s["tokens"]);
            if want.last().is_some_and(|t| p.eos_ids().contains(t)) {
                want.pop();
            }
            let req = VlRequest {
                task: VlTask::Spotting,
                ..VlRequest::default()
            };
            let t = std::time::Instant::now();
            let (out, stats) = p.recognize_rgb8_with_stats(&rgb, w, h, req).unwrap();
            eprintln!(
                "{prec} {fx} spotting: prompt {} tok, gen {} tok, {} spots, {:.3} s \
                 (vision {:.3} prefill {:.3} decode {:.3})",
                stats.prompt_tokens,
                out.tokens,
                out.spots.len(),
                t.elapsed().as_secs_f64(),
                stats.vision.as_secs_f64(),
                stats.prefill.as_secs_f64(),
                stats.decode.as_secs_f64(),
            );
            assert_eq!(
                stats.prompt_tokens as u64,
                s["prompt_tokens"].as_u64().unwrap(),
                "{prec} {fx} spotting prompt"
            );
            assert_eq!(out.tokens, want.len(), "{prec} {fx} spotting tokens");
            let (stream, _) = p.generate_rgb8(&rgb, w, h, req, None).unwrap();
            assert_eq!(stream.tokens, want, "{prec} {fx} spotting tokens");
            assert_eq!(out.markdown, s["plain"].as_str().unwrap());
            let reference = super::spotting::parse_spotting_output(s["markdown"].as_str().unwrap());
            assert_eq!(out.spots, reference, "{prec} {fx} spots");
        }
    }
}
