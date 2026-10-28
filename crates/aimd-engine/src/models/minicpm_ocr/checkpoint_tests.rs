//! MiniCPM-OCR tests against the transformers goldens in
//! `tests/golden/minicpm_ocr/` (the official implementation in torch bf16;
//! 4-bit = the same graph on `mx.quantize` weights): preprocessing
//! (hermetic), prompt ids (tokenizer only), then checkpoint-tier vision /
//! decoder / logprob parity and the `AIMD_E2E=1` greedy runs. All real-weight access goes through
//! the shared fixtures in [`super::test_support`].

use aimd_mlx::MlxDtype;

use super::config::PreprocessorConfig;
use super::generate::greedy_stream;
use super::pipeline::{BatchImage, OcrRequest, TEXT_TASK};
use super::preprocess::{
    build_prompt_text, crop_tokens, crops_rgb8, image_placeholder, patch_rows, pixel_lut,
    position_boundaries, position_buckets, slice_plan,
};
use super::test_support::*;
use crate::models::unlimited_ocr::test_support::{Parity, compare_rows, mlx_rows_f32};
use crate::tokenizer::Tokenizer;

const PRECISIONS: [Option<u32>; 2] = [None, Some(4)];

/// Hidden states vs torch bf16 on CPU. The port rounds like torch (one
/// bf16 rounding per op), but matmul accumulation order differs, and that
/// drift compounds through the 27 vision / 24 decoder layers. Measured on
/// Metal: rel rms <= 3.4e-2, max <= 8.5% (the deepest merger crop); mlx-vlm's
/// rounding was 2-3x further from torch at the same stages.
const STAGES: Parity = Parity {
    rel_rms: 5e-2,
    max_frac: 1.25e-1,
};
/// Step logprobs (f32 log-softmax of bf16 logits). Measured: rel rms
/// <= 6.6e-3, max <= 0.83%.
const LOGPROBS: Parity = Parity {
    rel_rms: 1e-2,
    max_frac: 2e-2,
};

fn on_metal() -> bool {
    aimd_mlx::metal_kernels_available().unwrap_or(false)
}

fn fnv1a64(data: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

fn stage_bytes(e: &serde_json::Value) -> Vec<u8> {
    let off = e["offset"].as_u64().unwrap() as usize;
    let len = e["bytes"].as_u64().unwrap() as usize;
    read_bytes(e["file"].as_str().unwrap())[off..off + len].to_vec()
}

fn pair(v: &serde_json::Value) -> (usize, usize) {
    let u = usizes(v);
    (u[0], u[1])
}

/// Hermetic: pixel table, position boundaries, slicing plans, and every
/// crop (size, patch grid, buckets, RGB8 and patch-row hashes).
#[test]
fn minicpm_ocr_preprocess_matches_processor() {
    let g = preprocess_golden();
    let cfg = PreprocessorConfig::default();
    let lut = pixel_lut(0.5, 0.5);
    let bits: Vec<u32> = lut.iter().map(|b| b.to_bits() as u32).collect();
    assert_eq!(bits, u32s(&g["pixel_lut_bf16"]), "pixel table");
    let want_b: Vec<f32> = g["pos_boundaries_f32"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let boundaries = position_boundaries(70);
    assert_eq!(boundaries, want_b, "position boundaries");

    for s in g["slicing"].as_array().unwrap() {
        let (w, h) = pair(&s["size_wh"]);
        let plan = slice_plan(w, h, &cfg);
        let tag = format!("{w}x{h}");
        assert_eq!(
            plan.grid,
            s["grid"].as_array().map(|_| pair(&s["grid"])),
            "{tag} grid"
        );
        assert_eq!(plan.source, pair(&s["source"]), "{tag} source");
        assert_eq!(
            plan.refine,
            s["refine"].as_array().map(|_| pair(&s["refine"])),
            "{tag} refine"
        );
    }

    let luts = [lut; 3];
    for (name, fx) in g["fixtures"].as_object().unwrap() {
        let (rgb, w, h) = fixture_image(name);
        assert_eq!((w, h), pair(&fx["size_wh"]), "{name} size");
        let (crops, _) = crops_rgb8(&rgb, w, h, &cfg).unwrap();
        let want = fx["crops"].as_array().unwrap();
        assert_eq!(crops.len(), want.len(), "{name} crop count");
        for (i, (c, e)) in crops.iter().zip(want).enumerate() {
            let tag = format!("{name} crop {i}");
            assert_eq!((c.width, c.height), pair(&e["size_wh"]), "{tag} size");
            let (gh, gw) = c.grid(14);
            assert_eq!((gh, gw), pair(&e["tgt_hw"]), "{tag} grid");
            assert_eq!(
                position_buckets(gh, &boundaries),
                u32s(&e["bucket_h"]),
                "{tag} bucket_h"
            );
            assert_eq!(
                position_buckets(gw, &boundaries),
                u32s(&e["bucket_w"]),
                "{tag} bucket_w"
            );
            assert_eq!(
                fnv1a64(&c.rgb),
                e["rgb8_fnv1a64"].as_str().unwrap(),
                "{tag} rgb8"
            );
            let rows: Vec<u8> = patch_rows(c, 14, &luts)
                .iter()
                .flat_map(|b| b.to_bits().to_le_bytes())
                .collect();
            assert_eq!(
                fnv1a64(&rows),
                e["patch_bf16_fnv1a64"].as_str().unwrap(),
                "{tag} patches"
            );
        }
    }
}

/// Tokenizer only: Rust prompt ids and image spans equal the processor's.
#[test]
fn minicpm_ocr_prompt_ids_match_processor() {
    let Some(dir) = snapshot(DENSE_REPO) else {
        eprintln!("skipping: {DENSE_REPO} not in HF cache");
        return;
    };
    let tok = Tokenizer::from_checkpoint(&dir).unwrap();
    let g = preprocess_golden();
    for (name, fx) in g["fixtures"].as_object().unwrap() {
        let tokens: Vec<usize> = fx["crops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let (gh, gw) = pair(&c["tgt_hw"]);
                crop_tokens(gh, gw)
            })
            .collect();
        let grid = fx["plan"]["grid"]
            .as_array()
            .map(|_| pair(&fx["plan"]["grid"]));
        let text = build_prompt_text(&image_placeholder(&tokens, grid), TEXT_TASK);
        let ids = tok.encode(&text).unwrap();
        assert_eq!(ids, u32s(&fx["prompt_ids"]), "{name} prompt ids");
    }
}

/// Vision stages (patch embed .. merger) vs transformers. Vision is dense at
/// every precision, so both manifests hold the same tower.
#[test]
fn minicpm_ocr_vision_stages_match_transformers() {
    let m = manifest();
    let mut fails = Vec::new();
    for bits in PRECISIONS {
        let Some(p) = shared_pipeline(bits) else {
            return;
        };
        let prec = prec_key(bits);
        for fx in ["small_text", "page_text"] {
            let (rgb, w, h) = fixture_image(fx);
            let prompt = p.prepare(&rgb, w, h, TEXT_TASK, None).unwrap();
            let dumps = m[prec]["fixtures"][fx]["vision"].as_array().unwrap();
            let mut taps = Vec::new();
            for (i, c) in prompt.crops.iter().enumerate() {
                let (gh, gw) = c.grid(14);
                let rows = patch_rows(c, 14, p.luts());
                let mut t = Vec::new();
                p.vision().forward(&rows, gh, gw, Some(&mut t)).unwrap();
                for (n, a) in t {
                    let n = if fx == "page_text" {
                        format!("{n}_crop_{i}")
                    } else {
                        n
                    };
                    taps.push((n, a));
                }
            }
            for e in dumps {
                let stage = e["stage"].as_str().unwrap();
                let got = &taps
                    .iter()
                    .find(|(n, _)| n == stage)
                    .unwrap_or_else(|| panic!("no tap {stage}"))
                    .1;
                assert_eq!(
                    got.shape().unwrap(),
                    usizes(&e["shape"]),
                    "{prec} {fx} {stage}"
                );
                let (g, cols) = mlx_rows_f32(got);
                compare_rows(
                    &format!("{prec} {fx} {stage}"),
                    &g,
                    cols,
                    &usizes(&e["rows"]),
                    &bf16_le(&stage_bytes(e)),
                    STAGES,
                    &mut fails,
                );
            }
        }
    }
    if on_metal() {
        assert!(
            fails.is_empty(),
            "vision parity failures:\n{}",
            fails.join("\n")
        );
    } else if !fails.is_empty() {
        eprintln!(
            "vision drift on a non-Metal backend (not enforced):\n{}",
            fails.join("\n")
        );
    }
}

/// Decoder: prefill hidden states (no cache split) and the first greedy
/// steps' tokens and strided logprobs vs transformers, on `small_text`.
#[test]
fn minicpm_ocr_decoder_matches_transformers() {
    let m = manifest();
    let stride = m["logprob_stride"].as_u64().unwrap() as usize;
    let mut fails = Vec::new();
    for bits in PRECISIONS {
        let Some(p) = shared_pipeline(bits) else {
            return;
        };
        let prec = prec_key(bits);
        let fx = &m[prec]["fixtures"]["small_text"];
        let (rgb, w, h) = fixture_image("small_text");
        let prompt = p.prepare(&rgb, w, h, TEXT_TASK, None).unwrap();
        let feats = p.crop_features(&prompt).unwrap();
        let embeds = p.splice(&prompt, &feats).unwrap();

        let dec = p.decoder();
        let n = prompt.ids.len();
        let mut taps = vec![("inputs_embeds".to_string(), embeds.try_clone().unwrap())];
        let mut caches = dec.make_cache();
        dec.forward(
            &embeds,
            &dec.rope(0, n).unwrap(),
            &mut caches,
            Some(&mut taps),
        )
        .unwrap();
        for e in fx["decoder"].as_array().unwrap() {
            let stage = e["stage"].as_str().unwrap();
            let got = &taps.iter().find(|(s, _)| s == stage).unwrap().1;
            let got = got.reshape(&[n, dec.hidden_size()]).unwrap();
            assert_eq!(got.shape().unwrap(), usizes(&e["shape"]), "{prec} {stage}");
            let (g, cols) = mlx_rows_f32(&got);
            compare_rows(
                &format!("{prec} decoder {stage}"),
                &g,
                cols,
                &usizes(&e["rows"]),
                &bf16_le(&stage_bytes(e)),
                STAGES,
                &mut fails,
            );
        }

        let lp = &fx["logprobs"];
        let steps = lp["steps"].as_array().unwrap();
        let mut cap = Vec::new();
        let stream =
            greedy_stream(dec, &embeds, p.eos_ids(), steps.len(), None, Some(&mut cap)).unwrap();
        let want: Vec<u32> = steps
            .iter()
            .map(|s| s["token"].as_u64().unwrap() as u32)
            .collect();
        // An EOS step ends the stream but still has its logprobs captured.
        let k = stream.tokens.len();
        assert!(k + 1 >= want.len(), "{prec} stopped after {k} tokens");
        assert_eq!(stream.tokens[..], want[..k], "{prec} greedy tokens");
        assert_eq!(cap.len(), want.len(), "{prec} captured steps");
        let ref_lp = f32_le(&read_bytes(lp["file"].as_str().unwrap()));
        let per = ref_lp.len() / steps.len();
        for (i, (s, got)) in steps.iter().zip(&cap).enumerate() {
            assert_eq!(got.dtype().unwrap(), MlxDtype::F32, "{prec} step {i} dtype");
            let all = got.astype(MlxDtype::F32).unwrap().to_f32_vec().unwrap();
            let strided: Vec<f32> = all.iter().step_by(stride).copied().collect();
            assert_eq!(strided.len(), per);
            let best = (0..all.len()).fold(0, |b, j| if all[j] > all[b] { j } else { b });
            assert_eq!(
                best as u32,
                u32s(&s["top_ids"])[0],
                "{prec} step {i} argmax"
            );
            compare_rows(
                &format!("{prec} step {i} logprobs"),
                &strided,
                per,
                &[0],
                &ref_lp[i * per..(i + 1) * per],
                LOGPROBS,
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

/// Load-time 4-bit quantizes exactly mlx-vlm's 96 modules (full-attention
/// q/k/v/o and every MLP); DeltaNet projections stay dense. Bit-exactness
/// of `mx.quantize` against `mlx_vlm.convert` is checked by the exporter.
#[test]
fn minicpm_ocr_4bit_quantizes_attention_and_mlp_only() {
    let m = manifest();
    let Some(p) = shared_pipeline(Some(4)) else {
        return;
    };
    let dec = p.decoder();
    let mut got = Vec::new();
    for l in 0..dec.config().num_hidden_layers {
        let names: &[&str] = if dec.config().is_linear(l) {
            assert!(!dec.layer_linear(l, "in_proj_qkv").is_quantized());
            assert!(!dec.layer_linear(l, "out_proj").is_quantized());
            &["gate_proj", "up_proj", "down_proj"]
        } else {
            &[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        };
        for n in names {
            assert!(dec.layer_linear(l, n).is_quantized(), "layer {l} {n}");
            let kind = if n.ends_with("proj") && ["q", "k", "v", "o"].contains(&&n[..1]) {
                "self_attn"
            } else {
                "mlp"
            };
            got.push(format!("language_model.model.layers.{l}.{kind}.{n}"));
        }
    }
    got.sort();
    let mut want: Vec<String> = m["quant_4bit"]["quantized_modules"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    want.sort();
    assert_eq!(got, want);
}

/// Tier 3: full greedy generation equals transformers (tokens and text) for
/// both fixtures in bf16 and 4-bit, serially and as one region batch.
#[test]
fn minicpm_ocr_e2e_greedy_matches_transformers() {
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
        let mut texts: Vec<String> = Vec::new();
        for fx in ["small_text", "page_text"] {
            let g = &m[prec]["fixtures"][fx]["greedy"];
            let mut want = u32s(&g["tokens"]);
            if want.last().is_some_and(|t| p.eos_ids().contains(t)) {
                want.pop();
            }
            let (rgb, w, h) = fixture_image(fx);
            let plain = OcrRequest {
                speculative: false,
                ..OcrRequest::default()
            };
            let (out, stats) = p.recognize_rgb8_with_stats(&rgb, w, h, plain).unwrap();
            eprintln!(
                "{prec} {fx}: prompt {} tok ({} crops), gen {} tok, vision {:.3} s, \
                 prefill {:.3} s, decode {:.3} s ({:.1} tok/s)",
                stats.prompt_tokens,
                stats.crops,
                out.tokens,
                stats.vision.as_secs_f64(),
                stats.prefill.as_secs_f64(),
                stats.decode.as_secs_f64(),
                out.tokens.saturating_sub(1) as f64 / stats.decode.as_secs_f64().max(1e-9),
            );
            assert_eq!(
                stats.prompt_tokens as u64,
                g["prompt_tokens"].as_u64().unwrap(),
                "{prec} {fx} prompt tokens"
            );
            let (stream, _) = p.generate_rgb8(&rgb, w, h, plain, None).unwrap();
            assert_eq!(stream.tokens, want, "{prec} {fx} tokens");
            assert_eq!(
                out.text,
                g["markdown"].as_str().unwrap(),
                "{prec} {fx} text"
            );
            texts.push(out.text.clone());

            // DSpark: the same greedy text, faster.
            if !p.has_drafter() {
                continue;
            }
            let (spec, sstats) = p
                .recognize_rgb8_with_stats(&rgb, w, h, OcrRequest::default())
                .unwrap();
            let acc = sstats.speculative.expect("speculative stats");
            eprintln!(
                "{prec} {fx} DSpark: gen {} tok, decode {:.3} s ({:.1} tok/s), \
                 {} rounds, accepted {}/{} drafts ({:.2} tok/round)",
                spec.tokens,
                sstats.decode.as_secs_f64(),
                spec.tokens.saturating_sub(1) as f64 / sstats.decode.as_secs_f64().max(1e-9),
                acc.rounds,
                acc.accepted,
                acc.drafted,
                spec.tokens.saturating_sub(1) as f64 / acc.rounds.max(1) as f64,
            );
            assert_eq!(spec.text, out.text, "{prec} {fx} DSpark text");
        }

        // Region batching: left-padded rows, rows of equal length, and
        // (with a drafter) batched DSpark verification give the same text.
        let images: Vec<_> = ["small_text", "page_text", "small_text"]
            .map(fixture_image)
            .into_iter()
            .collect();
        let want = [&texts[0], &texts[1], &texts[0]];
        for speculative in [false, true] {
            let req = OcrRequest {
                speculative,
                ..OcrRequest::default()
            };
            let batch: Vec<BatchImage<'_>> = images
                .iter()
                .map(|(rgb, w, h)| (rgb.as_slice(), *w, *h, req))
                .collect();
            let got = p.recognize_batch_rgb8(&batch).unwrap();
            for (k, ((out, _), want)) in got.iter().zip(want).enumerate() {
                assert_eq!(
                    &out.text, want,
                    "{prec} batch[{k}] speculative={speculative}"
                );
            }
        }
    }
}
