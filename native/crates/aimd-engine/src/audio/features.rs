//! The mel spectrogram pipeline: the exact Qwen3-ASR log-mel chain.
//!
//! Data flow (each f32 cast point mirrors where the torch reference holds
//! f32 values; everything else stays f64):
//!
//! 1. center-pad `n_fft/2 = 200` samples with reflect on both sides
//!    (virtual padding: frame samples are index-mapped, nothing is copied)
//! 2. periodic Hann window (f64 table cast through f32, like
//!    `torch.hann_window`), frame product cast through f32
//! 3. hand-written f64 FFT ([`super::fft`])
//! 4. power = `re^2 + im^2` for the `n_fft/2 + 1` one-sided bins, cast
//!    through f32 (the reference holds f32 magnitudes)
//! 5. filterbank matmul (f64 construction cast through f32, like the
//!    extractor's `torch.from_numpy(...).to(torch.float32)`, accumulated
//!    in f64)
//! 6. `clamp(min = 1e-10).log10()`
//! 7. floor at the global max `- 8.0`
//! 8. `(x + 4.0) / 4.0`, cast to f32
//!
//! Output layout matches torch: row-major `[n_mels, frames]`
//! (frequency-major), with `frames = floor(N / hop)` (the reference drops
//! the trailing STFT frame via `stft[..., :-1]`).
//!
//! # Numeric tolerance against the fixtures
//!
//! The committed fixtures are compared element-wise with
//! `abs <= 1e-2 or rel <= 1e-3`. Observed across all 114,048 fixture
//! elements (see `audio_mel_golden_parity` below): worst absolute
//! difference 4.676e-5, worst relative difference 1.752e-4 wherever the
//! reference is >= 0.05 in magnitude. (The raw worst rel of 0.21 occurs
//! only at near-zero elements, which pass via the abs branch.) The gap is
//! dominated by the torch reference's own f32 STFT round-off at quiet bins
//! ~200 dB below the spectral peak - the Rust pipeline is the f64-accurate
//! value there - so these figures are the floor for any implementation and
//! sit two orders of magnitude inside the gate. Values near the 1e-10
//! clamp and the max-8 floor agree to ~1e-7.

use super::config::MelConfig;
use super::error::AudioError;
use super::fft::{FftPlan, FftScratch};
use super::filterbank::mel_filterbank;

/// A computed log-mel spectrogram.
#[derive(Debug, Clone, PartialEq)]
pub struct MelSpectrogram {
    /// Number of mel channels (rows).
    pub n_mels: usize,
    /// Number of time frames (columns).
    pub frames: usize,
    /// Row-major `[n_mels, frames]` f32 values, frequency-major like torch.
    pub data: Vec<f32>,
}

impl MelSpectrogram {
    /// Time slice of one mel channel.
    pub fn row(&self, mel: usize) -> &[f32] {
        &self.data[mel * self.frames..(mel + 1) * self.frames]
    }
}

/// Compute the log-mel spectrogram of `samples` (16 kHz mono f32, `[-1, 1]`)
/// under `config`.
pub fn mel_spectrogram(samples: &[f32], config: &MelConfig) -> Result<MelSpectrogram, AudioError> {
    config.validate()?;

    let n = samples.len();
    let pad = config.n_fft / 2;
    // Reflect padding needs `pad <= N - 1` (same constraint as torch).
    if n < pad + 1 {
        return Err(AudioError::InputTooShort {
            samples: n,
            required: pad + 1,
        });
    }
    let total_frames = 1 + n / config.hop_length;
    let frames = if config.drop_last_frame {
        total_frames - 1
    } else {
        total_frames
    };
    if frames == 0 {
        return Err(AudioError::InputTooShort {
            samples: n,
            required: config.hop_length,
        });
    }

    // Periodic Hann, cast through f32 like torch's f32 window.
    let window: Vec<f64> = (0..config.n_fft)
        .map(|k| {
            let w = 0.5 - 0.5 * (std::f64::consts::TAU * k as f64 / config.n_fft as f64).cos();
            f64::from(w as f32)
        })
        .collect();

    let plan = FftPlan::new(config.n_fft);
    let mut scratch = FftScratch::default();
    // Filterbank cast through f32 like the extractor's torch tensor.
    let fb: Vec<f64> = mel_filterbank(config)
        .into_iter()
        .map(|w| f64::from(w as f32))
        .collect();

    let n_mels = config.n_mels;
    let n_bins = config.freq_bins();
    let mut mel = vec![0.0f64; n_mels * frames];
    let mut re = vec![0.0f64; config.n_fft];
    let mut im = vec![0.0f64; config.n_fft];
    let mut power = vec![0.0f64; n_bins];
    let mut column = vec![0.0f64; n_mels];

    for t in 0..frames {
        // Gather frame t through the virtual reflect padding.
        for (j, &w) in window.iter().enumerate() {
            let pos = t * config.hop_length + j;
            let frame_sample = if pos < pad {
                samples[pad - pos]
            } else if pos >= pad + n {
                // y[p + n + q] = x[n - 2 - q]
                samples[n - 2 - (pos - pad - n)]
            } else {
                samples[pos - pad]
            };
            // Frame product cast through f32 (the reference multiplies f32).
            re[j] = f64::from((f64::from(frame_sample) * w) as f32);
            im[j] = 0.0;
        }

        plan.fft_in_place(&mut re, &mut im, &mut scratch);

        for (bin, p) in power.iter_mut().enumerate() {
            let magnitude_sq = re[bin] * re[bin] + im[bin] * im[bin];
            // Power cast through f32 (the reference holds f32 magnitudes).
            *p = f64::from(magnitude_sq as f32);
        }

        // Mel column: filterbank matmul with f64 accumulation.
        column.fill(0.0);
        for (bin, &p) in power.iter().enumerate() {
            if p == 0.0 {
                continue; // exact: contributes nothing
            }
            let weights = &fb[bin * n_mels..(bin + 1) * n_mels];
            for (m, &w) in weights.iter().enumerate() {
                column[m] += w * p;
            }
        }
        for (m, &v) in column.iter().enumerate() {
            mel[m * frames + t] = v;
        }
    }

    // Log chain: clamp, log10, floor at global max - offset, normalize.
    let mut max_log = f64::NEG_INFINITY;
    for v in &mut mel {
        *v = v.max(config.clamp_min).log10();
        if *v > max_log {
            max_log = *v;
        }
    }
    let floor = max_log - config.floor_offset;
    let data = mel
        .iter()
        .map(|v| ((v.max(floor) + config.norm_offset) / config.norm_scale) as f32)
        .collect();

    Ok(MelSpectrogram {
        n_mels,
        frames,
        data,
    })
}

#[cfg(test)]
mod golden_tests {
    use std::path::{Path, PathBuf};

    use serde_json::Value;

    use super::mel_spectrogram;
    use crate::audio::config::MelConfig;
    use crate::audio::error::AudioError;
    use crate::audio::filterbank::mel_filterbank;

    /// Committed fixtures live in `native/golden/mel` (two levels above the
    /// crate manifest dir).
    fn golden_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/mel")
    }

    fn read_f32_le(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("fixture readable");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    fn read_f64_le(path: &Path) -> Vec<f64> {
        let bytes = std::fs::read(path).expect("fixture readable");
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect()
    }

    /// Parity gate: element-wise `abs <= 1e-2 OR rel <= 1e-3`. Observed
    /// against the committed fixtures: 4.676e-5 worst abs; 1.752e-4 worst
    /// rel where the reference is >= 0.05 in magnitude (torch's own f32
    /// STFT round-off at quiet bins).
    const ABS_TOL: f64 = 1e-2;
    const REL_TOL: f64 = 1e-3;

    #[test]
    fn audio_mel_golden_parity() {
        let dir = golden_dir();
        let manifest: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap())
                .unwrap();

        let mut worst_abs = 0.0f64;
        let mut worst_rel = 0.0f64;
        let mut worst_rel_substantial = 0.0f64;
        let mut checked = 0usize;

        for case in manifest["cases"].as_array().expect("manifest cases") {
            let name = case["name"].as_str().unwrap();
            let input = read_f32_le(&dir.join(case["input"].as_str().unwrap()));
            let reference = read_f32_le(&dir.join(case["output"].as_str().unwrap()));
            let meta: Value = serde_json::from_str(
                &std::fs::read_to_string(dir.join(case["meta"].as_str().unwrap())).unwrap(),
            )
            .unwrap();
            let config: MelConfig = serde_json::from_value(meta["config"].clone()).unwrap();
            assert_eq!(
                meta["frames"].as_u64().unwrap() as usize,
                input.len() / config.hop_length
            );

            let mel = mel_spectrogram(&input, &config).expect("mel computes");
            assert_eq!(
                mel.n_mels,
                meta["n_mels"].as_u64().unwrap() as usize,
                "{name} n_mels"
            );
            assert_eq!(mel.frames, reference.len() / mel.n_mels, "{name} frames");
            assert_eq!(mel.data.len(), reference.len(), "{name} element count");

            for (i, (&got_raw, &want_raw)) in mel.data.iter().zip(reference.iter()).enumerate() {
                let (got, want) = (f64::from(got_raw), f64::from(want_raw));
                let abs = (got - want).abs();
                let rel = abs / want.abs().max(1e-6);
                worst_abs = worst_abs.max(abs);
                worst_rel = worst_rel.max(rel);
                // Relative error is only meaningful away from zero; near-zero
                // elements pass via the abs branch.
                if want.abs() >= 0.05 {
                    worst_rel_substantial = worst_rel_substantial.max(rel);
                }
                checked += 1;
                assert!(
                    abs <= ABS_TOL || rel <= REL_TOL,
                    "{name}[{i}]: got {got}, want {want}, abs {abs:.3e}, rel {rel:.3e}"
                );
            }
        }

        eprintln!(
            "audio_mel_golden_parity: {checked} elements, worst abs {worst_abs:.3e}, \
             worst rel {worst_rel:.3e}, worst rel where |ref| >= 0.05 {worst_rel_substantial:.3e}"
        );
        assert!(checked > 0);
    }

    #[test]
    fn audio_filterbank_matches_golden_fixture() {
        let dir = golden_dir();
        let filters = read_f64_le(&dir.join("mel_filters.f64"));
        let config = MelConfig::default();
        let built = mel_filterbank(&config);
        assert_eq!(built.len(), filters.len());
        let worst = built
            .iter()
            .zip(filters.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        eprintln!("audio_filterbank_matches_golden_fixture: worst abs {worst:.3e}");
        assert!(worst < 1e-9, "filterbank drift {worst:.3e}");
    }

    #[test]
    fn audio_mel_fixture_config_matches_default() {
        let dir = golden_dir();
        let manifest: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap())
                .unwrap();
        let first = &manifest["cases"][0];
        let meta: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join(first["meta"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        let config: MelConfig = serde_json::from_value(meta["config"].clone()).unwrap();
        assert_eq!(config, MelConfig::default());
    }

    #[test]
    fn audio_mel_silence_is_constant() {
        let dir = golden_dir();
        let manifest: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap())
                .unwrap();
        let case = manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "silence")
            .expect("silence case");
        let input = read_f32_le(&dir.join(case["input"].as_str().unwrap()));
        let mel = mel_spectrogram(&input, &MelConfig::default()).unwrap();
        assert!(mel.data.iter().all(|v| *v == -1.5));
    }

    #[test]
    fn audio_mel_frame_counts() {
        let config = MelConfig::default();
        // Exact multiple of hop: 16000/160 = 100 frames after the drop.
        assert_eq!(
            mel_spectrogram(&vec![0.0f32; 16000], &config)
                .unwrap()
                .frames,
            100
        );
        // Not a multiple: floor(9760/160) = 61.
        assert_eq!(
            mel_spectrogram(&vec![0.0f32; 9760], &config)
                .unwrap()
                .frames,
            61
        );
        assert_eq!(
            mel_spectrogram(&vec![0.0f32; 16001], &config)
                .unwrap()
                .frames,
            100
        );
        // Minimum reflectable length (pad + 1 = 201): 2 frames total, 1 kept.
        assert_eq!(
            mel_spectrogram(&vec![0.0f32; 201], &config).unwrap().frames,
            1
        );
    }

    #[test]
    fn audio_mel_rejects_too_short_input() {
        let config = MelConfig::default();
        let err = mel_spectrogram(&vec![0.0f32; 200], &config).unwrap_err();
        assert!(matches!(
            err,
            AudioError::InputTooShort {
                samples: 200,
                required: 201
            }
        ));
    }

    #[test]
    fn audio_mel_rejects_unsupported_config() {
        let dithered = MelConfig {
            dither: 1e-3,
            ..MelConfig::default()
        };
        assert!(
            mel_spectrogram(&[0.0f32; 16000], &dithered)
                .is_err_and(|e| matches!(e, AudioError::InvalidConfig(_)))
        );

        let linear = MelConfig {
            power: 1.0,
            ..MelConfig::default()
        };
        assert!(
            mel_spectrogram(&[0.0f32; 16000], &linear)
                .is_err_and(|e| matches!(e, AudioError::InvalidConfig(_)))
        );

        let wideband = MelConfig {
            f_max: 9000.0,
            ..MelConfig::default()
        }; // above Nyquist
        assert!(
            mel_spectrogram(&[0.0f32; 16000], &wideband)
                .is_err_and(|e| matches!(e, AudioError::InvalidConfig(_)))
        );
    }
}
