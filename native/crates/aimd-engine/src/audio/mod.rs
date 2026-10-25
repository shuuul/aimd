//! Audio front-end: wav loading and mel spectrogram (WS-05).
//!
//! # Contract
//!
//! - [`load_wav`] reads PCM wav (the outer pipeline ffmpeg-normalizes files
//!   to 16 kHz mono s16/f32 wav) into `f32` samples in `[-1, 1]` and rejects
//!   everything else with typed [`AudioError`]s.
//! - [`mel_spectrogram`] produces the 128-bin log-mel matching the
//!   `Qwen3ASRFeatureExtractor` of `Qwen/Qwen3-ASR-1.7B-hf`, locked by the
//!   golden fixtures under `native/golden/mel/` (exported with
//!   `scripts/export_mel_reference.py`). Tolerance is element-wise because
//!   STFT is float-order sensitive, unlike the bit-exact quantization
//!   fixtures.
//! - No external DSP crate: the FFT is hand-written ([`fft`], iterative
//!   radix-2 for powers of two plus a four-step Cooley-Tukey decomposition
//!   for `n_fft = 400 = 2^4 * 5^2`).
//!
//! # Extracted reference parameter set
//!
//! Extracted from `AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B-hf")`
//! (transformers `Qwen3ASRFeatureExtractor`, `torch.stft` CPU path). This is
//! the complete set of parameters that affect the output:
//!
//! | Parameter | Value |
//! |-----------|-------|
//! | sampling_rate | 16000 |
//! | n_mels (feature_size) | 128 |
//! | n_fft | 400 (win_length == n_fft, periodic Hann) |
//! | hop_length | 160 |
//! | window | periodic Hann `0.5 - 0.5*cos(2*pi*k/n_fft)` |
//! | center / pad_mode | true / reflect (`n_fft/2 = 200` samples each side) |
//! | STFT normalization | none; one-sided -> `n_fft/2 + 1 = 201` bins |
//! | frames | `1 + floor(N/hop)`, then the last frame is dropped -> `floor(N/hop)` |
//! | power | 2.0 (magnitude squared) |
//! | mel filterbank | 201 -> 128, `f_min=0`, `f_max=8000`, slaney scale, slaney norm |
//! | dither | 0.0 |
//! | log chain | `clamp(min=1e-10).log10()`, floor at global max `- 8.0`, `(x + 4.0) / 4.0` |
//!
//! The filterbank is constructed in f64 (as HF does in numpy) and cast to
//! f32 (as the extractor casts it into torch). The Rust pipeline mirrors the
//! reference data flow: f64 STFT, power cast to f32, f64 filterbank
//! accumulation, f64 log chain, f32 output.
//!
//! # Caller-side padding (not part of `mel_spectrogram`)
//!
//! The Python extractor wrapper also zero-pads clips shorter than
//! `min_length` (8000 samples), pads/truncates to `max_length` (480000
//! samples = 30 s), and right-pads the mel time axis to a multiple of
//! `2 * n_window` (100 frames) for the encoder. Those are padding concerns
//! of the model graph (WS-06) and are intentionally excluded here; the
//! fixtures lock the core sample -> log-mel chain only.
//!
//! # Numeric tolerance
//!
//! Fixtures are compared element-wise with `abs <= 1e-2 or rel <= 1e-3`.
//! Observed against the committed fixtures (see `features.rs` tests): worst
//! absolute difference 4.676e-5 and worst relative difference 1.752e-4
//! wherever the reference is >= 0.05 in magnitude, bounded by torch's own
//! f32 STFT round-off at quiet bins - roughly two orders of magnitude
//! inside the gate. The committed fixtures were generated with the numpy
//! f64 mirror in `scripts/export_mel_reference.py`, which reproduces the
//! torch reference to the same bound, and are byte-reproducible across
//! regeneration runs.

pub mod config;
pub mod error;
pub mod features;
pub mod fft;
pub mod filterbank;
pub mod wav;

pub use config::{MelConfig, MelNorm, MelScale, PadMode, WindowKind};
pub use error::AudioError;
pub use features::{MelSpectrogram, mel_spectrogram};
pub use wav::{WavAudio, load_wav};
