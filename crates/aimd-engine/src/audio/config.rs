//! Mel configuration, serialized into the golden fixture metadata.
//!
//! The field set and the serialized (snake_case) names mirror
//! `scripts/export_mel_reference.py`'s `MEL_CONFIG`, so the Rust tests can
//! deserialize `config` straight out of a fixture `meta.json`. The defaults
//! are exactly the extracted `Qwen3ASRFeatureExtractor` parameters of
//! `Qwen/Qwen3-ASR-1.7B-hf` (see the module docs in [`crate::audio`]).

use serde::{Deserialize, Serialize};

use super::error::AudioError;

/// Mel frequency scale used to lay out the filterbank triangles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MelScale {
    /// Slaney (Auditory Toolbox) scale: linear below 1 kHz, log above.
    Slaney,
    /// HTK scale: `2595 * log10(1 + f/700)`.
    Htk,
}

/// Per-filter area normalization of the triangular filterbank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MelNorm {
    /// No normalization.
    None,
    /// Slaney area normalization: weights scaled by `2 / (band width in Hz)`.
    Slaney,
}

/// STFT analysis window. The reference uses a periodic Hann.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WindowKind {
    /// Periodic Hann: `0.5 - 0.5 * cos(2*pi*k/n_fft)` (`torch.hann_window`
    /// with its `periodic=True` default).
    Hann,
}

/// How the STFT handles frame edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PadMode {
    /// Reflect padding (`torch.stft` `center=True, pad_mode="reflect"`):
    /// the boundary sample is not repeated.
    Reflect,
}

/// Full parameter set of the mel front-end.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MelConfig {
    /// Expected sample rate in Hz (16000 for Qwen3-ASR).
    pub sampling_rate: u32,
    /// Number of mel filterbank channels (128 for Qwen3-ASR).
    pub n_mels: usize,
    /// STFT size (400 for Qwen3-ASR); the window length equals `n_fft`.
    pub n_fft: usize,
    /// Frame step in samples (160 for Qwen3-ASR).
    pub hop_length: usize,
    /// Lowest filterbank frequency in Hz (0.0).
    pub f_min: f64,
    /// Highest filterbank frequency in Hz (8000.0 = Nyquist).
    pub f_max: f64,
    /// Mel scale for filterbank layout (slaney).
    pub mel_scale: MelScale,
    /// Filterbank normalization (slaney).
    pub norm: MelNorm,
    /// Spectral exponent; only 2.0 (power spectrum) is implemented.
    pub power: f64,
    /// Center frames with `n_fft/2` reflect padding on both sides.
    pub center: bool,
    /// Edge handling for centered frames (reflect).
    pub pad_mode: PadMode,
    /// Analysis window (periodic hann).
    pub window: WindowKind,
    /// Power-domain clamp applied before the log (1e-10).
    pub clamp_min: f64,
    /// Log-domain floor offset from the global max (8.0).
    pub floor_offset: f64,
    /// Additive constant of the final normalization (4.0).
    pub norm_offset: f64,
    /// Divisor of the final normalization (4.0).
    pub norm_scale: f64,
    /// Drop the trailing STFT frame, matching `stft[..., :-1]` in the
    /// reference (final frame count becomes `floor(N/hop)`).
    pub drop_last_frame: bool,
    /// Additive Gaussian noise on frames; only 0.0 is supported (the
    /// reference default; the Rust engine has no dither RNG).
    pub dither: f64,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sampling_rate: 16000,
            n_mels: 128,
            n_fft: 400,
            hop_length: 160,
            f_min: 0.0,
            f_max: 8000.0,
            mel_scale: MelScale::Slaney,
            norm: MelNorm::Slaney,
            power: 2.0,
            center: true,
            pad_mode: PadMode::Reflect,
            window: WindowKind::Hann,
            clamp_min: 1e-10,
            floor_offset: 8.0,
            norm_offset: 4.0,
            norm_scale: 4.0,
            drop_last_frame: true,
            dither: 0.0,
        }
    }
}

impl MelConfig {
    /// Number of one-sided STFT bins (`n_fft/2 + 1`).
    pub fn freq_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Validate the configuration against what [`crate::audio::
    /// mel_spectrogram`] implements. The reference chain fixes several
    /// knobs; deviating values are rejected with typed errors instead of
    /// silently computing something the model never saw.
    pub fn validate(&self) -> Result<(), AudioError> {
        if self.n_fft < 2 || !self.n_fft.is_multiple_of(2) {
            return Err(AudioError::InvalidConfig(format!(
                "n_fft must be an even number >= 2, got {}",
                self.n_fft
            )));
        }
        if self.hop_length == 0 {
            return Err(AudioError::InvalidConfig("hop_length must be > 0".into()));
        }
        if self.n_mels == 0 {
            return Err(AudioError::InvalidConfig("n_mels must be > 0".into()));
        }
        if !self.f_min.is_finite() || self.f_min < 0.0 {
            return Err(AudioError::InvalidConfig(format!(
                "f_min must be finite and >= 0, got {}",
                self.f_min
            )));
        }
        let nyquist = f64::from(self.sampling_rate) / 2.0;
        if !self.f_max.is_finite() || self.f_max <= self.f_min || self.f_max > nyquist {
            return Err(AudioError::InvalidConfig(format!(
                "f_max must be finite, > f_min and <= Nyquist ({nyquist}), got {}",
                self.f_max
            )));
        }
        if self.power != 2.0 {
            return Err(AudioError::InvalidConfig(format!(
                "only power=2.0 (power spectrum) is implemented, got {}",
                self.power
            )));
        }
        if self.window != WindowKind::Hann {
            return Err(AudioError::InvalidConfig(
                "only the periodic hann window is implemented".into(),
            ));
        }
        if !self.center || self.pad_mode != PadMode::Reflect {
            return Err(AudioError::InvalidConfig(
                "only center=true with reflect padding is implemented".into(),
            ));
        }
        if self.dither != 0.0 {
            return Err(AudioError::InvalidConfig(format!(
                "dithering is not supported (reference default is 0.0), got {}",
                self.dither
            )));
        }
        if self.clamp_min <= 0.0 || !self.clamp_min.is_finite() {
            return Err(AudioError::InvalidConfig(format!(
                "clamp_min must be finite and > 0, got {}",
                self.clamp_min
            )));
        }
        if self.norm_scale == 0.0 || !self.norm_scale.is_finite() {
            return Err(AudioError::InvalidConfig(format!(
                "norm_scale must be finite and non-zero, got {}",
                self.norm_scale
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::MelConfig;

    #[test]
    fn audio_mel_config_serde_roundtrip() {
        let config = MelConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"mel_scale\":\"slaney\""));
        assert!(json.contains("\"pad_mode\":\"reflect\""));
        assert!(json.contains("\"window\":\"hann\""));
        let back: MelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back, config);
    }
}
