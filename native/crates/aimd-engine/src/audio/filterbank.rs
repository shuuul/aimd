//! Mel filterbank construction, mirroring `transformers.audio_utils`.
//!
//! Reproduces `mel_filter_bank(num_frequency_bins = n_fft/2 + 1,
//! num_mel_filters = n_mels, min_frequency = f_min, max_frequency = f_max,
//! sampling_rate = sr, norm = ..., mel_scale = ...)` from HF `transformers`
//! in f64. For the Qwen3-ASR reference (201 -> 128, 0..8000 Hz, slaney
//! scale + slaney norm) the construction is exact: the golden fixture
//! `mel_filters.f64` (exported from the live extractor) compares with zero
//! difference (see the tests).

use super::config::{MelConfig, MelNorm, MelScale};

/// Build the filterbank as a row-major `[freq_bin][mel]` f64 matrix with
/// `n_fft/2 + 1` rows and `n_mels` columns.
pub(crate) fn mel_filterbank(config: &MelConfig) -> Vec<f64> {
    let n_bins = config.freq_bins();
    let n_mels = config.n_mels;

    // Center points of the triangular filters: n_mels + 2 mel-spaced
    // frequencies mapped back to Hz.
    let mel_min = hertz_to_mel(config.f_min, config.mel_scale);
    let mel_max = hertz_to_mel(config.f_max, config.mel_scale);
    let mel_points = linspace(mel_min, mel_max, n_mels + 2);
    let filter_freqs: Vec<f64> = mel_points
        .iter()
        .map(|m| mel_to_hertz(*m, config.mel_scale))
        .collect();

    // FFT bin center frequencies: `linspace(0, sampling_rate // 2, n_bins)`
    // (integer division, as HF does).
    let fft_freqs = linspace(0.0, f64::from(config.sampling_rate / 2), n_bins);

    // Triangular filters: rising slope into the triangle center, falling
    // slope out of it, clamped at zero.
    let mut fb = vec![0.0f64; n_bins * n_mels];
    for (bin, &f) in fft_freqs.iter().enumerate() {
        for mel in 0..n_mels {
            let down = (f - filter_freqs[mel]) / (filter_freqs[mel + 1] - filter_freqs[mel]);
            let up = (filter_freqs[mel + 2] - f) / (filter_freqs[mel + 2] - filter_freqs[mel + 1]);
            let weight = down.min(up).max(0.0);
            fb[bin * n_mels + mel] = weight;
        }
    }

    // Slaney area normalization: constant energy per channel.
    if config.norm == MelNorm::Slaney {
        for mel in 0..n_mels {
            let enorm = 2.0 / (filter_freqs[mel + 2] - filter_freqs[mel]);
            for bin in 0..n_bins {
                fb[bin * n_mels + mel] *= enorm;
            }
        }
    }

    fb
}

fn hertz_to_mel(freq: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + freq / 700.0).log10(),
        MelScale::Slaney => {
            if freq >= 1000.0 {
                15.0 + (freq / 1000.0).ln() * (27.0 / 6.4f64.ln())
            } else {
                3.0 * freq / 200.0
            }
        }
    }
}

fn mel_to_hertz(mels: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 700.0 * (10f64.powf(mels / 2595.0) - 1.0),
        MelScale::Slaney => {
            if mels >= 15.0 {
                1000.0 * ((6.4f64.ln() / 27.0) * (mels - 15.0)).exp()
            } else {
                200.0 * mels / 3.0
            }
        }
    }
}

/// `numpy.linspace` with `endpoint=True` (last element exactly `stop`).
fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    assert!(n >= 2, "linspace needs at least 2 points");
    let step = (stop - start) / (n - 1) as f64;
    let mut v: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
    v[n - 1] = stop;
    v
}
