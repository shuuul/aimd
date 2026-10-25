//! Mel-frame -> audio-pad layout shared by the prompt builder and the MLX
//! tower (WS-06).
//!
//! The prompt needs the number of `<|audio_pad|>` tokens before the tower
//! runs, so the relation is computed from the mel frame count alone. It is
//! locked against the committed tokenizer fixtures and the encoder goldens.

/// Chunk length of the conv front-end: `n_window * 2`, hard-coded as `100`
/// inside mlx-audio's `_get_feat_extract_output_lengths`.
pub const CONV_CHUNK_FRAMES: usize = 100;

/// After-cnn frames produced by one full 100-frame chunk (50 -> 25 -> 13).
pub const FULL_CHUNK_AFTER_CNN: usize = 13;

/// Mel frames -> `<|audio_pad|>` token count, the exact
/// `_get_feat_extract_output_lengths` relation of mlx-audio (hard-coded to
/// its 100-frame conv chunking):
///
/// ```text
/// rem  = frames % 100
/// a1   = floor((rem - 1) / 2) + 1     // conv2d1 time output
/// a2   = floor((a1 - 1) / 2) + 1      // conv2d2
/// a3   = floor((a2 - 1) / 2) + 1      // conv2d3
/// pads = a3 + (frames / 100) * 13     // each full 100-frame chunk -> 13
/// ```
///
/// Floor division uses python semantics, so `rem == 0` contributes 0.
pub fn frames_to_audio_pads(frames: usize) -> usize {
    let rem = (frames % CONV_CHUNK_FRAMES) as i64;
    let a1 = (rem - 1).div_euclid(2) + 1;
    let a2 = (a1 - 1).div_euclid(2) + 1;
    let a3 = (a2 - 1).div_euclid(2) + 1;
    a3.max(0) as usize + (frames / CONV_CHUNK_FRAMES) * FULL_CHUNK_AFTER_CNN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_frames_to_audio_pads_formula() {
        // rem-only cases (frames < 100)
        assert_eq!(frames_to_audio_pads(1), 1);
        assert_eq!(frames_to_audio_pads(61), 8);
        assert_eq!(frames_to_audio_pads(99), 13);
        // whole chunks + remainder
        assert_eq!(frames_to_audio_pads(100), 13);
        assert_eq!(frames_to_audio_pads(150), 20);
        assert_eq!(frames_to_audio_pads(180), 23);
        assert_eq!(frames_to_audio_pads(200), 26);
        assert_eq!(frames_to_audio_pads(270), 35);
        assert_eq!(frames_to_audio_pads(400), 52);
        assert_eq!(frames_to_audio_pads(900), 117);
        assert_eq!(frames_to_audio_pads(1000), 130);
        // exact multiples contribute 0 from the remainder branch
        assert_eq!(frames_to_audio_pads(0), 0);
        // consistency with the generic conv chain for the 100-frame chunking
        let after_front_end = |len: usize| {
            let step = |n: usize| (n - 1) / 2 + 1;
            step(step(step(len)))
        };
        for frames in [1usize, 2, 61, 99, 100, 101, 150, 270, 400, 900, 1500] {
            let chunks = frames.div_ceil(100);
            let generic = (0..chunks)
                .map(|j| {
                    let r = frames % 100;
                    let len = if j + 1 == chunks && r != 0 { r } else { 100 };
                    after_front_end(len)
                })
                .sum::<usize>();
            assert_eq!(frames_to_audio_pads(frames), generic, "frames {frames}");
        }
    }
}
