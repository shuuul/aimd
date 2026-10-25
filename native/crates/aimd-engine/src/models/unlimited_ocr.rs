//! Unlimited-OCR graph (WS-07).
//!
//! Architecture (from `mlx-community/Unlimited-OCR-4bit/config.json`):
//! gundam preprocessing (1024 global view + 640 local crops) -> SAM ViT-B
//! vision tower (`deeplip_b_l`, 12 layers, width 768, window attention with
//! global layers at indexes 2/5/8/11) -> linear projector 2048->1280 ->
//! DeepSeek-V2-lite MoE decoder (12 layers, hidden 1280, 10-head MHA
//! v_head_dim 128, first layer dense, sliding_window 128, 64 routed experts
//! top-6 sigmoid greedy + 2 shared, vocab 129280).
//!
//! Generation: greedy with the sliding-window no-repeat n-gram guard
//! (n=35, window=128), max_tokens 8192, per-page single-image mode;
//! markdown normalization applied to raw output.

/// Placeholder for the WS-07 model surface.
pub struct UnlimitedOcrStub;
