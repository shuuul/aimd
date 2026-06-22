"""ASR constants."""

AUDIO_FILE_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".opus",
    ".wma",
    ".webm",
    ".mp4a",
}

VIDEO_FILE_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".ts",
    ".m4v",
}

AUDIO_EXTENSIONS = AUDIO_FILE_EXTENSIONS | VIDEO_FILE_EXTENSIONS

MLX_AUDIO_DEFAULT_MODEL = "mlx-community/Qwen3-ASR-1.7B-4bit"
MLX_AUDIO_MODELS = {
    "mlx-community/Qwen3-ASR-1.7B-4bit": "Qwen3-ASR 1.7B (4-bit quantized, default)",
    "mlx-community/Qwen3-ASR-1.7B-6bit": "Qwen3-ASR 1.7B (6-bit quantized)",
    "mlx-community/Qwen3-ASR-1.7B-8bit": "Qwen3-ASR 1.7B (8-bit quantized)",
    "mlx-community/Qwen3-ASR-0.6B-4bit": "Qwen3-ASR 0.6B (4-bit quantized)",
    "mlx-community/Qwen3-ASR-0.6B-6bit": "Qwen3-ASR 0.6B (6-bit quantized)",
    "mlx-community/Qwen3-ASR-0.6B-8bit": "Qwen3-ASR 0.6B (8-bit quantized)",
    "mlx-community/whisper-large-v3-turbo-asr-fp16": "Whisper large-v3-turbo ASR (fp16)",
    "distil-whisper/distil-large-v3": "Distil-Whisper large-v3",
    "mlx-community/parakeet-tdt-0.6b-v3": "NVIDIA Parakeet TDT 0.6B v3",
    "mlx-community/nemotron-3.5-asr-streaming-0.6b": "NVIDIA Nemotron 3.5 ASR streaming 0.6B",
    "mlx-community/Voxtral-Mini-3B-2507-bf16": "Voxtral Mini 3B (bf16)",
    "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit": "Voxtral Mini 4B Realtime (4-bit)",
    "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16": "Voxtral Mini 4B Realtime (fp16)",
    "mlx-community/VibeVoice-ASR-bf16": "VibeVoice-ASR (bf16, diarization/timestamps)",
    "mlx-community/Qwen2-Audio-7B-Instruct-4bit": "Qwen2-Audio 7B Instruct (4-bit)",
}

QWEN_ASR_DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
QWEN_ASR_MODELS = {
    "Qwen/Qwen3-ASR-1.7B": "Qwen3-ASR 1.7B (default)",
    "Qwen/Qwen3-ASR-0.6B": "Qwen3-ASR 0.6B",
}
