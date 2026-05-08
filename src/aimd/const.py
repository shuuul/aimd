"""Constants used throughout the aimd package."""

# =============================================================================
# FILE PROCESSING CONSTANTS
# =============================================================================

# Supported audio file extensions (including video formats that can be transcribed)
AUDIO_EXTENSIONS = {
    # Audio formats
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".opus",
    ".wma",
    ".webm",
    # Video formats (will extract audio for transcription)
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".ts",
    ".m4v",
}

# EPUB file extensions (require special handling for image extraction)
EPUB_EXTENSIONS = {".epub", ".mobi", ".azw3"}

# Supported transcription engines
TRANSCRIPTION_ENGINES = {"auto", "mlx", "qwen"}

# mlx-audio STT models (Apple Silicon, via mlx_audio.stt)
MLX_AUDIO_DEFAULT_MODEL = "mlx-community/Qwen3-ASR-1.7B-4bit"
MLX_AUDIO_MODELS = {
    "mlx-community/Qwen3-ASR-1.7B-4bit": "Qwen3-ASR 1.7B (4-bit quantized, default)",
    "mlx-community/Qwen3-ASR-1.7B-6bit": "Qwen3-ASR 1.7B (6-bit quantized)",
    "mlx-community/Qwen3-ASR-1.7B-8bit": "Qwen3-ASR 1.7B (8-bit quantized)",
    "mlx-community/Qwen3-ASR-0.6B-4bit": "Qwen3-ASR 0.6B (4-bit quantized)",
    "mlx-community/Qwen3-ASR-0.6B-6bit": "Qwen3-ASR 0.6B (6-bit quantized)",
    "mlx-community/Qwen3-ASR-0.6B-8bit": "Qwen3-ASR 0.6B (8-bit quantized)",
}

# Qwen3-ASR models (Linux/CUDA, via qwen-asr)
QWEN_ASR_DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
QWEN_ASR_MODELS = {
    "Qwen/Qwen3-ASR-1.7B": "Qwen3-ASR 1.7B (default)",
    "Qwen/Qwen3-ASR-0.6B": "Qwen3-ASR 0.6B",
}

# =============================================================================
# URL/VIDEO PROCESSING CONSTANTS
# =============================================================================

# Subtitle language preferences
ENGLISH_SUBTITLE_LANGUAGES = [
    "en-orig",
    "en",
    "en-US",
    "en-GB",
    "ai-en",
]

CHINESE_SUBTITLE_LANGUAGES = [
    "zh-orig",
    "zh-Hans",
    "zh-Hant",
    "zh-CN",
    "zh",
    "ai-zh",
    "zh-TW",
    "zh-HK",
]

# Forbidden subtitle types
FORBIDDEN_SUBTITLE_LANGUAGES = ["danmaku"]
