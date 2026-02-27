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
TRANSCRIPTION_ENGINES = {"auto", "yap", "mlx", "cuda", "cpu"}

# Supported locales for yap transcription
YAP_SUPPORTED_LOCALES = {"zh_CN", "en_US"}

# Mapping from Whisper language codes to yap locale codes.
# The CLI and public API use short codes (zh, en); yap needs full locales.
LANGUAGE_TO_YAP_LOCALE = {
    "zh": "zh_CN",
    "en": "en_US",
}

# Whisper model sizes (for faster-whisper on CPU/CUDA)
WHISPER_MODEL_SIZES = {"tiny", "base", "small", "medium", "large-v3-turbo"}

# mlx-audio STT models (Apple Silicon, via mlx_audio.stt)
MLX_AUDIO_DEFAULT_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
MLX_AUDIO_MODELS = {
    "mlx-community/Qwen3-ASR-0.6B-8bit": "Qwen3-ASR 0.6B (8-bit quantized)",
    "mlx-community/Qwen3-ASR-1.7B-8bit": "Qwen3-ASR 1.7B (8-bit quantized, default)",
    "mlx-community/whisper-large-v3-turbo-asr-fp16": "Whisper large-v3-turbo (fp16)",
    "mlx-community/parakeet-tdt-0.6b-v3": "Parakeet TDT 0.6B v3 (multilingual)",
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
