"""Constants used throughout the aimd package."""

# =============================================================================
# FILE PROCESSING CONSTANTS
# =============================================================================

# Supported audio file extensions
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".mp4"}

# EPUB file extensions (require special handling for image extraction)
EPUB_EXTENSIONS = {".epub", ".mobi", ".azw3"}

# Supported transcription engines
TRANSCRIPTION_ENGINES = {"auto", "yap", "mlx", "cuda", "cpu"}

# Supported locales for yap transcription
YAP_SUPPORTED_LOCALES = {"zh_CN", "en_US"}

# Whisper model sizes (for faster-whisper and mlx-whisper)
WHISPER_MODEL_SIZES = {"tiny", "base", "small", "medium", "large-v3-turbo"}

# MLX model mappings
MLX_MODEL_MAPPINGS = {
    "tiny": "whisper-tiny-mlx",
    "base": "whisper-base-mlx",
    "small": "whisper-small-mlx",
    "medium": "whisper-medium-mlx-8bit",
    "large-v3-turbo": "whisper-large-v3-turbo",
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
    "zh-CN",
    "zh",
    "ai-zh",
    "zh-TW",
    "zh-HK",
]

# Forbidden subtitle types
FORBIDDEN_SUBTITLE_LANGUAGES = ["danmaku"]
