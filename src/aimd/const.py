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
TRANSCRIPTION_ENGINES = {"auto", "mlx", "funasr"}

# FunASR models (CPU/CUDA, via funasr)
FUNASR_DEFAULT_MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"
FUNASR_MODELS = {
    "FunAudioLLM/Fun-ASR-Nano-2512": "Fun-ASR-Nano (800M, 31 languages, lyric recognition, default)",
    "FunAudioLLM/SenseVoiceSmall": "SenseVoice Small (234M, multilingual)",
}

# mlx-audio-plus Fun-ASR-Nano (Apple Silicon only; see mlx_engine)
MLX_AUDIO_DEFAULT_MODEL = "mlx-community/Fun-ASR-Nano-2512-4bit"
MLX_AUDIO_MODELS = {
    "mlx-community/Fun-ASR-Nano-2512-4bit": "Fun-ASR-Nano 4-bit (mlx-audio-plus, multilingual)",
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
