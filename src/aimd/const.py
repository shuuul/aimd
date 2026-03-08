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
TRANSCRIPTION_ENGINES = {"auto", "yap", "mlx", "qwen", "funasr"}

# Supported locales for yap transcription
YAP_SUPPORTED_LOCALES = {"zh_CN", "en_US"}

# Mapping from public language codes to yap locale codes.
# The CLI and public API use short codes (zh, en); yap needs full locales.
LANGUAGE_TO_YAP_LOCALE = {
    "zh": "zh_CN",
    "en": "en_US",
}

# FunASR models (CPU/CUDA, via funasr)
FUNASR_DEFAULT_MODEL = "FunAudioLLM/SenseVoiceSmall"
FUNASR_MODELS = {
    "FunAudioLLM/SenseVoiceSmall": "SenseVoice Small (234M, multilingual, default)",
    "FunAudioLLM/Fun-ASR-Nano-2512": "Fun-ASR-Nano (800M, 31 languages, lyric recognition)",
}

# mlx-audio STT models (Apple Silicon, via mlx_audio.stt)
MLX_AUDIO_DEFAULT_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
MLX_AUDIO_MODELS = {
    "mlx-community/Qwen3-ASR-0.6B-8bit": "Qwen3-ASR 0.6B (8-bit quantized)",
    "mlx-community/Qwen3-ASR-1.7B-8bit": "Qwen3-ASR 1.7B (8-bit quantized, default)",
    "mlx-community/parakeet-tdt-0.6b-v3": "Parakeet TDT 0.6B v3 (multilingual)",
}

# Qwen3-ASR models (Linux/CUDA, via qwen-asr)
QWEN_ASR_DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
QWEN_ASR_MODELS = {
    "Qwen/Qwen3-ASR-0.6B": "Qwen3-ASR 0.6B",
    "Qwen/Qwen3-ASR-1.7B": "Qwen3-ASR 1.7B (default)",
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
