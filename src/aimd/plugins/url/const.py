"""Media URL/subtitle constants."""

__all__ = [
    "CHINESE_SUBTITLE_LANGUAGES",
    "ENGLISH_SUBTITLE_LANGUAGES",
    "FORBIDDEN_SUBTITLE_LANGUAGES",
]

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

FORBIDDEN_SUBTITLE_LANGUAGES = ["danmaku"]
