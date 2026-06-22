"""Constants used throughout the aimd package."""

# =============================================================================
# FILE PROCESSING CONSTANTS
# =============================================================================

from aimd.media.const import AUDIO_EXTENSIONS

# Ebook file extensions (require special handling for image extraction)
BOOK_EXTENSIONS = {".epub", ".mobi", ".azw3"}

MARKITDOWN_FILE_EXTENSIONS = (
    AUDIO_EXTENSIONS
    | BOOK_EXTENSIONS
    | {
        ".csv",
        ".doc",
        ".docx",
        ".html",
        ".htm",
        ".json",
        ".md",
        ".pdf",
        ".ppt",
        ".pptx",
        ".txt",
        ".xls",
        ".xlsx",
        ".xml",
    }
)
