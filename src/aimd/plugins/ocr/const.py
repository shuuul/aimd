"""OCR file-extension constants."""

IMAGE_FILE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"})
OCR_DOCUMENT_EXTENSIONS = frozenset({".pdf"})
OCR_EXTENSIONS = IMAGE_FILE_EXTENSIONS | OCR_DOCUMENT_EXTENSIONS
