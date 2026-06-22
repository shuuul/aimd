"""OCR processing entrypoint scaffold.

The package exists so OCR can be added without growing aimd's transcript or
document conversion processors. A concrete engine will be wired here once the
OCR dependency and input contract are selected.
"""

from pathlib import Path


async def process_ocr(input_path: str | Path):
    """Process a scanned PDF or image with OCR.

    This is intentionally not wired into aimd's router yet.
    """
    raise NotImplementedError("OCR processing is not implemented yet")
