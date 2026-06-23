"""ASR model adapter protocol."""

from pathlib import Path
from typing import Protocol


class ASRModel(Protocol):
    """Common interface for concrete ASR model adapters."""

    model_id: str

    async def transcribe(
        self,
        file_path: Path,
        *,
        language: str | None = None,
        temp_dir: Path | None = None,
    ) -> str:
        """Transcribe one local audio/video file."""
