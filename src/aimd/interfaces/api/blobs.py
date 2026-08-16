"""Sidecar-owned blob storage for desktop workers."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from uuid import UUID, uuid4


class BlobNotFoundError(LookupError):
    """Raised when a blob identifier is unknown."""


class BlobIdError(ValueError):
    """Raised when a blob identifier is not a UUID."""


_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(filename: str | None) -> str:
    raw = Path(filename or "").name.strip()
    if not raw or raw in {".", ".."}:
        return "input.bin"
    cleaned = _UNSAFE_NAME.sub("_", raw).strip("._")
    return cleaned or "input.bin"


def parse_blob_id(value: str) -> str:
    """Return a canonical UUID string or raise BlobIdError."""
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise BlobIdError("blob_id must be a UUID") from exc


@dataclass(slots=True)
class StoredBlob:
    blob_id: str
    path: Path
    filename: str
    size: int


class BlobStore:
    """Store uploaded bytes under a sidecar-owned directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_environment(cls) -> "BlobStore":
        configured = os.getenv("AIMD_BLOB_DIR")
        if configured:
            return cls(configured)
        return cls(Path(tempfile.mkdtemp(prefix="aimd-blobs-")))

    def put(
        self,
        data: bytes,
        *,
        filename: str | None = None,
        blob_id: str | None = None,
    ) -> StoredBlob:
        stored_id = parse_blob_id(blob_id) if blob_id else str(uuid4())
        safe_name = _safe_filename(filename)
        blob_dir = self.root / stored_id
        blob_dir.mkdir(parents=True, exist_ok=True)
        path = blob_dir / safe_name
        path.write_bytes(data)
        meta = blob_dir / "meta.json"
        meta.write_text(
            json.dumps({"filename": safe_name}, indent=None), encoding="utf-8"
        )
        return StoredBlob(
            blob_id=stored_id, path=path, filename=safe_name, size=len(data)
        )

    def resolve(self, blob_id: str) -> StoredBlob:
        stored_id = parse_blob_id(blob_id)
        blob_dir = self.root / stored_id
        if not blob_dir.is_dir():
            raise BlobNotFoundError(stored_id)
        filename = self._filename_for(blob_dir)
        path = blob_dir / filename
        if not path.is_file():
            raise BlobNotFoundError(stored_id)
        return StoredBlob(
            blob_id=stored_id,
            path=path,
            filename=filename,
            size=path.stat().st_size,
        )

    @staticmethod
    def _filename_for(blob_dir: Path) -> str:
        meta_path = blob_dir / "meta.json"
        if meta_path.is_file():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            name = payload.get("filename")
            if isinstance(name, str) and name and (blob_dir / name).is_file():
                return name
        for child in sorted(blob_dir.iterdir()):
            if child.is_file() and child.name != "meta.json":
                return child.name
        return "input.bin"
