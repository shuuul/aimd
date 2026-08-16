"""Shared-filesystem allow-root policy for the local HTTP sidecar."""

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlparse

from aimd.interfaces.api.schemas import ProcessRequest


class PathAccessError(ValueError):
    """Raised when a sidecar request escapes its configured local roots."""


@dataclass(slots=True, frozen=True)
class PathPolicy:
    """Normalize local request paths and constrain them to canonical roots."""

    roots: tuple[Path, ...]

    @classmethod
    def from_roots(cls, roots: tuple[str | Path, ...]) -> "PathPolicy":
        canonical: list[Path] = []
        for root in roots:
            resolved = Path(root).expanduser().resolve(strict=True)
            if not resolved.is_dir():
                raise PathAccessError(f"Allowed root is not a directory: {resolved}")
            canonical.append(resolved)
        return cls(tuple(canonical))

    @classmethod
    def from_environment(cls) -> "PathPolicy":
        configured = os.getenv("AIMD_ALLOWED_ROOTS", "")
        roots = tuple(value for value in configured.split(os.pathsep) if value)
        return cls.from_roots(roots)

    def normalize_request(self, request: ProcessRequest) -> ProcessRequest:
        """Return a request whose local paths are absolute and allow-root checked."""
        values: dict[str, str] = {}
        if request.blob_id:
            pass
        elif request.input_source:
            parsed = urlparse(request.input_source)
            if parsed.scheme in {"http", "https"}:
                values["input_source"] = request.input_source
            elif "://" in request.input_source:
                raise PathAccessError(
                    "Only HTTP(S) URLs or local filesystem paths are allowed"
                )
            else:
                values["input_source"] = str(
                    self.normalize(request.input_source, "input")
                )
        else:
            raise PathAccessError("Exactly one of input_source or blob_id must be set")

        for field_name in ("output_file", "save_original", "cookies"):
            value = getattr(request, field_name)
            if value is not None:
                values[field_name] = str(self.normalize(value, field_name))
        return request.model_copy(update=values)

    def normalize(self, value: str | Path, purpose: str) -> Path:
        """Resolve symlinks and traversal before enforcing configured roots."""
        resolved = Path(value).expanduser().resolve(strict=False)
        if self.roots and not any(resolved.is_relative_to(root) for root in self.roots):
            raise PathAccessError(f"{purpose} path is outside configured allowed roots")
        return resolved
