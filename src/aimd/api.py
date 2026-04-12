"""FastAPI application entrypoint (optional dependency: ``aimd[api]``)."""

import os

__all__ = [
    "app",
    "create_app",
    "main",
    "HealthResponse",
    "EngineCapabilityResponse",
    "EnginesResponse",
    "ProcessRequest",
    "ProcessResponse",
]

_app = None


def __getattr__(name: str):
    if name == "app":
        global _app
        if _app is None:
            from .adapters.http.app import create_app

            _app = create_app()
        return _app
    from .adapters.http import app as http_module

    if hasattr(http_module, name):
        return getattr(http_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    """Run FastAPI service via uvicorn."""
    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit(
            "aimd-api requires optional dependencies. Install: "
            "uv sync --extra api   or   pip install 'aimd[api]'"
        ) from e
    host = os.getenv("AIMD_API_HOST", "127.0.0.1")
    port = int(os.getenv("AIMD_API_PORT", "8000"))
    uvicorn.run("aimd.api:app", host=host, port=port, reload=False)
