"""FastAPI application entrypoint."""

import os

from .adapters.http.app import (
    EngineCapabilityResponse,
    EnginesResponse,
    HealthResponse,
    ProcessRequest,
    ProcessResponse,
    create_app,
)

app = create_app()


def main() -> None:
    """Run FastAPI service via uvicorn."""
    import uvicorn

    host = os.getenv("AIMD_API_HOST", "127.0.0.1")
    port = int(os.getenv("AIMD_API_PORT", "8000"))
    uvicorn.run("aimd.api:app", host=host, port=port, reload=False)


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
