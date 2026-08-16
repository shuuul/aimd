"""HTTP API module for aimd."""

from importlib import import_module

__all__ = [
    "CancellationStatus",
    "HealthResponse",
    "JobCreated",
    "JobEvent",
    "JobSnapshot",
    "JobStage",
    "JobState",
    "ProcessArtifact",
    "ProcessRequest",
    "ProcessResponse",
    "app",
    "create_app",
    "main",
]


def __getattr__(name: str):
    if name in {"app", "create_app", "main"}:
        app_module = import_module(".app", __name__)

        return getattr(app_module, name)
    if name in __all__:
        schemas_module = import_module(".schemas", __name__)
        return getattr(schemas_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
