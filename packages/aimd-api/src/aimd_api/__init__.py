"""HTTP API package for aimd."""

from importlib import import_module

__all__ = [
    "EngineCapabilityResponse",
    "EnginesResponse",
    "HealthResponse",
    "ProcessRequest",
    "ProcessResponse",
    "app",
    "create_app",
    "main",
]


def __getattr__(name: str):
    if name in __all__:
        app_module = import_module(".app", __name__)

        return getattr(app_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
