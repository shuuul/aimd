"""HTTP API package for aimd."""

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
        from . import app as app_module

        return getattr(app_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
