"""MCP server entrypoint (optional dependency: ``aimd[mcp]``)."""

__all__ = ["mcp", "healthz", "list_engines", "process_input", "main"]


def __getattr__(name: str):
    try:
        from .adapters.mcp import server as mcp_server
    except ImportError as e:
        raise AttributeError(
            "MCP adapter requires optional dependencies. Install: "
            "uv sync --extra mcp   or   pip install 'aimd[mcp]'"
        ) from e
    return getattr(mcp_server, name)


def main() -> None:
    """Run MCP server over stdio."""
    try:
        from .adapters.mcp.server import main as _run
    except ImportError as e:
        raise SystemExit(
            "aimd-mcp requires optional dependencies. Install: "
            "uv sync --extra mcp   or   pip install 'aimd[mcp]'"
        ) from e
    _run()


if __name__ == "__main__":
    main()
