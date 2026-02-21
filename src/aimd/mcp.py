"""MCP server entrypoint."""

from .adapters.mcp.server import healthz, list_engines, mcp, main, process_input

__all__ = ["mcp", "healthz", "list_engines", "process_input", "main"]


if __name__ == "__main__":
    main()
