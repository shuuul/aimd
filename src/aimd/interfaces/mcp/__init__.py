"""MCP server module for aimd."""

from .app import healthz, main, mcp, process_input

__all__ = ["healthz", "main", "mcp", "process_input"]
