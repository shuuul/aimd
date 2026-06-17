"""Ensure the default CLI stack does not pull optional HTTP/MCP dependencies."""

import pytest


@pytest.mark.parametrize("module_name", ["fastapi", "mcp"])
def test_cli_import_does_not_load_optional_dependencies(module_name: str) -> None:
    import sys

    for name in list(sys.modules):
        if name == module_name or name.startswith(f"{module_name}."):
            del sys.modules[name]

    import aimd.cli  # noqa: F401

    assert module_name not in sys.modules
