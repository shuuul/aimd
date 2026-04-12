"""Ensure the default CLI stack does not pull optional HTTP/MCP dependencies."""


def test_cli_import_does_not_load_fastapi() -> None:
    import sys

    for name in list(sys.modules):
        if name == "fastapi" or name.startswith("fastapi."):
            del sys.modules[name]

    import aimd.cli  # noqa: F401

    assert "fastapi" not in sys.modules


def test_cli_import_does_not_load_mcp() -> None:
    import sys

    for name in list(sys.modules):
        if name == "mcp" or name.startswith("mcp."):
            del sys.modules[name]

    import aimd.cli  # noqa: F401

    assert "mcp" not in sys.modules
