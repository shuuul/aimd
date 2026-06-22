from importlib.metadata import entry_points


def test_aimd_markitdown_plugins_are_discoverable() -> None:
    plugin_names = {
        entry_point.name for entry_point in entry_points(group="markitdown.plugin")
    }

    assert "media" in plugin_names
    assert "book" in plugin_names
