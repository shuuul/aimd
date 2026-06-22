"""Platform and hardware detection helpers."""

from functools import lru_cache
import platform
import subprocess


@lru_cache(maxsize=1)
def is_apple_silicon() -> bool:
    """Return True when running on Apple Silicon macOS."""
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

    cpu_info = result.stdout.strip().lower()
    return "apple" in cpu_info and any(
        chip in cpu_info for chip in ("m1", "m2", "m3", "m4")
    )
