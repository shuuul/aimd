"""Shared version parsing helpers."""


def parse_version_tuple(version: str) -> tuple[int, ...]:
    """Parse a dotted version prefix into comparable ints."""
    parts: list[int] = []
    for piece in version.split("+")[0].split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)
