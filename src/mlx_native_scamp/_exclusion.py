from __future__ import annotations


def self_join_exclusion(window: int) -> int:
    """Return SCAMP's half-open self-join exclusion radius."""

    return (window + 3) // 4
