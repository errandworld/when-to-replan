"""Reproducible seeds, independent of PYTHONHASHSEED."""

from __future__ import annotations

import hashlib


def stable_seed(*parts: object) -> int:
    """A reproducible 31-bit seed from arbitrary parts."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF
