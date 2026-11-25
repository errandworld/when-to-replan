"""Load API keys from `.env` so a run does not depend on the shell it was started from.
 
Only KEY=VALUE lines are read; blank lines, comments and empty values are skipped, and a variable
already set in the real environment wins. The file is never printed or logged.
"""
from __future__ import annotations
 
import os
from pathlib import Path
 
ROOT = Path(__file__).resolve().parents[1]
 
 
def load_env(path: "Path | None" = None) -> list[str]:
    """Set variables from `.env` and return the names it set."""
    path = path or ROOT / ".env"
    if not path.is_file():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded