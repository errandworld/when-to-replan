"""Prompt files, loaded by name."""
from __future__ import annotations

from pathlib import Path

DIRECTORY = Path(__file__).parent


def load(name: str, **fields: str) -> str:
    """The prompt `name`.md, with each {field} replaced by the value given."""
    text = (DIRECTORY / f"{name}.md").read_text()
    for key, value in fields.items():
        text = text.replace("{" + key + "}", value)
    return text
