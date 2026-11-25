"""Cost measures (FP3 4.4): episode cost from M1 at the recorded price table, with cached and
uncached input priced separately. Prices come from configs/prices.json, never hard-coded."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PRICES = Path(__file__).resolve().parents[1] / "configs" / "prices.json"


def load_prices(path: Path = PRICES) -> dict[str, Any]:
    return json.loads(path.read_text())


def episode_cost(record: dict[str, Any], prices: dict[str, Any]) -> "float | None":
    """USD for one run record, or None if the price table has no numbers for the model."""
    p = prices["models"].get(record["model"], {})
    if any(p.get(k) is None for k in ("input", "cached_input", "output")):
        return None
    return (
        record["input_tokens"] * p["input"]
        + record["cached_input_tokens"] * p["cached_input"]
        + record["output_tokens"] * p["output"]
    ) / 1_000_000


def cost_per_success(records: list[dict[str, Any]], prices: dict[str, Any]) -> "float | None":
    """Total cost of all episodes in a cell (failures included) over its successes (FP2 4.5).
    None, "undefined", when there are no successes or a price is missing."""
    costs = [episode_cost(r, prices) for r in records]
    wins = sum(bool(r["success"]) for r in records)
    return None if wins == 0 or None in costs else sum(costs) / wins