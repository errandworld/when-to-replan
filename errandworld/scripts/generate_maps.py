"""Generate the fixed ErrandWorld maps (2 development maps, 5 test maps).

    python scripts/generate_maps.py            # writes data/maps/*.json and prints a summary
    python scripts/generate_maps.py --check    # verify the files on disk match the seeds

The development maps are used only for prompt development; the pilot and test sets share the
five test maps (ARCHITECTURE.md, Datasets). Seeds are fixed here so the maps are reproducible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from errandworld.maps import describe, generate_map, map_invariant_problems  # noqa: E402
from errandworld.scenario_io import load_map, save_map  # noqa: E402

MAP_DIR = ROOT / "data" / "maps"

# Shared parameters (design decision D6: sparse ring-with-chords maps so length comes from travel).
N_LOCATIONS = 15
AVG_DEGREE = 2.4
STATIONS_PER_TYPE = 2

MAP_SEEDS: dict[str, int] = {
    "dev_map_1": 1001,
    "dev_map_2": 1002,
    "test_map_1": 2001,
    "test_map_2": 2002,
    "test_map_3": 2003,
    "test_map_4": 2004,
    "test_map_5": 2005,
}


def build(map_id: str):
    m = generate_map(
        map_id, MAP_SEEDS[map_id], n_locations=N_LOCATIONS, avg_degree=AVG_DEGREE,
        stations_per_type=STATIONS_PER_TYPE,
    )
    problems = map_invariant_problems(m, stations_per_type=STATIONS_PER_TYPE)
    if problems:
        raise SystemExit(f"{map_id}: invariant violations: {problems}")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify existing files, write nothing")
    args = ap.parse_args()

    status = 0
    print(f"{'map':<12} {'locs':>4} {'roads':>5} {'deg':>5} {'diam':>4} {'bridges':>7}")
    for map_id in MAP_SEEDS:
        m = build(map_id)
        path = MAP_DIR / f"{map_id}.json"
        if args.check:
            if not path.exists() or load_map(path) != m:
                print(f"{map_id}: MISMATCH with {path}")
                status = 1
                continue
        else:
            save_map(m, path)
        d = describe(m)
        print(f"{map_id:<12} {d['locations']:>4} {d['roads']:>5} {d['avg_degree']:>5} "
              f"{d['diameter']:>4} {d['bridges']:>7}")
    if not args.check:
        print(f"wrote {len(MAP_SEEDS)} maps to {MAP_DIR.relative_to(ROOT)}/")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
