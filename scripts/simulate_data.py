"""Generate the synthetic event-level dataset and write it to parquet.

Usage: python scripts/simulate_data.py [seed]
"""

from __future__ import annotations

import sys
from pathlib import Path

from mmm_sports.simulate.dgp import generate_dataset

OUT_PATH = Path("data/generated/events.parquet")


def main(seed: int) -> None:
    df = generate_dataset(seed=seed)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df)} events to {OUT_PATH} (seed={seed})")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed)
