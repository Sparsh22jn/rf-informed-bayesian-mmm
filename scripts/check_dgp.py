"""Prints the response decomposition (baseline / controls / always-on /
event-targeted shares), asserts they're in the DESIGN.md §4 target band, and
prints the irreducible MAPE floor -- the true model scored against its own
output. Every later model's MAPE should be read against this number.

Usage: python scripts/check_dgp.py [seed]
"""

from __future__ import annotations

import sys

import numpy as np

from mmm_sports.simulate.dgp import generate_dataset
from mmm_sports.simulate.truth import ALWAYS_ON, EVENT_TARGETED

# (low, high) target share of mean(mu), per DESIGN.md §4.
TARGET_BANDS = {
    "baseline": (0.60, 0.70),
    "controls": (0.15, 0.25),
    "always_on": (0.05, 0.10),
    "event_targeted": (0.03, 0.08),
}


def main(seed: int) -> None:
    df = generate_dataset(seed=seed)
    mu_mean = df["mu"].mean()

    shares = {
        "baseline": df["contrib_baseline"].mean() / mu_mean,
        "controls": df["contrib_controls"].mean() / mu_mean,
        "always_on": sum(df[f"contrib_{ch}"].mean() for ch in ALWAYS_ON) / mu_mean,
        "event_targeted": sum(df[f"contrib_{ch}"].mean() for ch in EVENT_TARGETED) / mu_mean,
    }

    print("Response decomposition (share of mean mu):")
    for name, share in shares.items():
        low, high = TARGET_BANDS[name]
        print(f"  {name}: {share:.1%}  (target {low:.0%}-{high:.0%})")

    for name, share in shares.items():
        low, high = TARGET_BANDS[name]
        assert low <= share <= high, f"{name} share {share:.1%} outside target band {low:.0%}-{high:.0%}"

    mape_floor = np.mean(np.abs(df["viewership"] - df["mu"]) / df["viewership"])
    print(f"\nIrreducible MAPE floor (true model vs its own generated data): {mape_floor:.1%}")
    print("Every later model's holdout MAPE should be read against this number.")
    print("\nOK: all decomposition shares in band")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed)
