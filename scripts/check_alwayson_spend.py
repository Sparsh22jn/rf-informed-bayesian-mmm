"""Prints per-channel coefficient of variation for daily always-on spend and
asserts out_of_home's is low -- the "chunky, weakly identified" pathology.

Usage: python scripts/check_alwayson_spend.py [seed]
"""

from __future__ import annotations

import sys

from mmm_sports.simulate.spend import ALWAYS_ON_CHANNELS, generate_alwayson_spend

OOH_CV_THRESHOLD = 0.8


def main(seed: int) -> None:
    df = generate_alwayson_spend(seed=seed)

    print("Per-channel CV (std / mean) of daily always-on spend:")
    cv = {}
    for ch in ALWAYS_ON_CHANNELS:
        series = df[ch]
        cv[ch] = series.std() / series.mean()
        print(f"  {ch}: mean=${series.mean():,.0f}  std=${series.std():,.0f}  CV={cv[ch]:.3f}")

    assert cv["out_of_home"] < OOH_CV_THRESHOLD, (
        f"out_of_home CV {cv['out_of_home']:.3f} should be below {OOH_CV_THRESHOLD}"
    )
    print(f"\nOK: out_of_home CV {cv['out_of_home']:.3f} < {OOH_CV_THRESHOLD} threshold")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed)
