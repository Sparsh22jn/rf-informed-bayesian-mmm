"""Prints control distributions, their correlation matrix, and mean
viewership by broadcaster level and by tentpole tier.

Usage: python scripts/check_controls.py [seed]
"""

from __future__ import annotations

import sys

from mmm_sports.simulate.dgp import generate_dataset
from mmm_sports.simulate.schedule import BROADCASTERS, TENTPOLE_TIERS
from mmm_sports.simulate.truth import SCALAR_CONTROLS

CONTINUOUS_CONTROLS = ("tv_availability", "team_interest", "star_interest", "competitiveness")


def main(seed: int) -> None:
    df = generate_dataset(seed=seed)

    print("Control distributions:")
    print(df[list(CONTINUOUS_CONTROLS)].describe().round(1).to_string())

    print("\nCorrelation matrix (continuous controls):")
    print(df[list(CONTINUOUS_CONTROLS)].corr().round(3).to_string())

    print("\nMean viewership by broadcaster level:")
    by_broadcaster = df.groupby("broadcaster", observed=True)["viewership"].mean().reindex(BROADCASTERS)
    print(by_broadcaster.round(0).to_string())

    print("\nMean viewership by tentpole tier:")
    by_tier = df.groupby("tentpole_tier", observed=True)["viewership"].mean().reindex(TENTPOLE_TIERS)
    print(by_tier.round(0).to_string())

    assert set(SCALAR_CONTROLS) == {*CONTINUOUS_CONTROLS, "is_weekend", "tentpole_tier"}
    print(f"\nOK: all {len(SCALAR_CONTROLS)} scalar controls plus broadcaster present")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed)
