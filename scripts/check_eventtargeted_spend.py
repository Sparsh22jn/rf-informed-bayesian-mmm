"""Prints the event-targeted spend correlation matrix (incl. tv_linear) and
mean spend by tentpole tier; asserts the TV/CTV correlation and the
tier-concentration confound DESIGN.md §3 calls for.

Usage: python scripts/check_eventtargeted_spend.py [seed]
"""

from __future__ import annotations

import sys

import pandas as pd

from mmm_sports.simulate.event_spend import (
    EVENT_TARGETED_CHANNELS,
    generate_eventtargeted_spend,
)
from mmm_sports.simulate.schedule import TENTPOLE_TIERS, generate_schedule
from mmm_sports.simulate.spend import generate_alwayson_spend

TV_CTV_CORR_THRESHOLD = 0.5


def main(seed: int) -> None:
    schedule = generate_schedule(seed=seed)
    alwayson = generate_alwayson_spend(seed=seed)
    event_targeted = generate_eventtargeted_spend(schedule, alwayson, seed=seed)

    tv_linear_at_event = (
        alwayson.set_index("day_idx")["tv_linear"].reindex(schedule["day_idx"]).to_numpy()
    )
    merged = event_targeted.copy()
    merged["tv_linear"] = tv_linear_at_event
    merged["tentpole_tier"] = schedule["tentpole_tier"].to_numpy()

    corr = merged[["tv_linear", *EVENT_TARGETED_CHANNELS]].corr()
    print("Correlation matrix:")
    print(corr.round(3).to_string())

    print("\nMean event-targeted spend by tentpole tier:")
    by_tier = merged.groupby("tentpole_tier", observed=True)[list(EVENT_TARGETED_CHANNELS)].mean()
    by_tier = by_tier.reindex(TENTPOLE_TIERS)
    print(by_tier.round(0).to_string())

    tv_ctv_corr = corr.loc["tv_linear", "ctv"]
    assert tv_ctv_corr > TV_CTV_CORR_THRESHOLD, (
        f"corr(tv_linear, ctv) {tv_ctv_corr:.3f} should exceed {TV_CTV_CORR_THRESHOLD}"
    )
    for ch in EVENT_TARGETED_CHANNELS:
        means = by_tier[ch].to_numpy()
        assert (pd.Series(means).diff().dropna() > 0).all(), (
            f"{ch} mean spend should rise monotonically with tentpole tier: {means}"
        )

    print(f"\nOK: corr(tv_linear, ctv)={tv_ctv_corr:.3f} > {TV_CTV_CORR_THRESHOLD}, "
          f"all event-targeted channels rise monotonically with tier")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed)
