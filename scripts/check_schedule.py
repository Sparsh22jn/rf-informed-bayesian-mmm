"""Prints schedule summary stats and saves a calendar density plot.

Usage: python scripts/check_schedule.py [seed]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from mmm_sports.simulate.schedule import generate_schedule

FIGS_DIR = Path("docs/figs")


def main(seed: int) -> None:
    df = generate_schedule(seed=seed)

    print(f"Total events: {len(df)}")
    print("\nEvents per season:")
    print(df.groupby("season").size().to_string())

    print("\nn_events_on_date distribution (count of dates with N events):")
    dist = df.drop_duplicates("date")["n_events_on_date"].value_counts().sort_index()
    print(dist.to_string())

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3 * df["season"].nunique()))
    for season, group in df.groupby("season"):
        ax.scatter(
            group["date"],
            [season] * len(group),
            s=group["n_events_on_date"] * 8,
            alpha=0.5,
            label=f"season {season}",
        )
    ax.set_yticks(sorted(df["season"].unique()))
    ax.set_xlabel("date")
    ax.set_ylabel("season")
    ax.set_title("Event calendar density (marker size = events on that date)")
    fig.tight_layout()
    out_path = FIGS_DIR / "schedule_density.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed)
