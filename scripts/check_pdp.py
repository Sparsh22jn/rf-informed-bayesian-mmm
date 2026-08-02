"""Saves a PDP per media channel to docs/figs/ and prints the implied
response range (min -> max predicted viewership) per channel.

Requires artifacts/forest.joblib -- run scripts/fit_forest.py first.

Usage: python scripts/check_pdp.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from mmm_sports.models.forest import build_features, season_holdout_split
from mmm_sports.models.interpret import compute_pdp
from mmm_sports.simulate.truth import ALWAYS_ON, EVENT_TARGETED

ARTIFACT_PATH = Path("artifacts/forest.joblib")
DATA_PATH = Path("data/generated/events.parquet")
FIGS_DIR = Path("docs/figs")

COLOR_BLUE = "#2a78d6"
GRIDLINE = "#e1e0d9"


def main() -> None:
    assert ARTIFACT_PATH.exists(), f"{ARTIFACT_PATH} missing -- run scripts/fit_forest.py first"
    model = joblib.load(ARTIFACT_PATH)["model"]

    df = pd.read_parquet(DATA_PATH)
    train_df, _ = season_holdout_split(df)
    x_train = build_features(train_df)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    print("Implied response range per channel (PDP min -> max, thousands of viewers):")
    for channel in (*ALWAYS_ON, *EVENT_TARGETED):
        feature = f"spend_{channel}"
        grid, avg_pred = compute_pdp(model, x_train, feature)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grid, avg_pred, color=COLOR_BLUE, linewidth=2)
        ax.set_xlabel(f"{feature} ($)")
        ax.set_ylabel("predicted viewership (thousands)")
        ax.set_title(f"PDP -- {channel}")
        ax.grid(axis="y", linewidth=0.75, alpha=0.6, color=GRIDLINE)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGS_DIR / f"pdp_{channel}.png", dpi=120)
        plt.close(fig)

        direction = "+" if avg_pred[-1] >= avg_pred[0] else "-"
        print(
            f"  {channel}: low-spend={avg_pred[0]:.1f} -> high-spend={avg_pred[-1]:.1f}"
            f" ({direction}, range {avg_pred.max() - avg_pred.min():.1f})"
        )

    print(f"\nSaved 6 PDPs to {FIGS_DIR}")


if __name__ == "__main__":
    main()
