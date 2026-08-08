"""Saves the empirical RF response curve per channel with the true curve
overlaid, and prints each curve's implied range. For always-on channels the
true curve is an honest mismatch, not a bug: the RF only ever sees raw
same-day spend, never the true adstocked/diluted exposure (DESIGN.md §4),
so its curve and the true Hill curve describe different quantities.

Requires artifacts/forest.joblib -- run scripts/fit_forest.py first.

Usage: python scripts/check_rf_curves.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from mmm_sports.models.forest import build_features, season_holdout_split
from mmm_sports.models.interpret_curves import compute_empirical_curve
from mmm_sports.simulate.truth import ALWAYS_ON, EVENT_TARGETED, TRUTH
from mmm_sports.transforms import hill_saturation

ARTIFACT_PATH = Path("artifacts/forest.joblib")
DATA_PATH = Path("data/generated/events.parquet")
FIGS_DIR = Path("docs/figs")

COLOR_RF = "#2a78d6"
COLOR_TRUE = "#eb6834"
GRIDLINE = "#e1e0d9"


def main() -> None:
    assert ARTIFACT_PATH.exists(), f"{ARTIFACT_PATH} missing -- run scripts/fit_forest.py first"
    model = joblib.load(ARTIFACT_PATH)["model"]

    df = pd.read_parquet(DATA_PATH)
    train_df, _ = season_holdout_split(df)
    x_train = build_features(train_df)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'channel':<15}{'rf range':>20}{'true range':>20}")
    for channel in (*ALWAYS_ON, *EVENT_TARGETED):
        feature = f"spend_{channel}"
        grid, rf_curve = compute_empirical_curve(
            model, x_train, feature, dummy_prefixes=("broadcaster",)
        )
        true_curve = TRUTH.beta[channel] * hill_saturation(grid, TRUTH.k[channel], TRUTH.s[channel])
        rf_curve_baselined = rf_curve - rf_curve[0]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(grid, rf_curve_baselined, color=COLOR_RF, linewidth=2, label="RF (empirical, baselined at 0)")
        ax.plot(grid, true_curve, color=COLOR_TRUE, linewidth=2, label="true Hill curve")
        ax.set_xlabel(f"{feature} ($)")
        ax.set_ylabel("implied viewership contribution (thousands)")
        ax.set_title(f"Empirical response curve -- {channel}")
        ax.legend(frameon=False, loc="best")
        ax.grid(axis="y", linewidth=0.75, alpha=0.6, color=GRIDLINE)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGS_DIR / f"rf_curve_{channel}.png", dpi=120)
        plt.close(fig)

        rf_range = f"{rf_curve_baselined.min():.0f} -> {rf_curve_baselined.max():.0f}"
        true_range = f"{true_curve.min():.0f} -> {true_curve.max():.0f}"
        print(f"{channel:<15}{rf_range:>20}{true_range:>20}")

    print(f"\nSaved 6 empirical response curves to {FIGS_DIR}")


if __name__ == "__main__":
    main()
