"""Saves a PDP-vs-ALE overlay for tv_linear and ctv (the correlated pair)
and prints the max divergence between the two curves for each. Also saves
tv_linear's ALE curve computed separately within each tentpole tier --
holding the real confounder fixed, to check whether tv_linear's backwards
sign (task 3.3 finding) is a transitive confound through tentpole_tier
rather than a direct tv_linear/ctv effect.

Requires artifacts/forest.joblib -- run scripts/fit_forest.py first.

Usage: python scripts/check_ale.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from mmm_sports.models.forest import build_features, season_holdout_split
from mmm_sports.models.interpret import compute_ale, compute_pdp
from mmm_sports.simulate.schedule import TENTPOLE_TIERS

ARTIFACT_PATH = Path("artifacts/forest.joblib")
DATA_PATH = Path("data/generated/events.parquet")
FIGS_DIR = Path("docs/figs")

CHANNELS = ("tv_linear", "ctv")
COLOR_PDP = "#2a78d6"
COLOR_ALE = "#eb6834"
GRIDLINE = "#e1e0d9"

# Ordinal blue ramp (steps 250/400/550/700), same convention as the EDA
# notebook -- lightest for "regular", darkest for "championship".
TIER_RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")


def main() -> None:
    assert ARTIFACT_PATH.exists(), f"{ARTIFACT_PATH} missing -- run scripts/fit_forest.py first"
    model = joblib.load(ARTIFACT_PATH)["model"]

    df = pd.read_parquet(DATA_PATH)
    train_df, _ = season_holdout_split(df)
    x_train = build_features(train_df)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    print("Max |PDP - ALE| divergence (centered, thousands of viewers):")
    for channel in CHANNELS:
        feature = f"spend_{channel}"
        edges, ale_curve = compute_ale(model, x_train, feature)
        grid, pdp_curve = compute_pdp(model, x_train, feature, grid_values=edges)
        pdp_centered = pdp_curve - pdp_curve.mean()

        divergence = (ale_curve - pdp_centered)
        max_divergence = float(abs(divergence).max())

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(grid, pdp_centered, color=COLOR_PDP, linewidth=2, label="PDP (centered)")
        ax.plot(edges, ale_curve, color=COLOR_ALE, linewidth=2, label="ALE (centered)")
        ax.set_xlabel(f"{feature} ($)")
        ax.set_ylabel("effect on viewership (thousands, centered)")
        ax.set_title(f"PDP vs ALE -- {channel}")
        ax.legend(frameon=False, loc="upper left")
        ax.grid(axis="y", linewidth=0.75, alpha=0.6, color=GRIDLINE)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGS_DIR / f"pdp_ale_{channel}.png", dpi=120)
        plt.close(fig)

        print(f"  {channel}: {max_divergence:.1f}")

    print(f"\nSaved PDP/ALE overlays to {FIGS_DIR}")

    print("\ntv_linear ALE, stratified by tentpole tier (holding the real confounder fixed):")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tier, color in zip(TENTPOLE_TIERS, TIER_RAMP):
        x_tier = x_train[train_df["tentpole_tier"].to_numpy() == tier]
        n_bins = max(3, min(15, len(x_tier) // 6))
        edges, ale_curve = compute_ale(model, x_tier, "spend_tv_linear", n_bins=n_bins)
        direction = "+" if ale_curve[-1] >= ale_curve[0] else "-"
        print(f"  {tier} (n={len(x_tier)}, {n_bins} bins): {direction}, low={ale_curve[0]:.1f} high={ale_curve[-1]:.1f}")
        ax.plot(edges, ale_curve, color=color, linewidth=2, label=f"{tier} (n={len(x_tier)})")
    ax.set_xlabel("spend_tv_linear ($)")
    ax.set_ylabel("centered ALE effect (thousands)")
    ax.set_title("tv_linear ALE by tentpole tier")
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="y", linewidth=0.75, alpha=0.6, color=GRIDLINE)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "ale_tv_linear_by_tier.png", dpi=120)
    plt.close(fig)
    print(f"Saved {FIGS_DIR / 'ale_tv_linear_by_tier.png'}")


if __name__ == "__main__":
    main()
