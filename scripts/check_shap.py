"""Prints SHAP contribution shares next to the true shares (gap per media
channel), and saves a SHAP dependence scatter for tv_linear and ctv, each
colored by the other channel's spend -- the per-event complement to
3.2/3.3's averaged PDP/ALE curves.

Requires artifacts/forest.joblib -- run scripts/fit_forest.py first.

Usage: python scripts/check_shap.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from mmm_sports.models.forest import build_features, season_holdout_split
from mmm_sports.models.interpret_shap import compute_shap_values
from mmm_sports.simulate.truth import ALWAYS_ON, EVENT_TARGETED

ARTIFACT_PATH = Path("artifacts/forest.joblib")
DATA_PATH = Path("data/generated/events.parquet")
FIGS_DIR = Path("docs/figs")

BLUE_SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#104281"])
GRIDLINE = "#e1e0d9"
INK_MUTED = "#898781"

DEPENDENCE_PAIRS = (("tv_linear", "ctv"), ("ctv", "tv_linear"))


def main() -> None:
    assert ARTIFACT_PATH.exists(), f"{ARTIFACT_PATH} missing -- run scripts/fit_forest.py first"
    model = joblib.load(ARTIFACT_PATH)["model"]

    df = pd.read_parquet(DATA_PATH)
    train_df, _ = season_holdout_split(df)
    x_train = build_features(train_df)
    spend_columns = tuple(f"spend_{ch}" for ch in (*ALWAYS_ON, *EVENT_TARGETED))
    shap_df = compute_shap_values(model, x_train, zero_spend_columns=spend_columns)

    mu_mean = train_df["mu"].mean()
    print(f"{'channel':<15}{'shap_share':>12}{'true_share':>12}{'gap':>10}")
    for channel in (*ALWAYS_ON, *EVENT_TARGETED):
        feature = f"spend_{channel}"
        shap_share = shap_df[feature].mean() / mu_mean
        true_share = train_df[f"contrib_{channel}"].mean() / mu_mean
        gap = shap_share - true_share
        print(f"{channel:<15}{shap_share:>12.1%}{true_share:>12.1%}{gap:>+10.1%}")

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    for channel, other in DEPENDENCE_PAIRS:
        feature, other_feature = f"spend_{channel}", f"spend_{other}"

        fig, ax = plt.subplots(figsize=(7, 4.5))
        sc = ax.scatter(
            x_train[feature],
            shap_df[feature],
            c=x_train[other_feature],
            cmap=BLUE_SEQUENTIAL,
            s=20,
            alpha=0.85,
            edgecolor="none",
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(f"{other_feature} ($)")
        ax.axhline(0.0, color=INK_MUTED, linewidth=0.75)
        ax.set_xlabel(f"{feature} ($)")
        ax.set_ylabel("SHAP value (thousands of viewers)")
        ax.set_title(f"SHAP dependence -- {channel}, colored by {other}")
        ax.grid(axis="y", linewidth=0.75, alpha=0.6, color=GRIDLINE)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGS_DIR / f"shap_dependence_{channel}.png", dpi=120)
        plt.close(fig)

    print(f"\nSaved SHAP dependence plots to {FIGS_DIR}")


if __name__ == "__main__":
    main()
