"""Prints SHAP per-channel share tables sliced by broadcaster, season,
weekend, margin bin, and tentpole tier -- extending task 3.4's headline
comparison to see whether the tv_linear/ctv gap concentrates in particular
slices (e.g. does over-attribution to ctv track the tentpole confound?).

Requires artifacts/forest.joblib -- run scripts/fit_forest.py first.

Usage: python scripts/check_shap_slices.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from mmm_sports.models.forest import build_features, season_holdout_split
from mmm_sports.models.interpret_shap import compute_shap_values
from mmm_sports.simulate.schedule import BROADCASTERS, TENTPOLE_TIERS
from mmm_sports.simulate.truth import ALWAYS_ON, EVENT_TARGETED

ARTIFACT_PATH = Path("artifacts/forest.joblib")
DATA_PATH = Path("data/generated/events.parquet")

CHANNELS = (*ALWAYS_ON, *EVENT_TARGETED)
SPEND_COLUMNS = tuple(f"spend_{ch}" for ch in CHANNELS)
MARGIN_BINS = ("closest", "close", "wide", "blowout")


def _slice_table(shap_df: pd.DataFrame, train_df: pd.DataFrame, slice_values, order) -> pd.DataFrame:
    slice_values = pd.Series(slice_values).to_numpy()
    rows = []
    for level in order:
        mask = slice_values == level
        mu_mean = train_df.loc[mask, "mu"].mean()
        row = {"n": int(mask.sum())}
        for ch in CHANNELS:
            row[ch] = shap_df.loc[mask, f"spend_{ch}"].mean() / mu_mean
        rows.append(row)
    return pd.DataFrame(rows, index=list(order))


def main() -> None:
    assert ARTIFACT_PATH.exists(), f"{ARTIFACT_PATH} missing -- run scripts/fit_forest.py first"
    model = joblib.load(ARTIFACT_PATH)["model"]

    df = pd.read_parquet(DATA_PATH)
    train_df, _ = season_holdout_split(df)
    x_train = build_features(train_df)
    shap_df = compute_shap_values(model, x_train, zero_spend_columns=SPEND_COLUMNS)

    margin_bin = pd.qcut(train_df["competitiveness"], 4, labels=MARGIN_BINS)

    slices = {
        "broadcaster": (train_df["broadcaster"], BROADCASTERS),
        "season": (train_df["season"], sorted(train_df["season"].unique())),
        "is_weekend": (train_df["is_weekend"], (False, True)),
        "margin_bin (competitiveness quartile, closest=tightest game)": (margin_bin, MARGIN_BINS),
        "tentpole_tier": (train_df["tentpole_tier"], TENTPOLE_TIERS),
    }

    for name, (values, order) in slices.items():
        print(f"\n=== SHAP channel share (% of mean mu) by {name} ===")
        table = _slice_table(shap_df, train_df, values, order)
        table[list(CHANNELS)] = (table[list(CHANNELS)] * 100).round(1)
        print(table.to_string())


if __name__ == "__main__":
    main()
