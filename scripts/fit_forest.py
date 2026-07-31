"""Fit the Stage-1 Random Forest, print in-sample and holdout MAPE/R2 read
against the irreducible MAPE floor (DESIGN.md §4), and persist the model
for later PDP/ALE/SHAP tasks.

Usage: python scripts/fit_forest.py [seed]
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mmm_sports.models.forest import fit_forest, season_holdout_split

DATA_PATH = Path("data/generated/events.parquet")
ARTIFACT_PATH = Path("artifacts/forest.joblib")


def _mape_floor(df: pd.DataFrame) -> float:
    """True model (mu) vs its own noisy output (viewership) -- see check_dgp.py."""
    return float(np.mean(np.abs(df["viewership"] - df["mu"]) / df["viewership"]))


def main(seed: int) -> None:
    df = pd.read_parquet(DATA_PATH)
    result = fit_forest(df, seed=seed)
    train_df, holdout_df = season_holdout_split(df)

    print(f"Best hyperparameters: {result['best_params']}")
    print(f"Features ({len(result['feature_names'])}): {result['feature_names']}")

    floors = {"train": _mape_floor(train_df), "holdout": _mape_floor(holdout_df)}
    for split, split_df in (("train", train_df), ("holdout", holdout_df)):
        metrics = result[f"{split}_metrics"]
        floor = floors[split]
        print(
            f"{split} (n={len(split_df)}): MAPE={metrics['mape']:.1%}  R2={metrics['r2']:.3f}"
            f"  [irreducible floor: {floor:.1%}]"
        )
        if metrics["mape"] < floor:
            print(
                f"  note: RF MAPE is below the floor -- not evidence of beating irreducible"
                f" noise. MAPE's error is relative to observed y, not mu; a fitted model can"
                f" land closer to a specific noisy realization than the true mean does on a"
                f" finite holdout, especially where y is small (denominator sensitivity)."
            )

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, ARTIFACT_PATH)
    print(f"\nSaved fitted model + metrics to {ARTIFACT_PATH}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed)
