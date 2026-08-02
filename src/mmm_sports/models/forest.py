"""Stage 1 — Random Forest on observed media spend + controls.

Features are the *observed* spend columns (raw dollars, as an analyst would
actually see them), not the DGP's internal adstocked/diluted exposure --
that distinction matters: the RF only ever sees what a real analyst would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from mmm_sports.simulate.truth import ALWAYS_ON, EVENT_TARGETED

SPEND_COLUMNS = tuple(f"spend_{ch}" for ch in (*ALWAYS_ON, *EVENT_TARGETED))
CONTINUOUS_CONTROLS = ("tv_availability", "team_interest", "star_interest", "competitiveness")

PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [4, 8, None],
    "min_samples_leaf": [1, 3, 5],
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the RF feature matrix: media spend + controls, all numeric.

    Parameters
    ----------
    df : pd.DataFrame
        An events dataframe (or slice of one) with `spend_*`, control, and
        schedule columns -- e.g. from `data/generated/events.parquet`.

    Returns
    -------
    pd.DataFrame
        One row per event, numeric columns only: raw spend per channel,
        continuous controls, `is_weekend` (0/1), `tentpole_tier_rank`
        (0-3), `n_events_on_date`, and one-hot `broadcaster_*` columns.
    """
    features = df[list(SPEND_COLUMNS)].copy()
    for name in CONTINUOUS_CONTROLS:
        features[name] = df[name]
    features["is_weekend"] = df["is_weekend"].astype(int)
    features["tentpole_tier_rank"] = df["tentpole_tier"].cat.codes
    # n_events_on_date: an analyst always knows the day's schedule. Without
    # it the model can't see always-on attention dilution (DESIGN.md §4) --
    # a day's raw spend looks identical whether it aired alone or split
    # three ways, even though the true per-event exposure differs a lot.
    features["n_events_on_date"] = df["n_events_on_date"]
    broadcaster_dummies = pd.get_dummies(df["broadcaster"], prefix="broadcaster")
    return pd.concat([features, broadcaster_dummies], axis=1)


def season_holdout_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on all but the last season; hold out the last season entirely."""
    holdout_season = df["season"].max()
    return df[df["season"] < holdout_season], df[df["season"] == holdout_season]


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mape": float(np.mean(np.abs(y_true - y_pred) / y_true)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_forest(df: pd.DataFrame, seed: int) -> dict:
    """Fit an RF with a light hyperparameter search, season holdout.

    Parameters
    ----------
    df : pd.DataFrame
        Full events dataframe (all seasons).
    seed : int
        Random seed for the forest and the CV split.

    Returns
    -------
    dict
        `model` (fitted `RandomForestRegressor`), `best_params`,
        `feature_names`, `train_metrics`, `holdout_metrics` (each a dict
        with `mape` and `r2`).
    """
    train_df, holdout_df = season_holdout_split(df)
    x_train, y_train = build_features(train_df), train_df["viewership"].to_numpy()
    x_holdout, y_holdout = build_features(holdout_df), holdout_df["viewership"].to_numpy()

    search = GridSearchCV(
        RandomForestRegressor(random_state=seed),
        PARAM_GRID,
        cv=3,
        scoring="r2",
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_

    return {
        "model": model,
        "best_params": search.best_params_,
        "feature_names": list(x_train.columns),
        "train_metrics": _regression_metrics(y_train, model.predict(x_train)),
        "holdout_metrics": _regression_metrics(y_holdout, model.predict(x_holdout)),
    }
