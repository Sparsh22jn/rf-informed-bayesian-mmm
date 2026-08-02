"""Stage 1 interpretation layer: PDP and ALE on top of the already-fitted
Random Forest from `forest.py`. See `interpret_shap.py` for TreeSHAP.
See DESIGN.md §5 -- this is the "flexible model discovers the shape" half
of the project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def _predict_with_override(model: RandomForestRegressor, x: pd.DataFrame, feature: str, value: float) -> np.ndarray:
    x_override = x.copy()
    x_override[feature] = value
    return model.predict(x_override)


def compute_pdp(
    model: RandomForestRegressor,
    x: pd.DataFrame,
    feature: str,
    grid_values: np.ndarray | None = None,
    grid_resolution: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Partial dependence of `model`'s prediction on a single feature.

    For each grid point, every row in `x` has `feature` overridden to that
    value (everything else about the row left untouched) and the model's
    predictions are averaged -- including combinations that never actually
    occurred in the data if `feature` is correlated with another one. See
    `compute_ale` for the alternative that avoids this.

    Parameters
    ----------
    model : RandomForestRegressor
        Already fitted (e.g. loaded from `artifacts/forest.joblib`).
    x : pd.DataFrame
        Reference feature set PDP averages over -- typically the training
        features, same columns `model` was fit on.
    feature : str
        Column name in `x` to sweep.
    grid_values : np.ndarray, optional
        Explicit grid to evaluate at (e.g. to share `compute_ale`'s bin
        edges for a direct comparison). Defaults to `grid_resolution`
        evenly spaced points across `x[feature]`'s observed range.
    grid_resolution : int
        Number of grid points, only used when `grid_values` is None.

    Returns
    -------
    grid : np.ndarray, shape (n_grid,)
        Swept values of `feature`, raw dollars.
    average_prediction : np.ndarray, shape (n_grid,)
        Model's average predicted viewership (thousands) at each grid point.
    """
    if grid_values is None:
        grid_values = np.linspace(x[feature].min(), x[feature].max(), grid_resolution)
    grid_values = np.asarray(grid_values)
    average_prediction = np.array(
        [_predict_with_override(model, x, feature, v).mean() for v in grid_values]
    )
    return grid_values, average_prediction


def compute_ale(
    model: RandomForestRegressor, x: pd.DataFrame, feature: str, n_bins: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulated Local Effects of `model`'s prediction on a single feature.

    Bins `feature` into `n_bins` quantile buckets. Within each bucket, only
    the rows actually observed in that bucket are used, and `feature` is
    nudged just to the bucket's own edges (not the full observed range) --
    so every prediction the model is asked for reflects a combination close
    to something real, unlike PDP. Local effects are accumulated
    (cumulative sum) into a curve, then centered to a weighted mean of 0.

    Parameters
    ----------
    model : RandomForestRegressor
        Already fitted.
    x : pd.DataFrame
        Reference feature set, same columns `model` was fit on.
    feature : str
        Column name in `x` to bin and sweep.
    n_bins : int
        Number of quantile bins.

    Returns
    -------
    edges : np.ndarray, shape (n_edges,)
        Quantile bin edges of `feature`, raw dollars (n_edges <= n_bins + 1;
        duplicate quantiles collapse for sparse/skewed features).
    ale : np.ndarray, shape (n_edges,)
        Centered accumulated local effect at each edge, thousands of
        viewers -- directly comparable to a mean-centered PDP curve.
    """
    values = x[feature].to_numpy()
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    bin_idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)

    local_effects = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1)
    for k in range(len(edges) - 1):
        mask = bin_idx == k
        counts[k] = mask.sum()
        if counts[k] == 0:
            continue
        x_bin = x[mask]
        pred_hi = _predict_with_override(model, x_bin, feature, edges[k + 1])
        pred_lo = _predict_with_override(model, x_bin, feature, edges[k])
        local_effects[k] = (pred_hi - pred_lo).mean()

    ale = np.concatenate([[0.0], np.cumsum(local_effects)])
    segment_mean = (ale[:-1] + ale[1:]) / 2.0
    center = np.average(segment_mean, weights=counts)
    return edges, ale - center
