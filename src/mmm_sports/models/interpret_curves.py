"""Empirical response curves: sweep one channel's spend holding everything
else at a single representative event's observed values, reading straight
off the RF's prediction surface. Split from `interpret.py` to keep modules
under the line budget -- purpose differs from PDP/ALE too: this feeds
Phase 4's curve-fitting, which needs one clean curve per channel, not a
marginal average distorted by correlated features. See DESIGN.md §5/§6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def compute_empirical_curve(
    model: RandomForestRegressor,
    x: pd.DataFrame,
    feature: str,
    grid_values: np.ndarray | None = None,
    grid_resolution: int = 50,
    dummy_prefixes: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep `feature` against one representative synthetic event.

    Unlike `interpret.compute_pdp` (which overrides `feature` on every row
    in `x` and averages), this builds a single row -- the median of every
    other numeric column, and for each `dummy_prefixes` group (e.g. a
    one-hot `broadcaster_*` set) the single most common level turned on,
    not an independent per-column median, which could otherwise produce an
    invalid state with zero or multiple levels "on" at once -- and sweeps
    `feature` across that one fixed backdrop.

    Parameters
    ----------
    model : RandomForestRegressor
        Already fitted.
    x : pd.DataFrame
        Reference feature set the representative row is built from --
        typically the training features, same columns `model` was fit on.
    feature : str
        Column name in `x` to sweep.
    grid_values : np.ndarray, optional
        Explicit grid to evaluate at. Defaults to `grid_resolution` evenly
        spaced points across `x[feature]`'s observed range.
    grid_resolution : int
        Number of grid points, only used when `grid_values` is None.
    dummy_prefixes : tuple[str, ...]
        Column-name prefixes identifying one-hot groups (e.g. `("broadcaster",)`)
        to set by mode rather than per-column median.

    Returns
    -------
    grid : np.ndarray, shape (n_grid,)
        Swept values of `feature`, raw dollars.
    prediction : np.ndarray, shape (n_grid,)
        Model's predicted viewership (thousands) at each grid point, for
        the one representative event.
    """
    if grid_values is None:
        grid_values = np.linspace(x[feature].min(), x[feature].max(), grid_resolution)
    grid_values = np.asarray(grid_values)

    representative = x.median(numeric_only=True)
    for prefix in dummy_prefixes:
        group_cols = [c for c in x.columns if c.startswith(prefix)]
        mode_col = x[group_cols].sum().idxmax()
        for col in group_cols:
            representative[col] = 1.0 if col == mode_col else 0.0

    row = pd.DataFrame([representative])[x.columns]
    rows = pd.concat([row] * len(grid_values), ignore_index=True)
    rows[feature] = grid_values
    prediction = model.predict(rows)
    return grid_values, prediction
