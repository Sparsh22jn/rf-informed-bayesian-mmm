"""TreeSHAP on top of the already-fitted Random Forest from `forest.py`.
Split from `interpret.py` (PDP/ALE) to stay under the module line budget.
"""

from __future__ import annotations

import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor


def compute_shap_values(
    model: RandomForestRegressor,
    x: pd.DataFrame,
    zero_spend_columns: tuple[str, ...] | None = None,
    max_background: int = 50,
) -> pd.DataFrame:
    """Per-event, per-feature TreeSHAP contributions.

    Parameters
    ----------
    model : RandomForestRegressor
        Already fitted.
    x : pd.DataFrame
        Feature set to explain -- same columns `model` was fit on.
    zero_spend_columns : tuple[str, ...], optional
        Spend columns to use a zero-spend background for (interventional
        SHAP), rather than the default average-event background. Matters
        for comparing against `truth.py`'s contributions: those are defined
        relative to zero spend (`hill_saturation(0) = 0`), but a plain
        `TreeExplainer`'s default background is the *average* event -- a
        channel near its average spend gets near-zero SHAP credit under
        that baseline even if its true zero-to-ceiling contribution is
        large. Pass the media spend columns to make the comparison
        apples-to-apples. None uses the plain average-event background.
    max_background : int
        Background sample size when `zero_spend_columns` is set (only the
        non-spend columns vary across it, so a small sample is stable).

    Returns
    -------
    pd.DataFrame
        Same shape and index as `x`: one SHAP value per event per feature,
        thousands of viewers. Each row sums (plus the explainer's base
        value) to that event's model prediction.
    """
    if zero_spend_columns is None:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x)
    else:
        x = x.astype(float)
        background = x.copy()
        background[list(zero_spend_columns)] = 0.0
        if len(background) > max_background:
            background = background.sample(max_background, random_state=0)
        explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
        shap_values = explainer.shap_values(x, check_additivity=False)
    return pd.DataFrame(shap_values, columns=x.columns, index=x.index)
