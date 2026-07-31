"""The four continuous controls not already produced by `schedule.py`.

`broadcaster`, `is_weekend`, and `tentpole_tier` are generated in
`schedule.py` (they're intrinsic to the event itself); `tv_availability`,
`team_interest`, `star_interest`, and `competitiveness` are generated here.
All are raw scale -- `dgp.py` applies MinMax[0,1] scaling uniformly across
every scalar control before multiplying by `truth.py`'s coefficients, per
DESIGN.md §4's pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm_sports.simulate.schedule import N_SEASONS, OFFSEASON_GAP_DAYS, SEASON_LENGTH_DAYS

ROLLING_WINDOW_DAYS = 28  # "rolling 4-week" per DESIGN.md §3


def _ar1_daily_series(n_days: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Mean-reverting AR(1) latent series, used for interest indices."""
    x = np.empty(n_days)
    x[0] = rng.normal(0.0, sigma)
    innovations = rng.normal(0.0, sigma, size=n_days - 1)
    for t in range(1, n_days):
        x[t] = rho * x[t - 1] + innovations[t - 1]
    return x


def _rolling_interest_index(n_days: int, rng: np.random.Generator, level: float) -> np.ndarray:
    """AR(1) latent, smoothed with a 4-week rolling mean, centered on `level`.

    rho=0.8, not closer to 1 -- a near-unit-root AR(1) drifts slowly enough
    to spuriously correlate with any other slow-moving series (found via
    check_controls.py showing a ~0.7 correlation with tv_availability's
    seasonal pattern at rho=0.98, which isn't a deliberate confound).
    """
    latent = _ar1_daily_series(n_days, rho=0.8, sigma=1.0, rng=rng)
    smoothed = pd.Series(latent).rolling(ROLLING_WINDOW_DAYS, min_periods=1).mean().to_numpy()
    return np.clip(level + 10.0 * smoothed, 0.0, 100.0)


def generate_controls(schedule: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Generate `tv_availability`, `team_interest`, `star_interest`, `competitiveness`.

    Parameters
    ----------
    schedule : pd.DataFrame
        Output of `schedule.generate_schedule()` -- needs `event_id`,
        `date`, `day_idx`.
    seed : int
        Explicit seed for the controls RNG.

    Returns
    -------
    pd.DataFrame
        `event_id` plus the four controls, raw scale:
        `tv_availability` (index, ~10-100, seasonal),
        `team_interest` / `star_interest` (index, ~0-100, rolling 4-week),
        `competitiveness` (final margin, points, >= 0, lower = closer game).
    """
    rng = np.random.default_rng(seed)
    n_days = N_SEASONS * SEASON_LENGTH_DAYS + (N_SEASONS - 1) * OFFSEASON_GAP_DAYS

    team_interest_daily = _rolling_interest_index(n_days, rng, level=50.0)
    star_interest_daily = _rolling_interest_index(n_days, rng, level=50.0)

    day_idx = schedule["day_idx"].to_numpy()
    day_of_year = schedule["date"].dt.dayofyear.to_numpy()

    tv_availability = np.clip(
        60.0 + 30.0 * np.cos(2 * np.pi * day_of_year / 365.0) + rng.normal(0.0, 5.0, len(schedule)),
        10.0,
        100.0,
    )
    competitiveness = rng.exponential(scale=12.0, size=len(schedule))

    return pd.DataFrame(
        {
            "event_id": schedule["event_id"].to_numpy(),
            "tv_availability": tv_availability,
            "team_interest": team_interest_daily[day_idx],
            "star_interest": star_interest_daily[day_idx],
            "competitiveness": competitiveness,
        }
    )
