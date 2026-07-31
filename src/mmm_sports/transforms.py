"""Media transforms shared by the DGP and the Bayesian model.

Kept separate from `simulate/` because Phase 5's pytensor model reuses the
same adstock semantics (see DESIGN.md §4) — this module has no dependency on
`simulate.truth` and is safe for model code to import.
"""

from __future__ import annotations

import numpy as np


def geometric_adstock(spend: np.ndarray, alpha: float, l_max: int = 30) -> np.ndarray:
    """Causal geometric adstock over calendar time.

    adstocked[t] = sum_{l=0}^{l_max} alpha**l * spend[t - l], with spend
    before day 0 treated as zero (no pre-history) — i.e. a causal convolution
    of `spend` with the decay kernel, truncated to `l_max` lags.

    Parameters
    ----------
    spend : np.ndarray, shape (n_days,)
        Daily spend, raw dollars.
    alpha : float
        Decay rate per day, in [0, 1).
    l_max : int
        Maximum carryover lag, in days.

    Returns
    -------
    np.ndarray, shape (n_days,)
        Adstocked spend, raw dollars, same length as `spend`.
    """
    kernel = alpha ** np.arange(l_max + 1)
    adstocked = np.convolve(spend, kernel)[: len(spend)]
    return adstocked


def hill_saturation(spend: np.ndarray, k: float, s: float) -> np.ndarray:
    """Hill saturation curve.

    f(x) = x**s / (k**s + x**s) — diminishing returns as spend rises, with
    f(k) = 0.5 by construction (k is the half-saturation point).

    Parameters
    ----------
    spend : np.ndarray, shape (n,)
        Spend, raw dollars. Non-negative.
    k : float
        Half-saturation point, raw dollars.
    s : float
        Hill exponent, dimensionless, controls steepness.

    Returns
    -------
    np.ndarray, shape (n,)
        Saturation fraction, dimensionless, in [0, 1).
    """
    return spend**s / (k**s + spend**s)


def extract_event_level(
    adstocked_daily: np.ndarray, day_idx: np.ndarray, n_events_on_date: np.ndarray
) -> np.ndarray:
    """Sample the adstocked daily always-on series at event dates, diluted.

    X[e] = adstocked_daily[day_idx[e]] / n_events_on_date[e] — attention on a
    given day's broadcast splits across however many events aired that day.

    Parameters
    ----------
    adstocked_daily : np.ndarray, shape (n_days,)
        Adstocked always-on spend, raw dollars, indexed by calendar day.
    day_idx : np.ndarray, shape (n_events,)
        Calendar-day index of each event, into `adstocked_daily`.
    n_events_on_date : np.ndarray, shape (n_events,)
        Number of events sharing that event's exact date. >= 1.

    Returns
    -------
    np.ndarray, shape (n_events,)
        Diluted event-level spend, raw dollars.
    """
    return adstocked_daily[day_idx] / n_events_on_date
