"""Event-targeted spend generation.

Event-targeted channels are attributed to a specific event, no adstock --
see DESIGN.md §4. Spend deliberately concentrates on higher tentpole tiers
(the central confound, DESIGN.md §3), and `ctv` is deliberately correlated
with `tv_linear`'s daily spend -- planners buy them together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_TARGETED_CHANNELS = ("ctv", "paid_social", "paid_search")

# Multiplier on base_mean by tentpole-tier rank (regular, showcase, rivalry,
# championship) -- monotonic by design, since this concentration is the
# confound the whole project is built to measure. Kept to ~3x rather than
# more extreme spreads: a wider spread swamps the ctv/tv_linear correlation
# below with tier-driven variance that has nothing to do with tv_linear.
TIER_MULTIPLIER = (1.0, 1.5, 2.2, 3.2)

# base_mean: per-event spend at the "regular" tier, raw dollars.
# log_sigma: lognormal spread of the multiplicative noise.
# tv_corr_rho: target correlation with standardized tv_linear daily spend at
# the event's day, before the lognormal transform (ctv only). The realized
# Pearson correlation on raw spend comes out well below rho itself -- the
# lognormal transform and tier multiplier both attenuate it.
EVENT_PARAMS = {
    "ctv": {"base_mean": 40_000.0, "log_sigma": 0.3, "tv_corr_rho": 0.9},
    "paid_social": {"base_mean": 12_000.0, "log_sigma": 0.5},
    "paid_search": {"base_mean": 8_000.0, "log_sigma": 0.45},
}


def generate_eventtargeted_spend(
    schedule: pd.DataFrame, alwayson: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """Generate per-event spend for `ctv`, `paid_social`, `paid_search`.

    Parameters
    ----------
    schedule : pd.DataFrame
        Output of `schedule.generate_schedule()` -- needs `event_id`,
        `day_idx`, `tentpole_tier`.
    alwayson : pd.DataFrame
        Output of `spend.generate_alwayson_spend()` -- needs `day_idx`,
        `tv_linear`, used only to induce the `ctv` correlation.
    seed : int
        Explicit seed for the spend RNG.

    Returns
    -------
    pd.DataFrame
        `event_id` plus one column per channel in `EVENT_TARGETED_CHANNELS`,
        spend in raw dollars, one row per event.
    """
    rng = np.random.default_rng(seed)
    n = len(schedule)

    tier_rank = schedule["tentpole_tier"].cat.codes.to_numpy()
    tier_mult = np.array(TIER_MULTIPLIER)[tier_rank]

    tv_linear_daily = (
        alwayson.set_index("day_idx")["tv_linear"].reindex(schedule["day_idx"]).to_numpy()
    )
    z_tv = (tv_linear_daily - tv_linear_daily.mean()) / tv_linear_daily.std()

    data = {"event_id": schedule["event_id"].to_numpy()}
    for channel, params in EVENT_PARAMS.items():
        rho = params.get("tv_corr_rho", 0.0)
        z_noise = rng.normal(size=n)
        z = rho * z_tv + np.sqrt(1.0 - rho**2) * z_noise
        sigma = params["log_sigma"]
        # exp(sigma*z - sigma**2/2) has mean 1 -- tier_mult sets the level.
        data[channel] = params["base_mean"] * tier_mult * np.exp(sigma * z - 0.5 * sigma**2)

    return pd.DataFrame(data)
