"""Daily always-on spend generation.

Always-on channels run on the calendar, not per-event -- see DESIGN.md §4.
Flighting is modeled as alternating "on" (flight) and "off" (dark) day
blocks, scaled by a within-season ramp and small day-to-day noise. Zero
during off-season gaps: there is nothing to flight against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm_sports.simulate.schedule import N_SEASONS, OFFSEASON_GAP_DAYS, SEASON_LENGTH_DAYS

ALWAYS_ON_CHANNELS = ("tv_linear", "out_of_home", "display")

# Flight/dark block lengths (days), day-to-day noise SD (fraction of mean),
# and within-season ramp amplitude (fraction of mean, added linearly over
# the season). out_of_home's long flights + short dark spells + low noise
# is what produces its low variance -- realistic (long-run OOH contracts)
# and the source of its weak identifiability, per DESIGN.md §3.
FLIGHT_PARAMS = {
    "tv_linear": {
        "mean": 120_000.0,
        "flight_len": (7, 20),
        "dark_len": (10, 25),
        "noise_sd_frac": 0.30,
        "ramp_amplitude": 0.40,
    },
    "out_of_home": {
        "mean": 50_000.0,
        "flight_len": (60, 90),
        "dark_len": (3, 8),
        "noise_sd_frac": 0.05,
        "ramp_amplitude": 0.10,
    },
    "display": {
        "mean": 25_000.0,
        "flight_len": (15, 30),
        "dark_len": (10, 20),
        "noise_sd_frac": 0.20,
        "ramp_amplitude": 0.25,
    },
}


def _flight_mask(
    n_days: int,
    flight_len: tuple[int, int],
    dark_len: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Alternating on/off day blocks covering `n_days`, random phase."""
    mask = np.zeros(n_days, dtype=bool)
    pos = 0
    on = bool(rng.integers(0, 2))
    while pos < n_days:
        low, high = flight_len if on else dark_len
        length = int(rng.integers(low, high + 1))
        mask[pos : pos + length] = on
        pos += length
        on = not on
    return mask


def _season_series(channel: str, rng: np.random.Generator) -> np.ndarray:
    params = FLIGHT_PARAMS[channel]
    n = SEASON_LENGTH_DAYS
    mask = _flight_mask(n, params["flight_len"], params["dark_len"], rng)
    ramp = 1.0 + params["ramp_amplitude"] * (np.arange(n) / (n - 1))
    noise = np.clip(1.0 + rng.normal(0.0, params["noise_sd_frac"], size=n), 0.0, None)
    return params["mean"] * ramp * noise * mask


def generate_alwayson_spend(seed: int) -> pd.DataFrame:
    """Generate daily always-on spend across the full multi-season calendar.

    Parameters
    ----------
    seed : int
        Explicit seed for the spend RNG.

    Returns
    -------
    pd.DataFrame
        One row per calendar day (`day_idx`, aligned with `schedule.py`'s
        `day_idx`), plus one column per channel in `ALWAYS_ON_CHANNELS`,
        daily spend in raw dollars. Zero on off-season gap days.
    """
    rng = np.random.default_rng(seed)
    total_days = N_SEASONS * SEASON_LENGTH_DAYS + (N_SEASONS - 1) * OFFSEASON_GAP_DAYS
    stride = SEASON_LENGTH_DAYS + OFFSEASON_GAP_DAYS

    data = {ch: np.zeros(total_days) for ch in ALWAYS_ON_CHANNELS}
    for season in range(N_SEASONS):
        start = season * stride
        for ch in ALWAYS_ON_CHANNELS:
            data[ch][start : start + SEASON_LENGTH_DAYS] = _season_series(ch, rng)

    df = pd.DataFrame(data)
    df.insert(0, "day_idx", np.arange(total_days))
    return df
