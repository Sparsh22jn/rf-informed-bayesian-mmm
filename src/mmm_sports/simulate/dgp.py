"""Assemble the full data-generating process -- the only place all of
`simulate/`'s pieces come together into a response. See DESIGN.md §4 for the
pipeline this implements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm_sports.simulate.controls import generate_controls
from mmm_sports.simulate.event_spend import generate_eventtargeted_spend
from mmm_sports.simulate.schedule import generate_schedule
from mmm_sports.simulate.spend import generate_alwayson_spend
from mmm_sports.simulate.truth import ALWAYS_ON, EVENT_TARGETED, SCALAR_CONTROLS, TRUTH
from mmm_sports.transforms import extract_event_level, geometric_adstock, hill_saturation


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo)


def _scalar_control_values(schedule: pd.DataFrame, controls: pd.DataFrame) -> dict[str, np.ndarray]:
    """Raw-scale values for every `SCALAR_CONTROLS` entry, before MinMax."""
    tier_rank = schedule["tentpole_tier"].cat.codes.to_numpy().astype(float)
    return {
        "tv_availability": controls["tv_availability"].to_numpy(),
        "team_interest": controls["team_interest"].to_numpy(),
        "star_interest": controls["star_interest"].to_numpy(),
        "is_weekend": schedule["is_weekend"].to_numpy().astype(float),
        "competitiveness": controls["competitiveness"].to_numpy(),
        "tentpole_tier": tier_rank,
    }


def generate_dataset(seed: int) -> pd.DataFrame:
    """Generate the full synthetic event-level dataset.

    Parameters
    ----------
    seed : int
        Single seed driving every sub-generator (schedule, spend, controls,
        noise) -- same seed always reproduces the same dataset.

    Returns
    -------
    pd.DataFrame
        One row per event: schedule/control columns, raw spend per channel,
        `mu` (noise-free response), `viewership` (mu + noise, thousands),
        plus one `contrib_<channel>` column per media channel and
        `contrib_baseline` / `contrib_controls` for the decomposition.
    """
    schedule = generate_schedule(seed=seed)
    alwayson = generate_alwayson_spend(seed=seed)
    event_targeted = generate_eventtargeted_spend(schedule, alwayson, seed=seed)
    controls = generate_controls(schedule, seed=seed)

    day_idx = schedule["day_idx"].to_numpy()
    n_events_on_date = schedule["n_events_on_date"].to_numpy()

    contrib = {}
    for ch in ALWAYS_ON:
        adstocked = geometric_adstock(alwayson[ch].to_numpy(), TRUTH.alpha[ch], l_max=30)
        event_level = extract_event_level(adstocked, day_idx, n_events_on_date)
        contrib[ch] = TRUTH.beta[ch] * hill_saturation(event_level, TRUTH.k[ch], TRUTH.s[ch])

    for ch in EVENT_TARGETED:
        contrib[ch] = TRUTH.beta[ch] * hill_saturation(
            event_targeted[ch].to_numpy(), TRUTH.k[ch], TRUTH.s[ch]
        )

    scalar_values = _scalar_control_values(schedule, controls)
    controls_total = np.zeros(len(schedule))
    for name in SCALAR_CONTROLS:
        controls_total += _minmax(scalar_values[name]) * TRUTH.control_beta[name]
    broadcaster_effect = schedule["broadcaster"].map(TRUTH.broadcaster_beta).to_numpy()
    controls_total = controls_total + broadcaster_effect

    season_intercept = np.array(TRUTH.season_intercept)[schedule["season"].to_numpy()]

    mu = season_intercept + controls_total + sum(contrib.values())
    rng = np.random.default_rng(seed)
    viewership = mu + rng.normal(0.0, TRUTH.sigma, size=len(schedule))

    out = schedule.copy()
    out = out.merge(alwayson.rename(columns={ch: f"spend_{ch}" for ch in ALWAYS_ON})[
        ["day_idx", *[f"spend_{ch}" for ch in ALWAYS_ON]]
    ], on="day_idx", how="left")
    for ch in EVENT_TARGETED:
        out[f"spend_{ch}"] = event_targeted[ch].to_numpy()
    for name in ("tv_availability", "team_interest", "star_interest", "competitiveness"):
        out[name] = controls[name].to_numpy()
    for ch, values in contrib.items():
        out[f"contrib_{ch}"] = values
    out["contrib_baseline"] = season_intercept
    out["contrib_controls"] = controls_total
    out["mu"] = mu
    out["viewership"] = viewership

    return out
