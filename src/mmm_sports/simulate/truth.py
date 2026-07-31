"""Ground truth for the synthetic data-generating process.

The one place every true parameter is defined. Model code (`models/`,
`priors.py`, the app) must never import this module — only tests and
evaluation scripts may, to score recovered parameters against it. See
`CLAUDE.md` non-negotiables.

Units, throughout:
  alpha  -- geometric adstock decay, dimensionless in [0, 1). 0 for
            event-targeted channels: DESIGN.md §4 gives them no adstock.
  k      -- Hill half-saturation point, raw dollars.
  s      -- Hill exponent, dimensionless.
  beta   -- channel ceiling / ordinary control coefficient, thousands of
            viewers per full-saturation unit (media) or per unit of a
            MinMax[0,1]-scaled control.
  season_intercept -- thousands of viewers, one per season.
  sigma  -- residual noise SD, thousands of viewers.

`tentpole_tier` is ordinal (regular < showcase < rivalry < championship) and
enters `mu` as `control_beta["tentpole_tier"] * rank / (n_tiers - 1)` -- a
single large scalar coefficient, per DESIGN.md §3. `broadcaster` is nominal
and enters as a per-level vector instead, so it is *not* in `SCALAR_CONTROLS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from mmm_sports.simulate.schedule import BROADCASTERS, TENTPOLE_TIERS

CHANNELS = ("tv_linear", "out_of_home", "display", "ctv", "paid_social", "paid_search")
ALWAYS_ON = ("tv_linear", "out_of_home", "display")
EVENT_TARGETED = ("ctv", "paid_social", "paid_search")

SCALAR_CONTROLS = (
    "tv_availability",
    "team_interest",
    "star_interest",
    "is_weekend",
    "competitiveness",
    "tentpole_tier",
)
N_SEASONS = 3


@dataclass(frozen=True)
class Truth:
    alpha: Mapping[str, float]
    k: Mapping[str, float]
    s: Mapping[str, float]
    beta: Mapping[str, float]
    control_beta: Mapping[str, float]
    broadcaster_beta: Mapping[str, float]
    season_intercept: tuple[float, ...]
    sigma: float


TRUTH = Truth(
    alpha={
        "tv_linear": 0.75,
        "out_of_home": 0.55,
        "display": 0.40,
        "ctv": 0.0,
        "paid_social": 0.0,
        "paid_search": 0.0,
    },
    k={
        "tv_linear": 140_000.0,
        "out_of_home": 45_000.0,
        "display": 30_000.0,
        "ctv": 55_000.0,
        "paid_social": 15_000.0,
        "paid_search": 6_000.0,  # low saturation point, per DESIGN.md §3
    },
    s={
        "tv_linear": 2.0,
        "out_of_home": 1.8,
        "display": 1.5,
        "ctv": 2.0,
        "paid_social": 1.7,
        "paid_search": 1.6,
    },
    beta={
        "tv_linear": 235.0,
        "out_of_home": 65.0,
        "display": 2.0,  # near-zero true effect -- the "dead channel" pathology
        "ctv": 60.0,
        "paid_social": 40.0,
        "paid_search": 70.0,
    },
    control_beta={
        "tv_availability": 80.0,
        "team_interest": 120.0,
        "star_interest": 100.0,
        "is_weekend": 60.0,
        "competitiveness": -90.0,  # blowouts shed viewers
        "tentpole_tier": 450.0,  # the central confounder -- large by design
    },
    broadcaster_beta={
        "flagship": 200.0,
        "sibling_cable": 40.0,
        "regional_partner": -60.0,
        "streaming_exclusive": -120.0,
    },
    season_intercept=(900.0, 950.0, 1000.0),
    sigma=190.0,
)

assert set(TRUTH.broadcaster_beta) == set(BROADCASTERS)
assert len(TENTPOLE_TIERS) == 4  # regular + 3 tiers, matches the ordinal rank scaling above
