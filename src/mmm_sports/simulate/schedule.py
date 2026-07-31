"""Event calendar for the invented broadcaster.

Generates the event-level schedule that every later simulate/ module builds
on: one row per live event, with a `day_idx` into a single contiguous daily
calendar spanning all seasons (including off-season gaps), used downstream to
sample the adstocked always-on daily series at event dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_SEASONS = 3
SEASON_LENGTH_DAYS = 200
OFFSEASON_GAP_DAYS = 90
TARGET_EVENTS_PER_SEASON = 250
CLUSTER_WEEK_FRACTION = 0.18
CLUSTER_MULTIPLIER = 2.0

# Poisson rate of events on a given day, by weekday (Mon=0 ... Sun=6).
# Weekend nights carry the bulk of live sports programming.
WEEKDAY_RATE = {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45, 4: 1.0, 5: 2.2, 6: 2.2}

BROADCASTERS = ("flagship", "sibling_cable", "regional_partner", "streaming_exclusive")
BROADCASTER_SHARE = (0.40, 0.25, 0.20, 0.15)

# "regular" plus three invented tentpole tiers, increasing prominence.
TENTPOLE_TIERS = ("regular", "showcase", "rivalry", "championship")
TENTPOLE_SHARE = (0.72, 0.14, 0.09, 0.05)


def _season_dates(season_start: pd.Timestamp, rng: np.random.Generator) -> list[pd.Timestamp]:
    """Sample event dates for one season, clustering multi-event days in some weeks."""
    days = pd.date_range(season_start, periods=SEASON_LENGTH_DAYS, freq="D")
    n_weeks = int(np.ceil(SEASON_LENGTH_DAYS / 7))
    cluster_weeks = set(
        rng.choice(n_weeks, size=max(1, int(n_weeks * CLUSTER_WEEK_FRACTION)), replace=False)
    )

    dates: list[pd.Timestamp] = []
    for offset, day in enumerate(days):
        rate = WEEKDAY_RATE[day.weekday()]
        if (offset // 7) in cluster_weeks:
            rate *= CLUSTER_MULTIPLIER
        n_events_today = rng.poisson(rate)
        dates.extend([day] * n_events_today)
    return dates


def generate_schedule(seed: int, n_seasons: int = N_SEASONS) -> pd.DataFrame:
    """Generate the event schedule.

    Parameters
    ----------
    seed : int
        Explicit seed for the schedule RNG.
    n_seasons : int
        Number of invented seasons to generate, ~`TARGET_EVENTS_PER_SEASON`
        events each.

    Returns
    -------
    pd.DataFrame
        One row per event, columns: `event_id` (int), `season` (int, 0-indexed),
        `date` (datetime64), `day_idx` (int, offset in days from the start of
        the full multi-season calendar), `is_weekend` (bool, Sat/Sun),
        `broadcaster` (category, 4 nominal levels), `tentpole_tier` (ordered
        category, "regular" + 3 invented tiers), `n_events_on_date` (int,
        events sharing that exact date — used later to dilute always-on
        attention across simultaneous broadcasts).
    """
    rng = np.random.default_rng(seed)
    calendar_start = pd.Timestamp("2021-09-01")
    season_stride = SEASON_LENGTH_DAYS + OFFSEASON_GAP_DAYS

    rows = []
    for season in range(n_seasons):
        season_start = calendar_start + pd.Timedelta(days=season * season_stride)
        for date in _season_dates(season_start, rng):
            rows.append({"season": season, "date": date})

    df = pd.DataFrame(rows).sort_values(["date"]).reset_index(drop=True)
    df["event_id"] = df.index
    df["day_idx"] = (df["date"] - calendar_start).dt.days
    df["is_weekend"] = df["date"].dt.weekday.isin([5, 6])

    n_events = len(df)
    df["broadcaster"] = pd.Categorical(
        rng.choice(BROADCASTERS, size=n_events, p=BROADCASTER_SHARE),
        categories=BROADCASTERS,
    )
    df["tentpole_tier"] = pd.Categorical(
        rng.choice(TENTPOLE_TIERS, size=n_events, p=TENTPOLE_SHARE),
        categories=TENTPOLE_TIERS,
        ordered=True,
    )
    df["n_events_on_date"] = df.groupby("date")["event_id"].transform("count")

    return df[
        [
            "event_id",
            "season",
            "date",
            "day_idx",
            "is_weekend",
            "broadcaster",
            "tentpole_tier",
            "n_events_on_date",
        ]
    ]
