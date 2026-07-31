"""Worked example of event-level extraction: a single-event day vs a
triple-event day sampling the same underlying daily series.

Usage: python scripts/check_extraction.py [seed]
"""

from __future__ import annotations

import sys

import numpy as np

from mmm_sports.simulate.schedule import generate_schedule
from mmm_sports.transforms import extract_event_level


def main(seed: int) -> None:
    schedule = generate_schedule(seed=seed)

    single = schedule[schedule["n_events_on_date"] == 1].iloc[0]
    triple = schedule[schedule["n_events_on_date"] == 3].iloc[0]

    n_days = schedule["day_idx"].max() + 1
    rng = np.random.default_rng(seed)
    adstocked_daily = rng.uniform(50_000.0, 150_000.0, size=n_days)

    # Force both example days to the exact same underlying daily value, so
    # the only difference in the printed result is the dilution divisor.
    shared_value = 300_000.0
    adstocked_daily[single["day_idx"]] = shared_value
    adstocked_daily[triple["day_idx"]] = shared_value

    for label, row in [("single-event day", single), ("triple-event day", triple)]:
        day_idx = np.array([row["day_idx"]])
        n_events = np.array([row["n_events_on_date"]])
        result = extract_event_level(adstocked_daily, day_idx, n_events)
        print(f"{label}: date={row['date'].date()}, day_idx={row['day_idx']}, "
              f"n_events_on_date={row['n_events_on_date']}, "
              f"adstocked_daily={shared_value:,.0f}, "
              f"event-level={result[0]:,.0f}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed)
