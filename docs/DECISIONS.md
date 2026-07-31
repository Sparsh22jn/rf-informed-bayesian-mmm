# DECISIONS.md

Append-only log of decisions and discrepancies found while implementing
`TASKS.md`. Feeds `docs/blog/`.

---

## 2026-07-30 — Task 0.3: `DESIGN.md` internal consistency pass

Read `docs/DESIGN.md` end to end and cross-checked its most load-bearing
pieces of spec against each other and against the rest of the document: the
curve-fit approach in §6 (Gaussian-CDF summarization feeding the Hill prior
conversion, including the deliberate one-channel-uncovered fallback), and
the production model spec in §4 (hierarchical non-centered beta
parameterization, nominal `broadcaster` encoding, noise prior sized to the
10-12% MAPE floor target).

No contradictions found — every piece checked is consistent with the rest
of `docs/DESIGN.md`.

**Outcome:** no change to `docs/DESIGN.md` from this task.

---

## 2026-07-30 — Task 1.1: `simulate/schedule.py`

Per-day event counts drawn from a Poisson rate keyed on weekday (low
Mon–Thu, elevated Fri, high Sat/Sun), with ~18% of weeks per season flagged
as "cluster weeks" where the rate doubles — this is what produces multi-event
days without hand-scripting them. Weekday rates were tuned (0.35→0.45
weekday, 1.7→2.2 weekend) after an initial run landed at ~196 events/season
against a ~250 target; four seeds now land in the 210–276 range per season,
which reads as realistic season-to-season variance rather than a bug.

`day_idx` is offset from a single calendar start spanning all seasons plus
90-day off-season gaps between them, not reset per season — always-on
adstock in Phase 1.6+ needs one contiguous daily timeline to decay across.

`tentpole_tier` modeled as an *ordered* pandas categorical (`regular` <
`showcase` < `rivalry` < `championship`); `broadcaster` deliberately left
unordered per `DESIGN.md §3`.

---

## 2026-07-30 — Task 1.2: `simulate/truth.py`

`tentpole_tier` enters `mu` as a single scalar (`control_beta["tentpole_tier"]
* rank / 3`) since DESIGN §3 calls it "direct," while `broadcaster` gets a
per-level vector since DESIGN §3 calls it explicitly nominal — the task text's
"7 control coefficients (including tentpole_tier ... and broadcaster as a
per-level vector)" reads as one control conceptually represented as several
parameters, not an 8th control.

Event-targeted channels (`ctv`, `paid_social`, `paid_search`) get `alpha =
0.0` in `Truth` rather than omitting alpha for them — DESIGN §4 gives them no
adstock, and an explicit zero lets Phase 5.6 compare learned α against a real
target instead of a missing one.

Only the "dead `display` channel" pathology is encoded here (β ≈ 0). The other
two DESIGN §3 pathologies — low `out_of_home` spend variance, `tv_linear`/
`ctv` spend correlation — are properties of the spend generator, not of
`truth.py`'s parameters; they land in tasks 1.6/1.7.

All magnitudes (K, S, β, sigma) are placeholders chosen for realistic order of
magnitude, not tuned. Task 1.9 retunes them against the `DESIGN.md §4`
contribution-share targets once the full DGP pipeline exists.

---

## 2026-07-30 — Task 1.3: `transforms.py` geometric adstock

Placed at `src/mmm_sports/transforms.py` (top-level, not under `simulate/`)
per the tree in `DESIGN.md §11` — Phase 5's pytensor model reuses the same
adstock semantics later, and it has no dependency on `simulate.truth`, so
model code can import it safely.

Implemented as a single causal `np.convolve(spend, kernel)` truncated to
`len(spend)`, rather than a hand-rolled loop — convolution already treats
missing pre-day-0 spend as zero contribution for free. Verified by hand
against a single-spike and a two-spike case, plus an `l_max` truncation case,
rather than a round-trip/property test — DESIGN.md's own framing is that a
green test proves the *shape* is right, not that the numbers are.

---

## 2026-07-30 — Task 1.4: `transforms.py` Hill saturation

`hill_saturation(spend, k, s)` added alongside adstock in the same file —
one-line closed form, no edge-case handling needed since `k**s > 0` always
(no division-by-zero risk at `spend=0`). Test checks the general shape
(monotone increasing, bounded `[0,1)`) plus the one exact value the formula
guarantees analytically: `f(k) == 0.5`, the half-saturation point.

---

## 2026-07-30 — Task 1.5: `transforms.py` event-level extraction

`extract_event_level()` is pure indexing + division, no new logic to hand-
verify — the interesting behavior is the dilution, so `check_extraction.py`
pulls two *real* days out of `generate_schedule()` (one single-event, one
triple-event) and forces both to the same synthetic daily value before
extracting, so the printed side-by-side comparison isolates the divisor's
effect instead of also varying by coincidence.

---

## 2026-07-30 — Task 1.6: `simulate/spend.py` daily always-on spend

New module, not in `DESIGN.md §11`'s illustrative tree — `dgp.py` alone would
blow past the ~150-line module cap once it also holds spend and control
generation, so spend generation got its own file. Calendar length derived
directly from `schedule.py`'s `N_SEASONS`/`SEASON_LENGTH_DAYS`/
`OFFSEASON_GAP_DAYS` constants rather than passed in, so `day_idx` alignment
with the event schedule is automatic and can't drift.

`out_of_home`'s low variance comes from realistic flighting parameters (long
60–90 day flights, short 3–8 day dark spells, low day-to-day noise) rather
than an artificial variance clamp — `tv_linear` gets short bursty flights and
more dark time by contrast. Coefficient of variation is computed over the
*full* calendar (including off-season zero blocks all channels share), so an
initial 0.5 threshold was too tight — `out_of_home` lands ~0.6–0.65 across
seeds vs `display` ~1.1 and `tv_linear` ~1.5. Threshold set to 0.8, which
sits cleanly between those two clusters.

---

## 2026-07-30 — Task 1.7: `simulate/event_spend.py` event-targeted spend

Split into its own module rather than growing `spend.py` further (same
~150-line reasoning as 1.6) — `spend.py` stays always-on only, matching
DESIGN.md's own always-on/event-targeted mechanism split.

`ctv`'s correlation with `tv_linear` is generated properly (a standard
bivariate-normal construction, `z_ctv = rho*z_tv + sqrt(1-rho^2)*z_noise`,
then lognormal-transformed into a positive spend value), not faked after the
fact. The realized Pearson correlation on raw spend comes out well below the
underlying `rho` — both the lognormal transform and the tier multiplier
(itself uncorrelated with `tv_linear`, since `tv_linear` is always-on) add
variance that dilutes it. `rho=0.9` with `TIER_MULTIPLIER=(1.0,1.5,2.2,3.2)`
was the balance found by sweeping: wider tier spreads (tried up to 9x)
pushed the realized correlation as low as 0.40 by swamping it with
tier-driven variance unrelated to TV spend. Landed on ~0.54, stable across
seeds, with threshold set to 0.5 and tier concentration still clearly
monotonic (~3.2x regular to championship).
