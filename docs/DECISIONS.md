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

---

## 2026-07-30 — Task 1.8: `simulate/controls.py`

Only 4 of the 7 controls needed new generation code — `broadcaster`,
`is_weekend`, and `tentpole_tier` already exist in `schedule.py` since
they're intrinsic to the event itself, not separately observed signals.
`team_interest`/`star_interest` are AR(1) latents smoothed with a 28-day
rolling mean; the first attempt used `rho=0.98` and produced a ~0.7
correlation with `tv_availability`'s seasonal cosine pattern in
`check_controls.py`'s printed matrix — not a deliberate confound, just a
near-unit-root process drifting slowly enough to spuriously track any other
slow-moving series over a finite sample. Dropped to `rho=0.8`, confirmed
against several seeds that the resulting correlation isn't just smaller but
actually changes sign run to run (-0.18 to +0.40), which is the signature of
ordinary finite-sample noise rather than a structural leak.

All 6 scalar controls get MinMax[0,1] scaling and `broadcaster` gets its
per-level vector in `dgp.py` (1.9), not here — `controls.py` only produces
the 4 new controls at raw, human-interpretable scale.

---

## 2026-07-30 — Task 1.9: `simulate/dgp.py`

This is where `truth.py`'s β/control_beta/season_intercept — explicitly
left as untuned placeholders back in task 1.2 — actually got calibrated
against `DESIGN.md §4`'s contribution-share targets. First run (with the
original placeholder betas) came out at baseline 51% / controls 16% /
always-on 16% / event-targeted 17%, against targets of 60-70% / 15-25% /
5-10% / 3-8% — controls were already fine, but media was contributing
roughly double what it should have and baseline was correspondingly
crowded out.

Because `hill_saturation`'s output doesn't depend on β at all (only K and
S), each channel's mean contribution scales exactly linearly with its β —
so hitting the target absolute contribution for the always-on bucket and the
event-targeted bucket was a matter of computing the needed scale factor
(0.359 and 0.278 respectively) and applying it uniformly within each
bucket, preserving the relative proportions already set between channels.
Landed on `tv_linear=235, out_of_home=65, display=2` (always-on) and
`ctv=60, paid_social=40, paid_search=70` (event-targeted) — stable across
four seeds: baseline 65.7-67.6%, controls 19.4-20.8%, always-on 6.7-7.5%,
event-targeted 5.9-6.1%, all inside band.

`sigma` similarly needed retuning for the §4 noise-floor target (true model
scored against its own output, 10-12% MAPE): 150 produced ~8.8%, scaled up
to 190 to land at ~11.3-11.5% across the same four seeds.

`extract_event_level`'s dilution and `geometric_adstock`'s carryover are
applied to always-on channels before Hill saturation; event-targeted
channels go straight to Hill saturation, no adstock — matching the
DESIGN.md §4 pipeline exactly, not a simplified version of it.

---

## 2026-07-30 — Task 1.10: commit a data sample

`.gitignore`'s blanket `data/generated/` rule changed to `data/generated/*`
plus a `!data/generated/events.parquet` exception, rather than removing the
ignore rule entirely — every other seed/variant someone generates locally
stays untracked, only the deterministic seed=0 default output is committed.

Reproducibility verified directly rather than by asserting it in a test:
regenerated the file three times (twice via the script directly, once via
`make data` through the actual Makefile target) and confirmed identical
SHA-256 checksums (`906b88c5...`) each time — including catching that
`make data` requires the venv activated first (it invokes plain `python`,
which isn't the issue `make` surfaced, just a reminder to activate before
running it; not a code change).

---

## 2026-07-31 — Task 2.1: `notebooks/01_eda.ipynb`

`ipykernel` was missing from dev dependencies -- `nbconvert --execute` needs
a real kernel, not just the conversion library. Added to `pyproject.toml`
and registered a project-specific kernelspec (`mmm-sports-venv`) rather than
relying on whatever default `python3` kernel happens to exist on a given
machine, so the notebook's execution environment is pinned same as
everything else in this repo.

Always-on spend is *regenerated* in the notebook (`generate_alwayson_spend`,
same seed) rather than read from the event-level parquet, since the parquet
only has spend at the days events actually happened -- plotting only those
points would misrepresent the flighting pattern entirely (most calendar
days aren't event days). Event-targeted spend, by contrast, is genuinely
event-level, so it's read straight from the parquet.

`check_eda.py` parses a `Baseline share: XX.X%` line out of the notebook's
captured stream output via `nbconvert` + `nbformat`, rather than requiring
the notebook to write a separate machine-readable file -- keeps the
notebook itself the single source of truth for both the human-facing report
and the machine-checkable assertion.

One real bug caught in review, not by the tests: a `§` character (used in
"DESIGN.md §4") came out corrupted (`�`) in one printed string, apparently a
Windows console encoding issue somewhere between the kernel process and
nbconvert's output capture -- interestingly, the same character survived
fine in markdown cells, only a `print()`-ed one was affected. Fixed by
avoiding non-ASCII symbols in printed output ("section 4" instead of "§4");
markdown prose can keep using `§` since it isn't routed through the kernel's
stdout stream.

Diagnostics from `DESIGN.md`'s 2.1 spec: always-on zero-spend inflation
came out `tv_linear` 65.8%, `display` 53.2%, `out_of_home` 26.7% -- exactly
the ordering the flighting parameters in task 1.6 were built to produce.
Tentpole-tier leverage: `championship` events average 1885 (thousand)
viewers on 3.7% of events but 4.8% of total viewership, vs `regular`'s 1354
average on 74% of events and 70.2% of total viewership -- visible confound,
not yet corrected for.

---

## 2026-07-31 — Task 3.1: `models/forest.py`

RF features are the *observed* `spend_*` columns (raw dollars as an
analyst would see them), never the DGP's internal adstocked/diluted
exposure -- `events.parquet` carries `mu` and `contrib_*` ground-truth
columns for scoring purposes (per `CLAUDE.md`'s "only tests and evaluation
scripts" carve-out), but `build_features()` deliberately excludes all of
them from the model's inputs. "Season holdout" read literally: train on
seasons 0-1, hold out season 2 entirely, distinct from Phase 5's later
random/temporal split schemes.

Real finding, not a bug: holdout MAPE (6.6%, seed 0) comes in *below* the
irreducible MAPE floor for that same season (10.6%) -- confirmed
consistent across four seeds (RF holdout MAPE 6.6-9.0% vs floor 10.1-12.6%
each time), so not a fluke. Traced it rather than shipping the number
uncommented: MAPE's denominator is the observed `y`, not the true `mu`, and
is therefore most sensitive exactly where a large negative noise draw made
`y` small. At the five smallest-`y` holdout points, RF's prediction sits
notably *below* `mu` (e.g. `mu`=1069, RF predicted 843, actual y=749) --
not because it detected the noise, but because RF's leaf predictions are
finite-sample averages of *noisy* training `y`, not the true conditional
mean, so in sparser regions of feature space RF's own estimation noise can
happen to land closer to a specific holdout noise realization than the
unbiased `mu` does. `fit_forest.py` now prints the floor next to every
MAPE (per `DESIGN.md §4`'s instruction, missed in the first pass) with an
explanatory note whenever a model's MAPE comes in under it, so this doesn't
read as the model having beaten irreducible noise anywhere else in the repo
either.
