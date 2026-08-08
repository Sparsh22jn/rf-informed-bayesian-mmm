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

---

## 2026-08-01 — Task 3.2: PDP per channel

New `models/interpret.py` rather than adding to `forest.py` -- Stage 1's
full interpretation layer (PDP now, ALE/SHAP/empirical curves across
3.3-3.6) would push `forest.py` well past the ~150-line module cap if it
all lived in one file. `interpret.py` only wraps `sklearn.inspection.
partial_dependence`; `check_pdp.py` loads the model already persisted by
`fit_forest.py` rather than refitting.

Confirms `DESIGN.md §5`'s predicted pathology directly, not just in theory:
`ctv`'s PDP implied response range came out at 1175.4 (thousand viewers) --
**nearly 20x** its true ceiling (`beta_ctv=60`, `truth.py`). PDP averages over
the *observed* joint distribution of the other features while sweeping
`ctv`, and because `ctv` and `tv_linear` are correlated by design (task
1.7), high-`ctv` grid points disproportionately co-occur with high-
`tv_linear` events in the real data -- PDP silently credits `ctv` with
`tv_linear`'s effect. `out_of_home`'s PDP, by contrast, is flat noise across
a ~6-point range on a ~1400 baseline -- correctly reflecting that channel's
weak identifiability from task 1.6's low-variance flighting. Both are
exactly the pathologies these channels were built to produce, now visible
in an actual curve rather than asserted in a docstring. Task 3.3's ALE
overlay is where the `tv_linear`/`ctv` correction gets measured directly.

---

## 2026-08-01 — Task 3.3: ALE for `tv_linear` and `ctv`

`compute_pdp` reimplemented as a manual sweep-and-average loop (was
`sklearn.inspection.partial_dependence`) rather than adding a second grid
mechanism -- ALE needs a custom grid (its own quantile bin edges) to be
directly comparable to PDP on the same points, and `partial_dependence`'s
`grid_resolution` path doesn't expose that cleanly. Side effect: sklearn's
`partial_dependence` silently clips its default grid to the 5th-95th
percentile; the manual version sweeps the full observed range, so 3.2's
`ctv` range grew from 702.6 to 1175.4 (thousand viewers, ~20x its true
ceiling now vs. the earlier ~11x) -- updated 3.2's entry above and
`JOURNAL.md` to match rather than leave a stale number, since 3.2 hadn't
been pushed yet.

`ctv`'s PDP and ALE curves track each other closely (max divergence 104.4)
rather than diverging sharply as expected -- because the `tv_linear`/`ctv`
correlation (task 1.7) is smooth across the whole spend range, not
concentrated in extreme/unrealistic combinations, so ALE's local buckets
still contain correlated neighbors and can't fully separate the two.

**Real finding, investigated rather than shipped as-is:** `tv_linear`'s PDP
*and* ALE both showed a clearly negative relationship with viewership,
despite `tv_linear` having the strongest true positive effect of any
channel (`beta=235`, `truth.py`). Ruled out an RF-specific artifact first
-- a plain `LinearRegression` on the same features showed the same negative
sign, so not a tree-model quirk. Found and fixed a real bug along the way:
`build_features()` (task 3.1) omitted `n_events_on_date` entirely, even
though it's directly observable and the true always-on contribution is
diluted by it (`corr(contrib_tv_linear, n_events_on_date) = -0.415`). Added
it; holdout R² improved 0.805 -> 0.816, but the sign flip persisted, so
that wasn't the (sole) explanation.

The actual explanation is a three-way *transitive* confound: `tv_linear`
correlates with `ctv` (0.54, task 1.7), and `ctv` separately correlates very
strongly with `tentpole_tier` (0.765 -- tier-driven spend concentration,
also task 1.7), and `tentpole_tier` has a large genuine positive effect on
viewership (raw correlation 0.48, but a *negative* partial coefficient of
-138 in the fixed-dummy-trap linear regression -- the tier effect is
getting absorbed into the correlated spend channels instead). `tv_linear`
itself correlates with `tentpole_tier` at only 0.036 -- essentially zero,
direct. It's confounded by proximity, one hop away through `ctv`, not by
any real relationship of its own. This goes beyond what `DESIGN.md §3`
explicitly named (the `tv_linear`/`ctv` pair, and `tentpole_tier`'s dual
direct-effect-plus-spend-driver role, described separately) -- it shows
those two designed pathologies compounding into a transitive effect neither
one produces alone, and it's the concrete reason `tentpole_tier` has to be
an explicit model input in Phase 5's Bayesian model rather than something
inferred from spend alone.

---

## 2026-08-01 — Correction to the 3.3 entry above: the transitive-confound theory was wrong

Tested the transitive-tentpole-confound theory directly by adding a
tier-stratified `tv_linear` ALE to `check_ale.py` (computing the curve
separately within each of the 4 tentpole tiers, holding the tier constant).
If the theory were right, holding tier fixed should break the chain and let
`tv_linear`'s true positive slope emerge. It didn't -- all four tiers
still show `tv_linear` decreasing.

That result pointed at the actual cause: checked `corr(spend_tv_linear,
spend_ctv)` *within* each tier and found 0.89-0.93 -- noticeably
**stronger** than the 0.54 pooled correlation reported in 1.7 and 3.3,
not weaker. Pooling across tiers was diluting the correlation, not
concentrating it -- tier adds extra variance to `ctv` (via the tier
multiplier) that has nothing to do with `tv_linear`, which drags the
pooled correlation down. The tentpole-tier chain in the entry above isn't
the real mechanism; it was a plausible-looking hypothesis from the pooled
correlation numbers that a direct test disproved.

**Corrected explanation:** `tv_linear` and `ctv` are severely collinear
(~0.9, not ~0.5) once tier is accounted for, and both sit in the same
model as separate predictors. With two predictors that correlated, their
*individual* coefficients/partial effects become unstable and can land on
almost any sign while the model's combined fit stays fine -- classic
multicollinearity, not a multi-hop confound. This is a simpler, single-cause
story than the previous entry claimed, and it's a more direct hit on
`DESIGN.md §3`'s actually-designed pathology (the `tv_linear`/`ctv` pair)
than the transitive version was. Left the earlier entry as written, per
this file's append-only convention -- this correction is the record of
having checked it and found it wrong, not a reason to erase what was
believed at the time.

---

## 2026-08-02 — Task 3.4: TreeSHAP contribution decomposition

`interpret.py` split into `interpret.py` (PDP/ALE) and `interpret_shap.py`
(TreeSHAP) once adding SHAP pushed it to 167 lines -- past the ~150 cap.

The first version of `compute_shap_values` used `shap.TreeExplainer(model)`
with no background data, i.e. `tree_path_dependent` perturbation -- the
library default. Every media channel's SHAP share came back within a point
of zero, dwarfed by their true shares (tv_linear true 5.5% vs shap 0.3%,
etc.), which looked like a bug. It wasn't a computation error -- the
efficiency identity (`sum(mean(shap per feature)) + base_value ==
mean(prediction)`) checked out exactly -- it was a mismatched baseline.
`TreeExplainer`'s default background is the *average* event, so a channel's
SHAP value measures deviation from average spend, not distance from zero.
`truth.py`'s contributions, by contrast, are defined relative to zero spend
(`hill_saturation(0) = 0`). Two different reference points, not a
comparable pair of numbers.

Fixed by adding an interventional-SHAP path: a background set with every
media spend column forced to zero (other features left at each row's real
value), matching `truth.py`'s own convention. Hit a `shap` library dtype
error on the interventional path specifically (`Cannot cast array data
from dtype('O') to dtype('float64')`) caused by `pd.get_dummies`' `bool`-
dtype columns -- worked fine for the default path, broke the interventional
one. Fixed with an explicit `.astype(float)` before constructing the
background/foreground arrays.

With the zero-spend background, the numbers now read consistently with
3.2/3.3's findings rather than contradicting them: `ctv` shap=46.6% vs
true=1.8% (gap +44.8%) -- the same over-attribution PDP showed, now via a
completely different method. `tv_linear` shap=-1.5% vs true=5.5% (gap
-7.1%) -- same negative-credit finding as the ALE chart, confirmed again.
The two SHAP dependence scatters (spend vs. that event's own SHAP value,
colored by the other channel) show this isn't just a mean-level artifact:
`tv_linear`'s SHAP values sit at or below zero for essentially every single
event, not just on average.

---

## 2026-08-08 — Task 3.5: SHAP slices

Reused `compute_shap_values` with the same zero-spend background from 3.4
rather than adding new interpretation logic -- 3.5 is a slicing exercise
over 3.4's output, not a new method, so it stayed a script (no new `src/`
module). `margin_bin` didn't exist as a column -- built as quartiles of
`competitiveness` (`closest`/`close`/`wide`/`blowout`), the other four
slice dimensions were already there.

The `ctv` over-attribution isn't flat across slices -- it climbs
monotonically with `tentpole_tier`: 43.4% (regular) -> 51.4% (showcase) ->
54.0% (rivalry) -> 58.6% (championship), against `broadcaster`/`season`/
`is_weekend`/`margin_bin` all staying within a couple points of each other.
Directly confirms the misattribution concentrates exactly where the
`tv_linear`/`ctv` correlation and the tentpole-tier spend concentration
overlap most (high-tier games get proportionally more of both), rather
than being a uniform bias across the dataset.

---

## 2026-08-08 — Task 3.6: empirical response curves

New `interpret_curves.py` (not folded into `interpret.py`, which was
already near budget) -- `compute_empirical_curve` builds one representative
synthetic event (median for continuous columns; for the `broadcaster_*`
one-hot group specifically, the single most common level set to 1 rather
than an independent per-column median, which would otherwise produce an
invalid state with zero or several levels "on" at once) and sweeps just
the target channel against that fixed backdrop -- deliberately different
from 3.2's PDP (which averages over every row) because Phase 4 needs one
clean curve per channel to fit a Gaussian CDF to, not a marginal average
already distorted by correlated features.

Baselined the RF curve at its own starting grid point before comparing
to the true curve (which starts at exactly 0 by construction,
`hill_saturation(0)=0`) so the two are on the same footing. For
event-targeted channels the grid's start is the minimum *observed* spend,
not literally $0 (spend is never exactly zero there), so the baseline is
an approximation of true zero, not exactly it -- close enough given
`hill_saturation` at a small fraction of `K` is already near 0 for every
one of these channels.

A fourth independent method now, and it agrees with the first three: `ctv`
massively inflated (RF range 0->1274 vs true 6->57), `tv_linear` still
backwards (RF -172->0 vs true 0->186). New texture beyond the tv_linear/ctv
pair: `paid_search` recovers the *cleanest* of all six -- tracks the true
curve closely up to ~$10k then diverges as the grid runs into sparser data,
consistent with it being the fastest-saturating channel (smallest `K`) so
most of its true curve sits inside the well-observed low-spend region.
`paid_social` gets the right direction but undershoots the true ceiling
substantially and flattens early. `display`'s RF curve drifts to -77 despite
a true effect near 0 -- read as noise around a genuinely weak signal (the
whole curve, RF and true alike, is small relative to `tv_linear`/`ctv`'s
scale), not a story worth a theory the way the `tv_linear`/`ctv` pair was.
This variation in how well each channel recovers is exactly the texture
task 4.2 anticipates when it plans to leave one channel's curve fit
uncovered, falling back to a generic prior.
