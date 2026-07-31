# TASKS.md

One unchecked task per turn. Stop after each. Every task has a verify command.

If a task needs more than ~100 lines of change, stop and propose a split.

---

## Phase 0 — Orient

Scaffold, `.gitignore`, and `pyproject.toml` are already in place (built by
`bootstrap.sh`). Start by checking them, not by rebuilding them.

- [x] **0.1** Verify the scaffold: package imports, `reference/` untracked,
      dev dependencies installed.
      *Verify:* `pip install -e ".[dev]"` then `python -c "import mmm_sports"`;
      `git check-ignore -q reference/ && echo OK`

- [x] **0.2** `scripts/check_naming.py` + `scripts/naming_patterns.txt`.
      **Patterns, not a term list.** The file holds regexes matching
      *old-codebase naming conventions* — not client identifiers — so it is
      safe to commit and CI can run it. Seed it with at least:
      `[A-Z][A-Z0-9_]+\.[A-Z0-9_]+\.[A-Z0-9_]+` (three-part SQL identifiers),
      `BRAND_CH_\d+`, `TUNEIN_CH_\d+`, `CTRL_\d+`, `PAID_[A-Z_]+_(BRAND|TUNE_IN)`,
      `ACG_`, `GAME_MASTER`, `DAILY_IDX`, `SEASON_TYPE_IDX`.
      Scans **tracked files only**; `reference/` is excluded by construction.
      *Verify:* exits 0 on the current tree, non-zero when a string matching
      one of the patterns is planted in a tracked file
      *Note:* a whitelist of allowed vocabulary is unimplementable — it would
      flag every Python keyword and English word in a docstring. Patterns
      matching the shape of copy-pasted old code is the workable version.

- [x] **0.3** Internal consistency pass on `docs/DESIGN.md`: spot-check the
      curve-fit spec (§6) against the production model spec (§4) and against
      the rest of the document. Note any discrepancy in `docs/DECISIONS.md`.
      *Verify:* `docs/DECISIONS.md` has an entry

- [x] **0.4** `Makefile` (`data`, `test`, `lint`, `app`, `check` targets) and
      `.gitattributes` with `* text=auto eol=lf`. The latter matters: this repo
      is developed on Windows and deployed to Linux, and `bootstrap.sh` plus
      every future shell script needs LF endings.
      *Verify:* `make test` runs pytest; `git check-attr text -- bootstrap.sh`
      reports `text: auto`

- [x] **0.5** Untrack the bootstrap artifacts now that scaffolding is done:
      `git rm --cached bootstrap.sh START_HERE.md` (keep the local files).
      *Verify:* `git ls-files` no longer lists either

## Phase 1 — Simulation

- [x] **1.1** `simulate/schedule.py` — 3 seasons × ~250 events. Dates,
      broadcaster, weekend flag, multi-event days clustered in some weeks,
      three invented tentpole tiers.
      *Verify:* `check_schedule.py` prints events/season, the
      `n_events_on_date` distribution, saves a calendar density plot

- [x] **1.2** `simulate/truth.py` — frozen dataclass of every true parameter:
      per-channel α, K, S, β for all 6 channels; **7** control coefficients
      (including `tentpole_tier`, large, and `broadcaster` as a per-level
      vector); season intercepts; noise sigma. Encode the three pathologies
      from `DESIGN.md §3`.
      *Verify:* `pytest tests/test_truth.py` — ranges sane, pathologies present,
      and **asserts the tentpole coefficient is large relative to other
      controls** (it is the confounder; if it's small the design collapses)

- [x] **1.3** `transforms.py` — geometric adstock (numpy, convolution form).
      *Verify:* `pytest tests/test_transforms.py::test_adstock_decay` — known
      input, hand-computed expected output

- [x] **1.4** `transforms.py` — Hill saturation.
      *Verify:* `pytest ...::test_hill_monotone_and_bounded`

- [x] **1.5** `transforms.py` — event-level extraction: sample the adstocked
      daily series at `day_idx`, divide by `n_events_on_date`.
      *Verify:* `check_extraction.py` prints a worked example for a single-event
      day and a triple-event day side by side

- [x] **1.6** Daily always-on spend for `tv_linear`, `out_of_home`, `display`
      with realistic flighting (bursts, dark periods, seasonal ramp). Must
      produce low variance on `out_of_home`.
      *Verify:* `check_alwayson_spend.py` prints per-channel CV; **asserts**
      `out_of_home` CV is below threshold

- [x] **1.7** Event-targeted spend for `ctv`, `paid_social`, `paid_search`,
      **concentrated on higher tentpole tiers — this is the confound**
      (`DESIGN.md §3`). `ctv` correlated with `tv_linear`.
      *Verify:* `check_eventtargeted_spend.py` prints the correlation matrix
      and mean spend by tentpole tier; **asserts** `corr(tv_linear, ctv)`
      exceeds threshold **and** that event-targeted spend rises monotonically
      with tier

- [x] **1.8** **Seven** control variables per `DESIGN.md §3`, including
      `tentpole_tier` as a direct effect on `mu` and `broadcaster` as a nominal
      4-level categorical (per-level coefficients, not an ordinal slope).
      *Verify:* `check_controls.py` prints distributions, the correlation matrix,
      and mean viewership by broadcaster level and by tentpole tier

- [x] **1.9** `simulate/dgp.py` — assemble the response through the full
      two-mechanism pipeline. Tune until contribution shares hit the
      `DESIGN.md §4` targets.
      *Verify:* `scripts/simulate_data.py` writes `data/generated/events.parquet`;
      `check_dgp.py` prints the decomposition, **asserts shares are in band**,
      and prints the **irreducible MAPE floor** — the true model scored against
      its own output (`DESIGN.md §4`). Every later MAPE is read against this.

- [x] **1.10** Commit a data sample; make the full set reproducible from seed.
      *Verify:* fresh clone + `make data` reproduces byte-identical output

## Phase 2 — EDA

- [x] **2.1** `notebooks/01_eda.ipynb` — viewership distribution, spend over
      time, both mechanisms illustrated, baseline dominance made visible.
      Also print three diagnostic checks (not data treatment — this is
      synthetic ground truth, and the high-viewership tentpole events are
      deliberate signal, not outliers to clip; see `docs/DECISIONS.md`):
      a media spend **collinearity preview** (correlation matrix across
      channels), **zero-spend / flighting inflation** per channel (% of
      zero-spend days or events), and **tentpole-tier leverage** on the
      response distribution (viewership mean/share by tier). These foreshadow
      the 5.5c collinearity check and the tentpole confounder without
      duplicating either.
      The notebook must **write named figures to `docs/figs/`** and print a
      summary block, not just render inline.
      *Verify:* `scripts/check_eda.py` runs the notebook via `nbconvert`, then
      asserts each expected figure file exists and that the printed baseline
      share is inside the target band. Executing without error proves only that
      it didn't crash — which is not what this task is for.

## Phase 3 — Random Forest

- [x] **3.1** `models/forest.py` — RF on media + controls, season holdout,
      light hyperparameter search.
      *Verify:* `scripts/fit_forest.py` prints in-sample and holdout MAPE, R²

- [ ] **3.2** PDP per channel.
      *Verify:* `check_pdp.py` saves 6 PDPs and prints the implied response
      range per channel

- [ ] **3.3** ALE for `tv_linear` and `ctv`. Plot PDP and ALE on the same axes.
      *Verify:* `check_ale.py` saves the overlay and prints the max divergence
      between the two curves for the correlated pair

- [ ] **3.4** TreeSHAP — per-channel contribution decomposition.
      *Verify:* `check_shap.py` prints SHAP contribution shares **next to true
      shares** with the gap per channel

- [ ] **3.5** SHAP slices — by broadcaster, season, weekend, margin bin,
      tentpole tier.
      *Verify:* `check_shap_slices.py` prints the slice tables

- [ ] **3.6** Empirical response curve per channel: sweep spend, hold others at
      observed values, read the RF surface.
      *Verify:* `check_rf_curves.py` saves curves with the **true** curve
      overlaid

## Phase 4 — Prior bridge

- [ ] **4.1** `priors.py` — fit a Gaussian CDF to each RF curve → (MU, SIGMA, AMP).
      *Verify:* `check_curve_fit.py` prints fit quality per channel, saves overlays

- [ ] **4.2** Convert (MU, SIGMA, AMP) → Hill priors (K, S, β). Leave one
      channel uncovered so it falls back to a generic prior.
      *Verify:* `check_priors.py` prints RF-derived prior vs **true** (K, S, β)
      per channel with implied bias — this table is a headline result

## Phase 5 — Bayesian model

- [ ] **5.0** `transforms.py` — **vectorized pytensor adstock, no `scan`.**
      Precompute the `(n_days, l_max+1)` lag matrix in numpy; inside the model
      contract it against `alpha ** arange(l_max+1)`. See `DESIGN.md §4`.
      *Verify:* `pytest tests/test_transforms.py::test_pytensor_matches_numpy` —
      pytensor output equals the numpy convolution to 1e-8, plus a printed
      timing comparison against a `scan` reference implementation

- [ ] **5.1** `models/bayesian.py` — model definition only, no sampling.
      Non-centered channel coefficients, hierarchical season intercept,
      **α learned for all 6 channels** at full `l_max` (5.0 makes this
      affordable), Stage-2 priors wired in.
      *Verify:* `check_model_graph.py` renders graphviz, prints variable shapes

- [ ] **5.2** Prior predictive check.
      *Verify:* `check_prior_predictive.py` — 500 draws, asserts simulated
      viewership is plausible, saves the plot

- [ ] **5.3** Sample and persist. draws/tune 1000, 4 chains, target_accept 0.95.
      *Verify:* `scripts/fit_bayesian.py` writes `artifacts/bayesian.nc`;
      `check_diagnostics.py` prints r-hat, ESS, divergences

- [ ] **5.4** **`tests/test_recovery.py`**. Assert only what is stable:
      - **HARD ASSERT:** true `(α, K, S, β)` inside the 94% HDI for the three
        well-identified channels (`tv_linear`, `paid_social`, `paid_search`)
      - **HARD ASSERT (relative):** HDI width for `out_of_home` is at least 3×
        the width for `paid_search`. A statement about *relative* identifiability
        is robust to seed; asserting a channel fails to be identified is not —
        a lucky draw would fail a test that isn't broken.
      - **REPORT ONLY, no assertion:** whether truth falls inside the HDI for
        the pathological channels, with widths printed
      *Verify:* `pytest tests/test_recovery.py`, then read
      `check_recovery_report.py` output yourself

- [ ] **5.5** Posterior predictive check + **both holdouts**. Evaluate on the
      random-20%-stratified split and the temporal split (last N events of each
      season). See `DESIGN.md §7`.
      *Verify:* `check_holdouts.py` prints MAPE, R² and 94% coverage under both
      splits in one table, plus the recovery error under each

- [ ] **5.5b** Prior sensitivity — overlay prior and posterior per parameter.
      *Verify:* `check_prior_sensitivity.py` prints the share of parameters
      whose posterior moved materially off the prior, and names the ones that
      didn't

- [ ] **5.5c** Multicollinearity check — flag channel pairs with |r| > 0.7.
      *Verify:* `check_collinearity.py` prints the correlation matrix and
      **asserts** the `tv_linear`/`ctv` pair is flagged

- [ ] **5.6** Learned α vs **true** α per channel.
      *Verify:* `check_alpha.py` prints the comparison table with HDIs

- [ ] **5.7** **Prior ablation** — refit with generic priors instead of
      RF-derived ones. Does the RF prior help, hurt, or wash out?
      *Verify:* `check_prior_ablation.py` prints recovery error under both

- [ ] **5.8** Contribution decomposition: Bayesian vs SHAP vs truth.
      *Verify:* `check_scorecard.py` prints the three-way table

## Phase 6 — Lift calibration

- [ ] **6.1** `lift.py` — simulate a lift test on `out_of_home` from the true
      curve, realistic σ.
      *Verify:* `check_lift_sim.py` prints Δy_true, Δy_obs, σ, relative SE

- [ ] **6.2** Refit with `add_lift_test_measurements()`, persist separately.
      *Verify:* `fit_bayesian.py --lift` writes `artifacts/bayesian_lift.nc`

- [ ] **6.3** Before/after posterior for `out_of_home`, both against truth.
      *Verify:* `check_lift_effect.py` prints HDI width and distance from truth
      before and after; saves the overlay

## Phase 7 — Optimizer

- [ ] **7.1** Response curves: single-day and steady-state (`1/(1-α)`), with
      posterior bands.
      *Verify:* `check_response_curves.py` saves all 6 and prints the
      steady-state multiplier per always-on channel

- [ ] **7.2** `optimize.py` — SLSQP, total budget only.
      *Verify:* `pytest tests/test_optimize.py::test_budget_respected`

- [ ] **7.3** Per-channel floors and caps (contractual minimums).
      *Verify:* `pytest ...::test_floors_binding`

- [ ] **7.4** Max % change vs prior period + flighting constraints.
      *Verify:* `pytest ...::test_change_constraint`

- [ ] **7.5** Shadow prices — which constraints bind, and their cost in
      incremental viewership.
      *Verify:* `check_shadow_prices.py` prints the constraint table

- [ ] **7.6** Risk-adjusted objective (posterior quantile) vs optimizing the mean.
      *Verify:* `check_risk_adjusted.py` prints both allocations side by side

## Phase 8 — App

- [ ] **8.1** `app/Home.py` — artifact loading, caching, navigation. No logic.
      *Verify:* starts, loads in under 3s
- [ ] **8.2** Page — Data & EDA
- [ ] **8.3** Page — Random Forest (PDP/ALE, SHAP, empirical curves)
- [ ] **8.4** Page — Prior bridge (RF curve → Hill prior, vs truth)
- [ ] **8.5** Page — Bayesian diagnostics
- [ ] **8.6** Page — Response curves & ROAS
- [ ] **8.7** Page — Optimizer with constraint editor
- [ ] **8.8** Page — Scorecard (RF vs Bayesian vs truth)
      *Verify (each):* page renders, plus `scripts/check_app_purity.py` —
      AST-parses every file under `app/` and fails if any imports `pymc`,
      `sklearn`, `scipy`, `shap` or `arviz`, or defines a function longer than
      15 statements. "No business logic outside src/" isn't runnable as prose;
      this is a real proxy for it.

## Phase 9 — Ship

Some ship tasks are irreducibly manual. Where that's true it is labelled, so
the working agreement is honoured explicitly rather than broken silently.

- [ ] **9.1** README — synthetic data, invented domain, simulated lift test,
      limitations, above the fold.
      *Verify:* `scripts/check_readme.py` asserts the four required disclosures
      appear in the first 40 lines
- [ ] **9.2** CI: pytest + `check_naming.py` + `check_app_purity.py` on push.
      *Verify:* a deliberately failing commit on a scratch branch goes red
- [ ] **9.3** Deploy to Streamlit Community Cloud.
      *Verify:* `curl -sS -o /dev/null -w '%{http_code}' <url>` returns 200 and
      the page loads in under 3s
- [ ] **9.4** Blog posts in `docs/blog/` from `DECISIONS.md`. First one:
      **how much bias do ML-derived priors inject into a Bayesian MMM?**
      *Verify (manual):* human review — writing quality isn't machine-checkable

---

## Backlog — do not start without asking

- Geo-level simulation + CausalPy synthetic control for a real lift pipeline
- Weibull adstock (delayed peak) alternative
- Scale beyond 6 channels once recovery is clean
- Time-varying baseline
- Postseason as a second event type
