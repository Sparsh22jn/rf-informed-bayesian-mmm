# DESIGN.md

**Reference material describing the target system. This is not a task list.**
Work is sequenced in `TASKS.md`.

---

## 1. The problem

An invented broadcaster spends across paid media to drive viewership of live
events. The response unit is not a calendar week — it is an **individual
event**, irregularly spaced, with its own attributes. Media splits into two
mechanisms with different temporal behaviour, and the baseline (the audience
the event would have drawn anyway) dominates.

**Why this is harder than standard MMM:** ~750 observations, media explaining
roughly 10% of response, a ~65% baseline, two channel families needing
different exposure models. Getting a defensible answer is an identification
problem, not a fitting exercise.

---

## 2. The three-stage pipeline

```
                 ┌─────────────────────────────────────┐
   synthetic  ─→ │  STAGE 1: Random Forest             │
   panel         │  · fit on media + controls          │
                 │  · PDP / ALE per channel            │
                 │  · SHAP → contribution decomposition│
                 │  · empirical response curve/channel │
                 └────────────────┬────────────────────┘
                                  │
                 ┌────────────────▼────────────────────┐
                 │  STAGE 2: Prior bridge              │
                 │  · fit Gaussian CDF to each curve   │
                 │    → (MU, SIGMA, AMP)               │
                 │  · convert → Hill priors (K, S, β)  │
                 └────────────────┬────────────────────┘
                                  │
                 ┌────────────────▼────────────────────┐
                 │  STAGE 3: Bayesian MMM              │
                 │  · RF-informed per-channel priors   │
                 │  · hierarchical, non-centered betas │
                 │  · LEARNED adstock α                │
                 │  · posteriors → constrained optimizer│
                 └─────────────────────────────────────┘

   every stage scored against  truth.py
```

**The methodological argument this makes:** a flexible ML model discovers the
shape of the response; a Bayesian model imposes structure, quantifies
uncertainty, and enables constrained decision-making. Neither alone is enough.

---

## 3. Domain and naming

Invented broadcaster, invented events, invented calendar. All naming stays
inside this vocabulary — `scripts/check_naming.py` enforces it.

### Channels — 6 total

| Channel | Mechanism | Role in the DGP |
|---|---|---|
| `tv_linear` | always-on | strong, high adstock |
| `out_of_home` | always-on | **low spend variance** — chunky long flights, weakly identified |
| `display` | always-on | **near-zero true effect** — the channel everyone suspects |
| `ctv` | event-targeted | **correlated with `tv_linear`** — bought together |
| `paid_social` | event-targeted | moderate, fast |
| `paid_search` | event-targeted | strong, fast, low saturation point |

Three pathologies, each attached to a channel where it's *realistic*:
low-variance OOH, a correlated TV/CTV pair, a dead display channel. Realism
matters — an arbitrary assignment reads as contrived.

Six is a dial, not a law. Scale up once recovery is clean (backlog).

### Mechanism naming

**"Always-on" vs "event-targeted."** Every planner understands the distinction
and it maps exactly to the modeling split: calendar-adstocked versus
event-attributed.

### Controls — 7

| Code | Description | Expected sign |
|---|---|---|
| `tv_availability` | viewing availability index | + |
| `broadcaster` | network — **nominal, 4 levels** | mixed |
| `team_interest` | rolling 4-week interest index | + |
| `star_interest` | rolling 4-week performer index | + |
| `is_weekend` | weekend indicator | + |
| `competitiveness` | final margin (lower = closer) | − blowouts shed viewers |
| `tentpole_tier` | 3 invented tiers + regular | + large |

**`broadcaster` is nominal, not ordinal.** Generate it as a genuine categorical
with a distinct mean per level; model it as a vector of coefficients indexed by
broadcaster, non-centered. Encoding four networks as `0,1,2,3` against a single
slope imposes a linear order that doesn't exist and will quietly absorb variance
that belongs elsewhere.

### `tentpole_tier` is the model's central confounder — by design

Tentpole tier does **two** things, and that is the point:

1. It enters `mu` **directly** as a control with a large coefficient. A holiday
   showcase draws a huge audience regardless of what was spent on it.
2. It **drives event-targeted spend**. Planners concentrate tune-in budget on
   the games that were always going to be big.

So spend and outcome are both caused by a third variable. That is the exact
identification problem MMM exists to solve, and the reason naive attribution
massively over-credits event-targeted channels. Building it in deliberately
means:

- SHAP in Phase 3 will over-attribute to `paid_search` / `paid_social`, and
  ground truth lets you **measure by how much**
- the Bayesian model can only avoid the same trap because the control is in the
  design matrix — worth demonstrating with a fit that omits it
- the lift test in Phase 6 becomes motivated rather than decorative

An earlier draft had tentpole affecting spend only. That version had no
confounding in it at all, which would have made the whole project too easy.

---

## 4. Data design

**Unit:** one event. 3 invented seasons, ~250 events each.
**Response:** average audience in thousands, **raw scale**, range ~200–6000.

**Scale: raw dollars.** Spend is *not* normalized to [0,1]. Keeping dollars
means `K` is a half-saturation point in dollars and `β` is a viewership ceiling
— both directly interpretable and both directly comparable against `truth.py`.
Normalizing would make every recovered parameter unit-dependent and the
recovery test far less legible.

```
ALWAYS-ON
  daily calendar spend (raw $)
    → geometric adstock over calendar time (α per channel, l_max=30)
    → sample at event dates:  X[e] = adstocked[day_idx[e]]
    → ÷ n_events_on_date[e]              # dilution
    → Hill saturation (K in $, S)
    → × beta                              # beta in 000s of viewers

EVENT-TARGETED
  spend attributed to a specific event (raw $)
    → NO ADSTOCK
    → Hill saturation (K in $, S)
    → × beta

CONTROLS → MinMax[0,1] → × beta_controls

mu = intercept[season] + always_on + event_targeted + controls
       # controls include tentpole_tier (large, direct) and broadcaster
       # (nominal, per-level coefficient)
y  ~ Normal(mu, sigma)
```

Hill: `f(x) = x^S / (K^S + x^S)`

**Adstock implementation — no `scan`.** Geometric adstock inside PyMC is
usually written with `pytensor.scan`, which is slow enough to force
compromises (short `l_max`, few draws, α learned for only some channels).
Avoid it. Precompute a lag matrix `X_lag` of shape `(n_days, l_max+1)` once in
numpy, then inside the model:

```python
weights   = alpha[:, None] ** pt.arange(l_max + 1)   # (n_channels, l_max+1)
adstocked = (X_lag[None, :, :] * weights[:, None, :]).sum(axis=-1)
```

One vectorized contraction, no recursion. This is what makes learning **all**
α feasible at full `l_max` and full draws.

**Calibration targets** (design targets in `truth.py`; tune the DGP until the
simulation hits them): baseline ~60–70%, controls ~15–25%, always-on ~5–10%,
event-targeted ~3–8%. If media explains 40% of response, the simulation is too
easy and proves nothing.

**Noise target.** Set `sigma` so that the *true* model — the DGP itself,
evaluated with the true parameters — achieves roughly **10–12% MAPE** on the
data it generated. That number is the irreducible floor: no fitted model can
beat it, and every holdout metric should be read against it. A model at 13% is
close to optimal; the same 13% with a 3% floor would mean something entirely
different. Report the floor alongside every MAPE in the repo.

Sigma also sets the power of the recovery test. Too much noise and nothing is
identifiable, so the test proves only that the data was uninformative. Print
the floor at task 1.9 and sanity-check it before building anything on top.

---

## 5. Stage 1 — Random Forest

`models/forest.py`. Fit on media spend + controls, season holdout.

**Interpretation layer:**

- **PDP per channel** — and **ALE for `tv_linear` / `ctv`**. PDPs extrapolate
  into implausible regions when features are correlated, which is exactly the
  pathology built into that pair. Showing PDP *and* ALE side by side, and
  explaining why they disagree, is one of the strongest details in the project.
- **SHAP** — TreeSHAP for per-channel contribution decomposition, aggregated to
  channel shares and sliced by broadcaster / season / weekend / margin bin /
  tentpole tier.
- **Empirical response curve** — sweep spend for one channel holding others at
  observed values, read off the RF prediction surface.

**Scored against truth:** SHAP contribution shares vs true shares; RF holdout
MAPE and R².

**The honest caveat, and it is a feature:** SHAP on observational data is
*associational*, not causal. The RF will happily absorb baseline and
seasonality into correlated media. Ground truth lets you **measure that bias
directly** — which becomes the sharpest blog post in the set: *how much bias do
ML-derived priors inject into a Bayesian MMM?*

---

## 6. Stage 2 — Prior bridge

`priors.py`. Numerical curve matching, not an analytical conversion:

1. Summarize each RF response curve as a Gaussian CDF → `(MU, SIGMA, AMP)`
2. Evaluate that curve on a **grid of ~500 spend points** spanning 0 to roughly
   `(MU + 3·SIGMA)`, in raw dollars
3. **`scipy.optimize.curve_fit` a Hill function to that grid**, with bounded
   parameters and sensible initial guesses (`K₀ = MU`, `S₀ = 2`, `β₀ = AMP`)
4. Record fit **R²** per channel. A poor fit is itself information: it means
   the RF curve isn't Hill-shaped and the prior should be widened accordingly.
5. Fall back to an analytical approximation if `curve_fit` fails to converge
6. Emit a prior spec the Bayesian model consumes
7. **Deliberately leave one channel uncovered** so it falls back to a generic
   prior — mirrors incomplete upstream coverage

Curve-fitting beats a closed-form conversion because the two functional forms
don't map cleanly onto each other. Matching them numerically over the spend
range that actually occurs is the honest version, and the R² tells you when
it worked.

**Scored against truth:** print RF-derived prior vs true `(K, S, β)` per
channel with the implied bias. This table is the empirical answer to "are
ML-derived priors any good?"

---

## 7. Stage 3 — Bayesian MMM

`models/bayesian.py`. PyMC.

- Per-channel Hill priors from Stage 2; generic priors for the uncovered channel
- Hierarchical **non-centered** channel coefficients
- Season-level hierarchical intercept
- **Learned α** with per-channel Beta priors — the upgrade over fixing adstock
  decay up front
- Directional priors on controls (HalfNormal where sign is known, Normal where
  not)
- `y ~ Normal(mu, sigma)`, raw scale

Sampling: draws/tune 1000, 4 chains, `target_accept=0.95`. Prior predictive
check before fitting.

### Validation — two holdouts, deliberately

**Run both splits and report both.** This is a headline result, not a detail.

- **Random 20%, stratified by season.** Convenient, and what most MMM work
  does. It also **leaks**: adstock makes each event's exposure a smooth
  function of the prior 30 days, and the rolling interest controls are smooth
  too, so a randomly held-out event sits between two training events with
  near-identical features. The model interpolates rather than predicts.
- **Temporal.** Hold out the last N events of each season — within-season so
  the hierarchical season intercept stays estimable, unlike withholding a whole
  season. This is the deployment question: *predict events you haven't seen.*

Report both MAPEs side by side and explain the gap. With ground truth you can
go further and show which split yields better **parameter recovery**, not just
better fit. "Random holdout says 11%, temporal says 19%, here's why the first
number is a lie" is a stronger portfolio artifact than any single metric.

**Other checks**
- **`tests/test_recovery.py`** — hard-assert true `(α, K, S, β)` inside the 94%
  HDI for the three well-identified channels, and hard-assert HDI width for
  `out_of_home` is at least 3× the width for `paid_search` (a relative
  identifiability claim, robust to seed). Whether truth falls inside the HDI
  for the pathological channels is **reported, not asserted** — a lucky draw
  recovering a weak channel isn't a bug, and hard-asserting the opposite
  outcome would make the test fail on noise rather than on a real defect.
- **Prior sensitivity** — overlay prior and posterior for every parameter.
  Heavy overlap means the data isn't informative and the result is
  prior-driven. Report the share of parameters that moved.
- **Multicollinearity** — flag channel pairs with |r| > 0.7 before fitting, so
  the `tv_linear`/`ctv` pair is a documented expectation rather than a surprise
  in the posterior.
- r-hat < 1.01, ESS > 400, divergences < 1%
- Posterior predictive coverage

**Ablation worth running:** fit once with RF-informed priors and once with
generic priors. Does the RF prior help, hurt, or wash out? With ground truth
you can answer this, and almost nobody in the field has.

---

## 8. Lift test calibration

Simulate a geo lift test from the known true saturation curve, targeting
`out_of_home` (the weakly-identified channel):

1. Pick spend `x`, change `Δx`
2. `Δy_true = f_true(x + Δx) − f_true(x)`
3. `Δy_obs ~ Normal(Δy_true, σ)` with **realistic** σ — 20–40% relative
   standard error. Geo tests are noisy; small σ is cheating.
4. Feed via `add_lift_test_measurements()` — a likelihood term constraining the
   saturation curve, not a prior on ROAS

**Show:** posterior before vs after, both against truth, with HDI widths.
**Every writeup states this is a simulated experiment.**

---

## 9. Optimizer

`scipy.optimize` SLSQP / `trust-constr` over fitted response curves.

Use **steady-state** curves for forward-looking allocation: sustaining $X/day
accumulates by `1/(1-α)`, so high-α always-on channels are worth more under
sustained spend than a single-day curve implies. Event-targeted channels have
no α and no multiplier — which is precisely why the two families cannot be
optimized on the same footing.

Constraints: total budget · per-channel floors and caps (**contractual partner
minimums**) · max % change vs prior period · flighting limits.

**Report shadow prices.** "This partner floor costs ~2.1M incremental viewers
against the unconstrained allocation" is the most valuable number in the app.

Bayesian variant: optimize a posterior quantile (e.g. 20th percentile) rather
than the mean, and show the allocation shift away from wide-posterior channels.

---

## 10. Streamlit app

One app. Artifacts loaded, never fit.

1. **Data & EDA** — schedule, spend, viewership, the two mechanisms illustrated
2. **Random Forest** — PDP/ALE, SHAP contributions, empirical response curves
3. **Prior bridge** — RF curve → Hill prior, per channel, against truth
4. **Bayesian diagnostics** — traces, r-hat, PPC, holdout fit
5. **Response curves & ROAS** — single-day vs steady-state, uncertainty bands
6. **Optimizer** — budget slider, constraint editor, binding constraints,
   shadow prices
7. **Scorecard** — RF vs Bayesian vs truth, side by side

---

## 11. Repo layout

```
├── CLAUDE.md
├── TASKS.md
├── README.md
├── pyproject.toml
├── Makefile
├── .gitignore              # reference/, artifacts/, data/generated/
├── reference/              # gitignored — local working notes
├── data/generated/         # simulated panel (sample committed)
├── docs/
│   ├── DESIGN.md
│   ├── DECISIONS.md        # append-only; feeds the blog
│   └── blog/
├── scripts/                # entry points + check_*.py diagnostics
├── src/mmm_sports/
│   ├── simulate/
│   │   ├── schedule.py
│   │   ├── truth.py        # TRUE parameters — models never import this
│   │   └── dgp.py
│   ├── transforms.py       # adstock, saturation, event-level extraction
│   ├── models/
│   │   ├── forest.py       # RF + PDP/ALE + SHAP
│   │   └── bayesian.py
│   ├── priors.py           # RF curve → Hill prior bridge
│   ├── lift.py
│   ├── optimize.py
│   └── viz.py
├── tests/
├── artifacts/              # gitignored
└── app/
```

---

## 12. Done

- `pytest` green including recovery
- Holdout metrics for both models, produced by this repo's code
- Scorecard: RF vs Bayesian vs truth
- App deployed, loads in under 3s
- README stating plainly: synthetic data, invented domain, simulated lift test,
  known limitations
- A blog post per phase in `docs/blog/`
