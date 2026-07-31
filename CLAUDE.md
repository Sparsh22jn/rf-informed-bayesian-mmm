# CLAUDE.md

Project context and working agreement. Keep under ~70 lines — past that it
stops getting followed.

## What this is

An end-to-end marketing mix model for **live-event viewership**, built on
**fully synthetic data** generated from a known data-generating process. The
domain is an invented broadcaster.

Three stages, not two models bolted together:

1. **Random Forest** — flexible, non-parametric. PDP/ALE plus SHAP give
   per-channel contribution and an empirical response curve per channel.
2. **Prior bridge** — fit a parametric curve to each RF response curve, convert
   its parameters into Hill saturation priors.
3. **Bayesian MMM** — hierarchical, RF-informed priors, learned adstock decay,
   full posteriors feeding a constrained budget optimizer.

Because the data is synthetic, **every stage can be scored against known
truth.** That is the thing public MMM repos cannot do, and it is the point.

## Working agreement — READ THIS FIRST

- Implement **exactly ONE** unchecked task from `TASKS.md` per turn, then stop
  and report. Do **not** start the next task.
- Before writing code, state your approach in **fewer than 5 bullets** and wait.
- `docs/DESIGN.md` is **reference describing the target system**. NOT a task
  list. Never implement from it directly.
- Every task ends with a **single runnable command that passes or fails**. If a
  task can't be verified that way, say so and propose a split.
- Every task producing a module also produces a `scripts/check_*.py` that
  **prints real numbers**. The human runs it. A green test is not sufficient
  evidence an MMM is correct — it's usually silent when it's wrong.
- Modules under **~150 lines**. Split rather than grow.
- Append 2–3 lines to `docs/DECISIONS.md` after each task; also append a short
  narrative entry to `docs/blog/JOURNAL.md` (append-only — never edit past entries).

## Non-negotiables

- **Ground truth lives only in `src/mmm_sports/simulate/truth.py`.** Model code
  never imports it. Only tests and evaluation scripts do. Violating this
  invalidates every result in the repo.
- **Never fit a model inside Streamlit.** Fit offline, persist to `artifacts/`,
  load with `@st.cache_resource`.
- **Models and optimizer import and test without Streamlit.** No logic in app
  callbacks.
- All naming stays inside the invented domain — see `DESIGN.md §3`.
  `scripts/check_naming.py` enforces this in CI.
- `reference/` is gitignored and never read into `src/`.

## Core architecture

Two media mechanisms, modeled differently — this is what makes the project
distinct from standard weekly MMM:

- **Always-on** spend lives on the **daily calendar**, is adstocked over
  calendar time, then sampled at event dates and divided by the number of
  events that day (attention dilutes across simultaneous broadcasts).
- **Event-targeted** spend is already attributed to a specific event and gets
  **no adstock** — a "watch tonight" ad only affects tonight.

Both then pass through Hill saturation and a channel coefficient. Response is
raw viewership in thousands, not logged.

## Stack

Python 3.11+ · scikit-learn + SHAP · PyMC + PyMC-Marketing · scipy.optimize ·
arviz · Streamlit · pytest · pandas / numpy

## Conventions

- Package in `src/mmm_sports/`. Entry points in `scripts/`.
- Type hints on public functions. **Docstrings state units and array shapes.**
- Seeds explicit and configurable — never bare `np.random`.
- Spend in whole dollars, viewership in thousands. Unit confusion is the most
  common silent bug here; state units everywhere.
