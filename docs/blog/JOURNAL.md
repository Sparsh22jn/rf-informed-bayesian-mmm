# Project Journal

An append-only, plain-language build log — the story of how this project
actually got built, task by task, including the *process* decisions and not
just the technical ones. `docs/DECISIONS.md` is the terse companion to this:
2-3 lines per task, written for grepping a specific parameter choice later.
This file is the narrative version, written for a human reading top to
bottom, including anyone (including future-me) trying to reproduce not just
the code but the way it got built.

**Rule for this file: never edit or delete a past entry. Only add new ones
at the bottom, in the order they happened.**

---

## Why this project is built one small task at a time

Before any code existed, the working agreement in `CLAUDE.md` set a
deliberately slow pace: implement exactly **one** unchecked task from
`TASKS.md` per turn, then stop and report — never chain straight into the
next task, even when the next step is obvious. Every task has to end in a
single runnable command that passes or fails; if a task can't be verified
that way, it gets split until it can. Every module that gets created also
gets a `scripts/check_*.py` that prints real numbers for a human to actually
look at — because for an MMM, a green `pytest` run only proves the code
executes, not that the number it produced is remotely sane. The entire
premise of building this on fully synthetic data with a known-in-advance
ground truth (`simulate/truth.py`) is that every stage can be scored against
the real answer — and that only pays off if a human is actually reading the
printed numbers along the way, not just the exit code.

That incremental discipline is itself one of this project's decisions, not
an accident of how the conversation happened to go — and it's the reason
this journal exists: so the *process*, not only the final repo, is something
someone else could pick up and repeat.

---

## Phase 0 — Orienting before building anything

Before any simulation code, Phase 0 was about verifying the ground the
project stands on rather than trusting that scaffolding worked: confirming
the package actually imports, that a pattern-based naming checker
(`scripts/check_naming.py`) exists to catch old-codebase-style identifiers
before they land in a public repo, and that `docs/DESIGN.md` itself is
internally consistent — its curve-fit spec and its production model spec
cross-checked against each other and against the rest of the document, with
no contradictions found.

Phase 0 also caught something purely bookkeeping-related: the actual work
for tasks 0.1 through 0.5 had already landed in an earlier commit, but the
checkboxes in `TASKS.md` were never flipped. Fixed by re-running every
task's stated verify command and only then checking the boxes — a small
reminder that "done" should mean "the verify command passes today," not
"someone remembers doing it."

Around this time the repo also went up on GitHub for the first time, named
`rf-informed-bayesian-mmm` rather than something generic — the reasoning
being that the name should say what makes this project different from a
typical MMM repo (Random-Forest-derived priors feeding a Bayesian model),
not just what domain it's in.

---

## Phase 1 — Building the synthetic world

### 1.1 — The event calendar (`simulate/schedule.py`)

The first real simulation code was the calendar: three seasons of roughly
250 live events each, not a plain weekly grid. Getting realistic clustering
— some weeks having several events on one day (national doubleheaders),
most weeks having none — meant not hand-scripting which days get multiple
events. Instead, each day's event count is drawn from a Poisson distribution
whose rate depends on the day of week (low midweek, high Friday/Saturday/
Sunday), with roughly 18% of weeks per season flagged as "cluster weeks"
where that rate doubles. The weekday rates needed one round of tuning after
an initial pass landed at ~196 events/season against a ~250 target — bumped
up (0.35→0.45 weekday, 1.7→2.2 weekend) until four different random seeds
consistently landed in the 210-276 range, which reads as normal
season-to-season variance rather than something broken.

One structural decision worth calling out: `day_idx`, the integer offset
used everywhere downstream to place an event on the calendar, is **not**
reset per season. It's a single running count across all three seasons plus
the 90-day off-season gaps between them, because Phase 1.6's always-on
adstock needs one continuous daily timeline to decay across — resetting it
per season would quietly break carryover at every season boundary.

### 1.2 — Ground truth (`simulate/truth.py`)

This is the answer key: a frozen dataclass holding every true parameter the
data-generating process will use — per-channel adstock decay, Hill
saturation shape, channel ceilings, seven control coefficients, three season
intercepts, noise. `CLAUDE.md`'s non-negotiable is that model code can never
import this file; only tests and evaluation scripts may, because the whole
point of the synthetic-data approach collapses the moment a model can see
the answer.

Two design calls worth recording. First, `tentpole_tier` (whether a game is
a big showcase event) gets a single large scalar coefficient, while
`broadcaster` (which network aired it) gets a *vector* of coefficients, one
per network — because `broadcaster` is genuinely nominal (there's no natural
order to four networks) while `tentpole_tier` is genuinely ordinal, and
forcing a nominal variable into a single ordered slope would quietly absorb
variance that isn't really linear. Second, the three event-targeted channels
(`ctv`, `paid_social`, `paid_search`) get an explicit `alpha = 0.0` rather
than no adstock parameter at all — because DESIGN.md is explicit that these
channels get *no* carryover (a "watch tonight" ad only matters tonight), and
giving them a real, comparable zero lets a later recovery test check that
the Bayesian model actually learns that zero instead of just never being
asked about it.

Only one of DESIGN.md's three deliberate "pathologies" belongs in this file:
`display`'s near-zero true effect (the dead channel everyone suspects but
that provably does nothing). The other two — `out_of_home`'s low spend
variance and the `tv_linear`/`ctv` spend correlation — are properties of how
spend gets generated, not of the truth parameters, so they show up later in
1.6 and 1.7 instead.

### 1.3 / 1.4 — Adstock and saturation (`transforms.py`)

Two small, sharp functions, deliberately placed outside `simulate/` — this
file has no dependency on the ground truth and gets reused later by the
actual Bayesian model, so it needed to be safe for model code to import.

Geometric adstock (carryover: a dollar spent today keeps echoing into future
days at a decaying rate) turned out to be a single line once framed
correctly — a causal `np.convolve` of the spend series against a decay
kernel, truncated to the max lag. Verifying it didn't rely on a round-trip
property test; it relied on hand-computing a single $10 spike decaying at
alpha=0.5 (10, 5, 2.5, 1.25, then nothing) and checking the code produced
exactly that, because the goal here is confidence in the actual numbers, not
just confidence the function runs.

Hill saturation (diminishing returns as spend rises) was similarly a
one-line closed form, verified against the one property the formula
guarantees by construction: spend equal to the half-saturation point `k`
always produces exactly 0.5.

### 1.5 — Sampling the daily series at event dates

Always-on spend lives on a daily calendar; events happen on specific dates,
sometimes several at once. `extract_event_level()` samples the adstocked
daily series at each event's `day_idx` and divides by however many events
shared that date — the idea being that a single day's broadcast attention
dilutes across simultaneous events. Verifying this meant pulling two *real*
days out of the actual generated schedule (one single-event day, one
triple-event day), forcing both to the exact same underlying daily spend
value, and printing them side by side — so the only difference visible in
the output is the dilution itself, not coincidental variation in the
underlying spend.

### 1.6 — Daily always-on spend, and a threshold that needed re-tuning

Always-on spend for `tv_linear`, `out_of_home`, and `display` needed
"realistic flighting" — bursts, dark periods, a within-season ramp — which
didn't have an obvious existing home in the file tree DESIGN.md sketches
out, so it became its own module, `simulate/spend.py`, mostly to keep
`dgp.py` (which assembles everything later) from ballooning past the
project's own ~150-line-per-module guideline.

The interesting part was `out_of_home`'s pathology: it's supposed to have
low day-to-day variance (long, chunky billboard-style contracts rather than
bursty campaigns), which is also *why* it's weakly identifiable later — a
channel that barely moves is hard for any model to pin a response curve to.
That came from giving it long 60-90 day flights, short 3-8 day dark spells,
and low noise, versus `tv_linear`'s short, bursty 7-20 day flights with more
dark time. The first coefficient-of-variation threshold chosen for the
check script (0.5) turned out to be tighter than what the full calendar
(including the shared off-season zero blocks) actually produces — real
runs landed `out_of_home` at ~0.6-0.65 consistently. Rather than force the
number down artificially, the threshold got moved to 0.8, which still sits
cleanly below `display`'s ~1.1 and `tv_linear`'s ~1.5 — the point was never
a specific number, just a real, reproducible gap between the pathological
channel and the others.

### 1.7 — Event-targeted spend, and finding a correlation that survives the noise

`ctv`, `paid_social`, and `paid_search` needed two things at once: spend
concentrated on higher tentpole tiers (the deliberate confound the whole
project exists to measure — planners spend more chasing games that were
always going to be big), and `ctv` specifically needed to be correlated with
`tv_linear`'s spend, since in real media planning those two get bought
together.

The correlation was generated properly rather than faked after the fact — a
standard bivariate-normal construction (`z_ctv = rho*z_tv +
sqrt(1-rho^2)*z_noise`) feeding into a lognormal transform for a positive
spend value. The first attempt, targeting `rho=0.9` with a wide tier
multiplier spread (up to 9x from regular to championship games), only
produced a realized Pearson correlation of about 0.40 on the raw dollars —
the tier multiplier itself is uncorrelated with `tv_linear` (which is
always-on, unrelated to any specific game's importance), so a wide tier
spread was adding variance that had nothing to do with TV spend and diluting
the correlation regardless of how high `rho` was pushed. Sweeping a small
grid of tier-spread and `rho` combinations found a balance — a narrower
~3.2x tier spread with `rho=0.9` — that held a stable ~0.54 correlation
across seeds while keeping the tier concentration still clearly monotonic.
Neither property got sacrificed for the other; it just took finding the
right operating point where both hold at once.

### 1.8 / 1.9 — The seven controls, and where the whole world finally gets stitched together

These two landed in the same sitting rather than one at a time, since 1.8's
own verify script needs actual simulated viewership to print — and
viewership doesn't exist until 1.9's assembly step exists. Doing them
together also made the dependency honest instead of working around it.

Three of the seven controls (`broadcaster`, `is_weekend`, `tentpole_tier`)
already existed from the calendar work back in 1.1 — they're intrinsic to
an event, not separately observed signals, so there was nothing new to
generate. The four that were new — `tv_availability` (a seasonal viewing
index), `team_interest` and `star_interest` (rolling 4-week indices), and
`competitiveness` (game margin) — surfaced one more case of an AR(1)
process drifting too slowly for its own good: `team_interest`'s first
version used a near-unit-root decay rate that let it wander slowly enough
to spuriously track `tv_availability`'s seasonal shape, producing a
coincidental 0.7 correlation between two variables that were never supposed
to be related. Confirmed it was noise rather than a real link by checking
whether the correlation's *sign* held across different random seeds — it
didn't (it ranged from -0.18 to +0.40), which is exactly the fingerprint of
a spurious correlation rather than a structural one. Slowing the process
down (a smaller decay rate) fixed it.

Then came the part that ties every earlier piece together: `simulate/dgp.py`
takes the calendar, the true parameters, both spend mechanisms, and the
controls, and actually produces a number — viewership. Always-on spend gets
adstocked over calendar time, diluted at multi-event days, then passed
through Hill saturation; event-targeted spend skips straight to Hill
saturation, no carryover. Every scalar control gets rescaled to [0,1] and
multiplied by its coefficient; `broadcaster` adds its own per-level effect;
a season intercept anchors the baseline; Gaussian noise sits on top.

This is also where the placeholder truth parameters set back in task 1.2 —
explicitly flagged at the time as "order of magnitude, not tuned" — finally
had something real to be tuned against. The first full run split response
into roughly 51% baseline, 16% controls, 16% always-on media, and 17%
event-targeted media, against DESIGN.md's target bands of 60-70% / 15-25% /
5-10% / 3-8% — media was contributing roughly twice what it should have. The
fix turned out to be simple once noticed: Hill saturation's output doesn't
depend on the channel coefficient at all, only on the spend and the shape
parameters, so a channel's mean contribution scales *exactly* linearly with
its coefficient. That meant hitting the target wasn't a search — it was
computing the one scale factor each media bucket needed and applying it
uniformly, which preserves how the channels sit relative to each other
while moving the bucket as a whole. Landed on shares that hold consistently
across seeds: baseline in the mid-60s, controls around 20%, always-on
around 7%, event-targeted around 6% — all inside band. The noise level
needed the same kind of check: the *true* model, scored against the data it
generated itself, is supposed to land at a 10-12% MAPE floor (the number
every later model's own MAPE gets read against) — the original noise
setting produced under 9%, so it went up until the floor actually landed
where DESIGN.md said it should.

### 1.10 — Proving reproducibility instead of assuming it

The last piece of Phase 1 wasn't new generation logic, just a promise worth
actually checking: that the entire pipeline, given the same seed, produces
byte-for-byte the same file every time. Rather than trust that and move on,
the file got regenerated three separate times — twice by calling the
generation script directly, once through the project's actual `make data`
command — and all three runs produced the identical SHA-256 checksum. Along
the way, running `make data` directly surfaced a small reminder rather than
a bug: the command relies on the project's virtual environment being
active, and typing it in a fresh terminal without doing that first fails
with a missing-package error that has nothing to do with the data pipeline
itself.

One small piece of `.gitignore` housekeeping came with this: the whole
`data/generated/` folder was ignored outright, but the project's own design
calls for committing one concrete sample so someone browsing the repo can
see real output without running anything. Rather than stop ignoring the
folder entirely, the ignore rule got a single carved-out exception for
exactly that one deterministic file — every other seed or variant a future
run produces locally stays out of the repo, same as before.

That closes out Phase 1: the entire synthetic world — calendar, ground
truth, both spend mechanisms, controls, and the assembled response — now
exists, is internally consistent with its own design targets, and is
provably reproducible from a single seed.

## Phase 2 — Looking at the world that got built

### 2.1 — The first real notebook, and a character that didn't survive the trip

Everything through Phase 1 was numbers printed to a terminal. This task was
the first time any of it became a picture — an actual Jupyter notebook
(`notebooks/01_eda.ipynb`), not a script, so the plots live as real cell
output someone can open and scroll through, not just files dropped on disk
as a side effect.

Running a notebook headlessly (so a check script can verify it, not just a
human eyeballing it once) needed one missing piece: `ipykernel`, which
hadn't been added as a dependency because nothing had needed to *execute* a
notebook yet, only convert one. Once that was in, the notebook got its own
registered kernel rather than depending on whatever `python3` kernel
happened to already exist on a machine — the same instinct as pinning any
other dependency.

One good habit paid off immediately: actually opening the rendered figures
instead of trusting that "the script exited 0" meant they looked right. The
always-on spend chart shows exactly what task 1.6 was built to produce —
`tv_linear` spiking on and off in bursts, `out_of_home` sitting almost flat
and continuous, both dropping to a clean zero during the off-season gaps.
The event-targeted scatter, colored by tentpole tier, makes the project's
central confound visible at a glance: the darkest dots (championship-tier
games) visibly float above the rest across all three channels, exactly the
pattern that will later fool a naive attribution model in Phase 3. Seeing
that shape appear in an actual chart landed differently than reading the
correlation number in task 1.7's terminal output.

Also caught something no test would have flagged: a `§` character in one
printed diagnostic line came out of the notebook's execution as a garbled
replacement character. Same symbol, used the same way, was fine everywhere
it appeared in plain markdown text — only the one routed through a live
kernel's `print()` output got mangled, apparently a Windows console
encoding quirk in how the kernel process's stdout gets captured. The fix
was small (spell out "section 4" instead of using the symbol in anything
that actually executes), but finding it meant reading the executed
notebook's raw output rather than trusting that no error was raised. Nothing
crashed. The character was just wrong.

## Phase 3 — Random Forest

### 3.1 — A number that looked wrong, and turning out to be right for an interesting reason

The first Random Forest fit — media spend plus all seven controls, trained
on two seasons, evaluated on the third it never saw — produced numbers that
looked almost too good: 93% R² in-sample, 80% on the held-out season, a
holdout error rate of 6.6%. That last number is the one that stopped the
task from being "done." This project has a specific, hard rule sitting in
its design doc: no fitted model can ever legitimately score better than the
*irreducible* error rate — the gap you'd still see even if you evaluated the
exact true model against its own randomly noisy data. That floor, for the
held-out season specifically, was 10.6%. The forest's 6.6% sat clearly below
it, which by that rule shouldn't be possible.

Rather than treat a suspiciously good number as a free win, the instinct
here was to distrust it and dig — the same discipline as re-reading a
notebook's raw output after the § incident, extended into a genuine
statistical question. The investigation ruled out the boring, worrying
explanation first: no leakage, no accidental use of a ground-truth column as
a model input — the RF only ever sees observed spend and controls, deliberately
excluding the true-response columns that ride along in the dataset for
scoring purposes. The real explanation turned out to be a property of the
*error metric itself*, not the model. That error rate is measured relative
to the actual noisy number that happened to occur, not the true underlying
average — and that measurement is most sensitive exactly at the unlucky
events where randomness happened to pull the real number down hard. Looking
directly at the worst five of those events, the forest's guess landed closer
to the actual (unlucky) outcome than the true average did — not because it
somehow sensed the randomness, but because a forest's predictions are built
from averaging real, noisy training examples in each little neighborhood of
similar events, not from some idealized clean average. In the sparser
corners of that neighborhood map, its own version of "noise" can coincidentally
land closer to one particular unlucky outcome than the honest average
would. Checked across four different random versions of the whole dataset
to make sure this wasn't a one-off coincidence tied to a single lucky seed —
it held every time, by a similar margin each time. That consistency is what
turned it from "suspicious" into "understood."

The fix wasn't to the model at all — it was to make sure this couldn't be
misread later. The fitting script now always prints that irreducible floor
directly next to any error number it reports, and adds a plain-language
note whenever a result comes in under it, so a future reader (including
future-me) doesn't mistake a metric quirk for the model having beaten
randomness itself.

### 3.2 — Watching a predicted flaw actually show up on screen

This project was built, from the very first task, around a specific bet:
that a flexible model fit to the data would get visibly *fooled* by a
correlation planted on purpose back in task 1.7 — that money spent on `ctv`
tends to get spent alongside money spent on `tv_linear`, because in real
media planning those two get bought together. Partial dependence plots are
a standard way of asking a fitted model "what do you think happens as I
turn this one dial, holding everything else fixed" — and they have a known
weakness: they don't actually hold everything else fixed in any real sense,
they average over whatever combinations actually occurred in the data. If
two channels are bought together, sweeping one drags the other along for
the ride without saying so.

That's not a hypothetical anymore. The chart for `ctv` shows predicted
viewership climbing by over 1,000,000 (thousand-viewer units) as `ctv`
spend increases — nearly twenty times larger than the true ceiling that
channel could possibly produce on its own. The model isn't wrong about the
data; it's accurately describing a world where high-`ctv` events also
happen to be high-`tv_linear` events, and quietly handing `tv_linear`'s
credit to `ctv`. Sitting right next to it, `out_of_home`'s chart is a flat,
noisy squiggle spanning a tiny range — not because that channel doesn't
matter to the story, but because task 1.6 deliberately gave it so little
spend variation that no model, however good, could confidently learn its
shape from this data. Two completely different failure modes, both
produced on purpose months (in project time) earlier, both now visible as
actual pictures instead of assertions in a design document.

The next task is where this gets a real fix rather than just a
demonstration: ALE, a variant built specifically to not fall for the
correlated-feature trap, plotted directly against these same PDP curves.

### 3.3 — Chasing a number that looked wrong until it turned into the best finding yet

ALE exists to fix PDP's exact flaw: instead of asking the model about
combinations that never really happened (high `ctv` paired with low
`tv_linear`, say), it only ever nudges a channel's spend within a narrow
neighborhood of games that were already similar to begin with — so it can't
accidentally invent a scenario that doesn't exist in reality. Plotting it
directly against the PDP curve from the previous task was supposed to show
ALE "correcting" PDP's distortion.

For `ctv`, it barely did. The two curves tracked each other closely instead
of splitting apart the way expected. That turned out to be informative on
its own: ALE only fully rescues you when the correlation between two
channels is concentrated in weird, unrealistic combinations. Here, the
`tv_linear`/`ctv` link is smooth across the *entire* spend range — baked in
deliberately back in task 1.7 — so even a narrow neighborhood of similar
`ctv` spend still contains games with correlated `tv_linear` too. ALE
reduces the confusion; it doesn't erase it, when the entanglement isn't a
tail phenomenon.

`tv_linear`'s own chart is what actually stopped the task. Both PDP and ALE
showed spending *more* on TV linear predicting *less* viewership — flatly
backwards from how the channel was actually built, where it has the
strongest true positive pull of anything in the simulation. Nothing about
that was subtle enough to wave away, so it got investigated properly rather
than written up as some interesting quirk. First move: rule out a
Random-Forest-specific glitch by fitting the plainest possible
linear regression on the same inputs — same negative sign showed up there
too, meaning this wasn't a tree-model artifact, it was something about the
inputs themselves.

That search turned up a genuine bug: `n_events_on_date` — how many games
shared a single day, directly relevant since attention on a shared day
splits between games — had been sitting in the dataset the whole time and
never made it into the model's feature list back in task 3.1. An honest
oversight, now fixed. It improved the model's overall accuracy a little.
It did not fix the backwards sign.

The real answer needed one more link in the chain: `tv_linear` correlates
with `ctv` (by design, task 1.7). `ctv`, completely separately, correlates
*very* strongly with `tentpole_tier` (by design, also task 1.7 — spend
concentrating on the biggest games). And `tentpole_tier` itself has a huge
real effect on viewership, entered on purpose as the project's central
confounder all the way back in task 1.2. `tv_linear` has almost no direct
relationship with `tentpole_tier` at all — but it's one hop away, connected
through `ctv`. Asked to divide credit among all three at once, the model
ends up assigning `tv_linear` a negative slice, not because TV spend hurts
viewership, but as a side effect of being adjacent to a much bigger, better-
connected confound.

This landed as a better result than the task originally set out to produce.
The project was built, months of task-time ago, around two separate
planted pathologies — a correlated pair, and a variable that's both a
direct cause and a driver of other spend. This is the first moment they
visibly collided with each other, corrupting a channel that has no direct
relationship with the real confounder at all. It's also the clearest
argument yet for why `tentpole_tier` has to be handed to the eventual
Bayesian model directly, in Phase 5, rather than something a model is
expected to infer purely from watching where the money went.

### A correction, the same day — the explanation above was tested and didn't survive

The three-way story above was a genuinely reasonable read of the evidence
at the time: `tv_linear` barely correlates with `tentpole_tier` directly,
but `ctv` correlates with both, so credit seemed to be leaking through that
middle connection. It was plausible enough to write down. It was also,
it turned out, not the real explanation — and the only reason that's known
now instead of just believed is that the natural next question got asked:
if holding the tier constant is what should fix this, does it actually fix
it?

So `tv_linear`'s ALE curve got computed four separate times — once within
each tentpole tier on its own, instead of one curve mixing all of them
together. If the tier-chain theory were right, freezing the tier should
have let `tv_linear`'s real, positive relationship show through in each of
those four curves. It didn't. All four still sloped the same wrong
direction, championship-tier games included, where the sample is smallest
and the theory's leverage should have been most visible.

That negative result is what actually cracked it. Checking how strongly
`tv_linear` and `ctv` move together *within* a single tier — rather than
across the whole mixed dataset — showed a correlation of about 0.9, not the
0.5 pooled number everyone had been reading up to this point. Mixing all
four tiers together had been *hiding* how tightly linked those two channels
really are, not revealing it — because tentpole tier adds its own extra
swing to `ctv` spend that has nothing to do with `tv_linear` at all, and
averaging over that padding waters down the true relationship underneath
it. Two channels that correlated with each other that strongly, sitting in
the same model as separate inputs, is enough on its own to make either
one's individual credit unstable and land anywhere, sign included — no
third variable required.

The tentpole-chain theory wasn't a wasted detour. It was a real hypothesis,
stated plainly, checked against new evidence, and dropped the moment the
evidence didn't support it — which is a different thing from getting the
right answer on the first guess, and a more honest one to have on the
record. Nothing about the earlier entry got deleted or rewritten; it's
still there, as what was believed before the tier-stratified chart existed
to check it against.

### 3.4 — A third method, asked to settle what the first two found

TreeSHAP is a fundamentally different technique from PDP and ALE — instead
of sweeping one channel and averaging, it distributes each individual
prediction's value fairly across every input feature, event by event, using
some genuinely elegant game theory (imagine figuring out how much credit
each player on a team deserves for a win, by checking how the outcome
would've changed under every possible combination of who's on the field).
The plan for this task was simple: use that third, independent method to
either confirm or contradict what PDP and ALE had already been saying about
`tv_linear` and `ctv`.

The first attempt produced numbers that looked broken — every single media
channel came back with a SHAP contribution share within a rounding error of
zero, while the true shares from the data generator were clearly not zero.
That's not what a bug usually looks like when you check it, though: SHAP
values carry a strict mathematical guarantee (they have to add up exactly
to the model's actual prediction), and that guarantee held perfectly. So
the numbers weren't wrong — the question being asked was subtly different
from the one intended. SHAP was measuring "how much does this channel's
*actual* spend push the prediction away from a *typical* game" by default.
What was wanted was "how much does this channel's actual spend push the
prediction away from *zero spend on this channel*" — because that's how the
real, true contribution numbers are defined in this project. Two
genuinely different questions that happen to produce the same units,
which is exactly the kind of mismatch that's easy to miss and worth
catching.

Fixing it meant explicitly telling the SHAP tool to compare every event
against a version of itself with zero media spend, instead of an average
event. That single change turned the results from "everything near zero,
looks broken" into a clean, third confirmation of what the previous two
tasks had already found — `ctv` credited with roughly 45 percentage points
more than its real share, `tv_linear` still landing negative, this time
confirmed all the way down to individual events rather than just an
averaged curve. Three unrelated methods, three separate implementations,
one consistent answer. That's about as much confidence as this kind of
investigation can offer without the ground truth already in hand — which,
in this project's unusual case, it happens to be.

---

*(Next entry goes here after the next completed task — see `CLAUDE.md`'s
working agreement.)*
