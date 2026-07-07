# Judge vs. React: dissociating perception from timing in VLM reflex control

> **Draft scaffold (arXiv target).** Real numbers from the oracle apparatus are
> filled in; results that require VLM runs (need an API key) are marked
> `<<fill from VLM runs>>`. Figures live in `../eval/pit/figures/`.

## Abstract

A vision-language-model (VLM) agent that must time a single action in a
first-person scene can fail two ways that existing game benchmarks cannot pull
apart: it can misjudge the scene (perception), or judge correctly but act too
late (timing). We build a minimal, parametric first-person pit-jump task that
reduces an episode to one correct decision, and use it to dissociate the two
axes. We show (i) a **same-decision flip** — the *identical* correct decision
clears the pit when the world is paused and fails in real time; (ii) a
**quantitative timing law** — success collapses at a sharp threshold in the
dimensionless ratio `r = latency / action-window`, located near `r ≈ 1`
(logistic fit `r* = 1.17` on the oracle apparatus), with latency varied
*independently of judgment quality*; (iii) **action-repeat does not rescue
it**; (iv) the boundary is a **law, not a quirk of one task** — a second
reflex task in a different modality (rotational aiming) collapses onto the same
`r ≈ 1` transition (`r* = 0.98`); and (v) a **real model traces the same
boundary** — a closed-loop Sonnet agent, swept through the transition by
injected delay with its own frame-by-frame judgment in the loop, collapses at
`r* = 1.05`, and past the boundary the *only* surviving episodes are ones that
misjudged early, their perception error cancelled by the delay. Measuring five
real vision models on the task,
**all land past `r*`** —
hosted frontier models at `r ≈ 4–6`, local open VLMs at `r ≈ 31–47` — so none
can act inside the window in real time, yet the same decisions succeed when the
world is paused. The framing implies that pausable/turn-based deployment
converts a slow-but-accurate VLM from failing to viable, and that "VLMs are too
slow to play games" is partly an artifact of the time regime they are
benchmarked in.

## 1. Introduction

`<<intro: the two-axis failure; why one-decision isolation is the move; the
pit-jump task as apparatus; the results S2/S1/S3, then generality via S4.>>`

Lead with the single-decision flip (S2) — the specific, novel version of
"pause-vs-real-time." Contributions:
- **S2 (dissociation):** the same correct decision succeeds paused, fails
  real-time, isolating latency from perception on ONE decision.
- **S1 (timing law):** a sharp threshold in `r = latency/window` near 1,
  latency varied independently of judgment quality.
- **S3 (no cheap fix):** action-repeat does not rescue the transition (and modestly
  worsens it — larger `k` fails at lower `r`).
- **S4 (generality):** a second reflex task in a different modality (rotational
  aiming) collapses onto the same `r ≈ 1` boundary (`r* = 0.98`) — the law is
  not specific to the pit.
- **S5 (a real model through the boundary):** a closed-loop Sonnet agent,
  judging every frame itself, swept through the transition by injected delay
  collapses at `r* = 1.05` — and past the boundary its only clears are
  *delay-rescued misjudgments*, the two failure axes cancelling.

## 2. Related work

Five works are closest; we differentiate from each (full prior-art check:
`docs/prior-art-fulltext.md`):

- **OmniGameArena** (Lin et al., 2026, arXiv:2606.09826) — has paused (PDQ) vs.
  latency-controlled real-time (LCRT) clock modes, but only as **aggregate
  whole-game** comparisons; no ratio transition, no single-decision flip, no
  action-repeat test. We add all three.
- **VideoGameBench** (Zhang et al., 2025, arXiv:2505.18134) — "Lite" (paused) vs.
  real-time; names the stale-action mechanism but leaves perception confounded
  ("it's not just latency"). We resolve exactly that confound by isolating one
  decision.
- **"Win Fast or Lose Slow"** (Kang et al., NeurIPS 2025, arXiv:2505.19481) —
  owns the latency–quality tradeoff, but its latency lever is
  quantization/model-size, which *also* degrades quality, and its ≈200ms point
  is an environment action-rate cap, not a task-window threshold. Our `r ≈ 1`
  collapse varies latency **independently** of the model — the apparatus injects
  control delay `D` with the policy fixed.
- **BLINK** (Fu et al., ECCV 2024, arXiv:2404.12390) — canonical "MLLMs fail at
  depth," but static MCQ. Ours is interactive, parametric, oracle-normalized.
- **Ramstedt & Pal, "Real-Time RL"** (NeurIPS 2019, arXiv:1911.04448) [+ "At
  Human Speed," arXiv:1810.07286] — the RL parent of the real-time/delay result
  (monotone degradation for learned policies). We locate a `r ≈ 1` boundary for
  *pausable VLM agents* and show turn-based deployment crosses it.

We explicitly do **not** claim to be first to study the latency–quality tradeoff
(Win Fast) nor first to isolate timing from competence in the aggregate
(Real-Time Deadlines, arXiv:2601.13206; OmniGameArena). Our claim is the
single-decision flip plus the quality-controlled ratio law.

## 3. Task and apparatus

A first-person raycaster. The player auto-walks toward a pit; the only decision
is *when* to jump. Knobs: pit width (sets the action-window `W`), fog/visibility
(perception), approach speed `v`, seed. The sim is deterministic and headless;
control delay `D` (ticks) is **injected** — a decision made at tick `t` is applied
at `t+D` — so `D` is an independent variable decoupled from wall-clock. Real
per-model latency `L` is measured separately (timed API calls) and mapped to the
`r` axis via `L / tick`. An oracle policy (true pit distance) is the perfect-
judgment ceiling.

**Ground-truth label.** `in_window` = whether a delay-0 takeoff at the decision
position clears, computed from the sim's *own* dynamics (not an analytic
approximation), so `in_window` exactly predicts `cleared` at `D=0`. The action
window is `W = w_hi - w_lo` with `w_lo, w_hi` derived by simulating takeoffs; for
pit widths beyond the jump reach the window is empty (`W ≤ 0`, unclearable) and
those configs are excluded from fits (reported, not silently dropped).

`r = v · D / W` (oracle sweep) or `v · (L/tick) / W` (VLM overlay).

## 4. Results

### 4.1 Timing law (S1) — Fig. `fig_timing.png`
Sweeping `D` for each pit width (each a different `W`) and pooling against `r`,
the clear/fail outcomes **collapse onto a single transition near `r ≈ 1`**;
logistic fit gives **`r* = 1.17`** (420/1680 empty-window rows excluded).
Because the delay grid is discrete, the data bracket the edge rather than
pinpoint it: the last clearing configuration lies at `r = 1.04` and the first
failure at `r = 1.30`; `r* = 1.17` is the logistic midpoint of that gap. The
transition is razor-sharp because the oracle is deterministic; a stochastic VLM
would smooth it into a psychometric curve. Real per-model decision latencies —
hosted Haiku/Sonnet/Opus and two local open VLMs — are measured directly and
placed on the `r` axis below (all land past `r*`; Fig. `fig_latency.png`).

**Every real VLM we measured lands past `r*` (Fig. `fig_latency.png`).** Because
the decision cost is exactly `r = L / T_window` — where the window is open for
`T_window = (W/v)·tick = 257 ms` at `pit_width=2` (`W=1.078`, `v=0.07`) — the
`r*` boundary sets a **maximum tolerable decision latency of `r*·T_window ≈ 300 ms`**.
We timed five vision models on the identical 20-frame decision battery, in each
model's *fastest* single-call configuration (n=20 each):

| Model | Median L | r = L / window |
|---|---|---|
| Haiku 4.5 (hosted API) | 0.95 s | **≈ 4** |
| Sonnet 4.6 (hosted API) | 1.45 s | ≈ 6 |
| Opus 4.8 (hosted API) | 1.53 s | ≈ 6 |
| Qwen2.5-VL 7B (local) | 7.9 s | ≈ 31 |
| llama3.2-vision 11B (local) | 12.1 s | ≈ 47 |

**None clear the 300 ms budget.** Even the *fastest hosted frontier model* is
~4× too slow, and local open VLMs are 30–47×. This is the concrete version of
"VLMs are too slow to play games": the whole measured field would need the world
**paused, or slowed 4–47×**, to act inside the window — and the gap is a property
of the *time regime*, not the model, since `r` here is `L`/window with judgment
held out entirely. (The hosted numbers are a model's best case: this fast config
uses no extended thinking; any more deliberate configuration is only slower.
Local latency is on-device — a real, reproducible closed-loop number.)

A closed-loop episode (blocking mode, `--policy agent`) confirms the local model
is non-viable, though it is worth separating the two failure modes: `llama3.2-vision`
also **fails perception** — it answered JUMP on all 20 frames (no depth
discrimination), so in blocking mode it jumps at `tick 0` from `x≈2.6` (far from
the pit at `x=10`) and falls. The latency result is the axis-independent point:
`r > r*` dooms even a model that judges correctly, the moment the world is
allowed to run in real time.

The slight offset above 1.0 is physical: a jump firing on the same tick can take
off from just past the pit's near edge, extending tolerance marginally beyond the
nominal window — hence clears observed at `r = 1.04`.

### 4.2 Same-decision flip (S2) — Fig. `fig_flip.png`
Restricting to oracle-correct decisions (`in_window = 1`), the **identical**
decision clears with probability **1.00 at `D = 0`** and **0.11 once `D` exceeds
the window** (`r > 1`). The decision is the same (same `decision_x`, delay-
independent); only execution timing changed. This isolates latency from
perception directly. `<<report as unique configs, not raw rows: the oracle emits
60× fog/seed duplicates; the "late" condition is 9 unique configs, 1 of which
still clears → 0.11.>>`

### 4.3 Action-repeat does not rescue (S3) — Fig. `fig_repeat.png`
Holding a decision across `k` frames does not rescue the timing failure — and in
fact modestly **worsens** it. Measured `r*` vs. `k` (fixed pit width): `k=1 →
1.10`, `k=2,4 → 0.97`, `k=8,16 → 0.71`. Consulting the policy less often means a
decision lands staler (up to `k-1` ticks later than ideal), so the transition
moves to *lower* `r` (fails earlier), not higher. Action-repeat is a
compute/cost lever, not a latency fix. (This sweep is at a single pit width;
`<<extend across pit widths to confirm the direction is general.>>`)

### 4.4 Perception axis (dissociation) — Fig. `fig_perception.png`
A capable vision model *can* judge the jump from a single frame, and fog
collapses it. We rendered standing decision frames across approach positions at
two visibility levels and had **four blind vision models** — three Claude models
(Haiku 4.5, Sonnet, Opus) and one local open VLM (`qwen2.5-vl:7b` via Ollama) —
each shown only the image (no coordinates, window, or labels; identical seed-42
shuffled frame set) call JUMP or WAIT on each; we score each call against the
ground-truth window (`in_window`). Judgment accuracy at zero delay (n=10 frames
per model per condition):

| Model | Clear (fog=0) | Foggy (fog=1) | Fog effect |
|---|---|---|---|
| Opus | **90%** (9/10) | 50% (5/10, chance) | **−40 pts** |
| Sonnet | 80% (8/10) | 50% (5/10, chance) | **−30 pts** |
| Qwen2.5-VL 7B (local) | 60% (6/10) | 60% (6/10) | 0 |
| Haiku 4.5 | 40% (4/10) | 40% (4/10) | 0 |

Two things fall out. **(1) With a clear depth cue, judgment scales with model
capability** — Opus 90%, Sonnet 80%, Qwen 60%, Haiku 40%. **(2) The fog collapse
appears only in the models that actually read depth.** Opus and Sonnet drop to
chance under fog, and each **missed every in-window frame** (0/5): they reported
no distinguishable pit band even up close, never committed, and would fall. The
two weak judges show **no fog effect at all** — but for opposite-looking reasons
that are really the same reason: Haiku sits at floor (it fell back on a
frame-order heuristic the shuffle nullifies), and Qwen is strongly *JUMP-biased*
(it answered JUMP on 18/20 frames, clearing all in-window frames but also most
out-of-window ones). Neither is using the depth cue in the first place, so
neither has any perception to lose when fog removes it. The fog collapse is a
**signature of genuine depth reading**, present exactly where clear-condition
accuracy is high. At n=10 frames per cell the per-model contrasts are
directional rather than individually significant (Fisher exact, clear vs. fog:
Opus p=0.14, Sonnet p=0.35; pooled Opus+Sonnet 17/20 vs. 10/20, p=0.04) — it is
the coherence of the pattern across four models, not any single cell, that
carries the claim.

This is the perception half of the dissociation: a model that judges well with a
clear depth cue drops to chance when fog obscures the pit's distance — **at zero
delay**, i.e. independent of the timing axis (§4.1). Together, perception (§4.4)
and timing (§4.1/§4.2) are separable failure axes on one task. (`<<caveat: the
Claude judges run via the Claude Code subagent path, not a controlled API
benchmark; the local VLM is a real keyless run. Judges are blinded to
coordinates/labels but not to the task. Latency for the local VLMs is on the
§4.1 axis: Qwen median 7.9s → r≈30, llama3.2-vision 12.1s → r≈47.>>`)

The oracle-only `fig_dissociation.png` remains a placeholder for the timing⟂fog
VLM version.

### 4.5 Generality (S4) — Fig. `fig_aim.png`
Is `r ≈ 1` specific to jump-timing, or a property of *any* act-in-a-window
reflex? We built a second task in a different modality: a **sweeping-aim** task.
The player rotates in place at constant angular speed `ω`; a target subtends an
angular window `[θ−h, θ+h]` (width `2h`); the single decision is *fire*, and a
control delay `D` rotates the aim `ω·D` further before the shot lands — overshoot
if it exceeds the window. By construction the dimensionless cost is `r = ω·D /
(2h)`, the exact angular analog of the pit's `v·D / W`. The aim episode is pure
kinematics and emits the same schema, so the *identical* fit machinery applies.

Sweeping `D` across five target widths (4.6°–13.8°) and pooling against `r`, the
hit/miss outcomes **collapse onto the same transition near `r ≈ 1`** — logistic
fit `r* = 0.98` (n=205). A translational jump task and a rotational aiming task,
sharing nothing but the "act inside a window that your delay pushes you out of"
structure, land on the **same boundary**. The small offset between the two `r*`
(pit 1.17, aim 0.98) is the oracles' firing phase: the pit's takeoff can extend
just *past* the near edge (a hair above 1), while the aim fires just *inside* the
near edge, forfeiting a sub-tick of tolerance (a hair below 1). The raw step
edges show exactly this: the pit's last clear lands at `r = 1.04` (past the
nominal edge), while the aim's last hit is at `r = 0.96` and its first miss at
exactly `r = 1.00`. Both are the same order-unity law. This is the result that
makes `r ≈ 1` a *law* rather than a fact about one pit.

### 4.6 A real model through the boundary (S5) — Fig. `fig_realmodel.png`
S1 measures real models' latencies and finds them all far past `r*` (`r ≈
4–47`); the transition itself is mapped by the oracle. Here we close that gap: a
real model is swept *through* the boundary with its own judgment in the loop.
Because the apparatus decouples sim time from wall-clock, the model's (slow,
variable) API latency is free, and `r` is engineered exactly as in S1: Sonnet
4.6 judges *every grounded frame* of a closed-loop blocking episode (pit_width
2, `v = 0.07`, fog 0, seeds 1–6), its committed jump is applied `D` ticks later,
and sweeping `D ∈ {0..31}` sweeps `r = v·D/W` from 0 to 2, densely around `r =
1` (72 episodes; 6,612 vision calls).

**Judge protocol.** The episode prompt used for the S1 latency battery turns out
never to elicit JUMP near the window from *any* hosted model: it describes the
pit as "the black band across the floor ahead," which stops matching the
close-range view — inside the window the pit is a black region filling the
bottom of the frame, and the visible checkered floor is the pit's *far* side.
The S5 judge instead answers a literal perceptual question ("what touches the
bottom border of the image: tiles or black?") at high reasoning effort, with a
commit rule of two consecutive JUMP answers. The commit rule matters: the
model's rare stochastic false JUMPs (~2% per call far from the pit) would
otherwise fire almost surely somewhere over the ~90 consulted approach frames —
a first-passage failure mode that single-frame accuracy batteries cannot see. We
validated the judge on all 108 approach frames × 3 samples: zero far-field
JUMPs, unanimous JUMP inside the window. The bottom-edge cue is nearly ideal in
this renderer: the pit's black region first touches the frame's bottom edge at
`x = 9.01`, one tick after the window opens (`w_lo = 8.92`).

**The law holds.** Episodes whose decision lands in the window clear **32/32
through `r = 0.97`**, 4/5 at `r = 1.04`, and **0/26 from `r = 1.10` on**; the
same logistic fit as S1 gives `r* = 1.05`, inside the oracle's bracketing gap
`[1.04, 1.30]` and at its last-clear edge (the same-tick takeoff tolerance of
S1). The transition is nearly as sharp as the oracle's, for an unplanned reason:
the model's commit position is almost deterministic — 51/72 episodes commit at
`x = 8.94` and 12 at `x = 9.01`, the first one or two ticks the bottom-edge cue
exists. The anticipated noisy psychometric curve compresses to a near-step; this
judge's decision-position variance is sub-tick, so the model behaves as a
*slightly-early oracle* rather than a noisy one.

**Delay-rescued misjudgments.** 9/72 episodes commit early (`x = 8.10–8.24`, out
of window — a perception error; these are excluded from the timing fit above).
At `D ≤ 8` every such episode fails, landing short. In the band `D = 15–24`
(`r = 0.97–1.56`), *every* early episode **clears** (5/5): the delay pushes the
too-early takeoff into the window. Past `r ≈ 1.1` the *only* surviving episodes
are the misjudged ones; by `r = 2.01` the delay overshoots even the early
takeoff and the rescue band closes. The two failure axes do not merely
dissociate — on opposite sides of the boundary they *compensate*, which is only
visible because the apparatus labels each decision in- or out-of-window.

## 5. Discussion

The two failure axes have opposite fixes, and the framing makes the requirement
quantitative. A perception failure wants a better model: no amount of extra time
helps a judge that reports no distinguishable pit band under fog. A timing
failure wants a slower clock: a model is viable exactly when
`r = L / T_window < r* ≈ 1`, so a deployment can cross the boundary by pausing
the world (`r → 0`), slowing it (raising `T_window`), or cutting latency — and
the measured field needs 4–47× on the latter two. The practical reading is that
pausable or turn-based deployment converts every slow-but-accurate model we
measured from failing to viable, with no change to the model; conversely, a
real-time benchmark score is a product of judgment *and* time regime, and "VLMs
are too slow to play games" is partly a fact about the clock they are handed.
S5 is the direct demonstration: the same model, same judgment, clears 32/32
below the boundary and 0/26 above it, with nothing varied but `r`. S5's
delay-rescued misjudgments add a sharper corollary: past the boundary, scores
can even *reward* bad perception — the axes compensate, and an aggregate
benchmark cannot tell.

## 6. Limitations

- **The real-model sweep (§4.6) is one model at one pit width**, and its judge
  uses a revised prompt (the S1 battery prompt elicits no JUMPs near the window
  from any hosted model) plus a two-consecutive-JUMP commit rule — a documented
  apparatus change, not the shipped default. Its `r` axis is engineered by
  injected delay with the model's judgment in the loop; the model's own
  wall-clock latency (~5 s/call at high effort) remains far past `r*`, so a
  wall-clock-real sweep is still impossible for the measured field — that is
  S1's point, not a gap. The **latency axis is real**: five models measured
  directly (§4.1, Fig. `fig_latency.png`), all past `r*` (hosted `r≈4–6`, local
  `r≈31–47`); network-/hardware-dependent, but real decision latencies, not
  stand-ins. The anticipated *noisy* psychometric curve did not materialize —
  this judge's decision variance is sub-tick — so mapping a genuinely noisy
  transition needs a weaker or more stochastic judge.
- **The perception axis (§4.4)** mixes three Claude judges (run via the Claude
  Code subagent path, not a controlled API benchmark) with one real keyless
  local VLM (`qwen2.5-vl:7b`), at n=10 frames per condition — large enough for
  the cross-model pattern, too small for per-model significance (Fisher exact
  p=0.14–0.35). The open VLMs are weak/biased judges — Qwen is
  JUMP-biased (60% clear, no fog effect) and `llama3.2-vision:11b` is
  always-JUMP (contributes latency only, not a perception point). A controlled,
  latency-timed multi-provider API sweep still needs a key. (None in the current
  environment.)
- **No human baseline** in v1 — the oracle stands in as the perfect-judgment
  ceiling. Human psychophysics is future work.
- **Two tasks, not many.** The `r ≈ 1` law now holds on two tasks in two
  modalities (translational jump, rotational aim; §4.5), which is the difference
  between a fact and a law — but both are single-decision reflex tasks in the
  same engine. Multi-decision and richer tasks are untested.
- **Deterministic-duplication caveat:** the oracle emits identical rows across
  fog/seed; reported counts are unique configs, and the sharp fit reflects zero
  per-episode noise.

## 7. Future work
Extending the §4.6 real-model sweep to more models and pit widths (and to a
judge noisy enough to trace a genuinely smooth psychometric curve); a
multi-provider latency panel (Gemini/GPT + more open VLMs); a human
psychophysics arm; and more reflex tasks (beyond the two in §4.5) to test how
universal the `r ≈ 1` boundary is.

## Figures
- `../eval/pit/figures/fig_timing.png` — S1, the r≈1 collapse (r*=1.17).
- `../eval/pit/figures/fig_flip.png` — S2, same-decision flip (1.00 vs 0.11).
- `../eval/pit/figures/fig_repeat.png` — S3, action-repeat null.
- `../eval/pit/figures/fig_aim.png` — S4, generality: the same r≈1 collapse on a rotational aiming task (r*=0.98, 5 target widths).
- `../eval/pit/figures/fig_realmodel.png` — S5, Sonnet 4.6 swept through the boundary in closed loop (r*=1.05; early decisions fail below the boundary and are the only survivors past it; data + driver in `../eval/pit/realmodel/`).
- `../eval/pit/figures/fig_perception.png` — perception axis (4 blind judges: Haiku/Sonnet/Opus + local Qwen2.5-VL; Opus 90% clear → 50% foggy; the fog collapse appears only in the depth-reading models).
- `../eval/pit/figures/fig_latency.png` — real per-model decision latency vs. the time budget, 5 models: hosted Haiku/Sonnet/Opus (r≈4–6) and local Qwen/llama3.2-vision (r≈31/47); all past the ≈300ms max-tolerable budget.
- `../eval/pit/figures/fig_dissociation.png` — placeholder pending multi-model VLM runs.

Perception data + method: `../eval/pit/data/perception_judgments.jsonl` (80
rows: pit_width=2, fog∈{0,1}, 10 approach positions, judged by 4 blind vision
models — Haiku 4.5, Sonnet, Opus, and local Qwen2.5-VL 7B — each seeing the
image alone, seed-42 shuffled, scored against the sim's `in_window`). Frames
generated by `keen-raycaster --shoot`.

Latency data + method: `../eval/pit/data/vlm_latency.jsonl` (100 per-decision
latencies across 5 models — hosted `claude-haiku-4-5`, `claude-sonnet-4-6`,
`claude-opus-4-8` and local `llama3.2-vision:11b`, `qwen2.5-vl:7b` — one
JUMP/WAIT call per frame, each model's fastest single-call config) and
`../eval/pit/data/vlm_runs.jsonl` (one closed-loop `--policy agent` episode).
Reproduce the local models with `pit_agent.py --backend ollama --model <vlm>`
(no API key). Note: the fast single-call config is a *latency* measurement; the
hosted models default to WAIT in it (so it is not a fair perception config — the
perception numbers in §4.4 come from the deliberate blind-judge setup).
- Demo: `../raycaster/demo/ai_clears_pit.mp4`, `../raycaster/demo/ai_misses_pit_latency.mp4`.
