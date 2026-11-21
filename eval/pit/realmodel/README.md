# S5: a real model through the boundary

Data and driver for the paper's section 4.6 — claude-sonnet-4-6 judging every
frame of closed-loop blocking episodes, swept through the `r ≈ 1` boundary by
injected delay. 72 episodes (12 delays × 6 seeds, pit width 2), 6,540 judged
frames (6,612 vision calls; see the note on `n_calls` below). Result: in-window decisions clear 32/32 through `r = 0.97` and 0/26
from `r = 1.10`; logistic `r* = 1.05`; 9 early (out-of-window) decisions of
which 5 are delay-rescued.

| File | What it is |
|---|---|
| `sonnet_boundary_sweep.jsonl` | The 72 episode rows (same schema as `../data/vlm_runs.jsonl`, plus `n_calls`/`wall_s`). |
| `boundary_sweep.py` | Sweep driver: parallel episodes, resumable (skips (delay, seed) pairs already in the output). |
| `sweep_agent.py` | The revised S5 judge (see its docstring for exactly how it differs from the shipped `raycaster/pit_agent.py`). |
| `battery.py` | Frame-battery harness used to validate judge prompts against every approach frame. |
| `battery_v4_high.json` | Validation of the final judge: 108 frames × 3 samples. |
| `sonnet_boundary_sweep_fog1.jsonl` | The same 72-episode grid at fog 1 (`boundary_sweep.py --fog 1.0`): commit point spreads over five ticks, 35/72 early, in-window r* = 1.01, 22 early commits delay-rescued, 41/72 cleared. |
| `battery_fog.py` | Renders the 108 tick-grid approach frames at a chosen fog and runs the S5 judge on them; the gate before a fog sweep. |
| `battery_v4_high_fog1.json` | The final judge at fog 1: 108 frames × 3 samples (far-field false JUMP at 2.85, early JUMPs 8.38–8.87, two non-unanimous in-window frames). |
| `fig_realmodel.py` | Regenerates `../figures/fig_realmodel.png` from the JSONL, `fig_realmodel_fog.png` (clear vs fog) when the fog sweep exists, and `fig_realmodel_haiku.png` (Sonnet vs Haiku) when the Haiku sweep exists. |
| `haiku_boundary_sweep.jsonl` | The same 72-episode grid with Haiku 4.5 as the judge (`boundary_sweep.py --model claude-haiku-4-5-20251001`): commits at 9.08 (57/72), in-window r* = 0.96, 10 far-field early commits, 26/72. |
| `battery_v4_high_fog0_haiku.json` | Haiku 4.5 on the 108 frames × 3 samples (the gate before its sweep): far-field false JUMPs at 5.02–8.66, first in-window frame 0/3. |
| `sonnet_aim_sweep.jsonl` | The judge on the second scene (`aim_sweep.py`): the rendered aim task (turn in place, FIRE while the center line is over the target tile; `raycaster --task aim --policy agent`), 12 delays × 6 seeds: 72/72 commits in-window on ticks 45–49 (mode 49), 37/72 hits, 30/30 through D = 6, 5/6 and 2/6 at D = 7 and 8, 0/30 from D = 9, r* = 0.65. Same row schema (speed = omega, w_lo/w_hi = angular window, decision_x = aim angle) plus `call_log` of [aim angle, latency_s, action]. |
| `aim_sweep.py` | Driver for the aim-scene sweep (`--pilot`, `--effort`, `--delays`, `--target-width N` for a 2- or 3-tile target with its own delay grid, written to `sonnet_aim_sweep_wN.jsonl`). |
| `aim_battery.py` | Renders the 68 frames of the aim sweep (`--aim-frames`) and runs the aim judge on them; the gate before an aim sweep. `--ticks LO-HI --samples N` for a dense battery on the frames around the window; `--target-width N` for the wider scenes. |
| `battery_aim_high.json` | The aim judge (prompt v2, the both-sides rule) × 3 samples: unanimous on ticks 48-54, late on 45-47, one pre-window FIRE frame. `battery_aim_v1_high.json` is the first prompt ("touching"), which fired 8 ticks early. |
| `fig_aim_real.py` | Regenerates `../figures/fig_aim_real.png` from the aim sweep: outcome vs r, commit histogram, and per-call latency by frame (the stall on ticks 43–47). |
| `predict_from_battery.py` | No API. Predicts each injected-delay sweep (modal commit, early commits, clears per delay) from its judge's offline battery and the commit rule via `analyze.commit_first_passage`, and prints it against the sweep. |
| `sonnet_clock_sweep.jsonl` | Clock-rate sweep (`clock_sweep.py`): the same judge in an emulated real-time loop, 12 clock periods × 6 seeds, each call's measured latency replayed as its own delay. 31/72 cleared; flip at r = 1/3; deliberation stall on x ∈ [7.9, 8.4]. Rows carry `clock_s`, `effort`, `judge_x`, `delay_ticks_median`, and a per-call `call_log` of [px, latency_s, delay_ticks, raw answer, committed action, retries]. |
| `sonnet_clock_sweep_low.jsonl` | The same clock-rate sweep at low effort (`clock_sweep.py --effort low`): no stall (1.0 s flat) but 54% false JUMP on x ∈ [7.5, 8.92), 58/72 early commits, 8/72 cleared. |
| `clock_sweep.py` | Driver for the clock-rate sweep (`--effort`, `--clocks`, `--backend rules --latency` for a free self-test, `--task aim` for the aim scene with its own clock grid, written to `sonnet_aim_clock_sweep.jsonl`). |
| `fig_clock.py` | Regenerates `../figures/fig_clock.png` from the clock sweep(s). |
| `fig_firstpassage.py` | Regenerates `../figures/fig_firstpassage.png`: the final judge's per-frame JUMP rate from `battery_v4_high.json`, and the first-passage probability of a per-frame false-JUMP rate over the 90-frame approach under single-JUMP vs. two-consecutive commit rules. |

## Reproduce

Needs the sim built (`cmake --build raycaster/build` from the repo root),
Python with `anthropic` + `pillow`, and `ANTHROPIC_API_KEY` set. From the
repo root:

```bash
python3 eval/pit/realmodel/boundary_sweep.py --pilot 2   # 2 episodes at D=0
python3 eval/pit/realmodel/boundary_sweep.py --workers 4 # full 72-episode grid
python3 eval/pit/realmodel/battery_fog.py --fog 1.0      # judge battery under fog (gate)
python3 eval/pit/realmodel/boundary_sweep.py --fog 1.0 --workers 4   # fog-1 grid
python3 eval/pit/realmodel/fig_realmodel.py              # refit + figures
```

A note on `n_calls` in the three committed boundary sweeps: when those ran,
the agent also answered the sim's end-of-episode message with a model call on
the frame it had already committed on. So each row's `n_calls` is its judged
frames (`decision_tick + 1`) plus one, and that extra latency sample is in
`L_mean_s`. The sweep totals are 6,540 / 6,597 / 6,468 judged frames (Sonnet,
Sonnet fog 1, Haiku) against 6,612 / 6,669 / 6,540 calls. No decision, outcome
or fit depends on it. `sweep_agent.py` no longer makes that call, so new runs
log judged frames only. The clock-rate sweeps were never affected.

`sonnet_aim_sweep.jsonl` straddles that change: the 36 rows at D ≤ 7 were
run in October 2026 and carry the extra call (`n_calls == decision_tick + 2`);
the 36 rows at D ≥ 8 were run on 2026-11-21 and do not
(`n_calls == decision_tick + 1`). 3,528 judged frames, 3,564 calls. The
October run also wrote 32 rows at D ≥ 8 while the API was failing: the agent
died or never answered, and the sim of that date recorded each as a miss with
`decision_tick: -1`. Those rows were removed and the episodes re-run. Two
kept rows (D = 7, seeds 5 and 6) include calls that stalled for hours in
that outage; their decisions are unaffected (blocking mode) but their
`L_mean_s` is not a latency measurement.

Practical notes: one episode ≈ 93 vision calls ≈ 5–12 minutes of wall-clock
(sim time is decoupled from wall-clock, so API latency only costs patience,
not correctness). The full grid is ~6,600 calls — roughly $30–60 of Sonnet
usage. More than ~4 parallel workers risks rate-limit throttling; the agent
retries with backoff, but a gentler worker count is faster in practice.
Episodes already in the output file are skipped on relaunch, so an
interrupted sweep resumes for free.

The S5 regression test (`eval/pit/tests/test_paper_numbers.py::
test_s5_realmodel_boundary_sweep`) pins every published number to
`sonnet_boundary_sweep.jsonl`.
