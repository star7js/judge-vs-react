# S5: a real model through the boundary

Data and driver for the paper's section 4.6 — claude-sonnet-4-6 judging every
frame of closed-loop blocking episodes, swept through the `r ≈ 1` boundary by
injected delay. 72 episodes (12 delays × 6 seeds, pit width 2), 6,612 vision
calls. Result: in-window decisions clear 32/32 through `r = 0.97` and 0/26
from `r = 1.10`; logistic `r* = 1.05`; 9 early (out-of-window) decisions of
which 5 are delay-rescued.

| File | What it is |
|---|---|
| `sonnet_boundary_sweep.jsonl` | The 72 episode rows (same schema as `../data/vlm_runs.jsonl`, plus `n_calls`/`wall_s`). |
| `boundary_sweep.py` | Sweep driver: parallel episodes, resumable (skips (delay, seed) pairs already in the output). |
| `sweep_agent.py` | The revised S5 judge (see its docstring for exactly how it differs from the shipped `raycaster/pit_agent.py`). |
| `battery.py` | Frame-battery harness used to validate judge prompts against every approach frame. |
| `battery_v4_high.json` | Validation of the final judge: 108 frames × 3 samples. |
| `fig_realmodel.py` | Regenerates `../figures/fig_realmodel.png` from the JSONL. |

## Reproduce

Needs the sim built (`cmake --build raycaster/build` from the repo root),
Python with `anthropic` + `pillow`, and `ANTHROPIC_API_KEY` set. From the
repo root:

```bash
python3 eval/pit/realmodel/boundary_sweep.py --pilot 2   # 2 episodes at D=0
python3 eval/pit/realmodel/boundary_sweep.py --workers 4 # full 72-episode grid
python3 eval/pit/realmodel/fig_realmodel.py              # refit + figure
```

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
