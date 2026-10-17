# Judge vs. React

**Dissociating perception from timing in VLM reflex control.**

A vision-language model playing a real-time game can fail two ways that
benchmark scores conflate: it can misjudge the scene (perception), or judge
correctly and act too late (timing). This repo is a minimal, deterministic
first-person apparatus that pulls the two apart — plus the data, analysis, and
preprint built on it.

![Same agent, same decision — paused it clears, real-time it falls](raycaster/demo/ai_judge_vs_react.gif)

Same agent, both panels. Left: the world pauses while it thinks — it clears
every time. Right: real time, the decision lands late — it walks into the pit.
The judgment never changed; only the clock did.

## Results

- **A real model through the boundary (headline):** a closed-loop Sonnet 4.6
  agent, judging every frame itself and swept through the transition by
  injected delay, clears 32/32 in-window decisions below `r ≈ 1` and 0/26
  above it (`r* = 1.05`), with nothing varied but the delay. Past the boundary
  its only clears are *delay-rescued misjudgments* (early jumps pushed into
  the window by the delay): the two failure axes compensate.
- **Perception axis:** blind single-frame judgment at zero delay scales with
  model capability (Opus 90%, Sonnet 80%) and collapses to chance under fog,
  only in the models that read depth.
- **Latency axis:** the window is open ~257 ms; hosted frontier VLMs decide in
  ~1–1.5 s (`r ≈ 4–6`), local open VLMs in 8–12 s (`r ≈ 31–47`). All past the
  boundary. Pausable/turn-based deployment converts the same models from
  failing to viable.
- **Calibration, not a law:** `r = v·D / W` is the coordinate in which "a
  leading-edge policy clears iff `r < 1`" is exact by construction. Oracle
  sweeps on the pit (`r* = 1.17`, bracketed between 1.04 and 1.30 by the delay
  grid; 21 informative configurations) and on a rotational aiming task
  (`r* = 0.98`) confirm the apparatus reproduces it.
- **Control:** action-repeat adds polling delay and shifts the transition to
  lower explicit `r`; it is a cost lever, not a latency fix.

The preprint is [`paper/judge-vs-react.pdf`](paper/judge-vs-react.pdf); an
accessible writeup is [`paper/blog.md`](paper/blog.md).

## Reproduce

The sim is self-contained C (SDL2 + libm); analysis is Python via
[uv](https://docs.astral.sh/uv/).

```bash
# build the deterministic sim
cmake -B raycaster/build -S raycaster && cmake --build raycaster/build

# regenerate every figure + fit from the committed data
KEEN_BIN=raycaster/build/keen-raycaster \
  uv run --with matplotlib --with numpy --with scipy python -m eval.pit.analyze

# tests (5 shell + 35 pytest, incl. a regression suite pinning every paper number to the data)
( cd raycaster && for t in tests/*.sh; do bash "$t"; done )
KEEN_BIN=raycaster/build/keen-raycaster \
  uv run --with pillow --with pytest --with numpy --with scipy --with matplotlib \
  python -m pytest eval/pit/tests/ -q
```

The local-VLM paths (Ollama) and the blind Claude judges run without an API
key; only the hosted-API latency numbers required one. Sweeps:
`eval/pit/run_oracle_sweep.py`, `run_aim_sweep.py`, `run_vlm.py`.

## Layout

| Path | What it is |
|---|---|
| `raycaster/` | Deterministic C/SDL2 first-person sim: pit-jump episode, `--task aim` (second task), injected control delay, headless modes. `pit_agent.py` drives it as an external agent (rules / Claude vision / Ollama backends). |
| `eval/pit/` | Analysis (`analyze.py`: all figures + fits), sweep runners, committed data (`data/*.jsonl`), figures, tests. |
| `paper/` | Preprint (`.tex`/`.pdf`), working draft (`.md`), blog post, arXiv bundle + `make-arxiv.sh`. |
| `docs/prior-art-fulltext.md` | Full-text prior-art check behind the related-work section. |

## Origin

This apparatus was extracted from a larger project exploring first-person
Commander Keen-style play with vision models. The pit-jump task started as
"the hard part is judging jump depth from a flat frame" and became the
measurement instrument here.
