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

- **Same-decision flip:** the identical correct decision clears with
  probability 1.00 paused and 0.11 once delay pushes it past the window.
- **Timing law:** success collapses at a sharp threshold in
  `r = latency / action-window` near `r ≈ 1` (logistic fit `r* = 1.17`),
  with latency injected independently of judgment quality.
- **Action-repeat doesn't rescue it** (modestly worsens it).
- **Generality:** a second reflex task in a different modality (rotational
  aiming) lands on the same boundary (`r* = 0.98`).
- **Every real model measured is past the boundary:** the window is open
  ~257 ms; hosted frontier VLMs decide in ~1–1.5 s (`r ≈ 4–6`), local open
  VLMs in 8–12 s (`r ≈ 31–47`). Pausable/turn-based deployment converts the
  same models from failing to viable.

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

# tests (5 shell + 22 pytest)
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
