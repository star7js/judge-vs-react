"""S5 figure: a real model (Sonnet 4.6) swept through the r~1 boundary.

Reads sonnet_boundary_sweep.jsonl (closed-loop blocking episodes, injected
delay D swept at pit width 2), fits the same decreasing logistic as the
oracle figures (analyze.transition_fit) on the in-window-decision episodes,
and overlays the out-of-window (early) decisions as crosses -- including the
delay-rescued ones that clear past r ~ 1.

Run from the repo root with numpy/scipy/matplotlib available:
    python3 eval/pit/realmodel/fig_realmodel.py
Writes eval/pit/figures/fig_realmodel.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from eval.pit.analyze import transition_fit, _decreasing_logistic  # noqa: E402

DATA = Path(__file__).parent / "sonnet_boundary_sweep.jsonl"
OUT = REPO / "eval" / "pit" / "figures" / "fig_realmodel.png"


def main() -> None:
    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    W = rows[0]["w_hi"] - rows[0]["w_lo"]
    v = rows[0]["speed"]
    r = np.array([v * x["delay"] / W for x in rows])
    y = np.array([x["cleared"] for x in rows], dtype=float)
    inw = np.array([x["in_window"] == 1 for x in rows])

    fit = transition_fit(r[inw], y[inw])

    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(r[inw], y[inw] + rng.uniform(-0.02, 0.02, inw.sum()),
               s=22, alpha=0.6, color="C0",
               label=f"in-window decision (n={int(inw.sum())})")
    out = ~inw
    ax.scatter(r[out], y[out] + rng.uniform(-0.02, 0.02, out.sum()),
               s=40, alpha=0.9, color="C3", marker="x",
               label=f"early decision, out-of-window (n={int(out.sum())})")

    r_curve = np.linspace(0, max(r.max(), fit["r_star"] * 1.5, 2.0), 400)
    ax.plot(r_curve, _decreasing_logistic(r_curve, fit["r_star"], fit["slope"]),
            color="black", lw=2, label="logistic fit (in-window only)")
    ax.axvline(fit["r_star"], color="black", ls="--", lw=1,
               label=f"r* = {fit['r_star']:.2f}")
    ax.axvline(1.17, color="gray", ls=":", lw=1.5,
               label="oracle r* = 1.17 (S1)")

    ax.set_xlabel("r = speed * delay_ticks / W")
    ax.set_ylabel("cleared")
    ax.set_title("S5: Sonnet 4.6 swept through the boundary (closed loop)")
    ax.legend(loc="center left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    n_resc = int(sum(1 for x in rows if x["in_window"] == 0 and x["cleared"] == 1))
    print(f"[fig_realmodel] n={len(rows)} r_star={fit['r_star']:.4f} "
          f"slope={fit['slope']:.4f} early={int(out.sum())} rescued={n_resc} "
          f"-> {OUT}")


if __name__ == "__main__":
    main()
