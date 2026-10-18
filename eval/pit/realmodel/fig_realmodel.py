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
from eval.pit.analyze import ratio, transition_fit, _decreasing_logistic  # noqa: E402

DATA = Path(__file__).parent / "sonnet_boundary_sweep.jsonl"
DATA_FOG1 = Path(__file__).parent / "sonnet_boundary_sweep_fog1.jsonl"
DATA_HAIKU = Path(__file__).parent / "haiku_boundary_sweep.jsonl"
OUT = REPO / "eval" / "pit" / "figures" / "fig_realmodel.png"
OUT_FOG = REPO / "eval" / "pit" / "figures" / "fig_realmodel_fog.png"
OUT_HAIKU = REPO / "eval" / "pit" / "figures" / "fig_realmodel_haiku.png"
W_LO, W_HI = 8.921, 9.999


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def sweep_summary(rows: list[dict]) -> dict:
    """Numbers the paper quotes for a sweep: in-window counts by r band,
    logistic r* on in-window decisions, early commits and how many of them
    the delay rescued, and the commit-position distribution."""
    inw = [x for x in rows if x["in_window"] == 1]
    out = [x for x in rows if x["in_window"] == 0]
    r_of = lambda x: ratio(x, source="delay")
    fit = transition_fit([r_of(x) for x in inw], [x["cleared"] for x in inw]) if len(set(x["cleared"] for x in inw)) > 1 else None
    from collections import Counter
    commits = Counter(round(x["decision_x"], 2) for x in rows)
    return {
        "n": len(rows), "n_in": len(inw), "n_out": len(out),
        "below_0_98": (sum(x["cleared"] for x in inw if r_of(x) <= 0.98), sum(1 for x in inw if r_of(x) <= 0.98)),
        "edge": (sum(x["cleared"] for x in inw if 1.0 < r_of(x) < 1.09), sum(1 for x in inw if 1.0 < r_of(x) < 1.09)),
        "above_1_09": (sum(x["cleared"] for x in inw if r_of(x) >= 1.09), sum(1 for x in inw if r_of(x) >= 1.09)),
        "r_star": fit["r_star"] if fit else None,
        "early_cleared": sum(x["cleared"] for x in out),
        "commits": dict(sorted(commits.items())),
        "cleared_total": sum(x["cleared"] for x in rows),
    }


def fig_fog_comparison(rows0: list[dict], rows1: list[dict], out_path: Path) -> dict:
    """Fog 0 vs fog 1 sweeps side by side (see fig_two_sweeps)."""
    return fig_two_sweeps(rows0, rows1, out_path, names=("fog 0", "fog 1"),
                          title="Same judge, clear vs fog", tag="fig_realmodel_fog")


def fig_two_sweeps(rows0: list[dict], rows1: list[dict], out_path: Path, *,
                   names: tuple[str, str], title: str, tag: str,
                   hist_lo: float = 8.0, hist_hi: float = 9.1) -> dict:
    """Two 72-episode sweeps side by side. Left: P(clear) vs r for in-window
    decisions with each fit, early (out-of-window) decisions as crosses.
    Right: commit-position histograms against the window."""
    r_of = lambda x: ratio(x, source="delay")
    s0, s1 = sweep_summary(rows0), sweep_summary(rows1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    rng = np.random.default_rng(3)
    for rows, s, color, name in ((rows0, s0, "C0", names[0]), (rows1, s1, "C3", names[1])):
        inw = [x for x in rows if x["in_window"] == 1]
        out = [x for x in rows if x["in_window"] == 0]
        ax1.scatter([r_of(x) for x in inw], [x["cleared"] + rng.uniform(-0.025, 0.025) for x in inw],
                    s=20, alpha=0.6, color=color, label=f"{name}: in-window decision (n={len(inw)})")
        ax1.scatter([r_of(x) for x in out], [x["cleared"] + rng.uniform(-0.025, 0.025) for x in out],
                    s=36, alpha=0.9, color=color, marker="x", label=f"{name}: early decision (n={len(out)})")
        if s["r_star"] is not None:
            fit = transition_fit([r_of(x) for x in inw], [x["cleared"] for x in inw])
            rc = np.linspace(0, 2.1, 400)
            ax1.plot(rc, _decreasing_logistic(rc, fit["r_star"], fit["slope"]), color=color, lw=1.5,
                     label=f"{name}: logistic fit, r* = {fit['r_star']:.2f}")
    ax1.axvline(1.04, color="gray", ls=":", lw=1.2, label="oracle r* = 1.04")
    ax1.set_xlabel("r = speed * delay_ticks / W")
    ax1.set_ylabel("cleared")
    ax1.set_title(f"{title}: outcome vs r")
    ax1.legend(fontsize=6.5, loc="center left")

    bins = np.arange(hist_lo, hist_hi, 0.07) - 0.035
    ax2.axvspan(W_LO, W_HI, color="#1b9e77", alpha=0.15, lw=0, label="action window")
    for rows, color, name in ((rows0, "C0", names[0]), (rows1, "C3", names[1])):
        xs = [x["decision_x"] for x in rows]
        ax2.hist(xs, bins=bins, color=color, alpha=0.55, label=f"{name} commit position (n={len(xs)})")
    ax2.set_xlim(hist_lo, hist_hi)
    ax2.set_xlabel("commit position x (tiles); window opens at 8.92")
    ax2.set_ylabel("episodes")
    ax2.set_title(f"Commit-point spread: {names[0]} vs {names[1]}")
    ax2.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{tag}] {names[0]}: {s0}\n[{tag}] {names[1]}: {s1} -> {out_path}")
    return {names[0]: s0, names[1]: s1}


def main() -> None:
    rows = _load(DATA)
    # These rows carry both the injected delay and Sonnet's measured API
    # latency; the sweep's r axis is the injected delay.
    r = np.array([ratio(x, source="delay") for x in rows])
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
    ax.axvline(1.04, color="gray", ls=":", lw=1.5,
               label="oracle r* = 1.04 (1-tick calibration grid)")

    ax.set_xlabel("r = speed * delay_ticks / W")
    ax.set_ylabel("cleared")
    ax.set_title("Sonnet 4.6 swept through the boundary (closed loop)")
    ax.legend(loc="center left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    n_resc = int(sum(1 for x in rows if x["in_window"] == 0 and x["cleared"] == 1))
    print(f"[fig_realmodel] n={len(rows)} r_star={fit['r_star']:.4f} "
          f"slope={fit['slope']:.4f} early={int(out.sum())} rescued={n_resc} "
          f"-> {OUT}")
    if DATA_FOG1.exists():
        fig_fog_comparison(rows, _load(DATA_FOG1), OUT_FOG)
    else:
        print(f"[fig_realmodel_fog] skipped -- no {DATA_FOG1.name} (run boundary_sweep.py --fog 1.0)")
    if DATA_HAIKU.exists():
        fig_two_sweeps(rows, _load(DATA_HAIKU), OUT_HAIKU, names=("Sonnet 4.6", "Haiku 4.5"),
                       title="Two judges, clear weather", tag="fig_realmodel_haiku", hist_lo=4.9, hist_hi=9.3)
    else:
        print(f"[fig_realmodel_haiku] skipped -- no {DATA_HAIKU.name} (run boundary_sweep.py --model claude-haiku-4-5-20251001)")


if __name__ == "__main__":
    main()
