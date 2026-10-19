"""The judge on the second scene: Sonnet 4.6 swept through the aim task's
boundary (aim_sweep.py -> sonnet_aim_sweep.jsonl).

Left: outcome vs r = omega * D / (2h) for in-window commits with the
decreasing-logistic fit, early commits as crosses, the pit judge's r* for
comparison, and the cliff the commit offset predicts: with the judge's modal
commit x_c, r_pred = (w_hi - x_c) / (2h). Right: commit position in ticks
from the window's near edge, against the window (0 .. 2h/omega ticks).

    python3 eval/pit/realmodel/fig_aim_real.py [rows.jsonl]
Writes eval/pit/figures/fig_aim_real.png.
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from eval.pit.analyze import ratio, transition_fit, _decreasing_logistic  # noqa: E402

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "sonnet_aim_sweep.jsonl"
OUT = REPO / "eval" / "pit" / "figures" / "fig_aim_real.png"
PIT_RSTAR = 1.05


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def aim_summary(rows: list[dict]) -> dict:
    """Numbers the paper quotes: window in ticks, commit offsets (ticks from
    the near edge), modal commit, predicted cliff from the commit offset,
    in-window clears by delay, logistic r*, early commits and rescues."""
    w_lo, w_hi, omega = rows[0]["w_lo"], rows[0]["w_hi"], rows[0]["speed"]
    width_ticks = (w_hi - w_lo) / omega
    offs = [round((r["decision_x"] - w_lo) / omega) for r in rows]
    inw = [r for r in rows if r["in_window"] == 1]
    out = [r for r in rows if r["in_window"] == 0]
    r_of = lambda x: ratio(x, source="delay")
    fit = transition_fit([r_of(x) for x in inw], [x["cleared"] for x in inw]) if len({x["cleared"] for x in inw}) > 1 else None
    mode = Counter(offs).most_common(1)[0][0]
    by_delay = {}
    for r in inw:
        c = by_delay.setdefault(r["delay"], [0, 0])
        c[0] += r["cleared"]; c[1] += 1
    return {
        "n": len(rows), "n_in": len(inw), "n_out": len(out), "cleared_total": sum(r["cleared"] for r in rows),
        "width_ticks": width_ticks, "commit_offsets": dict(sorted(Counter(offs).items())),
        "commit_mode_ticks": mode, "r_pred_from_mode": (width_ticks - mode) / width_ticks,
        "r_star": fit["r_star"] if fit else None, "slope": fit["slope"] if fit else None,
        "in_window_by_delay": dict(sorted(by_delay.items())),
        "early_cleared": sum(r["cleared"] for r in out),
        "rule_holds": all(r["cleared"] == (w_lo <= r["decision_x"] + omega * r["delay"] <= w_hi) for r in rows),
        "L_mean_median": statistics.median(r["L_mean_s"] for r in rows),
        "calls": sum(r["n_calls"] for r in rows),
    }


def main() -> None:
    rows = _load(DATA)
    s = aim_summary(rows)
    w_lo, w_hi, omega = rows[0]["w_lo"], rows[0]["w_hi"], rows[0]["speed"]
    r_of = lambda x: ratio(x, source="delay")
    inw = [r for r in rows if r["in_window"] == 1]
    out = [r for r in rows if r["in_window"] == 0]
    rng = np.random.default_rng(11)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax1.scatter([r_of(x) for x in inw], [x["cleared"] + rng.uniform(-0.025, 0.025) for x in inw],
                s=22, alpha=0.65, color="C2", label=f"in-window commit (n = {len(inw)})")
    ax1.scatter([r_of(x) for x in out], [x["cleared"] + rng.uniform(-0.025, 0.025) for x in out],
                s=40, alpha=0.9, color="C3", marker="x", label=f"early commit (n = {len(out)})")
    if s["r_star"] is not None:
        rc = np.linspace(0, 1.4, 400)
        ax1.plot(rc, _decreasing_logistic(rc, s["r_star"], s["slope"]), color="C2", lw=2,
                 label=f"logistic fit, r* = {s['r_star']:.2f}")
    ax1.axvline(s["r_pred_from_mode"], color="black", ls="--", lw=1,
                label=f"predicted from modal commit: r = {s['r_pred_from_mode']:.2f}")
    ax1.axvline(PIT_RSTAR, color="gray", ls=":", lw=1.5, label=f"same judge on the pit: r* = {PIT_RSTAR:.2f}")
    ax1.set_xlabel("r = omega * delay_ticks / (2h)")
    ax1.set_ylabel("hit")
    ax1.set_title("Sonnet 4.6 on the aim scene: outcome vs r")
    ax1.legend(fontsize=7, loc="center left")

    offs = [(r["decision_x"] - w_lo) / omega for r in rows]
    bins = np.arange(-4, s["width_ticks"] + 4, 1.0) - 0.5
    ax2.axvspan(0, s["width_ticks"], color="#1b9e77", alpha=0.15, lw=0, label="target window (ticks)")
    ax2.hist(offs, bins=bins, color="C2", alpha=0.7, label=f"commit position (n = {len(rows)})")
    ax2.set_xlabel("commit position, ticks from the window's near edge")
    ax2.set_ylabel("episodes")
    ax2.set_title("Commit point on the aim scene")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"[fig_aim_real] {s} -> {OUT}")


if __name__ == "__main__":
    main()
