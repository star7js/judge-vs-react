"""Clock-rate sweep figure: the judge's own latency carried through r ~ 1.

Reads sonnet_clock_sweep.jsonl (clock_sweep.py). Each episode ran the S5
judge in an emulated real-time loop at clock period TAU seconds per tick:
every decision's measured latency L became its own delay of round(L / TAU)
ticks, and the judge was not asked again until it answered. r for an episode
is v * n / W with n the median emulated delay over its calls.

Prediction. A leading-edge judge that is sampled every n ticks and commits
on its k-th consecutive in-window JUMP takes off at x1 + k*v*n, where x1,
its first in-window sample, lies in [w_lo, w_lo + v*n). It clears iff that
takeoff is <= x_hi, i.e. iff r <= 1/k (best sampling phase) down to
1/(k+1) (worst). The S5 judge has k = 2, so its real-time flip is predicted
in r = [1/3, 1/2], not at the injected-delay boundary r = 1: in real time
the window must hold the latency twice (the commit rule) plus the phase of
the next sample.

    python3 eval/pit/realmodel/fig_clock.py
Writes eval/pit/figures/fig_clock.png.
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from eval.pit.analyze import _wilson  # noqa: E402

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "sonnet_clock_sweep.jsonl"
DATA_LOW = Path(__file__).parent / "sonnet_clock_sweep_low.jsonl"   # same sweep, effort low
OUT = REPO / "eval" / "pit" / "figures" / "fig_clock.png"
V, W_LO, W_HI, X_HI = 0.07, 8.921, 9.999, 10.065
W = W_HI - W_LO
TICK_NATIVE = 1 / 60


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def r_of(row: dict) -> float:
    return V * row["delay_ticks_median"] / W


def clock_summary(rows: list[dict]) -> dict:
    """Per clock period: episodes, clears, median emulated delay (ticks),
    median r, median per-call latency, calls per episode, how many episodes
    committed (judge said JUMP twice) and how many of those were in-window
    at the judged frame, and retried calls."""
    out = {}
    for tau in sorted({r["clock_s"] for r in rows}):
        g = [r for r in rows if r["clock_s"] == tau]
        lat = [c[1] for r in g for c in r["call_log"]]
        committed = [r for r in g if r["judge_x"] is not None]
        out[tau] = {
            "n": len(g), "cleared": sum(r["cleared"] for r in g),
            "n_ticks_median": statistics.median(r["delay_ticks_median"] for r in g),
            "r_median": statistics.median(r_of(r) for r in g),
            "latency_median_s": statistics.median(lat) if lat else None,
            "calls_per_episode": statistics.mean(r["n_calls"] for r in g),
            "committed": len(committed),
            "committed_in_window": sum(W_LO <= r["judge_x"] <= W_HI for r in committed),
            "retried_calls": sum(r.get("n_retried_calls", 0) for r in g),
        }
    return out


def latency_profile(rows: list[dict], step: float = 0.25) -> dict:
    """Per-call latency binned by the frame position the judge was shown,
    pooled over every call in the sweep: median, p90, max, n per bin. This
    is where the real-time failure comes from -- the judge deliberates
    longest on the frames just before the window opens."""
    bins: dict[float, list[float]] = {}
    for r in rows:
        for c in r["call_log"]:
            bins.setdefault(round(c[0] / step) * step, []).append(c[1])
    out = {}
    for k in sorted(bins):
        L = sorted(bins[k])
        out[round(k, 2)] = {"n": len(L), "median": statistics.median(L),
                            "p90": L[int(0.9 * (len(L) - 1))], "max": L[-1]}
    return out


def main() -> None:
    rows = _load(DATA)
    summ = clock_summary(rows)
    taus = sorted(summ)
    low = _load(DATA_LOW) if DATA_LOW.exists() and len(sys.argv) == 1 else []
    summ_low = clock_summary(low) if low else {}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # Left: outcome vs r (log), each episode a point; predicted band and r = 1.
    rng = np.random.default_rng(5)
    rs = np.array([r_of(r) for r in rows])
    ys = np.array([r["cleared"] for r in rows], dtype=float)
    ax1.axvspan(1 / 3, 1 / 2, color="#d95f02", alpha=0.18, lw=0,
                label="predicted flip, k = 2 commit rule: r in [1/3, 1/2]")
    ax1.axvline(1.0, color="gray", ls=":", lw=1.5, label="injected-delay boundary r = 1")
    ax1.scatter(rs, ys + rng.uniform(-0.03, 0.03, len(rs)), s=26, alpha=0.75, color="C0",
                label=f"high effort, episode (n = {len(rows)}); r = v * median emulated delay / W")
    for tau in taus:
        s = summ[tau]
        ax1.plot([s["r_median"]], [s["cleared"] / s["n"]], marker="_", ms=14, mew=2, color="C0")
    if low:
        rl = np.array([r_of(r) for r in low]); yl = np.array([r["cleared"] for r in low], dtype=float)
        ax1.scatter(rl, yl + rng.uniform(-0.03, 0.03, len(rl)) - 0.08, s=26, alpha=0.75, color="C3", marker="D",
                    label=f"low effort, episode (n = {len(low)}), offset for visibility")
        for tau in sorted(summ_low):
            s = summ_low[tau]
            ax1.plot([s["r_median"]], [s["cleared"] / s["n"] - 0.08], marker="_", ms=14, mew=2, color="C3")
    ax1.set_xscale("log")
    ax1.set_xlabel("r from the judge's own latency at clock period tau (log scale)")
    ax1.set_ylabel("cleared   (bars: fraction per clock period)")
    ax1.set_title("Sonnet 4.6, own latency in the loop: outcome vs r")
    ax1.legend(fontsize=7, loc="center left")

    # Right: fraction cleared vs clock period with Wilson CIs, plus latency.
    xs = np.arange(len(taus))
    bw = 0.38 if low else 0.7
    for k, (sm, color, name) in enumerate([(summ, "C0", "high effort (the S5 judge)")] + ([(summ_low, "C3", "low effort")] if low else [])):
        fr = [sm[t]["cleared"] / sm[t]["n"] if t in sm else 0 for t in taus]
        ci = [_wilson(sm[t]["cleared"], sm[t]["n"]) if t in sm else (0, 0) for t in taus]
        ax2.bar(xs + (k - 0.5) * bw if low else xs, fr, bw, color=color, alpha=0.75, label=name,
                yerr=[[max(0.0, f - c[0]) for f, c in zip(fr, ci)], [max(0.0, c[1] - f) for f, c in zip(fr, ci)]],
                capsize=2, error_kw={"lw": 0.8})
    for x, t in zip(xs, taus):
        s = summ[t]
        ax2.text(x, 1.02, f"r={s['r_median']:.2f}", ha="center", fontsize=6)
    if low:
        ax2.legend(fontsize=7, loc="upper left")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([("1/60" if abs(t - TICK_NATIVE) < 1e-6 else f"{t:g}") for t in taus], fontsize=8)
    ax2.set_ylim(0, 1.2)
    ax2.set_xlabel("clock period tau (seconds of wall-clock per sim tick); 1/60 is the sim's native rate")
    ax2.set_ylabel("fraction cleared (6 seeds, Wilson 95%)")
    ax2.set_title("Fraction cleared per clock period")

    # Right: per-call latency vs the frame position shown, every call pooled.
    prof = latency_profile(rows)
    pxs = [c[0] for r in rows for c in r["call_log"]]
    lats = [c[1] for r in rows for c in r["call_log"]]
    ax3.axvspan(W_LO, W_HI, color="#1b9e77", alpha=0.15, lw=0, label="action window")
    ax3.scatter(pxs, lats, s=8, alpha=0.35, color="C0", label=f"high effort, one call (n = {len(lats)})")
    ks = sorted(prof)
    ax3.plot(ks, [prof[k]["median"] for k in ks], color="black", lw=1.8, label="high: median per 0.25-tile bin")
    ax3.plot(ks, [prof[k]["p90"] for k in ks], color="black", lw=1, ls="--", label="high: 90th percentile")
    if low:
        pl = latency_profile(low)
        kl = sorted(pl)
        ax3.plot(kl, [pl[k]["median"] for k in kl], color="C3", lw=1.8, label=f"low effort: median (n = {sum(len(r['call_log']) for r in low)} calls)")
        ax3.plot(kl, [pl[k]["p90"] for k in kl], color="C3", lw=1, ls="--", label="low: 90th percentile")
    ax3.set_yscale("log")
    ax3.set_xlabel("frame position x shown to the judge (tiles); pit edge at 10")
    ax3.set_ylabel("call latency (s, log)")
    ax3.set_title("Call latency vs the frame shown")
    ax3.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    for t in taus:
        s = summ[t]
        print(f"[fig_clock] tau={t:.4f} cleared={s['cleared']}/{s['n']} n~{s['n_ticks_median']:.0f} "
              f"r~{s['r_median']:.2f} L~{s['latency_median_s']:.1f}s calls/ep={s['calls_per_episode']:.1f} "
              f"committed={s['committed']} (in-window {s['committed_in_window']}) retried={s['retried_calls']}")
    if low:
        for t in sorted(summ_low):
            s = summ_low[t]
            print(f"[fig_clock] LOW tau={t:.4f} cleared={s['cleared']}/{s['n']} n~{s['n_ticks_median']:.0f} "
                  f"r~{s['r_median']:.2f} L~{s['latency_median_s']:.1f}s calls/ep={s['calls_per_episode']:.1f} "
                  f"committed={s['committed']} (in-window {s['committed_in_window']}) retried={s['retried_calls']}")
    hot = {k: v for k, v in prof.items() if 7.9 <= k <= 8.35}
    base = [v["median"] for k, v in prof.items() if k < 7.5]
    print(f"[fig_clock] latency: median far-field {statistics.median(base):.1f}s; "
          + "; ".join(f"x~{k}: median {v['median']:.1f}s p90 {v['p90']:.1f}s max {v['max']:.1f}s (n={v['n']})" for k, v in hot.items()))
    print(f"[fig_clock] -> {OUT}")


if __name__ == "__main__":
    main()
