"""The judge on the second scene: Sonnet 4.6 swept through the aim task's
boundary (aim_sweep.py -> sonnet_aim_sweep.jsonl).

Left: outcome vs r = omega * D / (2h) for in-window commits with the
decreasing-logistic fit, early commits as crosses, the pit judge's r* for
comparison, and the cliff the commit offset predicts: with the judge's modal
commit x_c, r_pred = (w_hi - x_c) / (2h). Middle: commit position in ticks
from the window's near edge, against the window (0 .. 2h/omega ticks).
Right: every call's latency against the tick of the frame it judged; the
sweep is blocking, so latency costs nothing here, but it shows where the
judge deliberates.

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
    # The modal commit's exact position (offs are rounded to whole ticks for
    # the histogram; the first in-window tick is 0.88 ticks past the edge).
    x_mode = Counter(r["decision_x"] for r in rows).most_common(1)[0][0]
    by_delay = {}
    for r in inw:
        c = by_delay.setdefault(r["delay"], [0, 0])
        c[0] += r["cleared"]; c[1] += 1
    return {
        "n": len(rows), "n_in": len(inw), "n_out": len(out), "cleared_total": sum(r["cleared"] for r in rows),
        "width_ticks": width_ticks, "commit_offsets": dict(sorted(Counter(offs).items())),
        "commit_mode_ticks": mode, "commit_mode_delta": (x_mode - w_lo) / (w_hi - w_lo),
        "r_pred_from_mode": (w_hi - x_mode) / (w_hi - w_lo),
        "commit_ticks": dict(sorted(Counter(r["decision_tick"] for r in rows).items())),
        "judged_frames": sum(r["decision_tick"] + 1 for r in rows),
        "r_star": fit["r_star"] if fit else None, "slope": fit["slope"] if fit else None,
        "in_window_by_delay": dict(sorted(by_delay.items())),
        "early_cleared": sum(r["cleared"] for r in out),
        "rule_holds": all(r["cleared"] == (w_lo <= r["decision_x"] + omega * r["delay"] <= w_hi) for r in rows),
        "L_mean_median": statistics.median(r["L_mean_s"] for r in rows),
        "calls": sum(r["n_calls"] for r in rows),
    }


# A call over the client's 60 s timeout went through at least one retry, so
# its wall-clock is backoff, not deliberation.
RETRY_S = 60.0


def call_latencies(rows: list[dict]) -> dict[int, list[float]]:
    """Per-tick latencies (s) of the judged frames: one per (episode, tick),
    dropping the duplicate end-of-episode call the October rows carry and
    calls that hit a retry."""
    a0 = min(c[0] for r in rows for c in r["call_log"])
    omega = rows[0]["speed"]
    by: dict[int, list[float]] = {}
    for r in rows:
        seen = set()
        for aim, lat, _ in r["call_log"]:
            tick = round((aim - a0) / omega)
            if tick in seen:
                continue
            seen.add(tick)
            if lat <= RETRY_S:
                by.setdefault(tick, []).append(lat)
    return by


def latency_profile(rows: list[dict], bands=((0, 29), (30, 37), (38, 42), (43, 47), (48, 49))) -> dict:
    """Median / p90 / max / fraction over 10 s of call latency per tick band,
    plus the share of calls and of total thinking time each band takes."""
    by = call_latencies(rows)
    total_t = sum(sum(v) for v in by.values())
    total_n = sum(len(v) for v in by.values())
    out = {}
    for lo, hi in bands:
        v = sorted(x for t in range(lo, hi + 1) for x in by.get(t, []))
        out[(lo, hi)] = {
            "n": len(v), "median": statistics.median(v), "p90": v[int(0.9 * (len(v) - 1))], "max": v[-1],
            "over_10s": sum(x > 10 for x in v) / len(v),
            "share_calls": len(v) / total_n, "share_time": sum(v) / total_t,
        }
    return out


def main() -> None:
    rows = _load(DATA)
    s = aim_summary(rows)
    w_lo, w_hi, omega = rows[0]["w_lo"], rows[0]["w_hi"], rows[0]["speed"]
    r_of = lambda x: ratio(x, source="delay")
    inw = [r for r in rows if r["in_window"] == 1]
    out = [r for r in rows if r["in_window"] == 0]
    rng = np.random.default_rng(11)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.6))

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

    by = call_latencies(rows)
    first_in = min(r["decision_tick"] for r in rows) - round((min(r["decision_x"] for r in rows) - w_lo) / omega) + 1
    ax3.axvspan(first_in - 0.5, first_in - 0.5 + s["width_ticks"], color="#1b9e77", alpha=0.15, lw=0, label="target window")
    xs = [t + rng.uniform(-0.3, 0.3) for t, v in by.items() for _ in v]
    ys = [x for v in by.values() for x in v]
    ax3.scatter(xs, ys, s=5, alpha=0.25, color="C0", label=f"one call (n = {len(ys)})")
    ticks = sorted(by)
    ax3.plot(ticks, [statistics.median(by[t]) for t in ticks], color="black", lw=1.6, label="median per tick")
    ax3.set_xlabel("tick of the frame judged")
    ax3.set_ylabel("call latency (s)")
    ax3.set_title("Where the judge deliberates")
    ax3.legend(fontsize=8, loc="upper left")
    prof = latency_profile(rows)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"[fig_aim_real] {s} -> {OUT}")
    print("[fig_aim_real] latency by tick band: " + "; ".join(
        f"{lo}-{hi}: median {p['median']:.1f}s p90 {p['p90']:.1f}s max {p['max']:.1f}s >10s {p['over_10s']:.0%} "
        f"({p['share_calls']:.1%} of calls, {p['share_time']:.1%} of time)" for (lo, hi), p in prof.items()))


if __name__ == "__main__":
    main()
