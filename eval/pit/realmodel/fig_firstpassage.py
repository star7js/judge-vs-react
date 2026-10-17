"""First-passage figure: why a frame-by-frame judge needs a commit rule.

Left panel: the final S5 judge's per-frame JUMP rate on the validation
battery (battery_v4_high.json: 108 approach frames x 3 samples), against the
true window. Right panel: the probability that a per-frame false-JUMP rate p
fires at least once somewhere over the N approach frames before the window
-- for a single-JUMP commit rule and for the two-consecutive-JUMP rule the S5
judge uses -- assuming independent per-frame errors. A rate that is invisible
in a single-frame accuracy battery (1-2%) fires almost surely over 90 frames
unless the commit rule suppresses it.

Run from the repo root with numpy/matplotlib available:
    python3 eval/pit/realmodel/fig_firstpassage.py
Writes eval/pit/figures/fig_firstpassage.png.
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

BATTERY = Path(__file__).parent / "battery_v4_high.json"
OUT = REPO / "eval" / "pit" / "figures" / "fig_firstpassage.png"
W_LO, W_HI = 8.921, 9.999
CUE_ONSET_X = 9.01          # pit's black region first touches the frame's bottom edge
N_APPROACH = 90             # consulted frames from spawn (x = 2.57) to the window


def p_false_commit_single(p: float, n: int = N_APPROACH) -> float:
    """P(at least one JUMP in n independent frames at per-frame rate p)."""
    return 1.0 - (1.0 - p) ** n


def p_false_commit_two_consecutive(p: float, n: int = N_APPROACH) -> float:
    """P(at least one run of two consecutive JUMPs in n independent frames
    at per-frame rate p). Exact, by a two-state chain over 'last answer was
    WAIT' / 'last answer was JUMP' with absorption on the second JUMP."""
    no_commit_last_wait, no_commit_last_jump = 1.0, 0.0
    for _ in range(n):
        nw = (no_commit_last_wait + no_commit_last_jump) * (1.0 - p)
        nj = no_commit_last_wait * p
        no_commit_last_wait, no_commit_last_jump = nw, nj
    return 1.0 - (no_commit_last_wait + no_commit_last_jump)


def load_battery(path: Path = BATTERY) -> list[dict]:
    return json.loads(path.read_text())


def main() -> None:
    rows = load_battery()
    px = np.array([r["px"] for r in rows])
    rate = np.array([r["jump"] / r["n"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax1.axvspan(W_LO, W_HI, color="#1b9e77", alpha=0.15, lw=0, label="action window [w_lo, w_hi)")
    ax1.axvline(CUE_ONSET_X, color="gray", ls=":", lw=1.2,
                label=f"cue onset (black reaches bottom edge), x = {CUE_ONSET_X}")
    ax1.plot(px, rate, marker="o", ms=3.5, lw=1, color="#d95f02",
             label=f"P(JUMP) per frame, n = {rows[0]['n']} samples")
    ax1.set_xlabel("player x (tiles); pit near edge at x = 10")
    ax1.set_ylabel("P(JUMP) on the frame")
    ax1.set_ylim(-0.05, 1.08)
    ax1.set_title("Final S5 judge on every approach frame\n(108 frames x 3 samples)")
    ax1.legend(loc="center left", fontsize=7.5)

    ps = np.linspace(0, 0.05, 400)
    ax2.plot(ps * 100, [p_false_commit_single(p) for p in ps], color="#d95f02", lw=2,
             label="commit on a single JUMP")
    ax2.plot(ps * 100, [p_false_commit_two_consecutive(p) for p in ps], color="#1b9e77", lw=2,
             label="commit on two consecutive JUMPs (S5 rule)")
    for p in (0.01, 0.02):
        s, d = p_false_commit_single(p), p_false_commit_two_consecutive(p)
        ax2.plot([p * 100, p * 100], [s, d], color="gray", lw=0.8, ls=":")
        ax2.annotate(f"{s:.0%}", (p * 100, s), xytext=(4, 2), textcoords="offset points", fontsize=8, color="#d95f02")
        ax2.annotate(f"{d:.1%}", (p * 100, d), xytext=(4, -10), textcoords="offset points", fontsize=8, color="#1b9e77")
    ax2.set_xlabel("per-frame false-JUMP rate p (%)")
    ax2.set_ylabel(f"P(false commit somewhere in {N_APPROACH} approach frames)")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("A rate invisible per frame fires almost surely per episode\n(independent per-frame errors)")
    ax2.legend(loc="center right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    first_any = next(r["px"] for r in rows if r["jump"] > 0)
    first_unanimous = next(r["px"] for r in rows if r["jump"] == r["n"])
    print(f"[fig_firstpassage] frames={len(rows)} first any-JUMP x={first_any} "
          f"first unanimous x={first_unanimous}; "
          f"p=1%: single {p_false_commit_single(0.01):.3f} two-consecutive {p_false_commit_two_consecutive(0.01):.4f}; "
          f"p=2%: single {p_false_commit_single(0.02):.3f} two-consecutive {p_false_commit_two_consecutive(0.02):.4f} "
          f"-> {OUT}")


if __name__ == "__main__":
    main()
