"""What each judge's offline frame battery predicts for its closed-loop sweep.

A battery asks the judge about every approach frame a few times, with no sim
in the loop. From its per-frame JUMP/FIRE rates and the commit rule alone,
analyze.commit_first_passage gives a commit-tick distribution (assuming
independent answers across frames), and the landing rule turns that into
clears per delay. This script puts that prediction next to what the
72-episode sweep did, for the four injected-delay sweeps.

    python3 eval/pit/realmodel/predict_from_battery.py

No API calls: it reads the committed batteries and sweeps.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from eval.pit.analyze import battery_prediction  # noqa: E402

# Pit: frame k of the 108-frame battery is the frame the judge sees on tick k
# (x = 2.57 + 0.07 k); a commit on tick t at x lands at x + 0.07 D and clears
# iff that is in [w_lo, w_hi + same-tick tolerance] = [8.921, 10.065].
PIT_X = lambda t: 2.5 + 0.07 * (t + 1)
PIT_HIT = lambda t, d: 8.921 <= PIT_X(t) + 0.07 * d <= 10.065
PIT_IN = lambda t: 8.921 <= PIT_X(t) <= 9.999
# Aim scene: ticks 45..55 are inside the window; a commit on tick t hits iff
# 45 <= t + D <= 55.
AIM_HIT = lambda t, d: 45 <= t + d <= 55
AIM_IN = lambda t: 45 <= t <= 55

CASES = [
    ("Sonnet 4.6, pit, clear", "battery_v4_high.json", "sonnet_boundary_sweep.jsonl", PIT_HIT, PIT_IN),
    ("Sonnet 4.6, pit, fog 1", "battery_v4_high_fog1.json", "sonnet_boundary_sweep_fog1.jsonl", PIT_HIT, PIT_IN),
    ("Haiku 4.5, pit, clear", "battery_v4_high_fog0_haiku.json", "haiku_boundary_sweep.jsonl", PIT_HIT, PIT_IN),
    ("Sonnet 4.6, aim scene", "battery_aim_high.json", "sonnet_aim_sweep.jsonl", AIM_HIT, AIM_IN),
]


def predictions() -> dict[str, dict]:
    out = {}
    for name, battery, sweep, hit, in_window in CASES:
        b = json.load(open(HERE / battery))
        rows = [json.loads(l) for l in open(HERE / sweep) if l.strip()]
        out[name] = battery_prediction(b, rows, hit, in_window)
    return out


def main() -> None:
    print(f"{'sweep':26s} {'modal commit tick':>18s} {'mean commit tick':>18s} {'early commits':>15s} {'cleared /72':>13s}")
    print(f"{'':26s} {'pred':>9s}{'obs':>9s} {'pred':>9s}{'obs':>9s} {'pred':>8s}{'obs':>7s} {'pred':>7s}{'obs':>6s}")
    for name, p in predictions().items():
        print(f"{name:26s} {p['pred_mode']:9d}{p['obs_mode']:9d} {p['pred_mean']:9.1f}{p['obs_mean']:9.1f} "
              f"{p['pred_early']:8.1f}{p['obs_early']:7d} {p['pred_cleared']:7.1f}{p['obs_cleared']:6d}")
        print("    by delay (pred/obs): " + "  ".join(
            f"D={d}: {p['pred_by_delay'][d]:.1f}/{p['obs_by_delay'][d]}" for d in p["pred_by_delay"]))


if __name__ == "__main__":
    main()
