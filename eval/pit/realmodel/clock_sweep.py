"""Clock-rate sweep: the S5 judge's OWN latency carried through the boundary.

Every other closed-loop result injects a constant delay D. Here D is not
injected: each episode runs sweep_agent.py with --clock-s TAU, which replays
each decision's measured wall-clock latency L as round(L / TAU) sim ticks of
delay and does not consult the judge again until it has answered, exactly
what a real-time loop running TAU seconds per tick would do. Sweeping TAU
from the sim's native 1/60 s (the real-time regime, r ~ 10-20) up to several
seconds per tick (approaching blocking mode, r -> 0) carries the same judge,
with its real latency, through r ~ 1 with no knob on the delay itself.

Rows: the sim's --episode row (delay = 0) plus model/backend/L_mean_s from
run_vlm, plus clock_s, n_calls, mean/median of the per-call emulated
delay_ticks, and wall_s. Resumable on (clock_s, seed).

Rules-policy self-test (no API, --latency gives a fixed fake latency):
    python3 eval/pit/realmodel/clock_sweep.py --backend rules --latency 0.3 --clocks 0.01,0.02,0.03
Real run (repo root, ANTHROPIC_API_KEY set):
    python3 eval/pit/realmodel/clock_sweep.py --workers 4
"""
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Slow clocks need up to ~93 calls per episode; the runner's default is 120 s.
os.environ.setdefault("KEEN_VLM_TIMEOUT", "1800")
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from eval.pit.config import EpisodeConfig  # noqa: E402
import eval.pit.run_vlm as rv  # noqa: E402

rv.PIT_AGENT = HERE / "sweep_agent.py"
BINARY = str(REPO / "raycaster" / "build" / "keen-raycaster")
MODEL = "claude-sonnet-4-6"
# Seconds of wall-clock per sim tick. 1/60 is the sim's native rate (the
# regime the latency table places models in); the window is W/v = 15.4 ticks
# wide, so with L ~ 3-5 s the single-call boundary r = 1 sits near TAU ~ 0.25 s
# and the two-consecutive commit rule pushes the real flip to roughly half
# that r (two decisions must land inside the window).
CLOCKS = [1 / 60, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 4.0]
SEEDS = [1, 2, 3, 4, 5, 6]


def run_one(job):
    clock_s, seed, backend, model, fixed_latency, effort = job
    wid = os.getpid()
    os.environ["KEEN_AGENT_SOCK"] = f"/tmp/keen_clock_sock_{wid}"
    latency_log = f"/tmp/keen_clock_lat_{wid}.jsonl"
    frame_out = f"/tmp/keen_clock_frame_{wid}.bmp"
    cfg = EpisodeConfig(policy="agent", pit_near=10.0, pit_width=2, speed=0.07,
                        fog=0.0, seed=seed, delay=0, repeat=1, model=model)
    extra = ["--clock-s", repr(clock_s), "--effort", effort]
    if fixed_latency:
        extra += ["--latency", repr(fixed_latency)]
    t0 = time.time()
    try:
        row = rv.run_vlm_episode(BINARY, cfg, backend=backend, model=model,
                                 latency_log=latency_log, frame_out=frame_out,
                                 agent_extra=extra)
    except RuntimeError as exc:
        return {"clock_s": clock_s, "seed": seed, "error": str(exc)[:300]}
    entries = [json.loads(l) for l in open(latency_log)] if os.path.exists(latency_log) else []
    # The vision decider logs each call (raw JUMP/WAIT); realtime_step logs
    # the same call again with its emulated delay and the committed action.
    # Pair them by tick so each call appears once.
    raw_by_tick = {e["tick"]: e for e in entries if not e.get("emulated")}
    calls = [e for e in entries if e.get("emulated")]
    ns = [c["delay_ticks"] for c in calls]
    jumps = [c for c in calls if c["action"] == "jump"]
    row.update({
        "clock_s": clock_s, "effort": effort, "n_calls": len(calls),
        "n_retried_calls": sum(1 for c in calls if c.get("retries")),
        # frame the judge committed on (its takeoff is decision_x, n ticks later);
        # the first committed JUMP -- the loop keeps asking after it, so later
        # entries at the takeoff tick also say JUMP.
        "judge_x": jumps[0]["px"] if jumps else None,
        # [px, latency_s, delay_ticks, raw answer, committed action, retries]
        "call_log": [[c["px"], c["latency_s"], c["delay_ticks"],
                      raw_by_tick.get(c["tick"], {}).get("action"), c["action"], c.get("retries", 0)]
                     for c in calls],
        "delay_ticks_mean": round(statistics.mean(ns), 2) if ns else None,
        "delay_ticks_median": statistics.median(ns) if ns else None,
        "wall_s": round(time.time() - t0, 1),
    })
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--backend", default="vision", choices=["vision", "rules"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default="high", help="adaptive-thinking effort for the judge (S5: high)")
    ap.add_argument("--latency", type=float, default=0.0,
                    help="rules backend: fixed fake latency in seconds (self-test)")
    ap.add_argument("--clocks", default=None, help="comma-separated clock periods (s) override")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds override")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    clocks = [float(x) for x in a.clocks.split(",")] if a.clocks else CLOCKS
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else SEEDS
    short = a.model.split("-")[1] if a.backend == "vision" else "rules"
    tag = "" if a.effort == "high" else f"_{a.effort}"
    out_path = Path(a.out) if a.out else HERE / f"{short}_clock_sweep{tag}.jsonl"
    grid = [(c, s, a.backend, a.model if a.backend == "vision" else None, a.latency, a.effort)
            for c in clocks for s in seeds]
    done = set()
    if out_path.exists():
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    done.add((r["clock_s"], r["seed"]))
    todo = [g for g in grid if (g[0], g[1]) not in done]
    print(f"[clock] {len(todo)} episodes to run ({len(done)} done) -> {out_path}", flush=True)

    n_ok = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(run_one, todo):
            if "error" in row:
                print(f"[clock] FAILED clock={row['clock_s']} seed={row['seed']}: {row['error']}", flush=True)
                continue
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            n_ok += 1
            print(f"[clock] clock={row['clock_s']:.4f}s seed={row['seed']} cleared={row['cleared']} "
                  f"in_window={row['in_window']} decision_x={row['decision_x']} calls={row['n_calls']} "
                  f"n_ticks~{row['delay_ticks_median']} L_mean={row['L_mean_s']:.2f} retried={row['n_retried_calls']}", flush=True)
    print(f"[clock] finished: {n_ok}/{len(todo)} ok", flush=True)


if __name__ == "__main__":
    main()
