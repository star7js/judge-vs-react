"""Real-model sweep through the boundary on the second scene (the aim task).

Same judge machinery as boundary_sweep.py (sweep_agent.py, two-consecutive
commit rule, injected delay D), on the rendered aim scene: the player turns
in place at omega = 0.01 rad/tick and must FIRE while the center line is
over the target tile, whose angular extent is the window (2h = 0.1176 rad,
11.8 ticks). r = omega * D / (2h). The sim emits the pit's row schema
(speed <- omega, w_lo/w_hi <- angular window, decision_x <- aim angle), so
analyze.ratio() applies unchanged.

Usage (repo root, ANTHROPIC_API_KEY set):
    python3 eval/pit/realmodel/aim_sweep.py --pilot 2
    python3 eval/pit/realmodel/aim_sweep.py --workers 4
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("KEEN_VLM_TIMEOUT", "1800")

from eval.pit.config import EpisodeConfig  # noqa: E402
import eval.pit.run_vlm as rv  # noqa: E402

rv.PIT_AGENT = HERE / "sweep_agent.py"
BINARY = str(REPO / "raycaster" / "build" / "keen-raycaster")
MODEL = "claude-sonnet-4-6"
# r = 0.01*D/0.1176 = 0.085*D. The battery puts the judge's commit 2-3 ticks
# into the 11.8-tick window, so its cliff should sit near D ~ 8 (r ~ 0.7);
# dense there, and extending past r = 1 for comparison with the pit.
DELAYS = [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]
SEEDS = [1, 2, 3, 4, 5, 6]


def run_one(job):
    delay, seed, model, effort = job
    wid = os.getpid()
    os.environ["KEEN_AGENT_SOCK"] = f"/tmp/keen_aim_sock_{wid}"
    latency_log = f"/tmp/keen_aim_lat_{wid}.jsonl"
    frame_out = f"/tmp/keen_aim_frame_{wid}.bmp"
    # The pit fields are ignored by --task aim; they keep EpisodeConfig happy.
    cfg = EpisodeConfig(policy="agent", pit_near=10.0, pit_width=2, speed=0.07,
                        fog=0.0, seed=seed, delay=delay, repeat=1, model=model)
    t0 = time.time()
    try:
        row = rv.run_vlm_episode(BINARY, cfg, backend="vision", model=model,
                                 latency_log=latency_log, frame_out=frame_out,
                                 agent_extra=["--task", "aim", "--effort", effort],
                                 extra_cli=["--task", "aim"])
    except RuntimeError as exc:
        return {"delay": delay, "seed": seed, "error": str(exc)[:300]}
    calls = [json.loads(l) for l in open(latency_log)] if os.path.exists(latency_log) else []
    row["n_calls"] = len(calls)
    row["n_retried_calls"] = sum(1 for c in calls if c.get("retries"))
    row["call_log"] = [[c["px"], c["latency_s"], c["action"]] for c in calls]   # px is the aim angle
    row["task"] = "aim"
    row["effort"] = effort
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="run N pilot episodes at D=0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--delays", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    delays = [int(x) for x in a.delays.split(",")] if a.delays else DELAYS
    short = a.model.split("-")[1]
    tag = "" if a.effort == "high" else f"_{a.effort}"
    out_path = Path(a.out) if a.out else HERE / f"{short}_aim_sweep{tag}.jsonl"
    grid = [(0, s, a.model, a.effort) for s in range(1, a.pilot + 1)] if a.pilot else \
           [(d, s, a.model, a.effort) for d in delays for s in SEEDS]
    done = set()
    if out_path.exists():
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    done.add((r["delay"], r["seed"]))
    todo = [g for g in grid if (g[0], g[1]) not in done]
    print(f"[aim] {len(todo)} episodes to run ({len(done)} done) -> {out_path}", flush=True)

    n_ok = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(run_one, todo):
            if "error" in row:
                print(f"[aim] FAILED delay={row['delay']} seed={row['seed']}: {row['error']}", flush=True)
                continue
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            n_ok += 1
            print(f"[aim] delay={row['delay']} seed={row['seed']} cleared={row['cleared']} "
                  f"in_window={row['in_window']} decision_x={row['decision_x']} calls={row['n_calls']} "
                  f"L_mean={row['L_mean_s']:.2f}s retried={row['n_retried_calls']} wall={row['wall_s']}s", flush=True)
    print(f"[aim] finished: {n_ok}/{len(todo)} ok", flush=True)


if __name__ == "__main__":
    main()
