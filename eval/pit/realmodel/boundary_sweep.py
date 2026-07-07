"""Real-model (Sonnet) sweep through the r~1 boundary — parallel version.

Blocking agent episodes at pit width 2 (W=1.078, v=0.07), sweeping injected
delay D so r = v*D/W spans ~0 to ~2. Uses the scratchpad's patched binary
and sweep_agent.py (env-configurable socket => parallel episodes; structured
two-line prompt at medium effort => competent judge). Judgment noise comes
from the model; the r axis from D.

Resumable: (delay, seed) pairs already in the output file are skipped.

Usage: python3 boundary_sweep.py [--pilot N] [--workers K] [--out PATH]
Requires ANTHROPIC_API_KEY.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# A high-effort episode takes 400-700 s of wall-clock (~93 vision calls);
# run_vlm reads this at import time and its 120 s default would kill every
# episode. Must be set before the import.
os.environ.setdefault("KEEN_VLM_TIMEOUT", "1800")

from eval.pit.config import EpisodeConfig
import eval.pit.run_vlm as rv

rv.PIT_AGENT = HERE / "sweep_agent.py"  # revised prompt/effort/socket + debounce
BINARY = str(REPO / "raycaster" / "build" / "keen-raycaster")
MODEL = "claude-sonnet-4-6"
# r = 0.07*D/1.078 = 0.0649*D. Dense near the oracle boundary (D~15-20).
DELAYS = [0, 4, 8, 12, 14, 15, 16, 17, 18, 20, 24, 31]
SEEDS = [1, 2, 3, 4, 5, 6]


def run_one(job):
    delay, seed = job
    wid = os.getpid()
    os.environ["KEEN_AGENT_SOCK"] = f"/tmp/keen_sweep_sock_{wid}"
    latency_log = f"/tmp/keen_sweep_lat_{wid}.jsonl"
    frame_out = f"/tmp/keen_sweep_frame_{wid}.bmp"
    cfg = EpisodeConfig(policy="agent", pit_near=10.0, pit_width=2, speed=0.07,
                        fog=0.0, seed=seed, delay=delay, repeat=1, model=MODEL)
    t0 = time.time()
    try:
        row = rv.run_vlm_episode(BINARY, cfg, backend="vision", model=MODEL,
                                 latency_log=latency_log, frame_out=frame_out)
    except RuntimeError as exc:
        return {"delay": delay, "seed": seed, "error": str(exc)[:300]}
    row["n_calls"] = (sum(1 for _ in open(latency_log))
                      if os.path.exists(latency_log) else 0)
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="run N pilot episodes at D=0")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(HERE / "sonnet_boundary_sweep.jsonl"))
    a = ap.parse_args()

    out_path = Path(a.out)
    if a.pilot:
        grid = [(0, s) for s in range(1, a.pilot + 1)]
    else:
        grid = [(d, s) for d in DELAYS for s in SEEDS]

    done = set()
    if out_path.exists():
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    done.add((r["delay"], r["seed"]))
    todo = [g for g in grid if g not in done]
    print(f"[sweep] {len(todo)} episodes to run ({len(done)} done)", flush=True)

    n_ok = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(run_one, todo):
            if "error" in row:
                print(f"[sweep] FAILED delay={row['delay']} seed={row['seed']}: "
                      f"{row['error']}", flush=True)
                continue
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            n_ok += 1
            print(f"[sweep] delay={row['delay']} seed={row['seed']} "
                  f"cleared={row['cleared']} in_window={row['in_window']} "
                  f"decision_x={row['decision_x']} calls={row['n_calls']} "
                  f"L_mean={row['L_mean_s']:.2f}s wall={row['wall_s']}s", flush=True)
    print(f"[sweep] finished: {n_ok}/{len(todo)} ok", flush=True)


if __name__ == "__main__":
    main()
