"""Aim-task oracle sweep: the generality test for the r~1 timing law.

Runs the C sim's `--episode --task aim` path over a grid of target angular
widths and control delays, writing one JSONL row per episode. The aim episode
emits the SAME schema as the pit (`speed`<-omega, `w_lo/w_hi`<-angular window,
`pit_width`<-width_id), so `analyze.ratio()` / `analyze.transition_fit()` reuse
unchanged: r = omega*delay / (2h). If the clear/fail outcomes collapse near
r~1 like the pit's, the timing law is not specific to translational jumps.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

DEFAULT_OUT = "eval/pit/data/aim_sweep.jsonl"

# Target angular half-widths (rad) and their grouping id (the pit_width analog).
HALF_WIDTHS = {1: 0.04, 2: 0.06, 3: 0.08, 4: 0.10, 5: 0.12}
OMEGA = 0.01          # angular sweep speed (rad/tick) — the "speed" analog
THETA = 1.0           # target center angle (rad)
A0 = 0.5              # aim start (rad); below every window's near edge
DELAYS = list(range(0, 41))   # r = omega*delay/(2h) spans ~0..5 across widths


def _run(binary: str, half_width: float, width_id: int, delay: int, env: dict) -> dict:
    args = [binary, "--episode", "--task", "aim",
            "--omega", repr(OMEGA), "--half-width", repr(half_width),
            "--theta", repr(THETA), "--a0", repr(A0),
            "--width-id", str(width_id), "--delay", str(delay)]
    proc = subprocess.run(args, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"keen-raycaster exited {proc.returncode} for "
            f"(h={half_width}, delay={delay}).\nstderr:\n{proc.stderr}"
        )
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    if not lines:
        raise RuntimeError(
            f"no output for (h={half_width}, delay={delay}).\nstderr:\n{proc.stderr}"
        )
    return json.loads(lines[-1])


def run_sweep(binary: str, out: str,
              half_widths: dict[int, float] | None = None,
              delays: list[int] | None = None) -> int:
    """Run the aim grid and write JSONL rows to `out`. Returns rows written."""
    half_widths = half_widths or HALF_WIDTHS
    delays = delays if delays is not None else DELAYS
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy"}

    n = 0
    with open(out_path, "w") as f:
        for width_id, h in half_widths.items():
            for d in delays:
                row = _run(binary, h, width_id, d, env)
                f.write(json.dumps(row) + "\n")
                n += 1
    return n


def main() -> None:
    binary = os.environ.get("KEEN_BIN", "raycaster/build/keen-raycaster")
    out = Path(DEFAULT_OUT)
    total = len(HALF_WIDTHS) * len(DELAYS)
    print(f"Running aim sweep: {total} configs -> {out}")
    n = run_sweep(binary, str(out))
    print(f"Done: {n} rows -> {out}")


if __name__ == "__main__":
    main()
