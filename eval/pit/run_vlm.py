"""VLM episode runner: drives the C `--episode --policy agent` parametric
path with `raycaster/pit_agent.py` as the external decision-maker, measuring
real per-call decision latency and collecting results as JSONL.

`run_vlm_episode` is the reusable core (used by tests with the no-API
`rules` backend, and by `main()` with the `vision` backend against a real
model); `mean_latency` reads a `--latency-log` JSONL file written by
`pit_agent.py` and returns the mean `latency_s` (NaN if missing/empty, since
the `rules` backend logs no latency at all).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from eval.pit.config import EpisodeConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
PIT_AGENT = REPO_ROOT / "raycaster" / "pit_agent.py"

DEFAULT_OUT = "eval/pit/data/vlm_runs.jsonl"
# A local Ollama vision model can take ~12s per decision, so a full blocking
# episode may need minutes. Override with KEEN_VLM_TIMEOUT (seconds).
GAME_TIMEOUT_S = float(os.environ.get("KEEN_VLM_TIMEOUT", "120"))


def mean_latency(path: str) -> float:
    """Return the mean `latency_s` across a `--latency-log` JSONL file.

    Returns `float("nan")` if the file is missing or has no rows (the
    `rules` backend makes no API calls, so it writes nothing).
    """
    try:
        with open(path) as f:
            lines = [line for line in f if line.strip()]
    except FileNotFoundError:
        return float("nan")
    if not lines:
        return float("nan")
    values = [json.loads(line)["latency_s"] for line in lines]
    return sum(values) / len(values)


def _reset_file(path: str) -> None:
    """Remove any stale file at `path` so a run starts from a clean slate."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def run_vlm_episode(
    binary: str,
    cfg: EpisodeConfig,
    *,
    backend: str = "vision",
    model: str | None = None,
    latency_log: str,
    frame_out: str = "/tmp/keen_ray_frame.bmp",
) -> dict:
    """Run one `--policy agent` episode driven by `pit_agent.py` and return
    the merged result row.

    Launches the C game (non-blocking) and `pit_agent.py` (non-blocking) as
    two subprocesses, waits for the game to exit (it prints its one JSON
    result line to stdout and terminates on episode end), parses that
    result, and merges it with `{"model", "backend", "L_mean_s"}`.

    Raises `RuntimeError` if `backend == "vision"` and `ANTHROPIC_API_KEY`
    is unset, or if the game does not exit within `GAME_TIMEOUT_S`. The
    `ollama` backend needs no key (local server); the `model` is passed
    through to `pit_agent.py` for both `vision` and `ollama`.
    """
    if backend == "vision" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "run_vlm_episode(backend='vision') requires the ANTHROPIC_API_KEY "
            "environment variable to be set."
        )

    _reset_file(latency_log)
    _reset_file(frame_out)

    env = {**os.environ, "SDL_VIDEODRIVER": "dummy"}
    game = subprocess.Popen(
        [binary, *cfg.cli_args(), "--frame-out", frame_out],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    agent_args = [
        sys.executable, str(PIT_AGENT),
        "--backend", backend,
        "--frame", frame_out,
        "--latency-log", latency_log,
    ]
    if backend in ("vision", "ollama"):
        agent_args += ["--model", model]
    agent = subprocess.Popen(
        agent_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    try:
        try:
            stdout, stderr = game.communicate(timeout=GAME_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            game.kill()
            agent.kill()
            game.communicate()
            agent.communicate()
            raise RuntimeError(
                f"keen-raycaster did not exit within {GAME_TIMEOUT_S}s "
                f"(config: {' '.join(cfg.cli_args())})"
            )

        if game.returncode != 0:
            raise RuntimeError(
                f"keen-raycaster exited with code {game.returncode} for config "
                f"({' '.join(cfg.cli_args())}).\nstderr:\n{stderr}"
            )

        lines = [line for line in stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(
                f"keen-raycaster produced no stdout output for config "
                f"({' '.join(cfg.cli_args())}).\nstderr:\n{stderr}"
            )
        try:
            result = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"could not parse JSON from keen-raycaster output for config "
                f"({' '.join(cfg.cli_args())}): {exc}\nline was: {lines[-1]!r}"
            ) from exc

        try:
            agent.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            agent.kill()
            agent.wait()

        return {
            **result,
            "model": model,
            "backend": backend,
            "L_mean_s": mean_latency(latency_log),
        }
    finally:
        for proc in (game, agent):
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def main() -> None:
    """Run a small (model, pit_width, seed) grid and append results to
    `eval/pit/data/vlm_runs.jsonl`. Each episode runs at delay=0 to measure
    the model's real decision latency L AND its delay-0 judgment; the timing
    figure places r(L) on the oracle transition (see analyze.fig_timing).

    Defaults to the Claude `vision` backend (needs ANTHROPIC_API_KEY). Pass
    `--backend ollama --model llama3.2-vision:11b` to drive a local Ollama
    vision model instead — no key required."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--backend", default="vision", choices=["vision", "ollama"])
    ap.add_argument("--model", action="append",
                    help="model id; repeatable (default depends on backend)")
    ap.add_argument("--seeds", default="1,2,3",
                    help="comma-separated seeds (default 1,2,3)")
    ap.add_argument("--pit-widths", default="2",
                    help="comma-separated pit widths (default 2)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    binary = os.environ.get("KEEN_BIN", "raycaster/build/keen-raycaster")
    models = a.model or (["claude-haiku-4-5", "claude-opus-4-8"]
                         if a.backend == "vision" else ["llama3.2-vision:11b"])
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    pit_widths = [int(w) for w in a.pit_widths.split(",") if w.strip()]

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a") as f:
        for model in models:
            for pw in pit_widths:
                for seed in seeds:
                    cfg = EpisodeConfig(
                        policy="agent", pit_near=10.0, pit_width=pw, speed=0.07,
                        fog=0.0, seed=seed, delay=0, repeat=1, model=model,
                    )
                    row = run_vlm_episode(
                        binary, cfg, backend=a.backend, model=model,
                        latency_log=f"/tmp/keen_vlm_latency_{os.getpid()}.jsonl",
                    )
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"  backend={a.backend} model={model} pit_width={pw} "
                          f"seed={seed} -> cleared={row.get('cleared')} "
                          f"L={row.get('L_mean_s')}")


if __name__ == "__main__":
    main()
