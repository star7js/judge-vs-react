"""Oracle sweep driver: runs the C sim for a grid of configs and writes
one JSONL row per episode, merging the config with the sim's JSON result.

`run_sweep` is the reusable core (used by tests with a small grid);
`main()` runs the full default sweep and writes it to the repo's
`eval/pit/data/oracle_sweep.jsonl`, printing progress along the way.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

from eval.pit.config import EpisodeConfig, ROW_KEYS, grid

DEFAULT_OUT = "eval/pit/data/oracle_sweep.jsonl"
PROGRESS_INTERVAL = 200


def _config_label(cfg: EpisodeConfig) -> str:
    return " ".join(cfg.cli_args())


def _run_episode(binary: str, cfg: EpisodeConfig, env: dict) -> dict:
    """Run one config through the C sim binary and return the merged row.

    Raises RuntimeError, naming the failing config, on a nonzero exit,
    missing/unparseable stdout, or a result missing expected ROW_KEYS.
    """
    proc = subprocess.run(
        [binary, *cfg.cli_args()], env=env, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"keen-raycaster exited with code {proc.returncode} for config "
            f"({_config_label(cfg)}).\nstderr:\n{proc.stderr}"
        )

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(
            f"keen-raycaster produced no stdout output for config "
            f"({_config_label(cfg)}).\nstderr:\n{proc.stderr}"
        )
    last_line = lines[-1]
    try:
        result = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"could not parse JSON from keen-raycaster output for config "
            f"({_config_label(cfg)}): {exc}\nline was: {last_line!r}"
        ) from exc

    missing = set(ROW_KEYS) - set(result.keys())
    if missing:
        raise RuntimeError(
            f"keen-raycaster output missing expected keys {sorted(missing)} "
            f"for config ({_config_label(cfg)}): {result!r}"
        )

    return {**asdict(cfg), **result}


def run_sweep(binary: str, configs: list[EpisodeConfig], out: str) -> int:
    """Run each config through the C sim binary and write JSONL rows to `out`.

    For each config: invokes `binary` with `cfg.cli_args()` (SDL_VIDEODRIVER
    forced to "dummy" so the sim runs headless), parses the last non-empty
    stdout line as JSON, validates its keys are a superset of ROW_KEYS,
    merges `asdict(cfg)` with the parsed result, and appends one JSONL line
    to `out`. Returns the number of rows written.
    """
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy"}

    rows_written = 0
    with open(out_path, "w") as f:
        for cfg in configs:
            row = _run_episode(binary, cfg, env)
            f.write(json.dumps(row) + "\n")
            rows_written += 1

    return rows_written


def main() -> None:
    binary = os.environ.get("KEEN_BIN", "raycaster/build/keen-raycaster")
    configs = grid()
    out_path = Path(DEFAULT_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy"}

    total = len(configs)
    print(f"Running oracle sweep: {total} configs -> {out_path}")

    rows_written = 0
    with open(out_path, "w") as f:
        for i, cfg in enumerate(configs, start=1):
            row = _run_episode(binary, cfg, env)
            f.write(json.dumps(row) + "\n")
            rows_written += 1

            if i % PROGRESS_INTERVAL == 0 or i == total:
                print(f"  {i}/{total} rows written")

    print(f"Done: {rows_written} rows -> {out_path}")


if __name__ == "__main__":
    main()
