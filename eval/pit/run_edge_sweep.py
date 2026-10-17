"""Width-adapted oracle sweep: bracket the r~1 edge at every pit width.

The default oracle grid (`run_oracle_sweep`, D in {0, 4, ..., 24}) only
straddles the edge at pit width 2: width 1 (W = 2.08) never reaches r = 1
by D = 24, and width 3 (W = 0.08) is past r = 3 at the first nonzero delay.
This sweep runs every clearable width through a 1-tick delay grid, one seed
and fog 0 (the oracle ignores both), so the last clear and first failure
bracket the edge independently at each width. Writes
`eval/pit/data/oracle_edge_sweep.jsonl`.
"""
from __future__ import annotations

import os
from pathlib import Path

from eval.pit.config import grid
from eval.pit.run_oracle_sweep import run_sweep

DEFAULT_OUT = "eval/pit/data/oracle_edge_sweep.jsonl"
PIT_WIDTHS = [1, 2, 3]
DELAYS = list(range(0, 41))   # width 1 crosses r = 1 near D = 30


def main() -> None:
    binary = os.environ.get("KEEN_BIN", "raycaster/build/keen-raycaster")
    configs = grid(pit_width=PIT_WIDTHS, fog=[0.0], delay=DELAYS, repeat=[1], seed=[0])
    out = Path(DEFAULT_OUT)
    print(f"Running oracle edge sweep: {len(configs)} configs -> {out}")
    n = run_sweep(binary, configs, str(out))
    print(f"Done: {n} rows -> {out}")


if __name__ == "__main__":
    main()
