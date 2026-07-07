import json, os, tempfile
from eval.pit.config import grid
from eval.pit.run_oracle_sweep import run_sweep

BIN = os.environ.get("KEEN_BIN", "raycaster/build/keen-raycaster")

def test_small_sweep_writes_valid_rows():
    cfgs = grid(pit_width=[2], fog=[0.0], delay=[0, 20], seed=list(range(3)))
    with tempfile.NamedTemporaryFile("r+", suffix=".jsonl", delete=False) as f:
        n = run_sweep(BIN, cfgs, f.name)
        rows = [json.loads(l) for l in open(f.name)]
    assert n == len(rows) == 6
    d0 = [r for r in rows if r["delay"] == 0]
    d20 = [r for r in rows if r["delay"] == 20]
    assert all(r["cleared"] == 1 for r in d0)      # oracle clears with no delay
    assert all(r["fell"] == 1 for r in d20)        # and falls with big delay
