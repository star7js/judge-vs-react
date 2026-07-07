"""The aim sweep is the generality test: a small run must produce the pit's
schema and a transition near r~1 under `analyze.ratio()` (which must work on
aim rows with zero special-casing, since speed<-omega and w_lo/w_hi<-angle)."""
import os
from pathlib import Path

import pytest

from eval.pit.analyze import ratio, transition_fit, load_jsonl
from eval.pit.run_aim_sweep import run_sweep

BIN = os.environ.get("KEEN_BIN", "raycaster/build/keen-raycaster")
_HAS_BIN = Path(BIN).exists()
skip_no_bin = pytest.mark.skipif(not _HAS_BIN, reason=f"sim binary not built at {BIN}")


@skip_no_bin
def test_aim_sweep_schema_and_transition(tmp_path):
    out = tmp_path / "aim.jsonl"
    n = run_sweep(BIN, str(out),
                  half_widths={2: 0.06, 4: 0.10}, delays=list(range(0, 30)))
    rows = load_jsonl(str(out))
    assert n == len(rows) == 2 * 30

    # Same schema fields the analysis reads.
    for key in ("policy", "speed", "delay", "cleared", "w_lo", "w_hi", "pit_width"):
        assert key in rows[0]
    # speed carries omega; window is angular (small).
    assert rows[0]["speed"] == pytest.approx(0.01)
    assert rows[0]["w_hi"] - rows[0]["w_lo"] == pytest.approx(0.12, abs=1e-3)

    # A delay-0 shot always hits; the transition sits near r ~ 1.
    zero = [r for r in rows if r["delay"] == 0]
    assert all(r["cleared"] == 1 for r in zero)

    rr = [ratio(r) for r in rows]
    y = [r["cleared"] for r in rows]
    fit = transition_fit(rr, y)
    assert 0.8 < fit["r_star"] < 1.2, fit


@skip_no_bin
def test_aim_wider_target_tolerates_more_delay(tmp_path):
    """r = omega*delay/(2h): a wider window clears at a larger delay."""
    out = tmp_path / "aim2.jsonl"
    run_sweep(BIN, str(out), half_widths={1: 0.04, 5: 0.12}, delays=[16])
    rows = {r["pit_width"]: r for r in load_jsonl(str(out))}
    # delay=16: narrow (2h=0.08 -> r=2.0) misses; wide (2h=0.24 -> r=0.67) hits.
    assert rows[1]["cleared"] == 0
    assert rows[5]["cleared"] == 1
