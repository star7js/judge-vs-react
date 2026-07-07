"""Regression tests pinning every headline number in the paper to the
committed data, through the same code paths analyze.py uses. If a data
regen or an analyze.py change shifts any published value, this fails
before the paper silently drifts from the repo."""
import os
import statistics

from eval.pit.analyze import load_jsonl, ratio, transition_fit, window_width

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def _rows(name):
    return load_jsonl(os.path.join(DATA, name))


def _fit(rows):
    usable = [r for r in rows if window_width(r) > 0]
    return usable, transition_fit(
        [ratio(r) for r in usable], [r["cleared"] for r in usable]
    )


def test_s1_timing_law_r_star_and_exclusions():
    rows = _rows("oracle_sweep.jsonl")
    assert len(rows) == 1680
    assert sum(window_width(r) <= 0 for r in rows) == 420
    usable, fit = _fit(rows)
    assert round(fit["r_star"], 2) == 1.17


def test_s1_step_edge_bracket():
    # The paper reports the discrete-grid bracket around the transition:
    # last clear at r = 1.04, first failure at r = 1.30.
    rows = [r for r in _rows("oracle_sweep.jsonl") if window_width(r) > 0]
    last_clear = max(ratio(r) for r in rows if r["cleared"])
    first_fail = min(ratio(r) for r in rows if not r["cleared"])
    assert round(last_clear, 2) == 1.04
    assert round(first_fail, 2) == 1.30


def test_s2_same_decision_flip():
    # Unique configs (the oracle emits fog/seed duplicates), oracle-correct
    # decisions only: 3/3 clear at delay 0; 1/9 once delay exceeds the window.
    uniq = {}
    for r in _rows("oracle_sweep.jsonl"):
        uniq.setdefault((r["pit_width"], r["delay"], r["speed"]), r)
    correct = [r for r in uniq.values() if r["in_window"] == 1]
    d0 = [r for r in correct if r["delay"] == 0]
    late = [r for r in correct if r["delay"] > 0 and ratio(r) > 1]
    assert (sum(r["cleared"] for r in d0), len(d0)) == (3, 3)
    assert (sum(r["cleared"] for r in late), len(late)) == (1, 9)


def test_s3_action_repeat_ladder():
    rows = _rows("oracle_repeat_sweep.jsonl")
    expected = {1: 1.10, 2: 0.97, 4: 0.97, 8: 0.71, 16: 0.71}
    for k, want in expected.items():
        _, fit = _fit([r for r in rows if r["repeat"] == k])
        assert round(fit["r_star"], 2) == want, f"k={k}"


def test_s4_aim_generality():
    rows = _rows("aim_sweep.jsonl")
    assert len(rows) == 205
    usable, fit = _fit(rows)
    assert round(fit["r_star"], 2) == 0.98
    # Paper's step-edge bracket: last hit r = 0.96, first miss exactly r = 1.00.
    last_hit = max(ratio(r) for r in usable if r["cleared"])
    first_miss = min(ratio(r) for r in usable if not r["cleared"])
    assert round(last_hit, 2) == 0.96
    assert round(first_miss, 2) == 1.00


def test_perception_table_counts():
    rows = _rows("perception_judgments.jsonl")
    correct = {}
    for r in rows:
        key = (r["judge"].split(" ")[0].split("(")[0], r["fog"])
        n_ok, n = correct.get(key, (0, 0))
        correct[key] = (n_ok + r["correct"], n + 1)
    assert correct == {
        ("claude-opus", 0.0): (9, 10), ("claude-opus", 1.0): (5, 10),
        ("claude-sonnet", 0.0): (8, 10), ("claude-sonnet", 1.0): (5, 10),
        ("qwen2.5-vl-7b", 0.0): (6, 10), ("qwen2.5-vl-7b", 1.0): (6, 10),
        ("claude-haiku-4-5", 0.0): (4, 10), ("claude-haiku-4-5", 1.0): (4, 10),
    }


def test_perception_fog_inwindow_misses():
    # Opus and Sonnet each missed every in-window frame under fog (0/5).
    rows = _rows("perception_judgments.jsonl")
    for judge in ("claude-opus", "claude-sonnet"):
        fog_iw = [
            r for r in rows
            if r["judge"].startswith(judge) and r["fog"] == 1.0 and r["in_window"] == 1
        ]
        assert (sum(r["jump"] for r in fog_iw), len(fog_iw)) == (0, 5), judge


def test_latency_medians():
    rows = _rows("vlm_latency.jsonl")
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"].split(" ")[0], []).append(r["latency_s"])
    medians = {m: round(statistics.median(v), 2) for m, v in by_model.items()}
    assert all(len(v) == 20 for v in by_model.values())
    assert medians == {
        "claude-haiku-4-5": 0.95,
        "claude-sonnet-4-6": 1.45,
        "claude-opus-4-8": 1.53,
        "qwen2.5-vl-7b": 7.85,
        "llama3.2-vision:11b": 12.15,
    }


def test_window_budget_257ms():
    # T_window = (W / v) * tick = 257 ms at pit_width 2 (W = 1.078, v = 0.07).
    rows = [
        r for r in _rows("oracle_sweep.jsonl")
        if r["pit_width"] == 2 and window_width(r) > 0
    ]
    widths = {round(window_width(r), 3) for r in rows}
    assert widths == {1.078}
    t_ms = (1.078 / 0.07) * (1 / 60) * 1000
    assert round(t_ms) == 257


def test_s5_realmodel_boundary_sweep():
    # S5: Sonnet 4.6 swept through the boundary by injected delay, its own
    # frame judgment in the loop. Paper: 72 episodes, in-window decisions
    # clear 32/32 through r = 0.97, 4/5 at r = 1.04, 0/26 from r = 1.10 on;
    # logistic r* = 1.05; 9 early (out-of-window) decisions of which 5 are
    # delay-rescued; commit position 51x at x=8.94 and 12x at x=9.01.
    rows = _rows(os.path.join("..", "realmodel", "sonnet_boundary_sweep.jsonl"))
    assert len(rows) == 72
    v, w = 0.07, 1.078
    assert {round(window_width(r), 3) for r in rows} == {w}
    r_of = lambda row: v * row["delay"] / w

    inw = [r for r in rows if r["in_window"] == 1]
    out = [r for r in rows if r["in_window"] == 0]
    assert (len(inw), len(out)) == (63, 9)
    assert sum(r["cleared"] for r in out) == 5  # delay-rescued misjudgments

    below = [r for r in inw if r_of(r) <= 0.98]
    edge = [r for r in inw if 1.0 < r_of(r) < 1.09]
    above = [r for r in inw if r_of(r) >= 1.09]
    assert (sum(r["cleared"] for r in below), len(below)) == (32, 32)
    assert (sum(r["cleared"] for r in edge), len(edge)) == (4, 5)
    assert (sum(r["cleared"] for r in above), len(above)) == (0, 26)

    fit = transition_fit([r_of(r) for r in inw], [r["cleared"] for r in inw])
    assert round(fit["r_star"], 2) == 1.05

    commits = [round(r["decision_x"], 2) for r in rows]
    assert commits.count(8.94) == 51
    assert commits.count(9.01) == 12
