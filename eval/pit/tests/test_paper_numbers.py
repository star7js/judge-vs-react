"""Regression tests pinning every headline number in the paper to the
committed data, through the same code paths analyze.py uses. If a data
regen or an analyze.py change shifts any published value, this fails
before the paper silently drifts from the repo."""
import os
import statistics

import pytest

from eval.pit.analyze import load_jsonl, ratio, transition_fit, window_width

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def _rows(name):
    return load_jsonl(os.path.join(DATA, name))


def _fit(rows):
    usable = [r for r in rows if window_width(r) > 0]
    return usable, transition_fit(
        [ratio(r) for r in usable], [r["cleared"] for r in usable]
    )


def _unique_oracle_configs(rows):
    # The oracle ignores fog and seed, so the 1680 rows are 60x replicates of
    # 28 (pit_width, delay) configurations. Collapse to one row per config.
    uniq = {}
    for r in rows:
        uniq.setdefault((r["pit_width"], r["delay"]), r)
    return list(uniq.values())


def test_s1_calibration_r_star_and_counts():
    rows = _rows("oracle_sweep.jsonl")
    assert len(rows) == 1680
    assert sum(window_width(r) <= 0 for r in rows) == 420
    # Paper: 28 unique configs, 7 empty-window (pit width 4), 21 informative;
    # every fog/seed replicate of a config is identical.
    uniq = _unique_oracle_configs(rows)
    assert len(uniq) == 28
    assert sum(window_width(r) <= 0 for r in uniq) == 7
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault((r["pit_width"], r["delay"]), set()).add(
            (r["cleared"], r["decision_x"])
        )
    assert all(len(v) == 1 for v in by_cfg.values())
    usable, fit = _fit(rows)
    assert round(fit["r_star"], 2) == 1.17
    _, fit_uniq = _fit(uniq)
    assert round(fit_uniq["r_star"], 2) == 1.17


def test_s1_step_edge_bracket():
    # The paper reports the discrete-grid bracket around the transition:
    # last clear at r = 1.04, first failure at r = 1.30, both at pit width 2.
    # Within the default grid width 1 never fails and width 3 never clears
    # past D = 0, so the edge is bracketed at one width only.
    rows = [r for r in _rows("oracle_sweep.jsonl") if window_width(r) > 0]
    last_clear = max(rows, key=lambda r: ratio(r) if r["cleared"] else -1)
    first_fail = min(rows, key=lambda r: ratio(r) if not r["cleared"] else float("inf"))
    assert (round(ratio(last_clear), 2), last_clear["pit_width"]) == (1.04, 2)
    assert (round(ratio(first_fail), 2), first_fail["pit_width"]) == (1.30, 2)
    assert all(r["cleared"] for r in rows if r["pit_width"] == 1)
    assert all(not r["cleared"] for r in rows if r["pit_width"] == 3 and r["delay"] > 0)


def test_s1_edge_sweep_brackets_every_width():
    # 1-tick delay grid, widths 1-3, one seed, fog 0: 123 rows. Paper: the
    # edge is bracketed within a tick at every width -- width 1 [1.01, 1.04],
    # width 2 [1.04, 1.10], width 3 [0.89, 1.77] (its window is ~1.1 ticks
    # wide) -- and the pooled logistic fit gives r* = 1.04.
    from eval.pit.analyze import edge_brackets
    rows = _rows("oracle_edge_sweep.jsonl")
    assert len(rows) == 123
    assert {r["pit_width"] for r in rows} == {1, 2, 3}
    assert {r["fog"] for r in rows} == {0.0} and {r["seed"] for r in rows} == {0}
    b = edge_brackets(rows)
    got = {pw: (round(v["last_clear_r"], 2), round(v["first_fail_r"], 2)) for pw, v in b.items()}
    assert got == {1: (1.01, 1.04), 2: (1.04, 1.10), 3: (0.89, 1.77)}
    assert {pw: (v["last_clear_delay"], v["first_fail_delay"]) for pw, v in b.items()} == {
        1: (30, 31), 2: (16, 17), 3: (1, 2),
    }
    # Each width is a clean step: every delay below the edge clears, every
    # delay at or above it fails.
    for pw in (1, 2, 3):
        g = sorted((r for r in rows if r["pit_width"] == pw), key=lambda r: r["delay"])
        edge = b[pw]["first_fail_delay"]
        assert all(r["cleared"] == (r["delay"] < edge) for r in g), pw
    _, fit = _fit(rows)
    assert round(fit["r_star"], 2) == 1.04


def test_s2_same_decision_flip():
    # Unique configs, oracle-correct decisions only: 3/3 clear at delay 0;
    # of the 9 configs past r = 1, only the r = 1.04 same-tick case clears,
    # and 0/8 clear at r >= 1.30. The paper reports these counts, not a
    # pooled rate (six of the nine are pit width 3 at r = 3.5-21).
    correct = [
        r for r in _unique_oracle_configs(_rows("oracle_sweep.jsonl"))
        if window_width(r) > 0 and r["in_window"] == 1
    ]
    d0 = [r for r in correct if r["delay"] == 0]
    late = [r for r in correct if r["delay"] > 0 and ratio(r) > 1]
    past_edge = [r for r in late if round(ratio(r), 2) >= 1.30]
    assert (sum(r["cleared"] for r in d0), len(d0)) == (3, 3)
    assert (sum(r["cleared"] for r in late), len(late)) == (1, 9)
    assert [round(ratio(r), 2) for r in late if r["cleared"]] == [1.04]
    assert (sum(r["cleared"] for r in past_edge), len(past_edge)) == (0, 8)
    assert sum(r["pit_width"] == 3 for r in late) == 6


def test_s3_action_repeat_ladder():
    rows = _rows("oracle_repeat_sweep.jsonl")
    expected = {1: 1.10, 2: 0.97, 4: 0.97, 8: 0.71, 16: 0.71}
    for k, want in expected.items():
        _, fit = _fit([r for r in rows if r["repeat"] == k])
        assert round(fit["r_star"], 2) == want, f"k={k}"


def test_s4_aim_calibration():
    rows = _rows("aim_sweep.jsonl")
    assert len(rows) == 205
    assert len({(r["pit_width"], r["delay"]) for r in rows}) == 205  # all unique
    usable, fit = _fit(rows)
    assert round(fit["r_star"], 2) == 0.98
    # Paper: at every width the last hit lies at r = 0.87-0.96 and the first
    # miss at exactly r = 1.00.
    for wid in sorted({r["pit_width"] for r in usable}):
        g = [r for r in usable if r["pit_width"] == wid]
        last_hit = max(ratio(r) for r in g if r["cleared"])
        first_miss = min(ratio(r) for r in g if not r["cleared"])
        assert 0.87 <= round(last_hit, 2) <= 0.96, wid
        assert round(first_miss, 2) == 1.00, wid


def test_perception_v2_api_battery():
    # Paper's perception table: API battery, 100 balanced frames per cell
    # (50 in-window, 50 out), two prompts x two fog levels x three hosted
    # models, plus local Qwen2.5-VL 7B. Neutral prompt: all at chance
    # (Sonnet/Opus answered WAIT on 199/200 each). Cue prompt: 97/91/99%
    # clear; 68/93/94% fog. Qwen: constant WAIT (neutral) / constant JUMP
    # (cue), 50% in all four cells.
    from eval.pit.analyze import perception_v2_table
    rows = _rows("perception_judgments_v2.jsonl")
    assert len(rows) == 1600
    assert all(r["decision"] in ("JUMP", "WAIT") for r in rows)
    hosted = [r for r in rows if "API" in r["judge"]]
    assert len(hosted) == 1200
    cells = perception_v2_table(hosted)
    want = {
        ("claude-haiku-4-5 (API)", "blind-v1", 0.0): 52, ("claude-haiku-4-5 (API)", "blind-v1", 1.0): 48,
        ("claude-haiku-4-5 (API)", "cue-v1", 0.0): 97, ("claude-haiku-4-5 (API)", "cue-v1", 1.0): 68,
        ("claude-sonnet-4-6 (API)", "blind-v1", 0.0): 50, ("claude-sonnet-4-6 (API)", "blind-v1", 1.0): 50,
        ("claude-sonnet-4-6 (API)", "cue-v1", 0.0): 91, ("claude-sonnet-4-6 (API)", "cue-v1", 1.0): 93,
        ("claude-opus-4-8 (API)", "blind-v1", 0.0): 49, ("claude-opus-4-8 (API)", "blind-v1", 1.0): 50,
        ("claude-opus-4-8 (API)", "cue-v1", 0.0): 99, ("claude-opus-4-8 (API)", "cue-v1", 1.0): 94,
    }
    got = {(j, p, f): c["correct"] for (j, p, e, f), c in cells.items()}
    assert got == want
    for c in cells.values():
        assert (c["n"], c["n_in"], c["n_out"]) == (100, 50, 50)
    neutral_jumps = {
        j: sum(r["jump"] for r in hosted if r["judge"] == j and r["prompt"] == "blind-v1")
        for j in ("claude-sonnet-4-6 (API)", "claude-opus-4-8 (API)")
    }
    assert neutral_jumps == {"claude-sonnet-4-6 (API)": 0, "claude-opus-4-8 (API)": 1}
    # Sonnet's cue/clear errors are pre-window false JUMPs, not misses.
    s_clear = [r for r in hosted if r["judge"] == "claude-sonnet-4-6 (API)" and r["prompt"] == "cue-v1" and r["fog"] == 0.0]
    assert (sum(r["jump"] for r in s_clear if not r["in_window"]),
            sum(not r["jump"] for r in s_clear if r["in_window"])) == (9, 0)
    # Haiku under fog misses 30 of 50 in-window frames.
    h_fog = [r for r in hosted if r["judge"] == "claude-haiku-4-5 (API)" and r["prompt"] == "cue-v1" and r["fog"] == 1.0]
    assert sum(not r["jump"] for r in h_fog if r["in_window"]) == 30
    # Qwen2.5-VL 7B (local): a constant answer in each arm, so exactly chance.
    qwen = [r for r in rows if r["judge"] == "qwen2.5-vl-7b (local, Ollama)"]
    assert len(qwen) == 400 and all(r["effort"] == "n/a" for r in qwen)
    q_cells = perception_v2_table(qwen)
    assert {(p, f): c["correct"] for (j, p, e, f), c in q_cells.items()} == {
        ("blind-v1", 0.0): 50, ("blind-v1", 1.0): 50, ("cue-v1", 0.0): 50, ("cue-v1", 1.0): 50}
    assert sum(r["jump"] for r in qwen if r["prompt"] == "blind-v1") == 0
    assert sum(r["jump"] for r in qwen if r["prompt"] == "cue-v1") == 200
    # Under the cue prompt it echoed the prompt's first line, then said JUMP.
    assert all(r["raw"].startswith("What touches the bottom border") for r in qwen if r["prompt"] == "cue-v1")


def test_perception_table_counts():
    # The retracted subagent-judged pilot (n = 10 per cell); kept for the
    # record, no longer the paper's perception table.
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
    # Paper: in the fast configuration the hosted models answered WAIT on
    # nearly every frame (Haiku, Sonnet 20/20; Opus 18/20) -- these are
    # time-to-answer measurements, not judgments.
    waits = {}
    for r in rows:
        m = r["model"].split(" ")[0]
        waits[m] = waits.get(m, 0) + (r["decision"] == "WAIT")
    assert waits["claude-haiku-4-5"] == 20
    assert waits["claude-sonnet-4-6"] == 20
    assert waits["claude-opus-4-8"] == 18
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


def test_s5_judge_battery_and_first_passage():
    # Paper (first-passage section): the final judge's battery is 108 approach
    # frames x 3 samples; zero JUMPs on the first 79 frames; isolated JUMPs at
    # x = 8.10 (1/3) and 8.24 (2/3); unanimous JUMP from x = 8.87 on (one
    # frame before the window opens at 8.921). With independent per-frame
    # errors over 90 approach frames, a single-JUMP commit rule fires early
    # with probability 60% at p = 1% and 84% at p = 2%; the two-consecutive
    # rule 0.9% and 3.4%.
    from eval.pit.realmodel.fig_firstpassage import (
        load_battery, p_false_commit_single, p_false_commit_two_consecutive,
    )
    rows = load_battery()
    assert len(rows) == 108 and {r["n"] for r in rows} == {3}
    assert all(r["jump"] == 0 for r in rows[:79])
    assert rows[79]["px"] == 8.1
    pre_window = {r["px"]: r["jump"] for r in rows if r["px"] < 8.921 and r["jump"] > 0}
    assert pre_window == {8.1: 1, 8.24: 2, 8.87: 3}
    assert all(r["jump"] == 3 for r in rows if r["px"] >= 8.87)
    assert sum(r["n"] for r in rows if r["px"] < 8.1) == 237  # far-field samples, zero JUMPs
    assert round(p_false_commit_single(0.01), 2) == 0.60
    assert round(p_false_commit_single(0.02), 2) == 0.84
    assert round(p_false_commit_two_consecutive(0.01), 3) == 0.009
    assert round(p_false_commit_two_consecutive(0.02), 3) == 0.034
    # Sanity on the exact chain: p = 0 never fires, p = 1 always does.
    assert p_false_commit_two_consecutive(0.0) == 0.0
    assert abs(p_false_commit_two_consecutive(1.0) - 1.0) < 1e-12


def test_fog_judge_battery():
    # Paper (fog section): the S5 judge's battery at fog 1, 108 frames x 3
    # samples. Still unanimous on most in-window frames, but a far-field
    # false JUMP at x = 2.85 (1/3), isolated early JUMPs at 8.38, 8.73, 8.80,
    # unanimous from 8.87, and two non-unanimous in-window frames.
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "realmodel", "battery_v4_high_fog1.json")
    rows = json.load(open(path))
    assert len(rows) == 108 and {r["n"] for r in rows} == {3}
    pre = {r["px"]: r["jump"] for r in rows if r["px"] < 8.921 and r["jump"]}
    assert pre == {2.85: 1, 8.38: 1, 8.73: 2, 8.8: 2, 8.87: 3}
    missed = {r["px"]: r["jump"] for r in rows if 8.921 <= r["px"] < 9.999 and r["jump"] < 3}
    assert missed == {9.01: 2, 9.78: 1}


def test_fog_realmodel_boundary_sweep():
    # Paper (fog section): same judge, fog 1, 72 episodes. Commit point
    # spreads over five ticks; 35/72 commit early (vs 9/72 at fog 0).
    # In-window: 19/19 clear through r = 0.97, 0/18 from r = 1.04; r* = 1.01.
    # 22/35 early commits rescued, all with 4 <= D <= 17. Both sweeps clear
    # 41/72. Every episode in both sweeps obeys: cleared iff the takeoff
    # lands inside the window extended by the same-tick tolerance,
    # w_lo <= commit + v*D <= 10.06.
    from collections import Counter
    base = os.path.join(os.path.dirname(__file__), "..", "realmodel")
    fog0 = _rows(os.path.join(base, "sonnet_boundary_sweep.jsonl"))
    fog1 = _rows(os.path.join(base, "sonnet_boundary_sweep_fog1.jsonl"))
    assert len(fog1) == 72 and {r["fog"] for r in fog1} == {1.0}
    assert {(r["delay"], r["seed"]) for r in fog1} == {(r["delay"], r["seed"]) for r in fog0}
    r_of = lambda row: ratio(row, source="delay")

    commits = Counter(round(r["decision_x"], 2) for r in fog1)
    assert commits == {8.8: 11, 8.87: 24, 8.94: 19, 9.01: 17, 9.08: 1}
    inw = [r for r in fog1 if r["in_window"] == 1]
    out = [r for r in fog1 if r["in_window"] == 0]
    assert (len(inw), len(out)) == (37, 35)
    assert (sum(r["cleared"] for r in inw if r_of(r) <= 0.98), sum(1 for r in inw if r_of(r) <= 0.98)) == (19, 19)
    assert (sum(r["cleared"] for r in inw if r_of(r) >= 1.0), sum(1 for r in inw if r_of(r) >= 1.0)) == (0, 18)
    fit = transition_fit([r_of(r) for r in inw], [r["cleared"] for r in inw])
    assert round(fit["r_star"], 2) == 1.01
    assert sum(r["cleared"] for r in out) == 22
    # 21 rescued with 4 <= D <= 17 (every early commit in that band), plus one
    # 8.80 commit at D = 18 that lands exactly on the 10.06 tolerance edge.
    assert sum(1 for r in out if r["cleared"] and 4 <= r["delay"] <= 17) == 21
    assert all(r["cleared"] for r in out if 4 <= r["delay"] <= 17)
    assert [(r["decision_x"], r["delay"]) for r in out if r["cleared"] and r["delay"] > 17] == [(8.8, 18)]
    assert sum(r["cleared"] for r in fog1) == 41 == sum(r["cleared"] for r in fog0)
    assert (sum(r["cleared"] for r in fog0 if r["in_window"]), sum(r["cleared"] for r in fog0 if not r["in_window"])) == (36, 5)
    assert (sum(r["cleared"] for r in fog1 if r["in_window"]), sum(r["cleared"] for r in fog1 if not r["in_window"])) == (19, 22)
    for rows in (fog0, fog1):
        for r in rows:
            landing = r["decision_x"] + 0.07 * r["delay"]
            assert (8.921 <= landing <= 10.065) == bool(r["cleared"]), (r["decision_x"], r["delay"], r["cleared"])
    pooled = transition_fit([r_of(r) for r in fog1], [r["cleared"] for r in fog1])
    assert round(pooled["r_star"], 2) == 1.07 and pooled["slope"] < 20 and fit["slope"] > 300


def test_s5_realmodel_boundary_sweep():
    # S5: Sonnet 4.6 swept through the boundary by injected delay, its own
    # frame judgment in the loop. Paper: 72 episodes, in-window decisions
    # clear 32/32 through r = 0.97, 4/5 at r = 1.04, 0/26 from r = 1.10 on;
    # logistic r* = 1.05; 9 early (out-of-window) decisions of which 5 are
    # delay-rescued; commit position 51x at x=8.94 and 12x at x=9.01.
    rows = _rows(os.path.join("..", "realmodel", "sonnet_boundary_sweep.jsonl"))
    assert len(rows) == 72
    assert {round(window_width(r), 3) for r in rows} == {1.078}
    # Rows carry both the injected delay and the model's measured latency;
    # the sweep's r axis is the injected delay, and the shared helper must be
    # told so (it refuses to guess).
    with pytest.raises(ValueError):
        ratio(rows[0])
    r_of = lambda row: ratio(row, source="delay")

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


def test_delay_sweep_invariance():
    # Eq. (sweep): expected clears from each sweep's own commit distribution
    # match the observed 41/72 in both weathers, and a dense grid gives the
    # F-independent limit (x_hi - w_lo)/v per seed (16.3 ticks) to within
    # discretisation.
    from eval.pit.analyze import delay_sweep_expectation
    grid = [0, 4, 8, 12, 14, 15, 16, 17, 18, 20, 24, 31]
    want = {"sonnet_boundary_sweep.jsonl": 41.8, "sonnet_boundary_sweep_fog1.jsonl": 41.3}
    for fname, exp in want.items():
        rows = _rows(os.path.join("..", "realmodel", fname))
        e = delay_sweep_expectation(rows, grid, per_delay=6)
        assert sum(r["cleared"] for r in rows) == 41
        assert round(e["expected"], 1) == exp
        assert abs(e["dense_expected"] - 101) <= 1.0 and round(e["dense_limit"], 1) == 98.1


def test_clock_sweep():
    # The judge's own latency through the boundary (clock-rate emulation):
    # 72 episodes, no retries; 0/30 for tau <= 0.4 s, first clears at 0.5 s
    # (r = 0.32), 18/18 for tau >= 1.5 s; 10 of the 11 failures between
    # 0.5 and 1.0 s contain a >= 10 s call on a frame in [7.9, 8.4]; the
    # deliberation stall: 70 such calls, median 13.6 s, 47 over 10 s, vs
    # 1,878 others, median 2.6 s, 12 over 10 s; in-window frames JUMP 96/96.
    import statistics
    rows = _rows(os.path.join("..", "realmodel", "sonnet_clock_sweep.jsonl"))
    assert len(rows) == 72 and sum(r["cleared"] for r in rows) == 31
    assert sum(r["n_retried_calls"] for r in rows) == 0
    assert all(r["effort"] == "high" for r in rows)
    by_tau = {}
    for r in rows:
        by_tau.setdefault(r["clock_s"], []).append(r)
    cleared = {round(t, 4): sum(x["cleared"] for x in g) for t, g in by_tau.items()}
    assert cleared == {round(1 / 60, 4): 0, 0.1: 0, 0.2: 0, 0.3: 0, 0.4: 0, 0.5: 2, 0.6: 3,
                       0.8: 5, 1.0: 3, 1.5: 6, 2.0: 6, 4.0: 6}
    calls = [c for r in rows for c in r["call_log"]]
    assert len(calls) == 1948
    hot = [c[1] for c in calls if 7.9 <= c[0] <= 8.4]
    cold = [c[1] for c in calls if not 7.9 <= c[0] <= 8.4]
    assert (len(hot), len(cold)) == (70, 1878)
    assert round(statistics.median(hot), 1) == 13.6 and round(statistics.median(cold), 1) == 2.6
    assert (sum(h >= 10 for h in hot), sum(c >= 10 for c in cold)) == (47, 12)
    assert round(max(hot), 1) == 32.4
    inw = [c for c in calls if 8.921 <= c[0] <= 9.999]
    assert len(inw) == 96 and all(c[3] == "jump" for c in inw)
    fails = [r for r in rows if 0.5 <= r["clock_s"] <= 1.0 and not r["cleared"]]
    stalled = [r for r in fails if any(7.9 <= c[0] <= 8.4 and c[1] >= 10 for c in r["call_log"])]
    assert (len(fails), len(stalled)) == (11, 10)
    # takeoff = judged frame + v * emulated delay, for every committed episode
    for r in rows:
        if r["judge_x"] is not None and r["decision_x"] > 0:
            n = [c[2] for c in r["call_log"] if c[0] == r["judge_x"]][0]
            assert abs(r["decision_x"] - (r["judge_x"] + 0.07 * n)) < 0.011


def test_clock_sweep_low_effort():
    # Same clock sweep at low effort: no stall (median 1.0 s, max 2.5 s on
    # the [7.9, 8.4] frames), in-window JUMP 18/18, but 54% false JUMP on
    # x in [7.5, 8.921) (233/429; 122/128 on [8.0, 8.5)), 58/72 early
    # commits (every episode at tau >= 0.4 s), 8/72 cleared, all at
    # tau = 0.1-0.3 s. High-effort band rate for comparison: 17/223.
    import statistics
    rows = _rows(os.path.join("..", "realmodel", "sonnet_clock_sweep_low.jsonl"))
    assert len(rows) == 72 and sum(r["cleared"] for r in rows) == 8
    assert all(r["effort"] == "low" for r in rows) and sum(r["n_retried_calls"] for r in rows) == 0
    calls = [c for r in rows for c in r["call_log"]]
    assert len(calls) == 3416
    assert round(statistics.median(c[1] for c in calls), 1) == 1.0
    assert max(c[1] for c in calls if 7.9 <= c[0] <= 8.4) < 2.6
    inw = [c for c in calls if 8.921 <= c[0] <= 9.999]
    assert len(inw) == 18 and all(c[3] == "jump" for c in inw)
    band = [c for c in calls if 7.5 <= c[0] < 8.921]
    assert (sum(c[3] == "jump" for c in band), len(band)) == (233, 429)
    near = [c for c in band if 8.0 <= c[0] < 8.5]
    assert (sum(c[3] == "jump" for c in near), len(near)) == (122, 128)
    early = [r for r in rows if r["judge_x"] is not None and r["judge_x"] < 8.921]
    assert len(early) == 58
    assert all(r["judge_x"] is not None and r["judge_x"] < 8.921 for r in rows if r["clock_s"] >= 0.4)
    assert {round(r["clock_s"], 2) for r in rows if r["cleared"]} == {0.1, 0.2, 0.3}
    hi = _rows(os.path.join("..", "realmodel", "sonnet_clock_sweep.jsonl"))
    hband = [c for r in hi for c in r["call_log"] if 7.5 <= c[0] < 8.921]
    assert (sum(c[3] == "jump" for c in hband), len(hband)) == (17, 223)
