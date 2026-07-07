"""Fit-recovery tests for eval.pit.analyze, using only synthetic logistic
data (never fit against the deterministic oracle sweep itself)."""
import numpy as np

from eval.pit.analyze import psychometric_fit, ratio, transition_fit


def test_transition_recovers_known_threshold():
    rng = np.random.default_rng(0)
    r = np.linspace(0, 2, 400)
    p = 1 / (1 + np.exp(12 * (r - 1.0)))          # true r* = 1.0
    y = (rng.random(len(r)) < p).astype(int)
    out = transition_fit(r, y)
    assert abs(out["r_star"] - 1.0) < 0.1


def test_psychometric_recovers_known_threshold_and_slope():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 2, 400)
    true_threshold, true_slope = 1.0, 8.0
    p = 1 / (1 + np.exp(-true_slope * (x - true_threshold)))  # increasing sigmoid
    y = (rng.random(len(x)) < p).astype(int)
    out = psychometric_fit(x, y)
    assert abs(out["threshold"] - true_threshold) < 0.1
    assert out["slope"] > 0  # recovered slope should keep the increasing sign


def test_ratio_empty_window_is_inf():
    # pit_width=4 degenerate case: w_lo=10.0, w_hi=-1.0 -> W < 0 -> unclearable.
    row = {"speed": 0.07, "delay": 8, "w_lo": 10.0, "w_hi": -1.0}
    assert ratio(row) == float("inf")


def test_ratio_zero_width_window_is_also_inf():
    row = {"speed": 0.07, "delay": 8, "w_lo": 9.0, "w_hi": 9.0}
    assert ratio(row) == float("inf")


def test_ratio_oracle_uses_delay_in_ticks_directly():
    row = {"speed": 0.07, "delay": 10, "w_lo": 9.0, "w_hi": 9.7}
    # W = 0.7; r = 0.07 * 10 / 0.7 = 1.0
    assert abs(ratio(row) - 1.0) < 1e-9


def test_ratio_vlm_converts_L_mean_s_via_tick_s():
    row = {"speed": 0.07, "delay": 0, "L_mean_s": 10 / 60, "w_lo": 9.0, "w_hi": 9.7}
    # delay_ticks = L_mean_s / (1/60) = 10 ticks -> same r as the oracle case above
    assert abs(ratio(row, tick_s=1 / 60) - 1.0) < 1e-9


def test_ratio_prefers_L_mean_s_over_stale_delay_when_both_present():
    # VLM rows carry a config "delay" (usually 0) alongside a real measured
    # L_mean_s; the measured latency should win.
    row = {"speed": 0.07, "delay": 0, "L_mean_s": 10 / 60, "w_lo": 9.0, "w_hi": 9.7}
    assert abs(ratio(row) - 1.0) < 1e-9
