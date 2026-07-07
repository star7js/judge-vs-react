from eval.pit.config import EpisodeConfig, grid, ROW_KEYS


def test_cli_args_roundtrip():
    c = EpisodeConfig(policy="oracle", model=None, pit_near=10.0, pit_width=2,
                      speed=0.07, fog=0.0, seed=3, delay=8, repeat=1)
    a = c.cli_args()
    assert "--pit-width" in a and a[a.index("--pit-width")+1] == "2"
    assert "--delay" in a and a[a.index("--delay")+1] == "8"


def test_grid_is_deterministic_and_nonempty():
    g1, g2 = grid(), grid()
    assert g1 == g2 and len(g1) == 4*3*7*1*20


def test_row_keys_matches_c_sim_output_no_model():
    # ROW_KEYS must match the raw JSON keys emitted by the C sim exactly,
    # in order. The oracle sim does not emit a "model" key.
    assert ROW_KEYS == [
        "policy", "seed", "pit_near", "pit_width", "speed", "fog",
        "delay", "repeat", "cleared", "fell", "decision_tick",
        "decision_x", "in_window", "w_lo", "w_hi", "ticks",
    ]
    assert "model" not in ROW_KEYS


def test_grid_default_axis_values():
    g = grid()
    assert {c.pit_width for c in g} == {1, 2, 3, 4}
    assert {c.fog for c in g} == {0.0, 0.5, 1.0}
    assert {c.delay for c in g} == {0, 4, 8, 12, 16, 20, 24}
    assert {c.repeat for c in g} == {1}
    assert {c.seed for c in g} == set(range(20))
    assert all(c.policy == "oracle" for c in g)
    assert all(c.pit_near == 10.0 for c in g)
    assert all(c.speed == 0.07 for c in g)
    assert all(c.model is None for c in g)


def test_grid_supports_list_overrides():
    g = grid(pit_width=[2], fog=[0.0], delay=[0, 20], seed=list(range(3)))
    assert len(g) == 1 * 1 * 2 * 1 * 3
    assert {c.pit_width for c in g} == {2}
    assert {c.delay for c in g} == {0, 20}
    assert {c.seed for c in g} == {0, 1, 2}


def test_cli_args_oracle_full_sequence():
    c = EpisodeConfig(policy="oracle", model=None, pit_near=10.0, pit_width=2,
                       speed=0.07, fog=0.0, seed=3, delay=8, repeat=1)
    a = c.cli_args()
    assert a[:2] == ["--episode", "--policy"]
    assert a[2] == "oracle"
    assert "--agent" not in a


def test_cli_args_non_oracle_uses_uniform_policy_flag():
    # The parametric agent path (Task 5a) is `--episode --policy agent ...`;
    # `model` is a Python-side concern for pit_agent.py, never a C flag.
    c = EpisodeConfig(policy="agent", model="some-model", pit_near=10.0, pit_width=2,
                       speed=0.07, fog=0.0, seed=3, delay=8, repeat=1)
    a = c.cli_args()
    assert a[:3] == ["--episode", "--policy", "agent"]
    assert "--agent" not in a
    assert "--model" not in a
