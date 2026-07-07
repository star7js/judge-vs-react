"""Config grid + row schema for the pit-jump experiment sweep.

`EpisodeConfig` describes one invocation of the C sim
(`raycaster/build/keen-raycaster --episode ...`); `grid()` enumerates the
cartesian product of swept axes for a full experiment sweep; `ROW_KEYS` is
the exact, ordered set of JSON keys the C sim prints on its one-line
`--episode` output, used by a later task to validate raw C output rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

# Exact keys emitted by the C sim's --episode JSON line, in order. The
# oracle policy has no model, so the sim never emits a "model" key -- keep
# it out of ROW_KEYS. (EpisodeConfig still carries a `model` field; the
# later Python driver merges it into the final JSONL row separately.)
ROW_KEYS: list[str] = [
    "policy",
    "seed",
    "pit_near",
    "pit_width",
    "speed",
    "fog",
    "delay",
    "repeat",
    "cleared",
    "fell",
    "decision_tick",
    "decision_x",
    "in_window",
    "w_lo",
    "w_hi",
    "ticks",
]

# Default sweep axes.
_DEFAULT_PIT_WIDTH = [1, 2, 3, 4]
_DEFAULT_FOG = [0.0, 0.5, 1.0]
_DEFAULT_DELAY = [0, 4, 8, 12, 16, 20, 24]
_DEFAULT_REPEAT = [1]
_DEFAULT_SEED = list(range(20))


@dataclass
class EpisodeConfig:
    policy: str
    pit_near: float
    pit_width: int
    speed: float
    fog: float
    seed: int
    delay: int
    repeat: int
    model: str | None = None

    def cli_args(self) -> list[str]:
        # Uniform across all policies: the parametric agent path (Task 5a)
        # is `--episode --policy agent <flags>`, not the old fixed-pit
        # `--agent` C flag. `model` is a Python-side concern (passed to
        # pit_agent.py), never a C flag.
        args = ["--episode", "--policy", self.policy]
        args += [
            "--pit-near", _fmt(self.pit_near),
            "--pit-width", _fmt(self.pit_width),
            "--speed", _fmt(self.speed),
            "--fog", _fmt(self.fog),
            "--seed", _fmt(self.seed),
            "--delay", _fmt(self.delay),
            "--repeat", _fmt(self.repeat),
        ]
        return args


def _fmt(value: float | int) -> str:
    """Format a numeric CLI value so C's atof/atoi parse it as a plain number."""
    if isinstance(value, bool):  # guard: bool is an int subclass
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def grid(**overrides) -> list[EpisodeConfig]:
    """Enumerate the cartesian product of swept axes.

    With no overrides, returns the default sweep:
    pit_width x fog x delay x repeat x seed = 4*3*7*1*20 = 1680 configs,
    in deterministic, stable order. Any swept axis may be overridden with
    a list of values via keyword, e.g. grid(pit_width=[2], delay=[0, 20]).
    """
    pit_width = overrides.get("pit_width", _DEFAULT_PIT_WIDTH)
    fog = overrides.get("fog", _DEFAULT_FOG)
    delay = overrides.get("delay", _DEFAULT_DELAY)
    repeat = overrides.get("repeat", _DEFAULT_REPEAT)
    seed = overrides.get("seed", _DEFAULT_SEED)

    policy = overrides.get("policy", "oracle")
    pit_near = overrides.get("pit_near", 10.0)
    speed = overrides.get("speed", 0.07)
    model = overrides.get("model", None)

    configs = []
    for pw, f, d, r, s in product(pit_width, fog, delay, repeat, seed):
        configs.append(
            EpisodeConfig(
                policy=policy,
                pit_near=pit_near,
                pit_width=pw,
                speed=speed,
                fog=f,
                seed=s,
                delay=d,
                repeat=r,
                model=model,
            )
        )
    return configs
