import math
import os
import tempfile

import pytest

from eval.pit.config import EpisodeConfig
from eval.pit.run_vlm import run_vlm_episode

BIN = os.environ.get("KEEN_BIN", "raycaster/build/keen-raycaster")


@pytest.mark.skipif(not os.path.exists(BIN), reason=f"keen-raycaster binary not found at {BIN}")
def test_run_vlm_episode_rules():
    cfg = EpisodeConfig(policy="agent", model=None, pit_near=10.0, pit_width=2,
                         speed=0.07, fog=0.0, seed=1, delay=0, repeat=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        latency_log = os.path.join(tmpdir, "latency.jsonl")
        frame_out = os.path.join(tmpdir, "frame.bmp")
        result = run_vlm_episode(
            BIN, cfg, backend="rules", model=None,
            latency_log=latency_log, frame_out=frame_out,
        )

    assert result["cleared"] == 1
    assert result["policy"] == "agent"
    assert math.isnan(result["L_mean_s"])
    assert result["backend"] == "rules"
    assert result["model"] is None


def test_run_vlm_episode_vision_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = EpisodeConfig(policy="agent", model="claude-haiku-4-5", pit_near=10.0,
                         pit_width=2, speed=0.07, fog=0.0, seed=1, delay=0, repeat=1)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        run_vlm_episode(BIN, cfg, backend="vision", model="claude-haiku-4-5",
                         latency_log="/tmp/unused-latency.jsonl")


@pytest.mark.skipif(not os.path.exists(BIN), reason=f"keen-raycaster binary not found at {BIN}")
def test_run_vlm_episode_dead_agent_is_an_error_not_a_fall(tmp_path, monkeypatch):
    """An agent that crashes mid-episode leaves the player walking into the
    pit. That must surface as a RuntimeError (so sweep drivers record an
    error and retry), never as a `fell` row."""
    import eval.pit.run_vlm as rv

    crasher = tmp_path / "crashing_agent.py"
    crasher.write_text(
        "import os, socket, sys, time\n"
        "s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "for _ in range(50):\n"
        "    try:\n"
        "        s.connect(os.environ['KEEN_AGENT_SOCK']); break\n"
        "    except OSError:\n"
        "        time.sleep(0.1)\n"
        "buf = b''\n"
        "for _ in range(5):\n"
        "    while b'\\n' not in buf:\n"
        "        buf += s.recv(4096)\n"
        "    _, buf = buf.split(b'\\n', 1)\n"
        "    s.sendall(b'{\"action\":\"none\"}\\n')\n"
        "sys.exit('simulated API failure')\n"
    )
    monkeypatch.setattr(rv, "PIT_AGENT", crasher)
    # Short path: pytest's tmp_path can exceed the ~104-char sun_path limit on macOS.
    monkeypatch.setenv("KEEN_AGENT_SOCK", f"/tmp/keen_test_dead_agent_{os.getpid()}.sock")
    cfg = EpisodeConfig(policy="agent", pit_near=10.0, pit_width=2, speed=0.07,
                        fog=0.0, seed=1, delay=0, repeat=1)
    with pytest.raises(RuntimeError) as exc:
        rv.run_vlm_episode(BIN, cfg, backend="rules",
                           latency_log=str(tmp_path / "lat.jsonl"),
                           frame_out=str(tmp_path / "frame.bmp"))
    assert "simulated API failure" in str(exc.value)   # the agent's own error is surfaced
