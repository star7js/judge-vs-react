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
