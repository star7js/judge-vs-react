import json
import math
import os
import tempfile

from eval.pit.run_vlm import mean_latency


def test_mean_latency_two_rows():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"tick": 1, "px": 1.0, "latency_s": 0.2, "action": "none"}) + "\n")
        f.write(json.dumps({"tick": 2, "px": 2.0, "latency_s": 0.3, "action": "jump"}) + "\n")
        path = f.name
    assert mean_latency(path) == 0.25


def test_mean_latency_missing_file_is_nan():
    assert math.isnan(mean_latency("/tmp/does-not-exist-latency-log.jsonl"))


def test_mean_latency_empty_file_is_nan():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        path = f.name
    assert math.isnan(mean_latency(path))
