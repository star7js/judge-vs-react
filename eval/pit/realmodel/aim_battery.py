"""Frame battery for the aim judge (the second scene).

Renders every tick of the aim sweep (raycaster --episode --task aim
--aim-frames: from a0 to one window past the far edge, 68 frames), asks the
judge (sweep_agent.AIM_PROMPT, high effort) `--samples` times per frame, and
writes battery_aim_high.json: [{"tick", "aim", "in_window", "jump", "n"}].
Prints the FIRE map, the early/missed frames and token usage. The gate
before an aim sweep, as battery_fog.py is for the pit.

Usage (repo root, ANTHROPIC_API_KEY set):
    python3 eval/pit/realmodel/aim_battery.py --samples 3
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from PIL import Image

HERE = Path(__file__).parent
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
from sweep_agent import AIM_PROMPT, thinking_kwargs  # noqa: E402

MODEL = "claude-sonnet-4-6"


def render(binary: str, workdir: Path) -> list[dict]:
    proc = subprocess.run(
        [binary, "--episode", "--task", "aim", "--aim-frames", str(workdir / "f")],
        env={**os.environ, "SDL_VIDEODRIVER": "dummy"}, capture_output=True, text=True, check=True)
    return [json.loads(l) for l in proc.stdout.splitlines() if l.strip().startswith("{")]


def b64_of(path: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def ask(client, model: str, effort: str, b64: str) -> tuple[bool, str, dict]:
    for attempt in range(8):
        try:
            r = client.messages.create(
                model=model, **thinking_kwargs(model, effort),
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": AIM_PROMPT}]}])
            break
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(2.0 * 2 ** attempt, 60))
    text = "".join(b.text for b in r.content if b.type == "text").strip().upper()
    return "FIRE" in text, text, {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--binary", default=os.environ.get("KEEN_BIN", str(REPO / "raycaster/build/keen-raycaster")))
    ap.add_argument("--workdir", default="/tmp/keen_aim_battery")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tag = "" if a.model == MODEL else "_" + a.model.split("-")[1]
    eff = "" if a.effort == "high" else f"_{a.effort}"
    out = Path(a.out) if a.out else HERE / f"battery_aim_high{tag}{eff}.json"
    workdir = Path(a.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    rows = render(a.binary, workdir)
    client = anthropic.Anthropic(timeout=60.0, max_retries=0)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        b64s = list(pool.map(b64_of, [r["path"] for r in rows]))
        jobs = [(i, s) for i in range(len(rows)) for s in range(a.samples)]
        results = list(pool.map(lambda j: ask(client, a.model, a.effort, b64s[j[0]]), jobs))

    fires = [0] * len(rows)
    texts: dict[int, list[str]] = {}
    usage = {"input_tokens": 0, "output_tokens": 0}
    for (i, _), (f, text, u) in zip(jobs, results):
        fires[i] += int(f)
        texts.setdefault(i, []).append(text)
        for k in usage:
            usage[k] += u[k]
    battery = [{"tick": r["tick"], "aim": r["aim"], "in_window": r["in_window"], "jump": fires[i], "n": a.samples,
                "raw": texts[i]} for i, r in enumerate(rows)]
    out.write_text(json.dumps(battery))

    line = "".join("F" if b["jump"] == b["n"] else ("f" if b["jump"] else ".") for b in battery)
    win = "".join("W" if b["in_window"] else "." for b in battery)
    print(f"model={a.model} effort={a.effort} samples={a.samples} ({len(rows)} frames)")
    print("judge : " + line)
    print("window: " + win)
    early = [(b["tick"], b["jump"]) for b in battery if b["jump"] and not b["in_window"] and b["tick"] < min(x["tick"] for x in battery if x["in_window"])]
    late = [(b["tick"], b["jump"]) for b in battery if b["jump"] and not b["in_window"] and b["tick"] > max(x["tick"] for x in battery if x["in_window"])]
    missed = [(b["tick"], b["jump"]) for b in battery if b["in_window"] and b["jump"] < b["n"]]
    print(f"pre-window FIREs (tick, count): {early}")
    print(f"post-window FIREs (tick, count): {late}")
    print(f"in-window frames not unanimous (tick, count): {missed}")
    print(f"wrote {out}; token usage: {usage}")


if __name__ == "__main__":
    main()
