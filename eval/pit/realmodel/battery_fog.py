"""Frame battery for the S5 judge at a chosen fog level.

Renders every tick-grid approach frame (x = 2.57 .. 10.06, step 0.07; the
108 frames the final judge was validated on at fog 0) at the given fog, asks
the S5 judge (sweep_agent.VISION_PROMPT, high effort) `--samples` times per
frame, and writes battery_v4_high_fog{F}.json in the same schema as
battery_v4_high.json: [{"px", "jump", "n"}, ...]. Prints the JUMP map, the
early/missed frames, and token usage so cost is auditable.

This is the gate before a fog sweep: if the judge never answers JUMP inside
the window under fog, the sweep would measure a perception failure at every
delay and a shorter grid suffices.

Usage (repo root, ANTHROPIC_API_KEY set):
    python3 eval/pit/realmodel/battery_fog.py --fog 1.0 --samples 3
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
from sweep_agent import VISION_PROMPT  # noqa: E402  (the S5 judge's prompt)

W_LO, W_HI = 8.921, 9.999
PIT_NEAR = 10.0
X0, STEP, N_FRAMES = 2.57, 0.07, 108
MODEL = "claude-sonnet-4-6"


def render(binary: str, workdir: Path, fog: float) -> list[dict]:
    pxs = [round(X0 + k * STEP, 2) for k in range(N_FRAMES)]
    offs = ",".join(f"{px - PIT_NEAR:.3f}" for px in pxs)
    proc = subprocess.run(
        [binary, "--shoot", str(workdir / f"f_fog{fog:.0f}"), "--pit-width", "2",
         "--fog", repr(fog), "--offsets", offs],
        env={**os.environ, "SDL_VIDEODRIVER": "dummy"}, capture_output=True, text=True, check=True,
    )
    rows = [json.loads(l) for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    assert len(rows) == N_FRAMES, len(rows)
    for r, px in zip(rows, pxs):
        r["px"] = px
    return rows


def b64_of(path: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def ask(client, b64: str) -> tuple[bool, dict]:
    for attempt in range(8):
        try:
            r = client.messages.create(
                model=MODEL, max_tokens=1500,
                thinking={"type": "adaptive"}, output_config={"effort": "high"},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": VISION_PROMPT}]}],
            )
            break
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(2.0 * 2 ** attempt, 60))
    text = "".join(b.text for b in r.content if b.type == "text").strip().upper()
    return "JUMP" in text, {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fog", type=float, required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--binary", default=os.environ.get("KEEN_BIN", str(REPO / "raycaster/build/keen-raycaster")))
    ap.add_argument("--workdir", default="/tmp/keen_battery_fog")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out) if a.out else HERE / f"battery_v4_high_fog{a.fog:.0f}.json"
    workdir = Path(a.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    rows = render(a.binary, workdir, a.fog)
    client = anthropic.Anthropic()

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        b64s = list(pool.map(b64_of, [r["path"] for r in rows]))
        jobs = [(i, s) for i in range(len(rows)) for s in range(a.samples)]
        results = list(pool.map(lambda j: ask(client, b64s[j[0]]), jobs))

    jumps = [0] * len(rows)
    usage = {"input_tokens": 0, "output_tokens": 0}
    for (i, _), (j, u) in zip(jobs, results):
        jumps[i] += int(j)
        for k in usage:
            usage[k] += u[k]
    battery = [{"px": r["px"], "jump": jumps[i], "n": a.samples} for i, r in enumerate(rows)]
    out.write_text(json.dumps(battery))

    line = "".join("J" if b["jump"] == b["n"] else ("j" if b["jump"] else ".") for b in battery)
    print(f"fog={a.fog} model={MODEL} effort=high samples={a.samples} ({len(rows)} frames)")
    print(line)
    early = [(b["px"], b["jump"]) for b in battery if b["jump"] and b["px"] < W_LO]
    missed = [(b["px"], b["jump"]) for b in battery if b["jump"] < b["n"] and W_LO <= b["px"] < W_HI]
    first = next((b["px"] for b in battery if b["jump"]), None)
    print(f"pre-window JUMPs (px, count): {early}")
    print(f"in-window frames not unanimous (px, count): {missed}")
    print(f"first JUMP at px={first}; wrote {out}")
    print(f"token usage: {usage}")


if __name__ == "__main__":
    main()
