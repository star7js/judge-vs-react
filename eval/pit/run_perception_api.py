"""Blind single-frame perception battery through the API (replaces the n=10
subagent-judged pilot in `perception_judgments.jsonl`).

Renders a balanced set of standing frames at pit width 2 for each fog level
(half in-window, half out-of-window, at distinct positions), shuffles them
into neutral names with a fixed seed, and asks each judge for a one-word
JUMP/WAIT verdict from the image alone -- no coordinates, no window, no
labels. Each call is scored against the sim's own `in_window` label.

Judges: hosted Claude models via the Messages API (adaptive thinking, medium
effort, temperature default) and, if reachable, a local Ollama vision model
(temperature 0). Output rows use the original file's schema plus the exact
model id, prompt tag, latency and token usage, so cost is auditable.

Usage (from the repo root, ANTHROPIC_API_KEY set):
    python3 -m eval.pit.run_perception_api --out eval/pit/data/perception_judgments_v2.jsonl
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PIT_NEAR = 10.0
PIT_WIDTH = 2
N_PER_SIDE = 50
# Out-of-window positions span the far approach up to just before the window
# opens (w_lo = 8.921); in-window positions span the window itself.
OUT_RANGE = (6.00, 8.85)
IN_RANGE = (8.93, 9.99)
FOGS = (0.0, 1.0)
SEED = 42

HOSTED = {
    "claude-haiku-4-5 (API)": "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6 (API)": "claude-sonnet-4-6",
    "claude-opus-4-8 (API)": "claude-opus-4-8",
}
OLLAMA_MODEL = "qwen2.5vl:7b"
OLLAMA_LABEL = "qwen2.5-vl-7b (local, Ollama)"

PROMPTS = {
    # Neutral: describes the task, says nothing about what the pit looks like
    # or which part of the frame to read. Measures unprompted depth judgment.
    "blind-v1": (
        "This is a single first-person frame from a game. You are auto-walking "
        "forward toward a pit in the floor that you must jump over to reach the "
        "green beacon beyond it. If you jump too early you land short, inside the "
        "pit; if you jump too late you walk into it. Judging only from this image, "
        "is this the moment to jump -- would a jump from exactly where you are "
        "standing now clear the pit?\n"
        "Answer with exactly one word: JUMP or WAIT."
    ),
    # Cue-naming: the S5 judge's prompt (sweep_agent.VISION_PROMPT), which
    # tells the model exactly which pixels decide it. Measures judgment once
    # the cue is pointed out. Still blind to coordinates and labels.
    "cue-v1": None,  # filled in at import from sweep_agent
}


def _load_cue_prompt() -> str:
    here = Path(__file__).resolve().parent / "realmodel"
    sys.path.insert(0, str(here))
    from sweep_agent import VISION_PROMPT  # noqa: E402
    return VISION_PROMPT


PROMPTS["cue-v1"] = _load_cue_prompt()
PROMPT_TAG = "blind-v1"
PROMPT = PROMPTS[PROMPT_TAG]


def positions() -> list[float]:
    out = [OUT_RANGE[0] + i * (OUT_RANGE[1] - OUT_RANGE[0]) / (N_PER_SIDE - 1) for i in range(N_PER_SIDE)]
    inn = [IN_RANGE[0] + i * (IN_RANGE[1] - IN_RANGE[0]) / (N_PER_SIDE - 1) for i in range(N_PER_SIDE)]
    return [round(x, 3) for x in out + inn]


def render(binary: str, workdir: Path, fog: float) -> list[dict]:
    """Render one frame per position at `fog`; return the sim's manifest rows."""
    offs = ",".join(f"{px - PIT_NEAR:.3f}" for px in positions())
    prefix = str(workdir / f"fog{fog:.0f}")
    proc = subprocess.run(
        [binary, "--shoot", prefix, "--pit-width", str(PIT_WIDTH), "--fog", repr(fog),
         "--offsets", offs],
        env={**os.environ, "SDL_VIDEODRIVER": "dummy"}, capture_output=True, text=True, check=True,
    )
    rows = [json.loads(l) for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    assert len(rows) == 2 * N_PER_SIDE, len(rows)
    return rows


def png_b64(path: str) -> str:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


EFFORT = "medium"


def thinking_kwargs(model: str) -> dict:
    """Per-model thinking configuration. Haiku 4.5 predates adaptive thinking
    and takes a token budget (effort does not apply); the 4.6+ models take
    adaptive thinking + EFFORT."""
    if model.startswith("claude-haiku-4-5"):
        return {"thinking": {"type": "enabled", "budget_tokens": 2048}, "max_tokens": 4096}
    return {"thinking": {"type": "adaptive"}, "output_config": {"effort": EFFORT}, "max_tokens": 1500}


def ask_claude(client, model: str, b64: str) -> tuple[str, float, dict]:
    t0 = time.time()
    for attempt in range(8):
        try:
            r = client.messages.create(
                model=model, **thinking_kwargs(model),
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": PROMPT}]}],
            )
            break
        except Exception as exc:  # rate limits / transient errors
            if attempt == 7:
                raise
            time.sleep(min(2.0 * 2 ** attempt, 60))
    text = "".join(b.text for b in r.content if b.type == "text").strip()
    usage = {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}
    return text, time.time() - t0, usage


def ask_ollama(model: str, b64: str, url: str = "http://localhost:11434") -> tuple[str, float]:
    payload = json.dumps({"model": model, "stream": False, "options": {"temperature": 0},
                          "messages": [{"role": "user", "content": PROMPT, "images": [b64]}]}).encode()
    req = urllib.request.Request(url + "/api/chat", data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
    return (body.get("message", {}).get("content") or "").strip(), time.time() - t0


def ollama_available(model: str) -> bool:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            tags = [m["name"] for m in json.loads(r.read())["models"]]
        return model in tags
    except Exception:
        return False


def verdict(text: str) -> str | None:
    t = text.strip().upper()
    has_j, has_w = "JUMP" in t, "WAIT" in t
    if has_j and not has_w:
        return "JUMP"
    if has_w and not has_j:
        return "WAIT"
    # Both or neither: take the last word mentioned, else unparseable.
    for tok in reversed(t.replace(".", " ").split()):
        if tok in ("JUMP", "WAIT"):
            return tok
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/pit/data/perception_judgments_v2.jsonl")
    ap.add_argument("--binary", default=os.environ.get("KEEN_BIN", "raycaster/build/keen-raycaster"))
    ap.add_argument("--workdir", default="/tmp/keen_perception_v2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-ollama", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="subset of judge labels to run")
    ap.add_argument("--prompt", default="blind-v1", choices=sorted(PROMPTS))
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"],
                    help="adaptive-thinking effort for the 4.6+ hosted models (Haiku 4.5 ignores it)")
    a = ap.parse_args()
    global PROMPT, PROMPT_TAG, EFFORT
    PROMPT_TAG, PROMPT, EFFORT = a.prompt, PROMPTS[a.prompt], a.effort

    workdir = Path(a.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out = Path(a.out)

    # One shuffled, neutrally named frame set per fog level (same seed as the pilot).
    frames = []
    for fog in FOGS:
        rows = render(a.binary, workdir, fog)
        rng = random.Random(SEED)
        rng.shuffle(rows)
        for i, r in enumerate(rows):
            neutral = workdir / f"frame_fog{fog:.0f}_{i:03d}.bmp"
            os.replace(r["path"], neutral)
            frames.append({**r, "path": str(neutral), "fog": fog})
    print(f"[perception] {len(frames)} frames rendered "
          f"({sum(r['in_window'] for r in frames)} in-window)", flush=True)
    b64s = {f["path"]: png_b64(f["path"]) for f in frames}

    done = set()
    if out.exists():
        for line in open(out):
            if line.strip():
                r = json.loads(line)
                done.add((r["judge"], r.get("prompt", "blind-v1"), r.get("effort", "medium"),
                          r["fog"], r["px"]))

    judges: list[tuple[str, str]] = [(lbl, mid) for lbl, mid in HOSTED.items()]
    if not a.skip_ollama:
        if ollama_available(OLLAMA_MODEL):
            judges.append((OLLAMA_LABEL, OLLAMA_MODEL))
        else:
            print(f"[perception] Ollama model {OLLAMA_MODEL} not available; skipping local judge", flush=True)
    if a.only:
        judges = [(l, m) for l, m in judges if l in a.only]

    client = None
    if any("API" in l for l, _ in judges):
        import anthropic
        client = anthropic.Anthropic()

    total_usage = {"input_tokens": 0, "output_tokens": 0}
    for label, model in judges:
        effort = EFFORT if ("API" in label and not model.startswith("claude-haiku-4-5")) else "n/a"
        todo = [f for f in frames if (label, PROMPT_TAG, effort, f["fog"], f["px"]) not in done]
        print(f"[perception] {label} prompt={PROMPT_TAG} effort={effort}: {len(todo)} calls", flush=True)

        def one(f):
            b64 = b64s[f["path"]]
            if "API" in label:
                text, dt, usage = ask_claude(client, model, b64)
            else:
                text, dt = ask_ollama(model, b64)
                usage = None
            v = verdict(text)
            return {
                "judge": label, "model": model, "prompt": PROMPT_TAG, "effort": effort,
                "pit_width": f["pit_width"], "fog": f["fog"], "px": f["px"],
                "in_window": f["in_window"],
                "decision": v or "UNPARSED", "jump": int(v == "JUMP"),
                "correct": int((v == "JUMP") == bool(f["in_window"])) if v else 0,
                "raw": text[:200], "latency_s": round(dt, 3), "usage": usage,
            }

        workers = a.workers if "API" in label else 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(one, todo):
                with open(out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                if row["usage"]:
                    for k in total_usage:
                        total_usage[k] += row["usage"][k]
        rows = [json.loads(l) for l in open(out) if l.strip()]
        for fog in FOGS:
            g = [r for r in rows if r["judge"] == label and r["fog"] == fog
                 and r.get("prompt", "blind-v1") == PROMPT_TAG and r.get("effort", "medium") == effort]
            acc = sum(r["correct"] for r in g) / max(len(g), 1)
            jumps = sum(r["jump"] for r in g)
            print(f"[perception]   {label} fog={fog:.0f}: acc={acc:.0%} (n={len(g)}, JUMP on {jumps})", flush=True)
    print(f"[perception] hosted token usage: {total_usage}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
