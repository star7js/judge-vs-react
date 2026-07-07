#!/usr/bin/env python3
"""S5 judge: the revised pit-jump agent used for the real-model boundary
sweep (paper section 4.6). A modified copy of raycaster/pit_agent.py; the
shipped agent is unchanged. Differences, all documented in the paper:

  * VISION_PROMPT asks the literal bottom-edge question ("what touches the
    bottom border of the image: tiles or black?") -- the shipped prompt's
    "black band across the floor ahead" stops matching close-range frames and
    elicits no JUMPs near the window from any hosted model.
  * High reasoning effort (the shipped agent uses low): needed to judge the
    thin floor sliver in the 8.1-8.9 transition zone reliably.
  * Two-consecutive-JUMP commit rule: rare stochastic false JUMPs (~2%/call
    far from the pit) would otherwise fire almost surely somewhere over the
    ~90 consulted approach frames.
  * API calls retry with backoff (a dead agent mid-episode silently walks the
    player into the pit).
  * Socket path honors KEEN_AGENT_SOCK so parallel episodes don't collide.

Driven by boundary_sweep.py; validated frame-by-frame by battery.py
(battery_v4_high.json: 108 approach frames x 3 samples, zero far-field JUMPs,
unanimous JUMP inside the window).
"""
import argparse
import json
import os
import socket
import time

SOCK = os.environ.get("KEEN_AGENT_SOCK", "/tmp/keen_ray_agent.sock")

# The vision model plays from the image alone — describe the task, ask for a
# single-word verdict so the reply is tiny (cheap + low latency).
VISION_PROMPT = (
    "This is a first-person view of a game. You are auto-walking toward a pit "
    "you must JUMP over to reach the green beacon. The pit is PURE BLACK (a "
    "solid black area with no texture). The floor is dark-gray CHECKERED tiles. "
    "Decide using ONLY the very bottom border of the image: "
    "look at the thin horizontal strip touching the bottom edge of the frame.\n"
    "If that strip shows checkered floor tiles - even a narrow sliver of tiles "
    "below a black area - the pit has not reached you yet.\n"
    "Answer in exactly two lines.\n"
    "Line 1: What touches the bottom border of the image: tiles or black?\n"
    "Line 2: If tiles, say WAIT. If black, say JUMP."
)


def rules_decide(st: dict) -> bool:
    # Jump when grounded and the pit's near edge is within ~1 tile. Earlier than
    # that lands short inside the pit; the clear window is pit_dist in (0, 1].
    pd = st["pit_dist"]
    return st["on_ground"] == 1 and 0.0 < pd <= 1.0



def _create_with_retry(client, **kwargs):
    """Retry API calls with backoff. The sim clock is decoupled from
    wall-clock, so blocking here is free; a dead agent mid-episode is not."""
    delay_s = 2.0
    for attempt in range(10):
        try:
            return client.messages.create(**kwargs)
        except Exception as exc:
            if attempt == 9:
                raise
            print(f"  [vision] API error (attempt {attempt+1}): {exc} — "
                  f"retrying in {delay_s:.0f}s", flush=True)
            time.sleep(delay_s)
            delay_s = min(delay_s * 2, 60.0)

def make_vision_decider(frame_path: str, model: str, latency_log: str | None = None):
    """Build a decide(st)->bool backed by Claude vision. Frame comes from disk
    (the game's --frame-out), never from pit_dist — the view is the only input.

    If `latency_log` is set, appends one JSON line per vision decision:
    `{"tick", "px", "latency_s", "action"}`."""
    import base64
    import io

    import anthropic
    from PIL import Image

    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from the environment
    print(f"[pit-agent] vision model={model} frame={frame_path}", flush=True)

    def decide(st: dict) -> bool:
        if st["on_ground"] != 1:
            return False  # mid-jump — nothing to decide, save the API call
        try:
            with open(frame_path, "rb") as f:
                raw = f.read()
        except FileNotFoundError:
            return False  # no frame exported yet
        # Claude vision takes png/jpeg/gif/webp, not BMP — convert in memory.
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

        t0 = time.time()
        resp = _create_with_retry(
            client,
            model=model,
            max_tokens=1500,
            # Opus/Sonnet-4.6+ can't disable thinking; low effort keeps the
            # per-frame call fast and cheap for a reflex yes/no.
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }],
        )
        dt = time.time() - t0
        text = "".join(b.text for b in resp.content if b.type == "text").strip().upper()
        jump = "JUMP" in text
        print(f"  [vision] {dt:4.2f}s px={st['px']:.2f} -> {text!r}"
              f"{'  *** JUMP ***' if jump else ''}", flush=True)
        if latency_log:
            with open(latency_log, "a") as f:
                f.write(json.dumps({
                    "tick": st["tick"], "px": st["px"],
                    "latency_s": round(dt, 4),
                    "action": "jump" if jump else "none",
                }) + "\n")
        return jump

    return decide


def _frame_to_png_b64(frame_path: str):
    """Read the game's exported frame (BMP) and return base64 PNG, or None if
    no frame is on disk yet. Shared by the vision and ollama backends."""
    import base64
    import io

    from PIL import Image

    try:
        with open(frame_path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return None
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def make_ollama_decider(frame_path: str, model: str, base_url: str,
                        latency_log: str | None = None):
    """Build a decide(st)->bool backed by a local Ollama vision model. Same
    frame-only handicap as the Claude vision backend, but the call goes to
    Ollama's HTTP API on localhost — no API key, real on-device latency.

    Latency logged (per decision) is the wall-clock of the closed-loop call:
    `{"tick", "px", "latency_s", "action"}` — identical schema to the Claude
    backend so `run_vlm.mean_latency` reads either."""
    import json as _json
    import urllib.request

    url = base_url.rstrip("/") + "/api/chat"
    print(f"[pit-agent] ollama model={model} url={url} frame={frame_path}", flush=True)

    def _ask(b64: str) -> str:
        payload = _json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": VISION_PROMPT, "images": [b64]}],
            "stream": False,
            "options": {"temperature": 0},
        }).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = _json.loads(r.read())
        return (body.get("message", {}).get("content") or "").strip().upper()

    # Warm up so the logged latencies reflect steady state, not the one-time
    # model load (Ollama keeps the model resident between calls).
    try:
        _ask(_frame_to_png_b64(frame_path) or "")
    except Exception:
        pass

    def decide(st: dict) -> bool:
        if st["on_ground"] != 1:
            return False  # mid-jump — nothing to decide
        b64 = _frame_to_png_b64(frame_path)
        if b64 is None:
            return False  # no frame exported yet
        t0 = time.time()
        try:
            text = _ask(b64)
        except Exception as exc:
            print(f"  [ollama] call failed: {exc}", flush=True)
            return False
        dt = time.time() - t0
        jump = "JUMP" in text
        print(f"  [ollama] {dt:4.2f}s px={st['px']:.2f} -> {text!r}"
              f"{'  *** JUMP ***' if jump else ''}", flush=True)
        if latency_log:
            with open(latency_log, "a") as f:
                f.write(json.dumps({
                    "tick": st["tick"], "px": st["px"],
                    "latency_s": round(dt, 4),
                    "action": "jump" if jump else "none",
                }) + "\n")
        return jump

    return decide


def read_latest_state(sock: socket.socket, buf: bytearray):
    """Block for at least one complete line, then drain to the freshest state."""
    while b"\n" not in buf:
        d = sock.recv(8192)
        if not d:
            return None, buf
        buf += d
    # Non-blocking drain of anything else already queued (get the newest frame).
    sock.setblocking(False)
    try:
        while True:
            d = sock.recv(8192)
            if not d:
                break
            buf += d
    except (BlockingIOError, OSError):
        pass
    sock.setblocking(True)

    lines = buf.split(b"\n")
    buf = bytearray(lines[-1])
    complete = [l for l in lines[:-1] if l.strip()]
    if not complete:
        return None, buf
    return json.loads(complete[-1]), buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="rules", choices=["rules", "vision", "ollama"])
    ap.add_argument("--model", default="claude-opus-4-8",
                    help="vision model. vision backend: try claude-haiku-4-5; "
                         "ollama backend: e.g. llama3.2-vision:11b")
    ap.add_argument("--ollama-url", default="http://localhost:11434",
                    help="ollama backend: base URL of the local Ollama server")
    ap.add_argument("--frame", default="/tmp/keen_ray_frame.bmp",
                    help="path the game writes frames to (--frame-out)")
    ap.add_argument("--latency", type=float, default=0.0,
                    help="rules backend: seconds per decision (simulate a slow call)")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop after N seconds and print a summary (0 = until won)")
    ap.add_argument("--latency-log", default=None,
                    help="vision backend: append one JSON line per decision "
                         "with the measured API call latency (rules backend "
                         "makes no API call, so nothing is logged)")
    a = ap.parse_args()

    if a.backend == "vision":
        raw_decide = make_vision_decider(a.frame, a.model, a.latency_log)

        # Commit rule: fire only on two consecutive JUMP answers. A single
        # stochastic JUMP (~2% per call far from the pit) would otherwise
        # almost surely fire early somewhere over the ~90 approach ticks.
        _streak = {"n": 0}

        def decide(st):
            if raw_decide(st):
                _streak["n"] += 1
            else:
                _streak["n"] = 0
            return _streak["n"] >= 2
    elif a.backend == "ollama":
        decide = make_ollama_decider(a.frame, a.model, a.ollama_url, a.latency_log)
    else:
        decide = rules_decide

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for attempt in range(50):  # wait for the game to start listening
        try:
            sock.connect(SOCK)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.1)
    else:
        print(f"[pit-agent] could not connect to {SOCK} — is the game running with --agent?", flush=True)
        return
    print(f"[pit-agent] backend={a.backend} latency={a.latency}s", flush=True)

    buf = bytearray()
    jumps = 0
    while True:
        st, buf = read_latest_state(sock, buf)
        if st is None:
            print("[pit-agent] game closed", flush=True)
            return

        if a.latency > 0:
            time.sleep(a.latency)

        jump = decide(st)
        if jump:
            jumps += 1
            if a.backend == "rules":
                print(f"  px={st['px']:.2f} pit_dist={st['pit_dist']:.2f} "
                      f"on_ground={st['on_ground']} -> JUMP", flush=True)
        sock.sendall((json.dumps({"action": "jump" if jump else "none",
                                  "tick": st["tick"]}) + "\n").encode())

        if st.get("won"):
            print(f"[pit-agent] WON — cleared the pit (jumps={jumps})", flush=True)
            return


if __name__ == "__main__":
    main()
