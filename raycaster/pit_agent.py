#!/usr/bin/env python3
"""Pit-jump agent for the raycaster.

Connects to the raycaster's Unix socket, reads game state, and decides when to
jump. The player auto-walks forward; the only decision is jump-or-not.

Backends:
  rules   — jump when the pit's near edge is ~1 tile ahead (reads pit_dist)
  vision  — Claude judges the jump from the *rendered frame alone* (the honest
            test: no pit_dist, same handicapped view a human gets). The game
            must run with --frame-out <path> so a fresh frame is on disk each
            decision; this agent reads it, converts BMP->PNG, and asks Claude.
  ollama  — same honest frame-only test as `vision`, but judged by a LOCAL
            open VLM served by Ollama (no API key, real on-device latency).
            Needs a vision model pulled, e.g. `ollama pull llama3.2-vision:11b`.

--latency simulates a slow decision (rules backend only), to test how the
real-time loop tolerates lag. The vision backend has real latency — run it in
BLOCKING mode (game --agent, no --realtime) so lag becomes slow-mo, not a fall.

Usage:
  # rules, blocking:
  #   ./build/keen-raycaster --agent
  python3 pit_agent.py

  # vision, blocking (recommended): start the game exporting frames, then:
  #   ./build/keen-raycaster --agent --frame-out /tmp/keen_ray_frame.bmp
  uv run --with anthropic --with pillow python3 pit_agent.py \
      --backend vision --model claude-haiku-4-5
"""
import argparse
import json
import socket
import time

SOCK = "/tmp/keen_ray_agent.sock"

# The vision model plays from the image alone — describe the task, ask for a
# single-word verdict so the reply is tiny (cheap + low latency).
VISION_PROMPT = (
    "This is a first-person view of a game. You are auto-walking forward toward "
    "a pit you must JUMP to clear, then reach the green exit beacon beyond it. "
    "The pit is the black band across the floor ahead. Jump too early and you "
    "land short inside the pit; too late and you walk into it. Judge the depth: "
    "is NOW the right moment to jump to clear the pit? "
    "Answer with exactly one word: JUMP or WAIT."
)


def rules_decide(st: dict) -> bool:
    # Jump when grounded and the pit's near edge is within ~1 tile. Earlier than
    # that lands short inside the pit; the clear window is pit_dist in (0, 1].
    pd = st["pit_dist"]
    return st["on_ground"] == 1 and 0.0 < pd <= 1.0


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
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            # Opus/Sonnet-4.6+ can't disable thinking; low effort keeps the
            # per-frame call fast and cheap for a reflex yes/no.
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},
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
        decide = make_vision_decider(a.frame, a.model, a.latency_log)
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
