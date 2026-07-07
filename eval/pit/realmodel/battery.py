"""Battery-test a judge prompt against every approach frame.

Sends each battery/*.bmp frame to Sonnet with the given prompt (threaded),
prints a JUMP/WAIT map over px, and flags misfires: JUMP before the window
[8.921, 9.999) or WAIT inside it.
"""
import base64
import io
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from PIL import Image

HERE = Path(__file__).parent
W_LO, W_HI = 8.921, 9.999

PROMPTS = {
    "v2-bottomedge": (
        "This is a first-person view of a game. You are auto-walking toward a pit "
        "(the BLACK region on the floor). You must JUMP over it to reach the green "
        "beacon. The checkered floor visible in the middle of the image is the FAR "
        "side of the pit, not where you stand. "
        "Answer in exactly two lines.\n"
        "Line 1: Does the black region extend all the way down to the very bottom "
        "edge of the image? Answer yes or no.\n"
        "Line 2: If yes, say JUMP. If no, say WAIT."),
    "v3-floorbelow": (
        "This is a first-person view of a game. You are auto-walking toward a pit "
        "you must JUMP over to reach the green beacon. The pit is PURE BLACK "
        "(no texture at all). The floor you walk on is dark-gray CHECKERED tiles "
        "- checkered tiles are floor, not pit, no matter how dark. "
        "Answer in exactly two lines.\n"
        "Line 1: At the very bottom edge of the image (the strip right in front "
        "of your feet), do you see checkered floor tiles, or pure untextured "
        "black? Answer floor or black.\n"
        "Line 2: If floor, say WAIT. If black, say JUMP."),
}

client = anthropic.Anthropic()


def b64_of(p):
    img = Image.open(p).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def ask(prompt, b64):
    r = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1000,
        thinking={"type": "adaptive"}, output_config={"effort": "medium"},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/png", "data": b64}},
            {"type": "text", "text": prompt}]}])
    text = "".join(b.text for b in r.content if b.type == "text").strip().upper()
    return "JUMP" in text


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "v3-floorbelow"
    prompt = PROMPTS[name]
    frames = sorted(HERE.glob("battery/f_*.bmp"))
    pxs = [float(re.search(r"px(\d+\.\d+)", f.name).group(1)) for f in frames]

    with ThreadPoolExecutor(max_workers=8) as pool:
        b64s = list(pool.map(b64_of, frames))
        answers = list(pool.map(lambda b: ask(prompt, b), b64s))

    line = "".join("J" if a else "." for a in answers)
    print(f"prompt={name}  ({len(frames)} frames, px {pxs[0]}..{pxs[-1]})")
    print(line)
    early = [(px, f.name) for px, f, a in zip(pxs, frames, answers)
             if a and px < W_LO]
    misses = [px for px, a in zip(pxs, answers) if not a and W_LO <= px < W_HI]
    first_jump = next((px for px, a in zip(pxs, answers) if a), None)
    print(f"early JUMPs (<{W_LO}): {[e[0] for e in early]}")
    print(f"WAITs inside window: {misses}")
    print(f"first JUMP at px={first_jump}")


if __name__ == "__main__":
    main()
