# First-person pit demo clips

Two matched clips of the AI playing the raycaster pit task in first person — the
visual hook for the *Judge vs. React* writeup. They show the paper's core point
qualitatively: **the AI's failure is latency, not judgment.**

| Clip | Mode | Outcome |
|---|---|---|
| `ai_judge_vs_react.mp4` | **Side-by-side** (the two clips below, labelled) | The hero clip for the blog — same agent, paused clears vs. real-time falls |
| `ai_clears_pit.mp4` | Blocking (world waits for each decision) | **Clears** — jumps at the right moment, reaches the exit beacon |
| `ai_misses_pit_latency.mp4` | Real-time 60fps + 0.2s decision lag | **Misses** — makes the *same* correct call, but the lag lands it in the pit (repeatedly) |

`ai_judge_vs_react.gif` is committed (an exception to the `*.gif` ignore) so it
renders inline in `paper/blog.md` on GitHub.

Both use the rules agent (`pit_agent.py`, no API key). The same decision that wins
in blocking mode fails in real time once the 0.2s lag exceeds the ~1-tile jump
window — the on-screen version of the `r ≈ 1` transition measured in
`eval/pit/figures/fig_timing.png`.

`.mp4` files are committed (the shareable deliverable). `.gif` previews are
larger and regenerable, so they are git-ignored — regenerate them with the
palette commands below if you want inline previews.

## Regeneration

These clips are recorded from the **on-screen / real-time** game path (`--agent`,
with `--record` dumping frames), not the headless `--episode` sim — the demo
wants the real-time feel and 60fps vsync. Build first: `cmake --build build`.

### Clear (blocking)
```bash
rm -rf /tmp/rec && mkdir -p /tmp/rec
SDL_VIDEODRIVER=dummy ./build/keen-raycaster --agent --record /tmp/rec/f &   # headless, blocking
python3 pit_agent.py                                                          # rules agent clears the pit; game auto-exits ~40 frames after the win
ffmpeg -y -framerate 30 -i /tmp/rec/f_%05d.bmp \
  -vf "scale=iw*3:ih*3:flags=neighbor" -pix_fmt yuv420p demo/ai_clears_pit.mp4
```

### Miss (real-time + latency)
```bash
rm -rf /tmp/rec2 && mkdir -p /tmp/rec2
./build/keen-raycaster --agent --realtime --record /tmp/rec2/f &             # on-screen, 60fps vsync
python3 pit_agent.py --latency 0.2                                            # 0.2s lag per decision
sleep 14; pkill -f keen-raycaster; pkill -f pit_agent                         # ~14s captures several fall/reset cycles
ffmpeg -y -framerate 60 -i /tmp/rec2/f_%05d.bmp \
  -vf "scale=iw*3:ih*3:flags=neighbor" -pix_fmt yuv420p demo/ai_misses_pit_latency.mp4
```

### Side-by-side (`ai_judge_vs_react.mp4` + committed `.gif`)
Composited from the two clips above with a label banner (rendered via Pillow,
since this ffmpeg build lacks `drawtext`). Loops the shorter "clear" clip to
match the "miss" length.
```bash
# 1. render the banner (paper/blog uses it as the hero); needs Pillow
#    (a small script draws the two labels + a green check / red cross)
# 2. composite: loop clears, downsample misses to 30fps, hstack, banner on top
ffmpeg -y -stream_loop -1 -i ai_clears_pit.mp4 -i ai_misses_pit_latency.mp4 -i _banner.png \
  -filter_complex "[0:v]fps=30,scale=560:420:flags=neighbor[L];\
    [1:v]fps=30,scale=560:420:flags=neighbor[R];[L][R]hstack=inputs=2[vid];\
    [2:v][vid]vstack=inputs=2[v]" -map "[v]" -t 12 -r 30 -pix_fmt yuv420p ai_judge_vs_react.mp4
# 3. hero gif (committed): 12fps, 620px
ffmpeg -y -i ai_judge_vs_react.mp4 -vf "fps=12,scale=620:-1:flags=lanczos,palettegen=stats_mode=diff" _pal.png
ffmpeg -y -i ai_judge_vs_react.mp4 -i _pal.png \
  -lavfi "fps=12,scale=620:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" ai_judge_vs_react.gif
```

### GIF previews (optional, git-ignored)
```bash
# example for the clear clip; swap paths/framerate for the miss clip
cd /tmp/rec
ffmpeg -y -framerate 30 -i f_%05d.bmp -vf "scale=iw*2:ih*2:flags=neighbor,palettegen" pal.png
ffmpeg -y -framerate 30 -i f_%05d.bmp -i pal.png \
  -lavfi "scale=iw*2:ih*2:flags=neighbor[x];[x][1:v]paletteuse" \
  <repo>/raycaster/demo/ai_clears_pit.gif
```

Physics: `MOVE_SPEED=0.07`, `JUMP_V0=0.11`, `GRAVITY=0.005`, 60fps → a 0.2s
decision costs ~0.84 tiles of travel, about one jump window; see the design doc
`docs/superpowers/specs/2026-07-06-judge-vs-react-design.md` for the full framing.
