#!/usr/bin/env bash
# The rendered aim scene (map_load_aim + --aim-scene / --aim-frames): the
# window is the target tile's angular extent seen from the player, the oracle
# hits through delay 10 and overshoots from 11 (its leading-edge fire sits
# ~0.9 tick inside the near edge), and the frame battery covers the sweep.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./build/keen-raycaster
aim(){ SDL_VIDEODRIVER=dummy $BIN --episode --task aim --aim-scene --delay "$1"; }

aim 0 | grep -q '"w_lo":-1.6296'
aim 0 | grep -q '"w_hi":-1.5120'
aim 0 | grep -q '"cleared":1'
aim 10 | grep -q '"cleared":1'   # r = 0.85 -> hit
aim 11 | grep -q '"fell":1'      # r = 0.94 -> overshoot
aim 20 | grep -q '"fell":1'

tmp=$(mktemp -d)
SDL_VIDEODRIVER=dummy $BIN --episode --task aim --aim-frames "$tmp/f" > "$tmp/manifest.jsonl"
[ "$(wc -l < "$tmp/manifest.jsonl")" -eq 68 ]
[ "$(grep -c '"in_window":1' "$tmp/manifest.jsonl")" -eq 11 ]
[ -f "$tmp/f_045.bmp" ]
rm -rf "$tmp"

# The kinematic oracle path is unchanged by the scene flags.
SDL_VIDEODRIVER=dummy $BIN --episode --task aim --omega 0.01 --half-width 0.06 --theta 1.0 --a0 0.5 --delay 0 | grep -q '"w_lo":0.9400'
echo "PASS test_aim_scene"
