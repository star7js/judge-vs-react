#!/usr/bin/env bash
# Sweeping-aim task (the generality test for the r~1 timing law). Verifies the
# angular window, the delay-0 hit, and the overshoot transition at r ~ 1.
set -euo pipefail
BIN=./build/keen-raycaster
aim(){ SDL_VIDEODRIVER=dummy $BIN --episode --task aim \
       --omega 0.01 --half-width 0.06 --theta 1.0 --a0 0.5 --delay "$1"; }

# Window is theta +/- h = [0.940, 1.060]; a delay-0 shot hits inside it.
aim 0 | grep -q '"w_lo":0.9400'
aim 0 | grep -q '"w_hi":1.0600'
aim 0 | grep -q '"cleared":1'
aim 0 | grep -q '"in_window":1'

# Transition at r = omega*delay/(2h) ~ 1. Here 2h=0.12, omega=0.01, so r=1 at
# delay=12. Below it the shot still lands in the window; above it, it overshoots.
aim 11 | grep -q '"cleared":1'   # r=0.917 -> hit
aim 12 | grep -q '"fell":1'      # r=1.000 -> overshoot, miss
aim 20 | grep -q '"fell":1'      # r=1.667 -> miss

# A wider target tolerates a longer delay (r scales with 1/window): 2h=0.24,
# so r=1 at delay=24 -> a delay of 20 (r=0.83) still hits.
SDL_VIDEODRIVER=dummy $BIN --episode --task aim --omega 0.01 --half-width 0.12 \
  --theta 1.0 --a0 0.5 --delay 20 | grep -q '"cleared":1'

echo "PASS test_aim"
