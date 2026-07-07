#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."            # raycaster/
BIN=./build/keen-raycaster
OUT=$(mktemp)
SDL_VIDEODRIVER=dummy $BIN --episode --policy agent \
  --pit-near 10 --pit-width 2 --seed 1 --delay 0 --repeat 1 >"$OUT" 2>/dev/null &
GAME=$!
python3 pit_agent.py >/dev/null 2>&1 || true      # rules backend, connects, drives jump
wait $GAME 2>/dev/null || true
cat "$OUT"
grep -q '"policy":"agent"' "$OUT"
grep -q '"cleared":1' "$OUT"                        # rules agent clears the pit at delay 0
grep -q '"in_window":1' "$OUT"                      # decision_x~9.01 is correctly in-window now
echo "PASS test_agent_episode"
