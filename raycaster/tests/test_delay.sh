#!/usr/bin/env bash
set -euo pipefail
BIN=./build/keen-raycaster
run(){ SDL_VIDEODRIVER=dummy $BIN --episode --policy oracle --pit-near 10 --pit-width 2 --seed 1 --delay "$1" --repeat "${2:-1}"; }
run 0  | grep -q '"cleared":1'      # no delay -> clear
run 20 | grep -q '"fell":1'         # 20-tick delay (~1.4 tiles) -> fall
# action-repeat does NOT rescue a large delay:
run 20 6 | grep -q '"fell":1'

# Repeat changes the consult cadence: decision_tick snaps to the next multiple of `repeat`.
# These FAIL if `repeat` gating is reverted to a no-op (decision_tick would stay 91).
run(){ SDL_VIDEODRIVER=dummy $BIN --episode --policy oracle --pit-near 10 --pit-width 2 --seed 1 --delay 0 --repeat "$1"; }
run 1  | grep -q '"decision_tick":91'
run 6  | grep -q '"decision_tick":96'
run 50 | grep -q '"decision_tick":100'
echo "PASS test_delay repeat-discriminates"

echo "PASS test_delay"
