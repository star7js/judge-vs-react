#!/usr/bin/env bash
set -euo pipefail
BIN=./build/keen-raycaster
# Oracle with no delay must clear the default pit, deterministically.
A=$(SDL_VIDEODRIVER=dummy $BIN --episode --policy oracle --pit-near 10 --pit-width 2 --seed 1 --delay 0 --repeat 1)
B=$(SDL_VIDEODRIVER=dummy $BIN --episode --policy oracle --pit-near 10 --pit-width 2 --seed 1 --delay 0 --repeat 1)
echo "$A"
test "$A" = "$B"                              # determinism
echo "$A" | grep -q '"cleared":1'             # oracle clears at delay 0
echo "$A" | grep -q '"in_window":1'           # decision was inside the window
echo "PASS test_episode determinism+clear"

# Pin the takeoff window for the default config so a pit_window regression
# can't hide. The window is scanned directly from the sim's own dynamics
# (takeoff_clears): no PLAYER_R inset, since the fall check is a center-point
# tile test and the pit floor is non-solid (never gates on player radius).
echo "$A" | grep -q '"w_lo":8.921'
echo "$A" | grep -q '"w_hi":9.999'
echo "PASS test_episode window-values"

# Fractional pit_near must still align to the integer tile boundary that
# map_load_pit actually uses when placing FLOOR_PIT tiles, so the window
# for --pit-near 10.4 must match the integer --pit-near 10 case exactly.
C=$(SDL_VIDEODRIVER=dummy $BIN --episode --policy oracle --pit-near 10.4 --pit-width 2 --seed 1 --delay 0 --repeat 1)
echo "$C"
echo "$C" | grep -q '"w_lo":8.921'
echo "$C" | grep -q '"w_hi":9.999'
echo "$C" | grep -q '"cleared":1'
echo "$C" | grep -q '"in_window":1'
echo "PASS test_episode fractional-pit-near-alignment"
