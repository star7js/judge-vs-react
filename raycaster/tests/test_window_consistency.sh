#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."            # raycaster/
BIN=./build/keen-raycaster

# Regression guard for the apparatus-correctness bug: the pre-fix analytic
# window inset both edges by PLAYER_R even though the fall check is a pure
# center-point tile test (the pit floor is non-solid, so PLAYER_R never
# gates movement over it). That made in_window a too-narrow, directionally
# biased label — e.g. the rules agent's decision_x~=9.01 cleared the pit but
# was reported in_window:0. in_window must instead be derived from the sim's
# own dynamics (takeoff_clears), so it can never disagree with `cleared`
# for a delay-0 decision.
OUT=$(mktemp)
SDL_VIDEODRIVER=dummy $BIN --episode --policy agent \
  --pit-near 10 --pit-width 2 --seed 1 --delay 0 --repeat 1 >"$OUT" 2>/dev/null &
GAME=$!
python3 pit_agent.py >/dev/null 2>&1 || true
wait $GAME 2>/dev/null || true
cat "$OUT"
grep -q '"cleared":1' "$OUT"
grep -q '"in_window":1' "$OUT"    # the key guard: this was wrongly 0 before the fix

# Check the window bounds directly (not just the pinned constants below) so
# this also fails if PLAYER_R insetting is reintroduced under a different
# pit config than the pinned default.
DX=$(grep -o '"decision_x":[0-9.]*' "$OUT" | cut -d: -f2)
WLO=$(grep -o '"w_lo":[0-9.]*' "$OUT" | cut -d: -f2)
WHI=$(grep -o '"w_hi":[0-9.]*' "$OUT" | cut -d: -f2)
awk -v dx="$DX" -v lo="$WLO" -v hi="$WHI" 'BEGIN { exit !(dx >= lo && dx <= hi) }'

# Pin the corrected default window (no PLAYER_R inset): true clear region is
# [8.921, 9.999] for --pit-near 10 --pit-width 2.
A=$(SDL_VIDEODRIVER=dummy $BIN --episode --policy oracle --pit-near 10 --pit-width 2 --seed 1 --delay 0 --repeat 1)
echo "$A" | grep -q '"w_lo":8.921'
echo "$A" | grep -q '"w_hi":9.999'

echo "PASS test_window_consistency"
