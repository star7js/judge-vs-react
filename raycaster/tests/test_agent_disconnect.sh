#!/usr/bin/env bash
# An agent that dies mid-episode must not be recorded as a fall. With nobody
# answering, the player walks into the pit; before this guard the sim printed
# that as an ordinary {"fell":1,"decision_tick":-1} row and exited 0, so a
# crashed agent (API error, rate limit) was indistinguishable from a judge
# that never jumped. The game must exit nonzero and print no result row.
set -euo pipefail
cd "$(dirname "$0")/.."            # raycaster/
BIN=./build/keen-raycaster
export KEEN_AGENT_SOCK=/tmp/keen_ray_test_disconnect_$$.sock
OUT=$(mktemp)
# Both agent-driven tasks: the pit and the rendered aim scene.
for TASK_ARGS in "--pit-near 10 --pit-width 2" "--task aim"; do
: >"$OUT"
SDL_VIDEODRIVER=dummy $BIN --episode --policy agent \
  $TASK_ARGS --seed 1 --delay 0 --repeat 1 >"$OUT" 2>/dev/null &
GAME=$!
# Answer ten ticks with "none", then hang up.
python3 - <<'PY'
import os, socket, time
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
for _ in range(50):
    try:
        s.connect(os.environ["KEEN_AGENT_SOCK"]); break
    except OSError:
        time.sleep(0.1)
buf = b""
for _ in range(10):
    while b"\n" not in buf:
        buf += s.recv(4096)
    _, buf = buf.split(b"\n", 1)
    s.sendall(b'{"action":"none"}\n')
s.close()
PY
RC=0; wait $GAME || RC=$?
test "$RC" -eq 2                                   # distinct "agent died" exit code
if grep -q '"policy"' "$OUT"; then echo "FAIL: result row emitted for a dead agent ($TASK_ARGS)"; exit 1; fi
done
echo "PASS test_agent_disconnect"
