#!/usr/bin/env bash
# Load-guarded throughput bench for a fire-quad lane.
#
# Why this exists: the 2026-09-27 LIL-vs-nvidia comparison was invalidated because the
# user and background clients (hindsight extraction, devops cron jobs) were hitting
# 192.168.1.6:8888 while the bench ran. Concurrent load depresses per-request decode
# throughput by an unknown amount, so an unguarded number is not a measurement.
#
# Protocol enforced here:
#   1. require `Running: 0 reqs` for a continuous IDLE_WINDOW before starting
#   2. run the bench
#   3. re-read the engine log for any request that was NOT ours; if the count of
#      engine-reported request entries exceeds ours, the run was contended and the
#      result is printed as CONTENDED (discard it)
#
# Usage: bash bench-idle.sh <port> <label> [container]
#   bash bench-idle.sh 8888 lil-lane
set -euo pipefail
PORT="${1:?usage: bench-idle.sh <port> <label>}"
LABEL="${2:-lane-$PORT}"
CONTAINER="${3:-glm53_lil_tp4}"
HEAD="spark@spark-e116.local"
IDLE_WINDOW="${IDLE_WINDOW:-60}"     # seconds of guaranteed idleness required
POLL=10

running_now() {  # -> number of reqs in the newest engine line
  ssh -o BatchMode=yes "$HEAD" "docker logs --since ${POLL}s $CONTAINER 2>&1 | grep -oE 'Running: [0-9]+ reqs' | tail -1" \
    | grep -oE '[0-9]+' || echo 0
}

echo "[$LABEL] requiring ${IDLE_WINDOW}s of idle (Running: 0 reqs) before measuring"
idle=0
while (( idle < IDLE_WINDOW )); do
  n=$(running_now)
  if [[ "$n" != "0" ]]; then
    echo "  lane busy ($n reqs) — waiting for another ${IDLE_WINDOW}s window"
    idle=0
  else
    idle=$((idle + POLL))
  fi
  sleep "$POLL"
done
echo "[$LABEL] idle window satisfied; measuring"

T0=$(date +%s)
python3 "$(dirname "${BASH_SOURCE[0]}")/bench-warm.py" "http://spark-e116.local:${PORT}" "$LABEL"
T1=$(date +%s)

# Any request arrival we did not cause shows up as extra engine activity.
echo "[$LABEL] engine-side report for the measurement window:"
ssh -o BatchMode=yes "$HEAD" "docker logs --since $((T1 - T0 + 30))s $CONTAINER 2>&1 | grep 'Avg generation throughput' | tail -3"
echo "[$LABEL] if the engine numbers disagree with the wall-clock numbers above, the run was CONTENDED. Discard."
