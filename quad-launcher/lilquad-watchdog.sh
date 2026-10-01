#!/usr/bin/env bash
# lilquad-watchdog.sh — keep the LIL lane (:8888) from sitting half-up.
#
# Failure mode this targets (receipted 2026-09-28): a stalled request step makes the
# EngineCore hit its 300 s RPC tolerance and kill ITSELF, so rank 0's container exits
# while the three workers stay Up holding rendezvous state from a dead head. The
# endpoint dies and nothing self-heals (the launcher runs with --restart no).
#
# Policy (deliberately conservative):
#   * head container has EXITED            -> full clean restart (down + up)
#   * head up, stall signature recent      -> WARN only (engine is ~300 s from dying);
#                                            set FORCE_ON_STALL=1 to restart proactively
#   * healthy                              -> silent, exit 0
#
# Usage:  bash quad-launcher/lilquad-watchdog.sh
#         FORCE_ON_STALL=1 bash quad-launcher/lilquad-watchdog.sh
#         LIL_MAX_MODEL_LEN=393216 bash quad-launcher/lilquad-watchdog.sh   # restart ctx
#
# Exit: 0 healthy or warn-only | 1 restart performed | 2 could not determine state
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

HEAD=spark-e116.local
WORKERS=(gx10-dc90.local spark-3ff5.local gx10-2ca4.local)
KEY=/home/flowai/.ssh/spark_agent_key
NAME=glm53_lil_tp4
CTX="${LIL_MAX_MODEL_LEN:-393216}"
STALL_WINDOW_S="${STALL_WINDOW_S:-300}"   # look this far back in the head log for the stall signature
FORCE_ON_STALL="${FORCE_ON_STALL:-0}"
LOG="${LIL_WATCHDOG_LOG:-/tmp/lilquad-watchdog.log}"
LOCK=/tmp/lilquad-watchdog.lock

say() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

# One watchdog at a time: two concurrent restarts would fight over the ring.
exec 9>"$LOCK"
flock -n 9 || { say "another watchdog run holds the lock; exiting"; exit 0; }

head_running() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$HEAD" \
    "docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null" 2>/dev/null
}

worker_up() {
  local h="$1"
  ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 spark@"$h" \
    "docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null" 2>/dev/null
}

# Health is NOT a stall detector: a stalled engine still answers /health until it dies.
api_health() {
  curl -s -o /dev/null -w '%{http_code}' -m 5 "http://192.168.1.6:8888/health" 2>/dev/null
}

# The EngineCore emits `No available shared memory broadcast block found in 60 seconds` while
# waiting on a worker that never returns — but it ALSO emits it during boot (engine init /
# JIT / graph capture), which is benign. Discriminator: a real stall has NO completed
# generation after its last occurrence. So: warn only if the last stall line is NOT followed
# by a `POST /v1/chat/completions ... 200 OK`.
# Prints: "<stall_lines> <completions_after_last_stall>"   (both integers), or "" on failure.
stall_state() {
  local log last idx after
  log=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$HEAD" \
    "docker logs --since ${STALL_WINDOW_S}s $NAME 2>&1" 2>/dev/null) || return 1
  [[ -n "$log" ]] || { printf '0 0'; return 0; }
  last=$(printf '%s\n' "$log" | grep -n 'No available shared memory broadcast block' | tail -1 | cut -d: -f1)
  [[ -n "$last" ]] || { printf '0 0'; return 0; }
  after=$(printf '%s\n' "$log" | awk -v n="$last" 'NR>n' \
    | grep -c 'POST /v1/chat/completions HTTP/1.1" 200 OK')
  printf '%s %s' "$(printf '%s\n' "$log" | grep -c 'No available shared memory broadcast block')" "$after"
}

restart_lane() {
  local why="$1"
  say "RESTART: $why"
  say "taking all four ranks down"
  just lilquad-down >>"$LOG" 2>&1 || say "warn: lilquad-down returned non-zero"
  say "bringing up at ctx=$CTX"
  just lilquad-up LIL_MAX_MODEL_LEN="$CTX" >>"$LOG" 2>&1 || say "warn: lilquad-up returned non-zero"
  # The engine needs ~8-10 min (warm) to serve; report state, do not block forever.
  say "restart issued; endpoint will answer once load + JIT + graph capture finish"
  exit 1
}

hr=$(head_running)
if [[ "$hr" != "true" && "$hr" != "false" ]]; then
  say "ERROR: cannot read head container state (ssh/docker failure); not restarting blind"
  exit 2
fi

if [[ "$hr" == "false" ]]; then
  orphaned=()
  for h in "${WORKERS[@]}"; do
    [[ "$(worker_up "$h")" == "true" ]] && orphaned+=("$h")
  done
  if (( ${#orphaned[@]} > 0 )); then
    restart_lane "head container $NAME is not running; workers still up: ${orphaned[*]} (orphaned half-cluster)"
  fi
  restart_lane "head container $NAME is not running (api health $(api_health))"
fi

state=$(stall_state)
if [[ "$state" =~ ^([0-9]+)[[:space:]]+([0-9]+)$ ]]; then
  stalls=${BASH_REMATCH[1]}
  after=${BASH_REMATCH[2]}
  if (( stalls > 0 && after == 0 )); then
    say "WARN: $stalls stall-signature line(s) in the last ${STALL_WINDOW_S}s with ZERO completed generations after them — engine is within ~300 s of self-killing. api health $(api_health)"
    if [[ "$FORCE_ON_STALL" == "1" ]]; then
      restart_lane "FORCE_ON_STALL=1 and an unrecovered stall signature is present"
    fi
    exit 0
  fi
  (( stalls > 0 )) && say "note: $stalls boot-time/stale stall-signature line(s), but $after generation(s) completed after them — benign, ignoring"
else
  say "note: could not evaluate stall state; reporting health only"
fi

say "OK: head running, api health $(api_health), no unrecovered stall signature"
exit 0
