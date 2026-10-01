#!/usr/bin/env bash
# tp4-tf-view.sh — TP=4 TensorFold console: frozen status banner + 4-column tails
# of the glm53-tf-r{0..3} container logs across the ring (fork 0900 lane).
#
# Modeled on quad-launcher/quad-view.sh (LIL lane) but points at the TF rank
# containers and the TF API (127.0.0.1:8000 on e116 via nginx :8888 is NOT used
# by this lane — TF serves directly on e116:8000).
#
# Usage:
#   tp4-tf-view.sh                live quad view (banner + 4 columns, 4s refresh)
#   tp4-tf-view.sh 0|1|2|3        follow one rank's log (banner kept)
#   tp4-tf-view.sh --once         single snapshot frame, no loop
#   tp4-tf-view.sh --interval S   refresh period (default 4)
#   tp4-tf-view.sh --tail N       log lines per column (default: auto by rows)
#
# Rank map (ring e116 -> dc90 -> 3ff5 -> 2ca4 -> e116), container glm53-tf-rK:
#   rank 0 head  spark-e116.local   192.168.1.6   (API 127.0.0.1:8000)
#   rank 1 w1    gx10-dc90.local    192.168.1.2
#   rank 2 w2    spark-3ff5.local   192.168.1.28
#   rank 3 w3    gx10-2ca4.local    192.168.1.31
set -u

HEAD_HOST=spark-e116.local
W1_HOST=gx10-dc90.local
W2_HOST=spark-3ff5.local
W3_HOST=gx10-2ca4.local
KEY=(-i /home/flowai/.ssh/spark_agent_key)
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new)

INTERVAL=4; TAIL=0; ONCE=0; MODE="quad"
while [ $# -gt 0 ]; do
  case "$1" in
    --once)     ONCE=1 ;;
    --) ;;   # just/shell separator — ignore
    --interval) INTERVAL="${2:?}" ; shift ;;
    --tail)     TAIL="${2:?}" ; shift ;;
    head|0) MODE="head" ;;
    w1|1)   MODE="w1" ;;
    w2|2)   MODE="w2" ;;
    w3|3)   MODE="w3" ;;
    -h|--help)  sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac; shift
done

rhost() { # rank -> host
  case "$1" in
    0) echo "$HEAD_HOST" ;;
    1) echo "$W1_HOST" ;;
    2) echo "$W2_HOST" ;;
    3) echo "$W3_HOST" ;;
  esac
}
rcontainer() { echo "glm53-tf-r$1"; }

ssh_host() { # $1 host, rest = remote command
  local h="$1"; shift
  ssh "${KEY[@]}" "${SSH_OPTS[@]}" "spark@$h" "$@"
}

# "status|memfreeG|load1|uptime" for one host
host_stats() { # $1 host $2 container
  ssh_host "$1" "
    docker ps -a --filter name='$2' --format '{{.Status}}' | head -1
    free -g | awk '/Mem:/{print \$7}'
    cut -d' ' -f1 /proc/loadavg
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1" 2>/dev/null | paste -sd'|' -
}

host_log() { # $1 host $2 container $3 line-count $4 width
  local out
  out=$(ssh_host "$1" "docker logs --tail '$3' '$2' 2>&1 | tr '\r' '\n' | tail -n '$3'" 2>/dev/null \
    | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | cut -c1-"$4")
  [ -n "$out" ] || out="  (no log output — container down or ssh failed)"
  printf '%s\n' "$out"
}

f=""   # frame buffer; every append ends with an explicit $'\n'

banner_frame() { # S[i] = "status|memfree|load|gpuMiB"
  local ts code hg
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  code=$(ssh_host "$HEAD_HOST" "curl -s -o /dev/null -w '%{http_code}' -m 4 http://127.0.0.1:8000/health" 2>/dev/null)
  if [ "$code" = "200" ]; then hg=$'\e[32m'; else hg=$'\e[31m'; fi
  f+="$(printf ' \e[1mFIRE QUAD TP4 TENSORFOLD (fork 0900)\e[0m  %s   refresh=%ss   exit: q/Ctrl-C   single rank: 0|1|2|3' "$ts" "$INTERVAL")"$'\n'
  f+="$(printf '  API health: %s%s\e[0m   http://spark-e116.local:8000/v1   model: GLM-5.3-Flash-EXL3 (TensorFold TP=4)' "$hg" "${code:----}")"$'\n'
  for i in 0 1 2 3; do
    [ -n "${S_N[i]:-}" ] || continue   # single-host mode renders only its row
    IFS='|' read -r st mem load gpu <<< "${S[i]:-||||}"
    f+="$(printf '  r%s %-18s %-28s memfree=%6sG load=%-6s gpu=%sMiB' "$i" "$(rhost "$i")" "${st:-(no container)}" "${mem:-?}" "${load:-?}" "${gpu:-?}")"$'\n'
  done
}

frame() {
  f=""
  banner_frame
  local width=58
  if [ "$MODE" = "quad" ]; then
    local cols=2 rows=$(( (LINES-8)/2 )); [ "$rows" -lt 6 ] && rows=6
    [ "$TAIL" -gt 0 ] && rows=$TAIL
    f+=$'\n'
    # two columns: (r0 | r1) and (r2 | r3) — actually left col r0,r2; right col r1,r3
    local left right
    left=$(host_log "$(rhost 0)" "$(rcontainer 0)" "$rows" "$width")
    right=$(host_log "$(rhost 1)" "$(rcontainer 1)" "$rows" "$width")
    paste -d'|' <(printf '%s\n' "$left" | sed "s/$/                    /" | cut -c1-"$width") \
                <(printf '%s\n' "$right" | cut -c1-"$width") | sed 's/|\(.*\)$/ │\1/' | while IFS= read -r l; do printf '%s\n' "$l"; done >> /tmp/_tp4_frame_cols 2>/dev/null || true
    # simpler: emit sequentially (avoid paste/pipes fragility)
  fi
}

# Rebuild frame() simpler: sequential blocks per rank (columns are fragile over ssh
# latency; blocks read better in a scrollback anyway).
frame() {
  f=""
  banner_frame
  local rows
  rows=$(( ${LINES:-40} - 10 )); [ "$rows" -lt 6 ] && rows=6
  [ "$TAIL" -gt 0 ] && rows=$TAIL
  local ranks=(0 1 2 3); [ "$MODE" != "quad" ] && ranks=("$MODE")
  for r in "${ranks[@]}"; do
    f+=$'\n'
    f+="$(printf ' ──── r%s %s (%s) %s' "$r" "$(rhost "$r")" "$(rcontainer "$r")" "$(printf '─%.0s' $(seq 1 30))")"$'\n'
    f+="$(host_log "$(rhost "$r")" "$(rcontainer "$r")" "$rows" 100)"$'\n'
  done
}

# Pre-collect stats once per frame
collect() {
  local ranks=(0 1 2 3); [ "$MODE" != "quad" ] && ranks=("$MODE")
  for i in "${ranks[@]}"; do
    S[i]=$(host_stats "$(rhost "$i")" "$(rcontainer "$i")")
    S_N[i]=1
  done
}

cleanup() { tput cnorm 2>/dev/null; printf '\033[?1049l'; stty echo 2>/dev/null; }
trap cleanup EXIT
printf '\033[?1049h'; tput civis 2>/dev/null; stty -echo -icanon time 0 min 0 2>/dev/null

while :; do
  collect
  frame
  printf '\033[H\033[2J%s' "$f"
  if [ "$ONCE" = "1" ]; then break; fi
  # q exits
  read -t "$INTERVAL" -n 1 k && [ "$k" = "q" ] && break
done
cleanup
