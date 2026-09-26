#!/usr/bin/env bash
# quad-view.sh — fire-quad NVFP4 console: frozen status banner on top + 4-column
# tails of the glm53_tp4 container logs from all 4 switchless-ring hosts.
#
# Usage:
#   quad-view.sh                  live quad view (banner + 4 columns, 4s refresh)
#   quad-view.sh 0|1|2|3 (or head|w1|w2|w3)   follow one host's log (banner kept)
#   quad-view.sh --once           single snapshot frame, no loop
#   quad-view.sh --interval S     refresh period (default 4)
#   quad-view.sh --tail N         log lines per column (default: auto by rows)
#
# Rank map (ring e116 -> dc90 -> 3ff5 -> 2ca4 -> e116), container glm53_tp4 on
# every rank, API :8888 on the head:
#   rank 0 head  spark-e116.local   192.168.1.6
#   rank 1 w1    gx10-dc90.local    192.168.1.2
#   rank 2 w2    spark-3ff5.local   192.168.1.28
#   rank 3 w3    gx10-2ca4.local    192.168.1.31
set -u

HEAD_HOST=spark-e116.local
W1_HOST=gx10-dc90.local
W2_HOST=spark-3ff5.local
W3_HOST=gx10-2ca4.local
CONTAINER=glm53_tp4
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

ssh_host() { # $1 host, rest = remote command
  local h="$1"; shift
  if [ "$h" = "$W1_HOST" ] || [ "$h" = "$W2_HOST" ]; then
    ssh "${KEY[@]}" "${SSH_OPTS[@]}" "spark@$h" "$@"
  else
    ssh "${SSH_OPTS[@]}" "spark@$h" "$@"
  fi
}

# "status|memfreeG|load1|uptime" for one host
host_stats() { # $1 host
  ssh_host "$1" "
    docker ps -a --filter name='$CONTAINER' --format '{{.Status}}' | head -1
    free -g | awk '/Mem:/{print \$7}'
    cut -d' ' -f1 /proc/loadavg
    uptime -p 2>/dev/null | sed 's/^up //'" 2>/dev/null | paste -sd'|' -
}

host_log() { # $1 host $2 line-count $3 width
  local out
  out=$(ssh_host "$1" "docker logs --tail '$2' '$CONTAINER' 2>&1 | tr '\r' '\n' | tail -n '$2'" 2>/dev/null \
    | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | cut -c1-"$3")
  [ -n "$out" ] || out="  (no log output — container down or ssh failed)"
  printf '%s\n' "$out"
}

f=""   # frame buffer; every append ends with an explicit $'\n'

banner_frame() { # S[i] = "status|mem|load|up"
  local ts code hg i
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://spark-e116.local:8888/health 2>/dev/null)
  if [ "$code" = "200" ]; then hg=$'\e[32m'; else hg=$'\e[31m'; fi
  f+="$(printf ' \e[1mFIRE QUAD NVFP4 VIEW\e[0m  %s   refresh=%ss   exit: q/Ctrl-C   single host: just quad-logs 0|1|2|3' "$ts" "$INTERVAL")"$'\n'
  f+="$(printf '  API health: %s%s\e[0m   http://spark-e116.local:8888/v1   model: GLM-5.3-Flash-NVFP4 (TP=4 switchless ring)' "$hg" "${code:----}")"$'\n'
  for i in 0 1 2 3; do
    [ -n "${S_N[i]:-}" ] || continue   # single-host mode renders only its row
    [ -n "${S[i]:-}" ] || S[i]="ssh unreachable||||"
    local st mem ld up col
    IFS='|' read -r st mem ld up <<<"${S[i]}"
    if [[ "$st" == Up* ]]; then col=$'\e[32m'; else col=$'\e[31m'; fi
    f+="$(printf '  \e[1m%-4s\e[0m %-19s %-17s %s%-26s\e[0m memfree=%-4s load=%-5s %s' \
         "${S_N[i]}" "${S_H[i]}" "$CONTAINER" "$col" "${st:--}" "${mem:--}" "${ld:--}" "${up:--}")"$'\n'
  done
  local sep n=$(( COLW * 4 + 3 ))
  sep=''; for ((i=0; i<n; i++)); do sep+='─'; done
  f+="$sep"$'\n'
}

# pad plain text to width, optionally bold
padline() { # $1 width $2 bold(0/1) $3 text
  local w=$1 t=$3 pad
  pad=$(( w - ${#t} )); [ "$pad" -lt 0 ] && pad=0
  if [ "$2" = 1 ]; then printf ' \e[1m%s\e[0m%*s' "$t" "$pad" ""
  else printf ' %s%*s' "$t" "$pad" ""; fi
}

# ── single-host follow mode ──────────────────────────────────────────────────
if [ "$MODE" != "quad" ]; then
  case "$MODE" in
    head) H=$HEAD_HOST; N=head ;;
    w1)   H=$W1_HOST;   N=w1 ;;
    w2)   H=$W2_HOST;   N=w2 ;;
    w3)   H=$W3_HOST;   N=w3 ;;
  esac
  [ "$TAIL" -gt 0 ] || TAIL=25
  COLW=$(( $(tput cols 2>/dev/null || echo 120) - 2 )); [ "$COLW" -ge 80 ] || COLW=118
  S_N=("$N"); S_H=("$H"); S=("$(host_stats "$H")")
  banner_frame
  printf '%s' "$f"
  ssh_host "$H" "docker logs -f --tail $TAIL '$CONTAINER' 2>&1" | sed -u 's/\x1b\[[0-9;]*[A-Za-z]//g'
  exit 0
fi

# ── live quad view ───────────────────────────────────────────────────────────
TMP=$(mktemp -d); cleanup() { rm -rf "$TMP"; printf '\e[?25h'; exit 0; }
trap cleanup INT TERM
S_N=(head w1 w2 w3)

while :; do
  COLS=$(tput cols 2>/dev/null || echo 260); [ "$COLS" -ge 160 ] || COLS=260
  ROWS=$(tput lines 2>/dev/null || echo 45)
  COLW=$(( (COLS - 5) / 4 ))
  SHOW=$(( ROWS - 11 )); [ "$SHOW" -ge 12 ] || SHOW=12; [ "$SHOW" -le 90 ] || SHOW=90
  [ "$TAIL" -gt 0 ] && SHOW=$TAIL          # explicit --tail wins

  S_H=("$HEAD_HOST" "$W1_HOST" "$W2_HOST" "$W3_HOST")
  host_stats "$HEAD_HOST" >"$TMP/s1" 2>/dev/null &
  host_stats "$W1_HOST"   >"$TMP/s2" 2>/dev/null &
  host_stats "$W2_HOST"   >"$TMP/s3" 2>/dev/null &
  host_stats "$W3_HOST"   >"$TMP/s4" 2>/dev/null &
  host_log  "$HEAD_HOST"  "$SHOW" "$COLW" >"$TMP/l1" 2>/dev/null &
  host_log  "$W1_HOST"    "$SHOW" "$COLW" >"$TMP/l2" 2>/dev/null &
  host_log  "$W2_HOST"    "$SHOW" "$COLW" >"$TMP/l3" 2>/dev/null &
  host_log  "$W3_HOST"    "$SHOW" "$COLW" >"$TMP/l4" 2>/dev/null &
  wait
  S=("$(cat "$TMP/s1" 2>/dev/null)" "$(cat "$TMP/s2" 2>/dev/null)" "$(cat "$TMP/s3" 2>/dev/null)" "$(cat "$TMP/s4" 2>/dev/null)")

  f=$'\e[H\e[2J'
  banner_frame
  f+="$(padline "$COLW" 1 "head · spark-e116")$(padline "$COLW" 1 "w1 · gx10-dc90")$(padline "$COLW" 1 "w2 · spark-3ff5")$(padline "$COLW" 1 "w3 · gx10-2ca4")"$'\n'

  mapfile -t L1 < <(tail -n "$SHOW" "$TMP/l1")
  mapfile -t L2 < <(tail -n "$SHOW" "$TMP/l2")
  mapfile -t L3 < <(tail -n "$SHOW" "$TMP/l3")
  mapfile -t L4 < <(tail -n "$SHOW" "$TMP/l4")
  for ((i=0; i<SHOW; i++)); do
    a=${L1[$i]:-}; b=${L2[$i]:-}; c=${L3[$i]:-}; d=${L4[$i]:-}
    f+="$(padline "$COLW" 0 "$a")$(padline "$COLW" 0 "$b")$(padline "$COLW" 0 "$c")$(padline "$COLW" 0 "$d")"$'\n'
  done

  if [ "$ONCE" = 1 ]; then printf '%s' "$f"; break; fi
  printf '%s' "$f"
  # sleep for INTERVAL, but allow 'q' to quit when run interactively
  if [ -t 0 ]; then
    key=""
    read -r -s -t "$INTERVAL" -n 1 key 2>/dev/null || true
    [ "$key" = "q" ] && cleanup
  else
    sleep "$INTERVAL"
  fi
done
