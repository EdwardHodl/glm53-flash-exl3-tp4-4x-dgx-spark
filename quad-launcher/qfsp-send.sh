#!/usr/bin/env bash
# qfsp-send — consistent inter-node file transfer over the fire QFSP ring.
#
# Usage:
#   qfsp-send <src-node> <dst-node> <path>... [--verify]
#   qfsp-send e116 dc90 /home/spark/tp4-ship.tar          # edge A direct
#   qfsp-send e116 3ff5 ~/logs/boot.log --verify           # non-adjacent -> mgmt LAN
#   qfsp-send all 2ca4 /home/spark/notes.md                # fan out from every node
#
# Nodes resolve to their ring-edge 10.10.x address when src and dst share a
# cable (edges A-D), else the 192.168.1.x mgmt LAN. Ring paths are 10-40x the
# mgmt path and carry jumbo frames (MTU 9000) — always prefer them.
#
# Ring (MAC-verified 2026-09-25):
#   edge A e116 10.10.1.1 (f0)      <-> dc90 10.10.1.2 (enP2p1s0f1)
#   edge B dc90 10.10.3.1 (f0)      <-> 3ff5 10.10.3.2 (f1)
#   edge C 3ff5 10.10.2.3 (f0)      <-> 2ca4 10.10.2.4 (f1)
#   edge D 2ca4 10.10.0.3 (f0)      <-> e116 10.10.0.1 (f1)
set -euo pipefail
KEY=/home/flowai/.ssh/spark_agent_key
LOG=/home/spark/logs/qfsp

mgmt_ip() { case $1 in e116) echo 192.168.1.6;; dc90) echo 192.168.1.2;; 3ff5) echo 192.168.1.28;; 2ca4) echo 192.168.1.31;; *) return 2;; esac; }

# edge_of SRC DST — echoes "EDGE <src-addr-on-edge> <ssh-user@host-for-src>" if
# the pair is directly cabled, else "MGMT".
edge_of() {
  local a=$1 b=$2
  case "$a$b" in
    e116dc90|dc90e116) echo "A 10.10.1.0" ;;
    dc903ff5|3ff5dc90) echo "B 10.10.3.0" ;;
    3ff52ca4|2ca43ff5) echo "C 10.10.2.0" ;;
    2ca4e116|e1162ca4) echo "D 10.10.0.0" ;;
    *) echo "MGMT" ;;
  esac
}

# address of node X on edge E (the end that belongs to X)
addr_on() { # $1=node $2=edge-letter
  case "$1$2" in
    e116A) echo 10.10.1.1;; dc90A) echo 10.10.1.2;;
    dc90B) echo 10.10.3.1;; 3ff5B) echo 10.10.3.2;;
    3ff5C) echo 10.10.2.3;; 2ca4C) echo 10.10.2.4;;
    2ca4D) echo 10.10.0.3;; e116D) echo 10.10.0.1;;
    *) return 1;;
  esac
}

run_node() { local h=$1; shift; ssh -n -T -o BatchMode=yes -o ConnectTimeout=12 -o LogLevel=ERROR -i "$KEY" "spark@$(mgmt_ip "$h")" "$@"; }
node_out() { run_node "$@" 2>/dev/null; }

send_one() { # src dst path...
  local src=$1 dst=$2; shift 2
  local info edge subnet saddr
  info=$(edge_of "$src" "$dst"); edge=$(echo "$info" | cut -d' ' -f1); subnet=$(echo "$info" | cut -d' ' -f2)
  local dsthost=$(mgmt_ip "$dst")
  local t0=$(date +%s)
  for f in "$@"; do
    local sz=$(node_out "$src" "stat -c%s '$f' 2>/dev/null || echo 0" | tr -cd '0-9')
    if [ "$edge" = MGMT ]; then
      echo "[$(date +%H:%M:%S)] $src->$dst (mgmt LAN) $f ($sz bytes)"
      node_out "$src" "rsync -a --timeout=120 '$f' spark@$dsthost:'$f'"
    else
      saddr=$(addr_on "$src" "$edge")
      # From src, the dst end of the edge is addr_on dst edge. rsync DIRECT to it.
      local daddr=$(addr_on "$dst" "$edge")
      echo "[$(date +%H:%M:%S)] $src->$dst (edge $edge: $saddr->$daddr) $f ($sz bytes)"
      node_out "$src" "rsync -a --timeout=120 '$f' spark@$daddr:'$f'"
    fi
    local dt=$(( $(date +%s) - t0 ))
    # verify size on receipt
    local rsz=$(node_out "$dst" "stat -c%s '$f' 2>/dev/null || echo 0" | tr -cd '0-9')
    if [ "$rsz" = "$sz" ]; then
      echo "[$(date +%H:%M:%S)] $src->$dst OK: $f ($sz bytes in ${dt}s)"
    else
      echo "[$(date +%H:%M:%S)] $src->$dst SIZE MISMATCH: sent=$sz got=$rsz" >&2; return 1
    fi
  done
}

main() {
  [ $# -ge 3 ] || { grep '^#' "$0" | head -20; exit 2; }
  local src=$1 dst=$2; shift 2
  local paths=() verify=0
  for a in "$@"; do [ "$a" = "--verify" ] && verify=1 || paths+=("$a"); done
  if [ "$src" = all ]; then
    for n in e116 dc90 3ff5 2ca4; do [ "$n" = "$dst" ] || send_one "$n" "$dst" "${paths[@]}"; done
  elif [ "$dst" = all ]; then
    for n in e116 dc90 3ff5 2ca4; do [ "$n" = "$src" ] || send_one "$src" "$n" "${paths[@]}"; done
  else
    send_one "$src" "$dst" "${paths[@]}"
  fi
}
main "$@"
