#!/usr/bin/env bash
# fire-quad — fan the NVFP4 weights out from the head (e116) over the fabric.
# Ring: e116(0) -> dc90(1) -> 3ff5(2) -> 2ca4(3).
# Legs: e116->2ca4 (edge D, 10.10.0.3) and e116->dc90 (edge A, 10.10.1.2) in
# parallel, then dc90->3ff5 (edge B, 10.10.3.2). ~190 GiB per node.
# Run ON spark-e116:  ~/fire-quad/scripts/fanout-weights.sh
set -euo pipefail
SRC="$HOME/glm53-nvfp4"
NFILES_EXPECTED=44

count=$(ls "$SRC" | wc -l)
[[ "$count" -eq "$NFILES_EXPECTED" ]] || { echo "src incomplete: $count files (want $NFILES_EXPECTED)" >&2; exit 1; }
echo "source verified: $count files, $(du -shL "$SRC" | cut -f1)"

leg() { # $1=label $2=dest_host $3=ssh_opts
  local label=$1 host=$2 opts=$3
  echo "[$label] -> $host starting $(date +%H:%M:%S)"
  rsync -aL --info=progress2 -e "ssh $opts -o BatchMode=yes" "$SRC/" "spark@$host:glm53-nvfp4/"
  echo "[$label] -> $host done $(date +%H:%M:%S)"
}

leg 2ca4 10.10.0.3 "" &
P1=$!
leg dc90 10.10.1.2 "" &
P2=$!
wait $P1 $P2
echo "first hop complete; chaining dc90 -> 3ff5"
ssh -o BatchMode=yes spark@10.10.1.2 "rsync -aL --info=progress2 -e 'ssh -o BatchMode=yes' ~/glm53-nvfp4/ spark@10.10.3.2:glm53-nvfp4/"
echo "ALL LEGS DONE $(date +%H:%M:%S)"
