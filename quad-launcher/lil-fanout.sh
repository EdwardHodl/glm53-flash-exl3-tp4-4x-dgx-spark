#!/usr/bin/env bash
# fire-quad LIL lane — fan the LIL checkpoint out from the head over the ring.
# Runs ON spark-e116 via: just lilquad-fanout  (this file is piped to bash there).
# Legs mirror fanout-weights.sh: e116->2ca4 (10.10.0.3) and e116->dc90 (10.10.1.2)
# in parallel, then dc90->3ff5 (10.10.3.2). ~186 GiB per node.
set -euo pipefail
SRC="$HOME/glm53-nvfp4-lil"
n=$(ls "$SRC" | wc -l)
shards=$(ls "$SRC" | grep -c 'safetensors$' || true)
test -f "$SRC/hf_quant_config.json" || { echo "source incomplete: no hf_quant_config.json" >&2; exit 1; }
echo "source verified: $n files ($shards safetensors), $(du -shL "$SRC" | cut -f1)"

leg() { echo "[$1] -> $2 starting $(date +%H:%M:%S)"; rsync -aL --info=progress2 -e "ssh -o BatchMode=yes" "$SRC/" "spark@$2:glm53-nvfp4-lil/"; echo "[$1] done $(date +%H:%M:%S)"; }
leg 2ca4 10.10.0.3 & P1=$!
leg dc90 10.10.1.2  & P2=$!
wait $P1 $P2
echo "first hop complete; chaining dc90 -> 3ff5"
ssh -o BatchMode=yes spark@10.10.1.2 "rsync -aL --info=progress2 -e 'ssh -o BatchMode=yes' ~/glm53-nvfp4-lil/ spark@10.10.3.2:glm53-nvfp4-lil/"
echo "ALL LEGS DONE $(date +%H:%M:%S)"
