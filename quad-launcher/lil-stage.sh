#!/usr/bin/env bash
# fire-quad LIL lane — stage the local-inference-lab QAD checkpoint on all 4
# ranks: download to the head, fan out over the ring fabric, install
# instanttensor (0.2.0, --no-deps) into the running image on every rank.
#
# Phases:
#   download : hf download local-inference-lab/GLM-5.3-Flash-NVFP4 on e116 (~186 GiB)
#   fanout   : rsync e116 -> 2ca4 + dc90 in parallel, then dc90 -> 3ff5
#   image    : pip install --no-deps instanttensor into sm121-v11-dflash2 on all
#              4 ranks and commit it as glm53-lil:v11 (keeps the shared base
#              image untouched so the nvidia lane keeps booting)
#   check    : verify all 4 ranks (file counts + import + quant config)
# Usage: ./lil-stage.sh <download|fanout|image|check>
set -euo pipefail
PHASE="${1:?usage: lil-stage.sh <download|fanout|image|check>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="local-inference-lab/GLM-5.3-Flash-NVFP4"
IMAGE="ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2"
NEWIMAGE="glm53-lil:v11-instanttensor"
SSH_K="-i /home/flowai/.ssh/spark_agent_key"
HOSTS=(spark-e116.local gx10-dc90.local spark-3ff5.local gx10-2ca4.local)

case "$PHASE" in
download)
  ssh -o BatchMode=yes spark@spark-e116.local "setsid nohup ~/.local/bin/hf download \\
    local-inference-lab/GLM-5.3-Flash-NVFP4 \\
    --local-dir ~/glm53-nvfp4-lil > ~/logs/hf-lil-glm53.log 2>&1 < /dev/null & \\
    echo 'LIL download started; tail ~/logs/hf-lil-glm53.log on e116'"
  ;;
fanout)
  ssh -o BatchMode=yes spark@spark-e116.local 'bash -s' <<'REMOTE'
set -euo pipefail
SRC="$HOME/glm53-nvfp4-lil"
n=$(ls "$SRC" | wc -l)
shards=$(ls "$SRC" | grep -c "safetensors$" || true)
echo "source: $n files ($shards safetensors); verifying config present"
test -f "$SRC/config.json" && test -f "$SRC/hf_quant_config.json"
leg() { echo "[$1] -> $2 $(date +%H:%M:%S)"; rsync -aL --info=progress2 -e "ssh -o BatchMode=yes" "$SRC/" "spark@$2:glm53-nvfp4-lil/"; echo "[$1] done $(date +%H:%M:%S)"; }
leg 2ca4 10.10.0.3 & P1=$!
leg dc90 10.10.1.2 "" & P2=$!
wait $P1 $P2
ssh -o BatchMode=yes spark@10.10.1.2 "rsync -aL --info=progress2 -e 'ssh -o BatchMode=yes' ~/glm53-nvfp4-lil/ spark@10.10.3.2:glm53-nvfp4-lil/"
echo "ALL LEGS DONE $(date +%H:%M:%S)"
REMOTE
  echo "fanout complete"
  ;;
image)
  for h in "${HOSTS[@]}"; do
    echo "== building $NEWIMAGE on $h"
    if [[ "$h" == "gx10-dc90.local" || "$h" == "spark-3ff5.local" ]]; then K="$SSH_K"; else K=""; fi
    ssh -o BatchMode=yes $K spark@$h "bash -s" <<'REMOTE'
set -euo pipefail
D=$(mktemp -d)
printf 'FROM %s\nRUN pip install --no-deps --no-cache-dir instanttensor==0.2.0 && rm -rf /root/.cache/pip\n' \
  ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2 > "$D/Dockerfile"
docker build -q -t glm53-lil:v11-instanttensor "$D"
rm -rf "$D"
docker run --rm --gpus all glm53-lil:v11-instanttensor python3 -c 'import instanttensor; print("  image verify: instanttensor OK")' 2>/dev/null \
  || docker run --rm --gpus all --entrypoint python3 glm53-lil:v11-instanttensor -c 'import instanttensor; print("  image verify: instanttensor OK")'
REMOTE
  done
  echo "image built on all 4 ranks: glm53-lil:v11-instanttensor"
  ;;
deploy)
  for h in "${HOSTS[@]}"; do
    echo "== deploying launcher to $h"
    if [[ "$h" == "spark-e116.local" ]]; then
      scp -o BatchMode=yes quad-launcher/lil-rank-launcher.sh spark@$h:/home/spark/fire-quad/scripts/lil-rank-launcher.sh
    else
      scp -o BatchMode=yes $SSH_K quad-launcher/lil-rank-launcher.sh spark@$h:/home/spark/fire-quad/scripts/lil-rank-launcher.sh
    fi
  done
  echo "launcher deployed to all 4 ranks (~/fire-quad/scripts/lil-rank-launcher.sh)"
  ;;
check)
  cat > /tmp/lil-check-rank.sh <<'EOF'
d=$HOME/glm53-nvfp4-lil
n=$(ls $d 2>/dev/null | wc -l || echo 0)
echo "  files: $n"
test -f $d/hf_quant_config.json && echo "  quant config: present" || echo "  quant config: MISSING"
docker run --rm --entrypoint python3 glm53-lil:v11-instanttensor -c 'import instanttensor; print("  instanttensor: OK")' 2>/dev/null || echo "  instanttensor: NOT INSTALLED"
df -BG --output=avail $HOME | tail -1 | xargs echo "  disk avail:"
EOF
  for h in "${HOSTS[@]}"; do
    echo "== $h"
    if [[ "$h" == "spark-e116.local" ]]; then
      ssh -o BatchMode=yes spark@$h 'bash -s' < /tmp/lil-check-rank.sh
    else
      ssh -o BatchMode=yes $SSH_K spark@$h 'bash -s' < /tmp/lil-check-rank.sh
    fi
  done
  rm -f /tmp/lil-check-rank.sh
  ;;
*) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac
