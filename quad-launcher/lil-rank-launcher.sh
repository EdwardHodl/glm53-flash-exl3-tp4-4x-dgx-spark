#!/usr/bin/env bash
# fire-quad LIL lane — launch ONE rank of TP4 GLM-5.3-Flash on the local-inference-lab
# QAD checkpoint (local-inference-lab/GLM-5.3-Flash-NVFP4) with the instanttensor loader.
#
# Mirrors ~/fire-quad/scripts/rank-launcher.sh (the proven nvidia lane) line for line.
# Three deltas only:
#   1. MODEL_DIR  = $HOME/glm53-nvfp4-lil          (QAD checkpoint, 186 GiB, 59 files)
#   2. --load-format instanttensor                (new lever under test)
#   3. --quantization modelopt_mixed              (checkpoint is modelopt MIXED_PRECISION)
#   4. NAME/PORT  = glm53_lil_tp4 / 8889          (parallel trial lane; nvidia keeps :8888)
#
# Provenance (verified 2026-09-27 before staging, probe session):
#   hf_quant_config.json -> quant_method=modelopt, quant_algo=MIXED_PRECISION,
#   ignore=[] (no attention block quantized -> passes the tonyd2wild ModelOpt-attention
#   corruption class, vllm#54150). NVFP4 routed experts layers 3-44, MXFP8 MTP
#   experts layer 45. Our v11 image HAS modelopt_mixed and dispatches NVFP4 ->
#   ModelOptNvFp4FusedMoE (marlin) and MXFP8 -> ModelOptMxFp8FusedMoE; MarlinExperts
#   supports kMxfp8Static on CUDA >= 7.5, Triton fallback is device-generic. Verified
#   importable on GB10. instanttensor 0.2.0 must be baked in the derived image
#   glm53-lil:v11-instanttensor (`just lilquad-image`); plain pip downgrades the NCCL
#   wheel to 2.29.7, hence --no-deps in the image build.
#
# Usage: NODE_RANK=<0..3> ./lil-rank-launcher.sh <rank 0..3>   (run as a file on the node)
set -euo pipefail

# ─── site values (never committed; resolved per node) ────────────────────────
MASTER="192.168.1.6"
PORT="${LIL_PORT:-8889}"       # trial lane; head serves 0.0.0.0:8889
MPORT=29521                    # distinct rendezvous port from the nvidia lane (29520)
GID_INDEX="${GID_INDEX:-3}"

MODEL_DIR="$HOME/glm53-nvfp4-lil"
DRAFT_DIR="$HOME/glm53-dflash2-draft"
NCCL_DIR="${NCCL_DIR:-$HOME/nccl-switchless-v0.0.1}"
CACHE_DIR="$HOME/glm53-tp4-cache"
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHAT_TEMPLATE="$REPO_ROOT/chat_template.jinja"
# Allow the piped/`bash -s` invocation used by `just lilquad-up` as a fallback.
[[ -f "$CHAT_TEMPLATE" ]] || CHAT_TEMPLATE="$HOME/fire-quad/chat_template.jinja"
EXPECTED_TEMPLATE_SHA256=0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5

# RoCE HCAs: dc90's edge-A end is the P2 cage; all others cross rail then pair rail.
case "$(hostname)" in
  gx10-dc90) IB_HCA="rocep1s0f0,roceP2p1s0f1" ;;
  *)         IB_HCA="rocep1s0f0,rocep1s0f1"   ;;
esac
case "$(hostname)" in
  spark-e116) MGMT_IF="enP7s7" ;;
  *)          MGMT_IF="wlP9s9" ;;
esac

R="${1:?usage: lil-rank-launcher.sh <rank 0..3>}"
case "$R" in 0|1|2|3) ;; *) echo "rank must be 0, 1, 2, or 3" >&2; exit 2 ;; esac
IMAGE="glm53-lil:v11-instanttensor"
NAME="glm53_lil_tp4"

die() { printf 'pre-flight failed: %s\n' "$*" >&2; exit 1; }

# ─── pre-flight (fail closed, mirroring the nvidia lane) ─────────────────────
test -f "$MODEL_DIR/config.json"          || die "missing $MODEL_DIR/config.json"
test -f "$MODEL_DIR/hf_quant_config.json" || die "missing $MODEL_DIR/hf_quant_config.json"
test -f "$DRAFT_DIR/model.safetensors"    || die "missing $DRAFT_DIR/model.safetensors"
test -f "$NCCL_DIR/libnccl.so.2"          || die "missing $NCCL_DIR/libnccl.so.2"
test -f "$CHAT_TEMPLATE"                  || die "missing $CHAT_TEMPLATE"
mkdir -p "$CACHE_DIR"

# Every served byte is pinned (AGENTS.md): chat template must match the pinned revision.
printf '%s  %s\n' "$EXPECTED_TEMPLATE_SHA256" "$CHAT_TEMPLATE" | sha256sum --check --status - \
  || die "official GLM chat template does not match the pinned revision"
if [[ -f "$NCCL_DIR/SHA256SUMS" ]]; then
  (cd "$NCCL_DIR" && sha256sum --check --ignore-missing SHA256SUMS) \
    || die "published NCCL asset failed its embedded checksum"
fi

# ModelOpt guard (phase-2 flag from the plan): refuse a checkpoint with quantized attention.
QIGNORE=$(python3 - "$MODEL_DIR" <<'PY'
import json, sys
c = json.load(open(sys.argv[1] + "/hf_quant_config.json"))
ig = c.get("ignore", [])
print(f"{len(ig)}|{c.get('quant_method','?')}|{c.get('quant_algo','?')}")
PY
) || die "could not read hf_quant_config.json"
echo "  quant check: ignore=${QIGNORE%%|*} | $(echo "$QIGNORE" | cut -d'|' -f2- | tr '|' '/')"
[[ "${QIGNORE%%|*}" == "0" ]] || die "checkpoint ignore-list is non-empty (attention-quantized?) — refusing"
[[ "$QIGNORE" == *"modelopt"*"MIXED_PRECISION"* ]] || die "unexpected quant_method/algo: $QIGNORE"

# instanttensor must be importable in the derived image (built by `just lilquad-image`).
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "missing image $IMAGE (run: just lilquad-image)"
docker run --rm --entrypoint python3 "$IMAGE" -c "import instanttensor" 2>/dev/null \
  || die "instanttensor not importable in $IMAGE; run: just lilquad-image"

# Fabric + GPU state.
[[ -n "${MIP:-}" ]] || MIP=$(ip -4 route get "$MASTER" 2>/dev/null | grep -oE 'src [0-9.]+' | awk '{print $2}')
[[ -n "$MIP" ]] || die "could not discover this node's management IP"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
  die "another process is using the GPU; stop the nvidia lane first (just quad-down)"
fi
IFS=',' read -r -a hcas <<<"$IB_HCA"
for hca_spec in "${hcas[@]}"; do
  hca=${hca_spec%%:*}; port=1; [[ "$hca_spec" == *:* ]] && port=${hca_spec#*:}
  base="/sys/class/infiniband/${hca}/ports/${port}"
  [[ -d "$base" ]] || die "missing RDMA device ${hca}:${port}"
  [[ "$(<"$base/state")" == '4: ACTIVE' ]] || die "${hca}:${port} not ACTIVE"
  gid=$(cat "$base/gids/$GID_INDEX" 2>/dev/null || true)
  [[ -n "$gid" && "$gid" != '0000:0000:0000:0000:0000:0000:0000:0000' ]] || die "${hca}:${port} GID empty"
  ndev=$(cat "$base/gid_attrs/ndevs/$GID_INDEX" 2>/dev/null || true)
  [[ "$(<"/sys/class/net/$ndev/mtu")" == 9000 ]] || die "$ndev MTU not 9000"
done
printf 'lil-quad pre-flight (rank %s on %s): template+NCCL pinned, quant=MIXED_PRECISION, instanttensor OK, fabric clean\n' \
  "$R" "$(hostname)"

docker rm -f "$NAME" 2>/dev/null || true

if [[ "$R" = "0" ]]; then TAIL=(--host 0.0.0.0 --port "$PORT"); else TAIL=(--headless); fi

docker run -d --name "$NAME" --restart no \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  --network host --ipc host --shm-size 32g --gpus all \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_DIR:/model:ro" \
  -v "$DRAFT_DIR:/draft:ro" \
  -v "$NCCL_DIR:/opt/patched-nccl:ro" \
  -v "$CACHE_DIR:/cache" \
  -v "$CACHE_DIR/root-cache:/root/.cache" \
  -v "$CHAT_TEMPLATE:/opt/glm53/chat_template.jinja:ro" \
  -e LD_PRELOAD=/opt/patched-nccl/libnccl.so.2 -e VLLM_NCCL_SO_PATH=/opt/patched-nccl/libnccl.so.2 \
  -e NCCL_SKIP_TREE_CONNECT=1 -e NCCL_SWITCHLESS_RING_ONLY=1 \
  -e NCCL_SOCKET_IFNAME="$MGMT_IF" -e GLOO_SOCKET_IFNAME="$MGMT_IF" -e VLLM_HOST_IP="$MIP" \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$IB_HCA" \
  -e NCCL_IB_GID_INDEX="$GID_INDEX" -e NCCL_IB_SUBNET_PREFIX_LEN=24 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 \
  -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_ALGO=Ring -e NCCL_PROTO=LL,LL128,Simple -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" -e NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}" \
  -e TORCH_USE_RTLD_GLOBAL=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_RANGE=10.10.0.0/22 \
  -e VLLM_ONE_GPU_PER_NODE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -e HF_HOME=/cache/hf -e XDG_CACHE_HOME=/cache -e VLLM_CACHE_ROOT=/cache/vllm \
  -e NODE_RANK="$R" -e MASTER_ADDR="$MASTER" \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  "$IMAGE" \
    /model \
    --load-format instanttensor \
    --quantization modelopt_mixed \
    --served-model-name glm-5.3-flash --trust-remote-code \
    --tensor-parallel-size 4 --nnodes 4 --node-rank "$R" \
    --master-addr "$MASTER" --master-port "$MPORT" \
    --gpu-memory-utilization 0.85 --max-model-len 262144 \
    --max-num-seqs "${LIL_MAX_NUM_SEQS:-8}" --block-size 2304 --moe-backend marlin \
    --skip-mm-profiling --mm-processor-cache-gb 4 \
    --mm-processor-kwargs '{"max_pixels":6422528,"max_image_tokens":4096}' \
    --limit-mm-per-prompt '{"image":16,"video":0}' \
    --kv-cache-dtype fp8_e4m3 --kv-cache-memory 25769803776 \
    --speculative-config '{"method":"dflash","model":"/draft","num_speculative_tokens":7}' \
    --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45 \
    --chat-template /opt/glm53/chat_template.jinja \
    --default-chat-template-kwargs '{"reasoning_effort":"low"}' \
    --distributed-executor-backend mp \
    "${TAIL[@]}"

echo "launched $NAME rank=$R mgmt=$MIP (instanttensor, modelopt_mixed, seqs=${LIL_MAX_NUM_SEQS:-8})"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" \
  || { echo "$NAME exited; run: docker logs $NAME" >&2; exit 1; }
