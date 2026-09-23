#!/usr/bin/env bash
# launch-glm-exl3-worker.sh — GLM-5.3-Flash-EXL3 rank-N (--headless) inside container
# FIRE KIT (3-node TP=3 triangle). Ported from the sparks 2x kit worker script;
# changes for fire:
#   - NODE_RANK comes from env (1 on gx10-2ca4, 2 on spark-3ff5), both headless
#   - master addr is HEAD_LAN_IP (192.168.1.6)
#   - Per-node NCCL config from FABRIC_IFS/FABRIC_HCAS/FABRIC_IPS: resolves the
#     RoCE v2 GID index for EVERY (hca, ip) pair, fail-closed, comma lists.
#   - FABRIC_MODE=socket falls back to TCP transport (NCCL_IB_DISABLE=1).
#
# Runs INSIDE the fire-glm-w<rank> container as: bash /tmp/launch.sh
# REQUIRED env: same set as the head script plus NODE_RANK.
set -euo pipefail
say() { echo "[fire-glm-worker-r${NODE_RANK:-?}] $*"; }

# ── NCCL fabric config ───────────────────────────────────────────────────────
IFS=',' read -ra _HCAS <<< "${FABRIC_HCAS:-}"
IFS=',' read -ra _IPS  <<< "${FABRIC_IPS:-}"
IFS=',' read -ra _IFS  <<< "${FABRIC_IFS:-}"
# Keep the WHOLE fabric in the socket bootstrap set: every pair in the triangle
# has a direct QSFP link, but each link is its own /24, so pinning a single
# interface name would strand one peer. NCCL takes comma-separated prefixes.
NCCL_SOCKET_IFNAME=$(IFS=,; echo "${_IFS[*]:-enp1s0f1np1}")
export NCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"

if [ "${FABRIC_MODE:-ib}" = "ib" ]; then
  export NCCL_IB_DISABLE=0
  _hca_list=()
  _gid_list=()
  for idx in "${!_HCAS[@]}"; do
    _hca=${_HCAS[$idx]}; _ip=${_IPS[$idx]:-}
    [ -n "$_ip" ] || { say "FATAL: FABRIC_HCAS/FABRIC_IPS length mismatch"; exit 1; }
    _ip_hex=$(printf '%02x%02x:%02x%02x' $(echo "$_ip" | tr . ' '))
    _match=""
    for g in /sys/class/infiniband/${_hca}/ports/1/gids/*; do
      i=${g##*/}
      [ "$(cat /sys/class/infiniband/${_hca}/ports/1/gid_attrs/types/$i 2>/dev/null)" = "RoCE v2" ] || continue
      case "$(cat "$g" 2>/dev/null)" in
        *ffff:${_ip_hex}) _match=$i; break;;
      esac
    done
    if [ -z "$_match" ]; then
      say "FATAL: no RoCE v2 GID on ${_hca} matches ${_ip} (::ffff:${_ip_hex}) — cabling/netplan not up?"
      for i in 0 1 2 3 4 5 6 7; do
        say "  ${_hca} gid$i: $(cat /sys/class/infiniband/${_hca}/ports/1/gids/$i 2>/dev/null) $(cat /sys/class/infiniband/${_hca}/ports/1/gid_attrs/types/$i 2>/dev/null)"
      done
      exit 1
    fi
    _hca_list+=("${_hca}^1"); _gid_list+=("$_match")
    say "fabric: ${_hca} ip=${_ip} gid_index=$_match"
  done
  # All fabric ports in this triangular kit sit at the same RoCE v2 IPv4 GID
  # index (verified 3 on all 3 nodes), so emit a SINGLE index — NCCL applies one
  # value to every HCA in NCCL_IB_HCA. A comma list is emitted only if a future
  # layout genuinely differs (and then we warn, since some NCCL builds ignore it).
  _uniq=$(printf '%s\n' "${_gid_list[@]}" | sort -u | wc -l | tr -d ' ')
  if [ "$_uniq" -eq 1 ]; then
    export NCCL_IB_GID_INDEX="${_gid_list[0]}"
  else
    export NCCL_IB_GID_INDEX=$(IFS=,; echo "${_gid_list[*]}")
    say "WARN: per-HCA GID indices differ (${_gid_list[*]}) — this NCCL build may not honor a comma list"
  fi
  export NCCL_IB_HCA=$(IFS=,; echo "${_hca_list[*]}")
  say "NCCL_IB_HCA=$NCCL_IB_HCA NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
else
  export NCCL_IB_DISABLE=1
  export NCCL_NET=Socket
  say "FABRIC_MODE=socket — IB disabled, TCP transport over ${NCCL_SOCKET_IFNAME} (bring-up only)"
fi

ARGS=(
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 0.0.0.0
  --port "${PORT}"
  --tensor-parallel-size "${TP}"
  --nnodes "${NNODES}"
  --node-rank "${NODE_RANK}"
  --master-addr "${HEAD_LAN_IP}"
  --master-port "${MASTER_PORT}"
  --distributed-executor-backend mp
  --headless
  --tool-call-parser glm47
  --enable-auto-tool-choice
  --reasoning-parser glm45
  --enable-prefix-caching
  --no-enable-flashinfer-autotune
)
[ "${ENFORCE_EAGER:-1}" = "1" ] && ARGS+=(--enforce-eager)
[ -n "${QUANTIZATION:-}" ] && [ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ]  && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${MAX_NUM_SEQS:-}" ] && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
[ -n "${KV_CACHE_DTYPE:-}" ] && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
if [ "${SPEC_METHOD:-mtp}" = "dflash" ]; then
  ARGS+=(--speculative-config "$(python3 -c 'import json,os
spec={"method":"dflash","model":os.environ["DFLASH_MODEL_DIR"],"num_speculative_tokens":int(os.environ.get("DFLASH_TOKENS","7")),"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard"}
tp=os.environ.get("DFLASH_DRAFT_TP","").strip()
if tp:
    spec["draft_tensor_parallel_size"]=int(tp)
print(json.dumps(spec,separators=(",",":")))')")
elif [ "${SPEC_METHOD:-mtp}" = "none" ]; then
  :
elif [ "${MTP_TOKENS:-0}" != "0" ]; then
  ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi
if [ -n "${CHAT_TEMPLATE:-}" ] && [ -f "${CHAT_TEMPLATE}" ]; then
  ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
  ARGS+=(--language-model-only)
else
  ARGS+=(--limit-mm-per-prompt '{"image":4,"video":1}')
  if [ -n "${MM_IMAGE_TOKENS:-}" ]; then
    ARGS+=(--mm-processor-kwargs "{\"max_image_tokens\":${MM_IMAGE_TOKENS}}")
    say "MM_IMAGE_TOKENS=${MM_IMAGE_TOKENS} (per-image token cap)"
  fi
  if [ -n "${MM_PROCESSOR_CACHE_GB:-}" ]; then
    ARGS+=(--mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB}")
    say "MM_PROCESSOR_CACHE_GB=${MM_PROCESSOR_CACHE_GB} (host RAM held for processed media)"
  fi
  [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  EXTRA=(${EXTRA_ARGS})
  ARGS+=("${EXTRA[@]}")
fi

[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }
for p in patch_w28_correctness.py \
         patch_glm_video_placeholders.py patch_suppress_stops_in_reasoning.py \
         patch_scheduler_decode_floor.py patch_mamba_align_chunking.py \
         patch_glm5_drafter_group.py \
         patch_hybrid_prefix_hit.py patch_xgrammar_termination.py \
         patch_kpool_tail_slotmap.py patch_router_gemm_gb10.py patch_cache_reset.py \
         patch_fine_grained_apc.py patch_apc_per_group_retention.py \
         patch_mamba_align_state_free.py patch_mamba_null_gap_retirement.py \
         patch_reasoning_noopener.py \
         patch_apc_no_store.py patch_kv_capacity_log.py \
         patch_adaptive_k.py patch_dense_fp8.py patch_default_max_new_tokens.py; do
  if [ -f "/opt/glm53/$p" ]; then
    say "applying $p"
    python3 "/opt/glm53/$p" || { say "FATAL: $p failed"; exit 1; }
  fi
done
say "joining TP${TP} at ${HEAD_LAN_IP}:${MASTER_PORT} as rank ${NODE_RANK}"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"
