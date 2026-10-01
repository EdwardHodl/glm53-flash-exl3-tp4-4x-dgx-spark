# Justfile for the FIRE triangle — GLM-5.3-Flash-EXL3 TP=3 on 3x DGX Spark
# spark-e116.local (rank 0, head, API :8888) + gx10-2ca4.local (rank 1) + spark-3ff5.local (rank 2)
# Modeled on the sparks 2-node kit (../sparks/playbooks).

inventory := "playbooks/inventory.yml"
playbooks := "playbooks"

# List the fire triangle and per-node vars
list-inventory:
    @ansible-inventory -i {{inventory}} --graph --vars

# Sanity: SSH + sudo + GPU visible on all 3 nodes; show fabric port states
ping:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{justfile_directory()}}
    ansible fire_nodes -m ping -i {{inventory}}
    for h in spark-e116.local gx10-2ca4.local spark-3ff5.local; do
      echo "== $h"
      if [ "$h" = "spark-3ff5.local" ]; then K="-i /home/flowai/.ssh/spark_agent_key"; else K=""; fi
      ssh $K -o BatchMode=yes spark@$h \
        "hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; \
         for hca in /sys/class/infiniband/*/; do echo \"  \$(basename \$hca): \$(cat \$hca/ports/1/state 2>/dev/null)\"; done; \
         df -BG --output=avail /home/spark/.cache/huggingface | tail -1"
    done

# Check QSFP triangle readiness: each node's fabric IPs answering (post-cabling check)
fabric-check:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{justfile_directory()}}
    ansible-playbook -i {{inventory}} {{playbooks}}/fabric-check.yml

# Download EXL3 weights + DFlash2 on head, rsync to both workers (~164 GiB x3)
download-glm-exl3:
    @ansible-playbook -i {{inventory}} {{playbooks}}/download-glm-exl3.yml

# Build or fetch the EXL3 overlay image on all 3 nodes (GHCR pull by default;
# local build from glm/exl3/Dockerfile + ship with: just build-glm-exl3-rebuild)
build-glm-exl3:
    @ansible-playbook -i {{inventory}} {{playbooks}}/build-glm-exl3.yml

# Local build (~20-40 min on GB10) + docker save | ssh load to both workers
build-glm-exl3-rebuild:
    @ansible-playbook -i {{inventory}} -e build_exl3_image=true {{playbooks}}/build-glm-exl3.yml

# Start the FIRE TP=3 cluster (workers then head, API on spark-e116:8888).
# Usage: just start-glm-exl3           (IB fabric — requires QSFP triangle cabled + netplanned)
#        just start-glm-exl3 socket    (TCP transport bring-up fallback — slow)
start-glm-exl3 MODE="ib":
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{justfile_directory()}}
    export ANSIBLE_TIMEOUT=2400
    ansible-playbook -i {{inventory}} {{playbooks}}/start-glm-exl3.yml -e fabric_mode={{MODE}}

# Stop the FIRE cluster and release ports
stop-glm-exl3:
    @ansible-playbook -i {{inventory}} {{playbooks}}/stop-glm-exl3.yml

# Stream or snapshot FIRE container logs. Usage:
#   just log               head (live follow)
#   just log w1            gx10-2ca4 worker
#   just log w2            spark-3ff5 worker
#   just log status        one-shot status of all 3 nodes
log WHICH="head":
    #!/usr/bin/env bash
    exec ./glm/scripts/glm-exl3-progress.sh "$WHICH"

# One-shot health check on the API (spark-e116.local:8888)
health:
    #!/usr/bin/env bash
    curl -s -o /dev/null -w 'health => HTTP %{http_code}\n' -m 5 http://spark-e116.local:8888/health || true
    curl -s -m 5 http://spark-e116.local:8888/v1/models | python3 -m json.tool | head -10 || true

# Test the FIRE Models API + a short inference on spark-e116.local:8888
test-glm-exl3 +HOST="spark-e116.local":
    #!/usr/bin/env bash
    set -euo pipefail
    h="${HOST:-spark-e116.local}"
    p=8888
    m="GLM-5.3-Flash-EXL3"
    base="http://${h}:${p}"
    echo "=== Test 1: Health ==="
    curl -s -o /dev/null -w 'health => HTTP %{http_code}\n' "${base}/health" || true
    echo ""
    echo "=== Test 2: Models API ==="
    curl -s "${base}/v1/models" | python3 -m json.tool | head -10 || true
    echo ""
    echo "=== Test 3: Inference (thinking off) ==="
    curl -s "${base}/v1/chat/completions" -H "Content-Type: application/json" -d '{
      "model": "'"$m"'",
      "messages": [{"role":"user","content":"Reply exactly: OK FIRE GLM 5.3 EXL3 LIVE TP3"}],
      "max_tokens": 10,
      "temperature": 0,
      "chat_template_kwargs": {"enable_thinking": false}
    }' | python3 -m json.tool || true
    echo ""
    echo "=== Test 4: TP3 container view ==="
    ssh spark@spark-e116.local 'docker ps --filter name=glm53-exl3-tp3 --format "$(printf "\173\173.Names\175\175 \173\173.Status\175\175")"'

# ── mem-free ritual for the quad (GB10 page-cache drop before boots) ──
# ── TP=4 quad lane (ring: e116 r0 head -> dc90 r1 -> 3ff5 r2 -> 2ca4 r3) ─────

# Free page cache on all 4 nodes. REQUIRED before a TP4 start (GB10 unified
# memory: staging/page cache counts against CUDA; see README boot saga).
mem-free-quad:
    #!/usr/bin/env bash
    set -euo pipefail
    for h in spark-e116.local; do
      ssh -o BatchMode=yes spark@$h 'free -g | awk "/Mem:/{print \"  '\"$h\"' avail=\"\$7\"G\"}"; sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"'
    done
    for h in gx10-dc90.local gx10-2ca4.local spark-3ff5.local; do
      ssh -i /home/flowai/.ssh/spark_agent_key -o BatchMode=yes spark@$h \
        'free -g | awk "/Mem:/{print \"  '\"$h\"' avail=\"\$7\"G\"}"; sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"'
    done


# ── QUAD lane: NVFP4 nvidia weights + alexellis switchless recipe (4 ranks) ──
# nvidia/GLM-5.3-Flash-NVFP4, image sm121-v11-dflash2, API glm-5.3-flash :8888
# on spark-e116. Scripts: quad-launcher/ (site config, rank launcher, fabric
# verify). Ring + gates per QUAD-ALEXELLIS-PROPOSAL.md; port 8888 keeps fire
# crons working. TP=3/TP=4 EXL3 lanes removed 2026-09-26 (quad is the only lane).

QUAD_CFG := "quad-launcher/cluster.env"
QUAD_RECIPE := "/home/flowai/Workspaces/devops/recipes/glm53flash/glm-5.3-flash-4x-dgx-spark-switchless"

# Verify the switchless ring: GID index 3 RoCE-v2, MTU 9000, jumbo pings.
quad-verify:
    bash quad-launcher/verify-fabric.sh

# Bring up the quad (workers 3->2->1 headless, then head 0; preflights + gate).
# Runs the recipe's cluster-up.sh with our site config. Takes ~15-30 min cold.
quad-up:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{justfile_directory()}}" && just mem-free-quad
    cd "{{QUAD_RECIPE}}"
    GLM53_CLUSTER_CONFIG="{{justfile_directory()}}/{{QUAD_CFG}}" bash scripts/cluster-up.sh

# Remove the quad containers on all 4 ranks.
quad-down:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{QUAD_RECIPE}}
    GLM53_CLUSTER_CONFIG="{{justfile_directory()}}/{{QUAD_CFG}}" bash scripts/cluster-down.sh

# Read-only quad status (containers, GPU, API).
quad-status:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{QUAD_RECIPE}}
    GLM53_CLUSTER_CONFIG="{{justfile_directory()}}/{{QUAD_CFG}}" bash scripts/cluster-status.sh

# Follow one rank's container log. Usage: just quad-log 0|1|2|3
quad-log RANK="0":
    #!/usr/bin/env bash
    set -euo pipefail
    case {{RANK}} in
      0) h=spark-e116.local; k="" ;;
      1) h=gx10-dc90.local; k="-i /home/flowai/.ssh/spark_agent_key" ;;
      2) h=spark-3ff5.local; k="-i /home/flowai/.ssh/spark_agent_key" ;;
      3) h=gx10-2ca4.local; k="" ;;
      *) echo "rank must be 0-3"; exit 2 ;;
    esac
    exec ssh $k -o BatchMode=yes spark@$h "docker logs -f glm53_tp4"

# Quad console: frozen status banner (API health + per-rank container/mem/load)
# on top + 4-column live tails of the glm53_tp4 logs on all 4 ring hosts.
#   just quad-logs             live quad view
#   just quad-logs 0|1|2|3     follow one host's log (banner kept)
#   just quad-logs -- --once   single snapshot frame, no loop
quad-logs *ARGS="":
    #!/usr/bin/env bash
    exec ./quad-launcher/quad-view.sh {{ARGS}}

# Resume the HF download of nvidia/GLM-5.3-Flash-NVFP4 on the head (e116).
quad-download:
    ssh spark@spark-e116.local 'setsid nohup ~/.local/bin/hf download nvidia/GLM-5.3-Flash-NVFP4 > ~/logs/hf-nvidia-glm53-cache.log 2>&1 < /dev/null & echo "download resumed (cache layout)"'

# Fan weights out from the head to the 3 workers over the ring (190 GiB each).
quad-fanout:
    ssh spark@spark-e116.local 'setsid nohup ~/fire-quad/scripts/fanout-weights.sh > ~/logs/fanout-weights.log 2>&1 < /dev/null & echo "fanout started; tail ~/logs/fanout-weights.log on e116"'

# One-shot API health + short generation on the quad endpoint.
quad-test:
    #!/usr/bin/env bash
    set -euo pipefail
    base="http://spark-e116.local:8888"
    curl -s -o /dev/null -w 'health => HTTP %{http_code}\n' -m 5 "$base/health" || true
    curl -s -m 120 "$base/v1/chat/completions" -H "Content-Type: application/json" -d '{
      "model": "glm-5.3-flash",
      "messages": [{"role":"user","content":"Reply exactly: QUAD NVFP4 LIVE TP4"}],
      "max_tokens": 16,
      "temperature": 0,
      "chat_template_kwargs": {"reasoning_effort": "low"}
    }' | python3 -m json.tool || true

# ── LIL lane: local-inference-lab QAD checkpoint + instanttensor (trial lane) ──
# local-inference-lab/GLM-5.3-Flash-NVFP4 (quantization-aware distilled NVFP4,
# attention untouched) on the same ring/image family, --load-format instanttensor,
# modelopt_mixed quant. Provenance: QUAD-UPGRADE-PLAN.md Phase 5 + lil-rank-launcher.sh
# header. The nvidia lane (quad-*) is untouched; the LIL lane runs in PARALLEL on
# its own container (glm53_lil_tp4, rendezvous 29521) — STOP the nvidia lane first,
# both need all four GPUs. Promote = stop quad lane, start lil lane.
#   just lilquad-stage download|fanout|image|check   (staging phases)
#   just lilquad-up                                  (boot the LIL lane)
#   just lilquad-down / -status / -log / -test       (ops)
# The LIL lane now serves :8888 (promoted 2026-09-27), the contract the clients
# already point at. The nvidia lane (quad-*) keeps its own container name and
# would collide on port 8888, so keep exactly one lane up at a time.

# Download the LIL checkpoint to the head (resumable; ~186 GiB from HF CDN).
lilquad-download:
    bash quad-launcher/lil-stage.sh download

# Fan the checkpoint out from the head to the 3 workers over the ring.
lilquad-fanout:
    ssh -o BatchMode=yes spark@spark-e116.local 'bash -s' < quad-launcher/lil-fanout.sh

# Build the derived image glm53-lil:v11-instanttensor on all 4 ranks
# (base + instanttensor 0.2.0, --no-deps; NCCL wheel stays pinned).
lilquad-image:
    bash quad-launcher/lil-stage.sh image

# Verify staging on all 4 ranks (files, quant config, image import, disk).
lilquad-check:
    bash quad-launcher/lil-stage.sh check

# Bring the LIL quad up (deploy launcher, workers 3->2->1 headless, then head 0; :8888).
# Context is an ENVIRONMENT override, not a just argument:
#   LIL_MAX_MODEL_LEN=393216 just lilquad-up     # 384K  (promoted config)
#   LIL_MAX_NUM_SEQS=8 LIL_MAX_MODEL_LEN=393216 just lilquad-up
#   just lilquad-up                              # default 262144
# NOTE: `just lilquad-up LIL_MAX_MODEL_LEN=393216` FAILS (just reads it as a recipe name —
# "justfile does not contain recipe"); the recipe declares no parameters.
lilquad-up:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{justfile_directory()}}
    bash quad-launcher/lil-stage.sh deploy
    just mem-free-quad
    LIL=/home/spark/fire-quad/scripts/lil-rank-launcher.sh
    for r in 3 2 1; do
      h=$(case $r in 1) echo gx10-dc90.local;; 2) echo spark-3ff5.local;; 3) echo gx10-2ca4.local;; esac)
      echo "== launching LIL rank $r on $h"
      ssh -o BatchMode=yes -i /home/flowai/.ssh/spark_agent_key spark@$h \
        "NODE_RANK=$r LIL_PORT=8888 LIL_MAX_NUM_SEQS=${LIL_MAX_NUM_SEQS:-8} LIL_MAX_MODEL_LEN=${LIL_MAX_MODEL_LEN:-262144} bash $LIL $r" &
      sleep 3
    done
    echo "== launching LIL rank 0 on spark-e116.local (head)"
    ssh -o BatchMode=yes spark@spark-e116.local \
      "NODE_RANK=0 LIL_PORT=8888 LIL_MAX_NUM_SEQS=${LIL_MAX_NUM_SEQS:-8} LIL_MAX_MODEL_LEN=${LIL_MAX_MODEL_LEN:-262144} bash $LIL 0"
    echo "all 4 LIL ranks launched; watch with: just lilquad-log"

# Remove the LIL containers on all 4 ranks.
lilquad-down:
    #!/usr/bin/env bash
    set -euo pipefail
    for h in spark-e116.local gx10-dc90.local spark-3ff5.local gx10-2ca4.local; do
      k=""; [[ "$h" == "spark-e116.local" ]] || k="-i /home/flowai/.ssh/spark_agent_key"
      ssh -o BatchMode=yes $k spark@$h 'docker rm -f glm53_lil_tp4 2>/dev/null || true'
    done
    echo "LIL containers removed"

# Read-only LIL status: containers, GPU use, API health.
lilquad-status:
    #!/usr/bin/env bash
    set -euo pipefail
    for h in spark-e116.local gx10-dc90.local spark-3ff5.local gx10-2ca4.local; do
      k=""; [[ "$h" == "spark-e116.local" ]] || k="-i /home/flowai/.ssh/spark_agent_key"
      echo "== $h"
      ssh -o BatchMode=yes $k spark@$h 'docker ps -a --filter name=glm53_lil_tp4' || true
    done
    curl -s -o /dev/null -w 'API :8888/health => HTTP %{http_code}\n' -m 5 http://spark-e116.local:8888/health || true

# Tail the head rank's vLLM log.
lilquad-log:
    ssh -o BatchMode=yes spark@spark-e116.local 'docker logs -f --tail 80 glm53_lil_tp4'

# One-shot API health + generation + Korean corruption probe on the LIL endpoint (:8888).
lilquad-test:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{justfile_directory()}}
    base="http://spark-e116.local:8888"
    curl -s -o /dev/null -w 'health => HTTP %{http_code}\n' -m 5 "$base/health" || true
    curl -s -m 180 "$base/v1/chat/completions" -H "Content-Type: application/json" -d '{
      "model": "glm-5.3-flash",
      "messages": [{"role":"user","content":"Reply exactly: LIL QAD LIVE TP4"}],
      "max_tokens": 16, "temperature": 0,
      "chat_template_kwargs": {"reasoning_effort": "low"}
    }' | python3 -m json.tool || true
    # corruption probe (tonyd2wild #23 protocol): temperature 0, non-streaming
    curl -s -m 180 "$base/v1/chat/completions" -H "Content-Type: application/json" -d '{
      "model": "glm-5.3-flash",
      "messages": [{"role":"user","content":"안녕하세요, 오늘 날씨는 어떤가요? 짧게 한국어로 대답하세요."}],
      "max_tokens": 64, "temperature": 0
    }' | python3 quad-launcher/korean_probe.py

# ── glm53tf lane: TensorFold GLM-5.3-Flash TP=2 on a ring edge ────────────────
# Flint-proven recipe (5624dfc, image b9) on the fire quad. Default pair per
# TF4-SCOPING.md: spark-3ff5 (head, rank 0) + gx10-2ca4 (worker, rank 1), edge C
# (10.10.2.x). API 127.0.0.1:8000 on the head; the LIL lane keeps :8888.
# The LIL lane must be DOWN on both pair nodes (GB10 unified memory) — run
# `just lilquad-down` first if it is up, or accept the preflight abort.

# Stage + launch + verify the TensorFold TP=4 lane (playbook does everything)
start-glm53tf:
	@just tf4-notice || true
	ansible-playbook -i hosts {{TF4_PLAY}}
	@just tf4-notice || true

# Stop the TensorFold lane, assert containers + GPU released
stop-glm53tf:
	ansible-playbook -i hosts glm53tf/stop-glm53tf.yml

# Run-any-time verification (hosts, edge C fabric, RoCE, health, smokes, canary)
check-glm53tf:
	ansible-playbook -i hosts glm53tf/check-glm53tf.yml

# Tail the TensorFold swap/boot log on the TF head
glm53tf-progress:
	ssh -o BatchMode=yes -i /home/flowai/.ssh/spark_agent_key spark@spark-3ff5.local \
	  "tail -f /home/spark/logs/glm53-tf/swap.log"

# ── OS + driver update (run while inference is down; see TP4-OS-DRIVER-UPDATE.md) ──
# asserts GPUs idle, apt upgrade, reboot workers first, verify drivers; then the head
tp4-update:
	ansible-playbook -i hosts glm53tf/update-reboot.yml

# ══ TensorFold TP=4 lane on the fire quad (fork 0900; glm53-tensorfold:tp4) ══
# The ONLY entrypoint for launch + monitoring of this lane (user directive).
# Ring: e116 r0 (API :8000) → dc90 r1 → 3ff5 r2 → 2ca4 r3.
# Boot context 36864 (see TP4-LESSONS.md #9: 1M OOMKills rank 0 at graph capture).
# LIL lane must be DOWN (GB10 unified memory is shared).
TF4_PLAY := "glm53tf/start-glm53tf-tp4.yml"

# tf4-fabric: re-apply the switchless ring fabric (MTU 9000, GID-3, DOCKER-USER,
# compaction=0) — RUN after every reboot and after docker churn. Optional node arg.
tf4-fabric *ARGS="":
	bash quad-launcher/tf4-fabric.sh {{ARGS}}

# tf4-stage: sync recipe + config, verify 92 shards + drafter, build/ship image.
tf4-stage:
	ansible-playbook -i hosts {{TF4_PLAY}} --tags stage

# tf4-prepare: per-rank prepared folders on all 4 nodes (GPU work, ~6 min each).
tf4-prepare:
	ansible-playbook -i hosts {{TF4_PLAY}} --tags prepare

# tf4-up: full lane (stage → prepare → launch → health + smoke). ~35 min cold.
tf4-up:
	@just tf4-notice || true
	ansible-playbook -i hosts {{TF4_PLAY}}
	@just tf4-notice || true

# tf4-down: remove all four rank containers, assert GPUs released.
tf4-down:
	for h in 192.168.1.6 192.168.1.2 192.168.1.28 192.168.1.31; do \
	  ssh -o BatchMode=yes -i /home/flowai/.ssh/spark_agent_key spark@$$h \
	    'docker rm -f glm53-tf-r0 glm53-tf-r1 glm53-tf-r2 glm53-tf-r3 2>/dev/null || true; \
	     test -z "$$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d " ")" && echo "$$h: GPU idle" || echo "$$h: GPU STILL BUSY"'; \
	done

# tf4-status: whole-system view (containers + load phase per rank). --once default.
tf4-status:
	bash quad-launcher/tp4-tf-view.sh --once

# tf4-log N: follow rank N's container log (pull straight off the host).
tf4-log RANK="0":
	#!/usr/bin/env bash
	set -euo pipefail
	case {{RANK}} in
	  0) h=192.168.1.6 ;; 1) h=192.168.1.2 ;; 2) h=192.168.1.28 ;; 3) h=192.168.1.31 ;;
	  *) echo "rank must be 0-3"; exit 2 ;;
	esac
	exec ssh -o BatchMode=yes -i /home/flowai/.ssh/spark_agent_key spark@$h \
	  "docker logs -f --tail 40 glm53-tf-r{{RANK}}"

# tf4-logs: 4-column live tails of all ranks (frozen banner with API health).
tf4-logs *ARGS="":
	bash quad-launcher/tp4-tf-view.sh {{ARGS}}

# tf4-send: consistent QFSP file transfer between nodes (prefers the direct ring
# edge when the pair is cabled, falls back to mgmt LAN; verifies size on receipt).
#   just tf4-send e116 dc90 /home/spark/tp4-ship.tar
#   just tf4-send all 2ca4 /home/spark/notes.md      (fan-in from every node)
tf4-send *ARGS:
	bash quad-launcher/qfsp-send.sh {{ARGS}}

# tf4-test: health + models + 17*19 smoke on the rank-0 API (e116:8888 LAN).
tf4-test:
	#!/usr/bin/env bash
	set -euo pipefail
	base="http://192.168.1.6:8888"
	curl -s -o /dev/null -w 'health => HTTP %{http_code}\n' -m 5 "$base/health" || true
	curl -s -m 10 "$base/v1/models" | python3 -m json.tool | head -12 || true
	curl -s -m 120 "$base/v1/chat/completions" -H "Content-Type: application/json" -d '{
	  "model": "GLM-5.3-Flash-EXL3", "max_tokens": 200, "temperature": 0,
	  "messages": [{"role":"user","content":"17*19=? Answer with just the number."}]
	}' | python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print("smoke:", (m.get("content") or m.get("reasoning") or "").strip()[-40:])' || true

# tf4-host: publish the lane on the LAN via a socat proxy (192.168.1.6:8888 on e116 (client contract)
# -> 127.0.0.1:8888). The engine stays private; the proxy is the publish point and
# survives lane bounces. Verifies health + 17*19 smoke through the proxy from
# flowai (off-node) BEFORE declaring hosted. Idempotent.
tf4-host:
	#!/usr/bin/env bash
	set -euo pipefail
	exec bash quad-launcher/tf4-proxy.sh start

# tf4-unhost: close the published endpoint (engine stays up, bench-safe).
tf4-unhost:
	#!/usr/bin/env bash
	set -euo pipefail
	exec bash quad-launcher/tf4-proxy.sh stop

# tf4-proxy-status: proxy + engine state (non-halting, exit 0 always).
tf4-proxy-status:
	#!/usr/bin/env bash
	exec bash quad-launcher/tf4-proxy.sh status

# tf4-bench: clean RigMark window — tears down the proxy, waits for true idle,
# runs the suite against 127.0.0.1:8888 (no proxy hop), reminds to re-publish.
# Extra env passes through: just tf4-bench SKIP_PREFILL=1 TF4_LABEL=my-label
tf4-bench *ARGS="":
	#!/usr/bin/env bash
	set -euo pipefail
	exec bash quad-launcher/tf4-bench.sh {{ARGS}}

# quick non-halting proxy status banner (sourced by the serve recipes below)
tf4-notice:
	#!/usr/bin/env bash
	pid=$(ssh -o BatchMode=yes -i /home/flowai/.ssh/spark_agent_key spark@192.168.1.6 \
	  "pgrep -f socat -a 2>/dev/null | grep 'TCP-LISTEN:8888,bind=192.168.1.6' | grep -v pgrep | awk '{print \$1}' | head -1" 2>/dev/null || true)
	if [[ -n "$pid" ]]; then
	  echo ">> NOTICE: publish proxy RUNNING (pid $pid) — LAN clients on http://spark-e116.local:8888/v1"
	  echo ">>        External requests WILL hit the lane. For a clean bench: just tf4-unhost"
	else
	  echo ">> NOTICE: publish proxy NOT running — engine is private (bench-safe)."
	  echo ">>        LAN clients have no endpoint. To publish: just tf4-host"
	fi

