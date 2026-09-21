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

# Stop the FIRE cluster and release ports (old sparks stack untouched)
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
    ssh spark@spark-e116.local "docker ps --filter name=glm53-exl3-tp3 --format '{{.Names}} {{.Status}}'"

# ── TP=3 lane: upstream start-tp3.sh on the head (the supported TP3 path) ──
# TP=3 is NOT a flags-only change: 64 attention/KV heads pad to 66 (local 22),
# vocab 154880 -> 154944 (lcm 64,3), shared-expert 2048 -> 2112, and routed
# moe_intermediate_size=2048 is NOT padded — expert parallel (288/3 = 96 whole
# experts per rank) is the supported route. Drafter cannot shard by 3, so
# DFLASH_DRAFT_TP=1. Upstream launcher owns the patch layer:
#   overlay/tp3/patch_tp3_glm.py   rewrites the image's glm5next/nvidia/model.py
#   overlay/tp3/vllm/*             bind-mounted over vLLM internals
# Recipe tree + env are staged at /home/spark/glm53-tp3 on spark-e116.
tp3-status:
    ssh spark@spark-e116.local 'cd /home/spark/glm53-tp3 && ./start-tp3.sh status'

tp3-start:
    ssh spark@spark-e116.local 'cd /home/spark/glm53-tp3 && ./start-tp3.sh'

tp3-stop:
    ssh spark@spark-e116.local 'cd /home/spark/glm53-tp3 && ./start-tp3.sh stop'

# Follow TP3 logs. Usage:
#   just tp3-logs            frozen-banner console: health/host banner on top +
#                            3-column live tails of head/w1/w2 startup logs
#   just tp3-logs head|w1|w2 single host, banner kept (docker logs -f)
#   just tp3-logs-raw [n]    upstream launcher's own log view (no banner)
tp3-logs WHICH="":
    ./glm/scripts/tp3-tri-view.sh {{WHICH}}

tp3-logs-raw WHICH="":
    ssh -t spark@spark-e116.local 'cd /home/spark/glm53-tp3 && ./start-tp3.sh logs {{WHICH}}'

# Free page cache on all 3 nodes. REQUIRED before a TP3 start: the ~164 GiB
# staging rsync fills the page cache, and on GB10 unified memory that shows up
# as "Free memory on device cuda:0" — rank 2 failed the 0.80 gate at 76 GiB
# free until caches were dropped (43 G used / 89 G cache -> 115 G free).
mem-free:
    #!/usr/bin/env bash
    set -euo pipefail
    for h in spark-e116.local gx10-2ca4.local; do
      ssh -o BatchMode=yes spark@$h 'free -g | awk "/Mem:/{print \"  '"$h"' avail=\"\$7\"G\"}"; sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"'
    done
    ssh -i /home/flowai/.ssh/spark_agent_key -o BatchMode=yes spark@spark-3ff5.local \
      'free -g | awk "/Mem:/{print \"  spark-3ff5 avail=\"\$7\"G\"}"; sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"'

# Stage the TP3-prepared drafter (glm53-tp3-draft/<rev>) on BOTH workers.
# start-tp3.sh's prepare_tp3_draft only runs on the head, but every rank parses
# the speculative config in the mp executor, so workers without it die with
# "Invalid repository ID or local directory ... config.json". Run after any
# drafter revision change, before tp3-start.
tp3-drafter-sync:
    #!/usr/bin/env bash
    set -euo pipefail
    ssh -o BatchMode=yes spark@spark-e116.local 'set -e
      for ip in 10.10.0.2 10.10.1.3; do
        ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new spark@$ip "mkdir -p /home/spark/.cache/huggingface/glm53-tp3-draft"
        rsync -a --info=progress2 -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
          /home/spark/.cache/huggingface/glm53-tp3-draft/ spark@$ip:/home/spark/.cache/huggingface/glm53-tp3-draft/
      done
      echo "drafter staged on both workers"'
