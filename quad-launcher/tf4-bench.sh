#!/usr/bin/env bash
# tf4-bench: clean RigMark window for the TP4 lane — NO external requests.
# Flow: ensure engine is loopback-only -> assert proxy (if any) torn down ->
#       wait for true idle -> run rigmark against 127.0.0.1:8888 -> publish reminder.
#
# Usage: tf4-bench.sh [extra rigmark env as VAR=VAL ...]
#   e.g. just tf4-bench SKIP_PREFILL=1

set -euo pipefail
HEAD=192.168.1.6
SSH="ssh -o BatchMode=yes -i /home/flowai/.ssh/spark_agent_key spark@$HEAD"

# 1. engine must be up
up=$($SSH 'curl -s -m5 -o /dev/null -w %{http_code} http://127.0.0.1:8888/health 2>/dev/null' || true)
[[ "$up" == "200" ]] || { echo "ENGINE DOWN (health=$up) — run: just start-glm53tf"; exit 1; }

# 2. engine must be bound to loopback (HOST=127.0.0.1 in installed config)
bind=$($SSH 'grep -E "^HOST=" ~/glm53-tf/config/prod-tp4.env || true')
if [[ "$bind" != "HOST=127.0.0.1" ]]; then
  echo "WARNING: config bind is '$bind' (want HOST=127.0.0.1 for a clean bench window)."
  echo "  Continuing anyway — but if the engine answers on the LAN, the bench is contaminated."
fi

# 3. kill any publish proxy — zero external requests during the window
pid=$($SSH "pgrep -f socat -a 2>/dev/null | grep 'TCP-LISTEN:8889' | grep -v pgrep | awk '{print \$1}' | head -1" 2>/dev/null || true)
if [[ -n "$pid" ]]; then
  echo "Tearing down publish proxy (pid $pid)..."
  $SSH "kill $pid" 2>/dev/null || true
  sleep 1
fi
pid2=$($SSH "pgrep -f socat -a 2>/dev/null | grep 'TCP-LISTEN:8889' | grep -v pgrep | awk '{print \$1}' | head -1" 2>/dev/null || true)
[[ -z "$pid2" ]] || { echo "PROXY STILL RUNNING (pid $pid2) — aborting to avoid contamination"; exit 1; }
# verify LAN endpoint is actually closed
lan_code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://$HEAD:8888/health" 2>/dev/null) || lan_code="000"
[[ "$lan_code" != "200" ]] && echo "LAN :8888 CLOSED (bench-safe)" || echo "WARNING: :8888 still answers on LAN — bind is not loopback"

# 4. wait for true idle (inflight 0 for 2 consecutive checks, 15s apart)
for i in 1 2 3 4 5 6 7 8; do
  inflight=$($SSH 'curl -s -m5 http://127.0.0.1:8888/health | python3 -c "import json,sys; print(json.load(sys.stdin).get(\"inflight\", 99))"' 2>/dev/null || echo 99)
  if [[ "$inflight" == "0" ]]; then
    sleep 15
    inflight2=$($SSH 'curl -s -m5 http://127.0.0.1:8888/health | python3 -c "import json,sys; print(json.load(sys.stdin).get(\"inflight\", 99))"' 2>/dev/null || echo 99)
    [[ "$inflight2" == "0" ]] && { echo "IDLE CONFIRMED (inflight 0 across 15s)"; break; }
  fi
  echo "waiting for idle (inflight=$inflight)... attempt $i/8"
  sleep 20
done
inflight_final=$($SSH 'curl -s -m5 http://127.0.0.1:8888/health | python3 -c "import json,sys; print(json.load(sys.stdin).get(\"inflight\", 99))"' 2>/dev/null || echo 99)
[[ "$inflight_final" == "0" ]] || { echo "NOT IDLE (inflight=$inflight_final) — aborting, bench would be contaminated"; exit 1; }

# 5. run rigmark, streaming output
label="${TF4_LABEL:-glm53-flash-exl3-tensorfold-tp4-bench}"
cid="${COMPARISON_ID:-2026-10-fire-tp4-hillclimb}"
echo "=== RIGMARK starting (label=$label, comparison=$cid) ==="
$SSH "cd ~/glm53-tf && env LABEL_TENSORFOLD='$label' COMPARISON_ID='$cid' $* bash scripts/rigmark/run.sh tensorfold http://127.0.0.1:8888" 2>&1 | tail -40
rc=${PIPESTATUS[0]}
echo "=== RIGMARK done (rc=$rc) — results in ~/glm53-tf/results/rigmark/ on the head ==="
echo "NOTE: publish proxy is DOWN. When benching is finished, re-publish with: just tf4-host"
exit "$rc"
