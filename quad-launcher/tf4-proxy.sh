#!/usr/bin/env bash
# tf4-proxy: manage the LAN publish proxy for the TP4 lane on spark-e116.
#
# Design: the engine binds 127.0.0.1:8888 (private, bench-clean). A socat proxy
# on the head publishes it on the LAN as 0.0.0.0:8889. Benchmarks run against
# 127.0.0.1:8888 directly (no proxy hop, matches rigmark's "no proxy" rule);
# clients use 192.168.1.6:8889. The proxy survives lane bounces (rm->start).
#
# Commands: start | stop | status
#   start  - idempotent; installs socat if missing, verifies health + smoke
#   stop   - kills the proxy, asserts it is gone
#   status - prints proxy state + engine state (exit 0 always)

set -uo pipefail
HEAD=192.168.1.6
SSH="ssh -o BatchMode=yes -i /home/flowai/.ssh/spark_agent_key spark@$HEAD"
KEY=/home/flowai/.ssh/spark_agent_key

proxy_pid() { $SSH "pgrep -f 'socat' -a 2>/dev/null | grep 'TCP-LISTEN:8889' | grep -v pgrep | awk '{print \$1}' | head -1" 2>/dev/null; }

engine_health() { $SSH "curl -s -m5 -o /dev/null -w %{http_code} http://127.0.0.1:8888/health 2>/dev/null" 2>/dev/null; }

case "${1:-status}" in
  start)
    up=$(engine_health)
    [[ "$up" == "200" ]] || { echo "ENGINE DOWN: loopback health=$up — run: just start-glm53tf"; exit 1; }
    # install socat if missing (best-effort, needs passwordless sudo which e116 has)
    if ! $SSH "command -v socat >/dev/null" 2>/dev/null; then
      echo "installing socat on head..."
      $SSH "sudo -n apt-get install -y socat >/dev/null 2>&1" || { echo "socat install failed"; exit 1; }
    fi
    pid=$(proxy_pid)
    if [[ -n "$pid" ]]; then
      echo "PROXY ALREADY RUNNING (pid $pid) — idempotent, nothing to do"
    else
      $SSH "nohup socat TCP-LISTEN:8889,bind=0.0.0.0,fork,reuseaddr TCP:127.0.0.1:8888 >/dev/null 2>&1 & echo \$!" >/dev/null
      sleep 1
      pid=$(proxy_pid)
      [[ -n "$pid" ]] || { echo "PROXY FAILED TO START"; exit 1; }
      echo "PROXY STARTED (pid $pid)"
    fi
    # verify the published endpoint answers health off-node
    code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "http://$HEAD:8889/health")
    [[ "$code" == "200" ]] || { echo "PUBLISH VERIFY FAILED: LAN health via proxy = HTTP $code"; exit 1; }
    ans=$(curl -s -m 300 "http://$HEAD:8889/v1/chat/completions" -H "Content-Type: application/json" -d '{
      "model": "GLM-5.3-Flash-EXL3", "max_tokens": 40, "temperature": 0,
      "messages": [{"role":"user","content":"17*19=? Answer with just the number."}]
    }' | python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print((m.get("content") or "").strip())' 2>/dev/null)
    echo "PUBLISH SMOKE: $ans"
    [[ "$ans" == *323* ]] || { echo "PUBLISH SMOKE WRONG ANSWER: $ans"; exit 1; }
    echo "HOSTED: TP4 lane published on http://$HEAD:8889/v1 (engine private on 127.0.0.1:8888)"
    ;;
  stop)
    pid=$(proxy_pid)
    if [[ -n "$pid" ]]; then
      $SSH "kill $pid" && echo "PROXY STOPPED (was pid $pid)"
    else
      echo "proxy not running (nothing to stop)"
    fi
    sleep 1
    left=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://$HEAD:8889/health" 2>/dev/null) || left="000"
    [[ "$left" != "200" ]] && echo "PUBLISHED ENDPOINT CLOSED (LAN unreachable as expected)" || echo "WARNING: :8889 still answering"
    ;;
  status)
    pid=$(proxy_pid)
    up=$(engine_health)
    if [[ -n "$pid" ]]; then
      echo "PROXY:    RUNNING (pid $pid) — LAN 192.168.1.6:8889 -> 127.0.0.1:8888"
    else
      echo "PROXY:    NOT RUNNING — engine is PRIVATE (bench-safe), LAN clients have no endpoint"
    fi
    if [[ "$up" == "200" ]]; then
      echo "ENGINE:   UP (127.0.0.1:8888)"
    else
      echo "ENGINE:   DOWN (loopback health=$up)"
    fi
    ;;
  *)
    echo "usage: tf4-proxy.sh start|stop|status"; exit 2 ;;
esac
