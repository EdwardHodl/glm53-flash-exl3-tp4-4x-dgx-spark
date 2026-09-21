#!/usr/bin/env bash
# glm-exl3-progress.sh — FIRE kit container status / log streaming (3 nodes)
#
#   ./glm-exl3-progress.sh            tail head logs (last 100, live follow)
#   ./glm-exl3-progress.sh w1         tail gx10-2ca4 worker logs
#   ./glm-exl3-progress.sh w2         tail spark-3ff5 worker logs
#   ./glm-exl3-progress.sh status     one-shot status of all 3 nodes + API readiness
#
# NOTE: vLLM is launched detached (exec -d + nohup > /tmp/fire-glm-*.log &),
# so stdout goes to the file inside the container, NOT to `docker logs`.
# Plain `docker logs -f` follows an empty stream and hangs silently.
set -euo pipefail

KEY=""
HEAD_HOST=spark-e116.local
W1_HOST=gx10-2ca4.local
W2_HOST=spark-3ff5.local
W2_KEY="-i /home/flowai/.ssh/spark_agent_key"
HEAD_CT=fire-glm-head
W1_CT=fire-glm-w1
W2_CT=fire-glm-w2

_ssh() { # host, key-args, cmd...
  local host=$1; shift
  local keyargs=$1; shift
  ssh -o BatchMode=yes -o ConnectTimeout=8 $keyargs "$host" "$@"
}

_tail() { # host, keyargs, container, logfile
  _ssh "$1" "$2" \
    "docker exec $3 tail -n 100 -f $4 2>&1" || true
}

# Log lives INSIDE the container (nohup redirect), not on the host.
_progress() { # host, keyargs, container, logfile
  _ssh "$1" "$2" "
    docker ps --filter name=$3 --format '  {{.Names}}  {{.Status}}';
    docker exec $3 bash -c '
      if [ -f $4 ]; then
        grep -aE \"Loading safetensors shards|Completed|engine is ready|Uvicorn running|ERROR|Traceback\" $4 | tail -3;
        echo \"  (log $4, \$(wc -l < $4) lines)\";
      else
        echo \"  no log yet: $4\";
      fi
    ' 2>/dev/null || true" || true
}

case "${1:-head}" in
  status)
    echo "=== head (${HEAD_HOST} / ${HEAD_CT}) ==="
    _ssh "$HEAD_HOST" "$KEY" "
      docker ps --filter name=$HEAD_CT --format '  {{.Names}}  {{.Status}}';
      HTTP=\$(curl -sf -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8888/health 2>/dev/null || echo 000);
      echo \"  api :8888/health => \$HTTP\";
    " || true
    _progress "$HEAD_HOST" "$KEY" "$HEAD_CT" /tmp/fire-glm-head.log
    echo "=== worker 1 (${W1_HOST} / ${W1_CT}) ==="
    _progress "$W1_HOST" "$KEY" "$W1_CT" /tmp/fire-glm-w1.log
    echo "=== worker 2 (${W2_HOST} / ${W2_CT}) ==="
    _progress "$W2_HOST" "$W2_KEY" "$W2_CT" /tmp/fire-glm-w2.log
    ;;
  w1)
    _tail "$W1_HOST" "$KEY" "$W1_CT" /tmp/fire-glm-w1.log
    ;;
  w2)
    _tail "$W2_HOST" "$W2_KEY" "$W2_CT" /tmp/fire-glm-w2.log
    ;;
  *)
    _tail "$HEAD_HOST" "$KEY" "$HEAD_CT" /tmp/fire-glm-head.log
    ;;
esac
