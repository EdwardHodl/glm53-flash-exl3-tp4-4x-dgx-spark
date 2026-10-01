#!/usr/bin/env bash
# tf4-fabric.sh — re-apply the switchless-TP4 ring fabric on the fire quad.
# Adapted from the alexellis switchless recipe (glm53flash/glm-5.3-flash-4x-dgx-spark-
# switchless/scripts/fabric-setup.sh), site-fit to the fire ring (MAC-verified 2026-09-25):
#   edge A e116 f0 10.10.1.1 <-> dc90 enP2p1s0f1 10.10.1.2   (10.10.1.0/24)
#   edge B dc90 f0 10.10.3.1 <-> 3ff5 f1 10.10.3.2           (10.10.3.0/24)
#   edge C 3ff5 f0 10.10.2.3 <-> 2ca4 f1 10.10.2.4           (10.10.2.0/24)
#   edge D 2ca4 f0 10.10.0.3 <-> e116 f1 10.10.0.1           (10.10.0.0/24)
#
# RUN THIS after EVERY reboot and after ANY docker up/down churn — the ring
# addressing, MTU, ip_forward and DOCKER-USER chain are runtime-only and get wiped
# (alexellis gotchas.md #1). Persistent L2/L3 lives in netplan/quad/*-60-quad-ring.yaml.
#
# What it does per node: GPU-idle gate -> NM unmanage on the rails -> flush+MTU 9000
# -> addresses -> ip_forward + DOCKER-USER accept -> verify (RoCE-v2 GID-3 non-empty,
# MTU 9000, addr match) -> jumbo ping both edges from each node.
set -euo pipefail

KEY=/home/flowai/.ssh/spark_agent_key
GID_INDEX="${GID_INDEX:-3}"

# rank -> mgmt IP, (pair-rail-if pair-rail-addr pair-rail-hca) (cross-rail ...)
# Rail roles per node follow the cabling; "pair" = f1-role edge, "cross" = f0-role.
#   e116: cross=f0 (edge A), pair=f1 (edge D) | dc90: cross=f0 (edge B), pair=P2f1 (edge A)
#   3ff5: cross=f0 (edge C), pair=f1 (edge B) | 2ca4: cross=f0 (edge D), pair=f1 (edge C)
# mgmt bootstrap iface differs per node (enP7s7 on e116, wlP9s9 on the others).
node_cfg() {
  case $1 in
    e116) echo "192.168.1.6  enp1s0f0np0 10.10.1.1 rocep1s0f0  enp1s0f1np1 10.10.0.1 rocep1s0f1" ;;
    dc90) echo "192.168.1.2  enp1s0f0np0 10.10.3.1 rocep1s0f0  enP2p1s0f1np1 10.10.1.2 roceP2p1s0f1" ;;
    3ff5) echo "192.168.1.28 enp1s0f0np0 10.10.2.3 rocep1s0f0  enp1s0f1np1 10.10.3.2 rocep1s0f1" ;;
    2ca4) echo "192.168.1.31 enp1s0f0np0 10.10.0.3 rocep1s0f0  enp1s0f1np1 10.10.2.4 rocep1s0f1" ;;
    *) echo "unknown node $1 (use e116|dc90|3ff5|2ca4 or all)" >&2; exit 2 ;;
  esac
}

ssh_node() { ssh -n -o BatchMode=yes -o ConnectTimeout=12 -i "$KEY" "spark@$1"; }

# jumbo targets: each node pings its two direct neighbours (the only peers a ring
# rank exchanges collectives with — alexellis fabric.md).
jumbo_targets() {
  case $1 in
    e116) echo "10.10.1.2 10.10.0.3" ;;   # dc90 (edge A), 2ca4 (edge D)
    dc90) echo "10.10.1.1 10.10.3.2" ;;   # e116 (edge A), 3ff5 (edge B)
    3ff5) echo "10.10.3.1 10.10.2.4" ;;   # dc90 (edge B), 2ca4 (edge C)
    2ca4) echo "10.10.2.3 10.10.0.1" ;;   # 3ff5 (edge C), e116 (edge D)
  esac
}

apply_node() {
  local n=$1; read -r mgmt pair_if pair_addr pair_hca cross_if cross_addr cross_hca <<< "$(node_cfg "$n")"
  echo "== $n ($mgmt): rails $cross_if ($cross_addr, $cross_hca) / $pair_if ($pair_addr, $pair_hca)"
  local tmp=$(mktemp /tmp/tf4-fabric.XXXX.sh)
  {
    echo '#!/usr/bin/env bash'
    echo "set -e"
    echo "N='$n'"
    echo "PAIR_IF=$pair_if PAIR_ADDR=$pair_addr PAIR_HCA=$pair_hca"
    echo "CROSS_IF=$cross_if CROSS_ADDR=$cross_addr CROSS_HCA=$cross_hca"
    echo "GID_INDEX=$GID_INDEX"
    cat <<'REMOTE'
# GPU-idle gate: refuse to touch fabric under a running model (alexellis quiescent()).
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | grep -c . || true)
[ "$pids" = 0 ] || { echo "$N: GPU busy ($pids pids) — stop appliances first" >&2; exit 1; }
# Stop NetworkManager fighting for the rails.
for d in $PAIR_IF $CROSS_IF; do sudo nmcli device set $d managed no 2>/dev/null || true; done
# Flush stale, MTU 9000, bring up (MTU 1500 = silent 2.7x penalty — never skip).
sudo ip addr flush dev $PAIR_IF 2>/dev/null || true
sudo ip addr flush dev $CROSS_IF 2>/dev/null || true
sudo ip link set $PAIR_IF mtu 9000 up
sudo ip link set $CROSS_IF mtu 9000 up
sudo ip addr add $PAIR_ADDR/24 dev $PAIR_IF
sudo ip addr add $CROSS_ADDR/24 dev $CROSS_IF
# Forwarding + DOCKER-USER accept (docker churn drops these — re-assert).
sudo sysctl -qw net.ipv4.ip_forward=1 || true
for IF in $PAIR_IF $CROSS_IF; do
  sudo iptables -C DOCKER-USER -i $IF -j ACCEPT 2>/dev/null || sudo iptables -I DOCKER-USER -i $IF -j ACCEPT
  sudo iptables -C DOCKER-USER -o $IF -j ACCEPT 2>/dev/null || sudo iptables -I DOCKER-USER -o $IF -j ACCEPT
done
# compaction gotcha (drowzeys #1): re-assert while we are here (reboot wipes it).
sudo sysctl -qw vm.compaction_proactiveness=0 || true
check_rail() {
  hca=$1; dev=$2; want=$3
  state=$(cat /sys/class/infiniband/$hca/ports/1/state 2>/dev/null || echo MISSING)
  [ "$state" = "4: ACTIVE" ] || { echo "$N: $hca not ACTIVE ($state)" >&2; exit 1; }
  gid=$(cat /sys/class/infiniband/$hca/ports/1/gids/$GID_INDEX 2>/dev/null || true)
  [ "$gid" != "0000:0000:0000:0000:0000:0000:0000:0000" ] && [ -n "$gid" ] || { echo "$N: $hca GID $GID_INDEX empty — reboot node, re-apply, rerun" >&2; exit 1; }
  mtu=$(cat /sys/class/net/$dev/mtu)
  [ "$mtu" = 9000 ] || { echo "$N: $dev MTU $mtu != 9000" >&2; exit 1; }
  ip -4 -o addr show dev $dev | grep -q "$want/" || { echo "$N: $dev missing $want" >&2; exit 1; }
}
check_rail $PAIR_HCA $PAIR_IF $PAIR_ADDR
check_rail $CROSS_HCA $CROSS_IF $CROSS_ADDR
echo "   $N: rails clean (GID-3 RoCE v2 non-empty, MTU 9000, addrs OK)"
REMOTE
  } > "$tmp"
  scp -q -o BatchMode=yes -i "$KEY" "$tmp" "spark@$mgmt:/tmp/tf4-fabric-node.sh"
  ssh_node "$mgmt" "bash /tmp/tf4-fabric-node.sh"
  rm -f "$tmp"
  # Jumbo pings on both edges (necessary, not sufficient — the real proof is a
  # completed NCCL collective; see alexellis fabric.md silent-failure #2).
  for t in $(jumbo_targets "$n"); do
    ssh_node "$mgmt" "ping -q -M do -s 8972 -c 2 -W 2 $t >/dev/null" || { echo "$n: jumbo ping $t FAILED" >&2; exit 1; }
  done
  echo "   $n: jumbo OK"
}

nodes="${1:-all}"
if [ "$nodes" = "all" ]; then nodes="e116 dc90 3ff5 2ca4"; fi
for n in $nodes; do apply_node "$n"; done
echo "fire ring fabric applied + verified on: $nodes"
