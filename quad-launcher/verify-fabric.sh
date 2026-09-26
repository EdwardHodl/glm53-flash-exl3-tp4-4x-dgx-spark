#!/usr/bin/env bash
# fire-quad — verify the 4-node switchless ring (GIDs, MTU 9000, jumbo pings).
# Stand-in for the recipe's fabric-setup.sh fabric application: the ring here is
# netplan-persisted (netplan/quad/*.yaml), so this only VERIFIES.
set -u
NODES=("spark@192.168.1.6" "spark@192.168.1.2" "spark@192.168.1.28" "spark@192.168.1.31")
NAMES=(spark-e116 gx10-dc90 spark-3ff5 gx10-2ca4)
# per-node HCAs used by the ring (dc90's edge-A end is the P2 cage)
HCAS=("rocep1s0f0 rocep1s0f1" "rocep1s0f0 roceP2p1s0f1" "rocep1s0f0 rocep1s0f1" "rocep1s0f0 rocep1s0f1")
# jumbo-ping targets: [node] -> neighbor fabric IPs
TARGETS=("10.10.1.2 10.10.0.3" "10.10.3.2 10.10.1.1" "10.10.2.4 10.10.3.1" "10.10.0.1 10.10.2.3")

fail=0
for i in 0 1 2 3; do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "${NODES[$i]}" "
    for hca in ${HCAS[$i]}; do
      gid=\$(cat /sys/class/infiniband/\$hca/ports/1/gids/3 2>/dev/null || true)
      ndev=\$(cat /sys/class/infiniband/\$hca/ports/1/gid_attrs/ndevs/3 2>/dev/null || true)
      mtu=\$(cat /sys/class/net/\$ndev/mtu 2>/dev/null || echo 0)
      [[ -z "\$gid" || "\$gid" == "0000:0000:0000:0000:0000:0000:0000:0000" ]] && { echo "BADGID \$hca"; continue; }
      [[ \"\$mtu\" != 9000 ]] && echo \"BADMTU \$hca \$mtu\"
    done
    for t in ${TARGETS[$i]}; do
      ping -c1 -M do -s 8972 -W2 \$t >/dev/null || echo \"BADPING \$t\"
    done
  " 2>&1) || { echo "${NAMES[$i]}: SSH FAIL"; fail=1; continue; }
  bad=$(echo "$out" | grep -E '^BAD' || true)
  if [ -z "$bad" ]; then
    echo "${NAMES[$i]}: OK (2 HCAs gid3/mtu9000, 2 jumbo edges)"
  else
    echo "${NAMES[$i]}: FAIL — $bad"
    fail=1
  fi
done
exit $fail
