# netplan/deprecated — triangle-era fabric files (SUPERSEDED 2026-10-01)

These hold the OLD 3-node triangle addressing that conflicts with the quad ring:

- `spark-e116-enp1s0f0np0.yaml` — 10.10.1.1 on MTU 4092 (ring wants MTU 9000 + 10.10.0.1 on f1)
- `spark-3ff5-fabric.yaml` — 10.10.1.3 on f1 (ring moved it to 10.10.3.2)
- `gx10-2ca4-enp1s0f0np0.yaml` — 10.10.0.2/10.10.2.2 (ring: 10.10.0.3/10.10.2.4)

DO NOT deploy these. The live source of truth is `netplan/quad/*-60-quad-ring.yaml`
(persisted on the nodes at /etc/netplan/60-quad-ring.yaml) plus
`quad-launcher/tf4-fabric.sh` for the runtime state (MTU, ip_forward, DOCKER-USER,
GID-3 checks) that netplan cannot express and docker churn wipes.
