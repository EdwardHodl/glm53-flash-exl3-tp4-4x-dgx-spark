# fire-quad — netplan for the 4-node QSFP ring (rewired 2026-09-25)

Ring discovered by MAC probe (all four edges verified both directions, this session):

```
        e116 ────── dc90            edge A  10.10.1.0/24
        │             │             edge B  10.10.3.0/24
        │             │             edge C  10.10.2.0/24 (existing link3)
        2ca4 ────── 3ff5            edge D  10.10.0.0/24 (existing link2)
```

Ring order: **e116 (rank 0) → dc90 (rank 1) → 3ff5 (rank 2) → 2ca4 (rank 3) → e116**.

| edge | end A | end B | subnet | status at probe time |
|---|---|---|---|---|
| A | e116 `enp1s0f0np0` | dc90 `enP2p1s0f1np1` | 10.10.1.0/24 | UP, ping OK (1472) |
| B | dc90 `enp1s0f0np0` | 3ff5 `enp1s0f1np1` | 10.10.3.0/24 (new) | UP, ping OK (1472), proven with temp IPs |
| C | 3ff5 `enp1s0f0np0` | 2ca4 `enp1s0f1np1` | 10.10.2.0/24 | UP, 4054-byte jumbo OK |
| D | 2ca4 `enp1s0f0np0` | e116 `enp1s0f1np1` | 10.10.0.0/24 | UP, 4054-byte jumbo OK |

Final addressing (last octet = rank + 1, alexellis convention):

| node | rank | f0 addr | f1 addr | MTU target |
|---|---|---|---|---|
| spark-e116 | 0 | 10.10.1.1/24 (f0, edge A) | 10.10.0.1/24 (f1, edge D) | 9000 both |
| gx10-dc90 | 1 | 10.10.3.1/24 (**f0**, edge B) | 10.10.1.2/24 (**P2f1**, edge A) | 9000 both |
| spark-3ff5 | 2 | 10.10.2.3/24 (f0, edge C) | 10.10.3.2/24 (f1, edge B) | 9000 both |
| gx10-2ca4 | 3 | 10.10.0.3/24 (f0, edge D) | 10.10.2.4/24 (f1, edge C) | 9000 both |

Note dc90's edge-A end is **`enP2p1s0f1np1`** — the second PCIe view of cage P2, which
dc90's previous (deepseek-era) netplan already addressed. Keep it: the cable is what it is.

## Faults found during the trace (must fix in the new netplan)

1. **Duplicate IP**: dc90 `enp1s0f1np1` held `10.10.0.2/24` — the same address as 2ca4's
   `enp1s0f0np0` on the same subnet. ARP was flapping between two hosts on 10.10.0.0/24.
2. **dc90 f0 unaddressed** (edge B had no L3 path until this session's temp IPs).
3. **MTU mismatch**: dc90's two cable ports are at 1500; everything else 4092. Target 9000
   on all eight ports (2.7× all-reduce penalty below that — alexellis gotcha).
4. GID index 3: correct IPv4-mapped RoCE v2 GIDs on 7/8 ports; dc90 f0's GID slot was
   empty (no IPv4 assigned). Re-check after applying netplan — a mode switch can leave an
   empty GID slot (reboot the node if it doesn't repopulate).

## Wiring cheatsheet for the next probe

Peer MACs scrubbed from the public repo (map kept in the ops memory backup,
`memory/backup/fire-quad-mac-map-*.md`). Resolve at runtime with
`ip link show <iface>` on the peer or `ip neigh show dev <iface>`.

| node | port | peer MAC to expect | peer host |
|---|---|---|---|
| e116 f0 | `enp1s0f0np0` | `<see backup>` | dc90 P2f1 |
| e116 f1 | `enp1s0f1np1` | `<see backup>` | 2ca4 f0 |
| dc90 f0 | `enp1s0f0np0` | `<see backup>` | 3ff5 f1 |
| dc90 P2f1 | `enP2p1s0f1np1` | `<see backup>` | e116 f0 |
| 3ff5 f0 | `enp1s0f0np0` | `<see backup>` | 2ca4 f1 |
| 3ff5 f1 | `enp1s0f1np1` | `<see backup>` | dc90 f0 |
| 2ca4 f0 | `enp1s0f0np0` | `<see backup>` | e116 f1 |
| 2ca4 f1 | `enp1s0f1np1` | `<see backup>` | 3ff5 f0 |
