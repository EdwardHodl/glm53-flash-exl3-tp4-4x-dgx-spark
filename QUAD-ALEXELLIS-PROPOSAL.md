# fire-quad — adoption of alexellis 4x-switchless recipe (staged vs required)

Date: 2026-09-26. Status: PROPOSAL — nothing downloaded/rebuilt/restarted; all 4 crashed TP4 containers cleanly stopped
(verified: 0 tp4 containers, 0 GPU processes on all 4 nodes).

## Target configuration (per Edward's instruction — alexellis recipe, as closely as possible)

| item | REQUIRED (alexellis recipe) | STAGED / current state | action |
|---|---|---|---|
| Weights | `LibertAIDAI/GLM-5.3-Flash-NVFP4` @ `9e0d74e3` , ~181.3 GiB, at `$HOME/glm53-flash-nvfp4/` on every rank | NOT present on any node (EXL3 TR3-4bpw is staged instead) | **download 181 GiB x4** (or x1 + fabric rsync) |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2` at `$HOME/glm53-dflash2-draft/model.safetensors` | DFlash2 @ `7d74cdd8` present on all nodes in HF-cache layout | small copy/flat-layout link per node (~2.2 GB) |
| Image | `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (~31 GB) | e116 & 2ca4 have `sm121-v8` (older, no DFlash2 support needed here); dc90/3ff5 have NONE | **docker pull on 1 node + fabric fan-out** |
| Patched NCCL | switchless-nccl v0.0.1 (`libnccl.so.2.30.7`, SHA `78cb8387…`) at `$HOME/nccl-switchless-v0.0.1/` | ✅ INSTALLED on all 4 nodes, SHA-verified | none |
| Chat template | `templates/chat_template.jinja` from the alexellis repo (SHA `0c4099f3…`) | present in local repo clone | stage to each node |
| Launcher | `scripts/rank-launcher.sh <rank>` with site values | local repo clone ready; needs site edit (MASTER=192.168.1.6, PORT, SSH user/keys) | edit + distribute |
| Fabric | 4-edge ring, MTU 9000, routed diagonals, DOCKER-USER ACCEPT, GID index 3 | ✅ DONE and verified this session (8/8 jumbo pings, full-mesh pings, GIDs IPv4-mapped) | none |
| Serve args | TP4 fixed block: util 0.85, ctx 262144, seqs 6, block 2304, marlin, fp8_e4m3 KV 12 GiB/rank, DFlash2 k=7, glm47+glm45, master-port 29520 | (replaces EXL3 knob set: util 0.75, ctx 320k, seqs 8, k=3, KV 24 GB) | adopt as-is |

## Key deltas vs the abandoned EXL3-TP4 attempt

1. Quant/weights change: EXL3 TR3-4bpw → NVFP4 W4A16 (LibertAIDAI). ~181 GiB vs 164 GiB per node.
2. Image change: glm53-exl3:e3-v152 → tonyd2wild `sm121-v11-dflash2` (public GHCR; contains the sm121 kpool fix).
3. NCCL: same switchless v0.0.1 lib, but used exactly as alexellis does — installer-placed at `$HOME/nccl-switchless-v0.0.1`,
   mounted `/opt/patched-nccl` + `LD_PRELOAD` + `VLLM_NCCL_SO_PATH`, `NCCL_ALGO=Ring` etc. (already in place from the EXL3
   trail; provenance identical).
4. KV: 12 GiB/rank pin (alexellis qualified) → 1,576,246-token pool, 6x @262k. Context drops 320k → 262k.
5. Speculative: DFlash2 k=7 static (drafter must be present on all ranks — already true).
6. Serving endpoint: alexellis default API port 8000; fire crons/clients currently target 8888. DECISION NEEDED.

## Risks / notes

- The 7 failed EXL3 boots were launcher/NCCL-transport issues; the alexellis recipe is the exact pairing (image+NCCL+args)
  that was RigMark-validated on a 4-node GB10 ring — highest-probability path to a working quad.
- 2ca4 disk: 121G free < 181.3 GiB required. Options: (a) prune old images/weights on 2ca4 (~62G needed), or
  (b) stage weights only on 3 nodes and NFS-export from head (alexellis quickstart supports NFS share).
- Weights download: head e116 pulls at ~81 MB/s from HF CDN → ~40 min; fabric fan-out to workers at 200G ≈ minutes each.
- LibertAIDAI main branch has moved past 9e0d74e3 (current caca4e6a); recipe pins 9e0d74e3 — download with `--revision`
  to match the recipe exactly. (Note: tonyd2wild #23 flagged LibertAIDAI builds for ModelOpt token corruption on their
  stack; alexellis's TP4 receipts were measured on exactly this checkpoint, so we follow the recipe as instructed.)

## Execution checklist (on go-ahead)

1. [ ] Pull `sm121-v11-dflash2` on e116; fan out via docker save | ssh load over fabric (dc90/3ff5/2ca4).
2. [ ] Download LibertAIDAI NVFP4 @ 9e0d74e3 to e116 `~/glm53-flash-nvfp4/` (~40 min), verify config.json + shard count.
3. [ ] rsync weights to 3 workers over fabric edges (2ca4 pending disk decision); stage drafter flat copy + chat template.
4. [ ] Edit alexellis `rank-launcher.sh` site values (MASTER=192.168.1.6, MGMT_IF per node, PORT, SSH_USER, keys,
       CONTAINER_NAME=glm53_tp4, paths) + distribute; add `ring.txt`/nothing else — no other changes.
5. [ ] Launch workers 3→2→1 then head 0; gate: needle + tool-call + warm decode (their scripts/gate.sh).
6. [ ] Cutover: stop tp3 lane (already stopped), point :8888 (or chosen port) at the new head container.
