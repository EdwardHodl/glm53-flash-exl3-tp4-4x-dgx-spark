# fire — GLM-5.3-Flash-EXL3 TP=3 on the 3-Spark "fire triangle"


> **Public mirror:** the TP=4 TensorFold fork this cluster's quad lane is built on is
> published at [EdwardHodl/glm53-tensorfold-spark-tp4-fork](https://github.com/EdwardHodl/glm53-tensorfold-spark-tp4-fork)
> (fork of jayleaton/glm53-tensorfold-spark; see its FORK-NOTES.md). RigMark receipts for
> the quad TP=4 lane are in `results/rigmark/` and `TP4-HILLCLIMB-RECEIPTS.md`.

> **STATUS 2026-09-26:** this README documents the TP=3 triangle era. The cluster now
> runs the **TP=4 quad** (e116/dc90/3ff5/2ca4, alexellis lane, `quad-launcher/`); see
> `QUAD-ALEXELLIS-PROPOSAL.md`, `TP4-MIGRATION.md` and `quad-launcher/PROGRESS.md`.
> References below to the `../sparks/` kit are historical — that tree was retired 2026-09-26.

Three DGX Sparks in a triangle (like a fire triangle) serving
**GLM-5.3-Flash-EXL3 (TR3-4bpw)** with vLLM mp-executor **TP=3**. Modeled on the
2-node sparks kit (`../sparks/playbooks/glm`); the EXL3 overlay image, runtime
patch layer, chat template and engine knobs are vendored from it unchanged.

| Node | Rank | Role | LAN (192.168.1.x) | Fabric IPs (QSFP) |
|---|---|---|---|---|
| spark-e116.local | 0 | head — torch.dist master + OpenAI API `:8888` | 192.168.1.6 | 10.10.0.1 (f1), 10.10.1.1 (f0) |
| gx10-2ca4.local | 1 | worker (headless) | 192.168.1.31 | 10.10.0.2 (f1), 10.10.2.2 (f0) |
| spark-3ff5.local | 2 | worker (headless) | 192.168.1.28 | 10.10.1.3 (f0), 10.10.2.3 (f1) |

Engine config carried over from the TP=2 kit: exl3 fused MoE quant, fp8 KV
(packed fp8_ds_mla), DFlash2 k=7 speculative (pinned drafter
`7d74cdd881ed7e32c31175984a67823127b66cfe`), 384k context, `--max-num-seqs 4`,
MNBT 3584, 19 runtime overlay patches, vision on (image:4/video:1 capped at
2048 tokens/image). Containers are `fire-glm-head` / `fire-glm-w1` /
`fire-glm-w2` so the kit coexists with the old sparks TP=2 stack until cutover.

## Current state / what is NOT done yet

- **QSFP triangle cabling is NOT done.** Only link A (e116<->2ca4, the existing
  TP=2 fabric) exists. Links B and C need cables + netplan (below). Until then,
  bring up with `just start-glm-exl3 socket` (TCP over the 192.168.1.x LAN —
  functional bring-up, poor TP=3 performance; spark-3ff5's LAN link is
  **wireless** (wlP9s9), so socket mode is barely usable. Wire its `enP7s7`
  into the LAN switch too, or better: finish the QSFP triangle first).
- **spark-3ff5 needs**: `glm53-exl3:e3-v151` image + ~164 GiB weights
  (`just download-glm-exl3` stages all 3 nodes). It has 3149 GB free, docker
  29.2.1, same GB10/driver as the other nodes.
- **Cutover**: fire serves on the same `spark-e116:8888` endpoint as the old
  TP=2 stack — stop the old stack (`cd ../sparks/playbooks && just stop-glm-exl3`)
  before `just start-glm-exl3` (port 8888 collides).

## QSFP triangle cabling — DONE and verified (2026-09-21)

Three point-to-point links, MAC-verified by probe, all ping OK both directions:

| Link | End A | End B | Subnet |
|---|---|---|---|
| link1 | e116 `enp1s0f0np0` (`rocep1s0f0`) | 3ff5 `enp1s0f1np1` (`rocep1s0f1`) | 10.10.1.1 ↔ 10.10.1.3 |
| link2 | e116 `enp1s0f1np1` (`rocep1s0f1`) | 2ca4 `enp1s0f0np0` (`rocep1s0f0`) | 10.10.0.1 ↔ 10.10.0.2 |
| link3 | 2ca4 `enp1s0f1np1` (`rocep1s0f1`) | 3ff5 `enp1s0f0np0` (`rocep1s0f0`) | 10.10.2.2 ↔ 10.10.2.3 |

**The cages are cross-connected — not the f1↔f1 layout the first draft assumed.**
Each GB10 Spark exposes 4 CX7 netdevs (2 cages × 2 PCI paths; every cage answers
on 2 MACs). The old `10.10.0.1/10.10.0.2` pair used to be e116-f1↔2ca4-f1; today
e116-f1 lands on 2ca4-**f0**. Addressing was moved accordingly (e116/2ca4
`/etc/netplan/50-qfsp-static.yaml`, new file on 3ff5) and applied.

Facts pinned by probe:
- All 6 ports `4: ACTIVE`, 200 Gb/s.
- RoCE v2 IPv4 GID index is **3 on every fabric port on all 3 nodes** → a single
  `NCCL_IB_GID_INDEX=3` with `NCCL_IB_HCA=rocep1s0f0^1,rocep1s0f1^1` serves both
  cages per node.
- `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` must be the comma list of both fabric
  interfaces (`enp1s0f0np0,enp1s0f1np1`) — each link is its own /24, so a single
  name strands one peer. The launch scripts now do this.
- e116's `id_ed25519_shared` key is authorized on 3ff5, so staging rsyncs run over
  link1 at 200 Gb/s instead of the LAN/wifi.
- MTU is 1500 on all fabric links (matches the old working TP=2 fabric).

Re-check at any time with `just fabric-check` (6 directed pings).

## Bring-up sequence

### TP=3 lane (current) — upstream `start-tp3.sh`, staged on the head

TP=3 is not a flags-only change. Nothing in GLM-5.3-Flash divides by three:

| what | value | handled by |
|---|---|---|
| attention / KV heads | 64 | `--hf-overrides` pads to 66 (local 22) |
| vocab | 154880 | padded to 154944 (`lcm(64,3)=192`) |
| shared-expert intermediate | 2048 | padded to 2112 |
| routed `moe_intermediate_size` | 2048 | NOT padded — `--enable-expert-parallel` (288/3 = 96 whole experts/rank) |
| DFlash2 drafter | 32 heads / 8 kv | `DFLASH_DRAFT_TP=1` (rank 0 only) |

Recipe + env live at `/home/spark/glm53-tp3` on spark-e116 (`.env`, `.env.tp3`;
reference copies in `fire/recipe-env/`). Commands: `just tp3-status|tp3-start|tp3-stop|tp3-logs`.

Preconditions baked into the justfile:
- `just mem-free` — drop page cache on all 3 nodes before a start.
- `just tp3-drafter-sync` — stage `glm53-tp3-draft/<rev>` on BOTH workers. The
  launcher's `prepare_tp3_draft` only runs on the head, but every rank parses the
  speculative config under the mp executor, so a worker without it dies with
  "Invalid repository ID or local directory ... config.json".
- `GPU_MEM_UTIL=0.70`, `CG_ESTIMATE=1`, `MAX_MODEL_LEN=400000` (see below).

### The GB10 unified-memory trap (cost us 4 failed boots)

These nodes report **121.69 GiB TOTAL** — GPU memory IS host RAM, and page cache
competes with CUDA allocations. Symptoms and fixes, in the order they bite:

1. `ValueError: Free memory on device cuda:0 (76.17/121.69 GiB) ... less than
   desired GPU memory utilization (0.85)` → the ~164 GiB staging rsync left the
   page cache full. Fix: `sync; echo 3 > /proc/sys/vm/drop_caches` (3ff5 went
   43 G used / 89 G cache → 115 G free).
2. Kernel **global OOM** kills `VLLM::Worker_TP` mid-load (check `sudo dmesg -T |
   grep -i oom-kill`): the weight read refills the cache while vLLM holds its
   reservation. Fix: run a trimmer during boot — `for i in $(seq 1 70); do sync;
   echo 1 > /proc/sys/vm/drop_caches; sleep 15; done` (echo 1 = page cache only).
   **Kill the trimmer the moment health=200**: during serving it evicts the
   mmap'd expert weights and every token re-reads GBs from disk — measured
   ~1 tok/s with it running vs ~71 tok/s (DFlash2, k=7) without. It must never
   overlap with serving.
3. `CUDA out of memory. Tried to allocate 2.04 GiB ... 32.06 GiB is free` right
   after "Estimated CUDA graph memory": that 2.04 GiB IS the graph estimate —
   do NOT set `CG_ESTIMATE=0` (it removes the deduction and the capture then
   overruns the budget). Keep `CG_ESTIMATE=1`.
4. Budget arithmetic: at 0.80 util vLLM reserved 97.35 GiB and OOM'd; 0.74 still
   lost rank 2; **0.70 = 85.2 GiB** with ~16 GiB host usage leaves real headroom.
   Weights+overhead ≈ 67.4 GiB/rank, so KV ≈ 17.8 GiB at 0.70; 400k context
   needs ~295 block-ids/request, giving ~2-3 concurrent long requests.
5. The launcher's post-ready `boot-shape-warmup` sweep (DFlash2 ladder +
   3584→65536-token prefill probes) wedged the scheduler on this triangle: every
   probe >248 tokens FAILED and the engine stopped answering real requests
   (health=200 but generation hangs; two sweep instances were holding all 4
   sequence slots). Disabled with `GLM53_BOOT_SHAPE_WARMUP=0` in `.env.tp3`.
6. Cron/bare-request compatibility: upstream defaults `--reasoning-parser
   glm45`, which returns `content: null` for short bare requests (output lands
   in an unparseable reasoning span). The TP2 kit used `glm47` on the head.
   Patched `start-tp3.sh` lines 1476/1617 to `glm47` (pristine copy kept at
   `start-tp3.sh.orig-fire`). Verified: bare request → content + finish=stop.

Check per-node CUDA-visible memory directly:
`docker run --rm --gpus all --entrypoint python3 glm53-exl3:e3-v151 -c "import torch;f,t=torch.cuda.mem_get_info();print(f,t)"`

### TP=2 lane (ancestor, still in this kit)


```bash
cd ~/Workspaces/devops/fire
just ping                # SSH + GPU + HCA state on all 3
just fabric-check        # after cabling: triangle ACTIVE + pings
just build-glm-exl3      # GHCR image pulled+retagged on all 3 nodes
just download-glm-exl3   # ~164 GiB on head, rsync to both workers (LAN)
just start-glm-exl3      # workers (r1, r2) then head; waits for :8888 health + smoke test
just log status          # progress / status on all 3
just health && just test-glm-exl3
```

Fallback before cabling: `just start-glm-exl3 socket` (TCP transport; expect
very slow TP=3 over wifi to 3ff5).

## Cutover (making fire the main inference lane)

1. `cd ../sparks/playbooks && just stop-glm-exl3` (releases e116:8888).
2. `just start-glm-exl3` — fire takes the same endpoint
   `http://spark-e116.local:8888/v1`, model id `GLM-5.3-Flash-EXL3`, tailnet
   HTTPS `https://spark-e116.taile54e9f.ts.net/v1` (configured by the smoke
   play). Cron jobs / OpenAI clients need no changes.
3. Retire the qwen lane on spark-3ff5 (it holds ~98 GB RAM + the port-free
   192.168.1.28 node; `qwen38-sglang` container can stay until 3ff5's rank-2
   weights are staged, then stop it to free RAM before starting fire).

## Post-tune verification (2026-09-22)

Serving-shape round 2, live-verified on all three nodes. Context: the morning
probe battery showed TTFT under 4-5 stream load running 5.8x below the
published bench posture (34k prompt = 137.5 s, 247 tok/s effective) — the
fair-prefill floor (share 0.30 / chunk 256) was starving deferred prefills.

Changes (commit fafe1e8 + this receipt): KV pool pin
`--kv-cache-memory-bytes 20000000000` (~787k tok, 2.46x @320k, ~11 GiB
guaranteed capture slack), `GLM53_INDEXER_WORKSPACE=rightsize` (bench
posture), FAT tiers 1/1/1 (upstream contract), fair-prefill 0.50 / chunk 512,
fabric MTU 4092 (was 1500 since bring-up), playbook MNBT 3584→7168 parity.
3ff5 cleanup: ~80 GB dead qwen images pruned, qwen38 dashboard/keepalive
units disabled.

Boot saga (4 attempts, root causes, both real):

1. rightsize lets the KV pool SELF-FILL the whole utilization budget
   (1.16M tok @0.76); graph capture then exceeds the profiler's estimate and
   UMA refuses the overage against 30+ GiB "free" ("tried 2.35 GiB ... free"
   = accounting refusal, not capacity). Fix = the pin above.
2. A boot-window page-cache trimmer MUST wrap the whole load+capture window
   (~17.5 min, 70x15s) and MUST be sudo-wrapped — a non-sudo loop fails with
   EACCES on every write (2>/dev/null hides it). Kill at health=200.
3. Node reboots between attempts cleared UMA fragmentation from the failed
   boots. One-shot `mem-free` is NOT sufficient at 0.76+rightsize.

Receipt (same probe battery, thinking off, temp 0):

| Probe | pre-tune | post-tune | note |
|---|---|---|---|
| tiny ~190 tok, under load | 19.7 s | 1.2 s | 2 bg streams |
| ~5k prompt | 61.1 s | 4.3 s | |
| ~16.4k prompt | 74.3 s (220 tok/s) | 12.9 s (1,273 tok/s) | |
| ~34k prompt | 137.5 s (247 tok/s) | 20.1 s (1,689 tok/s) | at/above the 1,430-1,590 tok/s TP2 bench |
| 6x16.4k simultaneous | timeout | all TTFT 19.47 s | ~5k tok/s aggregate, uniform admission |

Live state: pool 786,634 tok (2.46x), util 0.76, seqs 6, ctx 320k, graphs
4/4 captured with zero OOM, tiny TTFT 0.45-0.49 s unloaded, smoke
`OK FIRE GLM 5.3 EXL3 LIVE TP3` finish=stop, prefix-cache multi-page hits
confirmed. Next queued experiment: `GLM53_ADAPTIVE_K=ema` (+5-21% decode at
temp 0 on TP2 receipts) — needs 3/5-token capture shapes; the KV pin's
~11 GiB capture slack is what was missing when the 40/48 ladder OOM'd.

## Layout

```
fire/
├── justfile                  # ping / fabric-check / build / download / start / stop / log / health / test
├── hosts                     # ini inventory (ssh keys)
├── glm/
│   ├── exl3/                 # vendored overlay + Dockerfile (from sparks kit; e3-v151)
│   └── scripts/              # launch-glm-exl3-{head,worker}.sh, glm-exl3-progress.sh
└── playbooks/
    ├── inventory.yml         # 3 nodes, node_rank, lan_ip, fabric_{ips,ifs,hcas}
    ├── fabric-check.yml      # QSFP triangle verification
    ├── build-glm-exl3.yml    # image on all 3 (GHCR pull or local build + ship)
    ├── download-glm-exl3.yml # weights on head + rsync to both workers
    ├── start-glm-exl3.yml    # TP=3 bring-up (7 plays, mirrors sparks kit)
    └── stop-glm-exl3.yml
```

## Knobs worth re-tuning after first light

- `dflash_tokens` (7): TP=2 receipts say k=7 wins single-stream; re-measure at
  TP=3 (2 consumers + cron traffic) via mean accepted length.
- ~~`max_num_seqs` / `max_num_batched_tokens` (4 / 3584)~~ → **done 2026-09-22:
  6 / 7168** live-verified (see Post-tune verification above). Next lever:
  `GLM53_ADAPTIVE_K=ema`.
- ~~`max_model_len` (384000)~~ → **320000** as of 2026-09-21 (pool/seqs
  arithmetic; queue threshold ~131k avg tokens/session). Raise only after
  re-pinning the KV pool with a fresh "Maximum concurrency" line in front of
  you — the current pin (20e9) sizes the pool to 320k-era receipts.
- `GLM53_SPINWAIT_MS` (default 2ms in the runtime knob form): TP=2 receipts
  measured 16ms at +0.95% median decode / −85.3% EngineCore CPU. Unmeasured at
  TP=3; candidate for the same maintenance window as adaptive-k.
