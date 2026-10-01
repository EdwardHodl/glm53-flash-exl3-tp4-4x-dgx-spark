# TP4 Lessons Learned — TensorFold fork on the fire quad (2026-09-30 → 10-01)

Running log of every failure, root cause, and fix from the TP=4 bring-up. Written
per user request: "record in a log the lessons learned and gotchas and tips and
pitfalls encountered so far."

## A. Fabric / NCCL

1. **NCCL needs ring-constrained path selection on our 4-edge ring.** The fire
   cabling is 4 disjoint /24s (A e116↔dc90 10.10.1.x, B dc90↔3ff5 10.10.3.x,
   C 3ff5↔2ca4 10.10.2.x, D 2ca4↔e116 10.10.0.x). Non-adjacent rank pairs share
   NO subnet, so stock NCCL graph search pairs wrong GIDs
   (`ibv_modify_qp ... local ::ffff:10.10.0.1 → remote ::ffff:10.10.3.1`,
   errno 110). The alexellis switchless knobs
   (`NCCL_SKIP_TREE_CONNECT=1 NCCL_SWITCHLESS_RING_ONLY=1 NCCL_MIN/MAX_NCHANNELS=4
   LD_PRELOAD=switchless libnccl`) were **proven on this exact cabling** by the
   nvidia/vLLM lane — removing them to "simplify" (drowzeys-style minimal env)
   was wrong: drowzeys' fabric is fully cross-connected, ours is a ring.
   *Tip: match the env to the topology, not to the cleanest recipe.*
2. **`ibv_reg_mr: Cannot allocate memory`** on 3ff5/dc90 post-reboot = HCA/RoCE
   not fully settled when ranks launch seconds after boot. GID table not yet
   populated → memory registration fails. *Fix: wait for all 4 IB devices to
   report PORT_ACTIVE before launching (fabric-check), don't boot immediately
   after a reboot.*
3. **Host memlock was 16 GB on dc90/3ff5 but unlimited on e116/2ca4** — an
   inconsistency across the quad (fixed via /etc/security/limits.conf 2026-10-01;
   needs re-login to apply). Containers set `--ulimit memlock=-1 --cap-add
   IPC_LOCK` regardless, but keep hosts consistent so `ibv_reg_mr` inside never
   fights a host cap.
4. **`--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1` is
   mandatory** on every rank container. Without it: `NET/IB` init fails
   ("connect to a peer closed"), NCCL error 2 everywhere. Proven with a
   standalone NCCL test harness before touching the engine.
5. **Per-rank HCA lists are correct as originally written** (e116:
   `rocep1s0f0,rocep1s0f1`; dc90: `roceP2p1s0f1,rocep1s0f0`; 3ff5:
   `rocep1s0f0,rocep1s0f1`; 2ca4: `rocep1s0f1,rocep1s0f0`) — they mirror each
   node's two ring edges. Do NOT collapse to one HCA per rank.
6. **`Spectrum-X (SPCX) plugin is not supported on device: skipping` WARN is
   benign** on CX7. drowzeys' logs show the same lines with a successful boot.
7. **NCCL bootstrap rides the mgmt LAN** (`NCCL_SOCKET_IFNAME`: e116 `enP7s7`,
   others `wlP9s9`); RDMA data rides `NCCL_IB_HCA`. This split (tp3-proven)
   works — keep bootstrap off the ring ports.

## B. GB10 unified memory

8. **Exit 137 (OOMKilled) is the ONLY OOM signal on GB10** — no CUDA OOM message
   (unified memory; `nvidia-smi` shows N/A). Always `docker inspect .State.OOMKilled`.
9. **Boot at `CONTEXT=36864`, not 1048576.** drowzeys' evidence: rank 0 holds
   `host anon 1.5 GiB, device free 20.7 GiB` after load at 36k context; the
   graph capture (buckets up to 64K) needs tens of GB more. Our 1M-context boot
   OOMKilled rank 0 twice (pre- and post-reboot, caches dropped). 1M is a later
   milestone; drowzeys reaches it with a different family (DCP-4 KV interleave)
   that glm5_next doesn't have.
10. **`vm.compaction_proactiveness=0` on every node** (drowzeys gotcha #1):
    proactive compaction stalls ranks ~130 ms under the GPU. Applied
    2026-10-01; **does not survive reboot** — re-apply after any reboot
    (update-reboot playbook should set it).
11. **Page cache counts against CUDA on GB10** — `drop_caches` before every
    boot (`just mem-free-quad`).

## C. Container/launch hygiene

12. **`docker rm -f` before every rank run** (rm → start, never restart). Stale
    containers caused name-conflict crashes ("container name already in use")
    and — worse — made us read STALE logs from old containers (r0 OOM verdict at
    16:01 was from a dead container). The drowzeys one-shot does this; our
    serve-tp4.sh now does too.
13. **Launch order: workers 3→2→1, rank 0 last**; workers waiting on rank 0's
    TCPStore is EXPECTED (they retry; store timeout 900 s in our fork). Late
    worker join is safe — don't panic-kill on "waiting for rank 0" lines.
14. **`pkill -f serve-tp4` can match its own ssh wrapper** — the script's own
    arg-builder was rewritten to avoid self-kill; use `scripts/serve-tp4.sh stop`.
15. **heredoc-in-ssh silently writes stale content** — always write_file + scp.
16. **Ship images via `docker save -o tar` → rsync → `docker load`**; the
    piped `docker save | ssh docker load` competes for one NIC, hangs in
    S/Sl, and orphans saves that survive SIGKILL. ~40-50 MB/s per stream;
    do NOT run ships in parallel with boots.
17. **`/tmp` does not survive reboots** — persistent state lives under
    `/home/spark/` (config, scripts, logs). The reboot mid-session erased
    launch scripts in /tmp on all nodes.

## D. Fork (0900-glm-tp4-core)

18. TensorFold 0.3.4 `glm5_next` hardcodes world=2 in ≥7 places; the fork
    threads `world` through (RankReader, split_bytes, _gather_ints). config.json
    `num_heads: 64` / `16` linear heads drive the shard math.
19. Boot reached: NCCL ring OK → 164 G weights → KV pool → 150 graphs captured
    → calibration → died at a final all_gather (mis-sized recv). v5 diagnostic
    image prints stack+sizes at the failing gather; every gather site audited.
    Remaining fix is in the fork, identified via that diagnostic — next boot
    with context 36864 will re-test (graph-capture memory pressure earlier may
    have masked the real bug).
20. Stay on forked 0.3.4 (not upstream 0.6.0): 168 commits of drift would break
    the 62-patch stack; recipe origin/main itself still vendors 0.3.4.

## E. Ops process

21. **Verify the container you're reading is the container that just ran**
    (compare `docker ps` created-time vs log timestamps) — stale-log reads cost
    us two false verdicts.
22. **Whole-system view first**: `just tf4-status` (per-node image/containers/
    checkpoints) before any single-node debugging.
23. **Only `just` commands for launch + monitoring** (user directive) — the
    justfile recipes (`tf4-*`) are the single entrypoint; scripts beneath them
    are implementation detail.
24. drowzeys repo (recipes/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark) is a
    different family (glm_moe_dsa 753B) — settings transfer, code does not.
    Their ANALYSIS-VS-OUR-FORK.md documents the cross-reference.
