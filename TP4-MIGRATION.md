# fire → quad (TP=4 switchless) + duo (TP=2) migration plan

Status: plan/checklist — nothing executed. Written 2026-09-25 from
`recipes/glm53flash/` repos at latest main (MiaAI-Lab +30, jnardiello +11,
tonyd2wild +1 pulled 2026-09-25).

---

## 1. Repo scan — what TP=3 and TP=4 recipes actually differ on

The three upstream sources that matter:

| repo | role | TP=4 evidence |
|---|---|---|
| `GLM-5.3-Flash-EXL3-2x-DGX-Sparks` (MiaAI-Lab) | EXL3-native launchers `start-tp3.sh` / `start-tp4.sh` — **this is the path to follow, same weights we run** | `.env.tp4.example` + README "Experimental: 4× Spark"; measured on a 4-Spark kit 2026-09-02/03 (receipts in punkjazz-labs repo) |
| `glm-5.3-flash-4x-dgx-spark-switchless` (alexellis) | the switchless **transport** recipe: patched NCCL + ring fabric + launcher discipline | NVFP4 weights, but transport is model-agnostic (docs say the same ring served EXL3 GLM-5.2 and DeepSeek-V4-Flash at TP4) |
| `GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless` (jnardiello) | full IaC of the same ring: per-node netplan renderer, GID handling, ops runbook | FP8 zai-org weights; useful for netplan/host-side automation |

### Model-shape differences (why TP=3 is special and TP=4 is not)

Nothing in GLM-5.3-Flash divides by 3, so fire's TP=3 carries four shape fixes:
head pad 64→66 (`TP3_HEAD_OVERRIDE=66`), vocab pad to 154944, shared-expert
2048→2112, expert-parallel 96 experts/rank, `DFLASH_DRAFT_TP=1`, KDA
f/g_b_proj kept BF16, `MM_ENCODER_TP_MODE=data`.

At TP=4 every one of those disappears:

| quantity | stock | at TP=4 |
|---|---|---|
| attention / KV heads | 64 | 64/4 = **16/rank, clean** — no `--hf-overrides`, no pad |
| routed experts | 288 | 288/4 = **72/rank, clean** — no `--enable-expert-parallel` needed |
| vocab / shared-expert / A_log | 154880 / 2048 / 64 | loaded stock (`patch_tp3_glm.py` not used) |
| DFlash2 drafter | 32 heads / 8 kv — cannot shard by 3 | 32/4, 8/4 divide → **`DFLASH_DRAFT_TP=4`** (draft sharded on all ranks — this was the TP=3 divisibility tax) |

Launcher patch surface (`start-tp4.sh` vs `start-tp3.sh`, counted this session):

- **gone at TP=4**: `patch_tp3_glm`, `patch_exl3_ep_shard`, `patch_exl3_expert_map`, `patch_adaptive_k` wiring, `patch_dense_fp8` (KDA BF16 workaround), `patch_flashkda_tp3`.
- **kept**: video placeholders, suppress-stops, decode-floor, mamba chunking/state-free, drafter-group, hybrid-prefix-hit, per-group retention, xgrammar, kpool-tail, spinwait, loadclone.
- **new, TP=4-only**: `patch_sparse_mla_slice.py` (`VLLM_SM120_SPARSE_MLA_SLICE_TOKENS=64`) — opt-in mitigation for the TP=4 all-rank stall (issues #128/#159; bounded final sparse-MLA call ≤64 query rows). Default 0; qualified only with `DFLASH_TOKENS=3`, mixed-prefill `off`, `--enforce-eager` on that kit.

### Serve-knob differences (upstream TP=4 measured values, `.env.tp4.example`)

| knob | fire TP=3 (live) | TP=4 upstream measured | why |
|---|---|---|---|
| `GPU_MEM_UTIL` | 0.76 | **0.75** | 0.85 left <1–2 GiB host/rank → two engine deaths (NVRM NV_ERR_NO_MEMORY) |
| `MAX_NUM_SEQS` | 6 | **8** | 135 vs 84 tok/s aggregate, 1.0 s vs 27 s worst TTFT |
| `DFLASH_TOKENS` | 7 | **3** | 53 % acceptance vs 30 % at 7; 31–33 vs 26–28 tok/s prose |
| `MAX_NUM_BATCHED_TOKENS` | 7168 | **2048** | cold prefill fabric-bound anyway (~1.15k tok/s @282k); 8192 costs +20 s TTFT under prefill |
| `GLM53_MIXED_PREFILL_CHUNK` | fair 0.50/512 | example keeps fair 0.30/256; TP=4 kit ran **off** | TP=4 receipt: off answers newcomer in 0.5 s, incumbent keeps ~90 % |
| `EXL3_TEMP_ROWS_FUSED` | 48 | 128 default | topology-sized |
| KV pin | `--kv-cache-memory-bytes 20000000000` | **start-tp4.sh drops the shared pin (#204)** — set a TP=4-scoped pin or let it self-size | TP=4 has ~½ the weights/rank; much more KV headroom |
| `MAX_MODEL_LEN` | 320000 | unset → follows `.env` (850000) | fire will pin 320000 |
| drafter cache | `GLM53_DRAFT_KV_COMPACT=1`, `SPINWAIT_MS=16` | compact defaults ON under dflash | carry the same two pins |

### Transport differences (the part worth doing deliberately)

| aspect | fire TP=3 today | quad TP=4 target |
|---|---|---|
| cabling | triangle: 3 point-to-point QSFP links | **closed ring**: 4 edges, each node wired to 2 neighbours — needs 1 new cable and a rewire of one (see §3) |
| addressing | 3 subnets 10.10.0/1/2.x | 4 point-to-point /24s, e.g. 10.10.1..4.<rank+1> |
| MTU | 4092 | **9000 mandatory** — 1500/4092 works silently ~2.7× slower on all-reduce |
| NCCL | stock 2.29.7 (image torch wheel), directed-ring env from `.env.tp3` | **switchless-nccl v0.0.1** (NCCL 2.30.7 + skip-tree-connect, `LD_PRELOAD`, lib SHA `78cb8387…`) — tree/PAT connect wedges on a 4-rank ring; plus `NCCL_ALGO=Ring`, `NCCL_IB_MERGE_NICS=0`, `NCCL_IB_SUBNET_AWARE_ROUTING=1`, channels 4/4 |
| fabric persistence | netplan (`/etc/netplan/50-qfsp-static.yaml`) | upstream uses **runtime-only** `fabric-setup.sh` after every boot/docker churn; jnardiello renders netplan instead — **prefer netplan (fire's existing pattern), keep the DOCKER-USER/ip_forward re-apply** |
| GID | index 3 verified on every port | same; alexellis adds the empty-GID-after-mode-switch trap → reboot node if `gids/3` is all zeroes |
| control plane | mgmt LAN (e116 wired; 2ca4/3ff5 **wifi**) | same pattern; master-addr = head mgmt IP; port 29521 |
| diagonal relay | n/a (triangle fully connected) | **A↔C / B↔D relay through a neighbour**: needs `net.ipv4.ip_forward=1` + DOCKER-USER ACCEPT for fabric ifs — the stock fire kit does not do this |

### Launch/lifecycle differences

- Order: workers **3 → 2 → 1**, head last (fire tp3: workers 1, 2).
- Always cycle **all four** containers together; a single-rank recreate breaks the torch group.
- Collective deadline ~90 s ("Connected all rings" normally <1 min); API readiness ≤15 min with visible progress.
- Start only on four **empty** GPUs; launcher refuses on competing compute apps (mind supervised stacks like the qwen lane leftovers).
- Gate before "up": 30K needle + tool-call + warm decode (`gate.sh` upstream). Jumbo ping ≠ proof.
- Performance expectation (NVFP4 receipts, transport-level): code decode 75.2 vs 42.6 tok/s (TP2), structured 2.0×, cold 64K prefill only 1.19× (fabric-bound), warm replay 3.56×. EXL3 TP=4 measured: decode 1.45× vs TP=2, cold prefill +29 % only.

---

## 2. Node assignment (TBD — confirm hardware)

Current triangle: e116 (head, wired LAN), 2ca4 (wifi), 3ff5 (wifi). A quad needs
a **4th GB10**. Constraints to honor when picking it:

1. Prefer a **wired-LAN** node for the head or move the head role — wifi control plane to two ranks is today's weakest link and TP=4 adds a 4th rendezvous member.
2. The 4th node needs: ~164 GiB weights + ~2.3 GB DFlash2 staged, `glm53-exl3:e3-v152` image (or current tag), docker, GB10/driver parity, `spark_agent_key` authorized.
3. Duo (TP=2) should reuse the **best-connected pair** (e116 + the fastest fabric peer).

Open question for you: which sparks form the quad, which two form the duo, and
which nodes stay free? The checklist below assumes e116 stays head of the quad.

---

## 3. Cabling plan — triangle → 4-edge ring

Ring order proposal (keeps two existing cables in place, adds two):

```
        e116 ────── 3ff5          (existing link1  e116.f0  10.10.1.1 ↔ 3ff5.f1  10.10.1.3)
        │             │
        │             │
        NEW ────── 2ca4           (existing link3  2ca4.f1 10.10.2.2 ↔ 3ff5.f0 10.10.2.3)
```

- Edge A (e116↔3ff5): keep link1 as 10.10.1.0/24.
- Edge B (3ff5↔2ca4): keep link3 as 10.10.2.0/24.
- Edge C (2ca4↔NEW): **new cable**, subnet 10.10.3.0/24.
- Edge D (NEW↔e116): **new cable**, subnet 10.10.4.0/24.

Verify every cage adjacency by MAC probe like the original fire bring-up — the
cages cross-connect unpredictably (fire README: "the cages are cross-connected,
not f1↔f1"). Do NOT assume which physical port lands on which peer until probed;
write netplan after the probe, not before.

For the duo (TP=2): a single cable between the pair, 10.10.x.0/24, MTU 9000 —
the sparks kit's proven layout.

---

## 4. Netplan / networking options (the "several options" you recalled)

Option 1 — **netplan persistent (recommended, matches fire today)**: per-node
`/etc/netplan/60-qfsp-ring.yaml`, renderer NetworkManager (or systemd-networkd
to match the existing 50-qfsp file — keep one renderer), both addressed ports
MTU 9000, second-PCIe views unaddressed but MTU 9000. Survives reboot; you still
re-apply DOCKER-USER + ip_forward after docker churn (tiny systemd unit or
justfile task).

Option 2 — **runtime-only fabric-setup.sh (upstream default)**: addresses, MTU,
routes, forwarding, iptables applied by SSH script on every boot. Zero
persistence; matches upstream docs and their failure-mode knowledge exactly.

Option 3 — **jnardiello render pipeline**: `cluster.env` + `render-netplan.sh`
generates per-node 40-cx7.yaml + iptables env files, deployed by script with
hostname checks. Most robust, most moving parts; adopt if you want the IaC.

Either way you need the L3 bits for the **diagonal relay**:
`net.ipv4.ip_forward=1` and DOCKER-USER ACCEPT in/out on both fabric ifs.
jnardiello's NCCL_IB_GID_INDEX=-1 (auto-selection with AF_INET + RoCEv2=2
constraints) is worth adopting later; explicit 3 is fire's verified value.

---

## 5. NCCL networking options

1. **Switchless NCCL v0.0.1 (upstream-canonical for TP=4 ring).** Install
   `nccl-2.30.7-switchless-hardened-sm121-linux-arm64` on each node
   (`install-switchless-nccl.sh`, pinned SHAs). Inject via the launcher's
   USE_HOST_NCCL=1 path (`-v $HOME/nccl-switchless-v0.0.1:/nccl:ro` +
   `LD_PRELOAD=/nccl/libnccl.so.2`) or bake into an image layer. Set
   `NCCL_SWITCHLESS_RING_ONLY=1`, `NCCL_SKIP_TREE_CONNECT=1`, `NCCL_ALGO=Ring`,
   `NCCL_IB_MERGE_NICS=0`, `NCCL_IB_SUBNET_AWARE_ROUTING=1`, channels 4/4,
   `NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0`, `NCCL_CROSS_NIC=1`.
   *Risk*: image ships NCCL 2.29.7 (torch wheel). Preloading a 2.30.7 SO has one
   documented hazard in this stack: the DeepEP duplicate-NCCL assert (start-tp4.sh:236).
   EXL3 path doesn't use DeepEP; validate at boot (`NCCL_DEBUG=INFO` version line).
2. **Stock 2.29.7 ring first (fire-style).** Fire TP=3 works on stock NCCL with
   the directed-ring env (MERGE_NICS=0, SUBNET_AWARE_ROUTING, CROSS_NIC=1,
   both-HCA list, GID 3). A TP=4 ring *may* form without skip-tree-connect —
   upstream says tree/PAT "wedges" on a ring, so expect a hang at tree connect
   and treat option 1 as the fix, not a mystery. Cheapest first bring-up: all
   fabric/launch checks green under stock NCCL proves everything except the
   tree-connect step.
3. **Build-your-own hardened NCCL** (alexellis switchless-nccl repo): only if
   you need to modify the patch. Not needed for this migration.

Recommendation: bring up with **option 2** to separate fabric risk from NCCL
risk, then flip to **option 1** in the same maintenance window and keep it
(option 1 is the qualified TP=4 configuration upstream).

---

## 6. Weights & image (unchanged, as requested)

- Weights: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` @ `25a44fdbf16862a46b7cc9921142c6c81350af2f`, already staged on e116/2ca4/3ff5 (164 GiB each). `start-tp4.sh` adopts the same cache via `MODEL`/`MODEL_REVISION` defaults — zero re-download. Stage on the 4th node (rsync over its new QSFP edge once cabled, or LAN).
- DFlash2 drafter `incoai/GLM-5.3-Flash-DFlash2` @ `7d74cdd881ed7e32c31175984a67823127b66cfe` — same handling.
- Image: `glm53-exl3:e3-v152` (current fire image, on e116+3ff5; **2ca4 needs v152**). `start-tp4.sh` defaults to GHCR `exl3-instanttensor` — override `IMAGE=glm53-exl3:e3-v152` in `.env.tp4`. If adopting switchless NCCL, either rebuild the e3 layer bumping the wheel to 2.30.7 or rely on LD_PRELOAD.
- Serving identity stays `GLM-5.3-Flash-EXL3` on `:8888` — clients/crons unaffected.

---

## 7. Plan

Phase 0 — decide (you): quad members + duo pair + which node is head; confirm a
4th GB10 exists with wired LAN.
Phase 1 — stage (1–2 h): clone kit to new quad dir (`fire-quad/`), copy
`start-tp4.sh` + overlays from repo main (newer than e116's deployed copy),
write `.env`/`.env.tp4` with fire's IPs/keys, `IMAGE=glm53-exl3:e3-v152`,
`MAX_MODEL_LEN=320000`, TP=4 knob set from §1, TP=4-scoped KV pin.
Phase 2 — cable + netplan (1–2 h): two new DACs, MAC-probe all four edges,
write netplans (MTU 9000), fabric-check equivalent (8 directed pings).
Phase 3 — bring-up stock NCCL (0.5–1 h + boot ~17 min): sync weights to rank 3,
start 3→2→1→0, expect possible tree-connect hang → switch to switchless NCCL
(Phase 4) if so.
Phase 4 — switchless NCCL (1 h): install pinned release on 4 nodes, enable
USE_HOST_NCCL=1 + selector envs, relaunch all four, confirm NCCL 2.30.7 +
"Connected all rings" <90 s.
Phase 5 — gate (30 min): 30K needle, tool-call, warm decode; quote numbers.
Phase 6 — duo (TP=2, 1 h): separate kit dir (`fire-duo/`), `start.sh`-style
2-rank env from fire's old TP=2 lane (already vendored), single cable, MTU 9000.
Phase 7 — cutover: stop tp3 stack (containers glm53-exl3-tp3-*), port 8888
release, start chosen lane(s); duo can serve on a different port simultaneously.
Phase 8 — soak + tune: mirror fire's post-tune round on TP=4 (adaptive-k ema,
fair-prefill share, spinwait 16) one knob at a time; watch the #128 stall class
→ set `VLLM_SM120_SPARSE_MLA_SLICE_TOKENS=64` if it appears (accept its
--enforce-eager qualification caveat or re-soak).

Downtime: fire TP=3 keeps serving until Phase 7; total cutover gap ≈ stop +
start ≈ 20–25 min.

---

## 8. Checklist

**Phase 0 — decisions**
- [ ] Pick quad members + rank order (head = wired-LAN node preferred)
- [ ] Pick duo pair
- [ ] Confirm 4th node: GB10, docker, disk ≥ 200 GiB free, key authorized

**Phase 1 — stage**
- [ ] `fire-quad/` kit dir: `start-tp4.sh` from repo main (md5 `1051a7be…`), overlay/, tests/
- [ ] `.env`: HEAD/WORKER×3 IPs, users, keys, `MODEL_REVISION=25a44fdb…`, `IMAGE=glm53-exl3:e3-v152`
- [ ] `.env.tp4`: TP=4 knob set (util .75, seqs 8, k=3, MNBT 2048, DFLASH_DRAFT_TP=4, MAX_MODEL_LEN=320000, KV pin TP=4-sized, DRAFT_KV_COMPACT=1, SPINWAIT_MS=16)
- [ ] 4th node: weights + drafter staged, image pulled/built
- [ ] 2ca4: pull `glm53-exl3:e3-v152`

**Phase 2 — fabric**
- [ ] Two new DACs installed per §3 map
- [ ] MAC-probe all four edges (expect surprises — cages cross-connect)
- [ ] Netplan on 4 nodes: addressed ports MTU 9000, 10.10.1–4.x /24s; second PCIe views MTU 9000
- [ ] 8 directed jumbo pings green
- [ ] DOCKER-USER + ip_forward re-apply mechanism (unit or just task)
- [ ] GID check: `gids/3` non-zero, RoCE v2, on all 8 ports

**Phase 3 — bring-up (stock NCCL)**
- [ ] GPUs idle on all 4 (no competing appliance)
- [ ] Start ranks 3→2→1, head 0
- [ ] Collective <90 s; if tree-connect hang → Phase 4
- [ ] health 200 within 15 min, progress visible

**Phase 4 — switchless NCCL**
- [ ] `install-switchless-nccl.sh` on 4 nodes (SHA-verified)
- [ ] USE_HOST_NCCL=1 (or image bake), `NCCL_SWITCHLESS_RING_ONLY=1`
- [ ] Boot log shows NCCL 2.30.7, no DeepEP duplicate-NCCL assert
- [ ] "Connected all rings", gate green

**Phase 5 — gate**
- [ ] 30K needle retrieval coherent
- [ ] Tool-call parses (glm47)
- [ ] Warm decode measured (throwaway turn first)
- [ ] `/v1/models` id `GLM-5.3-Flash-EXL3`

**Phase 6 — duo (TP=2)**
- [ ] `fire-duo/` kit: 2-rank env (existing tp2 lane values + MTU 9000)
- [ ] Pair cabled, pings green, weights staged
- [ ] Bring up on spare port (e.g. 8889) or after quad decision

**Phase 7 — cutover**
- [ ] Announce; stop fire tp3 (`just stop-glm-exl3` path / containers tp3-*)
- [ ] Start chosen stack; smoke test + crons hit `spark-e116:8888`

**Phase 8 — soak/tune**
- [ ] 24 h soak, watch for #128-style all-rank stall
- [ ] One-knob experiments: GLM53_ADAPTIVE_K=ema, fair-prefill share, spinwait
- [ ] Record receipts in fire README (post-tune table style)

**Rollbacks**
- fabric: revert netplan to 50-qfsp triangle file, reapply
- NCCL: USE_HOST_NCCL=0 → stock wheel
- whole quad: `start-tp4.sh stop` (its stop knows ranks 2/3), restart tp3 stack
- KV/capture OOM: drop KV pin / util to last-green values; node reboot clears UMA fragmentation (fire saga, 3 failed boots)
