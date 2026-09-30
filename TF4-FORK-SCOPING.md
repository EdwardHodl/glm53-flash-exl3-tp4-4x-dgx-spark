# TensorFold TP=4 fork — scoping notes (2026-10-01)

Follow-up to `TF4-SCOPING.md`. Written after pulling the latest
`recipes/glm53flash/*` clones and auditing the upstream TensorFold repo at
`recipes/TensorFold` (ashhart/TensorFold @ `9cd52ab`, **v0.5.0**, 2026-09-29).

---

## 1. What the latest pull shows

### Upstream TensorFold moved fast: 0.3.4 → 0.5.0 in two days

The recipe's submodule pins **v0.3.4** (`2f8e514`). Upstream main is now
**v0.5.0** (142 commits, releases 0.3.5 → 0.3.7, 0.4.0, 0.5.0). Highlights
relevant to a TP=4 fork:

- **0.5.0: Responses API on both servers** (`/v1/responses` as a chat completion
  with cache reuse — the thing flint lost vs the DSV41 vLLM lane, and one of the
  GLM53TF-TRIAL-PLAN's open questions, answered upstream).
- **0.5.0: long-context kernels** — 27B decode at 128k 18.4 → 38.9 tok/s; the
  same architecture likely transfers to glm5_next's sparse path.
- **0.5.0: `--parallel N`** on qwen3_5_moe and the 27B — *multiple requests on
  CUDA with exact solo-equal replies*. This is upstream's own answer to what
  jayleaton's recipe patches in by hand (batch v2 0120, sessions 0180). Upstream
  is absorbing the recipe's feature set into the engine proper.
- **The glm5_next CUDA engine in 0.5.0 is still strictly two-rank** — same
  anchors (see §2) — and it has NO batch/sessions/roce/vision (those exist only
  as recipe patches on the 0.3.4 base). **The fork base stays 0.3.4 + the
  recipe's 71 patches; upstream main is a reference, not a rebase target.**

### Recipe-folder drift (glm53flash pulls, 2026-10-01)

| repo | was | now | notable |
|---|---|---|---|
| `GLM-5.3-Flash-EXL3-2x-DGX-Sparks` (MiaAI) | 0 behind at our last check (2026-09-26) | **3 behind** (`534528b`) | new `feat/dense-h3-ablit` TP2 preset (ABLIT=1, BF16-o_proj variant) — TP2-only, not TP4 material |
| `GLM-5.3-Flash-4x-DGX-Spark-TP4` (knapcio) | 0 behind (2026-09-26) | **11 behind** (`770d115`) | **`TRANSPORT=switchless` mode added (PR #1, @othexmr)** — a TP4-only ring launcher for four directly-cabled Sparks; documented as *community-contributed, not tested on their (switched) fleet* |
| `glm-5.3-flash-4x-gx10` / `glm53-flash-exl3-2x` | 0 / 2 behind | 27 / 2 behind | SP-MoE BF16 gather fix; token-exact chat-segment cache |
| others (alexellis 4x switchless, RiNGSiDE, NVFP4 lanes) | 0-1 behind | unchanged-ish | — |

No TensorFold mentions anywhere in `glm53flash/` outside our own
`glm53-tensorfold-spark` clone and one stale reference in
`glm53-flash-exl3-2x-dgx-spark/docs/19-router-once.md`. **Nobody else has
run TensorFold on more than 2 nodes.** A TP=4 fork would be first.

### The transport references exist and are mature

A TF4 fork does NOT need to invent the transport. Reuse, in order of fit:

1. **`fire/`'s own quad ring** — already built, netplanned (`netplan/quad/`,
   persistent — better than alexellis's runtime-only fabric), MTU 9000, 4
   edges, `verify-fabric.sh` green, `nccl-switchless-v0.0.1` (NCCL 2.30.7,
   SHA-pinned) **already installed on all four nodes** for the alexellis lane.
2. **knapcio's new `TRANSPORT=switchless`** (11 commits): the newest
   four-PF/dual-PCI-domain pattern; its *caveats doc* (`docs/switchless.md`)
   lists exactly the failure modes a TF4 fork inherits — including
   "**RoCEnante is disabled in this mode because its all-peer fabric
   assumptions do not hold for the ring**" — which is the crux for
   TensorFold, whose RoCE all-gather (patches/0230) is also all-peer by
   design (HCAs paired by subnet between exactly two ranks).
3. **alexellis `glm-5.3-flash-4x-dgx-spark-switchless`** — the original
   `NCCL_SKIP_TREE_CONNECT=1 / NCCL_SWITCHLESS_RING_ONLY=1` launcher env
   (`scripts/rank-launcher.sh:149`), fabric.md's two-rail addressing
   template, and the switchless-nccl install script our nodes already ran.

### TensorFold's NCCL wrapper is transport-agnostic (good news)

`src/tensorfold/cuda/comm.py` loads **any** libnccl via ctypes:
candidates are `$TF_NCCL_LIB` first, then system paths. It sets no NCCL env
vars itself (the recipe's patch 0040 adds `nccl_env()` for `NCCL_PROTO`).
So pointing TensorFold at the switchless SO is one env var:

```
TF_NCCL_LIB=/home/spark/nccl-switchless-v0.0.1/lib/libnccl.so.2
```

plus the alexellis rank-launcher env (`NCCL_SWITCHLESS_RING_ONLY=1`,
`NCCL_SKIP_TREE_CONNECT=1`, `NCCL_ALGO=Ring`, GID/subnet knobs). No fork
needed for the transport layer. **The fork is needed only for the world-math.**

---

## 2. The fork: every 2-rank anchor in the glm5_next CUDA path

Full inventory (25 anchors across 11 files, verified against 0.5.0 which has
the same structure; the recipe's 0.3.4 base matches):

| # | file:line | anchor | generalization |
|---|---|---|---|
| 1 | `cli.py:111-116` | `--tp choices=(1, 2)`, `--rank choices=(0, 1)` | widen choices; `--master` already a list-friendly string |
| 2 | `engine.py:107` | `NCCL(rank, 2, master, port)` | `NCCL(rank, world, master, port)` — the class itself is **already world-generic** (TCPStore with `world`, `ncclCommInitRank(world, …)`, `all_gather` sized by `self.world`) |
| 3 | `engine.py:117-118` | `split_weights(rule), rank=rank, world=2` + `draft_geometry(text, 2, …)` | pass the CLI world through (functions are generic: `mla_geometry(t, world, …)`, `split_weights(rule, world=2)` has a default) |
| 4 | `engine.py:130-136` | `both[0]/both[1]` settings compare | loop over `both[i]`; error message lists all ranks |
| 5 | `weights.py:248` | `world = 2` in `load()` | parameter; thread from engine |
| 6 | `weights.py:255-256, 412` | `HL = cfg.heads // world`, `LL = cfg.lin_heads // world`, `vl = cfg.vocab // world` | already world-generic once `world` is a param |
| 7 | `split.py:84-118` | `half = rows // 2` (row/col/dim1 rules) | `half = rows // world`; divisibility checks `rows % 2` → `% world` |
| 8 | `split.py:145` | `rank_files(dir, 1 - rank)` ("holds the other rank's share") | scan for any `*.rank{R}.safetensors` with R ≠ rank |
| 9 | `split.py:200-201, 279` | split writer `--rank choices=(0,1)`; `shape[0] // 2` | loop writer over `range(world)`; `// world` |
| 10 | `dflash2.py:174-195` | drafter sharding `heads // self.world`, `kv_heads // world`, `inter // world` | already world-generic (uses `w.world` — DFlash2 32 heads / 8 kv / inter must divide by 4: **they do**, 8/2/… — this was the TP=3 tax, gone at 4) |
| 11 | `dflash2.py:254, 295-301, 370-371` | `packed` gather loops `for i in range(1, self.world)` | **already world-generic** — no change |
| 12 | `glue.py:176-231` | Triton `_hc_post`, `_residual_add` take `WORLD: tl.constexpr`, loop `tl.static_range(1, WORLD)` | **already world-generic** — no change |
| 13 | `decode.py:41-45` | `got.view(w.world, …)`, `torch.cat` over `range(w.world)` | **already world-generic** — no change |
| 14 | `forward.py` buffers | `HL = c.heads // w.world` etc. | generic via `w.world` |
| 15 | `geometry.py:95-107` | `split_weights(rule, world=2)` default | default only |
| 16 | `kernels/affine.py:23` | **`if world != 2 or rank not in (0, 1)` — "must split into two complete group-aligned halves"** | the one real kernel-level wall; the math (`k // (world * group)`) generalizes, but the guard is hard and it backs the packed-row `input_slice` used by the EXL3 loader. Change the guard to `k % (world * group)` + rank range check. **Highest-risk edit** — bit-exactness of packed 4-bit groups across 4-way splits needs verification per tensor class. |
| 17 | `server.py:339-340` | background-yield "not on two ranks" logic (`tp == 1` check) | re-derive for N ranks; or disable background-yield under TP>2 initially |
| 18 | `cli.py:344, 367, 371` | `rank of 2` strings, rank-1 "follows rank 0" | cosmetic + start order (all N ranks wait on rank 0's TCPStore — already handled by TCPStore semantics) |

**The real work is not the world math (mostly mechanical, ~half the anchors are
already world-generic) — it is three specific walls:**

- **W1 — affine.py's two-halves guard** (16): the packed-4-bit row split assumes
  exactly two group-aligned halves. Generalizing is math-wise trivial but needs
  bit-exactness re-validation on the EXL3 checkpoint across every tensor class
  the recipe touches (trellis/suh/svh/mcg, kv_b, gate/up/down).
- **W2 — the recipe's 10 TP-coupled patches**: batch v2 (0120), sessions (0180),
  session-nvme (0250), kv-pool (0290), prefix-share (0310), prefill-row-split
  (0320), roce (0230/0350), deep-verify (0380), vision (0500) each encode
  2-rank assumptions (peer handoffs, rank-0-only components, "the other
  rank's rows"). Each needs a per-patch pass; some (vision on rank 0,
  sessions tiering) generalize cleanly, some (roce, prefill row split)
  need the same world-math treatment as the engine.
- **W3 — the RoCE all-gather (0230/0350)**: designed for exactly two nodes
  (HCAs paired by IPv4 subnet, one pinned region per rank, direct QP pairs).
  On a 4-ring: either (a) run the RoCE fast path only for pair-local
  exchanges and NCCL for the rest (hybrid — unmeasured), (b) drop to NCCL
  for all model exchanges at TP=4 (loses the +10.8% decode but ships), or
  (c) write a ring-RoCE (all-peer → ring-neighbour) — real work. **Start
  with (b): `GLM53_TF_COMM_BACKEND=nccl` on fire-tp4.**

### Effort estimate (revised from TF4-SCOPING option B)

| work item | estimate | confidence |
|---|---|---|
| core world-math (engine/weights/split/cli, anchors 1-15) | 1-2 days incl. unit tests (upstream has `tests/`, and the CPU-side split has an emulator) | medium-high |
| affine.py 4-way group split + bit-exactness harness (reuse recipe's `exact`/`batchexact` gates) | 1-2 days | medium |
| per-patch pass over the 10 TP-coupled recipe patches | 2-4 days (vision/sessions mechanical; batch/verify moderate; roce → nccl-only is a deletion) | medium |
| ring transport bring-up on fire (TF_NCCL_LIB switchless + launcher env) | 0.5 day — the pieces exist and are proven | high |
| prepare-folders for 4 ranks (77 GB × 4 = 308 GB staging + disk on dc90/2ca4) | 0.5 day + pruning decisions | high |
| boot, canary, idle-guarded bench vs the vLLM alexellis lane receipts | 0.5 day | high |
| **total** | **~5-8 focused days**, first boot inside 2-3 days | — |

### What would make it cheaper

- **Ask upstream**: 0.3.4→0.5.0 in two days shows shhart moves quickly; the
  `adding-a-cuda-family.md` contract says "for two ranks, compare with that
  same two-rank engine's serial reference" — TP-N is on their radar at least
  as a pattern (qwen3_5's `distributed.py` already *tries* world-size-generic
  code with a two-rank guard). An issue/PR with our 4-node hardware offer is
  the highest-leverage 30 minutes available.
- **Do TP=3 instead?** No — nothing divides by 3 (heads 64, kv 8, inter; drafter
  32/8) — the TP=3 pad-tax the vLLM lane paid (66-head overrides, EP, vocab pad)
  would need re-implementing in a fork. 4 is the natural next world size. (TF4-
  SCOPING.md §1 said the same for the vLLM path.)

## 3. Decision framing (updates TF4-SCOPING §6)

The fork (option B) and the TP=2-on-an-edge lane (option A/C) are not mutually
exclusive — the TP=2 lane is the fork's **first milestone** anyway:

1. **Ship option A first** (flint-identical TP=2 on an edge, ~1 session of work
   including the playbook run) — proves staging, config, fabric pair on fire's
   hardware, and gives the fork a live 2-rank reference to diff against.
2. **Fork in a `recipes/TensorFold` branch** (or a `glm53-tensorfold-spark`
   patch-series extension — the patches ARE the fork surface; prefer extending
   the recipe's patch series over diverging the submodule, so upstream pulls
   stay possible). Sequence: world-math patch (0010-style, one patch) → affine
   4-way → per-patch TP-coupling pass → nccl-only roce bypass.
3. **The 4-ring transport is already done** — fire's netplan + switchless NCCL
   + launcher env all exist; the fork reuses them verbatim via `TF_NCCL_LIB`.

The disk question (dc90 18 G / 2ca4 48 G) becomes a **4-node** question for the
fork (308 GB across dc90+2ca4) instead of a 2-node one — pruning there is a
prerequisite either way.

## 4. Sources for this note

- `recipes/TensorFold` @ 9cd52ab (v0.5.0): cli.py:111-116, cuda/comm.py:36-72
  (world-generic NCCL class), cuda/geometry.py:95-107 (split_weights world
  param), cuda/kernels/affine.py:23 (two-halves guard), families/glm5_next/
  cuda/{engine.py:107-136, weights.py:248-256+412, split.py:84-118+145+279,
  dflash2.py:174-195+254+295-301+370-371, glue.py:176-231 (WORLD constexpr),
  decode.py:41-45, forward.py:27-31}, server.py:339-340; docs/recipes/
  glm-5.3-flash.md (two-rank contract), docs/recipes/adding-a-cuda-family.md:38.
- `recipes/glm53-tensorfold-spark` (0.3.4 base, submodule 2f8e514): patches
  0040/0120/0180/0230/0250/0290/0310/0320/0350/0380/0500 (the TP-coupled set).
- `recipes/glm53flash/`: knapcio TP4 11 new commits (TRANSPORT=switchless,
  docs/switchless.md), alexellis 4x switchless scripts (rank-launcher.sh:149,
  install-switchless-nccl.sh, docs/fabric.md), MiaAI EXL3 3 new commits
  (dense-h3 TP2 preset — not TP4 material), no other TensorFold references.
- `fire/`: netplan/quad (persistent ring, all four edges), hosts' nccl-switchless
  install verified live (spark-e116:~/nccl-switchless-v0.0.1 with
  libnccl.so.2.30.7), recipe-env/env.tp4 (site values).
