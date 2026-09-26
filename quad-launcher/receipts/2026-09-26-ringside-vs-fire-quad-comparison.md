# RiNGSiDE vs fire-quad — comparison and adoption recommendations

Date: 2026-09-26. Status: **analysis only; nothing executed, no node accessed.**
Subject: `othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE` @ `6606f76` (clone at
`~/Workspaces/devops/recipes/glm53flash/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE`, in sync with origin)
versus our NVFP4 TP4 lane in `~/Workspaces/devops/fire/`.

## 1. The two recipes as they stand

| | RiNGSiDE TP4 profile | fire-quad (ours) |
|---|---|---|
| Lab plan | release plan `89dfef84`, one measured window 2026-09-26 00:36-01:50 | deployed receipt `c6d7f86`, gate green 2026-09-26 00:04 |
| Engine | vLLM **v0.29.0 + GLM-5.3 backport**, build commit `98dff2a8`, tag `local/glm53-v029:ported-20260910`, base image sha `017fd0ba` | prebuilt `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`, digest `4def0ef6…`, **vllm 0.1.dev20051+g487ecf187** |
| Weights | RedHatAI/GLM-5.3-Flash-NVFP4 `18d55bfd` + nvidia `tokenizer.json` (NVIDIA checkpoint `423acf37` in the extra arm) | nvidia/GLM-5.3-Flash-NVFP4 `09b04e5e`, flat dirs |
| Drafter | incoai DFlash2 `7d74cdd8`, K=7 (`--speculative-config`, `kv_cache_dtype auto`) | incoai DFlash2 `7d74cdd8`, K=7 static |
| Delivery of changes | 54 single-file patches + 55 full files **mounted as an overlay** (`launch/render.py`), sha256-checked pre/post; `src/` carries 38 TP4 files; native libs built from `build/` | changes **baked in the image**; our site deltas live in `quad-launcher/rank-launcher.sh` + `cluster.env` (sha-pinned launcher/template/NCCL) |
| Transport | custom NCCL build: patch 0001 switchless ring + **patch 0002 dual-PF four-GID routing**, pinned to cable subnets 10.100.224-227/24 | alexellis switchless-nccl v0.0.1 (`78cb8387…`), 2 HCAs/rank pinned per node, MTU 9000, subnet-aware routing |
| Scheduler | `glm53_speedup` fork (scheduler + fair_prefill + policy + cadence + adaptive_chunk + graphs) with mixed-prefill cap 4608, split cadence, first-token repay, KDA checkpoints, replay boundary | stock v11 scheduler; our E27c/E29 port (jnardiello R10) **aborted** → `QUAD_E27C_SCHEDULER=off` |
| Kernel stack | b12x MoE (`--moe-backend b12x`, atomic prefill, deterministic decode epilogue), sparse-MLA plugin (`VLLM_PLUGINS=glm53_sparse_mla`), prefill RDMA ring, lean all-reduce (≤2 MiB), topk rowspread, tuned FP8 GEMMs, humming dense | `--moe-backend marlin`, stock attention/sampling, no native extensions of our own |
| Context / KV | `--max-model-len 262144`, `--kv-cache-memory 34359738368` (32 GiB/rank fp8_e4m3), documented **512K and 1M experiments** | `--max-model-len 262144`, `--kv-cache-memory 12884901888` (12 GiB/rank pin), pool 1,576,246 tokens |
| Admission | `--max-num-seqs 16`, `--max-num-batched-tokens 14336`, prefix-cache retention 2304, cudagraph capture 8-128 | `--max-num-seqs 6`, `--block-size 2304`, no MBT/reten/compilation-config |
| GPU mem | `--gpu-memory-utilization 0.87` | `0.85` |
| Determinism | **explicitly non-deterministic** (atomic MoE adds in prefill; ≥10-row decode launches) | undocumented; marlin MoE (deterministic decode by construction) + dflash accept counted per request |
| Published numbers | own bench table (RigMark fork, sparkDash, llama-benchy, tool-eval-bench) | gate only: needle + tool-call + warm decode 70-76 t/s |

## 2. Measured head-to-head on the same four-Spark ring

RiNGSiDE's own bench table measured a **tonyd2wild TP4 recipe on this ring's NCCL transport** — its engine is the
exact version ours runs (`vllm 0.1.dev20051+g487ecf187`), so it is the closest published proxy for fire-quad. Both
rows are non-deterministic NVFP4 MoE with K=7 dflash; the checkpoint differs (nvidia `423acf37` for tonyd2wild,
RedHatAI `18d55bfd` for RiNGSiDE), and RiNGSiDE ran medians of 2 while tonyd2wild is a single run.

| cell | RiNGSiDE TP4 | tonyd2wild TP4 (our engine) | delta |
|---|---:|---:|---:|
| decode code (tok/s) | 105.3 | 82.5 | **+27.6 %** |
| decode prose (tok/s) | 61.7 | 35.8 | **+72.3 %** |
| decode structured (tok/s) | 145.4 | 127.4 | +14.1 % |
| cold TTFT 8K (s) | 1.71 | 4.52 | 2.64x faster |
| cold TTFT 32K (s) | 6.46 | 14.16 | 2.19x faster |
| cold TTFT 64K (s) | 12.87 | 28.39 | 2.21x faster |
| short-code aggregate, 1 stream | 77.0 | 58.3 | +32.1 % |
| short-code aggregate, 6 streams | 191.9 | 92.4 | **+107.7 %** |
| sparkDash x1 | 151.3 | 120.8 | +25.2 % |
| staggered prefill-first newcomer L2/L4/L6 (s) | 2.42 / 2.47 / 2.55 | 7.93 / 8.05 / 8.16 | ~3.2x faster |
| staggered decode-first newcomer L2/L4/L6 (s) | 7.66 / 7.85 / 9.41 | 14.58 / 14.71 / 14.90 | 1.6-1.9x faster |
| admits | 16 | 6 | 2.7x |

At 8/12/16 streams RiNGSiDE reports 232.6 / 283.2 / 345.8 tok/s short-code aggregate and 154.0 / 194.4 / 225.6 prose
aggregate — a regime our `--max-num-seqs 6` cannot reach. Our own live gate (70-76 t/s warm decode) sits inside the
tonyd2wild band, so the lane is where the comparison row says it is.

RiNGSiDE's 512K/1M experiment (lab plan `89dfef84`, derived `6b38d997`): one boot with `--max-model-len 1048576` and
**24 GiB KV/GPU** completed a 520,014-token cold prefill in 116.0 s (4,483 tok/s), a 1,030,014-token prefill in
358.3 s (2,875 tok/s), and retrieved all three needles from a 1,000,083-token prompt; the pool was **4,065,684 KV
tokens shared** = 15.5x @262k, 7.75x @524k, 3.88x @1M. Their 512K recommendation is `--max-model-len 524288` +
`--kv-cache-memory 25769803776` (24 GiB/GPU), concurrency at 512K **an estimate**: 4 full 524K requests = 2,097,152
tokens, 6 = 3,145,728, 7 = 3,670,016, 8 = 4,194,304 (exceeds the observed pool).

## 3. Recommendations, one per item

Cost estimates assume the warm boot we already have (8-10 min) plus one gate run (~5 min).

### ADOPT NOW — launcher-only (no image change)

1. **Raise the KV pin 12 → 24 GiB/rank.** Adopt. Evidence on our own engine: the tonyd2wild TP4 arm ran 24.0 GiB/rank
   at 262,144 ctx on this ring; RiNGSiDE qualified 24 GiB at 1M ctx and observed 4.07M shared tokens. Expected pool
   ~3.1M tokens (≈12x @262k) instead of 1,576,246. Risk: our gotchas history is OOM-hang at *larger* pins under the
   JIT window; mitigate with the existing ritual (mem-free, 48G swap, no cache-drop daemon) and RiNGSiDE's rule:
   `MemAvailable >= 16 GiB` on every node after startup, stop if a rank falls below. Cost: one boot.
   **Verdict: adopt 24 GiB, not 32.**

2. **NCCL hardening env (3 lines).** Adopt: `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` (abort instead of hang — we hit
   wedged collectives twice), `TORCH_USE_RTLD_GLOBAL=1` (preloaded NCCL symbols are the ones used; we preload but
   never set this), `NCCL_IB_ROCE_VERSION_NUM=2` and `NCCL_IB_ADDR_RANGE=10.10.0.0/22` — the four ring cable
   subnets are 10.10.0.0/24, 10.10.1.0/24, 10.10.2.0/24 and 10.10.3.0/24 (`netplan/quad/*.yaml`), which a /22 covers
   exactly; this is the cheap equivalent of their subnet-pinned build. Cost: one boot, 4 lines.
   **Verdict: adopt.**

3. **Prefix-cache retention `--prefix-cache-retention-interval 2304`** (= block size). Adopt if the flag exists in
   our image (`docker run … /model --help | grep prefix-cache-retention`). Rationale: RiNGSiDE treats prefix retention
   as a weak-spot fix and sets it to one block; our cron/Hindsight clients replay near-identical prefixes
   (system prompt + template), so this turns prefill into cache hits. Cost: one boot; measure TTFT on a repeated
   prompt before/after. **Verdict: adopt (test).**

4. **Context budget hygiene: `--skip-mm-profiling` + `--limit-mm-per-prompt {"image":4,"video":1}`.** Adopt. Ours
   accepts `image:16` and profiles mm at boot; neither our agents nor crons send images. RiNGSiDE measured with the
   smaller budget. Cost: one boot, two flags. **Verdict: adopt** (keep `image:4` so vision still works if asked).

5. **Boot time: `--load-format instanttensor` + `INSTANTTENSOR_BACKEND=URING`.** Test. tonyd2wild's own TP4 recipe and
   RiNGSiDE both load with instanttensor; our cold boot is ~30 min dominated by 191G of shard load. If unsupported in
   `sm121-v11-dflash2` the engine fails at load — cheap to try, revert in one line. Cost: one boot.
   **Verdict: test once, keep only if it passes the gate and cuts load time materially.**

### ADOPT WITH A SEPARATE BOOT — context/throughput

6. **`--max-model-len 262144` → `524288`.** Adopt as the step after #1. Their setting is derived from a successful 1M
   boot; no image change. Validate 1 → 2 → 4 concurrent full-length requests (their estimated ceiling is 4 at 24 GiB;
   6 needs 3.15M tokens and is a later experiment). Our Hindsight repoint runs a 600K-context model; a 512K GLM
   endpoint changes that arithmetic. Cost: one boot + a long-context needle. **Verdict: adopt, concurrency target 4.**

7. **`--max-num-seqs 6` → 8, then test 16.** Adopt 8 first: our own EXL3 TP4 receipts measured 8 as the knob
   (135 vs 84 tok/s aggregate, 1.0 s vs 27 s worst TTFT), and RiNGSiDE's profile runs 16 with an explicit cudagraph
   grid. Going past the image's default captured sizes without graphs costs more than it gains, so pair 16 with
   `--compilation-config {"mode":3,"cudagraph_capture_sizes":[8,…,128],"compile_sizes":[8]}` mirrored from their
   plan — **only after confirming mode 3 exists in v11** (`--compilation-config` help) and reading
   `GLM53_SPEEDUP_GRAPH_MEMORY`-style graph-memory logs (ours has no such flag; use vLLM's memory log).
   Risk: extra captured graphs eat GB10 UMA. Cost: two boots. **Verdict: adopt 8; treat 16 as an experiment.**

### DEFER — the big item, needs its own decision

8. **The `glm53_speedup` scheduler + `glm53_prefill_rdma` + lean all-reduce stack.** This is where the +72 % prose,
   2.2x cold-prefill and 3.2x staggered-newcomer numbers come from. **Do not port file-by-file onto v11.** We already
   proved the failure class: the E27c port from the R10 fork died on 5 symbol drifts
   (`receipts/2026-09-26-e27c-port-reassessment.md`); RiNGSiDE's overlay targets vLLM **v0.29.0 backport**
   (`98dff2a8`), a third fork, and its `docker/base/v029-port/v029-glm53-local.patch` is 23,421 lines of exactly that
   porting work. Adoption path is one of:
   - (a) build RiNGSiDE's image from `docker/Dockerfile` on rank 0 and qualify it — their measured base image
     `017fd0ba` and native artefacts are **not published**, so a self-built image is unqualified by construction
     ("A self-built image is not the measured image"), build is **1.5-3 h**, and the Dockerfile has never been built
     (first build is its test; build inputs can disappear: fork commit + pinned base-image digests). This buys the
     measured numbers only after our own qualification.
   - (b) wait for a release of the same scheduler against the v11/tonyd2wild lineage.
   Cost: 1-2 days with qualification, not 1-2 h. **Verdict: defer; decide (a) vs (b) explicitly.** Note what changes
   if we take (a): `--tool-call-parser`/`--reasoning-parser` become `glm53` (our clients use glm47/glm45 semantics),
   the checkpoint becomes RedHatAI, and determinism becomes explicitly non-deterministic in ≥10-row decode launches —
   three client-visible changes that each need a gate re-run and a documented notice.

### REJECT / KEEP AS IS

9. **The custom NCCL build (patches 0001+0002).** Reject. Our switchless-nccl v0.0.1 is what RiNGSiDE's own tonyd2wild
   comparison row ran on ("on this ring's NCCL transport"), so the transport is not the delta; rebuilding NCCL from
   their tree is a new build with a hard subnet pin (10.100.224-227/24) that does not match our 10.10.0-3 addressing.
10. **`--gpu-memory-utilization 0.85` → `0.87`.** Skip. It is their fork's number; our margin on GB10 UMA is the
    binding constraint and #1 is the controlled way to grow KV.
11. **`--override-generation-config {"top_p":0.95}` and `{"clear_thinking": true}`.** Do not copy. `top_p 0.95` is the
    RedHatAI checkpoint's generation default, ours is the NVIDIA checkpoint; `clear_thinking` belongs to their
    template. Keep our `reasoning_effort low` kwarg (it is what keeps reasoning turns from eating the cron budget).
12. **`--per-request-spec-decode-metrics detailed` / `--cudagraph-metrics` / `--kv-cache-metrics` / profiler flags.**
    Skip. They require fork symbols (`RequestSpecDecodeMetrics`, `per_request_spec_decode_metrics`) that are absent
    in v11 — the same symbols that aborted our E27c port. Use our existing `/metrics` scrape instead.
13. **weightless GLP-44 steering (TP4 option, off by default).** Skip. Refusal suppression is not our use case and it
    costs 6-7 % prefill.
14. **Fabric management: runtime `fabric-setup.sh` vs our static netplan.** Keep ours (netplan + `verify-fabric.sh`).
    Cheapest port is their check semantics: `preflight_tp4.py` verifies each function's IPv4 address lies in **its
    cable's** subnet — we verify GID index/type/ndev/MTU but not "address belongs to this cable's subnet". Optional,
    ~20 lines in `verify-fabric.sh`. **Verdict: optional, low value.**

### ADOPT — measurement discipline (highest value per hour)

15. **Adopt RiNGSiDE's measurement protocol, not just its settings.** Adopt the RigMark fork they used
    (`othexmr/rigmark` branch `feat/staggered-arrival`, commit `40fabcaf`, protocol 1.1.0) with the cells they report:
    cold TTFT at 8K/32K/64K, single-stream decode on code/prose/structured, short-code aggregate at 1/6/8/16 streams,
    staggered prefill-first and decode-first newcomer latency, sparkDash x1. Our current gate (needle + tool-call +
    warm decode) proves it serves; it cannot compare recipes or prove a change helped. Cost: client-side setup, ~1 h;
    medians of 2 runs per cell instead of their single-run gate. **Verdict: adopt as the standard for any change from
    this list** — it is the only way "improvement" stops being an assertion. (AGENTS.md already requires measured
    results to be kept separate from estimates; this is that tool.)

## 4. Sequence

1. Launcher-only batch: #1 (24 GiB), #2 (NCCL env), #3 (retention), #4 (mm), one boot, full gate + `just quad-verify`.
2. #15 in parallel (client side, no node change), then one RigMark window to fix the baseline in our own numbers.
3. #5 (`instanttensor`) as a single test boot; revert if it fails.
4. #6 (512K @24 GiB) as its own boot with the concurrency ladder 1 → 2 → 4 and the 16 GiB MemAvailable margin check.
5. #7 (seqs 8, then 16 with compilation-config) as one or two boots.
6. Decision on #8 (build their image and qualify, or wait), then #9-#14 as-is.

Nothing here needs a node until you ask for it: steps 1-5 are launcher edits in
`quad-launcher/rank-launcher.sh` + `cluster.env` (sha-pinned there) with a receipt per boot, step 6 is a
build-and-qualify decision with a 1-2 day budget.

## 5. Governance notes

- RiNGSiDE's own rule (its `AGENTS.md`): a profile changes only through `sources/import_plan.py` from a lab plan, and
  `sources/verify.py` fails if a flag, mount or served file changes without a new plan. Any adoption that touches
  *their* profile needs a lab plan, not an edit. (Their 512K experiment is documented as generated-file edits that
  `launch/up.sh` would overwrite — do not promote it that way.)
- Our lane: site values stay in `site.env`/`cluster.env`; per-boot receipts under `quad-launcher/receipts/`;
  measured vs estimate labelled in every table above.
- Keep the non-deterministic MoE accumulation of both lanes visible: RiNGSiDE is non-deterministic by declaration
  (atomic adds), ours is marlin + dflash; do not publish a single-run "bit-identical" claim for either.
