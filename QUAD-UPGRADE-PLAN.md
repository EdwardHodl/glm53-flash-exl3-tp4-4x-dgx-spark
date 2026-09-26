# QUAD-UPGRADE-PLAN — unified adoption plan for the fire TP=4 GLM-5.3-Flash lane

Written 2026-09-26. Status: **plan — nothing executed.** This document supersedes,
for execution purposes, the three input analyses, which remain on file as evidence:

- `QUAD-ALEXELLIS-PROPOSAL.md` — the deployed baseline proposal (commit `c6d7f86`, GATE GREEN).
- `QUAD-RINGSIDE-PROPOSAL.md` — the full analysis of `othexmr/…RiNGSiDE` @ `6606f76`
  (moved onto this filename at 14:24; the short stub it replaced is retired).
- `QUAD-KNAPCIO-PROPOSAL.md` — the full analysis of `knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4`
  @ `dddb034` (clone: `recipes/glm53flash/GLM-5.3-Flash-4x-DGX-Spark-TP4`, moved onto
  this filename at 14:24).
- `quad-launcher/receipts/2026-09-26-ringside-vs-fire-quad-comparison.md` — independent
  RiNGSiDE review with the same-ring head-to-head table (our engine measured as the
  "tonyd2wild TP4" row).

Where the input documents disagreed, the unified plan states the resolution
explicitly (§ "Disagreements resolved").

**Correction note (2026-09-26 14:4x).** This plan was first drafted against the short
stubs that briefly occupied the two `QUAD-*` proposal filenames, so its disagreement
table originally described texts that no longer existed. The table below now cites the
consolidated analyses. **No resolution changes**: 24 GiB/rank, skip 0.87, defer RedHatAI,
seqs 8 first, RigMark as the gate all stand. Four items were added or strengthened from
the consolidated knapcio analysis: the prefix-cache repair (§Phase 1 item 3), the
FlashInfer workspace cache (§Phase 3 item 9g), a quality gate for the ulp-level overlays
(§Phase 3 item 9h), and the mm-budget decision (§Phase 1 item 4).

## 0. Ground truth established by the three analyses

- **Our engine was measured head-to-head.** RiNGSiDE's bench table ran a
  tonyd2wild TP4 recipe (the same `vllm 0.1.dev20051+g487ecf187` we serve) on the
  same class of switchless ring: prose decode 35.8 vs their 61.7 (+72%), code
  82.5 vs 105.3, cold TTFT 64K 28.4 s vs 12.9 s (2.2×), staggered prefill-first
  newcomer ~8 s vs ~2.5 s (3.2×). Our live gate (70–76 t/s warm decode) sits
  inside the tonyd2wild band — the lane is where the comparison says it is.
- **The transport is not the delta.** The tonyd2wild row ran on our
  switchless-nccl v0.0.1; do not rebuild NCCL.
- **knapcio builds on our exact image family** (tonyd2wild v11, `487ecf187`).
  Its scheduler/KDA/argmax/L2-prefetch work is bind-mountable onto our lane with
  no rebuild — unlike RiNGSiDE's `glm53_speedup`, which targets a vLLM v0.29.0
  backport (a third fork; porting it file-by-file repeats the E27c failure, 5
  boots / 5 symbol drifts).
- **knob conservatism is our biggest cheap loss.** seqs 6 / KV 12 GiB/rank were
  bring-up values, not measurements. Our own EXL3 TP4 receipt already showed
  seqs 8 = 135 vs 84 tok/s aggregate, 1.0 s vs 27 s worst TTFT.
- **The single biggest scheduler win is not portable.** RiNGSiDE's saturation
  scheduler lives in its fork; getting it means adopting its image wholesale
  (unpublished build, self-built = unqualified by construction, 1.5–3 h build
  + 1–2 days qualification, 3 client-visible changes: parser set, checkpoint,
  determinism). That is a separate decision (Phase 4), not a step.

## 1. Disagreements resolved

| Topic | RINGSiDE proposal (this session) | Independent receipt | knapcio proposal | **Resolution** |
|---|---|---|---|---|
| KV pin | **24 GiB/rank**, not 32 (RiNGSiDE's published default is 32) | **24 GiB/rank**, not 32 | 24 GiB/rank (their profile) | **24 GiB/rank** (`--kv-cache-memory 25769803776`). 32 is RiNGSiDE's fork number; our own evidence (tonyd2wild arm, RiNGSiDE 512K experiment, knapcio profile) all says 24. UMA margin is the binding constraint. |
| mem-util 0.85→0.87 | skip | skip | 0.78 | **Skip 0.87.** Grow KV via the pin; UMA OOM is our 6-boot history. |
| RedHatAI checkpoint switch | defer to Phase 4 | defer (their generation default; tied to image) | n/a (stays on nvidia) | **Defer to Phase 4.** On our checkpoint the top_p override is wrong (it is RedHat's default, not NVIDIA's). |
| max-num-seqs | **8 first**, 16 as an experiment (RiNGSiDE's published default is 16) | 8 first, 16 as experiment | 32 | **8 first** (our own EXL3 receipt), then a measured step to 16 with compilation-config only if mode 3 exists in v11. |
| Measurement protocol | **adopt RigMark protocol — named the highest-value item** | **adopt RigMark protocol** | their own sparkDash discipline + qeval/KLD quality gate | **Adopt RigMark fork** (`othexmr/rigmark`, branch `feat/staggered-arrival`, commit `40fabcaf`) as the standard throughput gate for every phase, and knapcio's `qeval`/KLD probes as the quality gate wherever numerics move. Our needle gate proves "serves"; it cannot prove "improved". |

## 2. Unified plan — six phases, strictly sequential

Every phase: one boot, receipt written to `quad-launcher/receipts/`, RigMark
window (medians of 2) + needle + tool-call gate before promotion. All launcher
changes are edits to `quad-launcher/rank-launcher.sh` + `cluster.env`
(sha-pinned) — no image change until Phase 4. Rollback per phase = revert the
edits + one boot. Estimated cost per boot ~10–30 min (warm JIT caches) + gate.

### Phase 1 — launcher-only batch (one boot, ~1.5 h total)

1. **KV pin 12 → 24 GiB/rank**: `--kv-cache-memory 25769803776`.
   Expected pool ~3.1M tokens (~12× @262k) vs today's 1,576,246.
   Guard: RiNGSiDE's rule — every rank `MemAvailable ≥ 16 GiB` after startup,
   stop otherwise; keep the existing ritual (mem-free, 48G swap, no cache-drop daemon).
2. **NCCL hardening env (4 lines)**: `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`,
   `TORCH_USE_RTLD_GLOBAL=1`, `NCCL_IB_ROCE_VERSION_NUM=2`,
   `NCCL_IB_ADDR_RANGE=10.10.0.0/22` (covers our four cable subnets exactly).
3. **Prefix caching and its repair (the largest measured win we have).**
   Add explicitly `--enable-prefix-caching --enable-chunked-prefill`, **and mount
   knapcio's `overlay/kv_cache_coordinator.py` over
   `vllm/v1/core/kv_cache_coordinator.py`** (tonyd2wild's DFlash2 prefix-cache
   repair, same image family). Receipt: repeated 20k-token prompt TTFT
   **8.1 s → 0.63 s** (knapcio commit `c3d81d2`). Additionally take RiNGSiDE's
   `--prefix-cache-retention-interval 2304` (= block size) if the flag exists in
   our image (`--help | grep prefix-cache-retention`).
   Verify: cached-token counters in the response plus repeated-prompt TTFT
   before/after; v11 may already enable prefix caching by default, so the flags
   are asserted for explicitness and the mount is the actual repair.
   sha-pin the mounted file from knapcio's `docs/runtime-source-manifest.json`
   (`accepted_sha256`) in `cluster.env`, next to the existing three pins.
4. **Context budget hygiene — pick one, deliberately**: either RiNGSiDE's
   `--skip-mm-profiling` + `--limit-mm-per-prompt {"image":4,"video":1}`, or
   knapcio's keep-`image:16` + `--mm-processor-cache-gb 4`
   + `--mm-processor-kwargs {"max_pixels":6422528,"max_image_tokens":4096}` and
   `video:0`. Resolution: **take `--skip-mm-profiling` plus knapcio's mm cache
   and keep `image:16`, `video:0`** — it is the superset of our current posture
   (we accept `image:16` today), it removes the profiling pass, and knapcio's
   cache is the documented fix for the `mm_hash` assertion they hit with many
   images. Dropping to `image:4`/`video:1` is RiNGSiDE's capability set, not
   ours; adopt it only if we decide the capability is unwanted.
5. **Baseline in our own numbers**: install the RigMark client (no node change)
   and take one window **before** promoting Phase 1 — this is the "before" column
   for every later phase. Cells: cold TTFT 8K/32K/64K, single-stream
   decode code/prose/structured, short-code aggregate 1/6/8/16, staggered
   prefill-first/decode-first newcomer, sparkDash x1.

Gate: `just quad-test` + `just quad-verify` + RigMark window, **plus one `qeval`
run** (the prefix-cache repair changes the KV/prefix path, so a correctness check
belongs in Phase 1 even though no numerics knob moved). Promote only if
TTFT/aggregate improve or hold, qeval ≥72/75, and the gate stays green.

### Phase 2 — boot-shape experiments (one boot each, promote/revert per boot)

6. **`--load-format instanttensor` + `INSTANTTENSOR_BACKEND=URING`** — cut the
   ~30 min shard load. Cheap to try, revert in one line if unsupported in
   sm121-v11-dflash2. Keep only if the gate passes and load time drops materially.
7. **512K context**: `--max-model-len 524288` (keep 24 GiB). Concurrency ladder
   1 → 2 → 4 with the long-context needle (RiNGSiDE's 1M experiment completed a
   1,030,014-token prefill at 2,875 tok/s with needle retrieval at 24 GiB/GPU;
   estimated ceiling 4 concurrent 524K requests). This changes the Hindsight
   repoint arithmetic (that model runs 600K ctx) — note it in the ops doc if promoted.

### Phase 3 — admission and knapcio overlays (our image, mountable)

8. **`--max-num-seqs 6 → 8`.** Then, as a separate boot, 16 with
   `--compilation-config {"mode":3,"cudagraph_capture_sizes":[8,…,128],"compile_sizes":[8]}`
   only after confirming mode 3 exists in v11; past default capture sizes without
   graphs costs more than it gains. Watch UMA (graph memory) at each step.
9. **knapcio overlay mounts, one group per boot, behind a `QUAD_LEVERS=on|off`
   flag** (the E27c rollback pattern; files from
   `recipes/glm53flash/GLM-5.3-Flash-4x-DGX-Spark-TP4/overlay/`, same image family):
   a. **Batch-uniform adaptive draft** (`LeversScheduler`,
      `GLM_LV_MODE=batch-uniform`, SPEC_TABLE `[[1,1,7],[2,2,5],[3,32,3]]`) —
      mixed-k steps fall back from full CUDA-graph replay; one uniform k per
      batch keeps replay. Their measured c4 prose aggregate 152.9 (median of 3).
   b. **KDA verify-state stash** (`GLM_KDA_STASH=1`,
      `GLM_KDA_STASH_NOCOPY=1`, `GLM_KDA_FLAG_FUSED=share`) — ulp-level FP32
      differences; target paths carry numerical checks.
   c. **Vocab-parallel greedy argmax** (`GLM_TARGET_VOCAB_ARGMAX=1`,
      `GLM_LV_ARGMAX_MINTOK=1`) — same idea as RiNGSiDE's
      `GLM53_GREEDY_ARGMAX_VERIFY`, written for our image family.
   d. **Replicated projection splitting** (`GLM_DS_SPLIT=fc:128,qkv_a:64,wq_b:16`,
      `GLM_DS_DRAFT_HEAD_FP8=1`, `GLM_DS_TOPK_ONE_GATHER=1`) — qualify per path;
      leave `GLM_LV_SPLIT_INEXACT=fc` off unless the target-verified path covers it.
   e. **kpool/indexer fix pack** (`GLM_KPOOL_FIX=1`) — padded seed stride,
      speculative ring, hybrid tail-slot mapping (the same class as RiNGSiDE's
      vllm#58454/#50843 ports, but same-image). Gate with needle + long-context sanity.
   f. **L2 prefetch** (`GLM_L2_PREFETCH=1` + AR knobs) — 0.61 ms/step at c1,
      CI-excluded vs a ±0.2 ms A/A band. Small, cleanly measured.
   g. **Boot warmup / fast load** (`GLM_FAST_LOAD=1`, `FI_CACHE_PERSIST=1`,
      `-e FLASHINFER_WORKSPACE_BASE=/cache/fi`, their `boot_warm.py`) — 129 s
      warm boots on their fleet. **Mounting `/root/.cache` alone does not
      deliver this**: FlashInfer does not read `XDG_CACHE_HOME` and rebuilds
      three modules per boot (top-k ~92 s, batch MLA ~7 s, sampling ~38 s), so
      the FlashInfer workspace base plus the Triton/Inductor/CUDA cache vars
      (we already set some of the latter) are the actual lever; `GLM_FAST_LOAD=1`
      removes a further ~230-280 s of file-backed mmap copies on the head.
   h. **Quality gate for the numerics-moving overlays (b, c, d).** Run knapcio's
      `bench/qeval.py` (require ≥72/75 at c1 and at c4) and their
      `bench/kld_probe.py` + `compare_kld_strict.py` teacher-forced probe against
      a retained reference. Decode throughput cannot qualify an ulp-level change
      or a projection split; their own standard is arithmetic checks on tensors
      (bit-exact or ULP) plus this panel, not greedy-text equality.
   Order a→h, gate between each (throughput gate as stated; the quality gate is
   mandatory for b, c and d); a failed mount reverts by unsetting one flag.
10. **Do NOT adopt (standing rejects, all three analyses agree):**
    - Atomic MoE accumulation (`B12X_GLM53_ATOMIC_*`) — makes outputs
      non-deterministic run-to-run. Our lane has no atomic adds (marlin MoE),
      but it is **kernel-deterministic, not bit-reproducible**: this engine
      moves greedy text with batching changes (knapcio's own scheduler docstring
      says so) and we have never measured run-to-run equality. State it exactly
      that way in any published number (AGENTS.md: keep the non-deterministic
      accumulation posture visible; do not call our lane "deterministic").
      Revisit only as part of a Phase 4 wholesale adoption.
    - RoCEnante collectives on our ring — built and measured on a switched
      two-rail fabric; transport qualification required first. Reject as a mount.
    - Their NCCL builds (RiNGSiDE 0001+0002, knapcio official NCCL host build) —
      our switchless v0.0.1 is proven; subnet pins don't match our addressing.
    - Fork-only flags (`--per-request-spec-decode-metrics detailed`,
      `--cudagraph-metrics`, `--kv-cache-metrics`, profiler env) — v11 lacks the
      symbols; same failure class as E27c. Use our `/metrics` scrape.
    - `top_p 0.95` override / `clear_thinking` template / weightless GLP-44
      steering — wrong checkpoint default, foreign template, not our use case.
    - TP2-lane knobs (e.g. `DFLASH_TOKENS=3`) into TP4.

### Phase 4 — decision gate: adopt the RiNGSiDE stack wholesale or wait

11. **Decision item (needs explicit user choice, 1–2 day budget):**
    - **(a)** Build RiNGSiDE's image from `docker/Dockerfile` on rank 0 and
      qualify it (~1.5–3 h build; first build is its test; inputs can
      disappear). Buys the +72% prose / 2.2× prefill scheduler stack only
      after **our** qualification. Client-visible changes to schedule:
      `glm53` parsers (clients use glm47/glm45 semantics), RedHatAI checkpoint
      (+ NVIDIA tokenizer, `top_p 0.95` default), and declared
      non-determinism in ≥10-row decode launches — each needs a gate re-run
      and a documented notice.
    - **(b)** Wait for the same scheduler to be released against the
      v11/tonyd2wild lineage.
    - **(c)** Stay on Phase-3 overlays (knapcio path) and revisit in a month.
    Recommendation: **run (c) first regardless** — it is cheap and image-safe —
    and decide (a) vs (b) only after Phase 3 results are in. If Phase 3 closes
    most of the gap, (b) likely wins.
12. **If (a) is chosen**, sequence inside it: build → CPU checks → one boot →
    RigMark window vs the Phase-3 baseline → decide determinism flags
    (flip `B12X_GLM53_ATOMIC_*` back off and re-qualify, accepting some decode
    loss) → client notice for parser/checkpoint changes.
13. **Deferred / skip for now:** `lossless8` weight conversion (knapcio) —
    offline converter, full numeric revalidation required; only worth it if we
    commit to the Marlin/RoCEnante path, which Phase 4 (a)/(b) may make moot.
    Optional low-value: extend `verify-fabric.sh` with per-cable subnet checks
    (~20 lines, RiNGSiDE semantics).

## 3. Execution rules

- One variable set per boot. No phase mixes a knob change with an overlay mount.
- Every boot gets a receipt file under `quad-launcher/receipts/` (pattern:
  `2026-09-26-e27c-port-reassessment.md`), recording: what changed, gate result,
  RigMark cells, promote/revert decision.
- Measured vs estimated is labelled in every table (AGENTS.md rule).
- Site values stay in `cluster.env`/`site.env`; never commit site specifics.
- Keep both lanes' determinism posture visible in any published number.
- Nothing here touches a node until you say go. Phases 1–3 are launcher edits
  with per-boot receipts; Phase 4 is a decision document with a day-scale budget.

## 4. First action

Phase 1, step 1–5 as one batch: four launcher edits + RigMark client install,
one boot, full gate, receipt. Tell me to start and I will execute Phase 1 end
to end (~1.5–2 h including the two boots of margin).
