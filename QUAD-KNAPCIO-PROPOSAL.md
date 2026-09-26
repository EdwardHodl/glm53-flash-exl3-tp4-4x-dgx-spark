# Proposal — adopt knapcio's TP4 levers onto fire-quad

Date: 2026-09-26. Status: **PROPOSAL — nothing executed; no node accessed, no container touched.**
Author: Hermes (devops), for Edward.
Companion documents: `QUAD-RINGSIDE-PROPOSAL.md` (the other upstream assessed this session) and
`quad-launcher/receipts/2026-09-26-ringside-vs-fire-quad-comparison.md`.

---

## 1. Source of the analysis (provenance)

| source | identity | what was taken |
|---|---|---|
| `~/Workspaces/devops/recipes/glm53flash/GLM-5.3-Flash-4x-DGX-Spark-TP4` | clone of `github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4`, HEAD `dddb034717cb84c362d8a370b81723fa4ee0c4d2` (Zbigniew, 2026-09-26 22:19:20 +0200), 19 commits, `main` in sync with `origin/main`, clean | the whole comparison target |
| `README.md` | upstream prose | serving profile, sparkDash table, prefill, warm boot 129 s, reproduce steps |
| `profiles/current.env`, `.env.example`, `start.sh` | the accepted LVKP-S-L2 launch | every flag, env var, mount and container option |
| `docs/results/2026-09-26-l2.md`, `docs/results/2026-09-26-l2-supplement.md`, `docs/results/2026-09-26-prefill.md` | their measured windows | decode table (per-stream + aggregate), prefill rates, hashes |
| `docs/history.md`, `docs/history-2026-09-18.md` | experiment log | what was promoted, what failed its A/A gate, L2 prefetch contrast (0.610 ms/step c1, CI) |
| `docs/validation.md` | their gates | sparkDash job validation, qeval ≥72/75, KLD panel, A/B discipline (CI wholly above 0.2 ms, A/A within ±0.2 ms) |
| `docs/install.md`, `Dockerfile.roce` | build and base pin | base image digest, NCCL 2.30.7 host build, RoCEnante layer |
| `docs/weights.md`, `scripts/build_lossless8.sh`, `scripts/glm_quant_mix.py`, `scripts/drafter_fp8.py` | weight preparation | `lossless8` conversion, revision pins, config hashes |
| `docs/runtime.md`, `docs/runtime-source-manifest.json` | publication boundary | per-file `accepted_sha256` / `published_sha256` for the overlay — our pin source |
| `NOTICE`, `CREDITS.md`, `LICENSES/*`, `roce/b12x/PROVENANCE.json` | attribution | licence boundaries, including the tonyd2wild gap |
| `overlay/*.py` (30 files) | the levers | scheduler, kpool/indexer fixes, parsers, fast load, KDA stash, L2 prefetch |
| `bench/*` (`final_sparkdash.py`, `run_prefill.py`, `prefill_checked.py`, `qeval.py`, `kld_probe.py`, `compare_kld_strict.py`, `hardset.py`, `conc_bench.py`, `bench_matrix.py`) | their harness | measurement standard |
| fire side: `quad-launcher/rank-launcher.sh` (sha `87143c23392e…`), `cluster.env`, `PROGRESS.md`, `QUAD-ALEXELLIS-PROPOSAL.md`, `receipts/baseline-metrics-20260926-0115.txt` | ours | the baseline being changed |

**The linchpin, verified by digest.** Their `Dockerfile.roce` pins
`ARG BASE_IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`
(install.md L19-L27) — the **same digest fire-quad runs** (`sm121-v11-dflash2`, "ARM64 tonyd2wild v11 DFlash2 with
vLLM487ecf187"). Their weights start from `nvidia/GLM-5.3-Flash-NVFP4` revision
`09b04e5e74bca08ca8549fc736d4cdd8624bfde3` (weights.md L6) — **our exact revision** — same drafter family (incoai
DFlash2), same `262144` context, same `2304` block size, same `marlin` MoE backend, same `fp8_e4m3` KV. Their overlay
docstrings say so in writing: `overlay/glm_l2_prefetch.py` and `overlay/glm_kda_stash.py` are "for the Tony v11 image
(vLLM 0.1.dev20051+g487ecf187)". Their files are therefore mounts **on our image**, not a different fork.

Second linchpin: they load their scheduler with `--scheduler-cls glm_levers_sched.LeversScheduler` plus
`-e PYTHONPATH=/overlay/overlay` (start.sh `sched`) — **not** by replacing `vllm/v1/core/sched/scheduler.py`. That is
exactly the route that failed our E27c port (`receipts/2026-09-26-e27c-port-reassessment.md`, 5 symbol drifts from
file replacement). This repo demonstrates a working scheduler change on our image.

---

## 2. What their accepted profile is, and what it measures

Accepted profile **LVKP-S-L2** (`profiles/current.env`): NVFP4 routed experts + `lossless8` 8-bit non-expert weights;
`MOE_BACKEND=marlin`; `MAX_MODEL_LEN=262144`; `KV_BYTES=25769803776` (24 GiB/rank fp8); **`MAX_SEQS=32`**;
`BATCHED_TOKENS=6912`; adaptive draft `K_HI=7`, `K_LO=3`, batch-uniform; `CAPTURE_SIZES` to **256** with
`FULL_DECODE_ONLY`; `GPU_UTIL=0.78`; `MM_IMAGES=16` + 4 GB mm processor cache; prefix caching, chunked prefill,
FlashInfer autotune off; loopback API on `8093`, alias `GLM-5.3-Flash-FP8`; container `--memory 112g`; RoCEnante
one-shot collectives (switched two-rail fabric).

Measured (their own, sparkDash DecodeBench, 256 output tokens, temperature 0, thinking off):

| prompt | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose aggregate (per stream) | 70.19 | 107.45 (57.95) | 152.89 (39.39) | 221.70 (28.83) | 312.69 (20.34) |
| code | 126.44 | 160.34 (81.53) | 185.37 (50.37) | 231.27 (31.77) | 308.26 (21.37) |
| structured | 161.11 | 140.48 (80.32) | 196.68 (52.98) | 242.16 (32.61) | 323.78 (23.67) |
| JSON | 124.25 | 115.55 (61.73) | 193.62 (51.44) | 303.75 (41.90) | **476.41** (32.25) |

Prose c1 = median of 5, prose c4 = median of 3; other cells single runs (labelled as such upstream).
Prefill (unique prefixes, concurrency 1, medians of 3): **2,184.83 tok/s** at 16,705 tokens (7.647 s to first token),
**2,188.94** at 33,574 (15.338 s), **2,178.06** at 67,304 (30.903 s). Quality: qeval **75/75** at c1 and c4; teacher-forced
KLD **0.0288366** (17 items, 6,618 positions); long-context registry sanity 32/32 at ~9.6k tokens. Boot: **129 s warm
cache** (271 s cold). L2 prefetch: +0.610 ms/step at c1 (95 % CI [0.548, 0.671]) with a passing duplicate-baseline A/A.

Where our lane stands (fire-quad): 262,144 ctx, **6 seqs**, 12 GiB KV pin, `marlin`, k=7 static dflash, no
`--enable-prefix-caching`/`--enable-chunked-prefill` flags, no mm processor cache, warm boot 8-10 min, gate warm decode
**70-76 t/s**, KV pool 1,576,246 tokens. **Our prefill is unmeasured** (the gate does not measure it) — so their prefill
numbers are a reference band, not a delta we can claim.

Cross-protocol honesty: their sparkDash code c1 (126.44) vs the tonyd2wild TP4 arm's sparkDash (120.8) is **+4.7 % on
one protocol**; their prose c1 (70.19) sits inside our gate band. Their large wins are at **concurrency** and **boot /
prefix reuse**, not single-stream prose. RiNGSiDE's RigMark prose cell (61.7) is a different protocol and is not
comparable to either.

---

## 3. What this repo has that fire-quad does not

| # | lever | their measured/claimed value | how it reaches the engine |
|---|---|---|---|
| A | requirement: prefix caching + chunked prefill | repeat 20k prompt TTFT **8.1 s → 0.63 s** (commit `c3d81d2`) | flags + mount `overlay/kv_cache_coordinator.py` (tonyd2wild's DFlash2 prefix-cache repair) |
| B | slab loader for GB10 file-backed copies | stock loader spends **~230-280 s** in `cuMemcpyHtoDAsync` on the head; warm boot 271 → 128 s | `overlay/glm_fast_load.py` + `GLM_FAST_LOAD=1` |
| C | FlashInfer JIT cache persistence | 3 modules rebuilt per boot (top-k ~92 s, batch MLA ~7 s, sampling ~38 s) | `FLASHINFER_WORKSPACE_BASE=/cache/fi` (+ Triton/Inductor/CUDA cache env) |
| D | kpool / indexer correctness bundle | vllm#57477 padded seed stride, #58454 spec-decode ring, #53906 hybrid tail slot map — long-context OOB class | `GLM_KPOOL_FIX=1` + 4 mounts (`glm5next_kpool_compress`, `glm5next_attention`, `mla_indexer`, `mamba_hybrid`) |
| E | parser fixes | literal tool delimiters + stop-anchor fixes (logprobz vllm#639/#640) | mounts over `vllm/parser/glm47_moe.py`, `vllm/parser/abstract_parser.py` — our exact parser names |
| F | KDA spec-verify stash | removes per-token fp32 state stores; ulp-level, not bit-exact | `GLM_KDA_STASH=1` + `glm_kda_stash.py`, `glm_kda_stash_fast.py`, `kda_stash.py` |
| G | L2 prefetch of next dense weights | **0.610 ms/step c1**, 0.727 ms c4, CI above the gate | `GLM_L2_PREFETCH=1` + `overlay/glm_l2_prefetch.py` |
| H | custom scheduler without file replacement | batch-uniform k∈{3,7} from a per-request acceptance EMA, rank-invariant by construction | `--scheduler-cls glm_levers_sched.LeversScheduler` + `PYTHONPATH` + policy JSON |
| I | admission + graphs | 32 sequences admitted; capture sizes to 256 | `MAX_SEQS=32`, explicit capture set, `FULL_DECODE_ONLY` |
| J | memory discipline | container capped at 112 GB on a 128 GB node; `--ulimit stack/nofile`; `spawn` workers | docker options |
| K | mm hygiene | 16 images/prompt with a 4 GB processor cache after an `mm_hash` assertion with many images | `--mm-processor-cache-gb 4`, `--mm-processor-kwargs {max_pixels:6422528,max_image_tokens:4096}`, video 0 |
| L | weight conversion | `lossless8`: ~18 GiB of BF16 non-expert tensors → MXFP8/block-128 FP8; retained BF16 for sensitive tensors | offline converter (`scripts/build_lossless8.sh`, `glm_quant_mix.py`, `drafter_fp8.py`) |
| M | measurement + quality gate | sparkDash matrix with strict job validation, qeval 75 tasks, KLD probe, A/B in-boot with duplicate A/A control | `bench/*` |

---

## 4. Recommendations, one per item

Cost baseline: our ritual `just mem-free-quad` → `just quad-up`, warm boot 8-10 min, gate ~5 min.

### Phase 1 — launcher-only (flags, mounts, docker options; 1-2 boots). Adopt first.

1. **Prefix caching + chunked prefill, explicit** — add `--enable-prefix-caching --enable-chunked-prefill`, and mount
   `overlay/kv_cache_coordinator.py` over `vllm/v1/core/kv_cache_coordinator.py` (sha from their manifest), sha-pinned
   in `cluster.env`. **Verdict: adopt.** This is the highest value per risk for our workload: Hindsight and the cron
   jobs replay near-identical long prefixes, and their receipt is 8.1 s → 0.63 s TTFT. Verify: TTFT of a repeated
   20k-token prompt before/after; cached-token counters in the response.
2. **KV pin 12 → 24 GiB** (`--kv-cache-memory 25769803776`). **Verdict: adopt** (agrees with the RiNGSiDE proposal's
   item 1; the tonyd2wild TP4 arm ran 24 GiB on our ring). Verify: pool tokens ≈2x, `MemAvailable ≥ 16 GiB` per node.
3. **FlashInfer workspace persistence** — `-e FLASHINFER_WORKSPACE_BASE=/cache/fi` (keep our `/root/.cache` mount;
   add Triton/Inductor/CUDA cache vars as they do). **Verdict: adopt.** We mount `/root/.cache`, but their note says
   FlashInfer does not read `XDG_CACHE_HOME` and rebuilds three modules per boot. Verify: boot time and
   `cache/fi` contents growing across boots.
4. **Slab loader** — mount `overlay/glm_fast_load.py`, `GLM_FAST_LOAD=1`. **Verdict: adopt as a test.** Their claim is
   ~230-280 s of head-side load cost removed; failure mode is a load-time error, revertible in one line.
5. **kpool/indexer + parser mounts** — `GLM_KPOOL_FIX=1` with its four mounts, plus `glm47_moe.py` and
   `abstract_parser.py`. **Verdict: adopt.** They are correctness fixes (spec-decode ring, hybrid tail slot mapping,
   padded seed stride) for our exact image, and the parser pair improves the tools/reasoning path we already use; no
   parser-name change, so no client-visible semantics change. Their warning stands: the kpool fixes are only safe
   together.
6. **Admission ladder + graphs** — `--max-num-seqs 8` first, then 16, and (only if the pool and graphs hold) 32 with
   capture sizes through 256 and `--max-cudagraph-capture-size 256`, `FULL_DECODE_ONLY`, `mode 0`.
   **Verdict: adopt 8 now; 16/32 as experiments.** Our cap of 6 truncates the exact regime their table monetizes
   (c16 aggregate 312-476 tok/s); graphs must exist for the sizes we admit.
7. **Docker hygiene** — `--memory 112g --memory-swap 112g`, `--ulimit stack=67108864`,
   `--ulimit nofile=1048576:1048576`, `-e VLLM_WORKER_MULTIPROC_METHOD=spawn`. **Verdict: adopt.** Our bring-up saga
   was a UMA memory equation (3 kernel OOMs); a container cap makes the failure land inside the container instead of
   the node, and `spawn` removes fork-related surprises under unified memory. Watch: a cap too low kills the engine
   at JIT peaks — start at 112 GB as they do on 128 GB nodes.
8. **mm hygiene** — `--mm-processor-cache-gb 4`, `--mm-processor-kwargs {"max_pixels":6422528,"max_image_tokens":4096}`,
   `{"image":16,"video":0}`. **Verdict: adopt if any client sends images; otherwise optional.** Note this *revises* the
   RiNGSiDE proposal's item 3 (which dropped to `image:4`): two upstreams both run 16 images, one with a processor
   cache. Keeping `image:16` plus their cache is the safer superset.
9. **Vision/mm parsing unchanged; sampling unchanged.** Keep `--reasoning-parser glm45 --tool-call-parser glm47`,
   `reasoning_effort low`, our alias `glm-5.3-flash`, port `8888`, host `0.0.0.0`. **Verdict: keep ours** — their
   `high` effort, loopback bind, `8093` and `GLM-5.3-Flash-FP8` alias are their cluster's choices, not levers.

### Phase 2 — the engine levers that need a receipt each (2-3 boots)

10. **KDA stash** (`GLM_KDA_STASH=1`, `GLM_KDA_STASH_NOCOPY=1`, `GLM_KDA_FLAG_FUSED=share`) with the three mounts.
    **Verdict: adopt as an experiment.** Their claim is ulp-level (not bit-exact) numerics — so it is a numerics
    change, not a pure win, and it interacts with prefix caching (their docstring: align-mode copies of intermediate
    spec slots). Verify with a quality gate, not with greedy-text equality.
11. **L2 prefetch** (`GLM_L2_PREFETCH=1`, `_MB=6`, `_AR_MB=4`, `_MAXTOK=32`). **Verdict: adopt as an experiment.** This
    is the file written for our image, with the strongest evidence in the repo: +0.610 ms/step at c1 under a passing
    duplicate-baseline A/A control. Needs the L2 CUDA body compile (per-node cache) — one cold boot first.
12. **Scheduler via `--scheduler-cls`** — mount their overlay dir, `-e PYTHONPATH=/overlay/overlay`, write
    `cache/levers_policy_final.json` **on the head only**
    (`{"k_lo":3,"k_hi":7,"up":0.58,"down":0.42,"alpha":0.15,"enabled":1,"mode":"batch-uniform","signal":"pos",...}`),
    and pass the same `VLLM_ADAPTIVE_K_*` env. **Verdict: adopt as an experiment, and retire the R10
    `scheduler-e27c` attempt in favour of this route.** Rationale: our E27c port died on file replacement; this route
    is a supported extension point and is rank-invariant by construction (the decision is made in the engine-core
    process and reaches all ranks through the scheduler output). Honest caveats: `mode=batch-uniform` chooses k_hi only
    if every running request is high, else k_lo, so a mixed batch runs at k_lo — that is the point (uniform decode
    length replays one graph); their own history shows the extra `GDN`/router ideas *failed* their A/A gate, so do not
    bundle those. Keep `QUAD_E27C_SCHEDULER=off` and never mount two schedulers at once.
13. **Vocab-parallel greedy selection** (`GLM_TARGET_VOCAB_ARGMAX=1`, `GLM_LV_ARGMAX_MINTOK=1` and the replicated
    projection splitting `GLM_DS_SPLIT=fc:128,qkv_a:64,wq_b:16`). **Verdict: defer.** These carry their own accuracy
    caveats (`GLM_LV_SPLIT_INEXACT=fc`) and depend on more of their overlay (target argmax, ds hooks/split, levers).
    Take them only after 10-12 are green, and only with the KLD/qeval gate.

### Phase 3 — weight conversion (`lossless8`), a separate project

14. **Convert the BF16 non-expert tensors to 8-bit** (KDA → MXFP8, MLA/shared → block-128 FP8, sensitive → BF16;
    drafter decoder projections → block-128 FP8, requiring their drafter revision `bf582e4e…`, not our `7d74cdd8`).
    **Verdict: defer — decide explicitly.** It is the memory lever that makes 32 sequences comfortable, it is *lossy*
    (`lossless8` is a name, not a guarantee), it needs an offline CPU-only converter container (their 16 GB cap is a
    cap, not a sizing) with same-filesystem hardlinks, and their own note: "conversion of all weights was not rerun for
    publication". Taking it means a new weight tree per node, a re-qualification with qeval + KLD, and a storage
    decision (4 nodes) — that is a project, not a boot.

### Phase 4 — adopt the measurement standard (highest value, CPU-side)

15. **Adopt their harness and discipline, not just their settings.** `bench/final_sparkdash.py` (strict job validation:
    20 jobs, 18 scored, 256 tokens/255 decode/no reasoning chunks), `bench/run_prefill.py` + `prefill_checked.py`
    (prompt tok/s to first token), `bench/qeval.py` (75-task gate, require ≥72/75), `bench/kld_probe.py` +
    `compare_kld_strict.py` (teacher-forced, top-20 folded-tail), `bench/conc_bench.py`, `bench/hardset.py`, and their
    A/B rule: one-boot balanced A/B with a duplicate A/A control, ≥20-30 rounds, CI wholly above 0.2 ms, arithmetic
    checked on tensors (bit-exact or ulp). **Verdict: adopt.** Our needle+tool-call+warm-decode gate proves the service
    serves; it cannot compare recipes or prove a change helped, and AGENTS.md requires measured results to be kept
    apart from estimates. Adaptation needed: their collector expects a sparkDash dashboard endpoint with logical id
    `spark-01` and port 8093 — our endpoint is `192.168.1.6:8888` with alias `glm-5.3-flash`; the endpoint-agnostic
    parts (qeval, kld, prefill, conc_bench) can run as-is, and the prefill runner is pure stdlib.

---

## 5. Items examined and rejected (with reasons)

| item | verdict | reason |
|---|---|---|
| RoCEnante / b12x RoCE one-shot collectives (`GLM_ROCE_ALLREDUCE=1`, `--disable-custom-all-reduce`, `B12X_ROCE_HCA`) | reject for now | designed for their **switched two-rail** fabric with all-peer paths; our ring is switchless with two neighbours per rank; their install.md says "do not substitute the historical switchless-patched library", and their own Dockerfile refuses to shadow a base b12x |
| their NCCL 2.30.7 stock build (`NCCL_HOST_DIR`, commit `73cf1122…`) | reject | our switchless-nccl v0.0.1 is what the ring is qualified on, and it is the library RiNGSiDE's comparison arm measured on this hardware |
| loopback bind, port 8093, alias `GLM-5.3-Flash-FP8` | reject | fire's crons and agents use `192.168.1.6:8888` `glm-5.3-flash`; their loopback-only posture is their security choice, not a performance lever (we deliberately carry the LAN exposure) |
| `DEFAULT_EFFORT=high` | skip | a client default, not a serving change; keep our `reasoning_effort low` so reasoning cannot eat a cron budget |
| `--dtype bfloat16` | skip | weights are NVFP4/FP8; no measured basis for a change on our lane |
| `--memory 112g` as a *tuning* target | adopt as hygiene only | it is a cap, not a sizing; do not raise it to "use more memory" — the KV pin (item 2) is the controlled way to spend memory |
| their `batch-max` / `per-request` levers modes, `end_drain`, `coalesce_ms` | defer | `end_drain`/`coalesce` need `GLM_END_DRAIN=1`/`GLM_IDLE_COALESCE_MS` boot edits (jnardiello E29) — the same cadence territory our E29 wiring touched; take them only after the batch-uniform mode is proven |
| their `GDN` + router-dedup experiments | reject | they failed their owner's own A/A control gate (history.md, both CI intervals outside ±0.2 ms) and are not in the accepted profile |

---

## 6. Risk register

| risk | phase | mitigation / rollback |
|---|---|---|
| a mounted overlay file breaks engine init (symbol/path drift) | 1, 2 | one mount class per boot; sha-pin each file from their `runtime-source-manifest.json`; start with the kpool + parser set, which is the least invasive |
| prefix caching exposes an indexer/KDA state bug at long context | 1, 2 | their kpool bundle (item 5) is the companion fix, and their commit message ties the repair to the mount; run the 20k repeat-prefix check plus a long-context needle |
| container memory cap kills the engine at JIT peaks | 1 | 112 GB as they run it; if the engine dies at capture, raise the cap rather than dropping the swap/mem-free ritual |
| L2 prefetch / KDA stash changes numerics | 2 | quality gate required (`qeval ≥72/75`, KLD probe); treat ulp-level as a numerics change and record it; revert in one mount |
| scheduler experiment regresses decode | 2 | A/B in-boot with duplicate A/A control per their rule; keep `QUAD_E27C_SCHEDULER=off`; never two schedulers |
| weight conversion (Phase 3) changes quality silently | 3 | do not start without a reference KLD panel and a retained BF16-attention reference; their published KLD value came from a private panel — a new public panel is a new measurement |
| licence boundary on tonyd2wild-origin files | all | their NOTICE §6 states the tonyd2wild tree has no LICENSE/NOTICE and they preserve attribution rather than assign a licence. Record the origin (`patches/…`, mount path) and consult upstream before redistributing; do not relicense |

---

## 7. Governance

- All mounts get a sha256 pin. Source of truth: `docs/runtime-source-manifest.json` (per-file `accepted_sha256` /
  `published_sha256`) and `roce/b12x/PROVENANCE.json` (the vendored b12x subset). Pin them in `cluster.env` next to the
  existing `EXPECTED_LAUNCHER_SHA256`, `EXPECTED_TEMPLATE_SHA256`, `EXPECTED_NCCL_SHA256`.
- Vendor the overlay into fire (`quad-launcher/overlay/`) with their `NOTICE`, `CREDITS.md` and `LICENSES/*` alongside,
  so a later upstream force-push cannot change what we serve; upstream authorship and headers stay intact
  (AGENTS.md: preserve upstream authorship, licences and notices; never relicense a third-party file).
- Site data (addresses, hosts, paths, interfaces) stays in `cluster.env`/`site.env`; nothing committed.
- Keep the non-deterministic MoE accumulation visible: their profile is atomic-add MoE + b12x like the others; do not
  publish a bit-reproducible claim for our lane, and label ulp-level changes (KDA stash) as such.
- They are named by owner, repository and measured commit (`knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4` @ `dddb034`) as
  AGENTS.md requires for comparison recipes.
- No node access, no container start, no remote, no push without an explicit go-ahead.

---

## 8. How the three upstreams rank (for planning)

| upstream | adoptability onto fire-quad | payoff evidence | effort |
|---|---|---|---|
| **knapcio** (this report) | **highest** — overlay files written for our exact image digest and engine, `--scheduler-cls` route proven on it, sha manifest provided | prefix cache 8.1→0.63 s; boot 271→128 s; L2 prefetch +0.61 ms/step; 32 seqs, c16 312-476 tok/s aggregate; full quality gate | 1-2 boots for Phase 1, 2-3 more for Phase 2 |
| RiNGSiDE | low for file-level porting, high as a *whole stack* | +72 % prose, 2.2x cold prefill, 3.2x mixed-load, but a different fork (vLLM v0.29.0 backport) | 1-2 days: build + qualify their image |
| tonyd2wild (already measured as our proxy) | reference row only | 82.5 code / 35.8 prose / 24 GiB KV / 6 seqs on this ring | none |

Practical read: knapcio's levers are the ones to harvest **now** on the running lane; RiNGSiDE remains the
big-rebuild decision already recorded in `QUAD-RINGSIDE-PROPOSAL.md`.

---

## 9. Open questions for the owner

1. Phase 1 go/no-go (flags + prefix-cache mount + kpool/parser mounts + FI cache + slab loader, 1-2 boots)?
2. Vendor their overlay into fire (recommended, with notices + sha pins) or mount directly from the recipes clone?
3. Replace the aborted R10 `scheduler-e27c` with the `--scheduler-cls` experiment (Phase 2, item 12)?
4. Phase 3 (weight conversion): yes / no / later — it changes numerics and needs its own qualification.
5. Adopt their harness as the lane's measurement standard (Phase 4), including adopting `qeval`/KLD as our quality
   gate in addition to the serving gate?
6. Confirm we do **not** adopt RoCEnante (switch-fabric assumption) on the switchless ring.

---

## 10. Execution checklist (filled in only on go-ahead)

- [ ] Phase 1: edit `rank-launcher.sh` (items 1-9), vendor + sha-pin the mounted files, re-pin
      `EXPECTED_LAUNCHER_SHA256`, push to 4 ranks, sha-verify 4/4.
- [ ] Phase 1: `just mem-free-quad` → `just quad-up` → per-node `MemAvailable ≥ 16 GiB` → `just quad-verify` → gate.
- [ ] Phase 1: receipts — pool tokens, boot time vs 8-10 min baseline, repeated-20k-prefix TTFT before/after, gate.
- [ ] Phase 4 (parallel, CPU-side): install their harness, run one sparkDash matrix + prefill probe + qeval on the
      current lane to fix **our** baseline numbers before any Phase 2 change.
- [ ] Phase 2: KDA stash, L2 prefetch, scheduler — one per boot, each with an in-boot A/B + duplicate A/A control.
- [ ] Phase 3: owner decision recorded (yes/no/later) with the reference panel and storage plan if yes.
