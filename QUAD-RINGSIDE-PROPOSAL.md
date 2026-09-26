# Proposal — adopt the measurable parts of RiNGSiDE into the fire-quad TP4 lane

Date: 2026-09-26. Status: **PROPOSAL — nothing executed; no node accessed, no container touched.**
Author: Hermes (devops), for Edward.
Companion receipt (the working analysis): `quad-launcher/receipts/2026-09-26-ringside-vs-fire-quad-comparison.md`.

---

## 1. Source of the analysis (provenance)

Every number and claim below is traceable to one of these, read this session, read-only.

| source | identity | what was taken from it |
|---|---|---|
| `~/Workspaces/devops/recipes/glm53flash/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE` | clone of `github.com/othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE`, HEAD `6606f7638aff044a8170e414818477679934a1c3` (2026-09-26 22:40:59 +0700), `git fetch origin` → in sync with `origin/main`, working tree clean | the whole comparison target |
| `README.md` §"TP4: trying a 512K context limit" (L67), §"Status (2026-09-25)" (L120), §"The measured launch form" (L171), §"Optional: weightless GLP-44 steering" (L195) | upstream prose | 512K/1M experiment, per-profile change list, launch form |
| `launch/profiles/tp4/profile.json`, `launch/profiles/tp4/canonical-launch.json` (620-token `argv`, sha-checked by `sources/verify.py`), `launch/profiles/tp4/override.json` (`{"forced_k":7,"mode":"saturation"}`) | the measured TP4 launch | every flag, env var and mount of their TP4 plan |
| `docs/configuration.md` §"Server arguments" (L244), §"Fabric and collectives" (L9), §"Scheduler and speculation" (L53), §"Kernels" (L95/L115) | their configuration reference | per-knob values + which served file reads them |
| `bench/results/README.md` §"How it was measured" (L30), §"Caveats" (L76), §"TP4" (L108, table rows L114-L116), §"The RiNGSiDE TP4 profile at 8 to 16 streams" (L118), §"Arm manifests" (L268, rows L276+) | their bench table + arm manifests | the head-to-head numbers used in §3 |
| `docs/limitations.md`, `docker/README.md` (build-time estimate L47, qualify-a-self-built-image L117), `fabric/README.md`, `launch/README.md` | their caveats | non-portability, rebuild cost, fabric constraints |
| `patches/nccl/0001,0002`, `patches/vllm/tp4/*` (27 files), `src/tp4/*` (38 files), `build/*` | their served-change surface | what would have to be ported or rebuilt |
| `sources/verify.py`, `sources/import_plan.py`, `sources.lock.json`, `AGENTS.md` (repo root) | their governance | "a profile changes only through `import_plan.py` from a lab plan" |
| `~/Workspaces/devops/fire/quad-launcher/rank-launcher.sh` (sha `87143c23392e7377db190a89a6a005ad6f4def037fe11d6f26e5daf3056c5835`, pinned in `cluster.env`) | our live launcher | the exact flags we run |
| `~/Workspaces/devops/fire/quad-launcher/cluster.env` | our site config | pins: launcher sha, template sha `0c4099f3…`, NCCL sha `78cb8387…`, timeouts 120/2400 |
| `~/Workspaces/devops/fire/quad-launcher/PROGRESS.md`, `receipts/2026-09-26-e27c-port-reassessment.md`, `QUAD-ALEXELLIS-PROPOSAL.md`, `TP4-MIGRATION.md`, `netplan/quad/*.yaml` | our receipts | serving state, E27c abort, our ring subnets 10.10.0-3.0/24, gate numbers |
| `~/Workspaces/devops/fire/quad-launcher/receipts/baseline-metrics-20260926-0115.txt` | our `/metrics` scrape | spec-decode counters: 361,017 drafted / 127,043 accepted = 35.2 % acceptance |

**Why this comparison is legitimate.** RiNGSiDE's own bench table (§TP4, L115) measured
`tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark` @ `2ac4e8dd` (arm manifest `tp4-tonyd2wild-2ac4e8dd`,
`bench/results/evidence/comparison-arms.json`; the table row does not name its lab plan) — engine
**vLLM `0.1.dev20051+g487ecf187`**, which is the exact engine our image
`ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` runs — and it ran it **on this ring's NCCL transport**, i.e.
our transport. That row is the apples-to-apples proxy for fire-quad.

The checkpoint confound is also removed by an additional arm: RiNGSiDE's release plan run on the **NVIDIA checkpoint**
(nvidia `423acf37`, the same weight family as ours) is lab plan `31978478`, medians of 3 runs — 103.8 / 59.8 / 147.4
tok/s decode, cold TTFT 1.69 / 6.50 / 13.13 s, aggregates 79.8 / 197.2 — versus the tonyd2wild row's
82.5 / 35.8 / 127.4 and 4.52 / 14.16 / 28.39. On identical weights the gap is therefore the stack, not the checkpoint.
Neither row is our own measurement, so nothing below is claimed as our measured result.

---

## 2. What our recipe is today (the baseline being changed)

- Ring: spark-e116 (rank 0, head, API `:8888`) → gx10-dc90 (1) → spark-3ff5 (2) → gx10-2ca4 (3) → e116; MTU 9000;
  switchless-nccl v0.0.1 (`78cb8387…`), 2 HCAs/rank pinned per node.
- Image: `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (digest `4def0ef6…`), engine `0.1.dev20051+g487ecf187`.
- Weights: `nvidia/GLM-5.3-Flash-NVFP4` @ `09b04e5e` (flat dirs, 191 G/rank); drafter `incoai/GLM-5.3-Flash-DFlash2`
  @ `7d74cdd8`, k=7.
- Serve flags: `--gpu-memory-utilization 0.85 --max-model-len 262144 --max-num-seqs 6 --block-size 2304
  --moe-backend marlin --kv-cache-dtype fp8_e4m3 --kv-cache-memory 12884901888` (12 GiB), parsers glm47/glm45,
  `--limit-mm-per-prompt {"image":16}`, `reasoning_effort low`.
- State: HTTP 200, gate GREEN 4/4, KV pool **1,576,246 tokens**, `QUAD_E27C_SCHEDULER=off` after the aborted port.

---

## 3. Findings

### 3.1 Measured head-to-head on the same ring (RiNGSiDE's own bench table, L114-L116)

RiNGSiDE TP4 = medians of 2 runs in one boot (lab plan `89dfef84`); tonyd2wild TP4 = a single run of arm
`tp4-tonyd2wild-2ac4e8dd` (lab plan not named in the table). Checkpoints differ (RedHatAI `18d55bfd` vs nvidia
`423acf37`-derived); that confound is removed by RiNGSiDE's own NVIDIA-checkpoint arm (lab plan `31978478`, medians of
3 runs): 103.8 / 59.8 / 147.4 tok/s decode and cold TTFT 1.69 / 6.50 / 13.13 s on our weight family — i.e. same-weights
evidence that the gap is the stack, not the checkpoint. Both NVFP4, both dflash k=7.

| cell | RiNGSiDE TP4 | tonyd2wild TP4 (our engine) | delta |
|---|---:|---:|---:|
| decode code / prose / structured (tok/s) | 105.3 / 61.7 / 145.4 | 82.5 / 35.8 / 127.4 | +27.6 % / **+72.3 %** / +14.1 % |
| cold TTFT 8K / 32K / 64K (s) | 1.71 / 6.46 / 12.87 | 4.52 / 14.16 / 28.39 | 2.64x / 2.19x / 2.21x faster |
| short-code aggregate C1 / C6 (tok/s) | 77.0 / 191.9 | 58.3 / 92.4 | +32.1 % / **+107.7 %** |
| sparkDash code x1 (tok/s) | 151.3 | 120.8 | +25.2 % |
| staggered prefill-first newcomer L2/L4/L6 (s) | 2.42 / 2.47 / 2.55 | 7.93 / 8.05 / 8.16 | ~3.2x faster |
| staggered decode-first newcomer L2/L4/L6 (s) | 7.66 / 7.85 / 9.41 | 14.58 / 14.71 / 14.90 | 1.6-1.9x faster |
| admitted sequences | 16 | 6 | 2.7x |
| 8 / 12 / 16 streams, short-code aggregate (tok/s) | 232.6 / 283.2 / 345.8 | not run (cap 6) | — |

Our live gate band (70-76 t/s warm decode) sits inside the tonyd2wild band, so the lane is where the row says it is.
Our own acceptance counter: 35.2 % (drafted/accepted from the 01:15 scrape) — no published acceptance rate exists for
the RiNGSiDE TP4 row, so no delta is claimed there.

### 3.2 Context and KV

RiNGSiDE's 512K/1M experiment (lab plan `89dfef84`, derived plan `6b38d997`, README L67): one boot at
`--max-model-len 1048576` with **24 GiB KV/GPU** completed a 520,014-token cold prefill in 116.0 s (4,483 tok/s), a
1,030,014-token prefill in 358.3 s (2,875 tok/s), retrieved all three needles from a 1,000,083-token prompt, and
reported a **shared pool of 4,065,684 KV tokens** = 15.5x @262k / 7.75x @524k / 3.88x @1M. Their published 512K
setting is `--max-model-len 524288` + `--kv-cache-memory 25769803776` (24 GiB), and their concurrency at 512K is an
**estimate**: 4 full-length = 2,097,152 tokens, 6 = 3,145,728, 7 = 3,670,016, 8 = 4,194,304 (exceeds the pool).

Our 12 GiB pin yields 1,576,246 tokens. Their own TP2 profile runs a *smaller* pin than the TP4 one for a memory
margin (10.5 GiB), and `docs/operations.md` §"Memory" warns to move pool size in small steps — which is why this
proposal asks for 24 GiB, not their measured 32 GiB default.

### 3.3 Structure (why most of their gains cannot be flag-copied)

| axis | RiNGSiDE | us | consequence |
|---|---|---|---|
| engine | vLLM **v0.29.0 GLM53 backport** (`98dff2a8`, `local/glm53-v029:ported-20260910`) | tonyd2wild v11 (`0.1.dev20051+g487ecf187`) | their overlay/patch set is written against a third fork; the same class of symbol drift that aborted our E27c port (`receipts/2026-09-26-e27c-port-reassessment.md`, 5 drifts) |
| delivery | 54 patches + 55 full files mounted as an overlay, sha256-checked pre/post | baked into the image; our deltas are launcher/`cluster.env` | their changes are portable *as files*, but only onto their base |
| scheduler | `glm53_speedup/` fork: mixed-prefill cap 4608, split cadence, adaptive admission, first-token repay, adaptive fair chunk, KDA checkpoints, replay boundary | stock v11 scheduler, E27c off | this is where the mixed-load numbers come from |
| kernels | b12x MoE/dense/KDA, sparse-MLA plugin, prefill RDMA ring, lean all-reduce ≤2 MiB, topk rowspread, tuned FP8 GEMMs, humming dense | marlin MoE, stock attention/sampling | compiled artefacts (`_C…so`, `glm53_fp8_gemm.so`) are ABI-bound to their base image |
| transport | custom NCCL (ring-only + dual-PF four-GID routing) pinned to 10.100.224-227/24 | alexellis switchless v0.0.1, 10.10.0-3.0/24 | theirs *is* the transport their comparison row ran on; no gain to chase |
| determinism | declared non-deterministic (atomic MoE adds; ≥10-row decode launches) | marlin MoE + dflash | adopting their stack changes what we can claim |
| client surface | `glm53` tool/reasoning parsers, RedHatAI checkpoint, `top_p 0.95`, `clear_thinking` | glm47/glm45, nvidia checkpoint, `reasoning_effort low` | three client-visible changes if we take their stack |

---

## 4. Proposal — four phases

Time estimates are wall-clock for one boot of our existing ritual (`just mem-free-quad` → `just quad-up`, warm boot
8-10 min, gate ~5 min) plus, where noted, CPU-side work.

### Phase 1 — launcher-only batch (one boot; ~20-30 min). Recommend executing first.

Edits are to `quad-launcher/rank-launcher.sh` only; then re-pin `EXPECTED_LAUNCHER_SHA256` in `cluster.env`.

| # | change | from → to | source |
|---|---|---|---|
| 1 | KV pool | `--kv-cache-memory 12884901888` → `25769803776` | tonyd2wild TP4 arm ran 24.0 GiB on this ring (bench arm manifest L276); RiNGSiDE 512K setting (README L67) |
| 2 | prefix cache | add `--prefix-cache-retention-interval 2304` | `docs/configuration.md` §Server arguments (TP4 2304 = block size); RiNGSiDE treats retention as a weak-spot fix |
| 3 | mm budget | `--limit-mm-per-prompt '{"image":16}'` → `'{"image":4,"video":1}'`, add `--skip-mm-profiling` | their measured TP4 args; no client of ours sends images |
| 4 | NCCL/lifecycle env | add `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `TORCH_USE_RTLD_GLOBAL=1`, `NCCL_IB_ROCE_VERSION_NUM=2`, `NCCL_IB_ADDR_RANGE=10.10.0.0/22` | their env set; /22 covers our four cable /24s exactly (`netplan/quad/*.yaml`) |

Verification (all must pass before accepting):
1. per node after startup: `MemAvailable >= 16 GiB` (16,777,216 kB) — their memory-margin rule; stop if any rank is below;
2. `just quad-verify` PASS (8 GIDs idx 3, MTU 9000, jumbo pings);
3. gate GREEN 4/4 (`just quad-up` gate) + `just quad-test`;
4. record the reported KV pool from rank 0's log (expect ≈2x today's 1,576,246 tokens);
5. `curl /v1/models` shows `max_model_len 262144` unchanged.

Rollback: `git checkout quad-launcher/rank-launcher.sh`, re-pin the old sha in `cluster.env`, one boot.
Risk: (1) is the only OOM-relevant change; the pin move is 12 → 24 GiB, below their default 32 GiB, and the
mem-free + 48 G swap ritual stays in place. Flags (2)-(4) are unverified against this image — if `--help` rejects
one, drop that line alone (each is independent).

**Expected value:** ~2x KV headroom (512K becomes reachable, and 6-8 concurrent requests stop competing for a small
pool); prefix retention should cut TTFT on our repeat-prefix cron/Hindsight traffic. Confidence: high for (1) and
(4), moderate for (2) and (3) (flag presence in the image unverified).

### Phase 2 — adopt a measurement standard (CPU-side ~1 h, no node change). Do this before Phase 3.

Adopt RiNGSiDE's protocol instead of our 3-item gate for before/after claims: the `othexmr/rigmark` fork
(branch `feat/staggered-arrival`, commit `40fabcaf`, protocol 1.1.0) with cells — cold TTFT 8K/32K/64K, single-stream
decode on code/prose/structured, short-code aggregate at 1/6/8/16 streams, staggered prefill-first and decode-first
newcomer latency, sparkDash x1, medians of 2 runs. Our needle+tool-call+decode gate proves the service serves; it
cannot prove a change helped. This is also what AGENTS.md's "keep measured results separate from estimates" needs.

### Phase 3 — throughput and context experiments (each its own boot)

| # | experiment | change | verify | keep only if |
|---|---|---|---|---|
| 5 | loader | `--load-format instanttensor` + `INSTANTTENSOR_BACKEND=URING` | engine starts, gate GREEN, rank-0 shard-load time vs today's ~16 s/shard | gate passes and load time drops materially |
| 6 | 512K context | `--max-model-len 524288` (KV stays at 24 GiB) | concurrency ladder 1 → 2 → 4 full-length; 16 GiB `MemAvailable` floor; three needles at ~500K | all ladder steps answer, no rank below the margin; record actual pool |
| 7 | admission | `--max-num-seqs 8`; then 16 paired with `--compilation-config {"mode":3,"cudagraph_capture_sizes":[8,…,128],"compile_sizes":[8]}` | decode at 1/6/8/16 streams (Phase 2 cells); engine memory/graph logs | aggregate improves and 1-stream decode does not regress |

Notes: 16 sequences past the image's default captured sizes can be *slower* without graphs — that is why the
compilation-config rides with it, and why mode 3 must be confirmed in `--compilation-config` help first. Going past
a recipe's numbers is fine only with a receipt per boot; each experiment is one boot and one revertible line.

### Phase 4 — the decision gate, not a task (owner call; 1-2 days if taken)

The scheduler + prefill-RDMA + lean all-reduce stack is where §3.1's +72 % prose and 3.2x newcomer latency come from.
Two routes:

- **(a) Build and qualify their stack.** `docker/Dockerfile` on rank 0 (their estimate 1.5-3 h, README L47/§Quick
  start step 3); their measured base image `017fd0ba` and native artefacts are **not published**, so a self-built
  image is unqualified by construction (`docs/limitations.md`), the Dockerfile has never been built, and build inputs
  can disappear (fork commit + pinned base image digests). Cost 1-2 days with our own qualification against Phase 2.
- **(b) Wait** for the same scheduler released against the v11/tonyd2wild lineage.
- **(c) Do not port file-by-file onto v11.** Our E27c attempt already failed this way; their overlay targets
  vLLM v0.29.0 backport, and `docker/base/v029-port/v029-glm53-local.patch` is 23,421 lines of exactly that porting.

Client-visible changes if (a) is taken, each needing a gate re-run and a written notice: `glm53` parsers replace
glm47/glm45; checkpoint becomes RedHatAI `18d55bfd`; determinism becomes explicit atomic-add non-determinism in
≥10-row decode launches; `top_p 0.95` becomes the default.

---

## 5. Items examined and rejected (with reasons)

| item | verdict | reason |
|---|---|---|
| their NCCL build (patches 0001+0002) | reject | our library is what their comparison row ran on; theirs hard-pins 10.100.224-227/24, which our 10.10.0-3 ring does not match — adoption is a new build plus a re-address |
| `--gpu-memory-utilization 0.85 → 0.87` | skip | their fork's number; UMA margin is our binding constraint and Phase 1's KV pin is the controlled way to grow KV |
| `--override-generation-config {"top_p":0.95}` | skip | it compensates a missing value in *their* checkpoint's `generation_config.json`; ours is the NVIDIA checkpoint |
| `--default-chat-template-kwargs {"clear_thinking":true}` | skip | belongs to their template; ours keeps `reasoning_effort low` so reasoning cannot eat a cron budget |
| `--per-request-spec-decode-metrics detailed`, `--cudagraph-metrics`, `--kv-cache-metrics`, `--enable-per-request-metrics`, `--profiler-config` | skip | need fork symbols (`RequestSpecDecodeMetrics`, `per_request_spec_decode_metrics`) absent in v11 — the exact symbols that aborted our E27c port; we already scrape `/metrics` |
| `--kernel-config {"enable_flashinfer_autotune":false}`, `--attention-config`, b12x backends | skip for now | require their kernels; revisit only under Phase 4(a) |
| weightless GLP-44 steering (their TP4 option, default off) | skip | refusal suppression is not our use case; their own measurement shows 6-7 % slower prefill |
| runtime `fabric-setup.sh` instead of our netplan | keep ours | static `netplan/quad/*.yaml` + `verify-fabric.sh` already PASS; optional hardening only: add their `preflight_tp4.py` "address lies in its cable's subnet" check (~20 lines) |
| their startup warm-up (`GLM53_STARTUP_PREFILL_WARMUP=17`) | skip as a copy | it is a served-file feature of their overlay; on our stack the warm boot is the equivalent |

---

## 6. Risk register

| risk | phase | mitigation / rollback |
|---|---|---|
| 24 GiB KV starves host memory on a low-free node (our history: 3 kernel OOMs, dc90) | 1 | `just mem-free-quad` everywhere, 48 G swap persisted, one-shot cache drop, `MemAvailable >= 16 GiB` per node as a stop rule, revert the pin in one boot |
| a flag does not exist in the v11 image (`--prefix-cache-retention-interval`, `--skip-mm-profiling`, `--load-format instanttensor`, `mode 3`) | 1, 3 | check `--help` on rank 0 before the boot; these flags are independent, drop only the rejected one and note it |
| 512K with 6+ concurrent full-length requests exceeds the pool | 3 | ladder 1 → 2 → 4, pool re-read after the boot; 8 full-length needs 4,194,304 tokens and is out of scope |
| more captured graphs push the engine over UMA | 3 | pair `--max-num-seqs 16` with graph-memory logging; adopt 8 first and stop there if the margin shrinks |
| Phase 4(a) burns 1-2 days for an unqualified image | 4 | decide (a) vs (b) explicitly; (b) costs nothing and keeps the lane green |
| mixing an experiment into the serving lane | all | one phase per boot, receipt per boot under `quad-launcher/receipts/`, `QUAD_E27C_SCHEDULER=off` untouched |

---

## 7. Governance

- Site data (addresses, host names, paths, interfaces) stays in `cluster.env`/`site.env` and is never committed.
- Every boot gets a receipt; measured values and estimates are labelled as such (Phase 3's numbers are estimates
  until a Phase 2 window runs).
- The non-deterministic MoE accumulation stays visible for both lanes: ours (marlin + dflash) and theirs
  (atomic adds, ≥10-row decode). No "bit-identical" claim is made anywhere above.
- Anything that touches the **RiNGSiDE repository's own profile** changes only through `sources/import_plan.py` from a
  lab plan — their `sources/verify.py` fails if a flag, mount or served file changes without a new plan, and their
  512K recipe is explicitly a generated-file experiment that `launch/up.sh` overwrites. We do not promote that path;
  we take the values into *our* launcher.
- Comparison recipes are named by GitHub owner, repository and measured commit, as AGENTS.md requires
  (`tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark` @ `2ac4e8dd`; `othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE`
  @ `6606f76`).
- No node access, no container start, no remote, no push without an explicit go-ahead.

---

## 8. Open questions for the owner

1. Phase 1 go/no-go (launcher-only batch, one boot, ~20-30 min)?
2. Checkpoint policy: keep `nvidia/GLM-5.3-Flash-NVFP4` @ `09b04e5e` (our current) or move to RedHatAI `18d55bfd` —
   RiNGSiDE's measured row is RedHatAI, their comparison row (our engine) is nvidia.
3. Is 512K context wanted for the Hindsight/cron traffic (heavy) or is the KV pool growth the only goal (Phase 1
   alone delivers ~2x pool at 262K)?
4. Phase 4: build and qualify their stack, or wait for a v11-lineage release?
5. Do we adopt the RigMark protocol as the standard change-verification (Phase 2), or keep the 3-item gate and accept
   weaker before/after evidence?

## 9. Execution checklist (filled in only on go-ahead)

- [ ] Phase 1: edit `rank-launcher.sh` (items 1-4), re-pin `EXPECTED_LAUNCHER_SHA256`, push to 4 ranks, sha-verify.
- [ ] Phase 1: `just mem-free-quad` → `just quad-up` → per-node `MemAvailable` check → `just quad-verify` → gate.
- [ ] Phase 1: receipt with pool tokens, gate output, per-node memory; commit.
- [ ] Phase 2: clone `othexmr/rigmark` @ `40fabcaf`, run the TP4 cell set, store medians in `bench/`-equivalent.
- [ ] Phase 3: boots 5, 6, 7 in order, each with a receipt and a keep/revert decision.
- [ ] Phase 4: owner decision recorded, then either build-and-qualify plan or a dated deferral.
