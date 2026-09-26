# fire-quad — progress log (NVFP4 TP=4 lane)

Chronological record of the quad lane: what changed, what was verified, what is
open. Companion receipts live in `quad-launcher/receipts/`.

## 2026-09-25 → 09-26 — initial bring-up (GATE GREEN)
- nvidia/GLM-5.3-Flash-NVFP4 (190.4 GiB, snapshot `09b04e5e`) staged flat on all
  4 ranks; 2ca4 EXL3 cache pruned (121G → 285G free) to make room.
- alexellis `glm-5.3-flash-4x-dgx-spark-switchless` adopted as the baseline
  recipe; site deltas: port 8888 (not 8000), nvidia weights (recipe pins
  LibertAIDAI), pinned per-node `MGMT_IF`, persistent `/root/.cache` (JIT),
  48G swap per node, symlink-free flat dirs.
- Fabric: switchless RoCE ring e116(0) → dc90(1) → 3ff5(2) → 2ca4(3), MTU 9000,
  `verify-fabric.sh` PASS. Weights fanned out over QSFP (~400 MB/s/leg).
- 6 failed boots during the bring-up saga (GB10 UMA OOM: tilelang/cicc compile
  spikes + page-cache flood; dc90 diagnosed near-bare — root cause was the
  memory equation, not stray services). Fixes: 48G swap persisted in fstab,
  dc90 reboot, no cache-drop daemons during boot, 2400s startup timeout.
- Final: all 4 `glm53_tp4` containers up, KV pool **1,576,246 tokens**
  (6.01× @262144), API HTTP 200 on http://192.168.1.6:8888, GATE GREEN 4/4
  (needle `INDIGO-OTTER-4417` @28,780 tok, glm47 tool-call, decode 70–76 t/s).
- Commit `c6d7f86` — deployed receipt written into QUAD-ALEXELLIS-PROPOSAL.md.

## 2026-09-26 00:17 — quad-view console
- `quad-launcher/quad-view.sh` + `just quad-logs` (4-column tails, frozen status
  banner: API health + per-rank container/mem/load/uptime); `--once`, `--tail N`,
  single-rank follow. Verified live against the serving quad.
- Skill `tp4-quad-spark` created (devops). Commit `697bf09`.

## 2026-09-26 01:13–10:23 — E27c/E29 scheduler port attempt → ABORTED, rolled back
Phase 0 / 1 / 2 executed per plan:
- Vendored jnardiello `queued-cadence/scheduler.py` (sha `f44c2872…` verified
  against their SHA256SUMS) into `quad-launcher/scheduler-e27c/`, plus stock v11
  scheduler snapshot for reference, and per-symbol probes of the image.
- Wired (initially unconditional): scheduler bind-mount over
  `/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py`,
  `--async-scheduling`, `--prefill-schedule-interval 8`,
  `VLLM_E27B_SHORT_PREFILL_TOKENS=2048`, `VLLM_E27C_CADENCE_WHEN_QUEUED=1`.
  (`--max-num-seqs 6` was already present and is confirmed live.)
- Staging: launcher + scheduler pushed to all 4 ranks, sha-verified 4/4;
  `EXPECTED_LAUNCHER_SHA256` pinned in cluster.env. Commit `980343e`.

Result: **5 boots, 5 R10-vs-v11 symbol drifts**, one surfaced per boot:
1. `vllm.v1.spec_decode.dynamic.acceptance_length` — module absent → stubbed.
2. `vllm.v1.metrics.stats.RequestSpecDecodeMetrics` — absent → inert stub.
3. `ModelConfig.uses_xdrope` (+ kwarg at two `from_request` sites) — absent →
   `getattr` default, kwargs dropped.
4. `ObservabilityConfig.per_request_spec_decode_metrics` — absent → `getattr`
   `'none'`.
5. `VllmConfig.is_mm_encoder_only` — absent (v11 has `is_encoder_only`) →
   **not fixed**; engine init aborted.

Root cause: the E27c/E29 scheduler is written against the FujitsuPolycom
**SparkRing R10** image (a vLLM fork); ours is tonyd2wild **sm121-v11**
(`vllm 0.1.dev20051+g487ecf187`). This is a fork symbol-map problem, not a
single bug — semantic drift is likely queued behind the attribute drift.

Rollback (per user instruction to pause once serving worked):
- Scheduler mount + async-scheduling + E27 envs made conditional on
  `QUAD_E27C_SCHEDULER` (**default `off`**); ported scheduler + shims stay
  vendored for a future retry. Serving config = pre-merge baseline.
- Bounce (down → drop caches → up) completed 10:23; **HTTP 200 + GATE GREEN 4/4**
  on all 4 ranks, containers uniform.
- Justfile: `tp3-*` and `tp4-*` EXL3 lanes **removed** (quad is the only lane);
  `mem-free-quad` + full `quad-*` lane retained.
- Commits: `980343e` (wiring), `e82b06c` (rollback + reassessment receipt).

## 2026-09-26 15:00-16:00 — Phase 1 launcher-only batch DEPLOYED (GATE GREEN)
Executed `QUAD-UPGRADE-PLAN.md` §Phase 1 after committing the analyses (9f2902f):
- Applied: KV pin 12 → **24 GiB/rank** (pool **1,576,246 → 3,154,184 tokens**, 12.03x @262k);
  4 NCCL/lifecycle env vars (`TORCH_USE_RTLD_GLOBAL`, `TORCH_NCCL_ASYNC_ERROR_HANDLING`,
  `NCCL_IB_ROCE_VERSION_NUM=2`, `NCCL_IB_ADDR_RANGE=10.10.0.0/22`); mm hygiene
  (`--skip-mm-profiling`, 4 GiB mm processor cache, `image:16`/`video:0`, max_pixels/max_image_tokens).
- Prefix caching + knapcio `kv_cache_coordinator` mount: **NOT adopted** — boot attempt 1 hung all
  four ranks after communicator init, before weight loading (mamba cache flipped to 'align'); bisect
  (Attempt A) proved the prefix set is the cause. Toggle `QUAD_PREFIX_REPAIR` defaulted **off**;
  overlay vendored + sha-pinned for the Phase 3 retry with the kpool/indexer fix pack.
- Bonus finding: **v11 caches prefixes by default** — 21.6k-token repeat TTFT 2.196 s vs a genuine
  cold 18.165 s = 8.27x, with no flag.
- Verified: health 200, `/v1/models` 262144, gate 4/4 (needle `INDIGO-OTTER-4417` @28,780; glm47
  tool call; warm decode 47.6 t/s), containers Up 4/4, `MemAvailable` 35-37G per node.
- Receipt: `receipts/2026-09-26-phase1-launcher-batch.md`; probe harness `probe-quad.py` +
  `receipts/probe-{before,after-phase1a}-20260926.json` (throughput = receipt, not rollback gate).
- Open: gate warm decode 47.6 t/s is below the historical 70-76 band with no probe-side regression —
  check in Phase 2. `deep_ep` import failure is pre-existing (verified with a throwaway container).

## Current serving conditions (solidified; Phase 1 applied 2026-09-26)
- Endpoint: http://192.168.1.6:8888 — `glm-5.3-flash`, HTTP 200.
- Config: NVFP4 nvidia weights, TP=4, `--max-model-len 262144`,
  `--max-num-seqs 6`, fp8_e4m3 KV pinned **24 GiB/rank**, DFlash2 k=7,
  `QUAD_E27C_SCHEDULER=off`, `QUAD_PREFIX_REPAIR=off` (see Phase 1 note above),
  `--skip-mm-profiling` + 4 GiB mm processor cache, `image:16`/`video:0`.
- KV pool **3,154,184 tokens** = 12.03× @262k (was 1,576,246 = 6.01×).
- Gate: needle + tool-call + decode 4/4 PASS; `just quad-test` OK.
- Probes: `quad-launcher/probe-quad.py` (decode / prefix / prefill / concurrency) — receipts in
  `quad-launcher/receipts/probe-*.json`.
- Ops: `just quad-up|quad-down|quad-status|quad-logs|quad-log N|quad-test|`
  `quad-verify|mem-free-quad`.

## Open items (reassessment pending)
- **Option 1** — systematic symbol-map sweep of the R10 scheduler against v11,
  then one boot (est 1–2 h). **Option 2** — stock `--async-scheduling` only
  (likely net loss; 2x lab measured a prose-C2 decode regression). **Option 3**
  — adopt the R10 stack wholesale (image + FP8 + SIRCL + SparkCache; ~1.5 d,
  discards NVFP4 lane). **Option 4** — stay as-is (recommended for now).
- **Phase 3 (SparkCache prefix cache) repriced**: rides the same R10 ecosystem
  (kv-transfer-config + SIRCL sidecar + connector namespace). Not the earlier
  half-day; defer until option 1 lands or option 3 is chosen.
- Retire/annotate the now-unreferenced viewers `glm/scripts/tp3-tri-view.sh` and
  `glm/scripts/tp4-quad-view.sh` (their just recipes were removed).
