# Phase 1 — launcher-only batch: DEPLOYED, GATE GREEN

Date: 2026-09-26. Status: **DEPLOYED and serving-verified** (health 200, gate 4/4 on
spark-e116:8888). Plan: `QUAD-UPGRADE-PLAN.md` §Phase 1. Second Phase-1 boot after a
bisect (Attempt A) — see "Boot attempt 1" below.

Rollback policy applied this session (Edward, 2026-09-26): throughput numbers are
**receipts, not rollback gates**. A phase rolls forward unless it makes the service
unservable and unfixable. Everything here is recorded so the change can be reverted
if wanted; nothing was reverted for a benchmark delta.

## 1. What changed (applied)

| # | change | value |
|---|---|---|
| 1 | KV pool per rank | `--kv-cache-memory 12884901888` → **`25769803776`** (12 → 24 GiB) |
| 2 | NCCL/lifecycle env | `TORCH_USE_RTLD_GLOBAL=1`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `NCCL_IB_ROCE_VERSION_NUM=2`, `NCCL_IB_ADDR_RANGE=10.10.0.0/22` |
| 3 | mm hygiene | `--skip-mm-profiling`, `--mm-processor-cache-gb 4`, `--mm-processor-kwargs {"max_pixels":6422528,"max_image_tokens":4096}`, `--limit-mm-per-prompt {"image":16,"video":0}` |
| 4 | prefix set | **NOT adopted — defaulted OFF** (`QUAD_PREFIX_REPAIR=${QUAD_PREFIX_REPAIR:-off}`). Overlay vendored + sha-pinned for the Phase 3 retry |

Unchanged: context `262144`, `--max-num-seqs 6`, `--block-size 2304`, `--moe-backend marlin`,
k=7 static dflash, parsers glm47/glm45, `reasoning_effort low`, port 8888, `QUAD_E27C_SCHEDULER=off`.

Files: `quad-launcher/rank-launcher.sh` (sha256 `246e06e27a76336bd99e387f0ef1b7b51ea62d275fc44fa54bf8e52c69794c46`,
re-pinned in `cluster.env`), `quad-launcher/overlay/kv_cache_coordinator.py`
(sha256 `d6410dfd4cb7174620787573d5695706965fea7b462961e7671c9d6c3fe5c9e7`, staged but not mounted),
`quad-launcher/overlay/PROVENANCE.md`, `quad-launcher/probe-quad.py` (new probe harness).

Staged and sha-verified on all four ranks: launcher + overlay file identical on
spark-e116, gx10-dc90, spark-3ff5, gx10-2ca4. `cluster-up` preflight passed 4/4
(launcher hash, NCCL hash `78cb8387…`, template hash, empty GPU, ≥2 ACTIVE RDMA links).

## 2. Boot attempt 1 — the prefix set hangs engine init (finding)

With `QUAD_PREFIX_REPAIR=on` (i.e. `--enable-prefix-caching --enable-chunked-prefill` plus the
mounted coordinator), all four ranks **stalled after communicator init** and never reached
weight loading:

- 22:14:48 `Mamba cache mode is set to 'align' for Glm5NextForConditionalGeneration by default
  when prefix caching is enabled`
- 22:14:56 EngineCore prints the V1 engine config and the DP group line — then silence.
- 22:15:12 workers finish NCCL/PYNCCL init (`vLLM is using nccl==2.30.7`, `Using ['PYNCCL']
  all-reduce backends`) — then silence for >5 minutes.
- Diagnosis: GPUs 0 % util on all four ranks, worker main thread in state `R` (spinning),
  head disk IO ~88 KB/10 s (i.e. **no weight loading**), container RSS ~4 GiB.
- `cluster-up.sh` aborted on its own 120 s `COLLECTIVE_TIMEOUT` (exit 1); per recipe discipline
  the four containers keep running — they were torn down manually.
- Not an import error: in-container import test of the mounted file's dependencies
  (`KVCacheMetricsCollector`, `BlockPool`) and of `vllm.v1.core.kv_cache_coordinator` itself all
  succeed, and the mount is in place (39,786 bytes).

**Bisect (Attempt A)** — prefix set OFF, everything else (KV 24 GiB, env, mm) ON → shards load
normally (15-17 s/shard, 33 shards), KV cache built, graphs captured, API ready, gate green.
So the hang is caused by the prefix set on this image, not by KV/env/mm.

**Interpretation:** enabling prefix caching flips the mamba cache to `align` mode, and this
image's hybrid/tail path appears to need the companion fixes knapcio ships with the same
repair — their kpool fix pack (`GLM_KPOOL_FIX=1`: padded seed stride, spec-decode ring, hybrid
tail-slot mapping) and KDA accumulate/stash handling in align mode. Their `kv_cache_coordinator`
docstring says exactly that align-mode prefix caching copies intermediate spec slots. Phase 1
took the repair without its companions; the retry belongs with Phase 3e/3b, which is what the
plan's own note ("the kpool fixes are only safe together") anticipates.

Also note: **prefix caching is already effective by default on this image** — no flag needed for
the repeat-prefix win (see §4). The mount adds the DFlash2-specific repair, which is what needs
companions.

## 3. Serving verification (Attempt A)

| check | result |
|---|---|
| `/health` | HTTP 200 |
| `/v1/models` | `glm-5.3-flash`, `max_model_len 262144` |
| gate 1 long-context needle (~30K) | PASS — `INDIGO-OTTER-4417` @ 28,780 prompt tokens |
| gate 2 tool call (glm47) | PASS — `get_weather({"city": "Bristol"})` |
| gate 3 warm decode | 177 tokens in 3.7 s = **47.6 t/s**, gate reports PASS |
| gate summary | **4 passed, 0 failed — GATE GREEN** |
| KV pool (rank 0 log) | **3,154,184 tokens**, 12.03x @262k (was 1,576,246 = 6.01x) |
| boot shape | 33 shards at 15-17 s/shard (~8 min), graph capture 33 s / 1.90 GiB |
| per-node `MemAvailable` after startup | e116 35G, dc90 36G, 3ff5 37G, 2ca4 35G (floor 16 GiB: PASS) |
| containers | `glm53_tp4` Up on all four ranks |

Honest flags:
- the gate's warm-decode cell reads **47.6 t/s**, below the historical 70-76 t/s band. Our own
  before/after probe (§4) shows no regression from Phase 1 on the same cells (single-stream prose
  19.9/25.7 t/s after vs 21.2/24.9 before; 6-stream mean per-stream 18.03 vs 16.79). Prompt/acceptance
  variance and warm state are the likely explanation; recorded as an open observation for Phase 2,
  not a basis for rollback.
- `deep_ep` fails to import in this image (`AssertionError: Duplicate/Invalid NCCL runtime …`).
  Verified **pre-existing**, not caused by Phase 1: a throwaway container run of the same image
  fails to `import deep_ep` both with and without `TORCH_USE_RTLD_GLOBAL=1` (the wording differs —
  "Duplicate NCCL runtime" vs "Invalid NCCL versions" — but the import fails either way). vLLM logs
  it as a WARNING; the gate passes without it. `TORCH_USE_RTLD_GLOBAL=1` is kept (it makes the
  preloaded patched NCCL the globally visible one, matching the RiNGSiDE/knapcio env sets).

## 4. Probe receipts (informational, per the rollback policy)

`quad-launcher/probe-quad.py`, same cells before and after, one boot apart:

| cell | before (`probe-before-20260926.json`) | after Phase 1 (`probe-after-phase1a-20260926.json`) |
|---|---|---|
| decode prose run1 / run2, tok/s | 21.21 / 24.93 | 19.94 / 25.72 |
| decode prose TTFT | 0.537 / 0.435 s | 0.348 / 0.443 s |
| prefix 21,606 tok: cold / repeat TTFT | 2.195 / 2.193 s (ratio 1.00 — see note) | **18.165 / 2.196 s (ratio 8.27)** |
| prefill 17,320 tok | 10.335 s → 1,675.9 tok/s | 10.159 s → 1,704.7 tok/s |
| 6 concurrent 256-token streams | mean 16.79 / aggregate 80.18 tok/s | mean 18.03 / aggregate 69.97 tok/s |

Note on the prefix row: in the **before** run the 21,606-token prompt was already in the prefix
cache (an earlier probe run had sent the identical text), so "cold" and "repeat" both hit and
both read 2.19 s. After the reboot the cache was empty, which is what exposes the real numbers:
a genuine cold prefill of 21.6k tokens costs 18.165 s, and the repeat costs 2.196 s — **the v11
image caches prefixes by default** (8.27x on the repeat). This is the strongest single result of
the session for our cron/Hindsight traffic.

## 5. Rollback

- `QUAD_PREFIX_REPAIR` is already `off` (nothing to undo for the hang).
- KV pin / env / mm: revert `rank-launcher.sh` to `git checkout` HEAD~ of the Phase-1 commit,
  re-pin `EXPECTED_LAUNCHER_SHA256`, one boot.
- The overlay file stays vendored and pinned; it is not mounted while the toggle is off.

## 6. What Phase 1 did NOT do (now scheduled)

- Prefix caching + repair mount → retry with the Phase 3e kpool/indexer fix pack (and 3b KDA
  stash), since the hang is in the align-mode hybrid/tail path that pack repairs.
- RigMark protocol install (plan item 5) — the local `probe-quad.py` harness covered the
  before/after column for this boot; the external RigMark client remains for the Phase 2 baseline.
- Admission step (seqs 6 → 8) is Phase 3 item 8, untouched here.
