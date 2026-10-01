# TP4 Hill-Climb Ladder — Benchmark Receipts

Purpose: one row per benched configuration. Each row records the config knobs actually
set at bench time, the RigMark receipt path, and the headline numbers. The ladder is
6 → 8 → 12 → 16 slots at 384k/512k context; the goal is 16 @ 512k.

Per-stream KV cost at 7.44 KB/tok/rank (latent fp8): 384k = 2.83 GiB, 512k = 3.77 GiB.
Reserve budget per rank: 25 GiB (OS 8 + BATCH_RESERVE 11 + session 6).

### Rung 2 — 512k × 12 slots (2026-10-01 17:35 UTC)
- **Config**: CONTEXT=524288 · GLM53_TF_BATCH=12 · KV_POOL=6,291,456 (44.63 GiB/rank) ·
  BATCH_RESERVE_GB=6 · BATCH_ADMIT_GB=2 · HOST=127.0.0.1
- **Receipt**: `~/glm53-tf/results/rigmark/tensorfold-20261001-173524/glm53-flash-exl3-tensorfold-tp4-bench.json`
- **Results**: conc 1: 65.3 stream / 0.392s · conc 2: 95.1 agg / 54.5 stream ·
  conc 4: 113.2 agg / 30.7 stream / TTFT 1.216s
- **vs rung 1 (8×512k)**: conc 1 stream DOWN 83→65 (-22%) — at 12 slots the decode batch
  splits rows thinner even at low concurrency; aggregate at c2/c4 unchanged (95/113).
  Verdict: 12 slots buys nothing at ≤4 concurrent streams and costs single-stream speed.
  8×512k is the better serving config unless real traffic exceeds 8 concurrent.
- **Boot notes**: two failed boots before green (RESERVE 14 then 11 both starved batch
  sizing — the sizing runs after pool pre-alloc from post-pool MemFree; RESERVE=6 gives
  (11.5-6-2)/0.16 ≈ 21 → capped at 12). Playbook context assert widened to accept 524288.

---

## Receipt log (newest at top)

### Rung 1 — 512k × 8 slots (2026-10-01 16:54 UTC)
- **Config**: CONTEXT=524288 · GLM53_TF_BATCH=8 · KV_POOL=4,194,304 (29.75 GiB/rank) ·
  BATCH_RESERVE_GB=13 · HOST=127.0.0.1 (proxy = only publish point)
- **Receipt**: `~/glm53-tf/results/rigmark/tensorfold-20261001-165437/glm53-flash-exl3-tensorfold-tp4-bench.json`
  (label `...-tp4-bench`, comparison `2026-10-fire-tp4-hillclimb`)
- **Results**: conc 1: 83.2 stream / 0.303s · conc 2: 95.7 agg / 55.9 stream ·
  conc 4: 114.0 agg / 35.0 stream / TTFT 1.216s
- **Card rows** (rigmark c5a0db01, 5/5 gates clean): CODE 96.0 tok/s (92.9–97.8) ·
  PROSE 57.7 (56.4–60.5) · STRUCTURED 131.6 (127.3–131.9) ·
  64K PREFILL cold 1,544 tok/s / warm replay 297,646 tok/s
- **vs 384k×6 (run C)**: identical within noise at conc ≤4 — the doubled per-stream KV
  budget is free at bench loads. Next: rung 2 (12×512k) to see where admission knobs bite.
- **Boot notes**: dc90 disk full again (sessions grew 4.4→23G under 384k live traffic —
  spilled KV pages); purged with sudo, 5.2G free. Session-spill growth is now the
  watch-item for higher rungs. Boot GREEN first try.

### Rung 0 — baseline re-bench (2026-10-01, pre-ladder)
- **Config**: CONTEXT=393216 (384k) · GLM53_TF_BATCH=6 · KV_POOL=2,359,296 (16.74 GiB/rank)
  BATCH_RESERVE_GB=11 · BATCH_PIECE=2048 · MAX_DRAFT_ROWS=16 · IMAGE=glm53-tensorfold:tp4
- **Bind**: 127.0.0.1:8888 (proxy :8889 torn down for the window)
- **Receipt**: _see run C (below) — this bench re-validates the same config after the
  hill-climb tooling landed; skip unless numbers diverge >5%._
- **Status**: superseded by receipt tensorfold-20261001-153725; not re-run (config unchanged).

### Run C — 384k × 6 slots (2026-10-01 15:37 UTC)
- **Config**: CONTEXT=393216 · BATCH=6 · KV_POOL=2,359,296 (16.74 GiB/rank) · commit 477f331
- **Receipt**: `~/glm53-tf/results/rigmark/tensorfold-20261001-153725/glm53-flash-exl3-tensorfold-tp4-384k-6slot.json`
- **Results**: conc 1: 83.3 stream / TTFT 0.315s · conc 2: 94.8 agg / 51.7 stream ·
  conc 4: 110.3 agg / 31.8 stream / TTFT 1.225s
- **Prefill (now passing at 384k)**: cold 1568 tok/s @ 33k, 1545 tok/s @ 65k; warm 173k/294k (cached)

### Run B — 36k × 4 slots (2026-10-01 14:52 UTC)
- **Config**: CONTEXT=36864 · BATCH=4 · KV_POOL=36864 · commit a46dc04 era
- **Receipt**: `~/glm53-tf/results/rigmark/tensorfold-20261001-145922/`
- **Results**: conc 1: 84.7 stream / 0.302s · conc 2: 99.9 agg / 56.4 stream ·
  conc 4: 113.5 agg / 32.6 stream / TTFT 1.042s
- **Note**: SKIP_PREFILL=1 (65k stage > 36k context → HTTP 400 in run A at 14:51)

---

## Ladder plan (each rung: bounce → bench → receipt row here)

| Rung | CONTEXT | BATCH | KV_POOL | KV GiB/rank | Admission knob change | Status |
|---|---|---|---|---|---|---|
| 8×512k | 524288 | 8 | 4,194,304 | 31.8 | BATCH_RESERVE_GB 11→13 | pending |
| 12×512k | 524288 | 12 | 6,291,456 | 47.7 | BATCH_ADMIT_GB 2→3, RESERVE→14 | pending |
| 16×512k | 524288 | 16 | 8,388,608 | 63.6 | RESERVE→16, watch MAX_DRAFT_ROWS=16 ceiling | pending |

Rollback at any rung: previous config is one git revert + bounce away.
