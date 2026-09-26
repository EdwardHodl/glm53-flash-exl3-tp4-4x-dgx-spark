# E27c/E29 scheduler port — reassessment (2026-09-26)

## Outcome
Port aborted after 5 boots. Quad restored to pre-merge config: GATE GREEN 4/4
(needle 28,780tok, tool-call glm47, decode 70-76 t/s band), KV pool 1,576,246
tokens (6.01x @262k). `QUAD_E27C_SCHEDULER=off` in rank-launcher.sh; the ported
scheduler + shims stay vendored in quad-launcher/scheduler-e27c/.

## R10-vs-v11 symbol drifts found (one per boot, whack-a-mole)
| # | Symbol | v11 status | Resolution |
|---|--------|-----------|------------|
| 1 | vllm.v1.spec_decode.dynamic.acceptance_length module | absent | whole-module stub |
| 2 | vllm.v1.metrics.stats.RequestSpecDecodeMetrics | absent | inert class stub |
| 3 | ModelConfig.uses_xdrope (+ kwarg at 2 from_request sites) | absent (has uses_mrope) | getattr default + kwarg dropped |
| 4 | ObservabilityConfig.per_request_spec_decode_metrics | absent | getattr 'none' |
| 5 | VllmConfig.is_mm_encoder_only | absent (has is_encoder_only) | NOT YET FIXED |

The port is not a mount-and-go: jnardiello's scheduler is built against the
FujitsuPolycom SparkRing R10 image (vllm fork), ours is tonyd2wild sm121-v11
(vllm 0.1.dev20051+g487ecf187). Symbols 1-4 were shimmed; #5 still aborts.

## Options
A. Systematic symbol-map pass: AST-extract every config/attr access in their
   scheduler, diff against v11 classes in one automated sweep, then one boot.
   Est 1-2h agent work + 1 boot cycle. Success not guaranteed (semantic drift
   may follow: e.g. ScheduledEncoderInputStats, adaptive-k plumbing).
B. Drop the scheduler port: stock `--async-scheduling` only. NOTE: the NVFP4-2x
   lab measured async-scheduling WITHOUT a hold-near-limit scheduler as a
   prose-C2 decode REGRESSION (16.98 -> 15.19 t/s) — so B is likely a net loss.
C. Go to the R10 stack wholesale (image flip + FP8 weights + SIRCL + SparkCache)
   = re-qualifying the entire recipe; ~1.5d+, discards NVFP4 lane.
D. Stay as-is (current). Revisit only if Phase-1 TTFT gains are proven worth it.

## Phase 3 (SparkCache) impact
SparkCache rides on the same R10 ecosystem (kv-transfer-config + SIRCL sidecar +
connector modules + E21/E22b image features). Porting it without R10 image
support is now scoped materially harder than the earlier half-day estimate:
the connector alone is fine (Apache-2.0, mountable) but the runtime transport
(SIRCL) and the config namespace assumptions are R10-coupled. Recommend:
defer Phase 3 until either (A) succeeds or (C) is chosen.

## Sequence for any retry
1. Fix #5 + sweep (A). 2. Import-test in image. 3. Toggle on + bounce.
4. Gate + C1/C2/C4 TTFT ladder vs baseline (needle/tool/decode must hold).
Rollback = QUAD_E27C_SCHEDULER=off (env flip), one boot.
