# Receipt — lilquad 384K context (longer sessions)

Date: 2026-09-27
Lane: `glm53_lil_tp4` on `http://192.168.1.6:8888`, 4× DGX Spark GB10, switchless RoCE ring
Change: `--max-model-len 262144 → 393216` (384K), made overridable as `LIL_MAX_MODEL_LEN`
Requested by: user ("longer sessions")

## Why 393,216 and not more

The KV pin (`--kv-cache-memory 25769803776`, 24 GiB, fp8_e4m3) and `--max-num-seqs 8` are
both unchanged, so context is bounded by what 8 streams can hold:

| `max_model_len` | blocks/req (block_size 2304) | 8 streams | pool | verdict |
|---|---|---|---|---|
| 262,144 (previous) | 114 | 912 | 1,369 blocks | fits, 1/3 of pool unused |
| **393,216 (384K, applied)** | **171** | **1,368** | 1,369 blocks | **fits — maximum that keeps 8 streams** |
| 524,288 (512K) | 228 | 1,824 | 1,369 | exceeds → 6 streams |
| 1,048,576 (1M) | 456 | 3,648 | 1,369 | exceeds → 3 streams |

384K is the exact break-even: 8 × 171 = 1,368 of 1,369 blocks. Anything larger trades
streams for context.

## Measured at boot (from rank 0's log, load-independent facts)

| Metric | 262K boot | 384K boot |
|---|---|---|
| endpoint `max_model_len` (`/v1/models`) | 262,144 | **393,216** |
| weight load | 46.06 GiB in 66–67 s | 46.06 GiB in 66.3 s |
| **KV cache pool** | 3,154,184 tokens (12.03× @262K) | **3,459,187 tokens (8.80× @384K)** |
| `max_num_seqs` | 8 | 8 |
| `moe_backend` / quant | marlin / modelopt_mixed | unchanged |

**Flagged, not explained:** the reported pool *grew* by 305,003 tokens when `max_model_len`
grew, despite the fixed 24 GiB pin. Possible causes are block-count rounding and
activation/graph reservation differences at the larger shape; the cause is **not**
established here. The number is a measured log value, and the 8.80× concurrency figure is
that measurement, not an estimate from the older pool.

## Verification (load-bearing test)

`quad-launcher/longctx_probe.py` — needle at ~90% depth in a numbered filler body, asks for
two facts. **The first attempt undershot**: 1,080,294 chars reported only 204,902 tokens
(5.27 chars/token on this tokenizer), i.e. it never crossed the old 262,144 cap, so the test
would have been vacuous. The probe now self-calibrates from the server-reported token count
and reconfirms, then asserts `prompt_tokens > 262144`.

| Run | result |
|---|---|
| attempt 1 (undershoot) | 204,902 tokens → needle PASS, but not above the old cap — rejected as vacuous |
| **attempt 2 (calibrated)** | **267,570 prompt tokens (>262,144) → above-262K PASS, needle PASS** |
| prefill | 267,570 tokens in 65.9 s ≈ **4,062 tok/s end-to-end** |

Also on the 384K boot:
- `/health` HTTP 200 — PASS
- instructed generation returns exactly `LIL QAD LIVE TP4` — PASS
- Korean-Hangul corruption probe = **0 U+FFFD** — PASS
- all 4 ranks `glm53_lil_tp4` Up

## Caveats kept visible

- Single-run measurements; one prompt for the long-context test; novel filler body, so
  prefix-cache effects do not inflate it (attempt 1's 5.4 s was a prefix hit on the previous
  run's text, not a real prefill of 205K tokens).
- Throughput figures elsewhere in this lane remain subject to the contention caveat in
  `2026-09-27-lil-lane-chunky-test.md`; the 4,062 tok/s prefill number was measured during
  a window with background clients present, so treat it as an upper-ish figure, not a
  qualified rate.
- Changing `max_model_len` upward does not raise the KV pool by design; it only widens what
  a single request may hold.

## How to change it

```
just lilquad-up LIL_MAX_MODEL_LEN=393216          # applied
just lilquad-up LIL_MAX_MODEL_LEN=524288          # 512K, costs 2 streams
just lilquad-up LIL_MAX_MODEL_LEN=1048576         # 1M native, costs 5 streams
just lilquad-up                                   # default 262144
```
