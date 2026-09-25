# GLM-5.3-Flash quantization evaluation — for the fire quad/duo reconfig

Written 2026-09-25. Companion to `TP4-MIGRATION.md`. Sources: HF model cards
(nvidia, RedHatAI, LibertAIDAI, brandonmusic, unsloth), the local recipe repos'
measured receipts (alexellis TP4, jnardiello FP8 TP4, tonyd2wild 2x/TP4,
MiaAI EXL3 2x/3x/4x, fire TP=3 live). GB10 node = 121.69 GiB unified memory.

---

## 1. The candidates that actually work on our stack

119 quants exist; most are non-starters on GB10 + vLLM multi-node. The field
reduces to four:

| quant | size on disk | dtype | runs on our launcher? |
|---|---|---|---|
| **EXL3 TR3-4bpw** (Mia-AiLab mirror of brandonmusic; fire's current weights) | **164 GiB** (staged, measured) | EXL3 uniform K4 routed experts (trellis-packed), custom runtime | ✅ fire TP=3 today; MiaAI `start-tp4.sh` TP=4; TP=2 |
| **nvidia/GLM-5.3-Flash-NVFP4** (2026-09-09) | ~181 GiB (169B params; attn/vision/embeddings stay BF16) | ModelOpt **W4A16** — experts + 3 dense MLP NVFP4, **all 45 attention blocks excluded** | ✅ vLLM `glm53-flash-arm64-cu130` / tonyd2wild images; alexellis TP4 recipe pattern |
| **RedHatAI/GLM-5.3-Flash-NVFP4** (2026-08-27) | ~178 GiB (~3 GiB/rank smaller) | compressed-tensors **W4A4** — MoE expert weights AND activations FP4, MTP head FP8 | ✅ same images; alexellis TP4 receipts used RedHatAI + LibertAIDAI |
| **FP8** (unsloth/zai-org) | ~310–330 GiB | FP8 experts + BF16 rest | ✅ jnardiello TP4 daily driver; **does not fit TP=2 or TP=3** |

Excluded, with reasons:
- **LibertAIDAI NVFP4** (181 GiB): quantizes attention → documented **intermittent
  corrupted token IDs** on this stack (vLLM #54150; tonyd2wild measured 4/9/8
  corrupted runs vs 0/0/0 for nvidia/RedHat). Needs a hand-written sparse-MLA
  kernel on sm_121. Avoid.
- **GGUF / MLX / AWQ-W4A16 / MXFP4**: llama.cpp has no `glm5_next` at all;
  AWQ/MXFP4 paths are single-node or CPU-offload oriented, no multi-node vLLM
  receipts on GB10.
- **EXL3 2.05/2.25bpw single-Spark**: different problem (one box, quality cliff).

---

## 2. Quality

| quant | evidence | verdict |
|---|---|---|
| EXL3 4bpw | brandonmusic KLD vs sealed BF16 teacher: **0.0246** (fp8-KV) / 0.0548 (nvfp4-KV), 2,047-position full mean; LAVD-low NVFP4-KV 3/10 gate fail (fp8-KV 8/10) | good but the *largest* deviation of the three; LAVD failure is KV-dtype-linked, fire runs fp8_ds_mla |
| nvidia NVFP4 | BF16→NVFP4 deltas: GPQA-D 0.9217→0.9211, SciCode 0.5621→**0.5769 (+1.5pt)**, MMMU-Pro 0.7688→0.763 (−0.6pt), AA-LCR 0.71→0.7106, IFBench 0.613→0.6054, TerminalBench 0.8258→**0.8315 (+0.6pt)** | **measured at BF16 parity**; attention held in high precision is the cleanest NVFP4 build for tool-calling |
| RedHat NVFP4 | GSM8K-Plat 97.74, MATH-500 94.87, AIME-25 86.67, GPQA-D 90.57 (no BF16 side-by-side on card) | strong; W4A4 activations give the fastest MoE GEMMs but no published BF16 delta |
| FP8 | NVIDIA/jnardiello treat as reference lane | nearest to BF16 by construction |

Tool-calling robustness matters most for us (crons + agents). The nvidia build's
0-corrupted-token record (#23) is the strongest signal available.

---

## 3. Speed (single-stream decode, measured receipts)

| config | code | prose | note |
|---|---|---|---|
| EXL3 TP=3 fire (live, k=7) | ~71 tok/s blended (120 tok/1.69 s) | ~71 | our own probe, 2026-09-21 |
| EXL3 TP=4 (MiaAI kit, k=3) | — | 31–33 | 1.45× TP=2 decode; cold prefill only +29 % (fabric-bound) |
| NVFP4 nvidia + DFlash2 k=7, TP=2 GB10 | 51.9 | 28.4 | tonyd2wild #23; spec-off ≈14 |
| **NVFP4 TP=4 switchless ring** | **75.2** | **29.8** | alexellis RigMark, 3 runs; structured 109.6; TTFT 0.35/0.27 s |
| **FP8 TP=4 switchless ring** | **62.95** | **34.13** | jnardiello E29 RigMark, 162/162; TTFT 0.386/0.376 s; C4 97.4 |

Read: NVFP4-TP4 wins code decode by ~1.23× over FP8-TP4; FP8 wins prose by
~1.15×. Both beat our current TP=3 on aggregate throughput at equal depth, and
TP=4 halves per-rank weights (41 GiB vs 54.7), which is where the KV gain comes
from. The brandonmusic 145–151 tok/s numbers are a different machine class
(2×96 GB RTX PRO 6000 @600 W + memory offsets) — not comparable to GB10.

---

## 4. Memory → context & concurrency (computed from measured anchors)

KV cost per logical token (measured, incl. DFlash2 k=7 drafter):
NVFP4-TP4 **7.98 KiB/tok/rank** · FP8-TP4 **12.63** · EXL3-TP3 fire **~27.2**
(fire's pin 20 GB → 786,634 tok pool).

Pool (logical tokens) and full-depth concurrent streams:

| config | KV/rank pin | pool | @131k | @262k | @320k | @524k | @1M |
|---|---|---|---|---|---|---|---|
| EXL3 TP=3 fire (today) | 20 GB | 787k | 6.0× | 3.0× | **2.46×** | 1.5× | 0.75× |
| EXL3 TP=4 (pin ~24 GB est.) | 24 GB | ~1.0M | 7.6× | 3.8× | 3.1× | 1.9× | 0.96× |
| NVFP4 TP=4 (alexellis pin) | 12 GB | 1.576M | 12.0× | **6.01×** | 4.9× | 3.0× | **1.5×** |
| NVFP4 TP=4 (pin 16 GiB) | 16 GB | 2.10M | 16× | 8.0× | 6.6× | 4.0× | 2.0× |
| FP8 TP=4 (jnardiello pin) | 16 GB | 1.365M | 10.4× | 5.21× | 4.3× | 2.6× | 1.3× |
| EXL3 TP=2 duo | — | ~0.4–0.5M est. | 3× | 1.7× | 1.4× | — | 0× |
| FP8 TP=2 | — | **does not fit** | | | | | |

Basis: per-rank free-for-KV ≈ util×121.69 − weights/rank − ~14 GiB overhead.
EXL3 TP=4 at util 0.75 leaves ~36 GiB/rank (weights 41 GiB) — a 24 GB pin is
conservative and still 2.5× fire's current pool. NVFP4 TP=4 at util 0.85 leaves
~44 GiB/rank (weights ~45 GiB). **EXL3 TP=2 duo is memory-starved**: 82 GiB
weights/rank leaves ~1–8 GiB — expect ≤2 concurrent mid-length sessions, that's
a cold-standby/cron lane, not a serving lane.

KDA caveat (applies to every quant): the 34 linear-attention layers' per-request
state cache competes with KV; concurrency and context trade off directly, and a
drafter multiplies the state pool. Don't chase the lastGiB without a boot
receipt in front of you (fire saga).

---

## 5. Quant ↔ netplan / NCCL / serving-runtime impact

**Quant choice does not touch netplan or the fabric.** The ring (4 edges,
MTU 9000, /24 point-to-point, DOCKER-USER relay rules) is quant-agnostic. What
quant changes is what rides *over* it:

| aspect | EXL3 (current) | NVFP4 (nvidia/RedHat) | FP8 |
|---|---|---|---|
| MoE kernel path | custom EXL3 trellis kernel (fused MoE, fat-expert tiers) | `marlin` / `flashinfer_cutlass` NVFP4 MoE (native FP4 tensor cores) | Triton FP8 MoE (GB10-tuned config) |
| All-reduce message volume | largest activations of the three (BF16 activations) | W4A4 (RedHat) moves the smallest activation volume; nvidia W4A16 same as BF16 | FP8 activations halve vs BF16 |
| NCCL config | unchanged | unchanged | unchanged |
| KV dtype | fp8_ds_mla (fire) — brandonmusic's NVFP4-KV variant is what failed LAVD | fp8_e4m3 explicit (never generic `fp8` — NoPE `pe_dim` trap) | fp8_e4m3 |
| Spec decoding | DFlash2 k=7 tuned | DFlash2 (drafter is dense, unaffected); **MTP broken on nvidia build**, works on RedHat | DFlash2 + SparkCache/SIRCL stack |
| Image/runtime | glm53-exl3:e3-v152 (our build, 19-patch overlay) | tonyd2wild `sm121-v11-dflash2` or vllm-openai glm53-flash arm64 | jnardiello R10 image + SparkCache/SIRCL |
| NCCL version gate | stock 2.29.7 OK (ring works today at TP=3) | **switchless-nccl 2.30.7 required for TP=4 ring** (tree-connect wedge) | same 2.30.7 requirement (SIRCL single-rail transport on top) |
| sm121 kernel fixes | EXL3 overlay handles it | top-k indexer kpool patch mandatory (else engine kill >24k ctx) | R10 image carries its own |

Net effect on the transport plan: identical netplan, identical switchless-NCCL
install for TP=4 regardless of quant. The only interaction is that NVFP4/FP8
recipes are *validated with* switchless-nccl 2.30.7 and the alexellis ring
pattern, while our EXL3 kit currently runs stock NCCL — one more reason to land
switchless NCCL in Phase 4 before or alongside any quant switch.

---

## 6. Recommendation

**Stay on EXL3 for the quad bring-up; switch to nvidia NVFP4 as the first
post-cutover experiment.**

1. **EXL3 now** — zero new variables during the TP=4 fabric/NCCL migration; the
   launcher, weights, image and every receipt carry over. Changing quant and
   topology in the same window would make every failure ambiguous.
2. **nvidia NVFP4 next** — BF16-parity quality with the cleanest tool-call
   record, 75.2 tok/s code decode measured on exactly our target topology
   (TP=4 switchless ring), and the best KV economics: at equal 12 GiB/rank pin
   it yields **2× fire's current pool** (1.58M tok, 6× @262k). It is the
   tonyd2wild DFlash2 default, so the drafter path is proven. Cost: new image
   family, SM121 kpool patch, ~181 GiB staging ×4 nodes, and re-baselining the
   serving stack (our 19-patch EXL3 overlay does not apply).
3. **FP8 only if** you later want jnardiello's SparkCache persistent prefix
   cache (58:1 prompt:completion agent traffic is exactly its use case) — but it
   needs 330 GiB staged per node, drops TP=2 entirely, and its decode receipts
   trail NVFP4 on code.
4. **Duo lane**: EXL3 is the only quant that fits TP=2 at all with usable KV.

Practical sequencing: land the quad on EXL3 (Phase 1–8 in TP4-MIGRATION.md),
gate it, then stage NVFP4 + tonyd2wild image on the same four nodes and A/B
under the same RigMark-style probes. Rollback is a container-name/env flip; the
fabric and NCCL work is shared.
