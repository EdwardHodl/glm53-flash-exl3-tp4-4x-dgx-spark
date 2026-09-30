# Prune receipt — 2026-10-01

Motivation: stage the TensorFold lane (TF4-SCOPING.md). A TF rank needs ~265 G
(164 G checkpoint + 77 G prepared + 24 G image). Executed the tier-1 prune on
all four nodes, keeping every rollback path.

## What was pruned

| node | freed | targets |
|---|---|---|
| gx10-2ca4 | **+316 G** (48 → 364 G) | deepseek-0731 cache (156 G), nvidia NVFP4 hub-cache duplicate (157 G — flat `~/glm53-nvfp4` kept), tonyd2wild image (31 G), pip cache (4.6 G) |
| gx10-dc90 | **+321 G** (18 → 339 G) | DeepSeek-V4-Flash-Vision-Exp (157 G), TR3-4bpw pack (164 G — copies survive on e116 + 3ff5), 4 retired images (95 G) |
| spark-e116 | **+297 G** (715 → 1012 G) | 38 retired images (~299 G) + 98.8 G build cache |
| spark-3ff5 | 42 G (2.5 T free, no pressure) | e3-v151/v152 retired EXL3 images |

## Kept (load-bearing)

- `~/glm53-nvfp4` + `~/glm53-nvfp4-lil` on **all four nodes** (rollback + live weights)
- `~/glm53-dflash2-draft` (drafter, all nodes)
- `glm53-flash-cache-sm121` JIT volume (2.2 G, LIL boot time)
- images: `glm53-lil:v11-instanttensor` everywhere; `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` on e116/3ff5 (nvidia rollback image)
- the LIL lane itself: `glm53_lil_tp4` up 35 h, health 200 verified post-prune

## Rollback adjacency

e116 and 3ff5 both hold complete `nvfp4` + `nvfp4-lil` packs; either can
re-fanout a pruned node (`fanout-weights.sh` / `lil-fanout.sh`, ~1 h per
164-191 G pack over the ring). dc90 and 2ca4 now have room to re-stage too
(339/364 G free).

## Next

Stage the TF checkpoint (164 G, HF-cache layout, from flint head spark-6efc)
onto the chosen pair per TF4-SCOPING.md phase 1 — pair decision still open
(3ff5+2ca4 edge C recommended; both nodes now have room).
