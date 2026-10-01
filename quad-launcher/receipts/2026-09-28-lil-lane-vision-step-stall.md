# Receipt — LIL lane outage: vision step stall → head-rank self-kill

Date: 2026-09-28
Lane: `glm53_lil_tp4`, `http://192.168.1.6:8888`, GLM-5.3-Flash NVFP4 (QAD/instanttensor), TP=4
Outage window: **2026-09-28 12:15:23 → 12:20:23 PDT** (19:15–19:20 UTC; container clocks are UTC)
Reported as: "i was doing a vision request when it crashed"
Config in effect: `max_model_len 393216` (the 384K config from `2026-09-27-lil-lane-384k-context.md`)

## What happened, in order (head rank 0, spark-e116)

| UTC | Event |
|---|---|
| 19:15:23 | One multimodal request scheduled as a **combined encoder + prefill step**: `scheduled_encoder_inputs={req:[0]}`, `num_scheduled_tokens=1320`, `num_computed_tokens=0`, `mm_features` = 1 image, `image_grid_thw=[1,70,72]` (5040 patches → 1260 mm tokens), `modality='image'` |
| 19:15:23 | Worker JIT-compiles `_fused_q_kv_rmsnorm_kernel` + `rotary_kernel` ("during inference" latency-spike warnings — normal, also seen on healthy runs) |
| 19:16:24 | EngineCore: `No available shared memory broadcast block found in 60 seconds` — the first 60 s of *no worker response* |
| 19:17:24, 19:18:24, 19:19:24 | Same warning, repeated (4× total) |
| 19:20:23 | `TimeoutError: RPC call to sample_tokens timed out` → `EngineDeadError` → APIServer graceful shutdown → **container exit code 0** |
| 19:20:25 | dockerd logs only `TaskDelete` for that container — no daemon-level error |

The 300 s gap between 19:15:23 and 19:20:23 is exactly the EngineCore RPC/dequeue tolerance
(`VLLM_ENGINE_ITERATION_TIMEOUT_S` default 300 s). **The engine hit its own stall guard and
killed itself.** The request never returned an error to the client — the client would have seen
a hard connection close or a hang.

## Failure mode that made it an outage (not just one failed request)

`--restart no` and TP=4 mean only **rank 0 died**. All three workers (dc90/3ff5/2ca4) stayed
`Up 35 hours` holding NCCL/rendezvous state from a dead head → half-cluster, endpoint :8888
dead, no self-heal. Detected via `just lilquad-status`; cleared with `just lilquad-down`.

## Ruled out (with the evidence)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Host/container OOM | **No** | `journalctl -k` from 00:00 today on the head: zero `oom` / `killed process`; dmesg ring buffer verified readable (`dmesg_restrict=1` + `sudo -n` works, buffer spans Sep 26 → Sep 28) and has **no** entry in the 12:10–12:25 PDT window |
| GPU fault (Xid / NVRM) | **No** | No `xid`/`nvrm` lines in the window; last NVRM entries are from Sep 26 23:42 |
| Hung-task watchdog | **No** | No `blocked for more than` in kernel log |
| Fabric / RDMA break | Unlikely | Workers stayed up and healthy; no `ibv_reg_mr` / RDMA errors |
| The image size / grid | **No — disproved** | Fresh boot: 980×1008 (grid 1×70×72, 1285 prompt tokens ≈ the crash's 1320) served in **1.7 s**; 15/15 mixed-size vision requests (56×56 → 980×1008, concurrency 2) all OK, max 3.3 s |
| Vision path structurally broken | **No — disproved** | Provider-side preprocessing demonstrably completed (the dump contains real `pixel_values`); the stall was in the encoder/execution step, and the same step now completes |
| exit code 137 (SIGKILL) | **No** | `Exited (0)`; a graceful vLLM shutdown sequence is in the log |
| Storage full | No | `/` on the head: 712 GB free (81% used) |

## Unresolved — stated as unresolved

The **exact kernel/trigger that failed to return is not identified**. Two candidate mechanisms
survive the evidence, and the logs cannot separate them:

1. **Unified-memory allocator stall after long uptime** (low–moderate confidence). The lane had
   been up **35 h** with `--skip-mm-profiling` (KV pool sized without accounting for the mm
   encoder's activation peak). On GB10 unified memory under fragmentation
   (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), a large encoder allocation can *block*
   rather than return `NV_ERR_NO_MEMORY` — which matches a silent 300 s stall with no kernel
   line. A fresh boot has a clean arena, which is consistent with the repro passing.
2. **Rare device-side infinite loop / race in the chunked-prefill + encoder step** (low–moderate
   confidence). A corrupted index in a fused kernel can hang the device without raising, so the
   worker never writes a response — also matching the evidence.

What would distinguish them: per-step host/unified memory counters at the moment of the stall.
**None were captured.** That is the instrumentation gap, and it is the reason this is filed as
unresolved rather than fixed.

## A process error in this session, recorded

`just lilquad-down` (`docker rm -f`) was run **before** archiving the dead head container's full
log. The tail was captured first, but the boot-time log — which would have shown the KV-pool and
memory picture for the 35-h-old engine — is gone. **Archive `docker logs <name> > file` before
any `docker rm`** on this lane. This directly reduced what could be diagnosed.

## Recovery performed

1. `just lilquad-down` (all 4 ranks) — cleared the orphaned workers.
2. `LIL_MAX_MODEL_LEN=393216 just lilquad-up` — restored the 384K config the lane was serving
   before the crash. (A first boot was made on the plain default `262144` and was corrected;
   the promoted contract is 384K. The argument form `just lilquad-up LIL_MAX_MODEL_LEN=393216`
   is wrong — just reads it as a recipe name and fails; see the 384K receipt, now corrected.)
3. Verification on the restored lane: `/health` 200; instructed generation returns exactly
   `LIL QAD LIVE TP4`; **15/15** vision requests across 56×56 / 224×224 / 448×448 / 700×700 /
   980×1008 OK, max latency 3.3 s; `/v1/models` reports `max_model_len 393216`.

## Guard added

`quad-launcher/lilquad-watchdog.sh` — samples the lane and reacts to the **orphaned-worker**
state (head container exited while workers are still up), which is what turned one stalled
request into a full outage. Policy:

- head container exited → full clean restart (`just lilquad-down` + `LIL_MAX_MODEL_LEN=393216
  just lilquad-up`); the workers are listed explicitly when they were left orphaned
- head up **and** the log shows the stall signature (`No available shared memory broadcast
  block found in 60 seconds`) with **zero completed generations after it** → **warn only**
  (the engine is ~300 s from self-killing); `FORCE_ON_STALL=1` restarts proactively
- head up and healthy → silent, exit 0

The stall test carries a discriminator because the same line is emitted **benignly during boot**
(engine init / JIT / graph capture). The first version of this script flagged the healthy lane
off that boot-time line — a monitor that fires on nothing. It now requires no `POST
/v1/chat/completions ... 200 OK` after the last stall line. Verified on both shapes: the real
crash log (4 stall lines, 0 completions after → WARN) and a boot log (1 stall line, 2
completions after → benign), plus a live run against the healthy lane → `OK, exit 0`.

Not armed by default. The lane is NOT yet monitored by any recurring job.

## Follow-ups (not done here)

1. Capture per-step unified-memory counters on the head during a vision request, so the
   allocator-stall hypothesis can be confirmed or killed.
2. Decide whether `--skip-mm-profiling` should stay off for this lane (the KV pool would shrink
   slightly but the mm encoder peak would be accounted for).
3. Add an uptime ceiling / scheduled refresh if the allocator-stall hypothesis holds
   (a 35-h-old engine is currently the only difference between the crash and the repro).
4. Client side: the crash returned no error, so client retry/graceful-timeout handling in the
   caller should be checked.
