# TensorFold on fire — TP=4 scoping plan (PLAN ONLY, nothing executed)

Written 2026-10-01, after the flint TP=2 TensorFold success (2026-09-29/30, recipe
`jayleaton/glm53-tensorfold-spark` @ `5624dfc`, image b9). Companion docs:
`TP4-MIGRATION.md` (the vLLM TP=4 plan — much of its transport/fabric analysis
carries over verbatim), `QUANT-EVAL.md`, `QUAD-ALEXELLIS-PROPOSAL.md`.

---

## 1. The blocker found first: TensorFold 0.3.4 is a TWO-RANK engine by construction

This changes the whole scope, so it is stated up front. Verified in the running
flint container (`glm53-tf-r0`, TensorFold 0.3.4 + 71 patches):

| evidence | file | line |
|---|---|---|
| `--tp` CLI arg is `choices=(1, 2)` | `tensorfold/cli.py` | 86 |
| `--rank` CLI arg is `choices=(0, 1)` | `tensorfold/cli.py` | 88 |
| `self.comm = comm if comm is not None else NCCL(rank, 2, master, port)` | `glm5_next/cuda/engine.py` | 212 |
| `world = 2` hardcoded in the weight loader | `glm5_next/cuda/weights.py` | 416 |
| split writer: `--rank`, `choices=(0, 1)`; `half = rows // 2` everywhere | `glm5_next/cuda/split.py` | 223-231, 88-114 |
| `swap()` raises `swap: two ranks only` (`peer = 1 - self.rank`) | `glm5_next/cuda/comm.py` | ~100 |
| fast prefill pair pipeline (pfpp) sends "the other rank's rows" | `glm5_next/cuda/pfpp.py` | 115 |
| RoCE comm pairs HCAs by IPv4 subnet between exactly two nodes | `glm5_next/cuda/roce.py` | 61-62 |

So **TP=4 TensorFold does not exist upstream** — not as an option, a flag, or a
roadmap item (`docs/MIA-AUDIT.md`: "#244 TP=4 switchless ring, #224 TP4 sparse
slice, TP3 work — Two nodes only"). Their own multi-node future is logged as an
audit item they declined for the 2-node kit.

### What TP=4 on fire CAN be, given that

Three realistic options, cheapest first:

**Option A — TensorFold on the fire DUO (TP=2), 2 nodes of the quad.**
Proven recipe, ~zero new engineering: fire has four CX7-linked nodes; any
pair forms a flint-identical lane (weights 166 GiB + image 24.4 GB +
prepared folders 77 GB/rank are the only staging). The remaining two nodes
keep the current vLLM LIL lane (or the alexellis TP4 lane) — two engines
side by side, one per pair. This is the "apply TensorFold to fire" move
that is actually available today.

**Option B — wait for/patch TP>2 upstream.** The blockers are mechanical,
not architectural: `world = 2` → world size param; `half = rows // 2` →
`rows // world`; `swap()` peer math → gather/scatter; pfpp's pair pipeline
→ generalised exchange. The all-gather itself (`ncclAllGather`, world-sized
recv) is already world-generic, and the RoCE runtime pairs by subnet — a 4-node
ring has 2 subnets per node and would need a real rewrite of the pairing rule.
Estimate for a competent fork: days, not hours, plus upstream rebase pain on
every pull — and it would be an unmeasured engine (upstream measured TP=2 only,
4 days of windows). Not recommended unless TP=4 is a hard requirement AND the
vLLM alexellis lane is judged insufficient.

**Option C — run TensorFold TP=2 on the two best-connected fire nodes as a
*third* lane** alongside the existing LIL (vLLM TP=4, NVFP4) and keep
flint's lane too. Same staging work as A, no lane disruption. The fire
quad's fabric (4-edge ring, MTU 9000, netplan `60-quad-ring.yaml` on all
four nodes, `verify-fabric.sh` green) is *already built* — any pair of nodes
has a direct 200 Gb edge.

**Recommendation: Option A/C (same work).** Pick the pair by the ring edges
and current disk: `spark-e116 (10.10.1.1 f0 / 10.10.0.1 f1, 715 G free)` +
`spark-3ff5 (10.10.2.3 f0 / 10.10.3.2 f1, 2.5 T free)` — these two share
**no cable** in the current ring (they connect via dc90), so either use the
edge they do have via a neighbour's relay (the ring is a ring; a direct
pair must be adjacent) — no: pick an **adjacent pair** on the ring instead:

- **e116 ↔ dc90** (edge A, 10.10.1.0/24): e116 has 715 G free, dc90 has
  **18 G free — cannot hold the 166 G checkpoint + 77 G prepared folder**.
- **dc90 ↔ 3ff5** (edge B): same dc90 problem.
- **3ff5 ↔ 2ca4** (edge C, 10.10.2.0/24): 3ff5 has 2.5 T, 2ca4 has 48 G
  (**too small for 166 G + 77 G = 243 G** unless pruned).
- **2ca4 ↔ e116** (edge D, 10.10.0.0/24): e116 715 G, 2ca4 48 G — same issue.

**Every fire node except e116 and 3ff5 is currently too full for the
TensorFold staging set.** So the pair choice follows the disk:

> **Pick: spark-3ff5 (rank 1, worker) + spark-e116 (rank 0, head)** — but they
> are not cable-adjacent in the quad ring. Two sub-options:
>
> 1. **Stage on e116 + 3ff5 anyway and free dc90/2ca4 first.** dc90 has
>    ~852 G used — its old deepseek-era trees (`/home/spark/models/*`) and
>    superseded HF caches are candidates to prune (flint's worker did the same
>    to reach 378 G free). After a prune, any adjacent pair works.
> 2. **Prefer edge C (3ff5 ↔ 2ca4)**: 2ca4 needs ~200 G freed to hold
>    checkpoint+prepared (48 G now). 2ca4 hosts old tonyd2wild caches and the
>    retired dflash2 drafter copies — prunable.
>
> This is a **user decision** (what may be pruned on dc90/2ca4), flagged in
> §5 as the single open question blocking execution.

---

## 2. What carries over from flint unchanged (the win)

The entire flint kit is reusable; only site values change:

| artifact | flint value | fire value |
|---|---|---|
| recipe checkout | `~/Workspaces/devops/recipes/glm53-tensorfold-spark` @ `5624dfc` | same tree, same image build |
| image | `glm53-tensorfold:b9` (24.4 GB, already on flint) | rebuild OR `docker save \| ssh docker load` from flint's head over LAN (flint 6efc → fire nodes is routable via 192.168.1.x) |
| weights | `neko-legends/GLM-5.3-Flash-Uncensored-EXL3` @ `07135ec0`, 166 GiB in HF-cache layout | stage once on the fire head (HF throttling lesson: download once, rsync the rest), then rsync the prepared folder to the worker over CX7 (NOT the raw checkpoint — the prepared folder is 77 GB/rank vs 166 GB source, and flint measured the rsync at ~25-35 MB/s disk-bound) |
| drafter | `incoai/GLM-5.3-Flash-DFlash2` @ `7d74cdd` (2.2 GiB, CC BY-NC-ND) | same, both nodes |
| config | `config/prod.env` | **new** `config/prod.env` for fire: different `WORKER_SSH`, `HEAD_IP` (the pair's CX7 edge subnet), `NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA`, `MASTER_PORT` (29551 is flint's; 29531 is fire-quad vLLM's — pick 29561), `IMAGE=glm53-tensorfold:b9` |
| ansible kit | `flint/glm53tf/{start,stop,check}-glm53tf.yml` | **copy to `fire/glm53tf/`**, re-target inventory groups, add a 2-node `fire-tf` inventory (or a `[fire_tf]` group in `fire/hosts`) |

Config knobs that stay as flint's W17 set (they are engine knobs, not site
knobs): `GLM53_TF_VISION=1`, `L2PF=1/8`, `MULTI_PREFILL=1`,
`BATCH_CAPTURE_AFTER=8`, `REASONING_FIELDS=reasoning`, `ADMIT_MEM=free` family
(0550 off), batch 4, sessions on, `KV_POOL_TOKENS=1048576`.

One knob needs a **fire-specific decision**: `GLM53_TF_COMM_BACKEND=roce`.
Flint uses RoCE all-gather on its single edge (both CX7 functions, GID 3,
proven `11.5 us eager at 16 KiB`). Fire's ring gives each node TWO edges on
TWO subnets — the RoCE runtime pairs "HCAs by IPv4 subnet" between the two
ranks, which works fine for a point-to-point pair sharing one subnet. Keep
`roce` (it measured +10.8% decode upstream) but pin the HCA list to the pair's
shared edge only, and keep `/cache/roce-failed` marker semantics (auto-fallback
to NCCL is the recipe's built-in safety net).

## 3. Fire-specific hazards (differences from flint)

1. **The quad ring is in active use.** `glm53_lil_tp4` (vLLM, NVFP4) is up on
   all four ranks right now serving `spark-e116:8888` (`glm-5.3-flash`,
   384k ctx). Any TensorFold lane on a fire node contends for:
   - the **GPU** (GB10 unified memory: 121.69 GiB total, page cache competes
     with CUDA — fire's own saga had 3 OOM boots from this);
   - the **CX7 ports** (each node has exactly 2 cage views; the ring uses
     both on every node — a TensorFold lane adds RoCE QPs on the pair's shared
     edge; NCCL/RoCE coexistence on the same HCA is untested here);
   - **:8888** (LIL lane owns it on e116 — the TensorFold lane must take
     another port, or replace it).
   The vLLM lane must be **stopped before the TensorFold containers start**
   (same one-lane-per-GPU rule as flint), or the TF lane takes a different
   port and a *different pair of nodes* than the ones vLLM ranks run on —
   but every fire node is a vLLM rank, so there is no free pair. **Conclusion:
   the TF lane on fire is a lane swap, not a lane addition** (unlike flint,
   where it replaced DSV41).
2. **Driver/GB10 parity**: flint's preflight warns driver 580.178.04 (head)
   vs 580.173.02 (worker). Fire's four nodes were driver-aligned for the
   quad build — verify with the same preflight check, do not assume.
3. **MTU**: fire's ring is MTU 9000 (alexellis-mandated); flint's link A is
   1500. The TF RoCE runtime does not care, but the staging rsync between
   fire nodes rides the 9000 path — faster than flint's ~25-35 MB/s
   disk-bound number suggests.
4. **`MAX_TOKENS`/context**: flint serves 1,048,576 ctx on 4 GiB-class
   session tiers; fire's GB10 nodes are the same 121.69 GiB unified-memory
   shape, so the same config values apply. No change.
5. **The `__main__` prepare bug**: still upstream (verified at `5624dfc`).
   The `scripts/prepare-main.sh` wrapper must ship with the recipe sync —
   the flint playbook's rsync includes it; keep that (do NOT exclude it the
   way the first flint sync did).

## 4. The playbook (concrete shape — mirrors `flint/glm53tf/` exactly)

```
fire/
  glm53tf/
    start-glm53tf.yml   # preflight -> gate -> stage -> image -> swap -> verify
    stop-glm53tf.yml    # stop + assert released
    check-glm53tf.yml   # run-any-time verification
```

Deltas from the flint playbooks (the only edits needed):

1. `hosts:` lines → a new `[fire_tf_head]` / `[fire_tf_worker]` pair in
   `fire/hosts` (e.g. `spark-e116` head + `spark-3ff5` worker, or per the
   §5 decision).
2. `recipe_dst` → `/home/spark/glm53-tf` (same path convention as fire-quad's
   `~/fire-quad`; distinct from `~/glm53-tp3`).
3. `log_dir` → `/home/spark/logs/glm53-tf`.
4. The `:8888 DSV41 gate` becomes a **`:8888 LIL-lane gate`** (refuse to start
   while `glm53_lil_tp4` holds the port) **or a port-swap assertion** depending
   on the §5 decision.
5. The foreign-GPU-container regex must whitelist BOTH `glm53-tf-*` and the
   lane that is allowed to be up (`glm53_lil_tp4` if co-tenancy were allowed —
   it is not; the assert becomes "GPU free after the stop playbook ran").
6. Fabric checks: check the pair's shared edge (e.g. edge D 10.10.0.x for
   e116↔2ca4, edge A 10.10.1.x for e116↔dc90) instead of flint's single link A
   — plus keep the RoCE GID-3 check on the specific HCAs of that edge.
7. Vision smoke, text smoke, canary, health, 1M-context asserts: byte-identical.
8. Image source: add an optional `docker pull`-style task that loads the image
   from flint's head (`spark-6efc`) via `docker save | ssh docker load` over
   the LAN, instead of building on fire (build works too; shipping is ~15 min
   at 24 GB over 1 GbE-class LAN — the CX7 links don't help node↔node except
   for the ring edges; measure and pick).

Justfile additions in `fire/`:

```just
# stage + swap + verify the TensorFold lane on the chosen pair
start-glm53tf:
	ansible-playbook -i hosts glm53tf/start-glm53tf.yml

# stop the TensorFold lane, assert released
stop-glm53tf:
	ansible-playbook -i hosts glm53tf/stop-glm53tf.yml

# run-any-time verification (hosts, fabric, RoCE, health, smokes)
check-glm53tf:
	ansible-playbook -i hosts glm53tf/check-glm53tf.yml

# tail the swap/boot log on the TF head
glm53tf-progress:
	ssh -i {{sshkey}} spark@{{tf_head}} "tail -f /home/spark/logs/glm53-tf/swap.log"
```

## 5. Execution plan (in fire's phase style)

**Phase 0 — decision (user):** which pair, and what gets pruned on the
non-e116 node to fit 166 G (checkpoint) + 77 G (prepared) + 24 G (image).
Also: does the TF lane *replace* the LIL lane's :8888 on e116, or take a new
port (and which)? Default recommendation: **edge C pair (3ff5 head, 2ca4
worker), TF API on a new port :8001 behind a new nginx :8889 LAN proxy,
LIL lane keeps :8888 untouched** — zero disruption to the live lane, at the
cost of pruning ~200 G on 2ca4.

**Phase 1 — stage (1-2 h):**
1. Prune 2ca4 (or chosen node): old tonyd2wild caches, superseded images
   (`ghcr.io/tonyd2wild/...`, `glm53-exl3:e3-v151`), retired dflash2 copies.
   Gate: ≥ 270 G free.
2. `rsync --ignore-existing` the checkpoint (166 G) e116→2ca4 over edge D
   (10.10.0.x, MTU 9000) — or `hf download` once on the head if the edge
   rsync is slower than the WAN was for flint (43 min for 180 G).
3. Copy `scripts/prepare-main.sh` WITH the recipe sync (the `__main__` bug fix).
4. Write `fire/glm53tf/` playbooks (§4) + `config/prod.env` with fire site
   values; pin `IMAGE=glm53-tensorfold:b9`, `MASTER_PORT=29561`.
5. Ship image: `ssh spark-6efc "docker save glm53-tensorfold:b9" | docker load`
   on both fire nodes (or build from the synced recipe — same 1-min cached
   build flint saw, then `docker save | ssh docker load` to the worker).

**Phase 2 — verify fabric pair (15 min):**
- directed pings on the pair's edge both directions at MTU 9000
- RoCE GID-3 present and IPv4-mapped on the edge's two ports
- `verify-fabric.sh` still green for the ring (TF staging must not disturb it)

**Phase 3 — prepare + launch (30-45 min):**
- `CONFIG=config/prod.env scripts/prepare.sh` (uses the wrapper; ~6 min/rank
  on GPU; **requires the LIL lane stopped on those two nodes** — GB10 unified
  memory will not hold both)
- `CONFIG=config/prod.env scripts/serve.sh start` (worker first, then head)
- watch `swap.log`: prepared-folder reuse (~11 s load vs 82 G fresh), NCCL
  init, RoCE all-gather connect, canary, warmups

**Phase 4 — verify (playbook `check-glm53tf`):** all 48 checks green, plus
`longctx_probe.py`-style needle at 200k+ and a vision probe through the LAN
proxy (if exposed).

**Phase 5 — receipts + docs:** devops.log entry, README section in `fire/`,
`fire/glm53tf/` committed.

**Rollback:** `just stop-glm53tf`; LIL lane untouched (it was never stopped if
the TF lane took its own port and the GPUs coexist — see Phase 3 note; if the
GB10 memory rule forced the LIL stop, `just lilquad-up` restores it).

## 6. Open questions (block execution, not planning)

1. **Pair + prune target** (dc90 at 18 G and 2ca4 at 48 G both need pruning to
   host a TF rank; e116+3ff5 are the only pair that needs no prune but is not
   cable-adjacent in the ring).
2. **Port/proxy plan**: replace LIL's :8888, or new :8001/:8889 alongside?
3. **Driver parity** on the chosen pair (fire nodes were aligned at the quad
   build; re-verify with the preflight task before boot).
4. Is the **2ca4↔3ff5 edge (edge C) free of the ring's own traffic** when the
   LIL lane is up? NCCL ring traffic between those two ranks would contend
   with TF's RoCE QPs on the same cable. (If the LIL lane keeps running, its
   edge-C all-reduces and TF's edge-C all-gathers share 200 Gb/s — probably
   fine, but it is an unmeasured co-tenancy.)
