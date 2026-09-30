# TP4-DAY-PLAN.md — serving window runbook for the fire quad TensorFold TP=4 attempt
# Written 2026-10-01 for the "tomorrow when they finish" window. Everything staged today is
# non-interrupting; only phase 2 needs the LIL lane down.

## State at end of 2026-10-01 (all verified)

| item | state |
|---|---|
| checkpoint (164 G, 92 shards) | staged on e116 `/home/spark/models/tf-uncensored` — fetch from flint head was at 48% when last checked; VERIFY COMPLETE before phase 1 (`ls model-00092*`) |
| image `glm53-tensorfold:tp4` | built on flint head (6efc), fork code verified inside; chain script `~/logs/ship-b9-chain.sh` on 6efc ships it to e116 (after checkpoint), then e116 ships to 2ca4. dc90/3ff5 ship from e116 via the playbook's stage phase |
| drafter | on all four nodes already (`~/glm53-dflash2-draft`, 2.2 G) |
| fork patch | `patches/0900-glm-tp4-core.patch` @ recipe `e67d343`; full stack (0001→0900) applies clean; patched tree compiles |
| launcher | `scripts/serve-tp4.sh` (start/stop/status); config `config/prod-tp4.env` |
| playbooks | `fire/glm53tf/start-glm53tf-tp4.yml` (stage / prepare / launch tags), plus the TP=2 pair playbook `start-glm53tf.yml` for reference |
| RigMark | `recipes/glm53-tensorfold-spark/scripts/rigmark/` (upstream's turnkey runner) — `scripts/rigmark/install.sh` then `run.sh`; needs the endpoint URL |
| fire LIL lane | UP on all 4 ranks (glm53_lil_tp4) — must come down for phase 2 |

## Phase 0 — preflight (10 min, no interruption)

```bash
cd ~/Workspaces/devops/fire
# 1. checkpoint complete?
ssh -i ~/.ssh/spark_agent_key spark@spark-e116.local 'ls /home/spark/models/tf-uncensored/model-00092* && du -sh /home/spark/models/tf-uncensored'
# 2. image on e116? (chain script may still be running; if not: run the stage phase)
# 3. switchless NCCL present on all 4 (installed for the alexellis lane)?
for h in spark-e116.local gx10-dc90.local spark-3ff5.local gx10-2ca4.local; do
  ssh -i ~/.ssh/spark_agent_key spark@$h 'ls ~/nccl-switchless-v0.0.1/lib/libnccl.so.2.30.7' 2>/dev/null
done
# 4. fabric green (ring up, MTU 9000)?
just quad-verify
```

## Phase 1 — stage (30-60 min, non-interrupting; safe while LIL serves)

```bash
cd ~/Workspaces/devops/fire
ansible-playbook -i hosts glm53tf/start-glm53tf-tp4.yml --tags stage
#   - rsyncs the recipe + config to e116
#   - ships the tp4 image to dc90/3ff5/2ca4 (24 GB each over the ring — ~5 min per hop)
#   - launches checkpoint rsyncs e116 -> dc90/3ff5/2ca4 (parallel, 164 G each; ring edges
#     are 200 Gb/s so this is disk-bound, ~20-35 min per node)
# wait for the checkpoint syncs (poll until model-00092 appears on all three workers):
for w in gx10-dc90.local spark-3ff5.local gx10-2ca4.local; do
  while ! ssh -i ~/.ssh/spark_agent_key spark@$w 'test -f /home/spark/models/tf-uncensored/model-00092-of-00092.safetensors' 2>/dev/null; do
    sleep 30; echo "waiting for $w"
  done; echo "$w done"
done
```

## Phase 2 — prepare + launch (needs LIL DOWN; ~30 min total)

```bash
cd ~/Workspaces/devops/fire
just lilquad-down                     # frees all four GPUs
# prepared rank folders (parallel across nodes, ~6 min each; e116's runs first)
ansible-playbook -i hosts glm53tf/start-glm53tf-tp4.yml --tags prepare
# launch + smoke
ansible-playbook -i hosts glm53tf/start-glm53tf-tp4.yml --tags launch
```

Rollback (any failure): `CONFIG=config/prod-tp4.env scripts/serve-tp4.sh stop`
(from ~/glm53-tf on e116) then `just lilquad-up`. Nothing on the LIL lane
changes; its weights, image and JIT cache were all kept.

## Phase 3 — verify serving (15 min)

```bash
# health + context + smoke are asserted by the launch phase. Additional:
ssh spark@spark-e116.local 'curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool'
# per-rank containers:
for h in spark-e116.local gx10-dc90.local spark-3ff5.local gx10-2ca4.local; do
  echo "== $h"; ssh -i ~/.ssh/spark_agent_key spark@$h 'docker ps --filter name=glm53-tf --format "{{.Names}} {{.Status}}"'
done
# NCCL version line in the boot log (should say 2.30.7 — the switchless lib):
ssh spark@spark-e116.local 'docker logs glm53-tf-r0 2>&1 | grep -i nccl | head -3'
```

## Phase 4 — RigMark (1-2 h; run from e116 against 127.0.0.1:8000)

```bash
ssh spark@spark-e116.local
cd ~/glm53-tf/scripts/rigmark && bash install.sh   # one-time; fetches the rigmark runner
bash run.sh --base-url http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 --name tensorfold-tp4-fire
# receipts land in ~/glm53-tf/results/rigmark/<name>/
```

Compare against:
- upstream's TP=2 RigMark (results/rigmark/tensorfold-20260930-w17-final-r*): code 67.9 / prose 43.0 / structured 88.8
- alexellis's published vLLM TP2 and TP4 cards (results/rigmark/reference-alexellis/)

## Phase 5 — optimization levers (each one boot; keep only what measures well)

In `config/prod-tp4.env`, one knob at a time, gate between boots:
1. `GLM53_TF_L2PF_MB=8 → 16` (L2 prefetch window; +0.8-2.4% decode on TP=2)
2. `GLM53_TF_BATCH=4 → 8` (more slots; KV pool is 1M shared — watch admission)
3. `GLM53_TF_MAX_DRAFT_ROWS=16 → 8` (verify window at 4 ranks: smaller may win)
4. vision on (`GLM53_TF_VISION=1` + 64/64 MB caches) — needs the 0.3.4-base vision
   patch re-verified at world 4 first (tower is rank-0-local; likely fine, test)
5. prefill row split (0320) generalized — the biggest known prefill win, needs
   the pair-pipeline rewritten for N peers (a day of work; only if prefill is
   the binding constraint after RigMark)

## Failure modes to watch (from TF4-FORK-SCOPING §2)

- `checkpoint heads does not divide by world 4` — heads=64 divides, kv=8 divides,
  inter=2048/… divides; should not fire. If it does, the config.json dims differ
  from the fork's assumptions — stop and re-read.
- NCCL ring wedges at tree connect → the switchless env is not reaching the
  containers (check TF_NCCL_LIB is mounted + NCCL_PASSTHROUGH=1 in config).
- A rank exits with "different settings" — the config file wasn't sourced by
  that rank's launcher; check serve-tp4.sh sourced the same file on all nodes.
```
