# TP4-OS-DRIVER-UPDATE.md — the quad's OS + driver update runbook (run before leaving, while inference is down)
# Written 2026-10-01. Companion: TP4-DAY-PLAN.md (the serving window).

## Current state (verified 2026-10-01, read-only)

| node | OS | kernel | driver | apt upgradable |
|---|---|---|---|---|
| spark-e116 | 24.04.4 | 6.17.0-1026-nvidia | 580.159.03 | 470 |
| gx10-dc90 | 24.04.4 | 6.17.0-1031-nvidia | 580.173.02 | 170 |
| spark-3ff5 | 24.04.5 | 7.0.0-1019-nvidia | 580.173.02 | 7 |
| gx10-2ca4 | 24.04.4 | 6.17.0-1026-nvidia | 580.159.03 | 469 |

Driver packaging: `nvidia-driver-580-open` (Ubuntu arm64 pkg). e116 + 2ca4 are two
releases behind (580.159.03); the target everywhere is the latest 580.178.04
(flint's head already runs it — driver-version mismatch was flagged as cosmetic
at flint's window; the fix is aligning the whole fleet).

3ff5 is already on 24.04.5 + a 7.0.0 kernel — it rebooted into a newer kernel
series at some point and only has 7 upgradable packages. It is the reference.

## Plan (order matters: update + reboot BEFORE the serving window)

Run after `just lilquad-down`, before the TF lane start. The playbook handles
update → reboot → wait → driver verify. Estimated 15-25 min per node, the three
workers in parallel, the head last (it holds the API).

```bash
cd ~/Workspaces/devops/fire
# 0. LIL down (frees the GPUs, removes the serving constraint)
just lilquad-down

# 1. update + upgrade + reboot, worker-first
ansible-playbook -i hosts glm53tf/update-reboot.yml

# 2. when all four are back and drivers verified (the playbook asserts this),
#    proceed to the serving window (TP4-DAY-PLAN.md phase 2+)
```

## Rollback

Driver: `sudo apt install nvidia-driver-580-open=580.159.03-0ubuntu0.24.04.1
nvidia-kernel-common-580=580.159.03-0ubuntu0.24.04.1` then reboot. Kernel: boot
the previous entry from grub. Nothing else changes; the containers, images and
weights are all on NVMe and survive reboots.

## Risks (from the flint window's driver-mismatch lesson)

- A driver version change requires a matching kernel module rebuild — the
  Ubuntu package does this via DKMS at install time; a reboot completes it.
  If a node comes back without a working GPU (`nvidia-smi` fails), stop, roll
  back the driver, and re-examine. Do not boot the TF lane on a half-updated node.
- The switchless NCCL is a userspace .so — driver-independent, survives updates.
- Docker restarts on boot: the LIL/TF containers have `-d` restart policy absent
  (the launchers create them explicitly), so a reboot leaves them stopped — good.
- The JIT cache volume (`glm53-flash-cache-sm121`, `glm53-tf-cache`) persists
  across reboots (named volumes) — boot time stays warm.
