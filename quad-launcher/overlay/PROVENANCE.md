# quad-launcher/overlay — vendored overlay mounts for the fire quad lane

Each file here is staged to `/home/spark/fire-quad/overlay/` on all four ranks and
bind-mounted read-only into the container. Every file is sha256-pinned in
`quad-launcher/rank-launcher.sh` (and the launcher's own hash in `cluster.env`).

## kv_cache_coordinator.py

- Source: `knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4` @ `dddb034717cb84c362d8a370b81723fa4ee0c4d2`
  (upstream file `overlay/kv_cache_coordinator.py`).
- sha256 `d6410dfd4cb7174620787573d5695706965fea7b462961e7671c9d6c3fe5c9e7` — equal to that
  repository's `docs/runtime-source-manifest.json` `accepted_sha256` **and**
  `published_sha256` (change: none), i.e. the byte-identical file its deployment served.
- Purpose: tonyd2wild's DFlash2 prefix-cache repair (the upstream commit carrying it is
  `c3d81d2`, "mount tonyd2wild's kv_cache_coordinator repair (repeat 20k prompt
  8.1 s -> 0.63 s TTFT)"). Mounted over
  `vllm/v1/core/kv_cache_coordinator.py`; enabled with `QUAD_PREFIX_REPAIR=on`
  (default), which also asserts `--enable-prefix-caching --enable-chunked-prefill`.
- Provenance boundary: the file carries its own vLLM `SPDX-License-Identifier: Apache-2.0`
  and "Copyright contributors to the vLLM project" header; the repair itself originates
  from tonyd2wild's 2-Spark recipe, whose public tree carries **no LICENSE/NOTICE**.
  knapcio preserves that attribution rather than assigning a licence (their NOTICE §6).
  We vendor the file unmodified with this notice and do not relicense it.
- Rollback: boot with `QUAD_PREFIX_REPAIR=off` (no mount, stock prefix-cache behaviour).
