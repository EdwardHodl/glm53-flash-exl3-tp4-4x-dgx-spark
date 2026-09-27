#!/usr/bin/env python3
"""Fair A/B bench: JIT warmup, then c1 (best of 2) and c4 aggregate.

Usage: python3 bench-warm.py <base_url> <label>
Warmup matters on this image: kernels JIT-compile lazily during inference
(jit_monitor mode=warn), so a cold lane measures compilation, not decode.
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.request

BASE = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else BASE

P = ("Write a detailed technical explanation of how tensor parallelism splits "
     "MoE expert weights across GPUs, in prose.")


def gen(prompt, max_tokens=256):
    body = json.dumps({
        "model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=900))
    dt = time.time() - t0
    return d["usage"]["completion_tokens"], dt, d["usage"]["completion_tokens"] / dt


print(f"[{LABEL}] warmup: 3 x 256-token generations (JIT + cudagraph capture)")
for _ in range(3):
    ct, dt, tps = gen(P, 256)
    print(f"  warmup run: {ct} toks in {dt:.1f}s = {tps:.1f} tok/s")

best = max(gen(P, 256)[2] for _ in range(2))
print(f"[{LABEL}] c1 decode (best of 2): {best:.1f} tok/s")

with cf.ThreadPoolExecutor(4) as ex:
    rs = list(ex.map(lambda _: gen(P, 256)[2], range(4)))
print(f"[{LABEL}] c4 aggregate: {sum(rs):.1f} tok/s (per-stream {[f'{x:.1f}' for x in rs]})")
