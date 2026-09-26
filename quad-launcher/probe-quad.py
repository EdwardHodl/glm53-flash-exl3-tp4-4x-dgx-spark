#!/usr/bin/env python3
"""fire-quad probe — stdlib-only decode / prefix-cache / prefill / concurrency probe.

Usage:
  python3 quad-launcher/probe-quad.py --base http://192.168.1.6:8888 --label before --out quad-launcher/receipts/probe-before.json
  ... --label after  --out quad-launcher/receipts/probe-after.json

Cells (all temperature 0, reasoning_effort low, chat_template_kwargs):
  decode    one stream, 256 output tokens on a prose prompt -> output tok/s and TTFT
  prefix    the same ~20k-token prompt twice -> TTFT cold and TTFT on the repeat (cache)
  prefill   a ~16k-token unique prompt, 1 output token -> prompt tok/s to first token
  conc      N concurrent 256-token streams -> aggregate and mean per-stream tok/s

This is a local stand-in for external benchmarks (sparkDash / RigMark); it exists so a
launcher change has a before/after column measured on our own endpoint.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.request

DEF_BASE = "http://192.168.1.6:8888"


def post_stream(base, model, prompt, max_tokens, extra=None):
    """Return (ttft_s, total_s, completion_tokens, prompt_tokens, text)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    if extra:
        body.update(extra)
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    text = []
    prompt_tokens = completion_tokens = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            usage = chunk.get("usage") or {}
            if usage.get("prompt_tokens") is not None:
                prompt_tokens = usage["prompt_tokens"]
                completion_tokens = usage.get("completion_tokens")
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content")
                reason = delta.get("reasoning_content")
                if piece is None and reason is None:
                    continue  # role-only chunk
                if ttft is None:
                    ttft = time.time() - t0
                text.append((piece or "") + (reason or ""))
    total = time.time() - t0
    return ttft, total, completion_tokens, prompt_tokens, "".join(text)[:160]


def notes(n_tokens, nonce):
    """Deterministic numbered-notes prompt of roughly n_tokens tokens."""
    lines = [f"# Session {nonce}: deterministic numbered notes."]
    while len("\n".join(lines)) < n_tokens * 4:
        i = len(lines)
        lines.append(f"{i:06d} note line alpha beta gamma delta epsilon zeta eta theta {i*7919%100003}")
    return "\n".join(lines)


def ttft_of(v):
    """Coerce an Optional ttft to a float (no content streamed -> use 0)."""
    return float(v) if v else 0.0


def decode(base, model, max_tokens):
    """Two runs of the same prose prompt: run 2 shows the warmed state."""
    prompt = ("Write a clear short paragraph explaining what a ring topology is "
              "in distributed inference, in plain prose.")
    out = {"cell": "decode", "streams": 1}
    for run in (1, 2):
        raw_ttft, total, ct, pt, _ = post_stream(base, model, prompt, max_tokens)
        ttft = ttft_of(raw_ttft)
        ct_i = int(ct or 0)
        dec = max(ct_i - 1, 1) / max(total - ttft, 1e-6)
        out[f"run{run}_ttft_s"] = round(ttft, 3)
        out[f"run{run}_completion_tokens"] = ct_i
        out[f"run{run}_decode_tokens_per_s"] = round(dec, 2)
        out[f"run{run}_total_s"] = round(total, 2)
        out["prompt_tokens"] = pt
    return out


def prefix(base, model, tokens):
    p = notes(tokens, "prefix")
    raw1, _, _, pt1, _ = post_stream(base, model, p, 1)
    raw2, _, _, _, _ = post_stream(base, model, p, 1)
    t1, t2 = ttft_of(raw1), ttft_of(raw2)
    return {
        "cell": "prefix",
        "prompt_tokens": pt1,
        "ttft_cold_s": round(t1, 3),
        "ttft_repeat_s": round(t2, 3),
        "speedup": round(t1 / max(t2, 1e-6), 2),
    }


def prefill(base, model, tokens):
    p = notes(tokens, f"prefill-{time.time()}")
    raw_ttft, _, _, pt, _ = post_stream(base, model, p, 1)
    ttft = ttft_of(raw_ttft)
    return {
        "cell": "prefill",
        "prompt_tokens": pt,
        "ttft_s": round(ttft, 3),
        "prompt_tokens_per_s": round((pt or 0) / max(ttft, 1e-6), 1),
    }


def conc(base, model, streams, max_tokens):
    prompt = ("Write a clear short paragraph explaining what a ring topology is "
              "in distributed inference, in plain prose.")
    t0 = time.time()

    def one(_):
        raw_ttft, total, ct, _, _ = post_stream(base, model, prompt, max_tokens)
        ttft = ttft_of(raw_ttft)
        return max(int(ct or max_tokens), 1) / max(total - ttft, 1e-6)

    with cf.ThreadPoolExecutor(max_workers=streams) as ex:
        rates = list(ex.map(one, range(streams)))
    wall = time.time() - t0
    return {
        "cell": f"conc{streams}",
        "streams": streams,
        "mean_stream_tokens_per_s": round(sum(rates) / len(rates), 2),
        "aggregate_tokens_per_s": round(streams * max_tokens / wall, 2),
        "wall_s": round(wall, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEF_BASE)
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--label", default="probe")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix-tokens", type=int, default=20000)
    ap.add_argument("--prefill-tokens", type=int, default=16000)
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--conc", type=int, default=8)
    args = ap.parse_args()

    cells = [
        decode(args.base, args.model, args.decode_tokens),
        prefix(args.base, args.model, args.prefix_tokens),
        prefill(args.base, args.model, args.prefill_tokens),
        conc(args.base, args.model, args.conc, args.decode_tokens),
    ]
    out = {"label": args.label, "base": args.base, "model": args.model, "cells": cells}
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"written {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
