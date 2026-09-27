#!/usr/bin/env python3
"""Korean-Hangul corruption probe (tonyd2wild #23 protocol).

Reads one chat-completion JSON response on stdin and reports the count of
U+FFFD replacement characters — the corruption signature of ModelOpt
checkpoints with quantized attention. Zero on clean checkpoints (3 passes
measured 0/0/0 for nvidia and RedHat, 4/9/8 for the ModelOpt-attention class).
"""
import json
import sys

try:
    d = json.load(sys.stdin)
    t = d["choices"][0]["message"]["content"]
    print("korean-probe U+FFFD count:", t.count("\ufffd"), "| tail:", t[-40:])
except Exception as e:  # noqa: BLE001 — probe must never crash the gate
    print("probe parse error:", e)
