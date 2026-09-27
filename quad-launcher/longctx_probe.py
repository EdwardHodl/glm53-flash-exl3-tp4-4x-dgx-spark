#!/usr/bin/env python3
"""Long-context acceptance test: prove the lane serves a prompt ABOVE the old 262,144 cap.

Builds a varied filler body with a needle at ~90% depth, then asks for the needle.
Pass = request accepted (no 400) AND needle returned, with server-reported
prompt_tokens actually above 262,144.

Note: repeated filler text tokenises ~6 chars/token on this tokenizer, which silently
undershoots the target (a first attempt aimed at 270K tokens landed at only 170K).
Numbering every line keeps the ratio near the normal ~4 chars/token.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://spark-e116.local:8888"
TARGET_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 270_000
CHARS_PER_TOKEN = 4

WORDS = ("maintenance archive entry inspection routine facility bay coolant pressure valve "
         "calibration log shift report note deviation nominal reading segment interval record").split()
UNIT = "Archive entry {i}: {w} {w2} no operational significance.\n"
NEEDLE = "CRITICAL RECORD: the vault code is PLUM-4417 and the shift supervisor is Dr. Mira Okonkwo. "

def build(target_tokens, chars_per_token):
    """Build the prompt aiming for target_tokens using the measured chars/token ratio."""
    target_chars = int(target_tokens * chars_per_token)
    lines, total, n = [], 0, 0
    while total < target_chars:
        line = UNIT.format(i=n, w=WORDS[n % len(WORDS)], w2=WORDS[(n * 7) % len(WORDS)])
        lines.append(line)
        total += len(line)
        n += 1
    cut = int(len(lines) * 0.9)  # needle at ~90% depth: not a lucky recency hit
    body = "".join(lines[:cut]) + NEEDLE + "\n" + "".join(lines[cut:])
    return ("The following is a long archive dump; answer only from it.\n\n" + body +
            "\n\nQuestion: What is the vault code and who is the shift supervisor? "
            "Answer with just those two values.")


def send(prompt):
    req_body = json.dumps({
        "model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", req_body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=1800))
    except urllib.error.HTTPError as e:
        print(f"FAIL: HTTP {e.code} — {e.read()[:400].decode(errors='replace')}")
        sys.exit(2)
    return d, time.time() - t0


# Relative filler tokenises ~5.27 chars/token on this tokenizer, so a fixed 4.0 target
# undershoots. Calibrate: send, read server-reported prompt_tokens, rescale, resend.
ratio = CHARS_PER_TOKEN
prompt = build(TARGET_TOKENS, ratio)
d, dt = send(prompt)
u = d["usage"]
print(f"attempt 1: {len(prompt):,} chars -> prompt_tokens={u['prompt_tokens']:,} "
      f"({len(prompt) / u['prompt_tokens']:.2f} chars/token) in {dt:.1f}s")

if u["prompt_tokens"] <= 262144:
    ratio = len(prompt) / u["prompt_tokens"]
    prompt = build(TARGET_TOKENS, ratio)
    d, dt = send(prompt)
    u = d["usage"]
    print(f"attempt 2: {len(prompt):,} chars (ratio {ratio:.2f}) -> "
          f"prompt_tokens={u['prompt_tokens']:,} in {dt:.1f}s")

txt = d["choices"][0]["message"]["content"].strip()
print(f"completion={u['completion_tokens']} | end-to-end {u['prompt_tokens'] / dt:.0f} tok/s")
above_cap = u["prompt_tokens"] > 262144
found = ("PLUM-4417" in txt) and ("Okonkwo" in txt)
print(f"above-262K: {'PASS' if above_cap else 'FAIL (did not exceed the old cap)'} | "
      f"needle: {'PASS' if found else 'FAIL'} | reply: {txt[:160]}")
sys.exit(0 if (above_cap and found) else 1)
