#!/usr/bin/env python3
"""[glm53-noopener] Reasoning parsers: opener-less generation is content.

LOCAL fix, found live 2026-09-01 while verifying the W16/W18/W25 merge.
BaseThinkingReasoningParser.extract_reasoning (basic_parsers.py) treats a
generation with NEITHER think token as reasoning_content ("For models that
may not generate start token, assume the reasoning content is always at the
start"). With our chat template (thinking off -> generation prompt ends at
the bare assistant token), every thinking-off reply has no think tokens, so
the chat endpoint returned content=null with the answer silently filed as
(reasoning|"") — empty reasoning, no content. The streaming path in the same
file already returns content for opener-less deltas; only the non-streaming
extract was inconsistent.

Semantics after this patch (deepseek_r1 serve parser):
  opener in generation, never closed      -> all reasoning (truncated think)
  close-only (prompt-side opener)         -> reasoning="" + content=rest (unchanged)
  full pair                               -> split (unchanged)
  NEITHER token (thinking off)            -> content=model_output  (THE FIX)
  opener in PROMPT only + close in gen    -> else-branch split (unchanged)

Known accepted ambiguity: a thinking-on generation that truncates before the
close token (opener in prompt, no close) is textually identical to a
thinking-off answer and now surfaces as content. Mainline vLLM accepts the
same ambiguity.

Patches the shared base — deepseek_r1, deepseek_v4, ernie45, holo2, kimi_k2
and kin inherit it; glm45/47 (engine adapters) are not on this serve path
anymore. Unconditional (correctness fix, not a tuning knob). Idempotent via
the [glm53-noopener] marker; ast.parse-validated after edit; fails closed on
anchor drift.
"""

import ast
import os
import sys

MARKER = "[glm53-noopener]"

OLD = """        # Check if the start token is present in the model output, remove it
        # if it is present.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        # For models that may not generate start token,
        # assume the reasoning content is always at the start.
        if self.end_token not in model_output:
            return model_output, None"""

NEW = """        # Check if the start token is present in the model output, remove it
        # if it is present.
        model_output_parts = model_output.partition(self.start_token)
        had_start = bool(model_output_parts[1])
        model_output = (
            model_output_parts[2] if had_start else model_output_parts[0]
        )

        # [glm53-noopener] Mainline no-opener semantics: a generation with
        # NEITHER think token is a direct answer (thinking off), not
        # reasoning. Keeps: opener-present-never-closed -> reasoning
        # (truncated think), close-only -> else-branch split.
        if self.end_token not in model_output and not had_start:
            return None, model_output

        # For models that may not generate start token,
        # assume the reasoning content is always at the start.
        if self.end_token not in model_output:
            return model_output, None"""

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/reasoning/basic_parsers.py"


def patch_file(path: str, dry_run: bool = False) -> int:
    with open(path, encoding="utf-8") as f:
        text = f.read()

    if MARKER in text:
        if NEW in text:
            ast.parse(text, filename=path)
            print(f"[patch_reasoning_noopener] {path}: already patched; no-op.")
            return 0
        raise AssertionError(
            f"{path}: marker present but the patched form is incomplete "
            "(partial/interrupted earlier edit?); refusing to touch it."
        )

    if OLD not in text:
        raise AssertionError(
            f"{path}: extract_reasoning no-opener block not found "
            "(vLLM layout changed?); refusing to guess."
        )
    if text.count(OLD) != 1:
        raise AssertionError(
            f"{path}: expected exactly one no-opener block, found {text.count(OLD)}."
        )
    text = text.replace(OLD, NEW, 1)

    try:
        ast.parse(text, filename=path)
    except SyntaxError as e:
        raise AssertionError(f"POST-EDIT ast.parse FAILED for {path}: {e}") from e

    if dry_run:
        print(f"[patch_reasoning_noopener] DRY RUN -- {path} not written.")
    else:
        tmp = path + ".glm53-noopener.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)  # atomic: never leave a marker-bearing partial file
        print(f"[patch_reasoning_noopener] {path}: opener-less -> content applied.")
    return 1


def main() -> int:
    # GLM53_NOOPENER_TARGET is a test hook only (synthetic source files).
    path = os.environ.get("GLM53_NOOPENER_TARGET", TARGET)
    if not os.path.isfile(path):
        # Fail closed: stop the rank before `exec vllm serve` rather than
        # boot silently unpatched.
        print(
            f"[patch_reasoning_noopener] FATAL: target not found: {path}",
            file=sys.stderr,
        )
        return 1
    patch_file(path, dry_run=os.environ.get("DRY_RUN") == "1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
