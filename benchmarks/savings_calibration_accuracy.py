"""How close are Headroom's per-request savings to the provider's own count?

For each sampled turn of real Claude Code transcripts this replays what the
proxy does in cache mode (earlier messages frozen, only new content
compressed) and compares three numbers for the SAME request:

* truth       -- Anthropic ``count_tokens`` of the original request minus that
                 of the forwarded request: the saving in real Claude tokens.
* raw         -- Headroom's local-tokenizer ``tokens_saved`` (what reports
                 showed before calibration).
* calibrated  -- ``SavingsCalibrator`` output, fed only what the proxy sees on
                 a live request: Anthropic's billed input for the forwarded
                 request and Headroom's local counts. No side calls.

A constant system prompt is sent with every request but NOT counted locally,
exactly like the proxy's message-only count, so the turn method's cancellation
of constant content is exercised too.

``count_tokens`` is free; the run makes two calls per sampled turn per model.

Usage::

    ANTHROPIC_API_KEY=... python benchmarks/savings_calibration_accuracy.py \\
        ~/.claude/projects/*/<session>.jsonl ... --models claude-sonnet-4-6,claude-sonnet-5-5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from headroom import compress
from headroom.proxy.savings_calibration import local_request_overhead
from headroom.tokenizers import get_tokenizer


def _agent_tools() -> list[dict[str, Any]]:
    """~15 tool definitions shaped like a coding agent's: long prose
    descriptions plus JSON schemas, roughly 20K tokens, the way Claude Code's
    tools dominate the fixed part of every request. Built deterministically
    from this repo's docs so runs are reproducible."""
    root = os.path.join(os.path.dirname(__file__), "..", "docs")
    prose: list[str] = []
    for dirpath, _dirs, files in sorted(os.walk(root)):
        for name in sorted(files):
            if name.endswith(".md"):
                with open(os.path.join(dirpath, name), encoding="utf-8", errors="ignore") as fh:
                    prose.append(fh.read())
    text = "\n".join(prose)
    tools = []
    for i in range(15):
        tools.append(
            {
                "name": f"tool_{i}",
                "description": text[i * 5_000 : (i + 1) * 5_000],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute path to operate on."},
                        "pattern": {"type": "string", "description": "Optional glob or regex."},
                        "limit": {"type": "integer", "description": "Maximum results."},
                    },
                    "required": ["path"],
                },
            }
        )
    return tools


TOOLS = _agent_tools()

SYSTEM = "You are a coding agent working in a large repository.\n" + "\n".join(
    f"Guideline {i}: keep diffs small, run the tests for module m{i}, explain risky changes."
    for i in range(150)
)


def load_conversation(path: str, max_chars: int) -> list[dict[str, Any]]:
    """Valid Anthropic messages (text, tool_use, tool_result) from a transcript."""
    msgs: list[dict[str, Any]] = []
    size = 0
    for line in open(path, encoding="utf-8"):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        message = entry.get("message")
        if entry.get("type") not in ("user", "assistant") or not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            continue
        blocks = []
        for block in content:
            kind = block.get("type")
            if kind == "text" and block.get("text", "").strip():
                blocks.append({"type": "text", "text": block["text"]})
            elif kind == "tool_use":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": block.get("input", {}),
                    }
                )
            elif kind == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list):
                    inner = "\n".join(x.get("text", "") for x in inner if x.get("type") == "text")
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["tool_use_id"],
                        "content": inner or "(empty)",
                    }
                )
        if not blocks:
            continue
        role = message["role"]
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += blocks
        else:
            msgs.append({"role": role, "content": blocks})
        size += len(json.dumps(blocks))
        if size > max_chars:
            break
    use_ids = {b["id"] for m in msgs for b in m["content"] if b["type"] == "tool_use"}
    result_ids = {
        b["tool_use_id"] for m in msgs for b in m["content"] if b["type"] == "tool_result"
    }
    for m in msgs:
        m["content"] = [
            b
            for b in m["content"]
            if not (b["type"] == "tool_use" and b["id"] not in result_ids)
            and not (b["type"] == "tool_result" and b["tool_use_id"] not in use_ids)
        ]
    msgs = [m for m in msgs if m["content"]]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


def count_tokens(key: str, model: str, messages: list[dict[str, Any]]) -> int:
    tool_names = {b["name"] for m in messages for b in m["content"] if b.get("type") == "tool_use"}
    tools = TOOLS + [
        {"name": n, "description": f"Tool {n}.", "input_schema": {"type": "object"}}
        for n in sorted(tool_names)
    ]
    body = json.dumps(
        {"model": model, "system": SYSTEM, "tools": tools, "messages": messages}
    ).encode()
    for attempt in range(6):
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages/count_tokens",
            data=body,
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        try:
            return int(json.load(urllib.request.urlopen(req, timeout=120))["input_tokens"])
        except urllib.error.HTTPError as err:
            if err.code in (429, 529) and attempt < 5:
                time.sleep(2**attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def turn_ends(msgs: list[dict[str, Any]], samples: int) -> list[int]:
    """Indices just after user turns, i.e. points where the proxy sends a request."""
    ends = [i + 1 for i, m in enumerate(msgs) if m["role"] == "user" and i >= 1]
    if len(ends) <= samples:
        return ends
    step = len(ends) / samples
    return sorted({ends[int(k * step)] for k in range(samples)} | {ends[-1]})


def run(paths: list[str], models: list[str], samples: int, max_chars: int) -> int:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("set ANTHROPIC_API_KEY", file=sys.stderr)
        return 2
    try:
        from headroom.transforms.kompress_compressor import warm_kompress_model

        warm_kompress_model()  # production compresses prose/code with Kompress too
    except Exception as err:  # pragma: no cover - benchmark only
        print(f"Kompress unavailable ({err}); structural compression only", file=sys.stderr)

    # (model, conversation, truth, raw, request_ratio)
    rows: list[tuple[str, str, int, int, float]] = []
    for model in models:
        local = get_tokenizer(model)
        for path in paths:
            msgs = load_conversation(path, max_chars)
            if len(msgs) < 6:
                continue
            conv = os.path.basename(path)[:8]
            forwarded: list[dict[str, Any]] = []
            for end in turn_ends(msgs, samples):
                original = msgs[:end]
                new = original[len(forwarded) :]
                result = compress(forwarded + new, model=model, frozen_message_count=len(forwarded))
                forwarded = list(result.messages)
                # Fixed part counted exactly as the proxy does (savings_calibration).
                names = {
                    b["name"] for m in original for b in m["content"] if b.get("type") == "tool_use"
                }
                tools = TOOLS + [
                    {"name": n, "description": f"Tool {n}.", "input_schema": {"type": "object"}}
                    for n in sorted(names)
                ]
                overhead, _ = local_request_overhead(model, {"system": SYSTEM, "tools": tools})
                local_fwd = local.count_messages(forwarded) + overhead
                saved_local = max(
                    local.count_messages(original) - local.count_messages(forwarded), 0
                )
                billed = count_tokens(key, model, forwarded)
                truth = max(count_tokens(key, model, original) - billed, 0)
                ratio = billed / local_fwd if local_fwd else 0.0
                rows.append((model, conv, truth, saved_local, ratio))
                print(
                    f"{model} {conv} msgs={end:4d} truth={truth:7,} raw={saved_local:7,} "
                    f"ratio={ratio:.3f}",
                    flush=True,
                )

    def report(name: str, preds: list[float], truths: list[int]) -> str:
        errs = sorted(abs(p / t - 1) * 100 for p, t in zip(preds, truths))
        bias = sum(preds) / sum(truths) * 100 - 100
        return (
            f"{name}: median |err| {statistics.median(errs):.1f}%, "
            f"p90 {errs[int(0.9 * (len(errs) - 1))]:.1f}%, total {bias:+.1f}%"
        )

    print("\nPer-request error vs Anthropic count_tokens (requests saving >= 200 tokens):")
    for model in models:
        sel = [r for r in rows if r[0] == model and r[2] >= 200]
        if len(sel) < 2:
            continue
        truths = [r[2] for r in sel]
        convs = sorted({r[1] for r in sel})
        # Removed-content correction, fitted leave-one-conversation-out so each
        # conversation is scored with a correction it did not help fit.
        hybrid = []
        for r in sel:
            fit = [x for x in sel if x[1] != r[1]] or sel
            corr = sum(x[2] for x in fit) / sum(x[3] * x[4] for x in fit)
            hybrid.append(r[3] * r[4] * corr)
        overall = sum(r[2] for r in sel) / sum(r[3] * r[4] for r in sel)
        print(f"  {model}: n={len(sel)} requests, {len(convs)} conversations")
        print("    " + report("raw (today)            ", [r[3] for r in sel], truths))
        print("    " + report("per-request ratio      ", [r[3] * r[4] for r in sel], truths))
        print("    " + report("ratio x correction (LOO)", hybrid, truths))
        print(f"    fitted removed-content correction: {overall:.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("transcripts", nargs="+")
    parser.add_argument("--models", default="claude-sonnet-4-6,claude-sonnet-5-5")
    parser.add_argument("--samples", type=int, default=12, help="turns sampled per transcript")
    parser.add_argument("--max-chars", type=int, default=300_000)
    args = parser.parse_args()
    return run(args.transcripts, args.models.split(","), args.samples, args.max_chars)


if __name__ == "__main__":
    sys.exit(main())
