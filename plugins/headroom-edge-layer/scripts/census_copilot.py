#!/usr/bin/env python3
"""Phase 0 token census — GitHub Copilot Chat (VS Code) sessions.

    python3 scripts/census_copilot.py --root "/mnt/c/Users/<you>/AppData/Roaming/Code/User/workspaceStorage"

Copilot Chat stores sessions completely differently from Claude Code: one
file per session at `<workspaceStorage>/<hash>/chatSessions/<uuid>.jsonl`,
but despite the extension each *line* is a full JSON snapshot of the whole
session (`{"kind": 0, "v": {...}}`) — VS Code re-saves the entire state on
every update rather than appending deltas. This script reads every line and
keeps the one with the most `requests`, which in practice is the fullest
snapshot for that file.

Schema drift is real and unavoidable here: this was reverse-engineered from
one real user's session history spanning Copilot Chat extension v0.59.0
through v0.65.0, and fields visibly moved between versions:

- `promptTokens`/`outputTokens` (sometimes `completionTokens`) sit directly
  on the request object in some versions, under `result.metadata` in
  others. Both locations are checked.
- Terminal output sometimes appears as a standalone `"kind": "terminal"`
  response part with `terminalCommandOutput.text`, sometimes folded into
  `toolSpecificData` on a `"kind": "toolInvocationSerialized"` wrapper
  instead. Both are read, deduplicated by whatever call-identifying field
  is available, but this dedup is best-effort — see the module-level
  ``KNOWN_LIMITATIONS`` below before trusting a bucket-level number.

**What's solid**: the total cost figure. `promptTokens`/`outputTokens` are
checked in both locations and priced from your own account's embedded model
catalog (found by scanning `inputState` for any object carrying `id`,
`inputCost` and `outputCost` — GitHub's "AI Credits" unit, 1 AIC = $0.01,
confirmed against GitHub's own docs). A model missing from every scanned
catalog is flagged as unpriced, never silently recorded as $0 — except
`ollama-models/*`, which really is $0 (local inference).

**What's approximate**: the per-bucket split. Tool calls are matched to
their real output content via `result.metadata.toolCallResults`, keyed by
tool-call ID, which is itself a VS Code-internal rich-text AST (nested
`type`/`ctor`/`children` trees, not plain strings) — extracted here by
recursively collecting every `"text"` value found anywhere in the tree.

Never reads `toolCallRounds[].thinking` — that's Claude's own raw thinking
block content passed through Copilot's metadata when the underlying model
is Claude, and this script has no legitimate reason to touch it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_SRC = _ROOT / "src"
if str(_PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_SRC))

from headroom_edge_layer.buckets import (  # noqa: E402
    BUCKET_FILE_READ,
    BUCKET_GREP,
    BUCKET_OTHER,
    BUCKET_SHELL_LOG,
    BUCKET_WEB,
    classify_tool_name,
)
from headroom_edge_layer.cache_safety import estimate_tokens  # noqa: E402

DEFAULT_ROOT = Path.home() / ".config" / "Code" / "User" / "workspaceStorage"

AIC_TO_USD = 0.01  # GitHub AI Credits, effective 2026-06-01 (docs.github.com/copilot/reference/copilot-billing)

KNOWN_LIMITATIONS = (
    "Bucket-level tokens can double-count terminal output when both a standalone "
    "'terminal' response part AND a toolInvocationSerialized wrapper describe the "
    "same call without a shared identifying field to dedupe on (seen across "
    "Copilot Chat extension v0.59.0-0.65.0). The request-level cost total does "
    "NOT have this problem -- it comes straight from promptTokens/outputTokens."
)

# Real toolId values observed in one user's actual session history (512
# requests, extension v0.59.0-0.65.0). Extend this as new toolIds show up --
# anything unmapped falls through to classify_tool_name() (catches "mcp_*"
# via its substring rule) and then BUCKET_OTHER.
_TOOL_ID_TO_BUCKET: dict[str, str] = {
    "copilot_readFile": BUCKET_FILE_READ,
    "copilot_findTextInFiles": BUCKET_GREP,
    "copilot_findFiles": BUCKET_GREP,
    "copilot_listDirectory": BUCKET_GREP,
    "copilot_searchCodebase": BUCKET_GREP,
    "run_in_terminal": BUCKET_SHELL_LOG,
    "get_terminal_output": BUCKET_SHELL_LOG,
    "send_to_terminal": BUCKET_SHELL_LOG,
    "kill_terminal": BUCKET_SHELL_LOG,
    "copilot_fetchWebPage": BUCKET_WEB,
    "vscode_fetchWebPage_internal": BUCKET_WEB,
    "read_page": BUCKET_WEB,
    "navigate_page": BUCKET_WEB,
    "click_element": BUCKET_WEB,
    "screenshot_page": BUCKET_WEB,
    "open_browser_page": BUCKET_WEB,
}


def bucket_for_tool_id(tool_id: str) -> str:
    if tool_id in _TOOL_ID_TO_BUCKET:
        return _TOOL_ID_TO_BUCKET[tool_id]
    classified = classify_tool_name(tool_id)
    return classified if classified != BUCKET_OTHER else BUCKET_OTHER


# ---------------------------------------------------------------------------
# Session loading: multi-line JSONL, keep the line with the most requests.
# ---------------------------------------------------------------------------
def load_best_snapshot(session_file: Path) -> dict[str, Any] | None:
    best_requests = -1
    best_v: dict[str, Any] | None = None
    try:
        with session_file.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v = obj.get("v") if isinstance(obj, dict) else None
                if not isinstance(v, dict):
                    continue
                requests = v.get("requests", [])
                if isinstance(requests, list) and len(requests) > best_requests:
                    best_requests = len(requests)
                    best_v = v
    except OSError:
        return None
    return best_v


def find_model_catalog(v: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    """Scans `v` for every embedded model-pricing entry it can find.

    A catalog entry is any dict carrying `id`, `inputCost` and `outputCost`
    (GitHub's own AICs/1M-tokens rates for that model, on this account).
    """
    catalog: dict[str, dict[str, float | None]] = {}

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if "id" in obj and "inputCost" in obj and "outputCost" in obj:
                catalog[obj["id"]] = {
                    "input": obj.get("inputCost"),
                    "output": obj.get("outputCost"),
                    "cache_read": obj.get("cacheCost"),
                    "cache_write": obj.get("cacheWriteCost"),
                }
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(v.get("inputState"))
    return catalog


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------
@dataclass
class CostEstimate:
    usd: float | None
    source: str  # "catalog" | "local_free" | "unpriced"


def normalize_model_id(model_id: str) -> str:
    """`"copilot/claude-sonnet-5"` and `"claude-sonnet-5"` both key the same
    catalog entry -- the `agent.modelId` and `toolCallRounds[].modelId`
    fields disagree on whether the `copilot/` prefix is present.
    """
    return model_id.split("/", 1)[1] if "/" in model_id and not model_id.startswith("ollama-models/") else model_id


def estimate_request_cost(
    model_id: str,
    *,
    prompt_tokens: int,
    output_tokens: int,
    catalog: dict[str, dict[str, float | None]],
) -> CostEstimate:
    if model_id.startswith("ollama-models/"):
        return CostEstimate(usd=0.0, source="local_free")

    entry = catalog.get(normalize_model_id(model_id))
    if entry is None or entry.get("input") is None or entry.get("output") is None:
        return CostEstimate(usd=None, source="unpriced")

    usd = (
        prompt_tokens * entry["input"] * AIC_TO_USD + output_tokens * entry["output"] * AIC_TO_USD
    ) / 1_000_000
    return CostEstimate(usd=usd, source="catalog")


# ---------------------------------------------------------------------------
# Tool-call content extraction (the rich-text AST walker)
# ---------------------------------------------------------------------------
def extract_text_from_value(value: Any, *, max_chars: int = 200_000) -> str:
    """Recursively collects every `"text"` string found anywhere in `value`.

    Copilot's `toolCallResults[id].content` is a VS Code-internal rich-text
    tree (`{"$mid":..., "value": {"node": {"type":1, "children": [...]}}}`),
    not a plain string -- the actual text sits in scattered leaf nodes'
    `"text"` fields. This walker doesn't care about the tree's structure,
    it just gathers every `"text"` value it finds, which is robust to the
    exact node-type/ctor shape changing between extension versions.
    """
    texts: list[str] = []
    total = 0

    def walk(obj: Any) -> None:
        nonlocal total
        if total >= max_chars:
            return
        if isinstance(obj, dict):
            text = obj.get("text")
            if isinstance(text, str):
                texts.append(text)
                total += len(text)
            for key, val in obj.items():
                if key != "text":
                    walk(val)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, str):
            texts.append(obj)
            total += len(obj)

    walk(value)
    return "\n".join(texts)[:max_chars]


# ---------------------------------------------------------------------------
# Per-request tool-call extraction, defensive across schema versions.
# ---------------------------------------------------------------------------
@dataclass
class ToolCallObservation:
    tool_id: str
    bucket: str
    tokens: int
    dedupe_key: str | None  # tool_call_id or terminal_command_id, when available


def extract_tool_calls(request: dict[str, Any]) -> list[ToolCallObservation]:
    result = request.get("result") if isinstance(request.get("result"), dict) else {}
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    tool_call_results = metadata.get("toolCallResults")
    tool_call_results = tool_call_results if isinstance(tool_call_results, dict) else {}

    observations: list[ToolCallObservation] = []
    seen_dedupe_keys: set[str] = set()

    response_parts = request.get("response")
    if not isinstance(response_parts, list):
        return observations

    for part in response_parts:
        if not isinstance(part, dict):
            continue
        kind = part.get("kind")

        if kind == "terminal":
            dedupe_key = part.get("terminalCommandId")
            if dedupe_key and dedupe_key in seen_dedupe_keys:
                continue
            output = part.get("terminalCommandOutput")
            text = output.get("text", "") if isinstance(output, dict) else ""
            observations.append(
                ToolCallObservation(
                    tool_id="run_in_terminal",
                    bucket=BUCKET_SHELL_LOG,
                    tokens=estimate_tokens(text) if text else 0,
                    dedupe_key=dedupe_key,
                )
            )
            if dedupe_key:
                seen_dedupe_keys.add(dedupe_key)
            continue

        if kind == "toolInvocationSerialized":
            tool_id = part.get("toolId", "unknown")
            tool_call_id = part.get("toolCallId")
            if tool_call_id and tool_call_id in seen_dedupe_keys:
                continue

            # Newer schema: terminal detail folded into toolSpecificData
            # instead of a standalone "terminal" part.
            tool_specific = part.get("toolSpecificData")
            if isinstance(tool_specific, dict) and tool_specific.get("kind") == "terminal":
                text = extract_text_from_value(tool_specific)
            elif tool_call_id and tool_call_id in tool_call_results:
                text = extract_text_from_value(tool_call_results[tool_call_id])
            else:
                # Last resort: the display-only invocation message. Weaker
                # signal (UI text, not real output), but better than a
                # silent zero when no content is recoverable at all.
                invocation = part.get("invocationMessage")
                text = invocation.get("value", "") if isinstance(invocation, dict) else ""

            observations.append(
                ToolCallObservation(
                    tool_id=tool_id,
                    bucket=bucket_for_tool_id(tool_id),
                    tokens=estimate_tokens(text) if text else 0,
                    dedupe_key=tool_call_id,
                )
            )
            if tool_call_id:
                seen_dedupe_keys.add(tool_call_id)

    return observations


def extract_request_usage(request: dict[str, Any]) -> tuple[int, int, str | None]:
    """Returns (prompt_tokens, output_tokens, model_id) -- both possible
    schema locations checked, per the module docstring's schema-drift note.
    """
    result = request.get("result") if isinstance(request.get("result"), dict) else {}
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}

    prompt_tokens = metadata.get("promptTokens", request.get("promptTokens", 0)) or 0
    output_tokens = metadata.get(
        "outputTokens", metadata.get("completionTokens", request.get("completionTokens", 0))
    ) or 0

    model_id = None
    agent = request.get("agent")
    if isinstance(agent, dict):
        model_id = agent.get("modelId")
    tool_call_rounds = metadata.get("toolCallRounds")
    if isinstance(tool_call_rounds, list) and tool_call_rounds:
        last_round = tool_call_rounds[-1]
        if isinstance(last_round, dict) and last_round.get("modelId"):
            model_id = last_round["modelId"]

    return int(prompt_tokens), int(output_tokens), model_id


# ---------------------------------------------------------------------------
# Aggregation and reporting -- same shape as census.py for comparability.
# ---------------------------------------------------------------------------
@dataclass
class BucketStats:
    tokens: int = 0
    occurrences: int = 0


@dataclass
class CensusResult:
    bucket_stats: dict[str, BucketStats] = field(default_factory=lambda: defaultdict(BucketStats))
    sessions_scanned: int = 0
    sessions_with_requests: int = 0
    total_requests: int = 0
    total_real_cost_usd: float = 0.0
    local_free_requests: int = 0
    unpriced_requests: int = 0
    catalog_models_seen: set[str] = field(default_factory=set)


def census_session(v: dict[str, Any], result: CensusResult) -> None:
    requests = v.get("requests", [])
    if not isinstance(requests, list) or not requests:
        return
    result.sessions_with_requests += 1
    catalog = find_model_catalog(v)
    result.catalog_models_seen.update(catalog.keys())

    for request in requests:
        if not isinstance(request, dict):
            continue
        result.total_requests += 1

        prompt_tokens, output_tokens, model_id = extract_request_usage(request)
        if model_id:
            estimate = estimate_request_cost(
                model_id, prompt_tokens=prompt_tokens, output_tokens=output_tokens, catalog=catalog
            )
            if estimate.source == "unpriced":
                result.unpriced_requests += 1
            elif estimate.source == "local_free":
                result.local_free_requests += 1
            else:
                result.total_real_cost_usd += estimate.usd or 0.0

        for observation in extract_tool_calls(request):
            stats = result.bucket_stats[observation.bucket]
            stats.tokens += observation.tokens
            stats.occurrences += 1


def format_report(result: CensusResult) -> str:
    lines: list[str] = []
    lines.append(f"Sessions scanned: {result.sessions_scanned}")
    lines.append(f"Sessions with at least one request: {result.sessions_with_requests}")
    lines.append(f"Total requests: {result.total_requests}")
    lines.append(f"Total estimated real cost (USD): {result.total_real_cost_usd:.4f}")
    if result.local_free_requests:
        lines.append(f"  ({result.local_free_requests} requests were local Ollama models -- $0, real usage)")
    if result.unpriced_requests:
        lines.append(
            f"WARNING: {result.unpriced_requests} requests used a model not found in any scanned "
            "catalog -- cost total above UNDERCOUNTS. Models seen in a catalog: "
            f"{sorted(result.catalog_models_seen) or '(none found)'}"
        )
    lines.append("")
    lines.append("| Bucket | Tool-call tokens (approximate, see limitations) | Occurrences |")
    lines.append("|---|---|---|")
    total_tokens = sum(s.tokens for s in result.bucket_stats.values()) or 1
    for bucket, stats in sorted(result.bucket_stats.items(), key=lambda kv: kv[1].tokens, reverse=True):
        share = stats.tokens / total_tokens
        lines.append(f"| {bucket} | {stats.tokens} ({share:.1%}) | {stats.occurrences} |")
    lines.append("")
    lines.append("Known limitations of the bucket breakdown above:")
    lines.append(f"  {KNOWN_LIMITATIONS}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()

    if not args.root.exists():
        print(f"No such directory: {args.root}", file=sys.stderr)
        return 1

    result = CensusResult()
    session_files = sorted(args.root.glob("*/chatSessions/*.jsonl"))
    if args.max_sessions:
        session_files = session_files[: args.max_sessions]

    for session_file in session_files:
        result.sessions_scanned += 1
        v = load_best_snapshot(session_file)
        if v is None:
            continue
        census_session(v, result)

    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
