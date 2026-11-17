"""Regenerate the "Proof" savings table published on the docs landing page.

WHY THIS EXISTS
---------------
docs/content/docs/index.mdx published four precise before/after token counts
with no reproducible source. The nearest harness,
``real_world_agent_benchmark.py``, seeded nothing, so its corpus differed on
every run and the published figures could not be reproduced by anyone,
including us. A number on the front page of the docs that nobody can
regenerate is a liability, not evidence.

This script fixes the reproducibility half. It seeds the generators, builds the
same scenarios, and measures tokens through the real tokenizer and the real
``compress()`` path. No network, no API key, no model call: the table is a
statement about token counts, and token counts are computable locally.

    uv run python benchmarks/index_proof_table.py

Deterministic: same seed in, same numbers out, on any machine.
"""

from __future__ import annotations

import argparse
import json

from real_world_agent_benchmark import (  # noqa: E402
    DEFAULT_SEED,
    create_codebase_exploration_scenario,
    create_issue_triage_scenario,
    create_sre_debugging_scenario,
    generate_github_code_search,
    seed_everything,
)

from headroom import CompressConfig, compress
from headroom.providers.openai_compatible import OpenAICompatibleTokenCounter

MODEL = "gpt-5.6"


def _tool_messages(tools: list[dict]) -> list[dict]:
    """The tool payloads as the proxy would actually see them on the wire."""
    return [
        {
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": json.dumps(t["result"]),
        }
        for i, t in enumerate(tools)
    ]


def measure(label: str, tools: list[dict], tok, config: CompressConfig) -> dict:
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Analyse the tool output and answer."},
        *_tool_messages(tools),
    ]
    before = sum(tok.count_text(m["content"]) for m in msgs if m["role"] == "tool")
    result = compress(msgs, model=MODEL, config=config)
    after = sum(
        tok.count_text(m["content"])
        for m in result.messages
        if m.get("role") == "tool" and isinstance(m.get("content"), str)
    )
    saved = before - after
    return {
        "scenario": label,
        "before": before,
        "after": after,
        "saved": saved,
        "savings_pct": (saved / before * 100) if before else 0.0,
        "transforms": sorted(set(result.transforms_applied)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    tok = OpenAICompatibleTokenCounter(model=MODEL)

    # One configuration: the default, i.e. what a coding agent actually gets.
    #
    # A second `protect_recent=0` run used to be printed here, and came out
    # identical in every row and in the total (#4732). Not a coincidence: the
    # router applies `protect_recent` only to content detected as SOURCE_CODE
    # ("Don't compress recent CODE"). These scenarios are JSON tool payloads --
    # JSON_ARRAY even under the native detector -- so the guard never fires and
    # the second table carried no information. Printing it again would only
    # relabel a duplicate, so it is stated here once instead.
    cfg = CompressConfig()

    # Built in a fixed order: every generator draws from the same global RNG,
    # so re-ordering these lines changes every number below.
    print(f"seed={args.seed}  model={MODEL}  tokenizer={type(tok._tokenizer).__name__}")

    seed_everything(args.seed)
    rows = [
        measure(
            "Code search (100 results)",
            [generate_github_code_search("JWT authentication middleware", num_results=100)],
            tok,
            cfg,
        ),
        measure("SRE incident debugging", create_sre_debugging_scenario().tools, tok, cfg),
        measure("Codebase exploration", create_codebase_exploration_scenario().tools, tok, cfg),
        measure("GitHub issue triage", create_issue_triage_scenario().tools, tok, cfg),
    ]

    print("\n=== default (protect_recent=4) ===")
    print(f"{'Scenario':<30} {'Before':>10} {'After':>10} {'Savings':>9}")
    print("-" * 62)
    for r in rows:
        print(f"{r['scenario']:<30} {r['before']:>10,} {r['after']:>10,} {r['savings_pct']:>8.0f}%")
    tb = sum(r["before"] for r in rows)
    ta = sum(r["after"] for r in rows)
    print("-" * 62)
    print(f"{'TOTAL':<30} {tb:>10,} {ta:>10,} {(tb - ta) / tb * 100:>8.0f}%")

    print(
        "\n\nMarkdown for docs/content/docs/index.mdx "
        "(default config, which must be stated on the page):\n"
    )
    print("| Scenario | Before | After | Savings |")
    print("|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['scenario']} | {r['before']:,} | {r['after']:,} | **{r['savings_pct']:.0f}%** |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
