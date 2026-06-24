"""Test CCR markers and content preservation in compressed output."""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

from examples.context_compression_demo import build_retriever_chunks
from headroom import compress


def main():
    chunks = build_retriever_chunks()
    retriever_json = json.dumps(chunks, indent=2)

    messages = [
        {"role": "user", "content": "What are the types of reward hacking discussed in the blogs?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "retrieve_blog_posts",
                        "arguments": json.dumps({"query": "types of reward hacking"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_001", "content": retriever_json},
    ]

    result = compress(messages, model="claude-sonnet-4-5-20250929")

    compressed_tool = str(result.messages[2].get("content", ""))

    print("=== Compressed tool output (FULL) ===")
    print(compressed_tool)
    print()
    print(f"Tokens: {result.tokens_before} -> {result.tokens_after} ({result.tokens_saved} saved)")
    print(f"Transforms: {result.transforms_applied}")
    print()

    # Check for CCR markers
    if "hash=" in compressed_tool:
        print("CCR MARKERS FOUND — LLM can retrieve originals")
    else:
        print("No CCR markers")

    print()

    # Check key content
    key_terms = {
        "reward tampering": False,
        "sycophancy": False,
        "specification gaming": False,
        "proxy gaming": False,
        "reward model hacking": False,
        "distribution shift": False,
    }
    for term in key_terms:
        key_terms[term] = term.lower() in compressed_tool.lower()
        status = "FOUND" if key_terms[term] else "MISSING"
        print(f"  {term}: {status}")

    found = sum(1 for v in key_terms.values() if v)
    print(f"\n{found}/{len(key_terms)} key concepts preserved in compressed output")


if __name__ == "__main__":
    main()
