"""Verify that SmartCrusher preserves ERROR entries.

This is critical - errors should NEVER be dropped during compression.
"""

import json

from headroom.config import SmartCrusherConfig
from headroom.providers import OpenAIProvider
from headroom.transforms import SmartCrusher

from .mock_tools import generate_log_entries


def main():
    print("\n" + "=" * 70)
    print("VERIFYING ERROR PRESERVATION IN SMARTCRUSHER")
    print("=" * 70)

    # Generate logs with some ERROR entries
    raw_output = generate_log_entries("test-service", count=200)
    data = json.loads(raw_output)

    # Count errors in original
    original_errors = [e for e in data["entries"] if e["level"] == "ERROR"]
    print(f"\nOriginal log entries: {len(data['entries'])}")
    print(f"ERROR entries in original: {len(original_errors)}")
    print("\nERROR messages found:")
    for err in original_errors[:5]:  # Show first 5
        print(f"  - {err['message'][:60]}...")

    # Apply SmartCrusher
    smart_config = SmartCrusherConfig(
        enabled=True,
        min_tokens_to_crush=200,
        max_items_after_crush=20,
    )

    provider = OpenAIProvider()
    tokenizer = provider.get_token_counter("gpt-4o")
    crusher = SmartCrusher(config=smart_config)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Find ERROR entries in the logs"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "function": {"name": "search_logs", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "content": raw_output, "tool_call_id": "call_1"},
    ]

    result = crusher.apply(messages, tokenizer=tokenizer)
    compressed_output = result.messages[-1]["content"]

    # Handle case where SmartCrusher may add markers
    # Try to find the JSON part
    try:
        compressed_data = json.loads(compressed_output)
    except json.JSONDecodeError:
        # Try to extract just the JSON object
        import re

        json_match = re.search(r"(\{.*\})", compressed_output, re.DOTALL)
        if json_match:
            compressed_data = json.loads(json_match.group(1))
        else:
            print("Could not parse compressed output:")
            print(compressed_output[:500])
            return

    # Count errors in compressed
    compressed_errors = [e for e in compressed_data["entries"] if e["level"] == "ERROR"]
    print(f"\nCompressed log entries: {len(compressed_data['entries'])}")
    print(f"ERROR entries preserved: {len(compressed_errors)}")
    print("\nERROR messages in compressed:")
    for err in compressed_errors[:5]:
        print(f"  - {err['message'][:60]}...")

    # Verification
    print("\n" + "=" * 70)
    if len(compressed_errors) >= len(original_errors):
        print("SUCCESS: All ERROR entries were preserved!")
    elif len(compressed_errors) > 0:
        print(f"PARTIAL: {len(compressed_errors)}/{len(original_errors)} ERROR entries preserved")
    else:
        print("FAILURE: ERROR entries were dropped!")
    print("=" * 70)

    # Show compression ratio
    original_count = len(data["entries"])
    compressed_count = len(compressed_data["entries"])
    reduction = (original_count - compressed_count) / original_count * 100
    print(
        f"\nCompression: {original_count} → {compressed_count} entries ({reduction:.1f}% reduction)"
    )
    print(f"But kept: {len(compressed_errors)} of {len(original_errors)} ERROR entries")


if __name__ == "__main__":
    main()
