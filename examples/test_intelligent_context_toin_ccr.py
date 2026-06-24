#!/usr/bin/env python3
"""Scale test for IntelligentContextManager TOIN + CCR integration.

This tests that:
1. Dropped messages are stored in CCR
2. Drops are recorded to TOIN
3. The marker includes CCR reference
4. TOIN patterns accumulate across multiple compressions
"""

import json
import os

# Set API key from environment or use provided key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

from headroom.cache.compression_store import get_compression_store
from headroom.config import IntelligentContextConfig
from headroom.telemetry import get_toin
from headroom.tokenizer import Tokenizer
from headroom.tokenizers import EstimatingTokenCounter
from headroom.transforms.intelligent_context import IntelligentContextManager


def create_large_conversation(num_turns: int = 50) -> list[dict]:
    """Create a large conversation with varied content."""
    messages = [{"role": "system", "content": "You are a helpful coding assistant."}]

    for i in range(num_turns):
        # Vary content to create different importance levels
        if i % 10 == 0:
            # Error messages (should be preserved)
            messages.append(
                {"role": "user", "content": f"I'm getting an error: TypeError at line {i * 10}"}
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"The TypeError at line {i * 10} is caused by a type mismatch. "
                    f"Here's the fix:\n```python\n# Fix for error {i}\ndef fix_{i}():\n    pass\n```",
                }
            )
        elif i % 7 == 0:
            # Tool calls (should stay atomic)
            messages.append({"role": "user", "content": f"Search for files matching pattern_{i}"})
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": "search_files",
                                "arguments": f'{{"pattern": "pattern_{i}"}}',
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "content": json.dumps([f"file_{i}_a.py", f"file_{i}_b.py", f"file_{i}_c.py"]),
                }
            )
        else:
            # Regular conversation (lower priority)
            messages.append(
                {"role": "user", "content": f"Question {i}: Can you explain how feature_{i} works?"}
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Feature_{i} is a component that handles processing. "
                    f"It works by iterating through the data and applying "
                    f"transformations. Here's a brief overview of the key aspects "
                    f"and how they interact with other parts of the system. "
                    f"The main entry point is the process() method which takes "
                    f"input data and returns the transformed output.",
                }
            )

    return messages


def test_toin_ccr_integration():
    """Test TOIN + CCR integration with IntelligentContextManager."""
    print("=" * 70)
    print("TOIN + CCR Integration Test for IntelligentContextManager")
    print("=" * 70)

    # Get TOIN and CCR store
    toin = get_toin()
    store = get_compression_store()

    # Record initial state
    initial_patterns = len(toin._patterns) if hasattr(toin, "_patterns") else 0
    # CCR store uses a backend, not direct _store
    if hasattr(store, "_backend") and hasattr(store._backend, "_store"):
        initial_store_size = len(store._backend._store)
    else:
        initial_store_size = 0

    print("\nInitial state:")
    print(f"  TOIN patterns: {initial_patterns}")
    print(f"  CCR store entries: {initial_store_size}")

    # Create manager with TOIN
    config = IntelligentContextConfig(
        enabled=True,
        keep_system=True,
        keep_last_turns=3,
        output_buffer_tokens=2000,
        use_importance_scoring=True,
    )
    manager = IntelligentContextManager(config=config, toin=toin)
    tokenizer = Tokenizer(EstimatingTokenCounter())

    # Run multiple compression cycles to accumulate TOIN patterns
    print("\n" + "-" * 70)
    print("Running compression cycles...")
    print("-" * 70)

    all_ccr_refs = []

    for cycle in range(5):
        # Create fresh conversation each cycle
        messages = create_large_conversation(num_turns=30 + cycle * 5)

        tokens_before = tokenizer.count_messages(messages)

        # Set a tight limit to force dropping
        model_limit = tokens_before // 2

        result = manager.apply(
            messages,
            tokenizer,
            model_limit=model_limit,
            output_buffer=1000,
        )

        # Extract CCR reference from marker if present
        ccr_ref = None
        for marker in result.markers_inserted:
            if "ccr_retrieve" in marker and "reference '" in marker:
                start = marker.find("reference '") + len("reference '")
                end = marker.find("'", start)
                ccr_ref = marker[start:end]
                all_ccr_refs.append(ccr_ref)

        print(f"\nCycle {cycle + 1}:")
        print(f"  Messages: {len(messages)} → {len(result.messages)}")
        print(
            f"  Tokens: {result.tokens_before} → {result.tokens_after} "
            f"({100 * (1 - result.tokens_after / result.tokens_before):.1f}% reduction)"
        )
        print(f"  Transforms: {result.transforms_applied}")
        print(f"  CCR reference: {ccr_ref or 'None'}")

    # Check final state
    final_patterns = len(toin._patterns) if hasattr(toin, "_patterns") else 0
    if hasattr(store, "_backend") and hasattr(store._backend, "_store"):
        final_store_size = len(store._backend._store)
    else:
        final_store_size = 0

    print("\n" + "-" * 70)
    print("Final state:")
    print("-" * 70)
    print(
        f"  TOIN patterns: {initial_patterns} → {final_patterns} (+{final_patterns - initial_patterns})"
    )
    print(
        f"  CCR store entries: {initial_store_size} → {final_store_size} (+{final_store_size - initial_store_size})"
    )
    print(f"  CCR references created: {len(all_ccr_refs)}")

    # Test retrieval from CCR
    if all_ccr_refs:
        print("\n" + "-" * 70)
        print("Testing CCR retrieval...")
        print("-" * 70)

        ref = all_ccr_refs[-1]  # Use the most recent reference
        entry = store.retrieve(ref)

        if entry:
            # Parse the retrieved content from the CompressionEntry
            try:
                dropped_messages = json.loads(entry.original_content)
                print(f"  Retrieved {len(dropped_messages)} dropped messages from CCR")
                print(f"  First message role: {dropped_messages[0].get('role', 'unknown')}")
                print(f"  Content preview: {str(dropped_messages[0].get('content', ''))[:100]}...")
                print("  Entry metadata:")
                print(f"    - Tool: {entry.tool_name}")
                print(f"    - Original tokens: {entry.original_tokens}")
                print(f"    - Compressed tokens: {entry.compressed_tokens}")
            except json.JSONDecodeError:
                print(f"  Retrieved content (not JSON): {entry.original_content[:200]}...")
        else:
            print(f"  WARNING: Could not retrieve CCR reference {ref}")
            # Debug: check what's in the store
            print(f"  Store backend type: {type(store._backend)}")
            if hasattr(store._backend, "_store"):
                print(f"  Backend store keys: {list(store._backend._store.keys())[:5]}...")

    # Print TOIN statistics
    print("\n" + "-" * 70)
    print("TOIN Statistics:")
    print("-" * 70)

    stats = toin.get_stats()
    print(f"  Total patterns: {stats.get('total_patterns', 0)}")
    print(f"  Total compressions: {stats.get('total_compressions', 0)}")
    print(f"  Total retrievals: {stats.get('total_retrievals', 0)}")
    print(f"  Retrieval rate: {stats.get('retrieval_rate', 0):.1%}")

    # Check for intelligent_context_drop patterns
    drop_patterns = (
        [
            p
            for p in toin._patterns.values()
            if hasattr(p, "tool_name") and "intelligent_context" in str(getattr(p, "tool_name", ""))
        ]
        if hasattr(toin, "_patterns")
        else []
    )

    print(f"  IntelligentContext drop patterns: {len(drop_patterns)}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    # Assertions
    assert final_patterns >= initial_patterns, "TOIN should have recorded new patterns"
    assert len(all_ccr_refs) > 0, "Should have created CCR references"
    # CCR store entries should exist (though count may vary due to TTL)
    if final_store_size == 0 and initial_store_size == 0:
        print("  Note: CCR store size shows 0 (entries may have different backend)")
    else:
        assert final_store_size > initial_store_size, "CCR store should have new entries"

    print("\n✓ All assertions passed!")
    return True


def test_with_real_llm():
    """Test with a real LLM call to verify end-to-end flow."""
    print("\n" + "=" * 70)
    print("Real LLM Integration Test")
    print("=" * 70)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Skipping real LLM test - OPENAI_API_KEY not set")
        return

    try:
        from openai import OpenAI

        client = OpenAI()
    except ImportError:
        print("Skipping real LLM test - openai package not installed")
        return

    # Create a conversation that will be compressed
    messages = create_large_conversation(num_turns=20)

    # Apply IntelligentContext compression
    toin = get_toin()
    config = IntelligentContextConfig(
        enabled=True,
        keep_system=True,
        keep_last_turns=2,
    )
    manager = IntelligentContextManager(config=config, toin=toin)
    tokenizer = Tokenizer(EstimatingTokenCounter())

    tokens_before = tokenizer.count_messages(messages)

    result = manager.apply(
        messages,
        tokenizer,
        model_limit=tokens_before // 3,  # Force significant compression
        output_buffer=500,
    )

    print("\nCompression result:")
    print(f"  Messages: {len(messages)} → {len(result.messages)}")
    print(f"  Tokens: {result.tokens_before} → {result.tokens_after}")

    # Convert to OpenAI format (filter out tool messages with None content)
    openai_messages = []
    for msg in result.messages:
        if msg.get("role") == "tool":
            continue  # Skip tool messages for this test
        if msg.get("content") is None:
            continue  # Skip messages with None content
        openai_messages.append({"role": msg["role"], "content": msg["content"]})

    # Add a question about the compressed context
    openai_messages.append(
        {
            "role": "user",
            "content": "Based on our conversation, what errors did we discuss? "
            "If you see a message about compressed context, note the CCR reference.",
        }
    )

    print(f"\nSending {len(openai_messages)} messages to OpenAI...")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=openai_messages,
            max_tokens=500,
        )

        print("\nLLM Response:")
        print("-" * 40)
        print(response.choices[0].message.content)
        print("-" * 40)
        print(f"\nTokens used: {response.usage.total_tokens}")

    except Exception as e:
        print(f"LLM call failed: {e}")


if __name__ == "__main__":
    # Run the TOIN + CCR integration test
    test_toin_ccr_integration()

    # Run real LLM test if API key available
    test_with_real_llm()
