"""Integration tests for Headroom Memory System.

These tests use REAL API calls - no mocks.
Tests verify the full flow from LLM tool calls to memory storage.

Requirements:
    - OPENAI_API_KEY environment variable must be set
    - Run with: pytest tests/test_memory_integration.py -v -s
"""

from __future__ import annotations

import os
import tempfile
import uuid

import pytest
from openai import OpenAI

# API keys must be set externally via environment variables
# Tests will be skipped if OPENAI_API_KEY is not available


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
class TestMemoryIntegration:
    """Integration tests for the memory system with real LLM calls."""

    @pytest.fixture
    def openai_client(self):
        """Create an OpenAI client."""
        return OpenAI()

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        # Cleanup
        try:
            os.unlink(f.name)
        except OSError:
            pass

    @pytest.fixture
    def user_id(self):
        """Generate a unique user ID for test isolation."""
        return f"test_user_{uuid.uuid4().hex[:8]}"

    # =========================================================================
    # Test 1: Verify optimized tools include pre-extraction fields
    # =========================================================================

    def test_optimized_tools_have_extraction_fields(self):
        """Verify that optimized tools include pre-extraction fields."""
        from headroom.memory.tools import get_memory_tools, get_memory_tools_optimized

        # Standard tools should NOT have facts/extracted_entities
        standard_tools = get_memory_tools()
        memory_save = next(t for t in standard_tools if t["function"]["name"] == "memory_save")
        props = memory_save["function"]["parameters"]["properties"]
        assert "facts" not in props, "Standard tools should not have 'facts'"
        assert "extracted_entities" not in props, (
            "Standard tools should not have 'extracted_entities'"
        )

        # Optimized tools SHOULD have facts/extracted_entities/extracted_relationships
        optimized_tools = get_memory_tools_optimized()
        memory_save_opt = next(t for t in optimized_tools if t["function"]["name"] == "memory_save")
        props_opt = memory_save_opt["function"]["parameters"]["properties"]
        assert "facts" in props_opt, "Optimized tools should have 'facts'"
        assert "extracted_entities" in props_opt, "Optimized tools should have 'extracted_entities'"
        assert "extracted_relationships" in props_opt, (
            "Optimized tools should have 'extracted_relationships'"
        )
        assert "background" in props_opt, "Optimized tools should have 'background'"

    # =========================================================================
    # Test 2: Verify wrapper uses correct tools based on optimized flag
    # =========================================================================

    def test_wrapper_uses_correct_tools(self, openai_client, temp_db_path, user_id):
        """Verify wrapper uses standard vs optimized tools correctly."""
        from headroom.memory import with_memory_tools
        from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

        config = LocalBackendConfig(db_path=temp_db_path)
        backend = LocalBackend(config)

        # Create non-optimized wrapper
        wrapper_standard = with_memory_tools(
            openai_client, backend=backend, user_id=user_id, optimized=False
        )

        # Create optimized wrapper
        wrapper_optimized = with_memory_tools(
            openai_client, backend=backend, user_id=user_id, optimized=True
        )

        # Verify internal flags are set correctly
        assert wrapper_standard._optimized is False
        assert wrapper_optimized._optimized is True
        assert wrapper_optimized._inject_extraction_prompt is True

    # =========================================================================
    # Test 3: Verify extraction prompt is injected in optimized mode
    # =========================================================================

    def test_extraction_prompt_injection(self, openai_client, temp_db_path, user_id):
        """Verify extraction prompt is injected into system message."""
        from headroom.memory import with_memory_tools
        from headroom.memory.backends.local import LocalBackend, LocalBackendConfig
        from headroom.memory.extraction import EXTRACTION_SYSTEM_PROMPT

        config = LocalBackendConfig(db_path=temp_db_path)
        backend = LocalBackend(config)

        wrapper = with_memory_tools(
            openai_client,
            backend=backend,
            user_id=user_id,
            optimized=True,
            inject_extraction_prompt=True,
        )

        # Get the completions object
        completions = wrapper.chat.completions

        # Test _prepare_messages with existing system message
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        prepared = completions._prepare_messages(messages)

        # Verify system message has extraction prompt appended
        assert len(prepared) == 2
        assert EXTRACTION_SYSTEM_PROMPT in prepared[0]["content"]
        assert "You are a helpful assistant." in prepared[0]["content"]

        # Test _prepare_messages without existing system message
        messages_no_system = [{"role": "user", "content": "Hello"}]
        prepared_no_system = completions._prepare_messages(messages_no_system)

        # Verify system message was inserted
        assert len(prepared_no_system) == 2
        assert prepared_no_system[0]["role"] == "system"
        assert EXTRACTION_SYSTEM_PROMPT.strip() in prepared_no_system[0]["content"]

    # =========================================================================
    # Test 4: LocalBackend accepts pre-extraction fields
    # =========================================================================

    @pytest.mark.asyncio
    async def test_local_backend_pre_extraction(self, temp_db_path, user_id):
        """Test LocalBackend save_memory with pre-extraction fields."""
        from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

        config = LocalBackendConfig(db_path=temp_db_path)
        backend = LocalBackend(config)

        # Save with pre-extraction fields
        # Note: relationships must reference entities that are in extracted_entities
        memory = await backend.save_memory(
            content="John works at Netflix using Python and TensorFlow.",
            user_id=user_id,
            importance=0.8,
            facts=["John works at Netflix", "John uses Python", "John uses TensorFlow"],
            extracted_entities=[
                {"entity": "John", "entity_type": "person"},
                {"entity": "Netflix", "entity_type": "organization"},
                {"entity": "Python", "entity_type": "technology"},
                {"entity": "TensorFlow", "entity_type": "technology"},
            ],
            extracted_relationships=[
                {
                    "source": "John",
                    "relationship": "works_at",
                    "destination": "Netflix",
                },
                {"source": "John", "relationship": "uses", "destination": "Python"},
                {
                    "source": "John",
                    "relationship": "uses",
                    "destination": "TensorFlow",
                },
            ],
        )

        # Verify memory was created
        assert memory is not None
        assert memory.user_id == user_id
        assert memory.metadata.get("_pre_extracted") is True
        assert memory.metadata.get("_fact_count") == 3

        # Verify entities were added to graph
        graph = await backend.get_graph()
        netflix_entity = await graph.get_entity_by_name(user_id, "Netflix")
        assert netflix_entity is not None
        assert netflix_entity.entity_type == "organization"

        python_entity = await graph.get_entity_by_name(user_id, "Python")
        assert python_entity is not None
        assert python_entity.entity_type == "technology"

        john_entity = await graph.get_entity_by_name(user_id, "John")
        assert john_entity is not None
        assert john_entity.entity_type == "person"

        # Verify relationships were added by querying via public API
        from headroom.memory.adapters.graph_models import RelationshipDirection

        # Verify John has outgoing relationships
        john_id = john_entity.id
        john_rels = await graph.get_relationships(john_id, RelationshipDirection.OUTGOING)
        assert len(john_rels) >= 3, (
            f"Expected John to have at least 3 outgoing relationships, got {len(john_rels)}"
        )

        await backend.close()

    # =========================================================================
    # Test 5: End-to-end with real LLM - Standard Mode
    # =========================================================================

    def test_e2e_standard_mode_llm_call(self, openai_client, temp_db_path, user_id):
        """Test end-to-end flow with real LLM call in standard mode."""
        from headroom.memory import with_memory_tools
        from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

        config = LocalBackendConfig(db_path=temp_db_path)
        backend = LocalBackend(config)

        client = with_memory_tools(
            openai_client,
            backend=backend,
            user_id=user_id,
            optimized=False,  # Standard mode
        )

        # Make a real LLM call that should trigger memory_save
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that remembers important user information. When the user shares personal information, save it to memory using the memory_save tool.",
                },
                {
                    "role": "user",
                    "content": "Hi! My name is Alex and I work as a data scientist at Google.",
                },
            ],
        )

        # Verify response was generated
        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0

        # Check if memory tool was called
        message = response.choices[0].message
        if message.tool_calls:
            # Verify memory_save was called
            tool_names = [tc.function.name for tc in message.tool_calls]
            print(f"Tools called: {tool_names}")

            # Check if auto-handled
            if hasattr(response, "_memory_tool_results"):
                print(f"Memory tool results: {response._memory_tool_results}")
                assert len(response._memory_tool_results) > 0

    # =========================================================================
    # Test 6: End-to-end with real LLM - Optimized Mode
    # =========================================================================

    def test_e2e_optimized_mode_llm_call(self, openai_client, temp_db_path, user_id):
        """Test end-to-end flow with real LLM call in optimized mode."""
        from headroom.memory import with_memory_tools
        from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

        config = LocalBackendConfig(db_path=temp_db_path)
        backend = LocalBackend(config)

        client = with_memory_tools(
            openai_client,
            backend=backend,
            user_id=user_id,
            optimized=True,  # Optimized mode - should extract facts/entities
        )

        # Make a real LLM call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "I'm Sarah, a software engineer at Microsoft. I use Python, React, and PostgreSQL daily.",
                },
            ],
        )

        # Verify response was generated
        assert response is not None
        assert response.choices is not None

        # Check if memory tool was called with pre-extraction
        message = response.choices[0].message
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc.function.name == "memory_save":
                    import json

                    args = json.loads(tc.function.arguments)
                    print(f"memory_save arguments: {json.dumps(args, indent=2)}")

                    # In optimized mode, LLM SHOULD include facts/entities
                    # (depends on LLM following the extraction prompt)
                    if "facts" in args:
                        print(f"Pre-extracted facts: {args['facts']}")
                    if "extracted_entities" in args:
                        print(f"Pre-extracted entities: {args['extracted_entities']}")
                    if "extracted_relationships" in args:
                        print(f"Pre-extracted relationships: {args['extracted_relationships']}")

        # Check auto-handled results
        if hasattr(response, "_memory_tool_results"):
            print(f"Memory tool results: {response._memory_tool_results}")

    # =========================================================================
    # Test 7: Verify memory search works after save
    # =========================================================================

    @pytest.mark.asyncio
    async def test_memory_search_after_save(self, temp_db_path, user_id):
        """Test that saved memories can be searched."""
        from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

        config = LocalBackendConfig(db_path=temp_db_path)
        backend = LocalBackend(config)

        # Save some memories
        await backend.save_memory(
            content="User prefers Python for backend development",
            user_id=user_id,
            importance=0.9,
            entities=["Python"],
            extracted_entities=[{"entity": "Python", "entity_type": "technology"}],
        )

        await backend.save_memory(
            content="User works at Netflix as a senior engineer",
            user_id=user_id,
            importance=0.8,
            entities=["Netflix"],
            extracted_entities=[{"entity": "Netflix", "entity_type": "organization"}],
        )

        # Search for memories
        results = await backend.search_memories(
            query="What programming language does the user prefer?",
            user_id=user_id,
            top_k=5,
        )

        assert len(results) > 0, "Expected at least one search result"
        print(f"Search results: {[(r.memory.content, r.score) for r in results]}")

        # Search with entity filter
        results_netflix = await backend.search_memories(
            query="Where does the user work?",
            user_id=user_id,
            entities=["Netflix"],
            top_k=5,
        )

        # Should find the Netflix-related memory
        assert any("Netflix" in r.memory.content for r in results_netflix), (
            "Expected Netflix in results"
        )

        await backend.close()

    # =========================================================================
    # Test 8: Test include_related graph expansion
    # =========================================================================

    @pytest.mark.asyncio
    async def test_include_related_graph_expansion(self, temp_db_path, user_id):
        """Test that include_related expands results via graph."""
        from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

        config = LocalBackendConfig(db_path=temp_db_path)
        backend = LocalBackend(config)

        # Save memories with related entities
        await backend.save_memory(
            content="Alice is a data scientist",
            user_id=user_id,
            importance=0.8,
            entities=["Alice"],
            extracted_entities=[{"entity": "Alice", "entity_type": "person"}],
        )

        await backend.save_memory(
            content="Alice works at Acme Corp",
            user_id=user_id,
            importance=0.8,
            entities=["Alice", "Acme Corp"],
            extracted_entities=[
                {"entity": "Alice", "entity_type": "person"},
                {"entity": "Acme Corp", "entity_type": "organization"},
            ],
            extracted_relationships=[
                {
                    "source": "Alice",
                    "relationship": "works_at",
                    "destination": "Acme Corp",
                }
            ],
        )

        await backend.save_memory(
            content="Acme Corp is a tech company in San Francisco",
            user_id=user_id,
            importance=0.7,
            entities=["Acme Corp", "San Francisco"],
            extracted_entities=[
                {"entity": "Acme Corp", "entity_type": "organization"},
                {"entity": "San Francisco", "entity_type": "location"},
            ],
        )

        # Search for Alice - should expand to related memories via graph
        results_with_related = await backend.search_memories(
            query="Tell me about Alice",
            user_id=user_id,
            top_k=10,
            include_related=True,
        )

        # Search without related
        results_without_related = await backend.search_memories(
            query="Tell me about Alice",
            user_id=user_id,
            top_k=10,
            include_related=False,
        )

        print(f"With related: {[r.memory.content for r in results_with_related]}")
        print(f"Without related: {[r.memory.content for r in results_without_related]}")

        # With related should potentially include the Acme Corp memory via Alice connection
        # (This depends on graph expansion finding the connection)
        assert len(results_with_related) >= len(results_without_related), (
            "include_related should return same or more results"
        )

        await backend.close()

    # =========================================================================
    # Test 9: Test MemorySystem tool dispatch
    # =========================================================================

    @pytest.mark.asyncio
    async def test_memory_system_tool_dispatch(self, temp_db_path, user_id):
        """Test MemorySystem processes tool calls correctly."""
        from headroom.memory.backends.local import LocalBackend, LocalBackendConfig
        from headroom.memory.system import MemorySystem

        config = LocalBackendConfig(db_path=temp_db_path)
        backend = LocalBackend(config)
        system = MemorySystem(backend, user_id=user_id)

        # Test memory_save dispatch
        save_result = await system.process_tool_call(
            "memory_save",
            {
                "content": "User likes dark mode",
                "importance": 0.7,
                "facts": ["Prefers dark mode"],
                "extracted_entities": [{"entity": "dark mode", "entity_type": "preference"}],
            },
        )

        assert save_result["success"] is True
        assert "memory_id" in save_result or "data" in save_result
        print(f"Save result: {save_result}")

        # Test memory_search dispatch
        search_result = await system.process_tool_call(
            "memory_search", {"query": "dark mode preferences", "top_k": 5}
        )

        assert search_result["success"] is True
        print(f"Search result: {search_result}")

        await backend.close()

    # =========================================================================
    # Test 10: Full flow - LLM saves, then retrieves via search
    # =========================================================================

    def test_full_flow_save_then_search(self, openai_client, temp_db_path, user_id):
        """Test complete flow: LLM saves memory, then searches for it."""
        import json

        from headroom.memory import with_memory_tools
        from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

        config = LocalBackendConfig(db_path=temp_db_path)
        backend = LocalBackend(config)

        client = with_memory_tools(
            openai_client,
            backend=backend,
            user_id=user_id,
            optimized=True,
        )

        # First: Have LLM save some information
        save_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Remember this: My favorite programming language is Rust and I'm working on a CLI tool called headroom.",
                },
            ],
        )

        print(f"Save response: {save_response.choices[0].message}")

        # Process tool calls if any
        if save_response.choices[0].message.tool_calls:
            print(
                f"Tool calls made: {[tc.function.name for tc in save_response.choices[0].message.tool_calls]}"
            )
            if hasattr(save_response, "_memory_tool_results"):
                print(f"Results: {save_response._memory_tool_results}")

        # Second: Ask LLM to recall the information
        recall_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "What is my favorite programming language? Search your memory.",
                },
            ],
        )

        print(f"Recall response: {recall_response.choices[0].message}")

        # Check if search was invoked
        if recall_response.choices[0].message.tool_calls:
            for tc in recall_response.choices[0].message.tool_calls:
                print(f"Tool: {tc.function.name}, Args: {tc.function.arguments}")
                if hasattr(recall_response, "_memory_tool_results"):
                    results = recall_response._memory_tool_results.get(tc.id, {})
                    print(f"Tool result: {json.dumps(results, indent=2, default=str)}")


class TestExtractionPrompts:
    """Tests for extraction prompt templates."""

    def test_extraction_prompts_exist_and_valid(self):
        """Verify extraction prompts are defined and non-empty."""
        from headroom.memory.extraction import (
            ENTITY_EXTRACTION_PROMPT,
            EXTRACTION_SYSTEM_PROMPT,
            FACT_EXTRACTION_PROMPT,
            RELATIONSHIP_EXTRACTION_PROMPT,
        )

        assert len(EXTRACTION_SYSTEM_PROMPT) > 100, "System prompt should be substantial"
        assert len(FACT_EXTRACTION_PROMPT) > 100, "Fact prompt should be substantial"
        assert len(ENTITY_EXTRACTION_PROMPT) > 100, "Entity prompt should be substantial"
        assert len(RELATIONSHIP_EXTRACTION_PROMPT) > 100, (
            "Relationship prompt should be substantial"
        )

        # Verify they mention key concepts
        assert "facts" in EXTRACTION_SYSTEM_PROMPT.lower()
        assert "entities" in EXTRACTION_SYSTEM_PROMPT.lower()
        assert "relationships" in EXTRACTION_SYSTEM_PROMPT.lower()


class TestWrapperToolsModule:
    """Tests for wrapper_tools.py module."""

    def test_wrapper_tools_imports(self):
        """Verify all necessary imports work."""
        from headroom.memory.wrapper_tools import (
            MemoryToolsChatCompletions,
            MemoryToolsCompletions,
            MemoryToolsWrapper,
            with_memory_tools,
        )

        assert with_memory_tools is not None
        assert MemoryToolsWrapper is not None
        assert MemoryToolsChatCompletions is not None
        assert MemoryToolsCompletions is not None

    def test_with_memory_tools_accepts_optimized_param(self):
        """Verify with_memory_tools accepts optimized parameter."""
        import inspect

        from headroom.memory.wrapper_tools import with_memory_tools

        sig = inspect.signature(with_memory_tools)
        params = list(sig.parameters.keys())

        assert "optimized" in params, "with_memory_tools should accept 'optimized' param"
        assert "inject_extraction_prompt" in params, (
            "with_memory_tools should accept 'inject_extraction_prompt' param"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
