"""Integration tests for proxy memory system with real API calls.

These tests require:
- ANTHROPIC_API_KEY environment variable set

Run with:
    ANTHROPIC_API_KEY=... uv run pytest tests/test_proxy_memory_integration.py -v

Test categories:
- TestMemoryHeaderValidation: User ID header validation
- TestMemoryToolInjection: Memory tools are injected
- TestMemorySaveAndSearch: End-to-end save/recall flow
- TestMemoryUserIsolation: User memory isolation
"""

import os
import tempfile
import time
from pathlib import Path

import pytest

# Set tokenizer parallelism before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


@pytest.fixture
def temp_memory_db():
    """Create temporary memory database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    # Cleanup
    Path(f.name).unlink(missing_ok=True)
    # Also cleanup related files (HNSW index, etc.)
    for suffix in ["-shm", "-wal", ".hnsw"]:
        Path(f.name + suffix).unlink(missing_ok=True)


@pytest.fixture
def memory_client(temp_memory_db):
    """Create test client with memory enabled."""
    config = ProxyConfig(
        optimize=False,  # Disable optimization for simpler tests
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        memory_enabled=True,
        memory_backend="local",
        memory_db_path=temp_memory_db,
        memory_inject_tools=True,
        memory_inject_context=True,
        memory_top_k=5,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def no_memory_client():
    """Create test client with memory disabled."""
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        memory_enabled=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def anthropic_api_key():
    """Get Anthropic API key from environment."""
    return os.environ.get("ANTHROPIC_API_KEY")


class TestMemoryHeaderValidation:
    """Test user ID header validation."""

    def test_missing_user_id_uses_default(self, memory_client, anthropic_api_key):
        """Request without x-headroom-user-id should use 'default' user for simple DevEx."""
        if not anthropic_api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        response = memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                # Note: NOT setting x-headroom-user-id - should default to "default"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        # Should succeed, not return 400
        assert response.status_code == 200

    def test_with_user_id_succeeds(self, memory_client, anthropic_api_key):
        """Request with x-headroom-user-id should succeed."""
        if not anthropic_api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        response = memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": "test-user-123",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello, just say hi back."}],
            },
        )
        assert response.status_code == 200

    def test_no_memory_client_doesnt_require_user_id(self, no_memory_client, anthropic_api_key):
        """When memory is disabled, user ID header should not be required."""
        if not anthropic_api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        response = no_memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                # No x-headroom-user-id
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello, just say hi."}],
            },
        )
        assert response.status_code == 200


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestMemoryToolInjection:
    """Test memory tool injection."""

    def test_memory_tools_are_available(self, memory_client, anthropic_api_key):
        """Memory tools should be available to the LLM."""
        response = memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": "test-user-tool-check",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "messages": [
                    {
                        "role": "user",
                        "content": "List the tools available to you. Just list the tool names.",
                    }
                ],
            },
        )
        assert response.status_code == 200

        # The response should mention memory tools
        content = response.json().get("content", [])
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")

        # At least one memory tool should be mentioned
        assert any(tool in text.lower() for tool in ["memory_save", "memory_search", "memory"]), (
            f"Memory tools not found in response: {text}"
        )


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestMemorySaveAndSearch:
    """Test memory save and search flow."""

    def test_save_memory_via_explicit_instruction(self, memory_client, anthropic_api_key):
        """LLM should be able to save memories when instructed."""
        user_id = f"test-user-save-{int(time.time())}"

        # Request that explicitly asks to save
        response = memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": user_id,
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": "Please save this to memory: My favorite programming language is Rust. "
                        "Use the memory_save tool to save this information.",
                    }
                ],
            },
        )
        assert response.status_code == 200

        # Check if response indicates tool was used
        resp_json = response.json()
        content = resp_json.get("content", [])

        # Response could be tool_use (if not handled) or text (if handled)
        # Either way, it should complete successfully
        assert content, "Response should have content"

    def test_save_and_recall_memory(self, memory_client, anthropic_api_key):
        """Save a memory and recall it in subsequent request."""
        user_id = f"test-user-recall-{int(time.time())}"

        # First request: save a memory with explicit instruction
        save_response = memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": user_id,
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": "Please remember this: My name is TestUser and I work at AcmeCorp. "
                        "Save this information using the memory_save tool.",
                    }
                ],
            },
        )
        assert save_response.status_code == 200

        # Wait a moment for memory to be indexed
        time.sleep(1)

        # Second request: ask about saved info
        # Memory context should be injected automatically
        recall_response = memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": user_id,
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 300,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is my name and where do I work? "
                        "Answer based on what you know about me.",
                    }
                ],
            },
        )
        assert recall_response.status_code == 200

        # Check if response mentions the saved info
        content = recall_response.json().get("content", [])
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")

        # Should mention at least one of the saved facts
        text_lower = text.lower()
        assert "testuser" in text_lower or "acmecorp" in text_lower or "acme" in text_lower, (
            f"Saved info not recalled: {text}"
        )


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestMemoryUserIsolation:
    """Test that memories are isolated per user."""

    def test_different_users_have_isolated_memories(self, memory_client, anthropic_api_key):
        """User A's memories should not appear for User B."""
        timestamp = int(time.time())
        user_a = f"user-a-isolation-{timestamp}"
        user_b = f"user-b-isolation-{timestamp}"
        secret_code = f"SECRETCODE{timestamp}"

        # Save memory for user A
        save_response = memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": user_a,
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Remember my secret code: {secret_code}. "
                        "Save this using the memory_save tool.",
                    }
                ],
            },
        )
        assert save_response.status_code == 200

        # Wait for memory to be indexed
        time.sleep(1)

        # Query as user B - should NOT have access to user A's memory
        response_b = memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": user_b,
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 300,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is my secret code? Search your memory for it.",
                    }
                ],
            },
        )
        assert response_b.status_code == 200

        # User B should NOT see user A's secret code
        content = response_b.json().get("content", [])
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")

        assert secret_code not in text, f"User B should not see User A's secret: {text}"


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestMemoryStats:
    """Test memory-related stats and health."""

    def test_health_endpoint_works_with_memory(self, memory_client):
        """Health endpoint should work when memory is enabled."""
        response = memory_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"

    def test_stats_endpoint_works_with_memory(self, memory_client):
        """Stats endpoint should work when memory is enabled."""
        response = memory_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "requests" in data


@pytest.fixture
def memory_client_global(temp_memory_db):
    """Memory-enabled client with GLOBAL storage mode.

    GLOBAL keeps every memory in a single SQLite file regardless of
    project routing, so tests that pre-seed via direct backend access
    are guaranteed to share the same DB as the proxy's runtime backend.
    """
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        memory_enabled=True,
        memory_backend="local",
        memory_db_path=temp_memory_db,
        memory_inject_tools=True,
        memory_inject_context=True,
        memory_top_k=5,
        memory_storage_mode="global",
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Helpers shared by the live-API tests below. Each live test follows the same
# three-step shape:
#   1. seed a memory with known content via a fresh ONNX-backed LocalBackend
#      pointed at the same db_path as the proxy backend (so the proxy reads
#      our row, and we know its exact ID up front),
#   2. install a recorder that captures every memory tool call the proxy
#      dispatches downstream of the model's tool_use blocks,
#   3. make a real Anthropic API request via TestClient and assert the
#      recorded calls match the expected verb + memory_id contract.
#
# The helpers below factor out (1) and (2) so each test body reads as the
# one-line intent it actually is.
# ---------------------------------------------------------------------------


def _seed_memory(*, db_path: str, user_id: str, content: str) -> str:
    """Save ``content`` for ``user_id`` via a fresh ONNX-backed LocalBackend
    pointed at ``db_path``. Returns the new memory's ID."""
    import asyncio

    from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

    async def _run() -> str:
        backend = LocalBackend(
            LocalBackendConfig(
                db_path=db_path,
                embedder_backend="onnx",
                embedder_model="all-MiniLM-L6-v2",
                vector_dimension=384,
            )
        )
        mem = await backend.save_memory(content=content, user_id=user_id)
        return mem.id

    mem_id = asyncio.run(_run())
    assert mem_id, "save_memory should return a usable id"
    return mem_id


def _install_tool_call_recorder(handler):
    """Monkey-patch ``handler._execute_memory_tool`` to record every call.

    Each recorded entry has ``tool_name``, ``input``, and ``result`` so the
    test can assert both what the model called AND what the proxy returned
    back to it (the dedup-hint path lives in the latter).

    Returns ``(recorded_list, restore_callable)``. Call ``restore_callable()``
    in a ``finally`` to put the original method back, no matter what the
    request body does."""
    recorded: list[dict] = []
    original_execute = handler._execute_memory_tool

    async def _capturing_execute(
        tool_name, input_data, user_id_arg, provider, request_context=None
    ):
        result = await original_execute(
            tool_name,
            input_data,
            user_id_arg,
            provider,
            request_context=request_context,
        )
        recorded.append({"tool_name": tool_name, "input": dict(input_data), "result": result})
        return result

    handler._execute_memory_tool = _capturing_execute  # type: ignore[assignment]

    def _restore() -> None:
        handler._execute_memory_tool = original_execute  # type: ignore[assignment]

    return recorded, _restore


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestMemoryIdAutoTailAndUpdate:
    """End-to-end live: model uses [memory_id] from auto-tail to call memory_update.

    Validates that the IDs we added to the auto-injected memory block
    (see ``MemoryHandler.search_and_format_context``) are extractable by
    a real Claude model and can be passed directly to ``memory_update``
    without an intervening ``memory_search`` round-trip.
    """

    def test_model_uses_memory_id_to_call_memory_update(
        self,
        memory_client_global,
        anthropic_api_key,
        temp_memory_db,
    ):
        memory_id = _seed_memory(
            db_path=temp_memory_db,
            user_id=(user_id := f"test-id-update-{int(time.time())}"),
            content="The user's favorite color is blue.",
        )
        # Let the SQLite write + index settle before the proxy reads.
        time.sleep(0.5)

        proxy = memory_client_global.app.state.proxy
        assert proxy.memory_handler is not None
        recorded, restore = _install_tool_call_recorder(proxy.memory_handler)

        try:
            response = memory_client_global.post(
                "/v1/messages",
                headers={
                    "x-api-key": anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "x-headroom-user-id": user_id,
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 800,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Quick correction: my favorite color is actually "
                                "green, not blue. Please call the memory_update "
                                "tool to fix the relevant memory in your context. "
                                "The relevant memories block lists each memory's "
                                "ID in square brackets — use that ID for "
                                "memory_id."
                            ),
                        }
                    ],
                },
            )
        finally:
            restore()

        assert response.status_code == 200, response.text

        # The model should have called memory_update at least once.
        update_calls = [c for c in recorded if c["tool_name"] == "memory_update"]
        assert update_calls, f"Expected at least one memory_update call. Recorded: {recorded}"

        # And it should reference the exact ID we seeded — i.e. the
        # model used the [id] from the auto-tail block, not a guess.
        assert any(c["input"].get("memory_id") == memory_id for c in update_calls), (
            f"Expected memory_update(memory_id={memory_id!r}); got inputs: "
            f"{[c['input'] for c in update_calls]}"
        )

    def test_model_uses_memory_id_to_call_memory_delete(
        self,
        memory_client_global,
        anthropic_api_key,
        temp_memory_db,
    ):
        """Same [id] handle, different destructive verb. Verifies the
        auto-tail bracketed ID is usable for memory_delete just as it
        is for memory_update — i.e. the handle is verb-agnostic."""

        memory_id = _seed_memory(
            db_path=temp_memory_db,
            user_id=(user_id := f"test-id-delete-{int(time.time())}"),
            content="The user used to work at AcmeCorp until 2024.",
        )
        time.sleep(0.5)

        proxy = memory_client_global.app.state.proxy
        assert proxy.memory_handler is not None
        recorded, restore = _install_tool_call_recorder(proxy.memory_handler)

        try:
            response = memory_client_global.post(
                "/v1/messages",
                headers={
                    "x-api-key": anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "x-headroom-user-id": user_id,
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 800,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Please remove the memory about where I used to "
                                "work (AcmeCorp). Call memory_delete directly — "
                                "do NOT call memory_search or memory_list first. "
                                "The memory's ID is shown in square brackets in "
                                "the relevant memories block at the end of this "
                                "message; pass that ID to memory_id."
                            ),
                        }
                    ],
                },
            )
        finally:
            restore()

        assert response.status_code == 200, response.text

        delete_calls = [c for c in recorded if c["tool_name"] == "memory_delete"]
        assert delete_calls, f"Expected at least one memory_delete call. Recorded: {recorded}"
        assert any(c["input"].get("memory_id") == memory_id for c in delete_calls), (
            f"Expected memory_delete(memory_id={memory_id!r}); got inputs: "
            f"{[c['input'] for c in delete_calls]}"
        )

    def test_dedup_hint_surfaces_seeded_id_when_memory_save_runs_on_near_duplicate(
        self,
        memory_client_global,
        anthropic_api_key,
        temp_memory_db,
    ):
        """Live verification of the memory_save → dedup-hint mechanism.

        Without this hint, ``memory_save`` on a near-duplicate would silently
        accumulate parallel rows, polluting the cache prefix and confusing
        the model on subsequent retrieval. The hint surfaces the existing
        row's ID in the tool result so the model has a directly addressable
        handle to consolidate via ``memory_update``.

        We assert the MECHANISM end-to-end:

          - Model fires ``memory_save`` on the prompted (similar) content.
          - The proxy's ``_execute_save`` returns a ``note`` containing the
            pre-seeded memory's exact ID.

        We DO NOT assert that the model actually consolidates — the hint
        text intentionally ends with "or ignore if these are distinct
        facts", so the model is free to decline. Whether it consolidates
        depends on its judgement about whether two phrasings are the same
        fact, which is intentionally outside this contract."""

        # Pre-seed a memory the new save will look semantically similar to.
        # We use content close enough that cosine similarity comfortably
        # clears DEDUP_HINT_THRESHOLD (0.75).
        seeded_id = _seed_memory(
            db_path=temp_memory_db,
            user_id=(user_id := f"test-dedup-{int(time.time())}"),
            content="The user prefers Python for data analysis work.",
        )
        time.sleep(0.5)

        proxy = memory_client_global.app.state.proxy
        assert proxy.memory_handler is not None
        recorded, restore = _install_tool_call_recorder(proxy.memory_handler)

        try:
            response = memory_client_global.post(
                "/v1/messages",
                headers={
                    "x-api-key": anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "x-headroom-user-id": user_id,
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Use memory_save DIRECTLY to store: "
                                "'User prefers Python for data science.' "
                                "Do NOT call memory_search or memory_list "
                                "first — I want to exercise the save path."
                            ),
                        }
                    ],
                },
            )
        finally:
            restore()

        assert response.status_code == 200, response.text

        # The model must have fired memory_save (we explicitly prompted
        # that path).
        save_calls = [c for c in recorded if c["tool_name"] == "memory_save"]
        assert save_calls, f"Expected memory_save call. Recorded: {recorded}"

        # The proxy's _execute_save must have returned a dedup hint
        # surfacing the seeded memory's exact ID — that's the mechanism
        # under test. The hint is a serialized JSON string with a "note"
        # field; assert the seeded ID is present in it.
        save_result = save_calls[0]["result"]
        assert isinstance(save_result, str), (
            f"Expected JSON-string tool result, got {type(save_result).__name__}: {save_result!r}"
        )
        assert "Similar memory exists" in save_result, (
            "Expected dedup hint in memory_save result (similarity should clear "
            f"DEDUP_HINT_THRESHOLD=0.75). Got: {save_result}"
        )
        assert seeded_id in save_result, (
            f"Expected dedup hint to surface seeded memory_id={seeded_id!r}; got: {save_result}"
        )
