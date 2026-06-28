"""Tests for CCR context tracker.

These tests verify that:
1. Compression events are tracked correctly
2. Query analysis finds relevant compressed contexts
3. Proactive expansion works as expected
4. LRU eviction and TTL work correctly
5. Expansion recommendations are appropriate
"""

import json
import time

import pytest

from headroom.cache.compression_store import (
    get_compression_store,
    reset_compression_store,
)
from headroom.ccr.context_tracker import (
    CompressedContext,
    ContextTracker,
    ContextTrackerConfig,
    ExpansionRecommendation,
    get_context_tracker,
    reset_context_tracker,
)


class TestContextTrackerBasics:
    """Test basic context tracking functionality."""

    @pytest.fixture(autouse=True)
    def reset_trackers(self):
        """Reset trackers before each test."""
        reset_context_tracker()
        reset_compression_store()
        yield
        reset_context_tracker()
        reset_compression_store()

    def test_track_compression(self):
        """Track a compression event."""
        tracker = ContextTracker()

        tracker.track_compression(
            hash_key="abc123",
            turn_number=1,
            tool_name="Bash",
            original_count=100,
            compressed_count=10,
            query_context="find all python files",
            sample_content='["src/main.py", "src/auth.py"]',
            workspace_key="ws-test",
        )

        assert "abc123" in tracker.get_tracked_hashes()
        stats = tracker.get_stats()
        assert stats["tracked_contexts"] == 1

    def test_track_multiple_compressions(self):
        """Track multiple compression events."""
        tracker = ContextTracker()

        for i in range(5):
            tracker.track_compression(
                hash_key=f"hash_{i}",
                turn_number=i,
                tool_name="Bash",
                original_count=100,
                compressed_count=10,
                workspace_key="ws-test",
            )

        hashes = tracker.get_tracked_hashes()
        assert len(hashes) == 5
        assert all(f"hash_{i}" in hashes for i in range(5))

    def test_tracking_disabled(self):
        """Tracking disabled doesn't store contexts."""
        config = ContextTrackerConfig(enabled=False)
        tracker = ContextTracker(config)

        tracker.track_compression(
            hash_key="abc123",
            turn_number=1,
            tool_name="Bash",
            original_count=100,
            compressed_count=10,
            workspace_key="ws-test",
        )

        assert len(tracker.get_tracked_hashes()) == 0

    def test_update_existing_hash(self):
        """Updating existing hash updates the context data."""
        tracker = ContextTracker()

        tracker.track_compression(
            hash_key="first",
            turn_number=1,
            tool_name=None,
            original_count=50,
            compressed_count=5,
            workspace_key="ws-test",
        )
        tracker.track_compression(
            hash_key="second",
            turn_number=2,
            tool_name=None,
            original_count=50,
            compressed_count=5,
            workspace_key="ws-test",
        )
        tracker.track_compression(
            hash_key="first",
            turn_number=3,
            tool_name=None,
            original_count=60,
            compressed_count=6,
            workspace_key="ws-test",
        )

        # Should still have 2 unique hashes
        hashes = tracker.get_tracked_hashes()
        assert len(hashes) == 2
        assert "first" in hashes
        assert "second" in hashes

        # The updated context should have the new values
        stats = tracker.get_stats()
        first_ctx = next(c for c in stats["contexts"] if c["hash"] == "first")
        assert first_ctx["items"] == "6/60"  # Updated values
        assert first_ctx["turn"] == 3


class TestLRUEviction:
    """Test LRU eviction at capacity."""

    def test_eviction_at_capacity(self):
        """Oldest entries evicted when at capacity."""
        config = ContextTrackerConfig(max_tracked_contexts=3)
        tracker = ContextTracker(config)

        for i in range(5):
            tracker.track_compression(
                hash_key=f"hash_{i}",
                turn_number=i,
                tool_name=None,
                original_count=100,
                compressed_count=10,
                workspace_key="ws-test",
            )
            time.sleep(0.01)  # Ensure different timestamps

        hashes = tracker.get_tracked_hashes()
        assert len(hashes) == 3
        # Should have the last 3
        assert "hash_2" in hashes
        assert "hash_3" in hashes
        assert "hash_4" in hashes
        # First 2 should be evicted
        assert "hash_0" not in hashes
        assert "hash_1" not in hashes


class TestQueryAnalysis:
    """Test query analysis for relevance detection."""

    @pytest.fixture(autouse=True)
    def reset_trackers(self):
        """Reset trackers before each test."""
        reset_context_tracker()
        reset_compression_store()
        yield
        reset_context_tracker()
        reset_compression_store()

    def test_analyze_query_finds_relevant_context(self):
        """Query analysis finds relevant compressed context."""
        # Use lower relevance threshold to make test more reliable
        config = ContextTrackerConfig(relevance_threshold=0.1)
        tracker = ContextTracker(config)

        tracker.track_compression(
            hash_key="auth_hash",
            turn_number=1,
            tool_name="Bash",
            original_count=100,
            compressed_count=10,
            query_context="find authentication files",
            # Use more explicit content with keywords that will match
            sample_content="authentication middleware handler login security",
            workspace_key="ws-test",
        )

        recommendations = tracker.analyze_query(
            query="show authentication middleware", current_turn=2, workspace_key="ws-test"
        )

        assert len(recommendations) >= 1
        assert recommendations[0].hash_key == "auth_hash"

    def test_analyze_query_no_match(self):
        """Query analysis returns empty for unrelated query."""
        tracker = ContextTracker()

        tracker.track_compression(
            hash_key="db_hash",
            turn_number=1,
            tool_name="Bash",
            original_count=100,
            compressed_count=10,
            query_context="find database files",
            sample_content='["database.py", "models.py"]',
            workspace_key="ws-test",
        )

        recommendations = tracker.analyze_query(
            query="What is the weather like?", current_turn=2, workspace_key="ws-test"
        )

        # Should not match unrelated query
        assert len(recommendations) == 0

    def test_analyze_query_keyword_overlap(self):
        """Query matches based on keyword overlap."""
        tracker = ContextTracker()

        tracker.track_compression(
            hash_key="python_files",
            turn_number=1,
            tool_name="Glob",
            original_count=200,
            compressed_count=20,
            query_context="find python files",
            sample_content='["main.py", "utils.py", "config.py", "test_main.py"]',
            workspace_key="ws-test",
        )

        # Query with overlapping keywords
        recommendations = tracker.analyze_query(
            query="Show me the main python file", current_turn=2, workspace_key="ws-test"
        )

        assert len(recommendations) >= 1

    def test_analyze_query_proactive_disabled(self):
        """No recommendations when proactive expansion disabled."""
        config = ContextTrackerConfig(proactive_expansion=False)
        tracker = ContextTracker(config)

        tracker.track_compression(
            hash_key="abc123",
            turn_number=1,
            tool_name="Bash",
            original_count=100,
            compressed_count=10,
            sample_content='["relevant.py"]',
            workspace_key="ws-test",
        )

        recommendations = tracker.analyze_query(
            query="Show me relevant files", current_turn=2, workspace_key="ws-test"
        )

        assert len(recommendations) == 0

    def test_analyze_query_respects_age(self):
        """Old contexts get lower relevance scores."""
        config = ContextTrackerConfig(max_context_age_seconds=2.0)
        tracker = ContextTracker(config)

        tracker.track_compression(
            hash_key="old_context",
            turn_number=1,
            tool_name="Bash",
            original_count=100,
            compressed_count=10,
            sample_content='["auth.py"]',
            workspace_key="ws-test",
        )

        # Wait for context to age
        time.sleep(2.1)

        recommendations = tracker.analyze_query(
            query="Show me the authentication code", current_turn=5, workspace_key="ws-test"
        )

        # Should not recommend aged-out context
        assert len(recommendations) == 0

    def test_analyze_query_max_recommendations(self):
        """Respects max proactive expansions limit."""
        config = ContextTrackerConfig(max_proactive_expansions=2)
        tracker = ContextTracker(config)

        # Track many relevant contexts
        for i in range(5):
            tracker.track_compression(
                hash_key=f"hash_{i}",
                turn_number=i,
                tool_name="Bash",
                original_count=100,
                compressed_count=10,
                sample_content=f'["python_{i}.py", "main.py"]',
                workspace_key="ws-test",
            )

        recommendations = tracker.analyze_query(
            query="Show me the python main file", current_turn=10, workspace_key="ws-test"
        )

        assert len(recommendations) <= 2


class TestRelevanceCalculation:
    """Test relevance score calculation."""

    def test_extract_keywords(self):
        """Keywords are extracted correctly."""
        tracker = ContextTracker()

        keywords = tracker._extract_keywords("Find authentication middleware files")

        assert "authentication" in keywords
        assert "middleware" in keywords
        assert "files" in keywords
        # Stop words should be filtered
        assert "the" not in keywords

    def test_exact_substring_match_bonus(self):
        """Exact substring matches get bonus score."""
        tracker = ContextTracker()

        tracker.track_compression(
            hash_key="exact_match",
            turn_number=1,
            tool_name=None,
            original_count=100,
            compressed_count=10,
            sample_content="authentication_middleware.py, auth_handler.py",
            workspace_key="ws-test",
        )

        # Query with exact substring match
        recommendations = tracker.analyze_query(
            query="authentication middleware", current_turn=2, workspace_key="ws-test"
        )

        assert len(recommendations) >= 1
        # Should have high relevance
        assert recommendations[0].relevance_score > 0.3


class TestExpansionExecution:
    """Test execution of expansion recommendations."""

    @pytest.fixture(autouse=True)
    def reset_stores(self):
        """Reset stores before each test."""
        reset_context_tracker()
        reset_compression_store()
        yield
        reset_context_tracker()
        reset_compression_store()

    def test_execute_full_expansion(self):
        """Execute full expansion retrieval."""
        store = get_compression_store()
        original = json.dumps([{"id": i} for i in range(100)])

        hash_key = store.store(
            original=original,
            compressed="[]",
            original_item_count=100,
        )

        tracker = ContextTracker()
        recommendations = [
            ExpansionRecommendation(
                hash_key=hash_key,
                reason="relevant to query",
                relevance_score=0.8,
            )
        ]

        results = tracker.execute_expansions(recommendations)

        assert len(results) == 1
        assert results[0]["type"] == "full"
        assert results[0]["item_count"] == 100

    def test_execute_nonexistent_hash(self):
        """Handle expansion of nonexistent hash gracefully."""
        tracker = ContextTracker()
        recommendations = [
            ExpansionRecommendation(
                hash_key="nonexistent123",
                reason="test",
                relevance_score=0.5,
            )
        ]

        results = tracker.execute_expansions(recommendations)

        # Should handle gracefully, no results
        assert len(results) == 0


class TestExpansionFormatting:
    """Test formatting of expansions for context."""

    def test_format_full_expansion(self):
        """Format full expansion for LLM context."""
        tracker = ContextTracker()

        expansions = [
            {
                "hash": "abc123",
                "type": "full",
                "content": '[{"id": 1}, {"id": 2}]',
                "item_count": 2,
                "reason": "relevant to query",
            }
        ]

        formatted = tracker.format_expansions_for_context(expansions)

        assert "[Proactive Context Expansion" in formatted
        assert "Expanded from earlier" in formatted
        assert '[{"id": 1}, {"id": 2}]' in formatted
        assert formatted.startswith("<headroom_proactive_expansion>\n")
        assert formatted.endswith("\n</headroom_proactive_expansion>")

    def test_format_empty_expansions(self):
        """Empty expansions return empty string."""
        tracker = ContextTracker()

        formatted = tracker.format_expansions_for_context([])

        assert formatted == ""

    def test_format_expansion_xml_wrapper(self):
        """Expansion output is wrapped in machine-readable XML provenance tag."""
        tracker = ContextTracker()
        expansions = [
            {
                "hash": "h1",
                "type": "full",
                "content": "expanded content",
                "item_count": 1,
                "reason": "high relevance",
            }
        ]
        result = tracker.format_expansions_for_context(expansions)
        assert result.startswith("<headroom_proactive_expansion>\n")
        assert result.endswith("\n</headroom_proactive_expansion>")
        assert "[Proactive Context Expansion" in result
        assert result.count("[End Proactive Expansion]") == 1

    def test_proactive_expansion_identifiable_after_injection(self):
        """Injected expansion carries XML provenance tag after full injection chain."""
        from headroom.ccr.context_tracker import ContextTracker
        from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

        tracker = ContextTracker()
        expansions = [
            {
                "hash": "h1",
                "type": "full",
                "content": "expanded context",
                "item_count": 1,
                "reason": "high relevance",
            }
        ]
        expansion_text = tracker.format_expansions_for_context(expansions)

        # Simulate a user turn that contains peer_turn markup
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<peer_turn from='AgentX'>some content</peer_turn>"}
                ],
            }
        ]
        result = AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
            messages, expansion_text, frozen_message_count=0
        )
        injected = result[0]["content"][0]["text"]
        # Headroom-injected content is identifiable by XML tag, distinct from peer content
        assert "<headroom_proactive_expansion>" in injected
        assert "</headroom_proactive_expansion>" in injected
        assert "<peer_turn from='AgentX'>" in injected  # peer content unchanged

    def test_format_expansion_xml_close_tag_in_payload_escaped(self):
        """Payload containing the XML close tag is escaped to keep wrapper boundaries intact."""
        tracker = ContextTracker()
        expansions = [
            {
                "hash": "h1",
                "type": "full",
                "content": "return '</headroom_proactive_expansion>'",
                "item_count": 1,
                "reason": "high relevance",
            }
        ]
        result = tracker.format_expansions_for_context(expansions)
        assert result.startswith("<headroom_proactive_expansion>\n")
        assert result.endswith("\n</headroom_proactive_expansion>")
        assert result.count("<headroom_proactive_expansion>") == 1
        assert result.count("</headroom_proactive_expansion>") == 1


class TestGlobalTracker:
    """Test global tracker singleton."""

    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        """Reset global tracker."""
        reset_context_tracker()
        yield
        reset_context_tracker()

    def test_singleton_pattern(self):
        """Global tracker uses singleton pattern."""
        tracker1 = get_context_tracker()
        tracker2 = get_context_tracker()

        assert tracker1 is tracker2

    def test_reset_clears_tracker(self):
        """Reset clears the global tracker."""
        tracker = get_context_tracker()
        tracker.track_compression(
            hash_key="test",
            turn_number=1,
            tool_name=None,
            original_count=10,
            compressed_count=1,
            workspace_key="ws-test",
        )

        assert len(tracker.get_tracked_hashes()) == 1

        reset_context_tracker()
        new_tracker = get_context_tracker()

        assert len(new_tracker.get_tracked_hashes()) == 0


class TestContextTrackerConfig:
    """Test context tracker configuration."""

    def test_default_config(self):
        """Default config values."""
        config = ContextTrackerConfig()

        assert config.enabled is True
        assert config.max_tracked_contexts == 100
        assert config.relevance_threshold == 0.3
        assert config.max_context_age_seconds == 300.0
        assert config.proactive_expansion is True
        assert config.max_proactive_expansions == 2

    def test_custom_config(self):
        """Custom config values."""
        config = ContextTrackerConfig(
            enabled=False,
            max_tracked_contexts=50,
            relevance_threshold=0.5,
            max_proactive_expansions=5,
        )

        assert config.enabled is False
        assert config.max_tracked_contexts == 50
        assert config.relevance_threshold == 0.5
        assert config.max_proactive_expansions == 5


class TestCompressedContextDataClass:
    """Test CompressedContext dataclass."""

    def test_create_context(self):
        """Create compressed context."""
        context = CompressedContext(
            hash_key="abc123",
            turn_number=5,
            timestamp=1234567890.0,
            tool_name="Bash",
            original_item_count=100,
            compressed_item_count=10,
            query_context="find files",
            sample_content='["file1.py", "file2.py"]',
            workspace_key="ws-test",
        )

        assert context.hash_key == "abc123"
        assert context.turn_number == 5
        assert context.tool_name == "Bash"
        assert context.original_item_count == 100
        assert context.compressed_item_count == 10


class TestExpansionRecommendationDataClass:
    """Test ExpansionRecommendation dataclass."""

    def test_expansion_recommendation(self):
        """Create an expansion recommendation (retrieval is always full)."""
        rec = ExpansionRecommendation(
            hash_key="abc123",
            reason="high relevance",
            relevance_score=0.9,
        )

        assert rec.hash_key == "abc123"
        assert rec.relevance_score == 0.9
        # Partial/search expansion fields no longer exist.
        assert not hasattr(rec, "expand_full")
        assert not hasattr(rec, "search_query")


class TestContextTrackerStats:
    """Test tracker statistics."""

    def test_stats_structure(self):
        """Stats have expected structure."""
        tracker = ContextTracker()

        tracker.track_compression(
            hash_key="test",
            turn_number=1,
            tool_name="Bash",
            original_count=100,
            compressed_count=10,
            workspace_key="ws-test",
        )

        stats = tracker.get_stats()

        assert "tracked_contexts" in stats
        assert "current_turn" in stats
        assert "config" in stats
        assert "contexts" in stats

        assert stats["tracked_contexts"] == 1
        assert len(stats["contexts"]) == 1

    def test_stats_context_details(self):
        """Stats include context details."""
        tracker = ContextTracker()

        tracker.track_compression(
            hash_key="abc123",
            turn_number=3,
            tool_name="Glob",
            original_count=50,
            compressed_count=5,
            workspace_key="ws-test",
        )

        stats = tracker.get_stats()
        context_stat = stats["contexts"][0]

        assert context_stat["hash"] == "abc123"
        assert context_stat["turn"] == 3
        assert context_stat["tool"] == "Glob"
        assert context_stat["items"] == "5/50"


class TestTrackerClear:
    """Test tracker clear functionality."""

    def test_clear_removes_all(self):
        """Clear removes all tracked contexts."""
        tracker = ContextTracker()

        for i in range(5):
            tracker.track_compression(
                hash_key=f"hash_{i}",
                turn_number=i,
                tool_name=None,
                original_count=10,
                compressed_count=1,
                workspace_key="ws-test",
            )

        assert len(tracker.get_tracked_hashes()) == 5

        tracker.clear()

        assert len(tracker.get_tracked_hashes()) == 0
        stats = tracker.get_stats()
        assert stats["current_turn"] == 0


# ============================================================================
# Workspace scoping (cross-project leak prevention).
#
# The bug: the ContextTracker is process-shared (one per proxy process,
# serving all sessions/projects). Without a workspace gate, Project A's
# compressed sample content keyword-matches Project B's later query and
# surfaces as "relevant" — which is exactly what Jocelyn reported on
# 2026-05-26: a tamag0 Python file (Ollama inference provider) appeared
# inside an unrelated daphni-rails Ruby/RSpec session.
#
# These tests pin the gate at the analyze_query level: entries are
# scoped by workspace_key, the same key the memory subsystem derives
# via ProjectResolver. Matching across workspaces is silently filtered;
# an empty workspace_key on analyze_query short-circuits to no
# recommendations (fail-closed per no-silent-fallbacks).
# ============================================================================


class TestWorkspaceScoping:
    """Cross-workspace leak prevention — the bug joce reported 2026-05-26."""

    @pytest.fixture(autouse=True)
    def reset_trackers(self):
        reset_context_tracker()
        reset_compression_store()
        yield
        reset_context_tracker()
        reset_compression_store()

    def test_same_workspace_match_works(self):
        """Within a single workspace, proactive expansion still functions normally."""
        config = ContextTrackerConfig(relevance_threshold=0.1)
        tracker = ContextTracker(config)

        tracker.track_compression(
            hash_key="auth_hash",
            turn_number=1,
            tool_name="Bash",
            original_count=100,
            compressed_count=10,
            workspace_key="ws-rails",
            query_context="find authentication files",
            sample_content="authentication middleware handler login security",
        )

        recommendations = tracker.analyze_query(
            query="show authentication middleware",
            current_turn=2,
            workspace_key="ws-rails",
        )

        # Same workspace — match expected (regression: don't accidentally over-filter).
        assert len(recommendations) >= 1
        assert recommendations[0].hash_key == "auth_hash"

    def test_cross_workspace_match_silently_filtered(self):
        """Workspace A's entry must NOT surface in Workspace B's analyze_query.

        This is the exact bug joce reported: Project tamag0 (workspace_key
        "ws-tamag0") had Python content stored; daphni-rails workspace
        queried for OAuth/session, the keyword overlap was high enough to
        score above threshold, and without scoping the Python content
        surfaced as "relevant" — wrong project, wrong language, real
        contamination risk.
        """
        config = ContextTrackerConfig(relevance_threshold=0.1)
        tracker = ContextTracker(config)

        # Workspace A: tamag0 Python code.
        tracker.track_compression(
            hash_key="tamag0_ollama_provider",
            turn_number=1,
            tool_name="Read",
            original_count=400,
            compressed_count=40,
            workspace_key="ws-tamag0",
            query_context="ollama inference provider",
            sample_content=(
                "class OllamaInferenceProvider provider auth login generate chat embed "
                "test_ollama_provider session token oauth user authentication middleware"
            ),
        )

        # Workspace B: daphni-rails Ruby code — entirely unrelated repo.
        # The query has heavy keyword overlap with the tamag0 sample
        # above (provider, oauth, session, authentication, middleware) —
        # exactly the surface-level lexical collision that triggered the
        # production bug.
        recommendations = tracker.analyze_query(
            query="OAuth provider session cookie middleware authentication for Rails",
            current_turn=2,
            workspace_key="ws-daphni-rails",
        )

        assert len(recommendations) == 0, (
            "Cross-workspace entry must NOT surface — this is the leak class "
            "Jocelyn reported on 2026-05-26 (tamag0 Python in daphni-rails Ruby session)."
        )

    def test_empty_workspace_key_returns_no_recommendations(self):
        """Empty workspace_key short-circuits to empty result set (fail-closed)."""
        config = ContextTrackerConfig(relevance_threshold=0.1)
        tracker = ContextTracker(config)

        tracker.track_compression(
            hash_key="some_hash",
            turn_number=1,
            tool_name="Bash",
            original_count=100,
            compressed_count=10,
            workspace_key="ws-real",
            sample_content="auth middleware",
        )

        # analyze_query with empty workspace_key — caller couldn't resolve a
        # project identity for the inbound request. Fail closed: no matches.
        recommendations = tracker.analyze_query(
            query="show auth middleware",
            current_turn=2,
            workspace_key="",
        )

        assert recommendations == [], (
            "Empty workspace_key must return [] — fail-closed per "
            "feedback_no_silent_fallbacks; otherwise an empty-keyed query "
            "would match nothing on the explicit-workspace branch but might "
            "still leak in any future fallback path."
        )

    def test_two_workspaces_each_see_only_their_own(self):
        """Tracking two workspaces in one tracker — each query sees only its own."""
        config = ContextTrackerConfig(relevance_threshold=0.1)
        tracker = ContextTracker(config)

        # Identical content surface in two different workspaces.
        for ws in ("ws-a", "ws-b"):
            tracker.track_compression(
                hash_key=f"hash-{ws}",
                turn_number=1,
                tool_name="Read",
                original_count=100,
                compressed_count=10,
                workspace_key=ws,
                query_context="authentication code",
                sample_content="auth middleware login session token oauth",
            )

        rec_a = tracker.analyze_query(
            query="show auth middleware",
            current_turn=2,
            workspace_key="ws-a",
        )
        rec_b = tracker.analyze_query(
            query="show auth middleware",
            current_turn=2,
            workspace_key="ws-b",
        )

        # Each workspace sees exactly its own entry — no leak in either direction.
        assert len(rec_a) == 1 and rec_a[0].hash_key == "hash-ws-a"
        assert len(rec_b) == 1 and rec_b[0].hash_key == "hash-ws-b"

    def test_workspace_label_propagates_to_format(self):
        """`format_expansions_for_context` emits the workspace label in the header."""
        tracker = ContextTracker()

        expansions = [
            {
                "hash": "h1",
                "type": "full",
                "content": "expanded content here",
                "item_count": 5,
                "reason": "high relevance",
            }
        ]

        # Without label: header has no workspace decoration.
        header_plain = tracker.format_expansions_for_context(expansions)
        assert "workspace:" not in header_plain

        # With label: provenance appears in header — same surface as the
        # memory-injection block (symmetric, see GH #462 Fix C).
        header_labeled = tracker.format_expansions_for_context(
            expansions, workspace_label="daphni-rails"
        )
        assert "workspace: daphni-rails" in header_labeled, (
            "Label must appear in the proactive-expansion header so the "
            "downstream model can reason about which project the expanded "
            "content came from."
        )

    def test_track_with_one_workspace_then_query_with_another_skips_silently(self):
        """LRU contents from prior workspace stay in tracker but don't leak.

        Regression for the production scenario: user works on tamag0,
        tracker accumulates entries. User switches to daphni-rails (same
        proxy process, ~minutes later). New queries on daphni-rails see
        an empty result set despite the LRU being non-empty — because the
        only entries present are scoped to tamag0.
        """
        config = ContextTrackerConfig(relevance_threshold=0.1)
        tracker = ContextTracker(config)

        # Populate workspace A.
        for i in range(5):
            tracker.track_compression(
                hash_key=f"tamag0-{i}",
                turn_number=i + 1,
                tool_name="Read",
                original_count=50,
                compressed_count=5,
                workspace_key="ws-tamag0",
                sample_content=f"file_{i}.py provider auth middleware session",
            )

        # Now query as workspace B. Same lexical surface, different identity.
        recommendations = tracker.analyze_query(
            query="show me the auth middleware provider",
            current_turn=10,
            workspace_key="ws-daphni-rails",
        )

        assert recommendations == [], (
            "Cross-workspace queries must return [] — even with a fully "
            "populated tracker. The workspace filter is the only gate."
        )
        # And the tracker itself still has its entries (we're filtering on
        # read, not purging on write — workspace A could come back and use
        # them again within the age window).
        assert len(tracker.get_tracked_hashes()) == 5
