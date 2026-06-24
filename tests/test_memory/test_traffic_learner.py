"""Tests for the Traffic Pattern Learner.

Tests pattern extraction from proxy traffic without requiring
a real memory backend.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from headroom.memory.traffic_learner import (
    ExtractedPattern,
    PatternCategory,
    TrafficLearner,
    _bash_binaries_match,
    _bash_first_binary,
    _classify_error,
    _commands_related_as_retry,
    _drop_contradictions,
    _is_error,
    _levenshtein,
    _load_persisted_patterns_from_sqlite,
    _normalize_bash_for_hash,
    _parse_iso_timestamp,
    _paths_related_as_typo,
    _patterns_to_recommendations,
    _project_for_pattern,
    _refine_error_recovery,
)

UTC = timezone.utc

# =============================================================================
# Error Classification Tests
# =============================================================================


class TestErrorClassification:
    def test_file_not_found(self):
        assert _classify_error("No such file or directory: foo.py") == "file_not_found"
        assert _classify_error("FileNotFoundError: [Errno 2]") == "file_not_found"

    def test_command_not_found(self):
        assert _classify_error("zsh: command not found: ruff") == "command_not_found"

    def test_module_not_found(self):
        assert _classify_error("ModuleNotFoundError: No module named 'foo'") == "module_not_found"

    def test_permission_denied(self):
        assert _classify_error("Permission denied: /etc/shadow") == "permission_denied"

    def test_not_an_error(self):
        assert _classify_error("Everything is fine, tests passed!") is None
        assert _classify_error("") is None

    def test_is_error_helper(self):
        assert _is_error("No such file or directory")
        assert not _is_error("All tests passed")
        assert not _is_error("")
        assert not _is_error("short")


# =============================================================================
# Recovery-pair relatedness heuristics
# =============================================================================


class TestPathsRelatedAsTypo:
    def test_identical_basename_different_dir_is_typo(self):
        # Same file in two locations — common path-typo case.
        assert _paths_related_as_typo("/a/state.rs", "/b/state.rs")

    def test_close_basename_is_typo(self):
        assert _paths_related_as_typo("/a/staet.rs", "/a/state.rs")
        assert _paths_related_as_typo("/a/App.tsx", "/a/app.tsx")

    def test_unrelated_files_in_same_dir_rejected(self):
        # The motivating bug: state.rs and lib.rs are unrelated files,
        # not typos, and should never be paired into a recovery rule.
        assert not _paths_related_as_typo("/src-tauri/src/state.rs", "/src-tauri/src/lib.rs")
        assert not _paths_related_as_typo("/x/models.py", "/x/views.py")

    def test_empty_or_equal_paths_rejected(self):
        assert not _paths_related_as_typo("", "/a/x")
        assert not _paths_related_as_typo("/a/x", "")
        assert not _paths_related_as_typo("/a/x", "/a/x")


class TestCommandsRelatedAsRetry:
    def test_python_to_python3_is_retry(self):
        assert _commands_related_as_retry("python test.py", "python3 test.py")

    def test_path_prefixed_binary_is_retry(self):
        assert _commands_related_as_retry("ruff check .", ".venv/bin/ruff check .")

    def test_extra_flag_is_retry(self):
        assert _commands_related_as_retry("cargo build", "cargo build --release")

    def test_different_binaries_rejected(self):
        assert not _commands_related_as_retry("grep -n foo bar.rs", "find . -name foo")

    def test_same_binary_unrelated_args_rejected(self):
        # The motivating bug: two grep calls sharing nothing but the
        # binary should not pair up. Different needles, different files.
        failed = (
            'grep -nE "smoke|HEADROOM_SMOKE_TEST_TIMEOUT|smoke_test" '
            "/Users/x/src-tauri/src/tool_manager.rs 2>&1 | head -20"
        )
        success = (
            'grep -nE "fn hf_hub_cache_dir|HF_HOME|HUGGINGFACE_HUB_CACHE" '
            "/Users/x/src-tauri/src/state.rs 2>&1 | head -10"
        )
        assert not _commands_related_as_retry(failed, success)

    def test_empty_or_equal_commands_rejected(self):
        assert not _commands_related_as_retry("", "ls")
        assert not _commands_related_as_retry("ls", "")
        assert not _commands_related_as_retry("ls", "ls")


class TestDropContradictions:
    def _read_recovery(self, failed: str, success: str) -> ExtractedPattern:
        return ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content=f"File `{failed}` does not exist. The correct path is `{success}`.",
            importance=0.7,
            entity_refs=[success],
            metadata={"error_category": "file_not_found", "failed_path": failed},
            evidence_count=5,
        )

    def test_drops_inverse_pairs(self):
        a_to_b = self._read_recovery("/x/a.rs", "/x/b.rs")
        b_to_a = self._read_recovery("/x/b.rs", "/x/a.rs")
        keep = self._read_recovery("/x/c.rs", "/x/d.rs")
        cleaned = _drop_contradictions([a_to_b, b_to_a, keep])
        assert keep in cleaned
        assert a_to_b not in cleaned
        assert b_to_a not in cleaned

    def test_passthrough_when_no_inverse(self):
        a_to_b = self._read_recovery("/x/a.rs", "/x/b.rs")
        cleaned = _drop_contradictions([a_to_b])
        assert cleaned == [a_to_b]

    def test_only_filters_error_recovery_category(self):
        env_pattern = ExtractedPattern(
            category=PatternCategory.ENVIRONMENT,
            content="File `x` does not exist. The correct path is `y`.",  # text alone
            importance=0.5,
            evidence_count=5,
        )
        cleaned = _drop_contradictions([env_pattern])
        assert cleaned == [env_pattern]

    def test_skips_error_recovery_with_non_canonical_content(self):
        """Bash recoveries don't match the Read regex; skip without crashing."""
        bash_pattern = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="Command `foo` fails (exit_code). Use `bar` instead.",
            importance=0.7,
            evidence_count=5,
        )
        cleaned = _drop_contradictions([bash_pattern])
        assert cleaned == [bash_pattern]


# =============================================================================
# Helper edge cases (branch coverage)
# =============================================================================


class TestLevenshtein:
    def test_equal_returns_zero(self):
        assert _levenshtein("abc", "abc") == 0

    def test_empty_a_returns_len_b(self):
        assert _levenshtein("", "abc") == 3

    def test_empty_b_returns_len_a(self):
        assert _levenshtein("abc", "") == 3

    def test_swap_when_a_longer(self):
        # Triggers the `len(a) > len(b)` swap branch.
        assert _levenshtein("abcdef", "abc") == 3

    def test_simple_substitution(self):
        assert _levenshtein("kitten", "sitting") == 3


class TestBashFirstBinary:
    def test_empty_returns_none(self):
        assert _bash_first_binary("") is None
        assert _bash_first_binary("   ") is None

    def test_strips_source_venv_prefix(self):
        assert _bash_first_binary("source .venv/bin/activate && pytest -x") == "pytest"

    def test_skips_env_var_assignments(self):
        assert _bash_first_binary("FOO=bar BAZ=qux python script.py") == "python"

    def test_returns_first_token_otherwise(self):
        assert _bash_first_binary("cargo test --release") == "cargo"


class TestBashBinariesMatch:
    def test_equal_strings_match(self):
        # Direct equality short-circuit; not exercised by the typical retry flow.
        assert _bash_binaries_match("cargo", "cargo")

    def test_basename_match_across_paths(self):
        assert _bash_binaries_match("ruff", ".venv/bin/ruff")
        assert _bash_binaries_match("/usr/bin/python3", "python3")

    def test_prefix_version_match(self):
        assert _bash_binaries_match("python", "python3")

    def test_unrelated_binaries_do_not_match(self):
        assert not _bash_binaries_match("grep", "find")


class TestPathsRelatedAsTypoEdgeCases:
    def test_root_paths_rejected(self):
        # After basename strip both sides become empty.
        assert not _paths_related_as_typo("/", "/")


class TestCommandsRelatedAsRetrySubstantiveToken:
    def test_substantive_token_beats_distance(self):
        # Edit distance is too high to pass the 40% gate, but both commands
        # share the substantive token "headroom-config", so the token-overlap
        # path accepts the pair.
        failed = "python -m foo --headroom-config=/etc/h.toml"
        success = "python -m bar --headroom-config=/etc/h.toml --extra"
        assert _commands_related_as_retry(failed, success)


# =============================================================================
# Traffic Learner Core Tests
# =============================================================================


class TestTrafficLearner:
    @pytest.fixture
    def learner(self):
        """Create a learner with low evidence threshold for testing."""
        return TrafficLearner(
            backend=None,
            user_id="test-user",
            min_evidence=1,  # Save on first sighting for tests
        )

    @pytest.mark.asyncio
    async def test_error_recovery_bash(self, learner: TrafficLearner):
        """Test error→recovery pattern extraction for Bash commands."""
        # First: a failed command
        await learner.on_tool_result(
            tool_name="Bash",
            tool_input={"command": "ruff check ."},
            tool_output="zsh: command not found: ruff",
            is_error=True,
        )

        # Then: the recovery
        await learner.on_tool_result(
            tool_name="Bash",
            tool_input={"command": "source .venv/bin/activate && ruff check ."},
            tool_output="All checks passed!",
            is_error=False,
        )

        stats = learner.get_stats()
        assert stats["patterns_extracted"] >= 1
        assert stats["requests_processed"] == 2

    @pytest.mark.asyncio
    async def test_error_recovery_read(self, learner: TrafficLearner):
        """Test error→recovery for Read tool (wrong path → correct path)."""
        await learner.on_tool_result(
            tool_name="Read",
            tool_input={"file_path": "/src/old_module.py"},
            tool_output="No such file or directory: /src/old_module.py",
            is_error=True,
        )

        await learner.on_tool_result(
            tool_name="Read",
            tool_input={"file_path": "/src/new_module.py"},
            tool_output="# Module content here\nclass Foo: pass",
            is_error=False,
        )

        stats = learner.get_stats()
        assert stats["patterns_extracted"] >= 1

    @pytest.mark.asyncio
    async def test_environment_venv_detection(self, learner: TrafficLearner):
        """Test detection of virtual environment activation patterns."""
        await learner.on_tool_result(
            tool_name="Bash",
            tool_input={"command": "source /project/.venv/bin/activate && pytest"},
            tool_output="5 passed in 2.1s",
            is_error=False,
        )

        stats = learner.get_stats()
        assert stats["patterns_extracted"] >= 1

    @pytest.mark.asyncio
    async def test_preference_extraction(self, learner: TrafficLearner):
        """Test extraction of user preference signals."""
        await learner.on_messages(
            [
                {"role": "user", "content": "don't use git push, I'll push manually"},
            ]
        )

        stats = learner.get_stats()
        assert stats["patterns_extracted"] >= 1

    @pytest.mark.asyncio
    async def test_preference_from_content_blocks(self, learner: TrafficLearner):
        """Test preference extraction from Anthropic content block format."""
        await learner.on_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "stop running the full test suite without asking"},
                    ],
                },
            ]
        )

        stats = learner.get_stats()
        assert stats["patterns_extracted"] >= 1

    @pytest.mark.asyncio
    async def test_evidence_accumulation(self):
        """Test that patterns need min_evidence before saving."""
        learner = TrafficLearner(backend=None, min_evidence=3)

        # Same error→recovery pattern 3 times
        for _ in range(3):
            await learner.on_tool_result(
                tool_name="Bash",
                tool_input={"command": "python test.py"},
                tool_output="command not found: python",
                is_error=True,
            )
            await learner.on_tool_result(
                tool_name="Bash",
                tool_input={"command": "python3 test.py"},
                tool_output="OK",
                is_error=False,
            )

        stats = learner.get_stats()
        assert stats["patterns_extracted"] >= 3

    @pytest.mark.asyncio
    async def test_dedup(self, learner: TrafficLearner):
        """Test that identical patterns are deduplicated."""
        # Same pattern twice
        for _ in range(2):
            await learner.on_tool_result(
                tool_name="Bash",
                tool_input={"command": "ruff check ."},
                tool_output="command not found: ruff",
                is_error=True,
            )
            await learner.on_tool_result(
                tool_name="Bash",
                tool_input={"command": ".venv/bin/ruff check ."},
                tool_output="OK",
                is_error=False,
            )

        # Should not double-count the same pattern
        stats = learner.get_stats()
        # First extraction saves, second is deduped
        assert stats["patterns_extracted"] >= 1

    @pytest.mark.asyncio
    async def test_extract_tool_results_from_messages(self, learner: TrafficLearner):
        """Test extraction of tool results from Anthropic message format."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": [{"type": "text", "text": "file1.py\nfile2.py"}],
                    }
                ],
            },
        ]

        results = learner.extract_tool_results_from_messages(messages)
        assert len(results) == 1
        assert results[0]["tool_name"] == "Bash"
        assert "file1.py" in results[0]["output"]
        assert not results[0]["is_error"]

    @pytest.mark.asyncio
    async def test_tool_history_bounded(self, learner: TrafficLearner):
        """Test that tool history stays within max_history."""
        for i in range(30):
            await learner.on_tool_result(
                tool_name="Read",
                tool_input={"file_path": f"/file{i}.py"},
                tool_output=f"content {i}",
                is_error=False,
            )

        assert len(learner._tool_history) <= learner._max_history

    @pytest.mark.asyncio
    async def test_no_pattern_from_success_only(self, learner: TrafficLearner):
        """Test that success without prior error doesn't generate error_recovery pattern."""
        await learner.on_tool_result(
            tool_name="Bash",
            tool_input={"command": "echo hello"},
            tool_output="hello",
            is_error=False,
        )

        stats = learner.get_stats()
        # Only environment patterns possible, no error_recovery
        assert stats["requests_processed"] == 1


# =============================================================================
# Pattern Model Tests
# =============================================================================


class TestExtractedPattern:
    def test_content_hash_deterministic(self):
        p1 = ExtractedPattern(
            category=PatternCategory.ENVIRONMENT,
            content="Use venv",
            importance=0.5,
        )
        p2 = ExtractedPattern(
            category=PatternCategory.ENVIRONMENT,
            content="Use venv",
            importance=0.8,  # Different importance, same hash
        )
        assert p1.content_hash == p2.content_hash

    def test_different_content_different_hash(self):
        p1 = ExtractedPattern(
            category=PatternCategory.ENVIRONMENT,
            content="Use venv",
            importance=0.5,
        )
        p2 = ExtractedPattern(
            category=PatternCategory.ENVIRONMENT,
            content="Use conda",
            importance=0.5,
        )
        assert p1.content_hash != p2.content_hash


# =============================================================================
# Project Routing
# =============================================================================


class TestProjectForPattern:
    def _project(self, path: str):
        from pathlib import Path as _P

        from headroom.learn.models import ProjectInfo

        p = _P(path)
        return ProjectInfo(name=p.name, project_path=p, data_path=p)

    def test_matches_longest_root(self):
        proj_a = self._project("/x/a")
        proj_b = self._project("/x/a/b")
        pattern = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="File `/x/a/b/foo.py` does not exist.",
            importance=0.5,
        )
        result = _project_for_pattern(pattern, [proj_a, proj_b])
        assert result is proj_b

    def test_returns_none_for_unanchored(self):
        proj_a = self._project("/x/a")
        pattern = ExtractedPattern(
            category=PatternCategory.PREFERENCE,
            content="User preference: use terse responses",
            importance=0.7,
        )
        assert _project_for_pattern(pattern, [proj_a]) is None

    def test_matches_via_entity_refs(self):
        proj = self._project("/x/a")
        pattern = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="Command failed.",
            importance=0.5,
            entity_refs=["/x/a/tool.py"],
        )
        assert _project_for_pattern(pattern, [proj]) is proj

    def test_windows_root_with_trailing_backslash_matches_child_path(self):
        proj = self._project(r"C:\Users\john.doe\repo\\")
        pattern = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content=r"File `C:\Users\john.doe\repo\src\main.py` does not exist.",
            importance=0.5,
        )

        assert _project_for_pattern(pattern, [proj]) is proj

    def test_no_false_match_on_prefix_boundary(self):
        # /x/ab should not match a project rooted at /x/a
        proj_a = self._project("/x/a")
        pattern = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="File `/x/ab/foo.py` does not exist.",
            importance=0.5,
        )
        assert _project_for_pattern(pattern, [proj_a]) is None


# =============================================================================
# Persisted-pattern loading from memory.db
# =============================================================================


class TestLoadPersistedPatterns:
    def _make_db(self, tmp_path, rows: list[dict]):
        import json as _json
        import sqlite3 as _sql

        db = tmp_path / "memory.db"
        conn = _sql.connect(db)
        conn.execute(
            "CREATE TABLE memories ("
            "id TEXT PRIMARY KEY, content TEXT NOT NULL, "
            "metadata TEXT NOT NULL DEFAULT '{}', "
            "entity_refs TEXT NOT NULL DEFAULT '[]', "
            "importance REAL NOT NULL DEFAULT 0.5, "
            "created_at TEXT)"
        )
        for i, r in enumerate(rows):
            conn.execute(
                "INSERT INTO memories "
                "(id, content, metadata, entity_refs, importance, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    str(i),
                    r["content"],
                    _json.dumps(r.get("metadata", {})),
                    _json.dumps(r.get("entity_refs", [])),
                    r.get("importance", 0.5),
                    r.get("created_at"),
                ),
            )
        conn.commit()
        conn.close()
        return db

    def test_dedupes_by_content_and_sums_evidence(self, tmp_path):
        db = self._make_db(
            tmp_path,
            [
                {
                    "content": "Command `foo` fails.",
                    "metadata": {
                        "source": "traffic_learner",
                        "category": "error_recovery",
                        "evidence_count": 2,
                    },
                },
                {
                    "content": "Command `foo` fails.",
                    "metadata": {
                        "source": "traffic_learner",
                        "category": "error_recovery",
                        "evidence_count": 3,
                    },
                },
            ],
        )
        patterns = _load_persisted_patterns_from_sqlite(db)
        assert len(patterns) == 1
        assert patterns[0].evidence_count == 5
        assert patterns[0].category == PatternCategory.ERROR_RECOVERY

    def test_skips_non_traffic_rows(self, tmp_path):
        db = self._make_db(
            tmp_path,
            [
                {
                    "content": "Something else",
                    "metadata": {"source": "other"},
                },
                {
                    "content": "From traffic",
                    "metadata": {
                        "source": "traffic_learner",
                        "category": "environment",
                    },
                },
            ],
        )
        patterns = _load_persisted_patterns_from_sqlite(db)
        assert len(patterns) == 1
        assert patterns[0].content == "From traffic"

    def test_reads_importance_column(self, tmp_path):
        db = self._make_db(
            tmp_path,
            [
                {
                    "content": "High-importance pattern",
                    "metadata": {
                        "source": "traffic_learner",
                        "category": "environment",
                    },
                    "importance": 0.85,
                },
            ],
        )
        patterns = _load_persisted_patterns_from_sqlite(db)
        assert len(patterns) == 1
        assert patterns[0].importance == 0.85

    def test_skips_unknown_category(self, tmp_path):
        db = self._make_db(
            tmp_path,
            [
                {
                    "content": "X",
                    "metadata": {"source": "traffic_learner", "category": "bogus"},
                },
            ],
        )
        assert _load_persisted_patterns_from_sqlite(db) == []


# =============================================================================
# Category → recommendation routing
# =============================================================================


class TestPatternsToRecommendations:
    def test_routes_preference_to_memory_file(self):
        from headroom.learn.models import RecommendationTarget

        patterns = [
            ExtractedPattern(
                category=PatternCategory.PREFERENCE,
                content="User prefers terse output",
                importance=0.8,
                evidence_count=3,
            ),
        ]
        recs = _patterns_to_recommendations(patterns)
        assert len(recs) == 1
        assert recs[0].target == RecommendationTarget.MEMORY_FILE
        assert "User prefers terse output" in recs[0].content

    def test_routes_environment_to_context_file(self):
        from headroom.learn.models import RecommendationTarget

        patterns = [
            ExtractedPattern(
                category=PatternCategory.ENVIRONMENT,
                content="Use uv run python",
                importance=0.7,
                evidence_count=4,
            ),
        ]
        recs = _patterns_to_recommendations(patterns)
        assert len(recs) == 1
        assert recs[0].target == RecommendationTarget.CONTEXT_FILE

    def test_groups_by_category(self):
        patterns = [
            ExtractedPattern(
                category=PatternCategory.ERROR_RECOVERY,
                content="A",
                importance=0.5,
                evidence_count=2,
            ),
            ExtractedPattern(
                category=PatternCategory.ERROR_RECOVERY,
                content="B",
                importance=0.5,
                evidence_count=5,
            ),
        ]
        recs = _patterns_to_recommendations(patterns)
        assert len(recs) == 1
        # B has higher evidence, should sort first
        lines = recs[0].content.splitlines()
        assert lines[0] == "- B"
        assert lines[1] == "- A"
        assert recs[0].evidence_count == 7


# =============================================================================
# Debounced flush worker
# =============================================================================


class TestFlushDebounce:
    @pytest.mark.asyncio
    async def test_flush_worker_rate_limits(self, monkeypatch):
        """Rapid dirty flags should not cause rapid flush_to_file calls."""
        from headroom.memory import traffic_learner as tl_mod

        # Shorten debounce for a fast test
        monkeypatch.setattr(tl_mod, "FLUSH_DEBOUNCE_SECONDS", 0.5)

        learner = TrafficLearner(backend=None, min_evidence=1)
        call_count = 0

        async def fake_flush() -> None:
            nonlocal call_count
            call_count += 1

        learner.flush_to_file = fake_flush  # type: ignore[method-assign]

        await learner.start()
        # Toggle dirty rapidly over ~1.2s, which permits at most ~2 flushes.
        for _ in range(30):
            learner._flush_dirty = True
            await __import__("asyncio").sleep(0.04)

        await learner.stop()

        # start() kicked a flush dirty→false at some point; stop() also calls
        # flush_to_file once (final flush). We want evidence the worker did
        # NOT call flush on every sleep tick — cap is generous.
        assert call_count <= 5, f"Expected few flushes, got {call_count}"
        assert call_count >= 1, "Expected at least one flush during the burst"


# =============================================================================
# Evidence-count persistence & re-sighting bumps
# =============================================================================


class _FakeBackend:
    """Minimal LocalBackend stand-in that persists to a real SQLite file.

    Provides just enough surface area for TrafficLearner: `_config.db_path`
    (read by `_resolve_backend_db_path`) and an `async save_memory` that
    inserts a row and returns an object with `.id`.
    """

    def __init__(self, db_path):
        import types as _types

        self._config = _types.SimpleNamespace(db_path=str(db_path))
        self._db_path = str(db_path)

    async def save_memory(
        self,
        *,
        content: str,
        user_id: str,
        importance: float,
        metadata: dict,
    ):
        import json as _json
        import sqlite3 as _sql
        import types as _types
        import uuid

        mid = str(uuid.uuid4())
        conn = _sql.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO memories (id, content, metadata, entity_refs, importance) "
                "VALUES (?,?,?,?,?)",
                (mid, content, _json.dumps(metadata), "[]", importance),
            )
            conn.commit()
        finally:
            conn.close()
        return _types.SimpleNamespace(id=mid)


def _init_db(path):
    import sqlite3 as _sql

    conn = _sql.connect(path)
    conn.execute(
        "CREATE TABLE memories ("
        "id TEXT PRIMARY KEY, content TEXT NOT NULL, "
        "metadata TEXT NOT NULL DEFAULT '{}', "
        "entity_refs TEXT NOT NULL DEFAULT '[]', "
        "importance REAL NOT NULL DEFAULT 0.5, "
        "created_at TEXT)"
    )
    conn.commit()
    conn.close()


def _read_traffic_rows(db_path):
    import json as _json
    import sqlite3 as _sql

    conn = _sql.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, content, metadata FROM memories "
            "WHERE json_extract(metadata, '$.source') = 'traffic_learner'"
        ).fetchall()
    finally:
        conn.close()
    return [(r[0], r[1], _json.loads(r[2])) for r in rows]


async def _wait_for_saved(learner: TrafficLearner, count: int, db_path) -> None:
    """Wait until at least `count` traffic_learner rows exist in the DB."""
    import asyncio as _asyncio

    for _ in range(100):
        if len(_read_traffic_rows(db_path)) >= count:
            return
        await _asyncio.sleep(0.02)
    raise AssertionError(
        f"Timeout waiting for {count} saved row(s); got {len(_read_traffic_rows(db_path))}"
    )


class TestEvidencePersistence:
    @pytest.mark.asyncio
    async def test_save_persists_actual_evidence_count(self, tmp_path):
        """The count written to the DB reflects total sightings, not the default 1."""
        db = tmp_path / "memory.db"
        _init_db(db)
        backend = _FakeBackend(db)
        learner = TrafficLearner(backend=backend, min_evidence=3)
        await learner.start()

        pattern_kwargs = {
            "category": PatternCategory.ENVIRONMENT,
            "content": "Use /usr/bin/python3 for system scripts.",
            "importance": 0.6,
        }
        for _ in range(3):
            await learner._accumulate(ExtractedPattern(**pattern_kwargs))
        await _wait_for_saved(learner, 1, db)
        await learner.stop()

        rows = _read_traffic_rows(db)
        assert len(rows) == 1
        assert rows[0][2]["evidence_count"] == 3

    @pytest.mark.asyncio
    async def test_resighting_bumps_persisted_row(self, tmp_path):
        """Sightings after save bump the existing row instead of creating duplicates."""
        db = tmp_path / "memory.db"
        _init_db(db)
        backend = _FakeBackend(db)
        learner = TrafficLearner(backend=backend, min_evidence=2)
        await learner.start()

        def mk() -> ExtractedPattern:
            return ExtractedPattern(
                category=PatternCategory.PREFERENCE,
                content="User preference: terse replies.",
                importance=0.7,
            )

        # Two sightings → save with evidence_count=2.
        await learner._accumulate(mk())
        await learner._accumulate(mk())
        await _wait_for_saved(learner, 1, db)

        # Three more sightings → three bumps.
        for _ in range(3):
            await learner._accumulate(mk())
        await learner.stop()

        rows = _read_traffic_rows(db)
        assert len(rows) == 1, "re-sightings must not create duplicate rows"
        assert rows[0][2]["evidence_count"] == 5

    @pytest.mark.asyncio
    async def test_hydrate_prevents_cross_session_duplicates(self, tmp_path):
        """A second session re-sighting an already-persisted pattern bumps, doesn't insert."""
        import json as _json
        import sqlite3 as _sql

        db = tmp_path / "memory.db"
        _init_db(db)

        # Session 1 row pre-seeded directly.
        seeded_content = "Command `foo` fails; use `bar` instead."
        conn = _sql.connect(db)
        conn.execute(
            "INSERT INTO memories (id, content, metadata, entity_refs, importance) "
            "VALUES (?,?,?,?,?)",
            (
                "seed-id",
                seeded_content,
                _json.dumps(
                    {
                        "source": "traffic_learner",
                        "category": "error_recovery",
                        "evidence_count": 2,
                    }
                ),
                "[]",
                0.7,
            ),
        )
        conn.commit()
        conn.close()

        # Session 2: fresh learner, hydrates from DB on start().
        backend = _FakeBackend(db)
        learner = TrafficLearner(backend=backend, min_evidence=2)
        await learner.start()

        def mk() -> ExtractedPattern:
            return ExtractedPattern(
                category=PatternCategory.ERROR_RECOVERY,
                content=seeded_content,
                importance=0.7,
            )

        # Two sightings: both should bump the seeded row (no duplicates).
        await learner._accumulate(mk())
        await learner._accumulate(mk())
        await learner.stop()

        rows = _read_traffic_rows(db)
        assert len(rows) == 1
        assert rows[0][0] == "seed-id"
        assert rows[0][2]["evidence_count"] == 4


# =============================================================================
# flush_to_file end-to-end + early-return paths
# =============================================================================


class _FakeWriteResult:
    def __init__(self, files_written):
        self.files_written = files_written


class _FakeWriter:
    def __init__(self):
        self.calls: list[tuple] = []
        self.files_to_return: list = []
        self.raise_on_write = False

    def write(self, recommendations, project, *, dry_run):
        self.calls.append((list(recommendations), project, dry_run))
        if self.raise_on_write:
            raise RuntimeError("boom")
        return _FakeWriteResult(list(self.files_to_return))


class _FakePlugin:
    def __init__(self, roots, writer, discover_raises=False):
        self._roots = roots
        self._writer = writer
        self._discover_raises = discover_raises

    def discover_projects(self):
        if self._discover_raises:
            raise RuntimeError("discover blew up")
        return list(self._roots)

    def create_writer(self):
        return self._writer


def _install_plugin_registry(monkeypatch, plugin):
    """Stub out headroom.learn.registry so flush_to_file uses our fake."""
    import sys
    import types as _types

    fake = _types.ModuleType("headroom.learn.registry")
    fake.auto_detect_plugins = lambda: [plugin] if plugin is not None else []  # type: ignore[attr-defined]
    fake.get_plugin = lambda agent_type: plugin  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.learn.registry", fake)
    import headroom.learn as learn_pkg

    monkeypatch.setattr(learn_pkg, "registry", fake, raising=False)


def _make_project(path):
    from pathlib import Path as _P

    from headroom.learn.models import ProjectInfo

    p = _P(path)
    return ProjectInfo(name=p.name, project_path=p, data_path=p)


class TestFlushToFile:
    @pytest.mark.asyncio
    async def test_end_to_end_writes_per_project(self, tmp_path, monkeypatch):
        """Happy path: anchored patterns → bucketed per project → writer called."""
        db = tmp_path / "memory.db"
        _init_db(db)
        backend = _FakeBackend(db)
        project_path = tmp_path.resolve()

        learner = TrafficLearner(backend=backend, agent_type="claude", min_evidence=2)
        writer = _FakeWriter()
        writer.files_to_return = [project_path / "CLAUDE.md"]
        proj = _make_project(str(project_path))
        plugin = _FakePlugin(roots=[proj], writer=writer)
        _install_plugin_registry(monkeypatch, plugin)

        # Need the save worker running so accumulated patterns actually land in
        # the DB where flush_to_file reads them.
        await learner.start()
        try:

            def mk() -> ExtractedPattern:
                return ExtractedPattern(
                    category=PatternCategory.ENVIRONMENT,
                    content=f"Use /usr/bin/python3 at {project_path}/main.py",
                    importance=0.6,
                )

            # Two sightings → save at evidence_count=2 (crosses live-flush gate).
            await learner._accumulate(mk())
            await learner._accumulate(mk())
            await _wait_for_saved(learner, 1, db)

            await learner.flush_to_file()
        finally:
            await learner.stop()

        assert len(writer.calls) >= 1
        recs, written_proj, dry_run = writer.calls[0]
        assert dry_run is False
        assert written_proj is proj
        assert len(recs) == 1
        assert "python3" in recs[0].content

    @pytest.mark.asyncio
    async def test_shutdown_flush_respects_min_evidence(self, tmp_path, monkeypatch):
        """Regression: stop() must not bypass the evidence gate.

        Earlier behavior collapsed min_evidence to 1 at shutdown, persisting
        every singleton pattern. This is exactly inverted: singletons are the
        *least* trustworthy patterns. The gate must use self._min_evidence at
        all times, including stop()'s final flush.
        """
        writer = _FakeWriter()
        proj = _make_project(str(tmp_path))
        plugin = _FakePlugin(roots=[proj], writer=writer)
        _install_plugin_registry(monkeypatch, plugin)

        learner = TrafficLearner(backend=None, agent_type="claude", min_evidence=5)
        # Singleton pattern: should NOT survive the shutdown flush.
        learner._pattern_counts["h"] = (
            ExtractedPattern(
                category=PatternCategory.ENVIRONMENT,
                content=f"singleton at {tmp_path}/main.py",
                importance=0.5,
                evidence_count=1,
            ),
            1,
        )
        await learner.stop()  # triggers a final flush_to_file
        assert writer.calls == [], "singleton survived the shutdown gate"

    @pytest.mark.asyncio
    async def test_early_returns_no_plugin(self, monkeypatch):
        """No plugin detected → flush is a no-op."""
        learner = TrafficLearner(backend=None, agent_type="unknown", min_evidence=1)
        _install_plugin_registry(monkeypatch, None)
        # Seed an accumulator entry so the check isn't vacuously "no patterns".
        learner._pattern_counts["h"] = (
            ExtractedPattern(
                category=PatternCategory.ENVIRONMENT,
                content="x",
                importance=0.5,
                evidence_count=2,
            ),
            2,
        )
        await learner.flush_to_file()  # returns without raising

    @pytest.mark.asyncio
    async def test_early_return_no_patterns(self, monkeypatch):
        """Empty accumulator and empty DB → flush returns without calling writer."""
        writer = _FakeWriter()
        plugin = _FakePlugin(roots=[_make_project("/x/a")], writer=writer)
        _install_plugin_registry(monkeypatch, plugin)

        learner = TrafficLearner(backend=None, agent_type="claude", min_evidence=1)
        await learner.flush_to_file()
        assert writer.calls == []

    @pytest.mark.asyncio
    async def test_discover_projects_failure_is_swallowed(self, monkeypatch):
        """If plugin.discover_projects raises, flush logs and returns."""
        writer = _FakeWriter()
        plugin = _FakePlugin(roots=[], writer=writer, discover_raises=True)
        _install_plugin_registry(monkeypatch, plugin)

        learner = TrafficLearner(backend=None, agent_type="claude", min_evidence=1)
        learner._pattern_counts["h"] = (
            ExtractedPattern(
                category=PatternCategory.ENVIRONMENT,
                content="whatever",
                importance=0.5,
                evidence_count=2,
            ),
            2,
        )
        await learner.flush_to_file()
        assert writer.calls == []  # no roots → short-circuits before writer

    @pytest.mark.asyncio
    async def test_unanchored_patterns_dropped(self, tmp_path, monkeypatch):
        """Patterns with no path anchoring are dropped before writer is called."""
        writer = _FakeWriter()
        project_path = tmp_path.resolve()
        plugin = _FakePlugin(roots=[_make_project(str(project_path))], writer=writer)
        _install_plugin_registry(monkeypatch, plugin)

        learner = TrafficLearner(backend=None, agent_type="claude", min_evidence=1)
        # Content has no absolute path — should be dropped as un-anchored.
        learner._pattern_counts["h"] = (
            ExtractedPattern(
                category=PatternCategory.PREFERENCE,
                content="User preference: use terse output",
                importance=0.7,
                evidence_count=2,
            ),
            2,
        )
        await learner.flush_to_file()
        assert writer.calls == []

    @pytest.mark.asyncio
    async def test_writer_exception_does_not_propagate(self, tmp_path, monkeypatch):
        """A writer raising should be logged; flush must not bubble the error."""
        writer = _FakeWriter()
        writer.raise_on_write = True
        project_path = tmp_path.resolve()
        plugin = _FakePlugin(roots=[_make_project(str(project_path))], writer=writer)
        _install_plugin_registry(monkeypatch, plugin)

        learner = TrafficLearner(backend=None, agent_type="claude", min_evidence=1)
        learner._pattern_counts["h"] = (
            ExtractedPattern(
                category=PatternCategory.ENVIRONMENT,
                content=f"Use {project_path}/tool.py",
                importance=0.6,
                evidence_count=2,
            ),
            2,
        )
        await learner.flush_to_file()  # must not raise
        assert len(writer.calls) == 1


# =============================================================================
# Internal helper edge cases — _resolve_backend_db_path / _collect_all_patterns
# / _hydrate_persisted_state / _bump_persisted_evidence
# =============================================================================


class TestBackendResolution:
    def test_resolve_none_backend(self):
        from headroom.memory.traffic_learner import _resolve_backend_db_path

        assert _resolve_backend_db_path(None) is None

    def test_resolve_backend_without_config(self):
        from headroom.memory.traffic_learner import _resolve_backend_db_path

        class _Bare:
            pass

        assert _resolve_backend_db_path(_Bare()) is None

    def test_resolve_backend_with_empty_db_path(self):
        import types as _types

        from headroom.memory.traffic_learner import _resolve_backend_db_path

        backend = _types.SimpleNamespace(_config=_types.SimpleNamespace(db_path=""))
        assert _resolve_backend_db_path(backend) is None


class TestCollectAllPatterns:
    @pytest.mark.asyncio
    async def test_merges_db_and_accumulator(self, tmp_path):
        """Patterns in both DB and accumulator get evidence_count summed by hash."""
        db = tmp_path / "memory.db"
        _init_db(db)
        backend = _FakeBackend(db)

        # Seed DB with a traffic_learner row at evidence_count=3.
        await backend.save_memory(
            content="shared pattern",
            user_id="t",
            importance=0.5,
            metadata={
                "source": "traffic_learner",
                "category": "environment",
                "evidence_count": 3,
            },
        )

        learner = TrafficLearner(backend=backend, min_evidence=1)
        # Same content in accumulator with count=2; hash matches.
        p = ExtractedPattern(
            category=PatternCategory.ENVIRONMENT,
            content="shared pattern",
            importance=0.5,
        )
        learner._pattern_counts[p.content_hash] = (p, 2)

        merged = learner._collect_all_patterns()
        assert len(merged) == 1
        assert merged[0].evidence_count == 3 + 2

    def test_handles_missing_db_gracefully(self, tmp_path):
        """A backend pointing to a nonexistent DB is skipped, not raised."""
        backend = _FakeBackend(tmp_path / "absent.db")  # file not created
        learner = TrafficLearner(backend=backend, min_evidence=1)
        merged = learner._collect_all_patterns()
        assert merged == []


class TestHydrateEdgeCases:
    @pytest.mark.asyncio
    async def test_no_backend(self):
        """start() with backend=None hydrates to empty state and still runs."""
        learner = TrafficLearner(backend=None, min_evidence=1)
        await learner.start()
        try:
            assert learner._saved_hashes == set()
            assert learner._persisted_ids == {}
        finally:
            await learner.stop()

    @pytest.mark.asyncio
    async def test_missing_db_file(self, tmp_path):
        """Backend with a db_path that doesn't exist → hydrate is a no-op."""
        backend = _FakeBackend(tmp_path / "not-there.db")
        learner = TrafficLearner(backend=backend, min_evidence=1)
        await learner._hydrate_persisted_state()
        assert learner._saved_hashes == set()
        assert learner._persisted_ids == {}


class TestBumpEdgeCases:
    @pytest.mark.asyncio
    async def test_bump_with_no_backend_is_noop(self):
        learner = TrafficLearner(backend=None, min_evidence=1)
        # Should not raise even with no backend.
        await learner._bump_persisted_evidence("some-id")

    @pytest.mark.asyncio
    async def test_bump_with_missing_db_is_noop(self, tmp_path):
        backend = _FakeBackend(tmp_path / "absent.db")
        learner = TrafficLearner(backend=backend, min_evidence=1)
        await learner._bump_persisted_evidence("some-id")  # no exception

    @pytest.mark.asyncio
    async def test_bump_unknown_id_is_noop(self, tmp_path):
        """Updating a non-existent memory id silently affects zero rows."""
        db = tmp_path / "memory.db"
        _init_db(db)
        backend = _FakeBackend(db)
        learner = TrafficLearner(backend=backend, min_evidence=1)
        await learner._bump_persisted_evidence("no-such-id")
        assert _read_traffic_rows(db) == []


# =============================================================================
# stop() cancels the flush task
# =============================================================================


class TestStopCancels:
    @pytest.mark.asyncio
    async def test_stop_cancels_flush_task(self):
        learner = TrafficLearner(backend=None, min_evidence=1)
        await learner.start()
        assert learner._flush_task is not None and not learner._flush_task.done()
        await learner.stop()
        assert learner._flush_task is None or learner._flush_task.done()


class TestNormalizedHash:
    """Error-recovery patterns hash on recovery intent, not literal text."""

    def _mk(self, **meta) -> ExtractedPattern:
        return ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content=f"content-{meta.get('tool', 'none')}-{len(meta)}",
            importance=0.7,
            metadata=meta,
        )

    def test_read_recovery_basename_hash(self):
        a = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="File `/a/state.rs` does not exist. The correct path is `/a/lib.rs`.",
            importance=0.7,
            metadata={"tool": "Read", "error_path": "/a/state.rs", "success_path": "/a/lib.rs"},
        )
        b = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="File `/b/state.rs` does not exist. The correct path is `/b/lib.rs`.",
            importance=0.7,
            metadata={"tool": "Read", "error_path": "/b/state.rs", "success_path": "/b/lib.rs"},
        )
        assert a.content_hash == b.content_hash

    def test_bash_recovery_tail_count_collapse(self):
        a = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="Command `cargo check` fails. Use `cargo check --manifest-path src-tauri/Cargo.toml | tail -10` instead.",
            importance=0.7,
            metadata={
                "tool": "Bash",
                "failed_cmd": "cargo check",
                "success_cmd": "cargo check --manifest-path src-tauri/Cargo.toml | tail -10",
            },
        )
        b = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="Command `cargo check` fails. Use `cargo check --manifest-path src-tauri/Cargo.toml | tail -50` instead.",
            importance=0.7,
            metadata={
                "tool": "Bash",
                "failed_cmd": "cargo check",
                "success_cmd": "cargo check --manifest-path src-tauri/Cargo.toml | tail -50",
            },
        )
        assert a.content_hash == b.content_hash

    def test_bash_recovery_pipe_boundary(self):
        a = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="x",
            importance=0.7,
            metadata={
                "tool": "Bash",
                "failed_cmd": "grep foo bar.txt",
                "success_cmd": "grep -n foo bar.txt | head -5",
            },
        )
        b = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="y",
            importance=0.7,
            metadata={
                "tool": "Bash",
                "failed_cmd": "grep foo bar.txt",
                "success_cmd": "grep -n foo bar.txt | wc -l",
            },
        )
        assert a.content_hash == b.content_hash

    def test_bash_recovery_different_primary_cmd_different_hash(self):
        a = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="x",
            importance=0.7,
            metadata={
                "tool": "Bash",
                "failed_cmd": "cargo check",
                "success_cmd": "cargo build",
            },
        )
        b = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="y",
            importance=0.7,
            metadata={
                "tool": "Bash",
                "failed_cmd": "cargo check",
                "success_cmd": "cargo test",
            },
        )
        assert a.content_hash != b.content_hash

    def test_non_error_recovery_unchanged(self):
        a = ExtractedPattern(
            category=PatternCategory.ENVIRONMENT,
            content="Use /usr/bin/python3.",
            importance=0.7,
        )
        b = ExtractedPattern(
            category=PatternCategory.ENVIRONMENT,
            content="Use /opt/bin/python3.",
            importance=0.7,
        )
        assert a.content_hash != b.content_hash

    def test_error_recovery_without_tool_falls_back_to_content(self):
        """Legacy error_recovery rows without a `tool` metadata key still work."""
        a = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="Some legacy bullet.",
            importance=0.7,
        )
        b = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="Some legacy bullet.",
            importance=0.7,
        )
        assert a.content_hash == b.content_hash


class TestRefineErrorRecovery:
    """Render-time pipeline: hard floor, re-validate, collapse, rank, cap."""

    def _mk_read(
        self,
        *,
        error_path: str,
        success_path: str,
        evidence: int = 1,
        last_seen: datetime | None = None,
    ) -> ExtractedPattern:
        now = datetime.now(UTC)
        return ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content=f"File `{error_path}` does not exist. The correct path is `{success_path}`.",
            importance=0.7,
            evidence_count=evidence,
            metadata={
                "tool": "Read",
                "error_path": error_path,
                "success_path": success_path,
            },
            last_seen_at=last_seen or now,
            first_seen_at=last_seen or now,
        )

    def test_drops_patterns_beyond_hard_floor(self, tmp_path):
        target = tmp_path / "lib.rs"
        target.write_text("pub fn x() {}")
        old = self._mk_read(
            error_path=str(tmp_path / "state.rs"),
            success_path=str(target),
            last_seen=datetime.now(UTC) - timedelta(days=22),
        )
        fresh = self._mk_read(
            error_path=str(tmp_path / "other.rs"),
            success_path=str(target),
        )
        refined = _refine_error_recovery([old, fresh])
        assert fresh in refined
        assert old not in refined

    def test_revalidates_read_success_path(self, tmp_path):
        present = tmp_path / "present.rs"
        present.write_text("x")
        p_ok = self._mk_read(
            error_path=str(tmp_path / "miss.rs"),
            success_path=str(present),
        )
        p_missing = self._mk_read(
            error_path=str(tmp_path / "other.rs"),
            success_path=str(tmp_path / "gone.rs"),
        )
        refined = _refine_error_recovery([p_ok, p_missing])
        assert p_ok in refined
        assert p_missing not in refined

    def test_collapses_ambiguous_error_path(self, tmp_path):
        a = tmp_path / "a.rs"
        a.write_text("x")
        b = tmp_path / "b.rs"
        b.write_text("y")
        c = tmp_path / "c.rs"
        c.write_text("z")
        error_path = str(tmp_path / "ambiguous.rs")
        group = [
            self._mk_read(error_path=error_path, success_path=str(a), evidence=3),
            self._mk_read(error_path=error_path, success_path=str(b), evidence=2),
            self._mk_read(error_path=error_path, success_path=str(c), evidence=1),
        ]
        refined = _refine_error_recovery(group)
        assert len(refined) == 1
        collapsed = refined[0]
        assert collapsed.metadata.get("collapsed") is True
        assert collapsed.evidence_count == 6
        assert "ambiguous.rs" in collapsed.content
        assert "Glob/Grep" in collapsed.content

    def test_single_success_path_not_collapsed(self, tmp_path):
        a = tmp_path / "a.rs"
        a.write_text("x")
        error_path = str(tmp_path / "only-one-target.rs")
        patterns = [
            self._mk_read(error_path=error_path, success_path=str(a), evidence=3),
            self._mk_read(error_path=error_path, success_path=str(a), evidence=2),
        ]
        refined = _refine_error_recovery(patterns)
        # Not collapsed — only one distinct success_path.
        assert all(p.metadata.get("collapsed") is not True for p in refined)
        assert len(refined) == 2

    def test_recency_ranking_prefers_fresh_over_stale_heavy(self, tmp_path):
        target = tmp_path / "lib.rs"
        target.write_text("x")
        # Heavy but old: evidence=10, seen 10 days ago → score ~10 * 0.5**2 = 2.5
        heavy_old = self._mk_read(
            error_path=str(tmp_path / "old.rs"),
            success_path=str(target),
            evidence=10,
            last_seen=datetime.now(UTC) - timedelta(days=10),
        )
        # Light but fresh: evidence=3, seen now → score ~3
        light_fresh = self._mk_read(
            error_path=str(tmp_path / "fresh.rs"),
            success_path=str(target),
            evidence=3,
        )
        refined = _refine_error_recovery([heavy_old, light_fresh])
        assert refined[0] is light_fresh
        assert refined[1] is heavy_old

    def test_section_cap_enforced(self, tmp_path):
        target = tmp_path / "lib.rs"
        target.write_text("x")
        patterns = [
            self._mk_read(
                error_path=str(tmp_path / f"miss_{i}.rs"),
                success_path=str(target),
                evidence=i + 1,
            )
            for i in range(25)
        ]
        refined = _refine_error_recovery(patterns)
        assert len(refined) == 15
        # Highest-evidence ones kept (all are equally fresh, so evidence wins).
        kept_evidence = sorted(p.evidence_count for p in refined)
        assert kept_evidence[0] >= 11  # Bottom of top-15 out of 1..25

    def test_read_recovery_without_success_path_not_revalidated(self):
        """Read patterns lacking `success_path` in metadata skip re-validation cleanly."""
        p = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="Some legacy Read bullet",
            importance=0.7,
            metadata={"tool": "Read", "error_path": "/something.rs"},
            last_seen_at=datetime.now(UTC),
        )
        refined = _refine_error_recovery([p])
        assert p in refined

    def test_bash_recoveries_not_revalidated(self, tmp_path):
        """Bash patterns pass through re-validation regardless of command content."""
        bash_pat = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="Command `x` fails. Use `y` instead.",
            importance=0.7,
            evidence_count=1,
            metadata={
                "tool": "Bash",
                "failed_cmd": "x",
                "success_cmd": "y",
            },
            last_seen_at=datetime.now(UTC),
        )
        refined = _refine_error_recovery([bash_pat])
        assert bash_pat in refined

    def test_empty_input_returns_empty(self):
        assert _refine_error_recovery([]) == []

    def test_missing_timestamps_survive_one_render(self):
        """Patterns without timestamps are kept rather than silently dropped."""
        p = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="legacy bullet",
            importance=0.7,
        )
        assert p.first_seen_at is None
        assert p.last_seen_at is None
        refined = _refine_error_recovery([p])
        assert p in refined

    def test_refined_empty_skips_section_in_recommendations(self, tmp_path):
        """If all error_recovery patterns fail re-validation, no recommendation is emitted."""
        # Only pattern is a Read recovery pointing at a nonexistent success_path.
        stale = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="File `/a.rs` does not exist. The correct path is `/gone.rs`.",
            importance=0.7,
            metadata={
                "tool": "Read",
                "error_path": "/a.rs",
                "success_path": str(tmp_path / "does-not-exist.rs"),
            },
            last_seen_at=datetime.now(UTC),
        )
        recs = _patterns_to_recommendations([stale])
        # Section should be skipped entirely — no recommendation produced.
        assert recs == []

    def test_oserror_during_revalidation_keeps_row(self, monkeypatch):
        """Transient OS errors during path checks should not drop the row."""
        p = ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content="File `/a.rs` does not exist. The correct path is `/b.rs`.",
            importance=0.7,
            metadata={"tool": "Read", "error_path": "/a.rs", "success_path": "/b.rs"},
            last_seen_at=datetime.now(UTC),
        )

        def _raise(self):
            raise OSError("simulated permission error")

        monkeypatch.setattr("pathlib.Path.exists", _raise)
        refined = _refine_error_recovery([p])
        assert p in refined


class TestNormalizeBashForHash:
    """Bash command normalization for hash-key collapse."""

    def test_empty_string_returns_empty(self):
        assert _normalize_bash_for_hash("") == ""

    def test_no_volatile_suffix_unchanged(self):
        assert _normalize_bash_for_hash("cargo check") == "cargo check"

    def test_strips_head_suffix(self):
        assert _normalize_bash_for_hash("grep foo bar | head -20") == "grep foo bar"

    def test_strips_tail_suffix(self):
        assert _normalize_bash_for_hash("cargo check | tail -5") == "cargo check"

    def test_strips_trailing_context_flags(self):
        # The regex is anchored to end-of-string; context flags must be trailing.
        assert _normalize_bash_for_hash("grep foo bar -A 3") == "grep foo bar"

    def test_strips_stderr_redirect(self):
        assert _normalize_bash_for_hash("cargo check 2>&1") == "cargo check"

    def test_cuts_at_first_chain(self):
        # && boundary collapses to just the primary command
        assert _normalize_bash_for_hash("cd /tmp && ls") == "cd /tmp"


class TestParseIsoTimestamp:
    """Edge-case coverage for _parse_iso_timestamp."""

    def test_none_returns_none(self):
        assert _parse_iso_timestamp(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_iso_timestamp("") is None

    def test_non_string_returns_none(self):
        assert _parse_iso_timestamp(12345) is None
        assert _parse_iso_timestamp(3.14) is None

    def test_invalid_format_returns_none(self):
        assert _parse_iso_timestamp("not an iso string") is None

    def test_naive_timestamp_assumed_utc(self):
        parsed = _parse_iso_timestamp("2026-04-20T12:00:00")
        assert parsed is not None
        assert parsed.tzinfo == UTC

    def test_aware_timestamp_preserved(self):
        parsed = _parse_iso_timestamp("2026-04-20T12:00:00+00:00")
        assert parsed is not None
        assert parsed.tzinfo is not None


class TestLoadPersistedPatternsTimestamps:
    """The sqlite load path reads first_seen_at / last_seen_at correctly."""

    def _make_db(self, tmp_path, rows: list[dict]):
        import json as _json
        import sqlite3 as _sql

        db = tmp_path / "memory.db"
        conn = _sql.connect(db)
        conn.execute(
            "CREATE TABLE memories ("
            "id TEXT PRIMARY KEY, content TEXT NOT NULL, "
            "metadata TEXT NOT NULL DEFAULT '{}', "
            "entity_refs TEXT NOT NULL DEFAULT '[]', "
            "importance REAL NOT NULL DEFAULT 0.5, "
            "created_at TEXT)"
        )
        for i, r in enumerate(rows):
            conn.execute(
                "INSERT INTO memories "
                "(id, content, metadata, entity_refs, importance, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    str(i),
                    r["content"],
                    _json.dumps(r.get("metadata", {})),
                    _json.dumps(r.get("entity_refs", [])),
                    r.get("importance", 0.5),
                    r.get("created_at"),
                ),
            )
        conn.commit()
        conn.close()
        return db

    def test_reads_timestamps_from_metadata(self, tmp_path):
        db = self._make_db(
            tmp_path,
            [
                {
                    "content": "env bullet",
                    "metadata": {
                        "source": "traffic_learner",
                        "category": "environment",
                        "evidence_count": 3,
                        "first_seen_at": "2026-04-10T10:00:00+00:00",
                        "last_seen_at": "2026-04-20T15:00:00+00:00",
                    },
                }
            ],
        )
        patterns = _load_persisted_patterns_from_sqlite(db)
        assert len(patterns) == 1
        p = patterns[0]
        assert p.first_seen_at is not None
        assert p.first_seen_at.year == 2026 and p.first_seen_at.month == 4
        assert p.last_seen_at is not None
        assert p.last_seen_at.day == 20

    def test_falls_back_to_created_at(self, tmp_path):
        """When metadata has no timestamps, `created_at` is used."""
        db = self._make_db(
            tmp_path,
            [
                {
                    "content": "env bullet",
                    "metadata": {
                        "source": "traffic_learner",
                        "category": "environment",
                        "evidence_count": 1,
                    },
                    "created_at": "2026-03-01T09:00:00+00:00",
                }
            ],
        )
        patterns = _load_persisted_patterns_from_sqlite(db)
        assert len(patterns) == 1
        assert patterns[0].first_seen_at is not None
        assert patterns[0].first_seen_at.month == 3
        # last_seen defaults to first_seen when metadata lacks both.
        assert patterns[0].last_seen_at == patterns[0].first_seen_at

    def test_collision_merges_timestamps_max_last_min_first(self, tmp_path):
        """Two rows collapsing to the same hash keep the widest timestamp range."""
        db = self._make_db(
            tmp_path,
            [
                {
                    "content": "dup bullet",
                    "importance": 0.4,
                    "metadata": {
                        "source": "traffic_learner",
                        "category": "preference",
                        "evidence_count": 2,
                        "first_seen_at": "2026-04-10T00:00:00+00:00",
                        "last_seen_at": "2026-04-15T00:00:00+00:00",
                    },
                },
                {
                    "content": "dup bullet",
                    "importance": 0.9,
                    "metadata": {
                        "source": "traffic_learner",
                        "category": "preference",
                        "evidence_count": 3,
                        "first_seen_at": "2026-04-01T00:00:00+00:00",
                        "last_seen_at": "2026-04-20T00:00:00+00:00",
                    },
                },
            ],
        )
        patterns = _load_persisted_patterns_from_sqlite(db)
        assert len(patterns) == 1
        p = patterns[0]
        assert p.evidence_count == 5
        # Higher importance wins when collision merges.
        assert p.importance == 0.9
        assert p.first_seen_at is not None and p.first_seen_at.day == 1
        assert p.last_seen_at is not None and p.last_seen_at.day == 20

    def test_non_numeric_importance_falls_back_to_default(self, tmp_path):
        """Rows with an unparseable importance value use 0.5."""
        import json as _json
        import sqlite3 as _sql

        db = tmp_path / "memory.db"
        conn = _sql.connect(db)
        conn.execute(
            "CREATE TABLE memories ("
            "id TEXT PRIMARY KEY, content TEXT NOT NULL, "
            "metadata TEXT NOT NULL DEFAULT '{}', "
            "entity_refs TEXT NOT NULL DEFAULT '[]', "
            "importance TEXT, "
            "created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO memories (id, content, metadata, importance) VALUES (?,?,?,?)",
            (
                "0",
                "bullet",
                _json.dumps(
                    {
                        "source": "traffic_learner",
                        "category": "environment",
                        "evidence_count": 1,
                    }
                ),
                "not-a-number",
            ),
        )
        conn.commit()
        conn.close()
        patterns = _load_persisted_patterns_from_sqlite(db)
        assert len(patterns) == 1
        assert patterns[0].importance == 0.5

    def test_malformed_metadata_json_skipped_gracefully(self, tmp_path):
        """Rows with invalid JSON metadata don't crash the load."""
        import sqlite3 as _sql

        db = tmp_path / "memory.db"
        conn = _sql.connect(db)
        conn.execute(
            "CREATE TABLE memories ("
            "id TEXT PRIMARY KEY, content TEXT NOT NULL, "
            "metadata TEXT NOT NULL DEFAULT '{}', "
            "entity_refs TEXT NOT NULL DEFAULT '[]', "
            "importance REAL NOT NULL DEFAULT 0.5, "
            "created_at TEXT)"
        )
        # Invalid JSON in metadata
        conn.execute(
            "INSERT INTO memories VALUES (?,?,?,?,?,?)",
            ("0", "bullet", "{not json", "[]", 0.5, None),
        )
        conn.commit()
        conn.close()
        # Should not raise — the row is simply skipped (no recognizable category).
        patterns = _load_persisted_patterns_from_sqlite(db)
        assert patterns == []


class TestBumpPersistsLastSeenAt:
    """_bump_persisted_evidence sets $.last_seen_at on every bump."""

    @pytest.mark.asyncio
    async def test_bump_sets_last_seen_at_in_metadata(self, tmp_path):
        import sqlite3 as _sql

        db = tmp_path / "memory.db"
        _init_db(db)
        # Seed a traffic_learner row with no last_seen_at.
        import json as _json

        conn = _sql.connect(db)
        conn.execute(
            "INSERT INTO memories (id, content, metadata) VALUES (?,?,?)",
            (
                "row-1",
                "bullet",
                _json.dumps(
                    {
                        "source": "traffic_learner",
                        "category": "environment",
                        "evidence_count": 1,
                    }
                ),
            ),
        )
        conn.commit()
        conn.close()

        backend = _FakeBackend(db)
        learner = TrafficLearner(backend=backend, min_evidence=1)
        await learner._bump_persisted_evidence("row-1")

        conn = _sql.connect(db)
        row = conn.execute("SELECT metadata FROM memories WHERE id='row-1'").fetchone()
        conn.close()
        meta = _json.loads(row[0])
        assert meta["evidence_count"] == 2
        assert "last_seen_at" in meta
        # Should be parseable back.
        parsed = _parse_iso_timestamp(meta["last_seen_at"])
        assert parsed is not None


class TestHydrateLegacyRow:
    """Legacy rows without `category` metadata fall back to literal-content hashing."""

    @pytest.mark.asyncio
    async def test_hydrate_legacy_row_without_category(self, tmp_path):
        import sqlite3 as _sql

        db = tmp_path / "memory.db"
        _init_db(db)
        import json as _json

        conn = _sql.connect(db)
        # No `category` key in metadata — must still hydrate.
        conn.execute(
            "INSERT INTO memories (id, content, metadata) VALUES (?,?,?)",
            (
                "legacy-1",
                "legacy bullet",
                _json.dumps({"source": "traffic_learner"}),
            ),
        )
        conn.commit()
        conn.close()

        backend = _FakeBackend(db)
        learner = TrafficLearner(backend=backend, min_evidence=1)
        await learner._hydrate_persisted_state()

        # Falls back to sha256(content) for the hash key.
        import hashlib as _h

        expected = _h.sha256(b"legacy bullet").hexdigest()[:16]
        assert expected in learner._saved_hashes
        assert learner._persisted_ids[expected] == "legacy-1"

    @pytest.mark.asyncio
    async def test_hydrate_skips_empty_content(self, tmp_path):
        """Rows with empty content are skipped during hydration."""
        import json as _json
        import sqlite3 as _sql

        db = tmp_path / "memory.db"
        _init_db(db)
        conn = _sql.connect(db)
        conn.execute(
            "INSERT INTO memories (id, content, metadata) VALUES (?,?,?)",
            ("empty", "", _json.dumps({"source": "traffic_learner"})),
        )
        conn.execute(
            "INSERT INTO memories (id, content, metadata) VALUES (?,?,?)",
            (
                "ok",
                "normal bullet",
                _json.dumps({"source": "traffic_learner", "category": "environment"}),
            ),
        )
        conn.commit()
        conn.close()

        backend = _FakeBackend(db)
        learner = TrafficLearner(backend=backend, min_evidence=1)
        await learner._hydrate_persisted_state()

        assert "empty" not in learner._persisted_ids.values()
        assert "ok" in learner._persisted_ids.values()

    @pytest.mark.asyncio
    async def test_hydrate_invalid_category_falls_back(self, tmp_path):
        """Unknown category values (e.g., typos) are handled as legacy rows."""
        import sqlite3 as _sql

        db = tmp_path / "memory.db"
        _init_db(db)
        import json as _json

        conn = _sql.connect(db)
        conn.execute(
            "INSERT INTO memories (id, content, metadata) VALUES (?,?,?)",
            (
                "bad-cat",
                "mystery bullet",
                _json.dumps({"source": "traffic_learner", "category": "mystery_type"}),
            ),
        )
        conn.commit()
        conn.close()

        backend = _FakeBackend(db)
        learner = TrafficLearner(backend=backend, min_evidence=1)
        # Must not raise.
        await learner._hydrate_persisted_state()


class TestCollectAllPatternsTimestamps:
    """_collect_all_patterns bumps last_seen_at on in-session re-sightings."""

    @pytest.mark.asyncio
    async def test_re_sighting_bumps_last_seen_at(self, tmp_path):
        """A persisted pattern re-observed in this session gets last_seen_at=now."""
        import json as _json
        import sqlite3 as _sql

        db = tmp_path / "memory.db"
        _init_db(db)
        old_last_seen = "2026-01-01T00:00:00+00:00"
        conn = _sql.connect(db)
        conn.execute(
            "INSERT INTO memories (id, content, metadata) VALUES (?,?,?)",
            (
                "seed-1",
                "some env bullet",
                _json.dumps(
                    {
                        "source": "traffic_learner",
                        "category": "environment",
                        "evidence_count": 1,
                        "first_seen_at": old_last_seen,
                        "last_seen_at": old_last_seen,
                    }
                ),
            ),
        )
        conn.commit()
        conn.close()

        backend = _FakeBackend(db)
        learner = TrafficLearner(backend=backend, min_evidence=1)

        # Simulate in-session accumulation of the same pattern.
        pattern = ExtractedPattern(
            category=PatternCategory.ENVIRONMENT,
            content="some env bullet",
            importance=0.7,
        )
        learner._pattern_counts[pattern.content_hash] = (pattern, 1)

        merged = learner._collect_all_patterns()
        assert len(merged) == 1
        m = merged[0]
        assert m.last_seen_at is not None
        # last_seen_at should be bumped past the stale 2026-01 timestamp.
        assert m.last_seen_at.year == datetime.now(UTC).year
        assert m.last_seen_at > _parse_iso_timestamp(old_last_seen)


# =============================================================================
# Regression tests for GH #464:
#   * <system-reminder> blocks must not feed _extract_preferences
#   * correction capture groups must end on a sentence boundary, not on a
#     fixed-length window
# =============================================================================


class TestStripSystemReminders:
    """Verify the literal-scan stripper does what the regex would do without
    introducing a new regex pattern into the learner."""

    def test_empty_and_no_tag_passthrough(self) -> None:
        assert TrafficLearner._strip_system_reminders("") == ""
        assert TrafficLearner._strip_system_reminders("hello world") == "hello world"

    def test_basic_strip(self) -> None:
        assert (
            TrafficLearner._strip_system_reminders("a<system-reminder>X</system-reminder>b") == "ab"
        )

    def test_case_insensitive_tag_name(self) -> None:
        assert (
            TrafficLearner._strip_system_reminders("a<System-Reminder>X</system-reminder>b") == "ab"
        )

    def test_multiple_reminders(self) -> None:
        text = "a<system-reminder>X</system-reminder>b<system-reminder>Y</system-reminder>c"
        assert TrafficLearner._strip_system_reminders(text) == "abc"

    def test_unclosed_reminder_drops_to_eos(self) -> None:
        # Malformed input — we'd rather drop than persist scaffolding.
        assert TrafficLearner._strip_system_reminders("hello <system-reminder>oops") == "hello "

    def test_realworld_colgrep_reminder(self) -> None:
        # The exact shape that produced 25× duplicate "User preference: of
        # Grep, Glob..." in the reporter's DB.
        text = (
            "<system-reminder>use colgrep instead of Grep, Glob. When spawning "
            "agents, mention colgrep features actively.</system-reminder>"
            "What is 2+2?"
        )
        assert TrafficLearner._strip_system_reminders(text) == "What is 2+2?"


class TestExtractPreferencesSystemReminderFiltering:
    """The high-value half of GH #464: system-reminder text must never flow
    into the preference extractor."""

    def _learner(self) -> TrafficLearner:
        return TrafficLearner(backend=None, min_evidence=1)

    def test_colgrep_reminder_yields_no_preference(self) -> None:
        learner = self._learner()
        text = (
            "<system-reminder>use colgrep instead of Grep, Glob. When spawning "
            "agents, mention colgrep features actively.</system-reminder>"
            "Hi there"
        )
        assert learner._extract_preferences(text) == []

    def test_observation_tag_reminder_yields_no_preference(self) -> None:
        learner = self._learner()
        text = (
            "<system-reminder>do not use <observation> tags. <observation> "
            "output will be DISCARDED and never reach the user.</system-reminder>"
            "Hello"
        )
        assert learner._extract_preferences(text) == []

    def test_dont_mention_reminder_yields_no_preference(self) -> None:
        learner = self._learner()
        text = (
            "<system-reminder>don't mention this reminder to the user.</system-reminder>List files"
        )
        assert learner._extract_preferences(text) == []

    def test_never_force_push_reminder_yields_no_preference(self) -> None:
        learner = self._learner()
        text = (
            "<system-reminder>never use git push --force on the main branch."
            "</system-reminder>OK got it"
        )
        assert learner._extract_preferences(text) == []


class TestExtractPreferencesRealCorrections:
    """Make sure the noise filter does not eat genuine user corrections."""

    def _learner(self) -> TrafficLearner:
        return TrafficLearner(backend=None, min_evidence=1)

    def test_dont_correction_with_sentence_boundary(self) -> None:
        learner = self._learner()
        out = learner._extract_preferences("don't use double quotes in the SQL, use single quotes.")
        assert len(out) == 1
        assert out[0].category is PatternCategory.PREFERENCE
        assert "double quotes" in out[0].content

    def test_no_use_correction(self) -> None:
        learner = self._learner()
        out = learner._extract_preferences("No, use httpx not requests.")
        assert len(out) == 1
        assert "httpx" in out[0].content

    def test_instead_correction(self) -> None:
        learner = self._learner()
        out = learner._extract_preferences("Instead, render the table with rich tables.")
        assert len(out) == 1
        assert "render the table" in out[0].content


class TestExtractPreferencesSentenceBoundary:
    """The tighter capture group must reject mid-sentence rambling so we
    never persist fragments like ``of Grep, Glob. When spawning agents…``."""

    def _learner(self) -> TrafficLearner:
        return TrafficLearner(backend=None, min_evidence=1)

    def test_long_unbroken_paragraph_yields_no_preference(self) -> None:
        learner = self._learner()
        # 100+ chars after the trigger word with no '.', '!', '?', or
        # '\n' anywhere — the kind of payload that would have matched
        # the old ``.{10,100}`` regex and produced a mid-word
        # truncation. The new bound forbids it: we need a terminator
        # OR end-of-string within 98 chars of the trigger.
        long_no_terminator = (
            "don't use Grep when running benchmarks because it floods the output "
            "buffer with a lot of irrelevant context that"
        )
        assert learner._extract_preferences(long_no_terminator) == []

    def test_short_utterance_without_terminator_still_matches(self) -> None:
        # Relaxation: a short user utterance without trailing
        # punctuation is a complete thought, not a truncation. End-of-
        # input counts as a boundary as long as the captured length
        # fits the 8–98 char window.
        learner = self._learner()
        out = learner._extract_preferences("don't use git push, I'll push manually")
        assert len(out) == 1
        assert "git push" in out[0].content

    def test_terminator_inside_window_captures_to_terminator(self) -> None:
        learner = self._learner()
        # The capture should end at the first '.', not include the
        # following sentence.
        out = learner._extract_preferences(
            "don't use Grep at all. Use ripgrep instead because it is faster."
        )
        assert len(out) == 1
        content = out[0].content
        assert "Use ripgrep instead" not in content
        assert "Grep" in content

    def test_trailing_terminator_is_stripped(self) -> None:
        learner = self._learner()
        out = learner._extract_preferences("Never commit secrets to git.")
        assert len(out) == 1
        # Pref must not end on its sentence terminator.
        assert not out[0].content.endswith(".")
        assert not out[0].content.endswith("!")
        assert not out[0].content.endswith("?")
