from __future__ import annotations

from types import SimpleNamespace

import pytest

import headroom.transforms.content_router as content_router_module
from headroom.transforms.content_detector import ContentType, DetectionResult
from headroom.transforms.content_router import (
    CompressionCache,
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
    RoutingDecision,
    _create_content_signature,
    _detect_content,
    _extract_json_block,
    is_mixed_content,
    split_into_sections,
)


def test_compression_cache_handles_hits_skips_evictions_and_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 112.0, 112.0])
    monkeypatch.setattr(content_router_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(content_router_module.time, "perf_counter_ns", lambda: 50)

    cache = CompressionCache(ttl_seconds=10)
    cache.put(1, "compressed", 0.4, "text")
    cache.mark_skip(2)

    assert cache.get(1) == ("compressed", 0.4, "text")
    assert cache.is_skipped(2) is True
    assert cache.size == 1
    assert cache.skip_size == 1

    cache.move_to_skip(1)
    assert cache.get(1) is None
    assert cache.is_skipped(1) is True

    # Expire both skip entries
    assert cache.is_skipped(2) is False
    assert cache.is_skipped(1) is False

    assert cache.stats["cache_hits"] == 1
    assert cache.stats["cache_skip_hits"] == 2
    assert cache.stats["cache_misses"] == 1
    assert cache.stats["cache_evictions"] >= 2

    cache.clear()
    assert cache.size == 0
    assert cache.skip_size == 0


def test_router_result_helpers_and_summary() -> None:
    pure = RouterCompressionResult(
        compressed="small",
        original="very large",
        strategy_used=CompressionStrategy.TEXT,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.PLAIN_TEXT,
                strategy=CompressionStrategy.TEXT,
                original_tokens=10,
                compressed_tokens=4,
            )
        ],
    )
    assert pure.total_original_tokens == 10
    assert pure.total_compressed_tokens == 4
    assert pure.compression_ratio == 0.4
    assert pure.tokens_saved == 6
    assert pure.savings_percentage == 60.0
    assert pure.summary() == "Pure text: 10→4 tokens (60% saved)"

    mixed = RouterCompressionResult(
        compressed="joined",
        original="original",
        strategy_used=CompressionStrategy.MIXED,
        sections_processed=2,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.PLAIN_TEXT,
                strategy=CompressionStrategy.TEXT,
                original_tokens=0,
                compressed_tokens=0,
            ),
            RoutingDecision(
                content_type=ContentType.SEARCH_RESULTS,
                strategy=CompressionStrategy.SEARCH,
                original_tokens=8,
                compressed_tokens=2,
            ),
        ],
    )
    assert mixed.routing_log[0].compression_ratio == 1.0
    assert mixed.summary().startswith("Mixed content: 2 sections, routed to ")


def test_content_signature_and_detection_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage-3d (PR5) wired `_detect_content` through the Rust chain
    (`headroom._core.detect_content_type` → magika → unidiff →
    PlainText). The pre-PR5 Python-side `_get_magika_detector`
    fallback path is gone.

    This test asserts the new contract:
    1. The detection helper delegates to the Rust binding.
    2. Whatever `ContentType` the Rust side returns flows back as a
       Python `DetectionResult` with that same `content_type`.
    """
    signature = _create_content_signature("search", "file.py:10:match", language="python")
    assert signature is not None
    assert len(signature.structure_hash) == 24

    # Monkeypatch the Rust binding to return a deterministic fake
    # result; verify _detect_content propagates the content_type
    # tag back as the Python ContentType enum.
    import headroom._core as _core

    # Pin the Rust backend so this test exercises the native delegation
    # path on every platform (Windows now defaults to the pure-Python
    # detector — see content_router._resolve_detect_backend).
    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "rust")

    fake_rust_result = SimpleNamespace(
        content_type="source_code",
        confidence=1.0,
        metadata={},
    )
    monkeypatch.setattr(_core, "detect_content_type", lambda content: fake_rust_result)

    result = _detect_content("def main(): pass")
    assert result.content_type is ContentType.SOURCE_CODE
    assert result.confidence == 1.0
    assert result.metadata == {}


def test_mixed_content_section_splitting_and_json_extraction() -> None:
    content = "\n".join(
        [
            "Intro paragraph with Several words included for prose detection.",
            "Another line with enough words to read as normal prose today.",
            "Third line adds more prose so the detector sees real text content.",
            "Fourth sentence keeps the count moving higher for prose patterns.",
            "Fifth sentence does the same for mixed content identification.",
            "Sixth sentence seals the prose threshold for the helper.",
            "```python",
            "def main():",
            "    return 1",
            "```",
            '[{"id": 1}]',
            "src/app.py:10:def main():",
            "src/app.py:11:return 1",
        ]
    )
    assert is_mixed_content(content) is True

    sections = split_into_sections(content)
    assert [section.content_type for section in sections] == [
        ContentType.PLAIN_TEXT,
        ContentType.SOURCE_CODE,
        ContentType.JSON_ARRAY,
        ContentType.SEARCH_RESULTS,
    ]
    assert sections[1].language == "python"
    assert sections[1].is_code_fence is True
    assert sections[2].content == '[{"id": 1}]'
    assert sections[3].end_line == 12

    json_block, end_idx = _extract_json_block(["[", '{"id": 1}', "]"], 0)
    assert json_block == '[\n{"id": 1}\n]'
    assert end_idx == 2
    assert _extract_json_block(["{", '"a": 1'], 0) == (None, 0)


def test_extract_json_block_ignores_brackets_inside_strings() -> None:
    """Brackets/braces inside JSON string values must not end the block early.

    Regression: counting raw ``[``/``]``/``{``/``}`` per line treated the
    ``]`` inside ``{"path": "a]b"}`` as a closing bracket, so the array was
    truncated mid-way and the remaining rows leaked into later sections.
    """
    import json as _json

    lines = [
        "[",
        '  {"path": "a]b"},',
        '  {"path": "c"}',
        "]",
    ]
    block, end_idx = _extract_json_block(lines, 0)
    assert end_idx == 3
    assert block is not None
    parsed = _json.loads(block)
    assert parsed == [{"path": "a]b"}, {"path": "c"}]

    # Braces inside a string value must likewise be ignored.
    obj_lines = [
        "{",
        '  "msg": "use {curly} and [square]",',
        '  "n": 1',
        "}",
    ]
    obj_block, obj_end = _extract_json_block(obj_lines, 0)
    assert obj_end == 3
    assert obj_block is not None
    assert _json.loads(obj_block) == {"msg": "use {curly} and [square]", "n": 1}


def test_split_into_sections_keeps_json_array_with_bracket_in_string() -> None:
    """A JSON array embedded in prose stays one JSON section, not fragments.

    With the bracket-in-string bug, the array below split into a truncated
    JSON section plus a stray ``]`` glued onto the trailing prose.
    """
    import json as _json

    content = "\n".join(
        [
            "prose line here that is long enough to matter",
            "[",
            '  {"path": "a]b"},',
            '  {"path": "c"}',
            "]",
            "trailing prose",
        ]
    )

    sections = split_into_sections(content)
    json_sections = [s for s in sections if s.content_type == ContentType.JSON_ARRAY]
    assert len(json_sections) == 1
    parsed = _json.loads(json_sections[0].content)
    assert parsed == [{"path": "a]b"}, {"path": "c"}]


def test_content_router_strategy_and_compress_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    router = ContentRouter(ContentRouterConfig(prefer_code_aware_for_code=False))

    monkeypatch.setattr(content_router_module, "is_mixed_content", lambda content: False)
    monkeypatch.setattr(
        content_router_module,
        "_detect_content",
        lambda content: DetectionResult(ContentType.SOURCE_CODE, 1.0, {}),
    )
    assert router._determine_strategy("code") is CompressionStrategy.KOMPRESS
    assert (
        router._strategy_from_detection(DetectionResult(ContentType.SEARCH_RESULTS, 1.0, {}))
        is CompressionStrategy.SEARCH
    )
    assert router._strategy_from_detection_type(ContentType.GIT_DIFF) is CompressionStrategy.DIFF
    assert (
        router._content_type_from_strategy(CompressionStrategy.PASSTHROUGH)
        is ContentType.PLAIN_TEXT
    )

    mixed_result = RouterCompressionResult(
        compressed="mixed",
        original="mixed",
        strategy_used=CompressionStrategy.MIXED,
    )
    pure_result = RouterCompressionResult(
        compressed="pure",
        original="pure",
        strategy_used=CompressionStrategy.TEXT,
    )
    monkeypatch.setattr(router, "_compress_mixed", lambda *args, **kwargs: mixed_result)
    monkeypatch.setattr(router, "_compress_pure", lambda *args, **kwargs: pure_result)

    monkeypatch.setattr(router, "_determine_strategy", lambda content: CompressionStrategy.MIXED)
    assert router.compress("mixed") is mixed_result

    monkeypatch.setattr(router, "_determine_strategy", lambda content: CompressionStrategy.TEXT)
    assert router.compress("pure") is pure_result
    assert router.compress("   ").strategy_used is CompressionStrategy.PASSTHROUGH


def test_force_kompress_bypasses_content_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    router = ContentRouter()
    router._runtime_force_kompress = True
    pure_result = RouterCompressionResult(
        compressed="pure",
        original="pure",
        strategy_used=CompressionStrategy.KOMPRESS,
    )

    monkeypatch.setattr(
        content_router_module,
        "is_mixed_content",
        lambda content: (_ for _ in ()).throw(AssertionError("mixed detection called")),
    )
    monkeypatch.setattr(
        content_router_module,
        "_detect_content",
        lambda content: (_ for _ in ()).throw(AssertionError("content detection called")),
    )
    monkeypatch.setattr(router, "_determine_strategy", lambda content: CompressionStrategy.MIXED)
    monkeypatch.setattr(router, "_compress_pure", lambda *args, **kwargs: pure_result)

    assert router.compress("large tool output") is pure_result


def test_normal_compress_path_still_uses_content_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ContentRouter()
    calls = {"mixed": 0, "detect": 0}
    pure_result = RouterCompressionResult(
        compressed="pure",
        original="pure",
        strategy_used=CompressionStrategy.TEXT,
    )

    def _fake_mixed(content: str) -> bool:
        calls["mixed"] += 1
        return False

    def _fake_detect(content: str) -> DetectionResult:
        calls["detect"] += 1
        return DetectionResult(ContentType.PLAIN_TEXT, 1.0, {})

    monkeypatch.setattr(content_router_module, "is_mixed_content", _fake_mixed)
    monkeypatch.setattr(content_router_module, "_detect_content", _fake_detect)
    monkeypatch.setattr(router, "_compress_pure", lambda *args, **kwargs: pure_result)

    assert router.compress("plain text") is pure_result
    assert calls["mixed"] > 0
    assert calls["detect"] > 0


def test_force_kompress_apply_uses_lightweight_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        def count_text(self, text: str) -> int:
            return len(text.split())

    router = ContentRouter(ContentRouterConfig(protect_recent_code=2))
    content = " ".join(["plain text payload"] * 80)

    monkeypatch.setattr(
        content_router_module,
        "_detect_content",
        lambda content: (_ for _ in ()).throw(AssertionError("content detection called")),
    )
    monkeypatch.setattr(
        content_router_module,
        "_regex_detect_content_type",
        lambda content: DetectionResult(ContentType.PLAIN_TEXT, 1.0, {}),
    )
    monkeypatch.setattr(
        router,
        "compress",
        lambda content, context="", bias=1.0: RouterCompressionResult(
            # CCR marker -> the original was stored and is retrievable, so the
            # #1307 reversibility gate accepts this lossy KOMPRESS tool result.
            compressed="compressed <<ccr:tool>>",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
            routing_log=[
                RoutingDecision(
                    content_type=ContentType.PLAIN_TEXT,
                    strategy=CompressionStrategy.KOMPRESS,
                    original_tokens=len(content.split()),
                    compressed_tokens=1,
                )
            ],
        ),
    )

    result = router.apply(
        [{"role": "tool", "content": content}],
        FakeTokenizer(),
        force_kompress=True,
        min_tokens_to_compress=10,
        protect_recent=2,
    )

    assert result.messages[0]["content"] == "compressed <<ccr:tool>>"


def test_force_kompress_apply_lightweight_detection_protects_recent_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        def count_text(self, text: str) -> int:
            return len(text.split())

    router = ContentRouter(ContentRouterConfig(protect_recent_code=2))
    content = "\n".join(
        [
            "def generated_function(value):",
            "    if value:",
            "        return str(value)",
        ]
        * 40
    )

    monkeypatch.setattr(
        content_router_module,
        "_detect_content",
        lambda content: (_ for _ in ()).throw(AssertionError("content detection called")),
    )
    monkeypatch.setattr(
        router,
        "compress",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("recent code should be protected")
        ),
    )

    result = router.apply(
        [{"role": "tool", "content": content}],
        FakeTokenizer(),
        force_kompress=True,
        min_tokens_to_compress=10,
        protect_recent=2,
    )

    assert result.messages[0]["content"] == content
    assert result.transforms_applied == ["router:protected:recent_code"]


def test_content_router_mixed_pure_apply_and_toin(monkeypatch: pytest.MonkeyPatch) -> None:
    router = ContentRouter()
    mixed_content = "\n".join(["before", "```python", "print('x')", "```", "after"])
    monkeypatch.setattr(
        content_router_module,
        "split_into_sections",
        lambda content: [
            SimpleNamespace(
                content="print('x')",
                content_type=ContentType.SOURCE_CODE,
                language="python",
                is_code_fence=True,
            ),
            SimpleNamespace(
                content="after text",
                content_type=ContentType.PLAIN_TEXT,
                language=None,
                is_code_fence=False,
            ),
        ],
    )
    monkeypatch.setattr(
        router,
        "_apply_strategy_to_content",
        lambda content, strategy, context, language=None, question=None, bias=1.0: (
            f"{strategy.value}:{content}",
            len(content.split()) - 1,
            [strategy.value],
        ),
    )
    result = router._compress_mixed(mixed_content, "ctx")
    assert result.strategy_used is CompressionStrategy.MIXED
    assert result.sections_processed == 2
    assert "```python\ncode_aware:print('x')\n```" in result.compressed

    monkeypatch.setattr(
        router,
        "_apply_strategy_to_content",
        lambda content, strategy, context, language=None, question=None, bias=1.0: (
            "shrunk",
            1,
            [strategy.value],
        ),
    )
    pure = router._compress_pure("some plain text", CompressionStrategy.TEXT, "ctx")
    assert pure.routing_log[0].content_type is ContentType.PLAIN_TEXT
    assert pure.total_original_tokens == 3
    assert pure.total_compressed_tokens == 1

    calls: list[dict] = []
    router._toin = SimpleNamespace(record_compression=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(content_router_module, "_create_content_signature", lambda **kwargs: "sig")
    router._record_to_toin(
        CompressionStrategy.TEXT,
        "original content",
        "small",
        original_tokens=10,
        compressed_tokens=4,
        language="python",
        context="question",
    )
    assert calls[0]["tool_signature"] == "sig"
    assert calls[0]["strategy"] == "text"
    assert calls[0]["query_context"] == "question"

    router._record_to_toin(
        CompressionStrategy.SMART_CRUSHER,
        "x",
        "x",
        original_tokens=10,
        compressed_tokens=4,
    )
    router._record_to_toin(
        CompressionStrategy.TEXT,
        "x",
        "x",
        original_tokens=2,
        compressed_tokens=2,
    )
    monkeypatch.setattr(content_router_module, "_create_content_signature", lambda **kwargs: None)
    router._record_to_toin(
        CompressionStrategy.TEXT,
        "x",
        "y",
        original_tokens=5,
        compressed_tokens=1,
    )
    assert len(calls) == 1


def test_diff_strategy_does_not_fallback_to_kompress_when_diff_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ContentRouter()
    diff = "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+a"

    class NoopDiffCompressor:
        def compress(self, content: str, context: str = "") -> SimpleNamespace:
            return SimpleNamespace(compressed=content)

    monkeypatch.setattr(router, "_get_diff_compressor", lambda: NoopDiffCompressor())

    def fail_kompress(*_args: object, **_kwargs: object) -> tuple[str, int]:
        raise AssertionError("Diff compression must not fallback to Kompress")

    monkeypatch.setattr(router, "_try_ml_compressor", fail_kompress)

    compressed, compressed_tokens, strategy_chain = router._apply_strategy_to_content(
        diff,
        CompressionStrategy.DIFF,
        context="",
    )

    assert compressed == diff
    assert compressed_tokens == len(diff.split())
    assert strategy_chain == ["diff"]


def test_log_strategy_does_not_fallback_to_kompress_when_log_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ContentRouter()
    log = "ERROR one\nERROR two\nERROR three"

    class NoopLogCompressor:
        def compress(self, content: str, bias: float = 1.0) -> SimpleNamespace:
            return SimpleNamespace(compressed=content)

    monkeypatch.setattr(router, "_get_log_compressor", lambda: NoopLogCompressor())

    def fail_kompress(*_args: object, **_kwargs: object) -> tuple[str, int]:
        raise AssertionError("Log compression must not fallback to Kompress")

    monkeypatch.setattr(router, "_try_ml_compressor", fail_kompress)

    compressed, compressed_tokens, strategy_chain = router._apply_strategy_to_content(
        log,
        CompressionStrategy.LOG,
        context="",
    )

    assert compressed == log
    assert compressed_tokens == len(log.split())
    assert strategy_chain == ["log"]


# ---------------------------------------------------------------------------
# Cache-safety tests for _process_content_blocks. These pin down the
# block-level invariants that protect upstream prefix caches:
#
#   * cache_control on a block is the client's explicit cache breakpoint —
#     never modified, regardless of role/type.
#   * assistant text blocks are part of the cache prefix in subsequent
#     turns; default-skipped, opt-in via compress_assistant_text_blocks.
#   * user/system text blocks are the prompt; never modified.
#   * tool/function text blocks are tool outputs; freely compressed.
#   * min_chars threshold gates short blocks.
# ---------------------------------------------------------------------------


def _make_router_with_mock_compress(monkeypatch: pytest.MonkeyPatch) -> ContentRouter:
    """Return a ContentRouter whose compress() always emits a half-length
    ``[compressed]`` payload at ratio 0.5 (passes the < min_ratio check)."""
    router = ContentRouter(ContentRouterConfig())

    def fake_compress(content, context: str = "", bias: float = 1.0):
        return SimpleNamespace(
            compressed=content[: len(content) // 2] + "[compressed]",
            compression_ratio=0.5,
            strategy_used=SimpleNamespace(value="text"),
        )

    monkeypatch.setattr(router, "compress", fake_compress)
    return router


def test_text_block_cache_control_protected_with_assistant_optin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _make_router_with_mock_compress(monkeypatch)
    long_text = "A" * 1000
    msg = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": long_text, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "B" * 1000},
        ],
    }
    counts: dict[str, int] = {
        "excluded_tool": 0,
        "user_msg": 0,
        "small": 0,
        "recent_code": 0,
        "analysis_ctx": 0,
        "ratio_too_high": 0,
        "non_string": 0,
        "content_blocks": 0,
    }
    result = router._process_content_blocks(
        msg,
        msg["content"],
        "",
        [],
        set(),
        route_counts=counts,
        compress_assistant_text_blocks=True,
    )
    blocks = result["content"]
    # cache_control'd block: untouched (defense in depth)
    assert blocks[0] == msg["content"][0]
    assert blocks[0]["text"] == long_text
    # Sibling non-cache_control'd block: compressed under opt-in
    assert "[compressed]" in blocks[1]["text"]
    assert counts["cache_control_protected"] == 1


def test_tool_result_cache_control_protected(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _make_router_with_mock_compress(monkeypatch)
    long_text = "Z" * 1000
    msg = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "abc",
                "content": long_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }
    result = router._process_content_blocks(
        msg,
        msg["content"],
        "",
        [],
        set(),
    )
    # cache_control hard-skip applies to tool_result too
    assert result["content"][0]["content"] == long_text


def test_assistant_text_blocks_skipped_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _make_router_with_mock_compress(monkeypatch)
    long_text = "X" * 1000
    msg = {"role": "assistant", "content": [{"type": "text", "text": long_text}]}
    result = router._process_content_blocks(
        msg,
        msg["content"],
        "",
        [],
        set(),
    )
    # Default OFF: assistant text untouched, restoring pre-#431 cache safety
    assert result["content"][0]["text"] == long_text


def test_assistant_text_blocks_opt_in_compresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _make_router_with_mock_compress(monkeypatch)
    long_text = "Y" * 1000
    msg = {"role": "assistant", "content": [{"type": "text", "text": long_text}]}
    result = router._process_content_blocks(
        msg,
        msg["content"],
        "",
        [],
        set(),
        compress_assistant_text_blocks=True,
    )
    assert "[compressed]" in result["content"][0]["text"]


def test_user_text_blocks_never_compressed_even_with_assistant_optin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _make_router_with_mock_compress(monkeypatch)
    long_text = "U" * 1000
    msg = {"role": "user", "content": [{"type": "text", "text": long_text}]}
    result = router._process_content_blocks(
        msg,
        msg["content"],
        "",
        [],
        set(),
        compress_assistant_text_blocks=True,  # MUST NOT bleed into user
    )
    assert result["content"][0]["text"] == long_text


def test_system_text_blocks_skipped_when_skip_system_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _make_router_with_mock_compress(monkeypatch)
    long_text = "S" * 1000
    msg = {"role": "system", "content": [{"type": "text", "text": long_text}]}
    result = router._process_content_blocks(
        msg,
        msg["content"],
        "",
        [],
        set(),
        skip_system=True,
        compress_assistant_text_blocks=True,
    )
    assert result["content"][0]["text"] == long_text


def test_tool_role_text_blocks_compressed_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _make_router_with_mock_compress(monkeypatch)
    long_text = "T" * 1000
    msg = {"role": "tool", "content": [{"type": "text", "text": long_text}]}
    result = router._process_content_blocks(
        msg,
        msg["content"],
        "",
        [],
        set(),
    )
    # tool role ≈ tool output — compress freely
    assert "[compressed]" in result["content"][0]["text"]


def test_unknown_role_text_blocks_skipped_for_safety(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _make_router_with_mock_compress(monkeypatch)
    long_text = "Q" * 1000
    msg = {"role": "developer", "content": [{"type": "text", "text": long_text}]}
    result = router._process_content_blocks(
        msg,
        msg["content"],
        "",
        [],
        set(),
        compress_assistant_text_blocks=True,
    )
    # Unknown role: be safe, don't compress
    assert result["content"][0]["text"] == long_text


def test_min_chars_gates_short_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _make_router_with_mock_compress(monkeypatch)
    short_text = "tiny"
    msg = {"role": "tool", "content": [{"type": "text", "text": short_text}]}
    result = router._process_content_blocks(
        msg,
        msg["content"],
        "",
        [],
        set(),
        min_chars=500,
    )
    assert result["content"][0]["text"] == short_text


def test_pinning_skips_already_compressed(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _make_router_with_mock_compress(monkeypatch)
    pinned = "Retrieve more: hash=abc " + "x" * 1000
    msg = {"role": "tool", "content": [{"type": "text", "text": pinned}]}
    result = router._process_content_blocks(
        msg,
        msg["content"],
        "",
        [],
        set(),
    )
    # Already-compressed marker keeps proxy idempotent across turns
    assert result["content"][0]["text"] == pinned


def test_detect_backend_env_python_forces_python_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HEADROOM_DETECT_BACKEND=python forces the pure-Python regex path."""
    import headroom._core as _core

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "python")

    called = []

    def _record(content: str):  # type: ignore[return]
        called.append(content)
        raise AssertionError("native must not be called with python backend")

    monkeypatch.setattr(_core, "detect_content_type", _record)

    # Should not raise — native detector must be bypassed entirely.
    result = _detect_content('[{"id": 1}]')
    assert result.content_type is ContentType.JSON_ARRAY
    assert called == [], "native detect_content_type was called despite python backend"


def test_detect_backend_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """HEADROOM_DETECT_BACKEND pins the detector on any platform."""
    resolve = content_router_module._resolve_detect_backend

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "python")
    assert resolve() == "python"

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "RUST")  # case-insensitive
    assert resolve() == "rust"

    # Unrecognized values fall back to the platform default.
    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "bogus")
    monkeypatch.setattr(content_router_module.sys, "platform", "linux")
    assert resolve() == "rust"


def test_detect_backend_defaults_to_python_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows defaults to the pure-Python detector (native ONNX hang, #845)."""
    monkeypatch.delenv("HEADROOM_DETECT_BACKEND", raising=False)

    monkeypatch.setattr(content_router_module.sys, "platform", "win32")
    assert content_router_module._resolve_detect_backend() == "python"

    monkeypatch.setattr(content_router_module.sys, "platform", "linux")
    assert content_router_module._resolve_detect_backend() == "rust"


def test_detect_content_python_backend_skips_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The python backend must not touch the native detector at all."""
    import headroom._core as _core

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "python")

    def _boom(_content: str) -> None:
        raise AssertionError("native detector must not be called")

    monkeypatch.setattr(_core, "detect_content_type", _boom)

    result = _detect_content('[{"id": 1}, {"id": 2}]')
    assert result.content_type is ContentType.JSON_ARRAY
