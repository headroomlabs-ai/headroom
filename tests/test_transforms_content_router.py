from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

import headroom._ort as ort_runtime
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
    _estimate_tokens,
    _extract_json_block,
    _strip_detection_envelope,
    is_mixed_content,
    split_into_sections,
)


@pytest.fixture(autouse=True)
def _reset_detect_module_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the module-level detect flags from leaking across tests.

    The circuit breaker (#575) is process-wide, so a test that trips it would
    otherwise force later tests onto the pure-Python path. ``monkeypatch.setattr``
    zeroes each flag for the test and auto-restores it afterward.
    """
    monkeypatch.setattr(content_router_module, "_detect_native_unhealthy", False)
    monkeypatch.setattr(content_router_module, "_detect_backend_warned", False)
    monkeypatch.setattr(content_router_module, "_detect_panic_warned", False)
    # Router unit tests replace ``headroom._core.detect_content_type`` with
    # deterministic fakes. Keep the separate ORT API-compatibility preflight
    # open so those fakes reach the watchdog/circuit-breaker behavior under
    # test; incompatibility itself is covered in test_ort_dylib.py (#2960).
    monkeypatch.setattr(ort_runtime, "rust_ort_runtime_compatible", lambda: True)


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


def test_native_detection_remains_bounded_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful native call must not disable the watchdog for later calls."""
    import headroom._core as _core

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "rust")
    monkeypatch.setattr(content_router_module, "_detect_timeout_secs", lambda: 0.01)
    calls = 0

    def _succeeds_then_hangs(_content: str) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(content_type="plain_text")
        threading.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(_core, "detect_content_type", _succeeds_then_hangs)

    assert _detect_content("first").content_type is ContentType.PLAIN_TEXT
    assert _detect_content("second").content_type is ContentType.PLAIN_TEXT
    assert content_router_module._detect_native_unhealthy is True


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


def test_short_instruction_with_embedded_json_compresses_without_kompress() -> None:
    rows = [
        {
            "id": i,
            "source": "opencode",
            "status": "ok",
            "message": "routine event with repeated fields",
            "needle": "CRITICAL_NEEDLE_42" if i == 17 else "ordinary",
        }
        for i in range(40)
    ]
    content = "\n".join(
        [
            "Compress this OpenCode context. Preserve critical values.",
            json.dumps(rows, indent=2),
            "Return a concise answer.",
        ]
    )
    router = ContentRouter(
        ContentRouterConfig(
            enable_kompress=False,
            skip_user_messages=False,
        )
    )

    assert is_mixed_content(content) is True
    assert router._determine_strategy(content) is CompressionStrategy.MIXED

    result = router.compress(content, context="CRITICAL_NEEDLE_42 opencode")

    assert result.strategy_used is CompressionStrategy.MIXED
    assert result.tokens_saved > 0
    assert "CRITICAL_NEEDLE_42" in result.compressed
    assert any(
        decision.strategy is CompressionStrategy.SMART_CRUSHER for decision in result.routing_log
    )


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
    assert router._determine_strategy("code") is CompressionStrategy.PASSTHROUGH
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

    monkeypatch.setattr(
        router, "_determine_strategy", lambda content, **_kwargs: CompressionStrategy.MIXED
    )
    assert router.compress("mixed") is mixed_result

    monkeypatch.setattr(
        router, "_determine_strategy", lambda content, **_kwargs: CompressionStrategy.TEXT
    )
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


@pytest.mark.parametrize(
    ("content_type", "multiplier"),
    [
        (ContentType.BUILD_OUTPUT, 0.50),
        (ContentType.SOURCE_CODE, 0.50),
        (ContentType.SEARCH_RESULTS, 0.70),
        (ContentType.PLAIN_TEXT, 1.00),
    ],
)
def test_domain_routing_scales_detected_content_bias(
    monkeypatch: pytest.MonkeyPatch,
    content_type: ContentType,
    multiplier: float,
) -> None:
    router = ContentRouter()
    observed: dict[str, float] = {}

    monkeypatch.setattr(content_router_module, "is_mixed_content", lambda _content: False)
    monkeypatch.setattr(
        content_router_module,
        "_detect_content",
        lambda _content: DetectionResult(content_type, 1.0, {}),
    )
    monkeypatch.setattr(
        router,
        "_determine_strategy",
        lambda _content, **_kwargs: CompressionStrategy.KOMPRESS,
    )

    def _capture_pure(*args, **kwargs) -> RouterCompressionResult:
        observed["bias"] = kwargs["bias"]
        return RouterCompressionResult(
            compressed="compressed",
            original="payload",
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    monkeypatch.setattr(router, "_compress_pure", _capture_pure)

    router.compress("payload", bias=1.6)

    assert observed["bias"] == pytest.approx(1.6 * multiplier)


def test_domain_routing_does_not_change_forced_kompress_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ContentRouter()
    router._runtime_force_kompress = True
    observed: dict[str, float] = {}

    def _capture_pure(*args, **kwargs) -> RouterCompressionResult:
        observed["bias"] = kwargs["bias"]
        return RouterCompressionResult(
            compressed="compressed",
            original="payload",
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    monkeypatch.setattr(router, "_compress_pure", _capture_pure)

    router.compress("payload", bias=1.6)

    assert observed["bias"] == pytest.approx(1.6)


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
        lambda content, context="", bias=1.0, precomputed_detection=None: RouterCompressionResult(
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
        lambda content, isolate=(): [
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
    assert pure.total_original_tokens == _estimate_tokens("some plain text")
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
    assert compressed_tokens == _estimate_tokens(diff)
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
    assert compressed_tokens == _estimate_tokens(log)
    assert strategy_chain == ["log"]


def test_smart_crusher_log_fallback_skipped_for_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truncated/invalid JSON must not reach the Log fallback (#1306).

    The native magika detector tags content by shape, so a truncated JSON
    tool output is classified as ``json_array`` and routed to SMART_CRUSHER.
    SmartCrusher returns it unchanged (can't parse broken JSON); Kompress
    passes it through. Without the JSON-validity guard, the LogCompressor
    would then collapse the whole thing to a single CCR-retrieval marker —
    99.9% data loss when CCR retrieval isn't configured. The guard skips
    the Log fallback for invalid JSON so the content passes through verbatim.
    """
    router = ContentRouter(ContentRouterConfig())
    # Truncated JSON object: starts valid, cut mid-string so json.loads fails.
    truncated = '{"rows": [{"address": "Addr1", "name": "PoolA"}, {"address": "Addr2", "nam'

    # SmartCrusher returns no savings (can't parse) → Kompress is a no-op
    # (model not loaded in tests) → Log fallback must be SKIPPED.
    class CollapsingLogCompressor:
        def compress(self, content: str, bias: float = 1.0) -> SimpleNamespace:
            # Simulate the collapse: LogCompressor treats the broken JSON as
            # a multi-line "log" and reduces it to a retrieval marker.
            return SimpleNamespace(
                compressed="\n[2 lines compressed to 0. Retrieve more: hash=deadbeef]"
            )

    monkeypatch.setattr(router, "_get_log_compressor", lambda: CollapsingLogCompressor())
    # Kompress not ready → returns content unchanged (real behavior in tests).
    monkeypatch.setattr(
        router,
        "_try_ml_compressor",
        lambda content, context, question=None: (content, len(content.split())),
    )

    compressed, compressed_tokens, strategy_chain = router._apply_strategy_to_content(
        truncated,
        CompressionStrategy.SMART_CRUSHER,
        context="",
    )

    # Content preserved verbatim — no collapse.
    assert compressed == truncated
    assert CompressionStrategy.LOG.value not in strategy_chain
    assert CompressionStrategy.KOMPRESS.value in strategy_chain


def test_smart_crusher_log_fallback_runs_for_valid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid JSON arrays still reach the Log fallback when SmartCrusher no-ops.

    The JSON-validity guard (#1306) only blocks the Log fallback for
    *invalid* JSON. Valid JSON arrays that SmartCrusher can't shrink (e.g.
    repetitive JSONL the proxy's own log dumps) still get the Log fallback,
    which is the intended "repetitive JSONL" path.
    """
    router = ContentRouter(ContentRouterConfig())
    valid_json = '[{"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}]'

    class NoopSmartCrusher:
        def crush(self, content: str, query: str = "", bias: float = 1.0) -> SimpleNamespace:
            return SimpleNamespace(compressed=content)

    class ShrinkingLogCompressor:
        def compress(self, content: str, bias: float = 1.0) -> SimpleNamespace:
            return SimpleNamespace(compressed="[compressed]")

    monkeypatch.setattr(router, "_get_smart_crusher", lambda: NoopSmartCrusher())
    monkeypatch.setattr(router, "_get_log_compressor", lambda: ShrinkingLogCompressor())
    # Kompress no-op → Log fallback fires. A faithful no-op reports the same
    # token count the router computed for the (unchanged) content — using
    # _estimate_tokens, not a naive word split, so it isn't mistaken for a
    # saving once #1857's whitespace-aware counting rates the JSON above 8.
    monkeypatch.setattr(
        router,
        "_try_ml_compressor",
        lambda content, context, question=None: (content, _estimate_tokens(content)),
    )

    compressed, _compressed_tokens, strategy_chain = router._apply_strategy_to_content(
        valid_json,
        CompressionStrategy.SMART_CRUSHER,
        context="",
    )

    assert CompressionStrategy.LOG.value in strategy_chain
    assert compressed == "[compressed]"


# ---------------------------------------------------------------------------
# MIXED false-positive on source code: Python with dict literals + docstrings
# triggers ``has_json_blocks`` and ``has_prose`` in ``is_mixed_content``,
# misclassifying pure code as MIXED.  The native detector (magika) correctly
# identifies it as SOURCE_CODE.  ``_determine_strategy`` must trust the
# detector over the regex heuristics when confidence is high.
# ---------------------------------------------------------------------------


def test_mixed_false_positive_on_source_code_overridden_by_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python code with dicts + docstrings must not be routed to MIXED.

    ``is_mixed_content`` returns True (dict literals → has_json_blocks,
    docstrings → has_prose), but the native detector says SOURCE_CODE
    with high confidence.  The detector must win.
    """
    router = ContentRouter(ContentRouterConfig())

    # Simulate the false positive: is_mixed_content says True, detector
    # says SOURCE_CODE with confidence 1.0 (real magika behaviour).
    monkeypatch.setattr(content_router_module, "is_mixed_content", lambda content: True)
    monkeypatch.setattr(
        content_router_module,
        "_detect_content",
        lambda content: DetectionResult(ContentType.SOURCE_CODE, 1.0, {}),
    )

    strategy = router._determine_strategy("python code with dicts")
    # With prefer_code_aware_for_code=True (default), source code routes to CodeAware.
    assert strategy is CompressionStrategy.CODE_AWARE
    assert strategy is not CompressionStrategy.MIXED


def test_mixed_still_used_when_detector_says_plain_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Genuine mixed content (detector says PLAIN_TEXT) still uses MIXED.

    The override only fires when the detector says SOURCE_CODE.  For
    actual mixed content (prose + JSON + code fences), the detector
    returns PLAIN_TEXT or another non-code type, and MIXED is correct.
    """
    router = ContentRouter(ContentRouterConfig())

    monkeypatch.setattr(content_router_module, "is_mixed_content", lambda content: True)
    monkeypatch.setattr(
        content_router_module,
        "_detect_content",
        lambda content: DetectionResult(ContentType.PLAIN_TEXT, 0.9, {}),
    )

    strategy = router._determine_strategy("genuinely mixed content")
    assert strategy is CompressionStrategy.MIXED


def test_mixed_still_used_when_detector_confidence_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low-confidence SOURCE_CODE detection does not override MIXED.

    If the detector is uncertain (confidence < 0.8), we keep the MIXED
    path rather than risking a false override.
    """
    router = ContentRouter(ContentRouterConfig())

    monkeypatch.setattr(content_router_module, "is_mixed_content", lambda content: True)
    monkeypatch.setattr(
        content_router_module,
        "_detect_content",
        lambda content: DetectionResult(ContentType.SOURCE_CODE, 0.5, {}),
    )

    strategy = router._determine_strategy("uncertain code")
    assert strategy is CompressionStrategy.MIXED


def test_source_code_passthrough_preserves_content_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Code detected as SOURCE_CODE with CodeAware disabled → passthrough.

    When ``prefer_code_aware_for_code=False`` (default), code must pass
    through unmangled.  Previously it fell to KOMPRESS which can destroy
    code semantics (98% compression, 11% recall on large blobs).  Now
    it explicitly uses PASSTHROUGH.
    """
    router = ContentRouter(ContentRouterConfig(prefer_code_aware_for_code=False))

    monkeypatch.setattr(content_router_module, "is_mixed_content", lambda content: False)
    monkeypatch.setattr(
        content_router_module,
        "_detect_content",
        lambda content: DetectionResult(ContentType.SOURCE_CODE, 1.0, {}),
    )

    code = "def hello():\n    print('world')\n"
    result = router.compress(code)
    assert result.compressed == code
    assert result.strategy_used is CompressionStrategy.PASSTHROUGH


def test_source_code_code_aware_enabled_uses_code_aware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When CodeAware is explicitly enabled, it is used for source code.

    The PASSTHROUGH fallback only applies when ``prefer_code_aware_for_code``
    is False.  If a user explicitly enables it, CODE_AWARE is used.
    """
    router = ContentRouter(
        ContentRouterConfig(
            enable_code_aware=True,
            prefer_code_aware_for_code=True,
        )
    )

    monkeypatch.setattr(content_router_module, "is_mixed_content", lambda content: False)
    monkeypatch.setattr(
        content_router_module,
        "_detect_content",
        lambda content: DetectionResult(ContentType.SOURCE_CODE, 1.0, {}),
    )

    strategy = router._determine_strategy("def foo(): pass")
    assert strategy is CompressionStrategy.CODE_AWARE


def test_source_code_passthrough_does_not_invoke_kompress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PASSTHROUGH for code must never call _try_ml_compressor.

    KOMPRESS on large code blobs can destroy semantics.  The PASSTHROUGH
    path must be a pure passthrough — no ML compressor invocation.
    """
    router = ContentRouter(ContentRouterConfig(prefer_code_aware_for_code=False))

    monkeypatch.setattr(content_router_module, "is_mixed_content", lambda content: False)
    monkeypatch.setattr(
        content_router_module,
        "_detect_content",
        lambda content: DetectionResult(ContentType.SOURCE_CODE, 1.0, {}),
    )

    def fail_kompress(*_args: object, **_kwargs: object) -> tuple[str, int]:
        raise AssertionError("PASSTHROUGH must not invoke Kompress")

    monkeypatch.setattr(router, "_try_ml_compressor", fail_kompress)

    code = "def hello():\n    return 42\n"
    result = router.compress(code)
    assert result.compressed == code
    assert result.strategy_used is CompressionStrategy.PASSTHROUGH


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

    def fake_compress(content, context: str = "", bias: float = 1.0, precomputed_detection=None):
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


# ---------------------------------------------------------------------------
# Cache-churn fix: HEADROOM_FREEZE_BLOCK_DECISION (default off).
#
# Root cause: ``min_ratio`` drifts every turn with context pressure, so a
# block whose compression ratio sits in [aggressive, relaxed) is compressed
# on a low-pressure turn and downgraded to passthrough on a high-pressure
# turn (or vice-versa). The block's prefix bytes flap across turns ⇒
# cache_write churn. The fix freezes the per-block compress/passthrough
# verdict on first sighting against the FIXED aggressive threshold.
# ---------------------------------------------------------------------------


class _ChurnTokenizer:
    """Word-count tokenizer; apply() only calls ``count_text``."""

    def count_text(self, text: str) -> int:
        return len(str(text).split())


def _churn_router(monkeypatch: pytest.MonkeyPatch, ratio: float) -> ContentRouter:
    """Router whose compress() always emits a deterministic payload at a
    fixed ratio. ``min_chars`` thresholds are relaxed so a 200-word tool
    message reaches the compression path."""
    cfg = ContentRouterConfig(min_ratio_relaxed=0.85, min_ratio_aggressive=0.65)
    router = ContentRouter(cfg)

    def fake_compress(content, context: str = "", bias: float = 1.0, precomputed_detection=None):
        return SimpleNamespace(
            # CCR marker keeps the compression recoverable so the #1307
            # tool-reversibility guard preserves it instead of restoring original.
            compressed="[C]" + content[:20] + " <<ccr:t>>",
            compression_ratio=ratio,
            strategy_used=CompressionStrategy.TEXT,
        )

    monkeypatch.setattr(router, "compress", fake_compress)
    # Don't let lazy ML model state interfere with the model_ready signal.
    monkeypatch.setattr(router, "_kompress_model_ready", lambda: True)
    return router


def _run_turn(router: ContentRouter, content: str, tokens_before: int):
    """Drive apply() for a single message with a rising context-pressure
    knob. model_limit fixed at 1000 so larger tokens_before ⇒ higher
    pressure ⇒ tighter min_ratio."""
    messages = [{"role": "tool", "tool_call_id": "t1", "content": content}]
    result = router.apply(
        messages,
        _ChurnTokenizer(),
        model_limit=1000,
        # Force a known tokens_before by padding nothing — apply recomputes
        # tokens_before from the messages, so we instead steer pressure via
        # model_limit below. (kept for clarity)
    )
    return result.messages[0]["content"]


def _content_of_n_words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_freeze_off_is_byte_identical_flapping_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) Flag OFF (default): a mid-zone block (ratio 0.75) flaps —
    compressed on a low-pressure turn, downgraded to passthrough once
    pressure tightens min_ratio below the ratio. This is the legacy churn
    and must be preserved byte-for-byte when the flag is off."""
    monkeypatch.delenv("HEADROOM_FREEZE_BLOCK_DECISION", raising=False)
    router = _churn_router(monkeypatch, ratio=0.75)
    content = _content_of_n_words(200)

    # Turn 1: low pressure (small model_limit denom via large limit) ->
    # min_ratio ~ relaxed (0.85). 0.75 < 0.85 -> compresses.
    low = router.apply(
        [{"role": "tool", "tool_call_id": "t1", "content": content}],
        _ChurnTokenizer(),
        model_limit=100000,  # pressure ~ 0 -> min_ratio ~ 0.85
    ).messages[0]["content"]
    assert low.startswith("[C]"), "turn1 should compress at low pressure"

    # Turn 2: high pressure -> min_ratio ~ aggressive (0.65). 0.75 < 0.65
    # is False -> cache-hit path downgrades via move_to_skip -> passthrough.
    high = router.apply(
        [{"role": "tool", "tool_call_id": "t1", "content": content}],
        _ChurnTokenizer(),
        model_limit=100,  # 200 words >> 100 -> pressure 1.0 -> min_ratio 0.65
    ).messages[0]["content"]
    assert high == content, "turn2 should flip to passthrough (legacy churn)"
    # The flap: bytes changed across turns.
    assert low != high


def test_freeze_on_pins_compress_verdict_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) Flag ON: a block that compresses on first sighting keeps the SAME
    verdict AND the SAME bytes on later turns despite rising pressure /
    drifting min_ratio. Use ratio 0.6 (< aggressive 0.65) so it compresses
    under the frozen threshold."""
    monkeypatch.setenv("HEADROOM_FREEZE_BLOCK_DECISION", "1")
    router = _churn_router(monkeypatch, ratio=0.60)
    content = _content_of_n_words(200)

    outs = []
    # Rising pressure across turns: model_limit shrinks -> min_ratio drifts
    # from ~0.85 down to ~0.65. Without freeze this is the flap zone.
    for limit in (100000, 1000, 300, 100):
        out = router.apply(
            [{"role": "tool", "tool_call_id": "t1", "content": content}],
            _ChurnTokenizer(),
            model_limit=limit,
        ).messages[0]["content"]
        outs.append(out)

    assert all(o.startswith("[C]") for o in outs), "verdict must stay compress"
    assert len(set(outs)) == 1, "bytes must be identical across all turns"


def test_freeze_on_pins_passthrough_verdict_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) Flag ON, mid-zone block (ratio 0.75): the RELAXED first-sighting
    threshold (0.85) compresses it on turn 1 (low pressure), freezes the
    compress verdict, and PINS it compressed on every later turn — even as
    rising pressure pulls ``min_ratio`` below 0.75, where the flag-off path
    would ``move_to_skip`` and bust the prefix cache. So the bytes stay
    compressed and identical, and the freeze-pin counter records the
    busts it avoided. (Contrast with the flag-off flapping baseline.)"""
    monkeypatch.setenv("HEADROOM_FREEZE_BLOCK_DECISION", "1")
    router = _churn_router(monkeypatch, ratio=0.75)
    content = _content_of_n_words(200)

    outs = []
    for limit in (100000, 1000, 300, 100):
        out = router.apply(
            [{"role": "tool", "tool_call_id": "t1", "content": content}],
            _ChurnTokenizer(),
            model_limit=limit,
        ).messages[0]["content"]
        outs.append(out)

    assert all(o.startswith("[C]") for o in outs), "verdict must stay compressed"
    assert len(set(outs)) == 1, "bytes identical (always compressed) across turns"
    assert router._freeze_pin_hits > 0, "freeze must pin compress over a tightening min_ratio"


def test_model_not_ready_passthrough_is_not_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caveat (1): a block that passes through ONLY because the ML model is
    not ready must NOT have its skip verdict frozen — once the model is
    ready it must be re-evaluated. We simulate not-ready (compress returns
    the content unchanged at ratio 1.0) then ready (real compression)."""
    monkeypatch.setenv("HEADROOM_FREEZE_BLOCK_DECISION", "1")
    cfg = ContentRouterConfig()
    router = ContentRouter(cfg)
    content = _content_of_n_words(200)

    # Phase 1: model NOT ready. _try_ml_compressor returns content unchanged
    # -> ratio 1.0 (passthrough). Verdict must NOT be frozen as skip.
    monkeypatch.setattr(router, "_kompress_model_ready", lambda: False)
    monkeypatch.setattr(
        router,
        "compress",
        lambda c, context="", bias=1.0, precomputed_detection=None: SimpleNamespace(
            compressed=c, compression_ratio=1.0, strategy_used=CompressionStrategy.TEXT
        ),
    )
    out1 = router.apply(
        [{"role": "tool", "tool_call_id": "t1", "content": content}],
        _ChurnTokenizer(),
        model_limit=1000,
    ).messages[0]["content"]
    assert out1 == content, "not-ready -> passthrough"
    # The verdict store must NOT contain a frozen skip for this block.
    assert all(v is True for v in router._frozen_verdicts.values()) or (
        not router._frozen_verdicts
    ), "not-ready passthrough must not be frozen as skip"

    # Phase 2: model now ready, real compression below the aggressive
    # threshold. Because the verdict was never frozen, the block must now be
    # re-evaluated and compress (not stuck on a frozen passthrough). Note:
    # the legacy byte skip-cache is keyed on the same content; we clear it to
    # isolate the verdict behaviour (the freeze fix governs the verdict, not
    # the existing TTL byte cache).
    router._cache.clear()
    monkeypatch.setattr(router, "_kompress_model_ready", lambda: True)
    monkeypatch.setattr(
        router,
        "compress",
        lambda c, context="", bias=1.0, precomputed_detection=None: SimpleNamespace(
            compressed="[C]"
            + c[:20]
            + " <<ccr:t>>",  # recoverable marker so #1307 tool-reversibility guard keeps the compression
            compression_ratio=0.50,
            strategy_used=CompressionStrategy.TEXT,
        ),
    )
    out2 = router.apply(
        [{"role": "tool", "tool_call_id": "t1", "content": content}],
        _ChurnTokenizer(),
        model_limit=1000,
    ).messages[0]["content"]
    assert out2.startswith("[C]"), "once model ready the block must compress"


# ---------------------------------------------------------------------------
# Cache-churn fix — CONTENT-BLOCK path (tool_result blocks).
#
# For Claude Code / Anthropic traffic the prefix is dominated by tool_result
# content-blocks routed through ``_compress_block_content``, not plain-string
# messages. These tests mirror the string-path freeze tests above for that
# dominant path.
# ---------------------------------------------------------------------------


def _tool_result_block_msg(content: str) -> dict:
    """A user message carrying a single ``tool_result`` content-block — the
    Anthropic shape that dominates Claude Code prefixes."""
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu1", "content": content},
        ],
    }


def _block_out(router: ContentRouter, content: str, model_limit: int) -> str:
    """Run apply() for a tool_result content-block and return the (possibly
    compressed) string content of that block."""
    result = router.apply(
        [_tool_result_block_msg(content)],
        _ChurnTokenizer(),
        model_limit=model_limit,
    )
    return result.messages[0]["content"][0]["content"]


def test_block_freeze_off_is_byte_identical_flapping_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) Block path, flag OFF (default): a mid-zone tool_result block
    (ratio 0.75) flaps — compressed at low pressure, downgraded to passthrough
    once pressure tightens min_ratio below the ratio. Legacy churn preserved."""
    monkeypatch.delenv("HEADROOM_FREEZE_BLOCK_DECISION", raising=False)
    router = _churn_router(monkeypatch, ratio=0.75)
    content = _content_of_n_words(200)

    low = _block_out(router, content, model_limit=100000)
    assert low.startswith("[C]"), "turn1 should compress at low pressure"

    high = _block_out(router, content, model_limit=100)
    assert high == content, "turn2 should flip to passthrough (legacy churn)"
    assert low != high, "flag-off block path must keep flapping (byte change)"


def test_block_freeze_on_pins_compress_verdict_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) Block path, flag ON: a tool_result block that compresses on first
    sighting keeps the SAME verdict AND SAME bytes across rising-pressure
    turns despite drifting min_ratio. ratio 0.6 < aggressive 0.65."""
    monkeypatch.setenv("HEADROOM_FREEZE_BLOCK_DECISION", "1")
    router = _churn_router(monkeypatch, ratio=0.60)
    content = _content_of_n_words(200)

    outs = [_block_out(router, content, model_limit=limit) for limit in (100000, 1000, 300, 100)]
    assert all(o.startswith("[C]") for o in outs), "verdict must stay compress"
    assert len(set(outs)) == 1, "block bytes must be identical across all turns"


def test_block_freeze_on_pins_passthrough_verdict_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) Block path, flag ON, mid-zone block (ratio 0.75): the RELAXED
    first-sighting threshold (0.85) compresses it on turn 1, freezes the
    compress verdict, and pins it compressed on every later turn even as
    rising pressure pulls min_ratio below 0.75 (where flag-off would
    move_to_skip). Bytes stay compressed/identical and pins are recorded."""
    monkeypatch.setenv("HEADROOM_FREEZE_BLOCK_DECISION", "1")
    router = _churn_router(monkeypatch, ratio=0.75)
    content = _content_of_n_words(200)

    outs = [_block_out(router, content, model_limit=limit) for limit in (100000, 1000, 300, 100)]
    assert all(o.startswith("[C]") for o in outs), "verdict must stay compressed"
    assert len(set(outs)) == 1, "block bytes identical (compressed) across turns"
    assert router._freeze_pin_hits > 0, "freeze must pin compress over a tightening min_ratio"


def test_block_model_not_ready_passthrough_is_not_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caveat (1) on the block path: a tool_result block that passes through
    ONLY because the ML model is not ready must NOT have its skip verdict
    frozen, so it is re-evaluated once the model is ready."""
    monkeypatch.setenv("HEADROOM_FREEZE_BLOCK_DECISION", "1")
    router = ContentRouter(ContentRouterConfig())
    content = _content_of_n_words(200)

    monkeypatch.setattr(router, "_kompress_model_ready", lambda: False)
    monkeypatch.setattr(
        router,
        "compress",
        lambda c, context="", bias=1.0, precomputed_detection=None: SimpleNamespace(
            compressed=c, compression_ratio=1.0, strategy_used=CompressionStrategy.TEXT
        ),
    )
    out1 = _block_out(router, content, model_limit=1000)
    assert out1 == content, "not-ready -> passthrough"
    assert all(v is True for v in router._frozen_verdicts.values()) or (
        not router._frozen_verdicts
    ), "not-ready block passthrough must not be frozen as skip"

    router._cache.clear()
    monkeypatch.setattr(router, "_kompress_model_ready", lambda: True)
    monkeypatch.setattr(
        router,
        "compress",
        lambda c, context="", bias=1.0, precomputed_detection=None: SimpleNamespace(
            compressed="[C]"
            + c[:20]
            + " <<ccr:t>>",  # recoverable marker so #1307 tool-reversibility guard keeps the compression
            compression_ratio=0.50,
            strategy_used=CompressionStrategy.TEXT,
        ),
    )
    out2 = _block_out(router, content, model_limit=1000)
    assert out2.startswith("[C]"), "once model ready the block must compress"


def test_frozen_verdicts_cleared_on_cache_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) ``_frozen_verdicts`` is cleared in lock-step with the cache so it
    cannot outlive the entries it shadows."""
    monkeypatch.setenv("HEADROOM_FREEZE_BLOCK_DECISION", "1")
    router = _churn_router(monkeypatch, ratio=0.60)
    content = _content_of_n_words(200)

    _block_out(router, content, model_limit=1000)
    assert router._frozen_verdicts, "a verdict should be frozen after a turn"

    router._cache.clear()
    assert not router._frozen_verdicts, "cache clear must also clear frozen verdicts"


def test_frozen_verdicts_is_size_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) ``_frozen_verdicts`` is capped with FIFO eviction so it cannot grow
    without bound across a long-lived process."""
    monkeypatch.setenv("HEADROOM_FREEZE_BLOCK_DECISION", "1")
    router = ContentRouter(ContentRouterConfig())
    router._frozen_verdicts_max = 8

    for i in range(50):
        router._record_frozen_verdict(i, True)

    assert len(router._frozen_verdicts) == 8, "store must stay capped at the max"
    # Oldest keys evicted, newest retained (insertion-order FIFO).
    assert set(router._frozen_verdicts) == set(range(42, 50))


def test_frozen_verdict_refuses_unrecoverable_lossy_block() -> None:
    """#1307: a "compress" verdict for a lossy-unmarked block with no CCR
    retrieval marker must NOT be frozen. Pinning it would keep serving an
    unrecoverable summary across turns, which the reversibility guard forbids.
    Recoverable compressions (marked, or a non-lossy strategy) may be pinned.
    """
    router = ContentRouter(ContentRouterConfig())

    # Lossy + no marker -> unrecoverable -> refuse to freeze.
    assert router._frozen_verdict_recoverable(CompressionStrategy.TEXT, "plain summary") is False
    # Lossy + CCR marker -> recoverable -> may freeze.
    assert router._frozen_verdict_recoverable(CompressionStrategy.TEXT, "summary <<ccr:t>>") is True
    # Non-lossy strategy -> recoverable regardless of marker.
    assert (
        router._frozen_verdict_recoverable(CompressionStrategy.SMART_CRUSHER, "no marker") is True
    )
    # Cache-hit path passes the strategy's .value string, not the enum.
    assert router._frozen_verdict_recoverable("text", "plain") is False
    assert router._frozen_verdict_recoverable("text", "x Retrieve more: hash=abc123") is True


def test_detect_timeout_secs_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The watchdog budget reads HEADROOM_DETECT_TIMEOUT_SECS; bad values → default."""
    get = content_router_module._detect_timeout_secs
    default = content_router_module._DEFAULT_DETECT_TIMEOUT_SECS

    monkeypatch.delenv("HEADROOM_DETECT_TIMEOUT_SECS", raising=False)
    assert get() == default

    monkeypatch.setenv("HEADROOM_DETECT_TIMEOUT_SECS", "0.25")
    assert get() == 0.25

    monkeypatch.setenv("HEADROOM_DETECT_TIMEOUT_SECS", "nope")
    assert get() == default

    monkeypatch.setenv("HEADROOM_DETECT_TIMEOUT_SECS", "0")
    assert get() == default


def test_rust_detect_watchdog_passes_through_result() -> None:
    """A fast native detector returns its result unchanged through the watchdog."""
    sentinel = SimpleNamespace(content_type="json_array", confidence=1.0, metadata={})
    out = content_router_module._rust_detect_watchdogged(lambda _content: sentinel, "payload", 5.0)
    assert out is sentinel


def test_rust_detect_watchdog_relays_native_error() -> None:
    """An exception raised inside the native detector propagates to the caller."""

    def boom(_content: str) -> None:
        raise ValueError("native boom")

    with pytest.raises(ValueError, match="native boom"):
        content_router_module._rust_detect_watchdogged(boom, "payload", 5.0)


def test_detect_content_watchdog_degrades_on_windows_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung native detect on Windows degrades to pure-Python, never deadlocks (#575)."""
    import threading as _threading

    import headroom._core as _core

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "rust")
    monkeypatch.setattr(content_router_module.sys, "platform", "win32")
    monkeypatch.setenv("HEADROOM_DETECT_TIMEOUT_SECS", "0.1")

    release = _threading.Event()

    def _hang(_content: str):
        release.wait()  # simulate the WaitOnAddress park (GIL released while waiting)
        return SimpleNamespace(content_type="plain_text", confidence=1.0, metadata={})

    monkeypatch.setattr(_core, "detect_content_type", _hang)

    try:
        # JSON content: the pure-Python regex fallback recognizes it as JSON_ARRAY,
        # proving we took the degrade path rather than the (hung) native result.
        result = _detect_content('[{"id": 1}]')
        assert result.content_type is ContentType.JSON_ARRAY
    finally:
        release.set()  # let the daemon worker finish so it does not linger


def test_detect_content_watchdog_uses_native_result_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On Windows with rust forced, a fast native result still flows through unchanged."""
    import headroom._core as _core

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "rust")
    monkeypatch.setattr(content_router_module.sys, "platform", "win32")

    fake = SimpleNamespace(content_type="source_code", confidence=1.0, metadata={})
    monkeypatch.setattr(_core, "detect_content_type", lambda _content: fake)

    result = _detect_content("def main(): pass")
    assert result.content_type is ContentType.SOURCE_CODE


def test_detect_content_circuit_breaker_skips_native_after_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After one watchdog timeout, native detection is disabled process-wide (#575)."""
    import threading as _threading

    import headroom._core as _core

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "rust")
    monkeypatch.setattr(content_router_module.sys, "platform", "win32")
    monkeypatch.setenv("HEADROOM_DETECT_TIMEOUT_SECS", "0.1")

    release = _threading.Event()
    calls = 0

    def _hang(_content: str):
        nonlocal calls
        calls += 1
        release.wait()  # park with GIL released, like the real WaitOnAddress hang
        return SimpleNamespace(content_type="plain_text", confidence=1.0, metadata={})

    monkeypatch.setattr(_core, "detect_content_type", _hang)
    try:
        first = _detect_content('[{"id": 1}]')
        second = _detect_content('[{"id": 2}]')
        assert first.content_type is ContentType.JSON_ARRAY
        assert second.content_type is ContentType.JSON_ARRAY
        assert calls == 1  # breaker tripped: native entered once, 2nd call skipped it
    finally:
        release.set()  # let the lone daemon worker finish


def test_strip_detection_envelope_isolates_tool_output_payload() -> None:
    """Only a whole-string tool-output envelope is unwrapped; content that
    merely mentions the tags, or has an empty body, is left untouched."""
    body = "def main():\n    return 1"
    wrapped = f"<returncode>0</returncode>\n<output>\n{body}\n</output>"
    assert _strip_detection_envelope(wrapped) == body
    # <output> alias tags and a bare envelope (no returncode) also unwrap.
    assert _strip_detection_envelope(f"<stdout>\n{body}\n</stdout>") == body
    # Non-envelope content is returned verbatim (no "<" fast-path + no match).
    prose = "see the <output> tag docs for details"
    assert _strip_detection_envelope(prose) == prose
    # Empty body never yields an empty probe — falls back to the original.
    empty = "<output>\n\n</output>"
    assert _strip_detection_envelope(empty) == empty


def test_detect_content_sees_through_tool_output_envelope() -> None:
    """Regression: a tool-result envelope's tags used to make the detector
    read the whole payload as markup and misroute code to the HTML extractor.
    Detection now runs on the inner payload, so the real type wins."""
    code = "\n".join(
        [
            "import os",
            "from pathlib import Path",
            "",
            "def main() -> int:",
            "    return len(os.listdir(Path.cwd()))",
        ]
    )
    wrapped = f"<returncode>0</returncode>\n<output>\n{code}\n</output>"
    assert _detect_content(wrapped).content_type is ContentType.SOURCE_CODE
    assert _detect_content(wrapped).content_type is _detect_content(code).content_type


def test_detect_content_overrides_html_misroute_for_grep_and_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the native detector (magika) tags dense grep output and
    build logs as HTML because file paths and </> read as markup. Routing those
    to the HTML article-extractor is lossy (it strips code + identifiers). When
    the structural log/search detectors positively claim the payload they
    override the HTML verdict (log checked first so tracebacks win); genuine
    HTML with no such structure is left as HTML."""
    import headroom._core as _core

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "rust")
    monkeypatch.setattr(
        _core,
        "detect_content_type",
        lambda content: SimpleNamespace(content_type="html", confidence=1.0, metadata={}),
    )

    # grep over HTML template files: native says html, but it is search results.
    grep = "\n".join(
        f'templates/pages/dashboard_{i}.html:{10 + i}:      <div class="card" data-id="{i}">'
        for i in range(6)
    )
    assert _detect_content(grep).content_type is ContentType.SEARCH_RESULTS

    # build/error log misread as html -> LOG wins (checked before search).
    build_log = "\n".join(
        [
            "ERROR failed to compile module widget",
            "WARNING deprecated call near <template>",
            "Traceback (most recent call last):",
            "ERROR build aborted after 2 retries",
        ]
    )
    assert _detect_content(build_log).content_type is ContentType.BUILD_OUTPUT

    # genuine HTML article: no grep/log structure -> override does not fire.
    html = (
        "<!DOCTYPE html>\n<html><head><title>x</title></head>"
        "<body><main><section><p>An article about widgets and gadgets.</p>"
        "</section></main></body></html>"
    )
    assert _detect_content(html).content_type is ContentType.HTML


def test_datetime_prefixed_user_prompt_survives_router() -> None:
    """Regression (2026-08-23): interactive wrap-copilot prompts were deleted.

    Copilot CLI prepends ``<current_datetime>…</current_datetime>`` to every
    interactive user turn; the ISO timestamp matched the grep ``file:line:``
    detector, the one-line prompt classified as SEARCH_RESULTS, and
    SearchCompressor kept only the datetime line — the model received no
    request and answered "How can I help you today?". The router must never
    route this shape to the search line-filter and must keep the prose.
    """
    prompt = (
        "<current_datetime>2026-08-23T09:57:59.792+02:00</current_datetime>\n"
        "\n"
        "Please update the PR desc and check .overlay/ for hints."
    )
    result = ContentRouter().compress(prompt)
    assert result.strategy_used is not CompressionStrategy.SEARCH
    assert "Please update the PR desc" in result.compressed
