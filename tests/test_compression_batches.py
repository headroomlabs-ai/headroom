from __future__ import annotations

import re
import time

from headroom.transforms.compression_batches import (
    CompressionBatchEntry,
    build_compression_batches,
    compress_batch_with_router,
)
from headroom.transforms.compression_units import CompressionUnit, RoutedCompressionUnit
from headroom.transforms.content_router import CompressionStrategy, RouterCompressionResult


def _entry(index: int, text: str) -> CompressionBatchEntry:
    unit = CompressionUnit(
        text=text,
        provider="openai",
        endpoint="responses",
        role="tool",
        item_type="local_shell_call_output",
        cache_zone="live",
        mutable=True,
        min_bytes=512,
    )
    return CompressionBatchEntry(
        entry_id=f"u{index}",
        routed=RoutedCompressionUnit(unit=unit, slot=(index, ("output", None))),
    )


def test_small_units_over_floor_form_one_batch():
    entries = [_entry(index, "x" * 150) for index in range(4)]

    batches, skipped = build_compression_batches(entries, min_batch_bytes=512)

    assert len(batches) == 1
    assert [entry.entry_id for entry in batches[0].entries] == ["u0", "u1", "u2", "u3"]
    assert batches[0].text_bytes == 600
    assert skipped == []


class _CharacterCounter:
    def count_text(self, text: str) -> int:
        return len(text)


class _ShorteningRouter:
    def __init__(self) -> None:
        self.calls = 0

    def compress(self, content: str, **_kwargs) -> RouterCompressionResult:
        self.calls += 1
        return RouterCompressionResult(
            compressed=content.replace("x" * 150, "x"),
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )


def test_batch_compresses_entries_with_one_router_call():
    entries = [_entry(index, "x" * 150) for index in range(4)]
    batches, _ = build_compression_batches(entries, min_batch_bytes=512)
    router = _ShorteningRouter()

    results = compress_batch_with_router(
        batches[0],
        router=router,
        tokenizer=_CharacterCounter(),
    )

    assert router.calls == 1
    assert [slot for slot, _ in results] == [entry.routed.slot for entry in entries]
    assert [result.compressed for _, result in results] == ["x"] * 4
    assert all(result.modified for _, result in results)


def test_under_floor_tail_is_skipped_without_a_batch():
    entries = [_entry(index, "x" * 150) for index in range(3)]

    batches, skipped = build_compression_batches(entries, min_batch_bytes=512)

    assert batches == []
    assert [entry.entry_id for entry in skipped] == ["u0", "u1", "u2"]


def test_sixteen_under_floor_entries_are_skipped_before_a_new_batch_starts():
    entries = [_entry(index, "x" * 30) for index in range(17)]

    batches, skipped = build_compression_batches(entries, min_batch_bytes=512)

    assert batches == []
    assert [entry.entry_id for entry in skipped] == [f"u{index}" for index in range(17)]


class _CorruptingRouter:
    def compress(self, content: str, **_kwargs) -> RouterCompressionResult:
        return RouterCompressionResult(
            compressed="missing protected tags",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )


def test_missing_batch_tags_passes_through_every_entry():
    entries = [_entry(index, "x" * 150) for index in range(4)]
    batches, _ = build_compression_batches(entries, min_batch_bytes=512)

    results = compress_batch_with_router(
        batches[0],
        router=_CorruptingRouter(),
        tokenizer=_CharacterCounter(),
    )

    assert [result.compressed for _, result in results] == ["x" * 150] * 4
    assert [result.reason for _, result in results] == ["batch_invalid"] * 4
    assert not any(result.modified for _, result in results)


def test_many_small_entries_create_no_more_than_sixteen_per_batch():
    entries = [_entry(index, "x" * 100) for index in range(381)]

    batches, skipped = build_compression_batches(entries, min_batch_bytes=512)

    assert skipped == []
    assert len(batches) <= 24
    assert all(len(batch.entries) <= 16 for batch in batches)
    assert all(batch.text_bytes <= 2048 for batch in batches)


class _LossyShellRouter:
    def compress(self, content: str, **_kwargs) -> RouterCompressionResult:
        return RouterCompressionResult(
            compressed=content.replace("line alpha beta gamma\n" * 7, "summary"),
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )


def test_batch_keeps_structured_shell_output_without_a_ccr_marker():
    entries = [_entry(index, "line alpha beta gamma\n" * 7) for index in range(4)]
    batches, _ = build_compression_batches(entries, min_batch_bytes=512)

    results = compress_batch_with_router(
        batches[0],
        router=_LossyShellRouter(),
        tokenizer=_CharacterCounter(),
    )

    assert not any(result.modified for _, result in results)
    assert [result.reason for _, result in results] == ["lossy_unrecoverable_tool_output"] * 4


class _MarkerStrippingRouter:
    def compress(self, content: str, **_kwargs) -> RouterCompressionResult:
        return RouterCompressionResult(
            compressed=content.replace("word " * 30, "x").replace(
                "[100 items compressed to 10. Retrieve more: hash=abc123]", ""
            ),
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )


def test_batch_preserves_ccr_markers_when_router_would_remove_them():
    marker = "[100 items compressed to 10. Retrieve more: hash=abc123]"
    original = f"{'word ' * 30}\n{marker}\n"
    entries = [_entry(index, original) for index in range(4)]
    batches, _ = build_compression_batches(entries, min_batch_bytes=512)

    results = compress_batch_with_router(
        batches[0],
        router=_MarkerStrippingRouter(),
        tokenizer=_CharacterCounter(),
    )

    assert all(result.modified for _, result in results)
    assert all(marker in result.compressed for _, result in results)


class _MarkerMovingRouter:
    def compress(self, content: str, **_kwargs) -> RouterCompressionResult:
        placeholders = re.findall(r"\[\[HEADROOM_BATCH_CCR_[^]]+\]\]", content)
        moved = content.replace(placeholders[0], "", 1).replace(
            placeholders[1], f"{placeholders[1]}{placeholders[0]}", 1
        )
        return RouterCompressionResult(
            compressed=moved.replace("word " * 30, "x"),
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )


def test_batch_rejects_ccr_marker_moved_to_another_entry():
    marker = "[100 items compressed to 10. Retrieve more: hash=abc123]"
    original = f"{'word ' * 30}\n{marker}\n"
    entries = [_entry(index, original) for index in range(4)]
    batches, _ = build_compression_batches(entries, min_batch_bytes=512)

    results = compress_batch_with_router(
        batches[0],
        router=_MarkerMovingRouter(),
        tokenizer=_CharacterCounter(),
    )

    assert [result.compressed for _, result in results] == [original] * 4
    assert [result.reason for _, result in results] == ["batch_invalid"] * 4


class _CjkShorteningRouter:
    def compress(self, content: str, **_kwargs) -> RouterCompressionResult:
        return RouterCompressionResult(
            compressed=content.replace("你" * 150, "短"),
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )


def test_batch_uses_utf8_bytes_for_cjk_small_units():
    entries = [_entry(index, "你" * 150) for index in range(4)]

    batches, skipped = build_compression_batches(entries, min_batch_bytes=512)
    results = compress_batch_with_router(
        batches[0],
        router=_CjkShorteningRouter(),
        tokenizer=_CharacterCounter(),
    )

    assert skipped == []
    assert batches[0].text_bytes == 1800
    assert all(result.modified for _, result in results)
    assert [result.compressed for _, result in results] == ["短"] * 4


def test_batch_after_the_request_deadline_passes_through_without_routing(monkeypatch):
    from headroom.transforms.content_router import ContentRouter

    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "20000")
    entries = [_entry(index, "x" * 150) for index in range(4)]
    batches, _ = build_compression_batches(entries, min_batch_bytes=512)
    router = ContentRouter()

    def _not_called(*_args, **_kwargs):
        raise AssertionError("a batch past the deadline must not enter the router")

    router.compress = _not_called

    results = compress_batch_with_router(
        batches[0],
        router=router,
        tokenizer=_CharacterCounter(),
        deadline_started_at=time.perf_counter() - 21,
    )

    assert [result.compressed for _, result in results] == ["x" * 150] * 4
    assert {result.reason for _, result in results} == {"deadline_exceeded"}


def test_build_compression_batches_skips_protected_units():
    from headroom.transforms.compression_units import compress_unit_with_router

    cases = [
        # (kwargs for CompressionUnit, expected_reason, expected_category)
        ({"mutable": False}, "immutable", "immutable"),
        ({"role": "system"}, "protected_system_message", "protected_role"),
        ({"role": "developer"}, "protected_system_message", "protected_role"),
        ({"role": "user", "metadata": {}}, "protected_user_message", "protected_role"),
        ({"role": "assistant", "metadata": {}}, "protected_assistant_message", "protected_role"),
        ({"cache_zone": "prefix"}, "cache_zone_prefix", "cache_zone"),
    ]

    entries = []
    for idx, (kw, _expected_reason, _expected_category) in enumerate(cases):
        unit_kwargs = {
            "text": "x" * 150,
            "provider": "openai",
            "endpoint": "responses",
            "role": "tool",
            "item_type": "message",
            "cache_zone": "live",
            "mutable": True,
            "min_bytes": 512,
        }
        unit_kwargs.update(kw)
        unit = CompressionUnit(**unit_kwargs)
        entries.append(
            CompressionBatchEntry(
                entry_id=f"u{idx}",
                routed=RoutedCompressionUnit(unit=unit, slot=(idx, ("output", None))),
            )
        )

    batches, skipped = build_compression_batches(entries, min_batch_bytes=512)

    assert batches == []
    assert len(skipped) == len(cases)

    router = _ShorteningRouter()
    tokenizer = _CharacterCounter()
    for entry, (_kw, expected_reason, expected_category) in zip(skipped, cases, strict=True):
        single_result = compress_unit_with_router(
            entry.routed.unit,
            router=router,
            tokenizer=tokenizer,
        )
        assert not single_result.modified
        assert single_result.reason == expected_reason
        assert single_result.reason_category == expected_category
    assert router.calls == 0


def test_batch_refuses_protected_units_without_routing():
    from headroom.transforms.compression_batches import CompressionBatch

    cases = [
        ({"mutable": False}, "immutable", "immutable"),
        ({"role": "system"}, "protected_system_message", "protected_role"),
        ({"role": "developer"}, "protected_system_message", "protected_role"),
        ({"role": "user", "metadata": {}}, "protected_user_message", "protected_role"),
        ({"role": "assistant", "metadata": {}}, "protected_assistant_message", "protected_role"),
        ({"cache_zone": "prefix"}, "cache_zone_prefix", "cache_zone"),
    ]

    for kw, expected_reason, expected_category in cases:
        entries = []
        for idx in range(2):
            unit_kwargs = {
                "text": "x" * 150,
                "provider": "openai",
                "endpoint": "responses",
                "role": "tool",
                "item_type": "message",
                "cache_zone": "live",
                "mutable": True,
                "min_bytes": 512,
            }
            unit_kwargs.update(kw)
            unit = CompressionUnit(**unit_kwargs)
            entries.append(
                CompressionBatchEntry(
                    entry_id=f"u{idx}",
                    routed=RoutedCompressionUnit(unit=unit, slot=(idx, ("output", None))),
                )
            )

        batch = CompressionBatch(entries=tuple(entries), text_bytes=300)

        def _not_called(*_args, **_kwargs):
            raise AssertionError("a protected batch must not enter the router")

        class _AssertRouter:
            compress = _not_called

        results = compress_batch_with_router(
            batch,
            router=_AssertRouter(),
            tokenizer=_CharacterCounter(),
        )

        assert len(results) == 2
        for _slot, res in results:
            assert not res.modified
            assert res.compressed == "x" * 150
            assert res.reason == expected_reason
            assert res.reason_category == expected_category


def test_batch_allows_permitted_user_and_assistant_units():
    entries = []
    for idx, (role, meta_key) in enumerate(
        [("user", "compress_user"), ("assistant", "compress_assistant")]
    ):
        unit = CompressionUnit(
            text="x" * 300,
            provider="openai",
            endpoint="responses",
            role=role,
            item_type="message",
            cache_zone="live",
            mutable=True,
            min_bytes=512,
            metadata={meta_key: "true"},
        )
        entries.append(
            CompressionBatchEntry(
                entry_id=f"u{idx}",
                routed=RoutedCompressionUnit(unit=unit, slot=(idx, ("output", None))),
            )
        )

    # Test permitted user units
    user_entries = [entries[0], entries[0]]
    user_batches, user_skipped = build_compression_batches(user_entries, min_batch_bytes=512)
    assert len(user_batches) == 1
    assert user_skipped == []

    router = _ShorteningRouter()
    user_results = compress_batch_with_router(
        user_batches[0],
        router=router,
        tokenizer=_CharacterCounter(),
    )
    assert router.calls == 1
    assert all(res.modified for _, res in user_results)

    # Test permitted assistant units
    assistant_entries = [entries[1], entries[1]]
    asst_batches, asst_skipped = build_compression_batches(assistant_entries, min_batch_bytes=512)
    assert len(asst_batches) == 1
    assert asst_skipped == []

    asst_results = compress_batch_with_router(
        asst_batches[0],
        router=router,
        tokenizer=_CharacterCounter(),
    )
    assert router.calls == 2
    assert all(res.modified for _, res in asst_results)
