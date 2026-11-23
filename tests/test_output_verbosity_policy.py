"""Tests for pure output verbosity steering policy."""

from __future__ import annotations

import pytest

from headroom.proxy import output_verbosity_policy
from headroom.proxy.output_verbosity_policy import (
    STEERING_SENTINEL,
    STEERING_SUFFIX,
    register_verbosity_text,
    replace_or_append_steering_block,
    steering_text,
)


def test_level_zero_and_unknown_levels_have_no_steering_text() -> None:
    assert steering_text(0) is None
    assert steering_text(99) is None


def test_steering_text_is_wrapped_and_byte_stable() -> None:
    first = steering_text(2)
    second = steering_text(2)
    assert first == second
    assert first is not None
    assert first.startswith(f"{STEERING_SENTINEL}\n")
    assert first.endswith(f"\n{STEERING_SUFFIX}")
    assert "Never restate code" in first


def test_replace_or_append_adds_block_to_nonempty_instructions() -> None:
    block = steering_text(3)
    assert block is not None
    updated, changed = replace_or_append_steering_block("System.", block)
    assert changed is True
    assert updated == f"System.\n\n{block}"


def test_replace_or_append_uses_block_for_empty_instructions() -> None:
    block = steering_text(1)
    assert block is not None
    updated, changed = replace_or_append_steering_block("   ", block)
    assert changed is True
    assert updated == block


def test_replace_or_append_replaces_existing_complete_block_once() -> None:
    old = steering_text(1)
    new = steering_text(4)
    assert old is not None
    assert new is not None
    updated, changed = replace_or_append_steering_block(f"System.\n\n{old}\n\nTail.", new)
    assert changed is True
    assert old not in updated
    assert updated == f"System.\n\n{new}\n\nTail."


def test_replace_or_append_replaces_unclosed_sentinel_to_end() -> None:
    new = steering_text(2)
    assert new is not None
    updated, changed = replace_or_append_steering_block(
        f"System.\n\n{STEERING_SENTINEL}\nold text without close",
        new,
    )
    assert changed is True
    assert updated == f"System.\n\n{new}"


def test_replace_or_append_is_idempotent_when_block_matches() -> None:
    block = steering_text(2)
    assert block is not None
    updated, changed = replace_or_append_steering_block(f"System.\n\n{block}", block)
    assert changed is False
    assert updated == f"System.\n\n{block}"


def test_registered_text_replaces_a_level_wrapped_as_usual(monkeypatch) -> None:
    monkeypatch.setattr(
        output_verbosity_policy, "VERBOSITY_LEVELS", dict(output_verbosity_policy.VERBOSITY_LEVELS)
    )
    register_verbosity_text(4, "Extension wording.")
    assert steering_text(4) == f"{STEERING_SENTINEL}\nExtension wording.\n{STEERING_SUFFIX}"
    assert "Never restate code" in (steering_text(3) or "")


@pytest.mark.parametrize(
    ("level", "text"),
    [(5, "x"), (0, "x"), (4, "  "), (4, f"a {STEERING_SENTINEL} b"), (4, f"a {STEERING_SUFFIX}")],
)
def test_register_rejects_unknown_levels_and_unsafe_text(monkeypatch, level, text) -> None:
    monkeypatch.setattr(
        output_verbosity_policy, "VERBOSITY_LEVELS", dict(output_verbosity_policy.VERBOSITY_LEVELS)
    )
    before = dict(output_verbosity_policy.VERBOSITY_LEVELS)
    with pytest.raises(ValueError):
        register_verbosity_text(level, text)
    assert output_verbosity_policy.VERBOSITY_LEVELS == before
