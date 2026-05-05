"""Detector-only tests for ``CacheAligner`` (PR-A2 / P2-23).

After PR-A2, ``CacheAligner`` is a detector-only transform: it never
mutates messages, only emits warnings. These tests pin that contract.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from headroom import OpenAIProvider, Tokenizer
from headroom.config import CacheAlignerConfig
from headroom.transforms.cache_aligner import (
    CacheAligner,
    detect_volatile_content,
)

_provider = OpenAIProvider()


def _tokenizer() -> Tokenizer:
    counter = _provider.get_token_counter("gpt-4o")
    return Tokenizer(counter, "gpt-4o")


@pytest.fixture
def tokenizer() -> Tokenizer:
    return _tokenizer()


def _system_user_messages(system_text: str) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": "hello"},
    ]


@contextmanager
def _capture_warnings() -> Iterator[list[str]]:
    """Capture warning messages from the cache_aligner logger.

    Bypasses pytest's ``caplog`` so logger-propagation tweaks elsewhere in
    the test suite don't break detection assertions.
    """
    captured: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    target_logger = logging.getLogger("headroom.transforms.cache_aligner")
    prev_level = target_logger.level
    target_logger.setLevel(logging.WARNING)
    handler = _ListHandler(level=logging.WARNING)
    target_logger.addHandler(handler)
    try:
        yield captured
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(prev_level)


# ---------------------------------------------------------------------------
# Detector tests on the ``apply`` surface
# ---------------------------------------------------------------------------


def test_volatile_uuid_detected_warned_not_rewritten(tokenizer: Tokenizer) -> None:
    system_text = (
        "You are a helpful assistant.\n"
        "Session ID: 550e8400-e29b-41d4-a716-446655440000\n"
        "Help the user."
    )
    messages = _system_user_messages(system_text)

    aligner = CacheAligner(CacheAlignerConfig(enabled=True))
    with _capture_warnings() as records:
        result = aligner.apply(messages, tokenizer)

    # Bytes preserved — never rewritten.
    assert result.messages[0]["content"] == system_text
    assert result.messages[1]["content"] == "hello"

    # Warning surfaced (in result + via the logger).
    assert result.warnings, "expected a warning for detected UUID"
    assert any("uuid" in w.lower() for w in result.warnings)
    assert any("uuid" in rec.lower() for rec in records)


def test_volatile_iso8601_detected_warned(tokenizer: Tokenizer) -> None:
    system_text = "You are a logging assistant.\nCurrent time: 2024-01-15T10:30:00\nHelp the user."
    messages = _system_user_messages(system_text)

    aligner = CacheAligner(CacheAlignerConfig(enabled=True))
    with _capture_warnings() as records:
        result = aligner.apply(messages, tokenizer)

    assert result.messages[0]["content"] == system_text
    assert any("iso8601" in w.lower() for w in result.warnings)
    assert any("iso8601" in rec.lower() for rec in records)


def test_volatile_jwt_detected_warned(tokenizer: Tokenizer) -> None:
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    system_text = f"You are an assistant.\nAuth Token: {jwt}\nHelp the user."
    messages = _system_user_messages(system_text)

    aligner = CacheAligner(CacheAlignerConfig(enabled=True))
    with _capture_warnings() as records:
        result = aligner.apply(messages, tokenizer)

    assert result.messages[0]["content"] == system_text
    assert any("jwt" in w.lower() for w in result.warnings)
    assert any("jwt" in rec.lower() for rec in records)


def test_volatile_hex_hash_detected_warned(tokenizer: Tokenizer) -> None:
    sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    system_text = f"You are a security assistant.\nHash: {sha256}\nHelp."
    messages = _system_user_messages(system_text)

    aligner = CacheAligner(CacheAlignerConfig(enabled=True))
    with _capture_warnings() as records:
        result = aligner.apply(messages, tokenizer)

    assert result.messages[0]["content"] == system_text
    assert any("hex_hash" in w.lower() for w in result.warnings)
    assert any("hex_hash" in rec.lower() for rec in records)


def test_no_false_positives_on_normal_prose(tokenizer: Tokenizer) -> None:
    system_text = (
        "You are a helpful assistant. Answer the user's questions clearly. "
        "Be polite, concise, and accurate. If you do not know the answer, "
        "say so plainly."
    )
    messages = _system_user_messages(system_text)

    aligner = CacheAligner(CacheAlignerConfig(enabled=True))
    with _capture_warnings() as records:
        result = aligner.apply(messages, tokenizer)

    assert result.messages[0]["content"] == system_text
    assert result.warnings == [], f"unexpected warnings: {result.warnings}"
    assert records == [], f"unexpected log records: {records}"


def test_apply_never_mutates_input(tokenizer: Tokenizer) -> None:
    """The detector never mutates the caller's input list."""
    system_text = "Session: 550e8400-e29b-41d4-a716-446655440000"
    messages = _system_user_messages(system_text)
    snapshot = [dict(m) for m in messages]

    aligner = CacheAligner(CacheAlignerConfig(enabled=True))
    aligner.apply(messages, tokenizer)

    for original, current in zip(snapshot, messages, strict=True):
        assert original == current


def test_apply_does_not_attempt_rewrite_on_dynamic_content(
    tokenizer: Tokenizer,
) -> None:
    """``transforms_applied`` must be empty even when warnings fire."""
    system_text = (
        "Session: 550e8400-e29b-41d4-a716-446655440000\n"
        "Hash: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    messages = _system_user_messages(system_text)
    aligner = CacheAligner(CacheAlignerConfig(enabled=True))
    result = aligner.apply(messages, tokenizer)
    assert result.transforms_applied == []
    # System bytes must equal input bytes (no rewrite).
    assert result.messages[0]["content"] == system_text


def test_should_apply_false_when_disabled(tokenizer: Tokenizer) -> None:
    messages = _system_user_messages("Session: 550e8400-e29b-41d4-a716-446655440000")
    aligner = CacheAligner(CacheAlignerConfig(enabled=False))
    assert not aligner.should_apply(messages, tokenizer)


def test_should_apply_false_without_system_message(tokenizer: Tokenizer) -> None:
    messages = [{"role": "user", "content": "hello"}]
    aligner = CacheAligner(CacheAlignerConfig(enabled=True))
    assert not aligner.should_apply(messages, tokenizer)


# ---------------------------------------------------------------------------
# Pure-function detection tests
# ---------------------------------------------------------------------------


def test_detect_volatile_content_uuid() -> None:
    findings = detect_volatile_content("Session: 550e8400-e29b-41d4-a716-446655440000")
    labels = [f.label for f in findings]
    assert "uuid" in labels


def test_detect_volatile_content_iso_date() -> None:
    findings = detect_volatile_content("Now: 2024-01-15T10:30:00")
    labels = [f.label for f in findings]
    assert "iso8601" in labels


def test_detect_volatile_content_jwt() -> None:
    # ggignore: canonical fake JWT, payload {"sub":"1"}, no privilege.
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"  # noqa: S105
    findings = detect_volatile_content(f"Token: {jwt}")
    labels = [f.label for f in findings]
    assert "jwt" in labels


def test_detect_volatile_content_md5() -> None:
    findings = detect_volatile_content("Hash: d41d8cd98f00b204e9800998ecf8427e")
    labels = [f.label for f in findings]
    assert "hex_hash" in labels


def test_detect_volatile_content_sha1() -> None:
    findings = detect_volatile_content("Hash: da39a3ee5e6b4b0d3255bfef95601890afd80709")
    labels = [f.label for f in findings]
    assert "hex_hash" in labels


def test_detect_volatile_content_returns_truncated_sample() -> None:
    long_token = "0123456789abcdef0123456789abcdef"  # 32-char hex (MD5-shaped)
    findings = detect_volatile_content(f"hash: {long_token}")
    assert findings
    # Sample is truncated for any token longer than 16 chars.
    assert "..." in findings[0].sample


def test_detect_volatile_content_empty_string_no_findings() -> None:
    assert detect_volatile_content("") == []


def test_detect_volatile_content_normal_text_no_findings() -> None:
    findings = detect_volatile_content("You are a helpful assistant. Be polite.")
    assert findings == []


def test_alignment_score_perfect_when_no_dynamic_content() -> None:
    aligner = CacheAligner()
    score = aligner.get_alignment_score([{"role": "system", "content": "You are helpful."}])
    assert score == 100.0


def test_alignment_score_decreases_with_dynamic_content() -> None:
    aligner = CacheAligner()
    score = aligner.get_alignment_score(
        [
            {
                "role": "system",
                "content": (
                    "Session: 550e8400-e29b-41d4-a716-446655440000\n"
                    "Hash: d41d8cd98f00b204e9800998ecf8427e"
                ),
            }
        ]
    )
    assert score < 100.0


def test_no_regex_imported() -> None:
    """The cache_aligner module must not import ``re`` (build constraint)."""
    import headroom.transforms.cache_aligner as mod

    # ``re`` should not be in the module's globals.
    assert "re" not in mod.__dict__, (
        "headroom.transforms.cache_aligner must not depend on the regex module"
    )
