"""Decision matrix for what the proxy does when compression fails.

Driven by Camille's bug report (2026-05-21): Codex threads were
locking with "ran out of room in the model's context window" because
the WS /v1/responses handler forwarded the *original* uncompressed
frame on compression timeout (fail-open). The upstream then rejected
the oversized frame, and Codex's auto-compaction never fired because
its ``total_usage_tokens`` heuristic had been hidden from cumulative
context pressure by Headroom's earlier successful compressions.

These tests pin :func:`headroom.proxy.helpers.decide_compression_failure_action`
so the matrix is reviewable in one place and regressions are loud.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from headroom.proxy.helpers import (
    WS_COMPRESSION_FAIL_OPEN_ENV,
    WS_COMPRESSION_OVERSIZE_BYTES_DEFAULT,
    WS_COMPRESSION_OVERSIZE_BYTES_ENV,
    decide_compression_failure_action,
)


@contextmanager
def _env(**overrides: str | None) -> Iterator[None]:
    """Temporarily set / unset env vars for a single test."""
    saved: dict[str, str | None] = {}
    for key, value in overrides.items():
        saved[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, prior in saved.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


def test_timeout_refuses_without_client_override_regardless_of_frame_size() -> None:
    """asyncio.TimeoutError → refuse, even for a tiny non-Codex frame.

    Compression timeout fires after ``COMPRESSION_TIMEOUT_SECONDS``, which
    means the pipeline already started work on the frame. A small frame
    that nevertheless timed out is a strong "something is wrong" signal —
    safer to surface to the client than to forward.
    """
    with _env(**{WS_COMPRESSION_FAIL_OPEN_ENV: None, WS_COMPRESSION_OVERSIZE_BYTES_ENV: None}):
        action = decide_compression_failure_action(asyncio.TimeoutError(), frame_bytes=128)
    assert action.refuse is True
    assert action.reason == "timeout"
    assert action.frame_bytes == 128


def test_codex_client_timeout_fails_open_without_env_override() -> None:
    """Codex direct-proxy traffic should keep flowing on compression timeout."""
    with _env(**{WS_COMPRESSION_FAIL_OPEN_ENV: None, WS_COMPRESSION_OVERSIZE_BYTES_ENV: None}):
        action = decide_compression_failure_action(
            asyncio.TimeoutError(),
            frame_bytes=128,
            client="codex",
        )
    assert action.refuse is False
    assert action.reason == "client_override:codex"
    assert action.frame_bytes == 128


def test_non_codex_timeout_still_refuses_without_env_override() -> None:
    """The Codex override must not restore global fail-open behavior."""
    with _env(**{WS_COMPRESSION_FAIL_OPEN_ENV: None, WS_COMPRESSION_OVERSIZE_BYTES_ENV: None}):
        action = decide_compression_failure_action(
            asyncio.TimeoutError(),
            frame_bytes=128,
            client="claude-code",
        )
    assert action.refuse is True
    assert action.reason == "timeout"


def test_small_transient_error_falls_through_to_passthrough() -> None:
    """Non-timeout error on a small frame: forward original (legacy)."""
    with _env(**{WS_COMPRESSION_FAIL_OPEN_ENV: None, WS_COMPRESSION_OVERSIZE_BYTES_ENV: None}):
        action = decide_compression_failure_action(
            RuntimeError("transient pipeline glitch"),
            frame_bytes=4 * 1024,
        )
    assert action.refuse is False
    assert action.reason == "small_frame_transient"
    assert action.frame_bytes == 4 * 1024


def test_oversize_frame_any_error_refuses() -> None:
    """Non-timeout error on a large frame: refuse — upstream would reject."""
    big = WS_COMPRESSION_OVERSIZE_BYTES_DEFAULT + 1024
    with _env(**{WS_COMPRESSION_FAIL_OPEN_ENV: None, WS_COMPRESSION_OVERSIZE_BYTES_ENV: None}):
        action = decide_compression_failure_action(
            RuntimeError("compressor crashed"), frame_bytes=big
        )
    assert action.refuse is True
    assert action.reason.startswith("oversize:")
    assert str(big) in action.reason


@pytest.mark.parametrize(
    "value",
    ["1", "true", "yes", "on", "TRUE", "Yes"],
)
def test_env_fail_open_overrides_everything(value: str) -> None:
    """Operator opt-in fail-open: don't refuse, even on timeout."""
    big = WS_COMPRESSION_OVERSIZE_BYTES_DEFAULT + 1
    with _env(**{WS_COMPRESSION_FAIL_OPEN_ENV: value, WS_COMPRESSION_OVERSIZE_BYTES_ENV: None}):
        action = decide_compression_failure_action(asyncio.TimeoutError(), frame_bytes=big)
    assert action.refuse is False
    assert action.reason == "env_override:fail_open"


def test_custom_threshold_via_env() -> None:
    """Operator can lower the oversize threshold per environment."""
    with _env(
        **{
            WS_COMPRESSION_FAIL_OPEN_ENV: None,
            WS_COMPRESSION_OVERSIZE_BYTES_ENV: "1024",
        }
    ):
        # Frame above the custom 1 KiB threshold → refuse
        action = decide_compression_failure_action(RuntimeError(), frame_bytes=2048)
    assert action.refuse is True
    assert "threshold=1024" in action.reason

    with _env(
        **{
            WS_COMPRESSION_FAIL_OPEN_ENV: None,
            WS_COMPRESSION_OVERSIZE_BYTES_ENV: "1024",
        }
    ):
        # Frame at/below the custom threshold → passthrough
        action_small = decide_compression_failure_action(RuntimeError(), frame_bytes=512)
    assert action_small.refuse is False
    assert action_small.reason == "small_frame_transient"


def test_invalid_threshold_env_falls_back_to_default() -> None:
    """A typo'd env var must not blow up every WS frame."""
    big = WS_COMPRESSION_OVERSIZE_BYTES_DEFAULT + 1
    with _env(
        **{
            WS_COMPRESSION_FAIL_OPEN_ENV: None,
            WS_COMPRESSION_OVERSIZE_BYTES_ENV: "not-an-int",
        }
    ):
        action = decide_compression_failure_action(RuntimeError(), frame_bytes=big)
    assert action.refuse is True
    assert f"threshold={WS_COMPRESSION_OVERSIZE_BYTES_DEFAULT}" in action.reason


def test_threshold_zero_or_negative_ignored() -> None:
    """A bogus 0 / negative threshold must NOT silently disable the gate."""
    big = WS_COMPRESSION_OVERSIZE_BYTES_DEFAULT + 1
    with _env(
        **{
            WS_COMPRESSION_FAIL_OPEN_ENV: None,
            WS_COMPRESSION_OVERSIZE_BYTES_ENV: "0",
        }
    ):
        action = decide_compression_failure_action(RuntimeError(), frame_bytes=big)
    # default threshold still applies
    assert action.refuse is True
    assert f"threshold={WS_COMPRESSION_OVERSIZE_BYTES_DEFAULT}" in action.reason
