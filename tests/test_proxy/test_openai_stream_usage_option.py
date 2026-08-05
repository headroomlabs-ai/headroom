"""Regression: the direct OpenAI streaming path must not override an explicit
client ``stream_options.include_usage``.

The handler injects ``include_usage`` so it can count tokens from the trailing
usage chunk. Forcing it on a client that passed ``include_usage: false`` (or that
never opted in) makes the upstream append a usage-only chunk (``choices: []``)
the client did not ask for — and the common ``chunk.choices[0].delta`` loop then
raises ``IndexError``. The injection now only fills in the option when the client
left the choice open.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from headroom.proxy.handlers.openai import _apply_stream_usage_option  # noqa: E402


def test_respects_explicit_include_usage_false():
    body = {"stream_options": {"include_usage": False}}
    _apply_stream_usage_option(body)
    assert body["stream_options"]["include_usage"] is False


def test_preserves_explicit_include_usage_true():
    body = {"stream_options": {"include_usage": True}}
    _apply_stream_usage_option(body)
    assert body["stream_options"]["include_usage"] is True


def test_injects_when_stream_options_absent():
    body = {"messages": []}
    _apply_stream_usage_option(body)
    assert body["stream_options"] == {"include_usage": True}


def test_fills_in_when_dict_present_without_key():
    body = {"stream_options": {"continuous_usage_stats": True}}
    _apply_stream_usage_option(body)
    assert body["stream_options"] == {
        "continuous_usage_stats": True,
        "include_usage": True,
    }
