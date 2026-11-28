"""Cross-turn cache-safety regression for the OpenAI-format proxy path.

Companion to test_cross_turn_cache_safety.py (Anthropic path, commit 248ae0f3 /
#1850). All three OpenAI-path ``tracker.update_from_response`` call sites
(openai.py:3413, openai.py:3710, streaming.py:1935) used to omit
``original_messages``, so ``PrefixCacheTracker`` fell back to storing the
FORWARDED (compressed) bytes as "originals". The overlay's append-only check
then diverged at message 0 on turn 2 and never replayed the cached prefix,
busting the provider's prompt cache on every turn.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.cache.prefix_tracker import PrefixCacheTracker, overlay_cached_prefix
from headroom.proxy.server import ProxyConfig, create_app

# ─── Mechanism-level: proves the fix is load-bearing ────────────────────
#
# Mirrors test_cross_turn_cache_safety.py's drive loop, isolating exactly the
# OpenAI-handler bug: calling update_from_response WITHOUT original_messages
# (the old, broken call shape) vs WITH it (current code).


def _drive_two_turns(*, record_originals: bool) -> tuple[list[dict], list[dict]]:
    """Simulate 2 append-only turns. Returns (turn1_forwarded, turn2_overlaid)."""
    tracker = PrefixCacheTracker("openai")

    turn1_original = [{"role": "user", "content": "please read this long file: " + "x" * 100}]
    turn1_forwarded = [{"role": "user", "content": "SHORT"}]  # turn 1: pipeline compresses it

    tracker.update_from_response(
        cache_read_tokens=0,
        cache_write_tokens=50,
        messages=turn1_forwarded,
        **({"original_messages": turn1_original} if record_originals else {}),
    )

    # Turn 2: client resends turn 1 verbatim (append-only) plus a new message.
    # The pipeline skips compressing already-forwarded content this turn (as
    # the real one does for a frozen/stable prefix), so it emits the RAW
    # original bytes unless the overlay steps in.
    turn2_original = [*turn1_original, {"role": "user", "content": "continue"}]
    turn2_pipeline_output = list(turn2_original)

    overlaid = overlay_cached_prefix(
        turn2_pipeline_output,
        turn2_original,
        tracker.get_last_original_messages(),
        tracker.get_last_forwarded_messages(),
    )
    return turn1_forwarded, overlaid


def test_missing_original_messages_busts_cache_on_turn_2() -> None:
    """Regression: omitting original_messages (the pre-fix call shape) busts the cache."""
    turn1_forwarded, overlaid = _drive_two_turns(record_originals=False)
    # Without originals recorded, the tracker's "originals" == turn1_forwarded
    # (the compressed bytes), which never equals turn2's raw client prefix ->
    # append-only guard fails -> overlay is a no-op -> turn 2 forwards the raw
    # bytes, diverging from what the provider actually cached.
    assert overlaid[0] != turn1_forwarded[0]


def test_recording_original_messages_keeps_prefix_byte_identical() -> None:
    """Fix: passing original_messages lets the overlay replay the cached prefix."""
    turn1_forwarded, overlaid = _drive_two_turns(record_originals=True)
    assert overlaid[0] == turn1_forwarded[0]


# ─── Handler-level: proves the real openai.py wiring is correct ─────────


def _make_client() -> TestClient:
    config = ProxyConfig(
        optimize=True,
        mode="token",
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    return TestClient(create_app(config))


def test_openai_direct_path_threads_original_messages_into_tracker() -> None:
    """handle_openai_chat (no-backend path) must record raw client messages as
    originals, not the pipeline's compressed output — else the two-turn
    overlay replay (the cache-safety mechanism) never engages."""
    with _make_client() as client:
        proxy = client.app.state.proxy
        real_tracker = PrefixCacheTracker("openai")
        proxy.session_tracker_store.compute_session_id = (
            lambda request, model, messages, **_kwargs: "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: real_tracker

        calls = {"n": 0}

        def _fake_apply(**kwargs):
            calls["n"] += 1
            msgs = list(kwargs["messages"])
            if calls["n"] == 1:
                # First turn: pipeline compresses the long message.
                msgs = [{**m, "content": "SHORT"} for m in msgs]
            # Later turns: pipeline skips the already-stable prefix, emitting
            # whatever the caller sent (mirrors real frozen-prefix behavior).
            return SimpleNamespace(
                messages=msgs,
                transforms_applied=["test:compress"] if calls["n"] == 1 else [],
                timing={},
                tokens_before=100,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.openai_pipeline.apply = _fake_apply

        captured_bodies: list[dict] = []

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):
            captured_bodies.append(body)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_1",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                },
            )

        proxy._retry_request = _fake_retry

        turn1_original = [{"role": "user", "content": "please read this long file: " + "x" * 100}]
        r1 = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer [REDACTED:Bearer token]"},
            json={"model": "gpt-4o-mini", "messages": turn1_original, "stream": False},
        )
        assert r1.status_code == 200, r1.text
        turn1_forwarded = captured_bodies[0]["messages"]
        assert turn1_forwarded[0]["content"] == "SHORT"

        turn2_original = [*turn1_original, {"role": "user", "content": "continue"}]
        r2 = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer [REDACTED:Bearer token]"},
            json={"model": "gpt-4o-mini", "messages": turn2_original, "stream": False},
        )
        assert r2.status_code == 200, r2.text
        turn2_forwarded = captured_bodies[1]["messages"]

        # The fix: turn 2 replays turn 1's compressed prefix byte-for-byte
        # instead of re-forwarding the raw original, which would bust the
        # provider's cache.
        assert turn2_forwarded[0] == turn1_forwarded[0] == {"role": "user", "content": "SHORT"}
