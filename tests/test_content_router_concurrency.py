"""Regression test for issue #3486: ContentRouter per-request state isolation.

``ContentRouter`` is instantiated once at proxy startup and shared across
all concurrent requests (see ``headroom/proxy/server.py``, which dispatches
``pipeline.apply()`` calls onto a real ``ThreadPoolExecutor``). ``apply()``
stores per-call runtime state -- including the F2.2
``self._runtime_compression_policy`` -- as plain, unsynchronized instance
attributes (``content_router.py`` around line 4837). A second concurrent
``apply()`` call can overwrite that attribute before the first call has
finished reading it, corrupting the first request's view of its own
compression policy with a *different* request's policy.

This test forces that interleaving deterministically: Thread A starts an
``apply()`` call under the Subscription policy (``toin_read_only=True`` --
must NEVER write to the shared TOIN learning pool) and is paused, via a
patched ``_record_to_toin``, right before it reads
``self._runtime_compression_policy``. While paused, Thread B runs a
complete ``apply()`` call under the PAYG policy, which overwrites the
shared attribute. Thread A is then resumed: on the current (unfixed) code
it reads Thread B's PAYG policy instead of its own Subscription policy and
incorrectly writes Subscription-mode content into the shared, cross-user
TOIN learning pool -- a privacy-relevant cross-tenant leak the F2.2 gate
exists specifically to prevent.

See ``tests/test_compression_policy_toin_gate.py`` for the (non-concurrent)
F2.2 gate tests this mirrors, and
``tests/test_code_compressor_thread_safety.py`` for the
Event/ThreadPoolExecutor style this follows.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from headroom.proxy.auth_mode import AuthMode
from headroom.telemetry.toin import TOINConfig, get_toin, reset_toin
from headroom.transforms.compression_policy import policy_for_mode
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig


@pytest.fixture
def fresh_toin():
    """Per-test TOIN instance backed by a tempdir to avoid global drift."""
    reset_toin()
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = str(Path(tmpdir) / "toin.json")
        toin = get_toin(TOINConfig(storage_path=storage, auto_save_interval=0))
        yield toin
        reset_toin()


@pytest.fixture
def tokenizer():
    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer

    provider = OpenAIProvider()
    token_counter = provider.get_token_counter("gpt-4o")
    return Tokenizer(token_counter, "gpt-4o")


class _FakeKompress:
    """Deterministic stand-in for the ML Kompress stage.

    Truncates to the first 20 words, so the compressed result is always
    reliably shorter than the (160-item) input regardless of whether the
    real ML model is installed/ready -- guarantees the
    ``original_tokens > compressed_tokens`` check in ``_record_to_toin``
    passes on every run.
    """

    def is_ready(self) -> bool:
        return True

    def ensure_background_load(self) -> None:
        pass

    def compress(self, content: str, **kwargs):
        compressed = " ".join(content.split()[:20]) + " Retrieve more: hash=deadbeef"
        return SimpleNamespace(
            compressed=compressed, compressed_tokens=len(compressed.split())
        )


def _kompress_forcing_tool_message() -> list[dict]:
    """A tool_result message that reaches ContentRouter's own
    ``_record_to_toin`` call site via the real ``apply()`` pipeline.

    Mirrors ``test_force_kompress_...`` in
    ``tests/test_transforms/test_content_router.py``. Confirmed by direct
    instrumentation that with ``force_kompress=True`` this content lands on
    the router's own TOIN-recording call (content_router.py:3694) -- NOT
    the SmartCrusher path, which has its own separate, already-tested gate.
    """
    tool_content = " ".join(
        f'{{"file":"src/module_{i}.py","line":{i},"text":"repeated search payload"}}'
        for i in range(160)
    )
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_search_1",
                    "content": tool_content,
                }
            ],
        }
    ]


def test_concurrent_apply_calls_leak_compression_policy_across_requests(
    fresh_toin, tokenizer, monkeypatch
):
    """#3486: a concurrent PAYG ``apply()`` call must not flip a
    Subscription request's TOIN write-gate to "write enabled".

    ``ContentRouter`` is a shared singleton (one instance for the whole
    proxy process); ``apply()`` stashes the per-call ``CompressionPolicy``
    on ``self._runtime_compression_policy`` with no locking and no
    per-request scoping. This test proves that a second, unrelated
    concurrent request corrupts the first request's view of its own
    policy -- Subscription-mode content ends up written into the
    cross-user TOIN learning pool, which the F2.2 gate exists specifically
    to prevent.
    """
    router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
    monkeypatch.setattr(router, "_get_kompress", lambda: _FakeKompress())

    subscription_policy = policy_for_mode(AuthMode.SUBSCRIPTION)
    payg_policy = policy_for_mode(AuthMode.PAYG)

    thread_a_ready = threading.Event()
    proceed = threading.Event()
    thread_a_ident: dict[str, int] = {}

    original_record_to_toin = ContentRouter._record_to_toin

    def patched_record_to_toin(self, *args, **kwargs):
        if threading.get_ident() == thread_a_ident.get("id"):
            thread_a_ready.set()
            # Give Thread B a chance to run its *entire* apply() call and
            # overwrite self._runtime_compression_policy before Thread A's
            # _record_to_toin reads it below.
            if not proceed.wait(timeout=5):
                raise TimeoutError("Thread B did not signal proceed in time")
        return original_record_to_toin(self, *args, **kwargs)

    monkeypatch.setattr(ContentRouter, "_record_to_toin", patched_record_to_toin)

    errors: list[BaseException] = []

    def run_thread_a():
        thread_a_ident["id"] = threading.get_ident()
        try:
            router.apply(
                _kompress_forcing_tool_message(),
                tokenizer,
                force_kompress=True,
                target_ratio=0.10,
                compress_user_messages=True,
                min_tokens_to_compress=10,
                read_protection_window=0,
                compression_policy=subscription_policy,
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    pre = sum(p.total_compressions for p in fresh_toin._patterns.values())

    thread_a = threading.Thread(target=run_thread_a)
    thread_a.start()

    assert thread_a_ready.wait(timeout=5), "Thread A never reached _record_to_toin"

    # Thread B: a fully independent, concurrent request under PAYG.
    # Runs to completion while Thread A is paused mid-call.
    router.apply([], tokenizer, compression_policy=payg_policy)

    proceed.set()
    thread_a.join(timeout=5)
    assert not thread_a.is_alive(), "Thread A did not finish"
    assert not errors, f"Thread A raised: {errors}"

    post = sum(p.total_compressions for p in fresh_toin._patterns.values())
    assert post == pre, (
        "Thread A's request used the Subscription policy "
        "(toin_read_only=True) and must NEVER write to the shared TOIN "
        "learning pool -- even though an unrelated concurrent PAYG "
        "request ran in between. A write here means "
        "self._runtime_compression_policy leaked across concurrent "
        "requests on the shared ContentRouter singleton (#3486)."
    )
