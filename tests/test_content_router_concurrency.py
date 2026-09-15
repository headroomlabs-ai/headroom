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
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


# =============================================================================
# PR #3556 review gap: the ContextVar fix only isolates the OUTER apply()
# call. It does not propagate into the worker threads apply() itself spawns
# for parallel compression (Pass 2 in content_router.py): the real
# `ThreadPoolExecutor` fan-out (`executor.submit(self._timed_compress, ...)`)
# and the single-pending-task deadline watchdog (`threading.Thread(target=
# _run, ...)`). `ContextVar.get()` on a freshly spawned thread returns the
# var's *global default* -- an all-defaults `_PerRequestRuntimeState()` --
# not the value `apply()` bound on the calling thread, because plain
# `threading.Thread`/`ThreadPoolExecutor.submit` do not copy the caller's
# `contextvars.Context`. The two tests below reproduce this with only ONE
# request in flight (no cross-request race needed): a single `apply()` call
# whose own `force_kompress` / `target_ratio` / `kompress_model` /
# `compression_policy` silently revert to defaults inside the fan-out.
# =============================================================================


def _two_distinct_tool_messages() -> list[dict]:
    """Two *flat string* tool messages -- large enough to each become a
    separate Pass-2 ``pending_tasks`` entry (not small, not cached) and
    distinct enough that neither collapses into the other's cache key.

    Flat-string ``content`` (not a content-block list) is required to reach
    the Pass-2 ``pending_tasks`` / ``ThreadPoolExecutor`` fan-out at all --
    Anthropic-style block content (``content: [{"type": "tool_result", ...}]``)
    is routed through ``_process_content_blocks`` instead, which always
    compresses inline on the calling thread and never spawns a worker.
    """

    def _blob(tag: str) -> str:
        return " ".join(
            f'{{"file":"src/{tag}_{i}.py","line":{i},"text":"repeated search payload {tag}"}}'
            for i in range(160)
        )

    return [
        {"role": "tool", "tool_call_id": "call_alpha", "content": _blob("alpha")},
        {"role": "tool", "tool_call_id": "call_beta", "content": _blob("beta")},
    ]


def _single_flat_tool_message(tag: str = "solo") -> list[dict]:
    """One flat-string tool message -- exactly one Pass-2 ``pending_tasks``
    entry, which routes through the single-task deadline-watchdog branch
    (content_router.py ~5478-5509), not the ``ThreadPoolExecutor`` branch.
    """
    blob = " ".join(
        f'{{"file":"src/{tag}_{i}.py","line":{i},"text":"repeated search payload {tag}"}}'
        for i in range(160)
    )
    return [{"role": "tool", "tool_call_id": f"call_{tag}", "content": blob}]


def test_parallel_fanout_workers_lose_this_requests_runtime_state(tokenizer, monkeypatch):
    """PR #3556 review (JerrettDavis): the ``ThreadPoolExecutor`` fan-out
    workers don't see the request's own runtime overrides.

    Two large tool messages force Pass 2 into the real parallel-compression
    branch (``max_workers=2 > 1``, ``len(pending_tasks)=2 > 1``). Each
    worker's ``compress()`` call is spied on to record what it observes for
    ``_runtime_force_kompress`` / ``_runtime_target_ratio`` /
    ``_runtime_kompress_model`` / ``_runtime_compression_policy`` -- these
    must equal what THIS SAME, single, apply() call actually requested. On
    the current (PR #3556) code, the worker threads instead observe the
    ContextVar's global default (False / None / None / None) because
    ``ThreadPoolExecutor.submit`` does not copy the calling thread's
    ``contextvars.Context``.
    """
    router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
    monkeypatch.setattr(router, "_get_kompress", lambda: _FakeKompress())
    monkeypatch.setenv("HEADROOM_COMPRESS_WORKERS", "4")

    policy = policy_for_mode(AuthMode.SUBSCRIPTION)
    main_thread_ident = threading.get_ident()
    observed: list[tuple[int, bool, float | None, str | None, Any]] = []
    observed_lock = threading.Lock()
    original_compress = ContentRouter.compress

    def spy_compress(self, content, *args, **kwargs):
        snapshot = (
            threading.get_ident(),
            getattr(self, "_runtime_force_kompress", False),
            getattr(self, "_runtime_target_ratio", None),
            getattr(self, "_runtime_kompress_model", None),
            getattr(self, "_runtime_compression_policy", None),
        )
        with observed_lock:
            observed.append(snapshot)
        # Widen the overlap window between the two fan-out workers. This
        # also exercises the failure mode of a *naive* fix that captures
        # one `contextvars.copy_context()` and shares it across concurrent
        # `executor.submit(ctx.run, ...)` calls: a `Context` object cannot
        # be `.run()` by two threads at once and raises `RuntimeError` --
        # that would surface here as an exception out of `router.apply()`
        # below, failing this test just as loudly as the wrong-value case.
        time.sleep(0.05)
        return original_compress(self, content, *args, **kwargs)

    monkeypatch.setattr(ContentRouter, "compress", spy_compress)

    router.apply(
        _two_distinct_tool_messages(),
        tokenizer,
        force_kompress=True,
        target_ratio=0.33,
        kompress_model="req-model-A",
        compression_policy=policy,
        min_tokens_to_compress=10,
    )

    assert len(observed) >= 2, f"expected both messages to reach compress(), got {observed}"
    assert all(ident != main_thread_ident for ident, *_ in observed), (
        "compress() ran on the apply()-calling thread, not a fan-out worker "
        "-- this test no longer exercises the parallel ThreadPoolExecutor "
        "path (content_router.py Pass 2)"
    )
    for ident, force_kompress, target_ratio, kompress_model, seen_policy in observed:
        assert force_kompress is True, (
            f"fan-out worker thread {ident} saw force_kompress={force_kompress!r} "
            "instead of this request's True -- ContextVar.get() on the worker "
            "thread returned the default _PerRequestRuntimeState instead of "
            "the state apply() bound on the calling thread"
        )
        assert target_ratio == 0.33, f"fan-out worker thread {ident} saw target_ratio={target_ratio!r}"
        assert kompress_model == "req-model-A", (
            f"fan-out worker thread {ident} saw kompress_model={kompress_model!r}"
        )
        assert seen_policy is policy, (
            f"fan-out worker thread {ident} saw compression_policy={seen_policy!r} "
            "instead of this request's own policy -- TOIN write-gating inside "
            "the fan-out is silently ungated, the exact class of leak #3486 "
            "was meant to close"
        )


def test_single_pending_task_watchdog_thread_loses_this_requests_runtime_state(
    tokenizer, monkeypatch
):
    """PR #3556 review gap, single-task variant: the deadline-bounded
    watchdog ``threading.Thread`` (content_router.py ~5487-5509) doesn't see
    the request's own runtime overrides either.

    Exactly one pending task takes the ``len(pending_tasks) == 1`` branch,
    where (with the default ``HEADROOM_COMPRESSION_DEADLINE_MS``, which is
    truthy) ``compress()`` runs inside a bare watchdog ``threading.Thread``
    rather than inline. Same underlying bug as the ThreadPoolExecutor case:
    that thread was never spawned with a copy of the calling thread's
    ``contextvars.Context``.
    """
    router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
    monkeypatch.setattr(router, "_get_kompress", lambda: _FakeKompress())
    monkeypatch.delenv("HEADROOM_COMPRESSION_DEADLINE_MS", raising=False)

    main_thread_ident = threading.get_ident()
    observed: list[tuple[int, bool, float | None, str | None]] = []
    original_compress = ContentRouter.compress

    def spy_compress(self, content, *args, **kwargs):
        observed.append(
            (
                threading.get_ident(),
                getattr(self, "_runtime_force_kompress", False),
                getattr(self, "_runtime_target_ratio", None),
                getattr(self, "_runtime_kompress_model", None),
            )
        )
        return original_compress(self, content, *args, **kwargs)

    monkeypatch.setattr(ContentRouter, "compress", spy_compress)

    router.apply(
        _single_flat_tool_message(),
        tokenizer,
        force_kompress=True,
        target_ratio=0.21,
        kompress_model="req-model-B",
        min_tokens_to_compress=10,
    )

    assert len(observed) == 1, f"expected exactly one pending task, got {observed}"
    ident, force_kompress, target_ratio, kompress_model = observed[0]
    assert ident != main_thread_ident, (
        "compress() ran inline on the apply()-calling thread -- this test no "
        "longer exercises the single-task watchdog-thread branch"
    )
    assert force_kompress is True, (
        f"watchdog thread {ident} saw force_kompress={force_kompress!r} instead "
        "of this request's True"
    )
    assert target_ratio == 0.21, f"watchdog thread {ident} saw target_ratio={target_ratio!r}"
    assert kompress_model == "req-model-B", f"watchdog thread {ident} saw kompress_model={kompress_model!r}"


# =============================================================================
# A second, previously-uncaught gotcha found during review: `apply()` sets
# TWO MORE plain, unsynchronized `self.` attributes at its top --
# `_protect_read_tool_ids` and `_protect_read_msg_indices` (content_router.py
# ~4902 / ~4924, gated by HEADROOM_PROTECT_READS) -- the exact same pattern
# #3486 fixed for `_runtime_compression_policy` et al, but these two were
# never migrated into `_PerRequestRuntimeState`/the ContextVar. They remain
# vulnerable to the ORIGINAL cross-request race: a second concurrent
# apply() call can overwrite Thread A's protected-tool-id set before Thread
# A finishes reading it (`_process_content_blocks`, content_router.py
# ~6345), silently disabling read protection for Thread A's own file-read
# content -- a genuine file read gets lossy-compressed instead of passed
# through byte-exact.
# =============================================================================


def _read_protected_tool_messages(tool_use_id: str, command: str, body: str) -> list[dict]:
    """One assistant `bash` tool_use + its tool_result, Anthropic block shape."""
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": "bash",
                    "input": {"command": command},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": body}
            ],
        },
    ]


# Plain prose, not JSON/CSV/log/diff-shaped, so `_read_output_should_be_protected`
# defaults to True (protect unless confidently a non-code data type).
def _protectable_file_body(tag: str) -> str:
    return "\n".join(
        f"{i}\tline {i} of the {tag} file with a handful of words in it" for i in range(1, 250)
    )


def test_concurrent_apply_calls_leak_read_protection_state_across_requests(
    tokenizer, monkeypatch
):
    """New gotcha (found in review, not in the original #3486 report):
    ``_protect_read_tool_ids`` was never migrated into the ContextVar-backed
    ``_PerRequestRuntimeState`` this PR introduces, so it's still vulnerable
    to the exact cross-request race #3486 was filed to fix -- just for a
    different field.

    Thread A reads its own file (``a_module.py``) via ``cat`` with
    ``HEADROOM_PROTECT_READS=1`` and must get that content back byte-exact.
    It's paused, via a patched ``_process_content_blocks``, right after
    ``_protect_read_tool_ids`` is set but before it's read. While paused,
    Thread B runs a complete, unrelated ``apply()`` call for a DIFFERENT
    file/tool_use_id, overwriting the shared ``_protect_read_tool_ids``
    attribute. Thread A resumes and reads (now Thread B's) ids -- its own
    tool_use_id no longer matches, so read protection silently fails to
    apply and its file content gets lossy-compressed.
    """
    monkeypatch.setenv("HEADROOM_PROTECT_READS", "1")

    # Control: confirm read protection genuinely applies to this exact
    # content/shape on a fresh, non-racing router, so a later "content
    # changed" assertion actually indicts the race -- not an unrelated
    # config/detection issue.
    baseline_router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
    monkeypatch.setattr(baseline_router, "_get_kompress", lambda: _FakeKompress())
    file_a = _protectable_file_body("a_module")
    baseline_result = baseline_router.apply(
        _read_protected_tool_messages("toolu_a_read", "cat a_module.py", file_a),
        tokenizer,
        force_kompress=True,
        min_tokens_to_compress=10,
    )
    assert baseline_result.messages[1]["content"][0]["content"] == file_a, (
        "sanity check failed: read protection did not protect this content in "
        "isolation, so the race assertion below would not be meaningful"
    )

    router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
    monkeypatch.setattr(router, "_get_kompress", lambda: _FakeKompress())

    thread_a_ready = threading.Event()
    proceed = threading.Event()
    thread_a_ident: dict[str, int] = {}

    original_process_content_blocks = ContentRouter._process_content_blocks

    def patched_process_content_blocks(self, *args, **kwargs):
        if threading.get_ident() == thread_a_ident.get("id"):
            thread_a_ready.set()
            if not proceed.wait(timeout=5):
                raise TimeoutError("Thread B did not signal proceed in time")
        return original_process_content_blocks(self, *args, **kwargs)

    monkeypatch.setattr(
        ContentRouter, "_process_content_blocks", patched_process_content_blocks
    )

    errors: list[BaseException] = []
    result_a: dict[str, Any] = {}

    def run_thread_a():
        thread_a_ident["id"] = threading.get_ident()
        try:
            result_a["result"] = router.apply(
                _read_protected_tool_messages("toolu_a_read", "cat a_module.py", file_a),
                tokenizer,
                force_kompress=True,
                min_tokens_to_compress=10,
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread_a = threading.Thread(target=run_thread_a)
    thread_a.start()

    assert thread_a_ready.wait(timeout=5), "Thread A never reached _process_content_blocks"

    # Thread B: a fully independent, concurrent request for a DIFFERENT file.
    # Runs to completion while Thread A is paused mid-call.
    file_b = _protectable_file_body("b_module")
    router.apply(
        _read_protected_tool_messages("toolu_b_read", "cat b_module.py", file_b),
        tokenizer,
        force_kompress=True,
        min_tokens_to_compress=10,
    )

    proceed.set()
    thread_a.join(timeout=5)
    assert not thread_a.is_alive(), "Thread A did not finish"
    assert not errors, f"Thread A raised: {errors}"

    tool_result_block = result_a["result"].messages[1]["content"][0]
    assert tool_result_block["content"] == file_a, (
        "Thread A's genuine file-read content was mutated -- its own "
        "_protect_read_tool_ids entry ('toolu_a_read') was overwritten by "
        "concurrent Thread B's apply() call ('toolu_b_read') before Thread A "
        "could read it. _protect_read_tool_ids/_protect_read_msg_indices are "
        "plain, unsynchronized self attributes (content_router.py ~4902, "
        "~4924) that PR #3556 did not migrate into the ContextVar-backed "
        "_PerRequestRuntimeState -- same #3486 race, different field."
    )


# =============================================================================
# Follow-up review (of THIS fix, not just #3556): the two tests above each
# exercise one dimension of concurrency in isolation -- either two concurrent
# apply() calls with no internal fan-out (the original #3486 shape), or one
# apply() call whose internal fan-out is inspected without any OTHER apply()
# call running at the same time. Neither proves the per-task
# `contextvars.copy_context()` snapshot in Pass 2 is truly request-exclusive
# under the combined, real-world shape: many concurrent proxy requests, each
# ALSO fanning out internally. In particular this guards against a regression
# where a future edit hoists `copy_context()` out of the per-task loop (or
# otherwise shares one snapshot across submissions), which would only be
# visible once TWO DIFFERENT requests are racing each other's fan-outs at
# the same time.
# =============================================================================


def _two_distinct_tool_messages_tagged(tag: str) -> list[dict]:
    """Two large flat-string tool messages carrying `tag` in their content,
    so a spy on `compress()` can attribute each fan-out worker's call back
    to the `apply()` call (request) it belongs to.
    """

    def _blob(sub: str) -> str:
        return " ".join(
            f'{{"file":"src/{tag}_{sub}_{i}.py","line":{i},"text":"{tag} payload"}}'
            for i in range(160)
        )

    return [
        {"role": "tool", "tool_call_id": f"call_{tag}_alpha", "content": _blob("alpha")},
        {"role": "tool", "tool_call_id": f"call_{tag}_beta", "content": _blob("beta")},
    ]


def test_concurrent_apply_calls_with_internal_fanout_do_not_cross_contaminate(
    tokenizer, monkeypatch
):
    """Combines both concurrency dimensions in one test: two concurrent
    ``apply()`` calls (cross-request concurrency, the original #3486 shape)
    where EACH call also internally fans out via the real
    ``ThreadPoolExecutor`` (Pass 2, the PR #3556 review gap this fix closes).

    Confirms request A's fan-out workers only ever observe request A's
    runtime state and request B's workers only ever observe request B's --
    i.e. that the per-task ``copy_context()`` snapshot is genuinely
    request-exclusive, not just "correct when only one apply() call is
    running at a time."
    """
    router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
    monkeypatch.setattr(router, "_get_kompress", lambda: _FakeKompress())
    monkeypatch.setenv("HEADROOM_COMPRESS_WORKERS", "4")

    observed: list[tuple[str, bool, float | None, str | None]] = []
    observed_lock = threading.Lock()
    original_compress = ContentRouter.compress

    def spy_compress(self, content, *args, **kwargs):
        if "reqA" in content:
            tag = "reqA"
        elif "reqB" in content:
            tag = "reqB"
        else:
            tag = "unknown"
        with observed_lock:
            observed.append(
                (
                    tag,
                    getattr(self, "_runtime_force_kompress", False),
                    getattr(self, "_runtime_target_ratio", None),
                    getattr(self, "_runtime_kompress_model", None),
                )
            )
        # Widen the overlap window so reqA's and reqB's fan-outs genuinely
        # interleave rather than one finishing before the other starts.
        time.sleep(0.05)
        return original_compress(self, content, *args, **kwargs)

    monkeypatch.setattr(ContentRouter, "compress", spy_compress)

    errors: list[BaseException] = []

    def run(tag: str, target_ratio: float, kompress_model: str) -> None:
        try:
            router.apply(
                _two_distinct_tool_messages_tagged(tag),
                tokenizer,
                force_kompress=True,
                target_ratio=target_ratio,
                kompress_model=kompress_model,
                min_tokens_to_compress=10,
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread_a = threading.Thread(target=run, args=("reqA", 0.11, "model-A"))
    thread_b = threading.Thread(target=run, args=("reqB", 0.77, "model-B"))
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=10)
    thread_b.join(timeout=10)

    assert not errors, f"apply() raised: {errors}"
    assert not thread_a.is_alive() and not thread_b.is_alive(), "a thread did not finish in time"

    reqA_obs = [o for o in observed if o[0] == "reqA"]
    reqB_obs = [o for o in observed if o[0] == "reqB"]
    assert len(reqA_obs) >= 2, f"expected reqA's fan-out to reach compress() twice, got {observed}"
    assert len(reqB_obs) >= 2, f"expected reqB's fan-out to reach compress() twice, got {observed}"

    for tag, force_kompress, target_ratio, kompress_model in reqA_obs:
        assert force_kompress is True, f"reqA worker saw force_kompress={force_kompress!r}"
        assert target_ratio == 0.11, (
            f"reqA fan-out worker saw target_ratio={target_ratio!r} instead of reqA's own "
            "0.11 -- cross-request contamination from reqB's concurrently-running fan-out"
        )
        assert kompress_model == "model-A", (
            f"reqA fan-out worker saw kompress_model={kompress_model!r} instead of "
            "reqA's own 'model-A' -- cross-request contamination from reqB"
        )

    for tag, force_kompress, target_ratio, kompress_model in reqB_obs:
        assert force_kompress is True, f"reqB worker saw force_kompress={force_kompress!r}"
        assert target_ratio == 0.77, (
            f"reqB fan-out worker saw target_ratio={target_ratio!r} instead of reqB's own "
            "0.77 -- cross-request contamination from reqA's concurrently-running fan-out"
        )
        assert kompress_model == "model-B", (
            f"reqB fan-out worker saw kompress_model={kompress_model!r} instead of "
            "reqB's own 'model-B' -- cross-request contamination from reqA"
        )
