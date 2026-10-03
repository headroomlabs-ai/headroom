"""The direct chat-completions path must answer the retrieval tool it injects.

`headroom_retrieve` is injected into a non-streaming chat request by default
(`ccr_inject_tool=True`), but `handlers/openai.py`'s direct HTTP branch had no
CCR tool-call handling at all — the code said so itself. Every other path
resolves the call: `gateway_turn.py:1003`, `handlers/gemini.py:837`,
`handlers/anthropic.py:4286`, `handlers/openai.py` for the custom backend and
for Responses. Only the direct chat branch did not.

So the proxy advertised a tool it would not answer. A model that called it sent
the client a `tool_calls` entry for a tool the client never declared and cannot
implement, which stalls any agent loop driven off `finish_reason`.

The accounting pair is the part worth keeping honest. A resolved turn makes two
billed upstream calls, and the continuation resends the whole conversation, so
counting only the last one hides roughly half the turn. The split matters too:
the forwarded body keeps the provider's own usage untouched (byte-faithful
forwarding), while Headroom's cost tracking folds in the extra call — the same
division the turn-hook re-drive path already uses via `TurnHookUsage`.
"""

from __future__ import annotations

import json
import logging

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.cache.compression_store import (  # noqa: E402
    get_compression_store,
    reset_compression_store,
)
from headroom.ccr.tool_injection import CCR_TOOL_NAME  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

ORIGINAL = "row 1: the original uncompressed rows\nrow 2: more of them"


@pytest.fixture(autouse=True)
def reset_store():
    reset_compression_store()
    yield
    reset_compression_store()


def _stored_hash() -> str:
    return get_compression_store().store(
        original=ORIGINAL,
        compressed="[2 items compressed to 0]",
        original_item_count=2,
        compressed_item_count=0,
    )


def _config() -> ProxyConfig:
    # No backend configured -> the direct OpenAI HTTP path.
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        ccr_handle_responses=True,
        ccr_inject_tool=True,
    )


def _retrieve_call_response(hash_key: str) -> dict:
    """An upstream reply in which the model asks to retrieve."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": CCR_TOOL_NAME,
                                "arguments": json.dumps({"hash": hash_key}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


def _final_response() -> dict:
    """The continuation reply, after the tool result was supplied."""
    return {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "there are 2 rows"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 400, "completion_tokens": 10, "total_tokens": 410},
    }


class _CollectHandler(logging.Handler):
    """Collect `headroom.*` records regardless of propagation.

    `caplog` attaches to the root logger, but the proxy's own startup
    (`helpers._setup_file_logging`) sets `propagate = False` on `headroom`, so
    records never reach root — and it does that *during* `create_app`, after
    conftest's autouse reset has already run. A handler on the `headroom`
    logger itself still fires, because `propagate` only governs whether records
    are passed further up. Anything asserting on log contents here has to go
    through this, or it silently asserts against an empty string.
    """

    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.DEBUG)
        self._sink = sink
        self.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink.append(self.format(record))
        except Exception:  # pragma: no cover - a test handler must not raise
            pass


def _run(
    upstream_sequence: list, log_capture: list[str] | None = None
) -> tuple[httpx.Response, list[dict], list]:
    """Post one chat turn, serving `upstream_sequence` to successive calls.

    Each entry is either a payload dict (served as `200`) or an explicit
    `(status, payload)` pair — non-2xx continuations are the whole point of the
    status-handling tests, and `_retry_request` hands those back without
    raising.

    Returns the response, the request bodies that went upstream, and the
    RequestOutcome objects the proxy recorded.
    """
    sent: list[dict] = []
    outcomes: list = []
    remaining = list(upstream_sequence)

    def _next() -> tuple[int, dict]:
        entry = remaining.pop(0) if remaining else upstream_sequence[-1]
        if isinstance(entry, tuple):
            return entry
        return 200, entry

    async def fake_retry(method, url, headers, req_body, *args, **kwargs):
        sent.append(req_body)
        status, payload = _next()
        return httpx.Response(status, json=payload, headers={"content-type": "application/json"})

    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy._retry_request = fake_retry

        # The continuation on this path may go out through the shared client
        # rather than _retry_request; capture both so the test does not depend
        # on which transport the implementation picks.
        class _FakeHTTPClient:
            async def post(self, url, content=None, headers=None, **kwargs):  # noqa: ANN001
                import json as _json

                sent.append(_json.loads(content) if content else {})
                status, payload = _next()
                return httpx.Response(
                    status, json=payload, headers={"content-type": "application/json"}
                )

        proxy.http_client = _FakeHTTPClient()

        _real_record = proxy._record_request_outcome

        async def _capture(outcome, *args, **kwargs):  # noqa: ANN001
            outcomes.append(outcome)
            return await _real_record(outcome, *args, **kwargs)

        proxy._record_request_outcome = _capture

        # After startup, so the proxy's own logging setup cannot undo it.
        _handler: logging.Handler | None = None
        if log_capture is not None:
            _headroom_logger = logging.getLogger("headroom")
            _headroom_logger.disabled = False
            _headroom_logger.setLevel(logging.DEBUG)
            _handler = _CollectHandler(log_capture)
            _headroom_logger.addHandler(_handler)

        try:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "how many rows?"}],
                    "stream": False,
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "Read", "description": "read a file"},
                        }
                    ],
                },
                headers={"Authorization": "Bearer test-key", "x-api-key": "test-key"},
            )
        finally:
            if _handler is not None:
                logging.getLogger("headroom").removeHandler(_handler)
        return resp, sent, outcomes


def test_a_retrieval_call_is_resolved_before_the_client_sees_it():
    """The client must never receive a headroom_retrieve tool call."""
    hash_key = _stored_hash()
    resp, sent, _outcomes = _run([_retrieve_call_response(hash_key), _final_response()])

    assert resp.status_code == 200, resp.text
    body = resp.json()
    message = body["choices"][0]["message"]

    tool_calls = message.get("tool_calls") or []
    names = [tc.get("function", {}).get("name") for tc in tool_calls]
    assert CCR_TOOL_NAME not in names, (
        "the proxy injected headroom_retrieve and then handed the model's call "
        "straight to the client, which cannot implement it"
    )
    assert message.get("content") == "there are 2 rows"
    assert len(sent) == 2, f"expected an original call plus a continuation, got {len(sent)}"


def test_the_continuation_carries_the_retrieved_content():
    """The retrieved original has to reach the model, not just be looked up."""
    hash_key = _stored_hash()
    _resp, sent, _outcomes = _run([_retrieve_call_response(hash_key), _final_response()])

    assert len(sent) == 2
    continuation = sent[1]
    serialized = str(continuation)
    assert ORIGINAL.split("\n")[0] in serialized, (
        "the continuation request did not carry the retrieved original content"
    )


def test_the_forwarded_body_keeps_the_provider_usage_untouched():
    """Byte-faithful forwarding: the client sees the provider's own final usage.

    Rewriting `usage` in the forwarded body would misreport what the upstream
    call actually returned. Extra calls belong in Headroom's own accounting,
    which the next test covers — this is the same split the turn-hook re-drive
    path already uses.
    """
    hash_key = _stored_hash()
    resp, _sent, _outcomes = _run([_retrieve_call_response(hash_key), _final_response()])
    assert (resp.json().get("usage") or {}).get("prompt_tokens") == 400


ERROR_BODY = {"error": {"message": "invalid api key", "type": "invalid_request_error"}}

# A value that must never be logged. Upstream error bodies are untrusted and
# routinely carry credential fragments, tenant identifiers and excerpts of the
# request that produced them.
SENTINEL = "sk-live-SENTINEL-must-never-be-logged-8f3a1c"
ERROR_BODY_WITH_SECRET = {
    "error": {
        "message": f"authentication failed for key {SENTINEL} (tenant acme-prod-42)",
        "type": "invalid_request_error",
    }
}


@pytest.mark.parametrize("status", [401, 429, 500])
def test_a_failed_continuation_body_never_reaches_the_logs(status):
    """The status is diagnostic enough; the body is untrusted content.

    Two paths would leak it, and both have to stay closed:

    * anything this module logs itself, and
    * the exception's own message — `handle_response` logs `repr(e)` when a
      continuation raises (`ccr/response_handler.py:541`), so a preview placed
      on the exception reaches the log even if this module never logs it.

    The positive assertion at the end is load-bearing: it is what proves the
    capture is actually wired to the `headroom` logger. Without it the negative
    assertions pass against an empty string and the test is worthless.
    """
    hash_key = _stored_hash()
    logs: list[str] = []
    resp, sent, _outcomes = _run(
        [_retrieve_call_response(hash_key), (status, ERROR_BODY_WITH_SECRET)],
        log_capture=logs,
    )
    text = "\n".join(logs)

    assert len(sent) == 2, "the continuation should still have been attempted"
    # Proves the capture works, so the assertions below mean something.
    assert f"HTTP {status}" in text, (
        f"the {status} status was not logged, so either the failure is invisible "
        "or this test is not capturing headroom's logs at all"
    )
    assert SENTINEL not in text, (
        "an upstream error body reached the logs; even a bounded prefix is an "
        "exfiltration path for credentials and tenant identifiers"
    )
    assert "acme-prod-42" not in text, "a tenant identifier reached the logs"
    assert SENTINEL not in resp.text, "the upstream error body reached the client"


@pytest.mark.parametrize(
    "status",
    [
        401,  # any 4xx: `_retry_request` returns it without raising
        429,  # retries exhausted, returned verbatim
        500,  # retries exhausted, returned via the HTTPStatusError branch
    ],
)
def test_a_non_2xx_continuation_never_reaches_the_client_as_200(status):
    """An upstream error must not be served under the first call's `200`.

    `_retry_request` deliberately does not raise for anything it will not retry,
    so a continuation failure looks exactly like a success unless the status is
    checked. Parsing that body and returning it as the model's answer produced
    an HTTP 200 whose payload was an error object with no `choices` at all — a
    shape no client expects, and one that hides an auth or quota failure.
    """
    hash_key = _stored_hash()
    resp, sent, _outcomes = _run([_retrieve_call_response(hash_key), (status, ERROR_BODY)])

    assert len(sent) == 2, "the continuation should still have been attempted"
    body = resp.json()
    assert "error" not in body, (
        f"a {status} continuation body was forwarded as the model's reply "
        f"under HTTP {resp.status_code}"
    )
    assert body.get("choices"), "the forwarded body lost its choices"


@pytest.mark.parametrize("status", [401, 429, 500])
def test_a_failed_continuation_forwards_the_original_reply_unchanged(status):
    """Fail open, and byte-for-byte.

    `handle_response` swallows a continuation failure and returns the response
    it already had, which is the same contract a transport error or timeout on
    that call already gets. The client therefore sees the model's original
    reply — including the unresolved tool call it cannot run, which is strictly
    better than an error object dressed as a completion, and is the documented
    behaviour for every other way a continuation can fail.

    Byte fidelity matters here too: nothing was resolved, so the upstream bytes
    must be forwarded as they arrived rather than re-serialized from a parsed
    dict.
    """
    hash_key = _stored_hash()
    original = _retrieve_call_response(hash_key)
    resp, _sent, _outcomes = _run([original, (status, ERROR_BODY)])

    assert resp.status_code == 200
    assert resp.json() == original, "the original upstream reply was not forwarded unchanged"
    assert resp.content == httpx.Response(200, json=original).content, (
        "the body was re-serialized instead of forwarded verbatim"
    )


@pytest.mark.parametrize("status", [401, 429, 500])
def test_a_failed_continuation_is_not_billed(status):
    """A non-2xx continuation returns no usage, so it must not be counted.

    Only the first call was billed. Recording the error response — or settling
    against the wrong object — would inflate the turn with tokens that were
    never charged, which is the same class of bug in the opposite direction as
    the dropped first call.
    """
    hash_key = _stored_hash()
    _resp, sent, outcomes = _run([_retrieve_call_response(hash_key), (status, ERROR_BODY)])

    assert len(sent) == 2
    assert outcomes, "no RequestOutcome was recorded"
    recorded = outcomes[-1].provider_input_tokens
    assert recorded == 100, (
        f"cost tracking saw {recorded} input tokens; only the first call "
        "(100) was billed, the continuation returned an error with no usage"
    )


def test_headroom_accounting_counts_both_upstream_calls():
    """A retrieval turn makes two billed calls; cost tracking must see both.

    `handle_response` replaces the response rather than merging usage, so the
    pre-continuation call's tokens are invisible unless they are folded in
    explicitly. The continuation resends the whole conversation, so counting
    only the last call hides roughly half the turn — the same mistake the
    re-drive block warns about ("counting only the last one lets the feature
    hide its own overhead behind the saving it is claiming").
    """
    hash_key = _stored_hash()
    _resp, sent, outcomes = _run([_retrieve_call_response(hash_key), _final_response()])
    assert len(sent) == 2
    assert outcomes, "no RequestOutcome was recorded"

    recorded = outcomes[-1].provider_input_tokens
    assert recorded >= 100 + 400, (
        f"cost tracking saw {recorded} input tokens for a turn that made two "
        "upstream calls costing 100 + 400; the first call was dropped"
    )
