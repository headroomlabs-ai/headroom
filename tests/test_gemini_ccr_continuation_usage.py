"""A Gemini CCR continuation with a present-null usage count must not 502.

The initial-response and non-CCR extraction sites guard against Gemini
returning a *present-null* ``promptTokenCount`` (a key that is present with a
JSON ``null`` value, which ``.get(key, default)`` returns as ``None`` rather
than the default). The CCR-continuation site re-read the continuation's
``usageMetadata`` with a bare ``.get(key, prior)`` and skipped that guard, so a
present-null count on the continuation turned the ``max(0, prompt - cache_read)``
arithmetic into ``None`` math, raised ``TypeError``, and the outer handler
masked a successful 200 as a synthetic 502.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from headroom.proxy.handlers.gemini import GeminiHandlerMixin


class _FakeRequest:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.query_params: dict[str, str] = {}
        self.url = SimpleNamespace(path="/v1beta/models/gemini-pro:generateContent", query="")
        # Every real Starlette Request carries a scope, and the Gemini handler
        # binds the savings-attribution ledger to it (#3051). Without this the
        # double is a shape that cannot occur in production.
        self.scope: dict = {"type": "http", "method": "POST"}


class _CcrToolCallResponse:
    """Initial 200 carrying a CCR tool call and a valid promptTokenCount."""

    status_code = 200
    content = json.dumps(
        {
            "candidates": [
                {"content": {"parts": [{"functionCall": {"name": "headroom_retrieve"}}]}}
            ],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 5},
        }
    ).encode()
    headers = {"content-type": "application/json"}

    def json(self) -> object:
        return json.loads(self.content)


class _CcrConfig:
    enabled = True


class _CcrHandler:
    """Stub CCR handler whose continuation reports a present-null usage count."""

    config = _CcrConfig()

    def has_ccr_tool_calls(self, resp_json, provider) -> bool:  # noqa: ANN001
        return True

    async def handle_response(self, resp_json, contents, native_fns, api_call_fn, provider):  # noqa: ANN001, ANN201
        return {
            "candidates": [{"content": {"parts": [{"text": "resolved"}]}}],
            # The continuation turn omits real counts as JSON null.
            "usageMetadata": {
                "promptTokenCount": None,
                "candidatesTokenCount": None,
                "cachedContentTokenCount": None,
            },
        }

    def residual_ccr_status(self, final_resp_json, provider):  # noqa: ANN001, ANN201
        return None  # not RESIDUAL_CCR_ERROR


class _FakeMetrics:
    def __init__(self) -> None:
        self.failed: list[str] = []

    async def record_failed(self, *, provider: str, model: str = "") -> None:
        self.failed.append(f"{provider}:{model}")


class _Handler(GeminiHandlerMixin):
    GEMINI_API_URL = "https://gemini.example"

    def __init__(self) -> None:
        self.memory_handler = None
        self.rate_limiter = None
        self.usage_reporter = None
        self.config = SimpleNamespace(
            optimize=False,
            anthropic_pre_upstream_memory_context_timeout_seconds=0.1,
        )
        self.metrics = _FakeMetrics()
        self.ccr_response_handler = _CcrHandler()
        self.outcomes: list = []

    async def _next_request_id(self) -> str:
        return "req-ccr-1"

    async def _retry_request(self, method, url, headers, body):  # noqa: ANN001, ANN201
        return _CcrToolCallResponse()

    async def _record_request_outcome(self, outcome) -> None:  # noqa: ANN001
        self.outcomes.append(outcome)

    async def _count_tokens_offloaded(self, model, messages):  # noqa: ANN001, ANN201
        return SimpleNamespace(), 100


@pytest.mark.asyncio
async def test_ccr_continuation_present_null_usage_does_not_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def payload(request):  # noqa: ANN001, ANN201
        return {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", payload)

    handler = _Handler()
    response = await handler.handle_gemini_generate_content(_FakeRequest(), "gemini-pro")

    # Before the fix this raised TypeError on the None arithmetic and the outer
    # handler returned a synthetic 502 with a recorded failure.
    assert response.status_code == 200
    assert handler.metrics.failed == []
    assert handler.outcomes[0].status_code == 200
    # The pre-continuation count (100) survives as the fallback.
    assert handler.outcomes[0].optimized_tokens == 100
    assert handler.outcomes[0].cache_read_tokens == 0
