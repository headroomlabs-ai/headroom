"""Gemini memory context must reach the wire on entries with non-text parts (#4419).

The Gemini handler converts ``contents[]`` to OpenAI-shaped messages, appends
the memory block to the latest user message, then rebuilds ``contents[]``.
Entries carrying a non-text part (``functionResponse``, ``inlineData``, ...)
are restored verbatim by ``_rebuild_gemini_contents`` so compression never
rewrites them. When the latest user entry was one of those, the rebuild also
discarded the memory text, while ``injected_live_zone_tail_gemini`` had already
been logged. The memory block must instead land on the preserved entry's text.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

import pytest

from headroom.proxy.handlers.gemini import GeminiHandlerMixin
from headroom.proxy.memory_handler import MemoryHandler

MEMORY_BLOCK = "<memory>user prefers metric units</memory>"


class _FakeRequest:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {"x-headroom-user-id": "alice"}
        self.query_params: dict[str, str] = {}
        self.url = SimpleNamespace(path="/v1beta/models/gemini-pro:generateContent", query="")
        self.scope: dict = {"type": "http", "method": "POST"}
        self.client = SimpleNamespace(host="127.0.0.1", port=50000)


class _UpstreamResponse:
    status_code = 200
    content = json.dumps(
        {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1},
        }
    ).encode()
    headers = {"content-type": "application/json"}

    def json(self) -> object:
        return json.loads(self.content)


class _MemoryHandler:
    """Memory handler double: real tail-append, fixed search result."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(inject_context=True, project_root_override="")

    async def search_and_format_context(self, *args: Any, **kwargs: Any) -> str:
        return MEMORY_BLOCK

    _append_to_latest_user_tail = staticmethod(MemoryHandler._append_to_latest_user_tail)


class _NoCcr:
    config = SimpleNamespace(enabled=False)

    def has_ccr_tool_calls(self, resp_json: Any, provider: str) -> bool:
        return False


class _Handler(GeminiHandlerMixin):
    GEMINI_API_URL = "https://gemini.example"

    def __init__(self) -> None:
        from headroom.cache.prefix_tracker import SessionTrackerStore

        self.memory_handler = _MemoryHandler()
        self.rate_limiter = None
        self.usage_reporter = None
        self.session_tracker_store = SessionTrackerStore()
        self.config = SimpleNamespace(
            optimize=False,
            anthropic_pre_upstream_memory_context_timeout_seconds=1.0,
            ccr_inject_tool=False,
        )
        self.metrics = SimpleNamespace()
        self.ccr_response_handler = _NoCcr()
        self.outcomes: list = []
        self.forwarded: list[dict] = []

    async def _next_request_id(self) -> str:
        return "req-mem-1"

    async def _retry_request(self, method, url, headers, body):  # noqa: ANN001, ANN201
        self.forwarded.append(copy.deepcopy(body))
        return _UpstreamResponse()

    async def _record_request_outcome(self, outcome) -> None:  # noqa: ANN001
        self.outcomes.append(outcome)

    async def _count_tokens_offloaded(self, model, messages):  # noqa: ANN001, ANN201
        return SimpleNamespace(), 10


async def _send(monkeypatch: pytest.MonkeyPatch, body: dict) -> dict:
    async def payload(request):  # noqa: ANN001, ANN202
        return body

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", payload)
    monkeypatch.setenv("HEADROOM_MEMORY_INJECTION_MODE", "live_zone_tail")

    handler = _Handler()
    response = await handler.handle_gemini_generate_content(_FakeRequest(), "gemini-pro")
    assert response.status_code == 200
    (forwarded,) = handler.forwarded
    return forwarded


def _texts(entry: dict) -> list[str]:
    return [p["text"] for p in entry["parts"] if isinstance(p.get("text"), str)]


@pytest.mark.asyncio
async def test_memory_reaches_latest_user_entry_with_function_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function_response = {"functionResponse": {"name": "get_weather", "response": {"temp_c": 21}}}
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": "What is the weather?"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {}}}]},
            {"role": "user", "parts": [function_response, {"text": "Answer in one line."}]},
        ]
    }
    original = copy.deepcopy(body)
    latest_entry = body["contents"][-1]

    forwarded = await _send(monkeypatch, body)

    latest = forwarded["contents"][-1]
    # The non-text part is still forwarded untouched...
    assert latest["parts"][0] == function_response
    # ...and the memory block is on that entry's text, not dropped.
    assert _texts(latest) == ["Answer in one line.\n\n" + MEMORY_BLOCK]
    # Older turns are not touched.
    assert forwarded["contents"][:2] == original["contents"][:2]
    # The parsed request entry is copied, not mutated in place.
    assert latest_entry == original["contents"][-1]


@pytest.mark.asyncio
async def test_memory_reaches_latest_user_entry_with_inline_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": "Hi"}]},
            {"role": "model", "parts": [{"text": "Hello"}]},
            {"role": "user", "parts": [{"text": "Describe this"}, image, {"text": "briefly"}]},
        ]
    }

    forwarded = await _send(monkeypatch, body)

    entry = forwarded["contents"][-1]
    assert entry["parts"][1] == image
    # The block is appended after the last text part, matching the bytes the
    # text-only path produces for the joined message text.
    assert _texts(entry) == ["Describe this", "briefly\n\n" + MEMORY_BLOCK]


@pytest.mark.asyncio
async def test_text_only_latest_entry_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [{"text": "hi"}]},
            {"role": "user", "parts": [{"text": "what did I say?"}]},
        ]
    }

    forwarded = await _send(monkeypatch, body)

    assert forwarded["contents"][-1] == {
        "role": "user",
        "parts": [{"text": "what did I say?\n\n" + MEMORY_BLOCK}],
    }
    assert forwarded["contents"][0] == {"role": "user", "parts": [{"text": "hello"}]}


@pytest.mark.asyncio
async def test_older_preserved_entry_is_not_rewritten_or_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The latest user entry has no text, so the tail append targets an older
    user entry. That entry is preserved and already in the provider's prompt
    cache: skip the injection instead of rewriting it or logging a drop."""
    logged: list[dict] = []
    monkeypatch.setattr(
        "headroom.proxy.helpers.log_memory_injection", lambda **kw: logged.append(kw)
    )
    image = {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": "Hi"}]},
            {"role": "model", "parts": [{"text": "Hello"}]},
            {"role": "user", "parts": [image, {"text": "What city is this?"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "lookup", "args": {}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "lookup", "response": {}}}]},
        ]
    }
    original = copy.deepcopy(body)

    forwarded = await _send(monkeypatch, body)

    assert forwarded["contents"] == original["contents"]
    assert logged == []


def test_carry_skips_when_target_entry_is_not_preserved() -> None:
    contents = [{"role": "user", "parts": [{"text": "a"}]}]
    preserved: dict[int, dict] = {}
    before = [{"role": "user", "content": "a"}]
    after = [{"role": "user", "content": "a\n\nmem"}]

    assert GeminiHandlerMixin._carry_tail_append_into_preserved(
        contents, set(), preserved, before, after
    )
    assert preserved == {}


_IMAGE_ENTRY = {
    "role": "user",
    "parts": [{"inlineData": {"mimeType": "image/png", "data": "AA=="}}, {"text": "a"}],
}
_BEFORE = [{"role": "user", "content": "a"}]
_SYSTEM = {"role": "system", "content": "s"}


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(_BEFORE, [*_BEFORE, {"role": "user", "content": "b"}], id="length-changed"),
        pytest.param(_BEFORE, list(_BEFORE), id="nothing-changed"),
        pytest.param(
            _BEFORE,
            [{"role": "user", "content": [{"type": "text", "text": "a\n\nmem"}]}],
            id="non-string-content",
        ),
        pytest.param(_BEFORE, [{"role": "user", "content": "rewritten"}], id="not-an-append"),
        pytest.param(
            [_SYSTEM, {"role": "user", "content": "x"}],
            [_SYSTEM, {"role": "user", "content": "x\n\nmem"}],
            id="no-matching-entry",
        ),
    ],
)
def test_carry_leaves_preserved_entry_alone_for_unrecognized_shapes(
    before: list[dict], after: list[dict]
) -> None:
    """Anything other than one string message gaining a suffix is not a tail
    append this helper understands: keep today's behavior untouched."""
    # "no-matching-entry" maps to a second text entry that does not exist.
    contents = [_IMAGE_ENTRY] if len(before) == 1 else [{"role": "user", "parts": []}]
    preserved = {0: _IMAGE_ENTRY}

    assert GeminiHandlerMixin._carry_tail_append_into_preserved(
        contents, {0}, preserved, before, after
    )
    assert preserved == {0: _IMAGE_ENTRY}
