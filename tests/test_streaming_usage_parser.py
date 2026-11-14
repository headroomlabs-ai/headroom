"""Final provider usage must survive progressive and partial backend frames.

The OpenAI-via-backend stream path is fed cumulative usage snapshots: a
provider may repeat `usage` on several content chunks with climbing counts
before its terminal frame. Accounting must record the LAST snapshot, not
the first nonzero one, and a frame that omits a counter must leave that
counter alone rather than resetting it to zero.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from headroom.proxy.handlers.streaming import StreamingMixin


@pytest.mark.parametrize("responses", [False, True], ids=["chat", "responses"])
@pytest.mark.parametrize(
    ("last", "expected"),
    [
        (
            {
                "input_tokens": 13,
                "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 9, "cache_write_tokens": 1},
            },
            (13, 5, 9, 1),
        ),
        ({"output_tokens": 5}, (10, 5, 2, 3)),
        (
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            },
            (0, 0, 0, 0),
        ),
        ({}, (10, 1, 2, 3)),
    ],
    ids=["latest-snapshot", "partial", "explicit-zero", "absent"],
)
def test_backend_stream_records_latest_present_usage(responses, last, expected):
    initial = {
        "input_tokens": 10,
        "output_tokens": 1,
        "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 3},
    }

    def frame(usage):
        if responses:
            data = {"response": {"usage": usage}}
        else:
            names = {
                "input_tokens": "prompt_tokens",
                "output_tokens": "completion_tokens",
                "input_tokens_details": "prompt_tokens_details",
            }
            data = {"usage": {names[key]: value for key, value in usage.items()}}
        return f"data: {json.dumps(data)}\n\n"

    class Backend:
        name = "openai"

        async def stream_openai_message(self, body, headers):
            yield frame(initial)
            yield 'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
            yield frame(last)
            yield "data: [DONE]\n\n"

    class Handler(StreamingMixin):
        config = SimpleNamespace(ccr_inject_tool=False)
        anthropic_backend = Backend()

        async def _record_request_outcome(self, outcome):
            self.outcome = outcome

    handler = Handler()

    async def consume():
        response = await handler._stream_openai_via_backend(
            {"messages": []}, {}, "model", "request", 0.0, 10, 10, 0, [], {}, 0.0
        )
        async for _ in response.body_iterator:
            pass

    asyncio.run(consume())
    outcome = handler.outcome
    assert outcome.status_code == 200
    assert (
        outcome.provider_input_tokens,
        outcome.output_tokens,
        outcome.cache_read_tokens,
        outcome.cache_write_tokens,
    ) == expected
    assert outcome.cache_inferred is False
