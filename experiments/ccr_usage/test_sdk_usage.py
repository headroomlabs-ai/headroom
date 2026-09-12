"""Offline SDK decoding check; no provider calls or credentials are used."""

import unittest

import anthropic
import httpx2 as httpx
from reproduce_usage import sse

from headroom.ccr.response_handler import _combine_anthropic_usage


class SDKUsageChecks(unittest.TestCase):
    def test_unknown_usage_survives_json_and_stream_decoding(self):
        message = _combine_anthropic_usage(
            {},
            {
                "id": "msg_synthetic",
                "type": "message",
                "role": "assistant",
                "model": "synthetic",
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 11, "output_tokens": 7},
            },
        )
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                response = (
                    httpx.Response(
                        200, content=sse(message), headers={"content-type": "text/event-stream"}
                    )
                    if streaming
                    else httpx.Response(200, json=message)
                )
                with anthropic.Anthropic(
                    api_key="synthetic",
                    max_retries=0,
                    http_client=httpx.Client(
                        transport=httpx.MockTransport(lambda _, response=response: response)
                    ),
                ) as client:
                    args = {"model": "synthetic", "max_tokens": 10, "messages": []}
                    if streaming:
                        with client.messages.stream(**args) as stream:
                            result = stream.get_final_message()
                    else:
                        result = client.messages.create(**args)
                self.assertEqual(result.content[0].text, "done")
                for key in message["usage"]:
                    self.assertIsNone(getattr(result.usage, key))


if __name__ == "__main__":
    unittest.main()
