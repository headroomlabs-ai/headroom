"""Validate the reproduction's usage decoder against independent wire fixtures."""

import unittest

from reproduce_usage import read_usage


class DecoderChecks(unittest.TestCase):
    def test_cumulative_output_is_replaced_not_summed(self):
        raw = b"""event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":11,"output_tokens":0,"cache_read_input_tokens":13,"cache_creation_input_tokens":17}}}

event: message_delta
data: {"type":"message_delta","usage":{"output_tokens":3}}

event: message_delta
data: {"type":"message_delta","usage":{"output_tokens":7}}

event: message_stop
data: {"type":"message_stop"}

"""
        self.assertEqual(
            read_usage(raw, "text/event-stream"),
            {
                "input_tokens": 11,
                "output_tokens": 7,
                "cache_read_input_tokens": 13,
                "cache_creation_input_tokens": 17,
            },
        )
        with self.assertRaises(AssertionError):
            read_usage(raw.split(b"event: message_stop")[0], "text/event-stream")

    def test_json_preserves_separate_cache_buckets(self):
        raw = b'{"type":"message","usage":{"input_tokens":33,"output_tokens":21,"cache_read_input_tokens":39,"cache_creation_input_tokens":51}}'
        self.assertEqual(
            read_usage(raw, "application/json"),
            {
                "input_tokens": 33,
                "output_tokens": 21,
                "cache_read_input_tokens": 39,
                "cache_creation_input_tokens": 51,
            },
        )


if __name__ == "__main__":
    unittest.main()
