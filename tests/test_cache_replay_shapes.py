"""Freeze eligibility must match the bytes the cache can actually replay."""

from __future__ import annotations

import copy
import unittest

from headroom.cache.compression_cache import CompressionCache
from headroom.proxy.session_engine import (
    FREEZE_POLICY_CONFIRMED_CLAMP,
    FREEZE_POLICY_REPLAYABLE,
    prepare_turn,
)


class TestCacheReplayShapes(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = CompressionCache()
        self.cache.store_compressed(self.cache.content_hash("A\nB"), "short", 1)
        self.blocks = [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]
        self.messages = [
            {"role": "tool", "tool_call_id": "c1", "content": self.blocks},
            {"role": "user", "content": "next"},
        ]

    def test_multiblock_cache_hit_remains_mutable(self) -> None:
        before = copy.deepcopy(self.messages)
        prep = prepare_turn(self.cache, self.messages, policy=FREEZE_POLICY_REPLAYABLE)
        self.assertEqual(prep.frozen_message_count, 0)
        self.assertEqual(prep.pipeline_input, before)
        self.assertEqual(self.messages, before)
        self.assertEqual(self.cache.get_stats()["hits"], 0)
        self.assertNotIn(self.cache.content_hash("A\nB"), self.cache._stable_hashes)

    def test_stable_joined_hash_does_not_prove_shape_replay(self) -> None:
        self.cache.mark_stable(self.cache.content_hash("A\nB"))
        self.assertEqual(self.cache.compute_frozen_count(self.messages), 0)

    def test_prefix_stops_before_unsupported_shape(self) -> None:
        messages = [{"role": "user", "content": "earlier"}, *self.messages]
        self.assertEqual(self.cache.compute_frozen_count(messages), 1)

    def test_confirmed_policy_is_still_clamped_by_replayability(self) -> None:
        prep = prepare_turn(
            self.cache,
            self.messages,
            policy=FREEZE_POLICY_CONFIRMED_CLAMP,
            tracker_frozen=1,
        )
        self.assertEqual(prep.frozen_message_count, 0)

    def test_explicit_caller_pin_still_wins(self) -> None:
        prep = prepare_turn(
            self.cache, self.messages, policy=FREEZE_POLICY_REPLAYABLE, explicit_frozen=1
        )
        self.assertEqual(prep.frozen_message_count, 1)
        self.assertEqual(prep.pipeline_input, self.messages)

    def test_supported_string_and_single_block_replay(self) -> None:
        for content in ["A\nB", [{"type": "text", "text": "A\nB", "custom": [1]}]]:
            with self.subTest(content=content):
                messages = [{"role": "tool", "content": content}, self.messages[-1]]
                prep = prepare_turn(self.cache, messages, policy=FREEZE_POLICY_REPLAYABLE)
                self.assertEqual(prep.frozen_message_count, 1)
                expected = (
                    "short"
                    if isinstance(content, str)
                    else [{"type": "text", "text": "short", "custom": [1]}]
                )
                self.assertEqual(prep.pipeline_input[0]["content"], expected)

    def test_anthropic_replay_contract_is_unchanged(self) -> None:
        content = [{"type": "tool_result", "tool_use_id": "c1", "content": "A\nB"}]
        prep = prepare_turn(
            self.cache,
            [{"role": "user", "content": content}, self.messages[-1]],
            policy=FREEZE_POLICY_REPLAYABLE,
        )
        self.assertEqual(prep.frozen_message_count, 1)
        self.assertEqual(prep.pipeline_input[0]["content"][0]["content"], "short")

    def test_malformed_and_media_shapes_remain_mutable(self) -> None:
        for content in [
            None,
            {},
            [],
            [{"type": "text", "text": 7}],
            [
                {"type": "image_url", "image_url": {"url": "synthetic"}},
                {"type": "text", "text": "A\nB"},
            ],
        ]:
            with self.subTest(content=content):
                messages = [{"role": "tool", "content": content}, self.messages[-1]]
                self.assertEqual(self.cache.compute_frozen_count(messages), 0)

    def test_textless_anthropic_prefix_stops_at_unsupported_openai_shape(self) -> None:
        textless = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t0", "content": []}],
        }
        for blocks in [
            self.blocks,
            [
                {"type": "image_url", "image_url": {"url": "synthetic"}},
                {"type": "text", "text": "A\nB"},
            ],
        ]:
            with self.subTest(blocks=blocks):
                messages = [textless, {"role": "tool", "content": blocks}, self.messages[-1]]
                before = copy.deepcopy(messages)
                stats_before = self.cache.get_stats()
                prep = prepare_turn(self.cache, messages, policy=FREEZE_POLICY_REPLAYABLE)
                self.assertEqual(prep.frozen_message_count, 1)
                self.assertEqual(prep.pipeline_input, before)
                self.assertEqual(messages, before)
                self.assertEqual(self.cache.get_stats()["hits"], 0)
                self.assertEqual(
                    self.cache.get_stats()["tokens_saved"], stats_before["tokens_saved"]
                )

    def test_textless_anthropic_prefix_still_obeys_confirmed_clamp(self) -> None:
        textless = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t0", "content": []}],
        }
        messages = [textless, *self.messages]
        for confirmed in [0, 1, 2]:
            with self.subTest(confirmed=confirmed):
                prep = prepare_turn(
                    self.cache,
                    messages,
                    policy=FREEZE_POLICY_CONFIRMED_CLAMP,
                    tracker_frozen=confirmed,
                )
                self.assertEqual(prep.frozen_message_count, min(confirmed, 1))

    def test_cached_text_after_textless_anthropic_result_remains_mutable(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t0", "content": []},
                    {"type": "tool_result", "tool_use_id": "t1", "content": "A\nB"},
                ],
            },
            self.messages[-1],
        ]
        before = copy.deepcopy(messages)
        prep = prepare_turn(self.cache, messages, policy=FREEZE_POLICY_REPLAYABLE)
        self.assertEqual(prep.frozen_message_count, 0)
        self.assertEqual(prep.pipeline_input, before)
        self.assertEqual(messages, before)
        self.assertEqual(self.cache.get_stats()["hits"], 0)

    def test_supported_stable_hash_and_trailing_live_zone_are_unchanged(self) -> None:
        cache = CompressionCache()
        cache.mark_stable(cache.content_hash("stable"))
        messages = [{"role": "tool", "content": "stable"}, self.messages[-1]]
        self.assertEqual(cache.compute_frozen_count(messages), 1)
        self.assertEqual(cache.compute_frozen_count(messages[:1]), 0)


if __name__ == "__main__":
    unittest.main()
