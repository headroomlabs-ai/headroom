"""Tests for Mechanism B: activity-based hold-back Read maturation.

The invariants under test, beyond decision behavior:
1. No cached byte is ever mutated — frozen-prefix content and content
   carrying a client cache_control breakpoint are untouched.
2. Replay is deterministic — once matured, the same marker is applied on
   every subsequent request, byte-identical.
3. Holding is derived from the conversation (file activity), so the
   decision survives state loss; only matured markers are stateful.
"""

from __future__ import annotations

import pytest

from headroom.config import ReadMaturationConfig
from headroom.transforms.read_maturation import (
    ReadMaturationManager,
    relocate_cache_breakpoint,
)

CONTENT = "     1\tdef foo():\n     2\t    return 42\n" * 60  # > 2048B
SMALL = "     1\tok\n"


def anthropic_read(tc_id: str, file_path: str, content: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tc_id, "name": "Read", "input": {"file_path": file_path}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tc_id, "content": content}],
        },
    ]


def anthropic_edit(tc_id: str, file_path: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tc_id,
                    "name": "Edit",
                    "input": {"file_path": file_path, "old_string": "a", "new_string": "b"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tc_id, "content": "ok"}],
        },
    ]


def openai_read(tc_id: str, file_path: str, content: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc_id,
                    "function": {"name": "Read", "arguments": f'{{"file_path": "{file_path}"}}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": tc_id, "content": content},
    ]


def quiet(n: int) -> list[dict]:
    """n assistant turns with no file activity (advances the quiet clock)."""
    return [
        {"role": "assistant", "content": [{"type": "text", "text": f"thinking {i}"}]}
        for i in range(n)
    ]


def base_conv() -> list[dict]:
    return [{"role": "user", "content": "look"}, *anthropic_read("r1", "/x/foo.py", CONTENT)]


def manager(**overrides) -> ReadMaturationManager:
    cfg = ReadMaturationConfig(enabled=True, **overrides)
    return ReadMaturationManager(cfg)


def read_content(res, idx=2):
    return res.messages[idx]["content"][0]["content"]


class TestActivityDecision:
    def test_disabled_is_noop(self):
        m = ReadMaturationManager(ReadMaturationConfig(enabled=False))
        res = m.apply(base_conv())
        assert res.messages == base_conv()
        assert res.holding_msg_indices == []

    def test_fresh_read_holds_verbatim(self):
        res = manager().apply(base_conv())
        assert read_content(res) == CONTENT
        assert res.holding_msg_indices == [2]
        assert res.holding_reads == 1
        assert res.newly_matured == 0

    def test_holds_while_file_quiet_below_quiesce(self):
        msgs = [*base_conv(), *quiet(4)]  # quiet = 4 < 5
        res = manager(quiesce_turns=5).apply(msgs)
        assert res.holding_msg_indices == [2]
        assert read_content(res) == CONTENT

    def test_matures_after_quiesce(self):
        msgs = [*base_conv(), *quiet(5)]  # quiet = 5 >= 5
        res = manager(quiesce_turns=5).apply(msgs)
        assert res.newly_matured == 1
        assert res.holding_msg_indices == []
        marker = read_content(res)
        assert "compressed after use" in marker
        assert "/x/foo.py" in marker
        assert "Retrieve original: hash=" in marker
        assert res.bytes_saved > 0

    def test_file_activity_resets_quiet_clock(self):
        # read, 4 quiet, edit same file, 4 quiet: quiet=4 < 5 → still held
        msgs = [*base_conv(), *quiet(4), *anthropic_edit("e1", "/x/foo.py"), *quiet(4)]
        res = manager(quiesce_turns=5).apply(msgs)
        assert res.holding_msg_indices == [2]
        # one more quiet turn → file quiesced → matures
        res = manager(quiesce_turns=5).apply([*msgs, *quiet(1)])
        assert res.newly_matured == 1

    def test_activity_on_other_file_does_not_reset(self):
        msgs = [*base_conv(), *quiet(3), *anthropic_edit("e1", "/x/OTHER.py"), *quiet(1)]
        # foo.py quiet for 5 assistant turns (3 quiet + edit-turn + 1 quiet)
        res = manager(quiesce_turns=5).apply(msgs)
        assert res.newly_matured == 1

    def test_max_hold_caps_busy_files(self):
        # File touched every turn — never quiesces — but the hold cap fires.
        msgs = base_conv()
        for i in range(6):
            msgs += anthropic_edit(f"e{i}", "/x/foo.py")
        res = manager(quiesce_turns=100, max_hold_turns=6).apply(msgs)
        assert res.newly_matured == 1
        assert res.holding_msg_indices == []

    def test_replay_is_deterministic_and_stateful(self):
        m = manager(quiesce_turns=5)
        matured_msgs = [*base_conv(), *quiet(5)]
        a = read_content(m.apply(matured_msgs))
        # Replay applies even when the conversation grows and the file is
        # touched again later (matured is final).
        later = [*matured_msgs, *anthropic_edit("e1", "/x/foo.py")]
        b = read_content(m.apply(later))
        c = read_content(m.apply([*later, *quiet(3)]))
        assert a == b == c

    def test_small_reads_ignored(self):
        msgs = [
            {"role": "user", "content": "look"},
            *anthropic_read("r1", "/x/a.py", SMALL),
            *quiet(10),
        ]
        res = manager().apply(msgs)
        assert res.holding_msg_indices == []
        assert read_content(res) == SMALL

    def test_frozen_prefix_untouched(self):
        msgs = [*base_conv(), *quiet(10)]
        m = manager(quiesce_turns=5)
        res = m.apply(msgs, frozen_message_count=len(msgs))
        assert res.holding_msg_indices == []
        assert read_content(res) == CONTENT

    def test_respects_lifecycle_markers(self):
        marker = "[Read content stale: /x/foo.py ... Retrieve original: hash=abc123]" + " " * 2048
        msgs = [
            {"role": "user", "content": "look"},
            *anthropic_read("r1", "/x/foo.py", marker),
            *quiet(10),
        ]
        res = manager().apply(msgs)
        assert res.holding_msg_indices == []
        assert res.newly_matured == 0

    def test_client_breakpoint_on_fresh_read_is_held_and_relocated(self):
        """Claude Code parks its tail breakpoint on the newest block —
        right after a Read, that's the Read result itself. The read must
        still be held (verbatim) and relocation must move the breakpoint
        off it, otherwise the verbatim form gets cache-written."""
        msgs = base_conv()
        msgs[2]["content"][0]["cache_control"] = {"type": "ephemeral"}
        res = manager().apply(msgs)
        assert res.holding_msg_indices == [2]
        assert res.messages[2]["content"][0]["content"] == CONTENT

        out = relocate_cache_breakpoint(res.messages, res.holding_msg_indices)
        # Breakpoint stripped from the held read, re-anchored before it.
        assert "cache_control" not in out[2]["content"][0]
        assert out[1]["content"][-1].get("cache_control") == {"type": "ephemeral"}

    def test_openai_format(self):
        msgs = [{"role": "user", "content": "look"}, *openai_read("r1", "/x/foo.py", CONTENT)]
        m = manager(quiesce_turns=5)
        res = m.apply(msgs)
        assert res.holding_msg_indices == [2]
        res = m.apply([*msgs, *quiet(5)])
        assert "compressed after use" in res.messages[2]["content"]

    def test_files_mature_independently(self):
        # foo.py quiet for ages; bar.py just read → foo matures, bar holds.
        msgs = [
            *base_conv(),
            *quiet(6),
            *anthropic_read("r2", "/x/bar.py", CONTENT),
        ]
        res = manager(quiesce_turns=5).apply(msgs)
        assert res.newly_matured == 1  # foo.py
        assert "compressed after use" in read_content(res, 2)
        assert res.holding_msg_indices == [10]  # bar.py result message
        assert read_content(res, 10) == CONTENT

    def test_decision_survives_state_loss(self):
        # A fresh manager (proxy restart) makes the same holding/matured
        # decision because holding is derived from the conversation.
        msgs = [*base_conv(), *quiet(5)]
        first = read_content(manager(quiesce_turns=5).apply(msgs))
        second = read_content(manager(quiesce_turns=5).apply(msgs))
        assert "compressed after use" in first and "compressed after use" in second
        # Markers differ only if CCR hashing differed — it must not.
        assert first == second


class TestCcrIntegration:
    def test_original_stored_and_retrievable(self):
        from headroom.cache.backends.memory import InMemoryBackend
        from headroom.cache.compression_store import CompressionStore

        store = CompressionStore(backend=InMemoryBackend())
        m = ReadMaturationManager(
            ReadMaturationConfig(enabled=True, quiesce_turns=5), compression_store=store
        )
        res = m.apply([*base_conv(), *quiet(5)])

        marker = read_content(res)
        ccr_hash = marker.split("hash=")[1].rstrip("]")
        entry = store.retrieve(ccr_hash)
        assert entry is not None
        assert entry.original_content == CONTENT
        assert entry.compression_strategy == "read_maturation"


class TestProxyWiring:
    def test_proxy_config_flag_default_off(self):
        from headroom.proxy.models import ProxyConfig

        assert ProxyConfig().read_maturation is False
        assert ProxyConfig(read_maturation=True).read_maturation is True

    def test_session_state_rides_on_prefix_tracker(self):
        """The handler's session-state flow: a manager attached to the
        tracker carries matured markers across requests."""
        from headroom.cache.prefix_tracker import PrefixCacheTracker
        from headroom.config import ReadMaturationConfig
        from headroom.transforms.read_maturation import (
            ReadMaturationManager,
            relocate_cache_breakpoint,
        )

        tracker = PrefixCacheTracker("anthropic")
        assert tracker.read_maturation_manager is None

        # Request 1: fresh read — held; breakpoint relocated.
        tracker.read_maturation_manager = ReadMaturationManager(
            ReadMaturationConfig(enabled=True, quiesce_turns=5)
        )
        msgs = base_conv()
        msgs[2]["content"][0] = {
            **msgs[2]["content"][0],
        }
        res = tracker.read_maturation_manager.apply(msgs)
        assert res.holding_msg_indices == [2]
        out = relocate_cache_breakpoint(res.messages, res.holding_msg_indices)
        assert len(out) == len(msgs)

        # Request N (file quiet): same manager matures and replays.
        later = [*base_conv(), *quiet(5)]
        res = tracker.read_maturation_manager.apply(later)
        assert res.newly_matured == 1
        replay = tracker.read_maturation_manager.apply(later)
        assert replay.replacements_applied == 1
        assert replay.newly_matured == 0


class TestBreakpointRelocation:
    def _msgs_with_tail_breakpoint(self) -> list[dict]:
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "earlier turn"}]},
            *anthropic_read("r1", "/x/foo.py", CONTENT),
        ]
        msgs[-1]["content"][-1] = {
            **msgs[-1]["content"][-1],
            "cache_control": {"type": "ephemeral"},
        }
        return msgs

    @staticmethod
    def _breakpoint_indices(msgs: list[dict]) -> list[int]:
        return [
            i
            for i, m in enumerate(msgs)
            if isinstance(m.get("content"), list)
            and any(isinstance(b, dict) and "cache_control" in b for b in m["content"])
        ]

    def test_noop_without_holds(self):
        msgs = self._msgs_with_tail_breakpoint()
        assert relocate_cache_breakpoint(msgs, []) is msgs

    def test_relocates_before_held_read(self):
        msgs = self._msgs_with_tail_breakpoint()
        out = relocate_cache_breakpoint(msgs, [2])

        # Held region [2:] carries no breakpoint; re-anchored on the
        # latest eligible message before it (index 1 — the assistant
        # tool_use message), so everything up to but excluding the held
        # Read still gets cached.
        assert self._breakpoint_indices(out) == [1]
        assert out[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
        assert len(self._breakpoint_indices(out)) <= len(self._breakpoint_indices(msgs))

    def test_noop_when_no_breakpoint_in_held_region(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "x"}]},
            *anthropic_read("r1", "/x/foo.py", CONTENT),
        ]
        out = relocate_cache_breakpoint(msgs, [2])
        assert self._breakpoint_indices(out) == []

    def test_originals_not_mutated(self):
        msgs = self._msgs_with_tail_breakpoint()
        before = [str(m) for m in msgs]
        relocate_cache_breakpoint(msgs, [2])
        assert [str(m) for m in msgs] == before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
