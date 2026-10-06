"""Cache mode must survive a client re-serializing the model's last reply.

Claude Code sends the model's Bash call back with the leading ``cd <cwd> &&``
stripped and without ``caller``. Before the fix the whole history counted as
diverged, cache mode fell back to raw forwarding, and every earlier compressed
message was replaced by its original: a full cache re-write (observed 66.8K
tokens per occurrence in benchmarks/tool_search_vs_native).
"""

import os
import stat

import pytest

from headroom.cache.prefix_tracker import extract_cache_stable_delta

USER = {"role": "user", "content": "fix the bug"}
RESULT_ORIGINAL = {
    "role": "user",
    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x" * 5000}],
}
RESULT_COMPRESSED = {
    "role": "user",
    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x…"}],
}
CALL = {
    "role": "assistant",
    "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a"}}],
}

RAW_REPLY = {
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Listing files."},
        {
            "type": "tool_use",
            "id": "t2",
            "name": "Bash",
            "input": {"command": "cd /work && git ls-files"},
            "caller": {"type": "direct"},
        },
    ],
}
CLIENT_REPLY = {
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Listing files."},
        {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "git ls-files"}},
    ],
}
NEXT = {
    "role": "user",
    "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "a.py"}],
}


def test_rewritten_reply_keeps_compressed_prefix():
    previous_original = [USER, CALL, RESULT_ORIGINAL, RAW_REPLY]
    previous_forwarded = [USER, CALL, RESULT_COMPRESSED, RAW_REPLY]
    current = [USER, CALL, RESULT_ORIGINAL, CLIENT_REPLY, NEXT]

    delta = extract_cache_stable_delta(current, previous_original, previous_forwarded)

    assert delta is not None
    prefix, appended = delta
    assert prefix[2] == RESULT_COMPRESSED  # earlier compression replayed, cache intact
    assert prefix[3] == CLIENT_REPLY  # the client's own version of the reply
    assert appended == [NEXT]


def test_different_turn_is_still_a_divergence():
    other = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Listing files."},
            {"type": "tool_use", "id": "t9", "name": "Bash", "input": {"command": "ls"}},
        ],
    }
    previous_original = [USER, CALL, RESULT_ORIGINAL, RAW_REPLY]
    previous_forwarded = [USER, CALL, RESULT_COMPRESSED, RAW_REPLY]
    assert (
        extract_cache_stable_delta(
            [USER, CALL, RESULT_ORIGINAL, other, NEXT], previous_original, previous_forwarded
        )
        is None
    )


def test_earlier_history_edit_is_still_a_divergence():
    edited_user = {"role": "user", "content": "something else"}
    previous_original = [USER, CALL, RESULT_ORIGINAL, RAW_REPLY]
    previous_forwarded = [USER, CALL, RESULT_COMPRESSED, RAW_REPLY]
    assert (
        extract_cache_stable_delta(
            [edited_user, CALL, RESULT_ORIGINAL, CLIENT_REPLY, NEXT],
            previous_original,
            previous_forwarded,
        )
        is None
    )


@pytest.mark.skipif(not hasattr(os, "fchmod"), reason="POSIX file modes only")
def test_debug_dump_is_owner_only_even_if_it_exists(tmp_path, monkeypatch):

    from headroom.proxy.handlers.anthropic import _dump_prefix_mismatch

    monkeypatch.setenv("HEADROOM_DEBUG_PREFIX_MISMATCH", str(tmp_path))
    existing = tmp_path / "req1.json"
    existing.write_text("{}")
    os.chmod(existing, 0o644)  # left world-readable by an earlier run

    _dump_prefix_mismatch("req1", [USER, CLIENT_REPLY], [USER, RAW_REPLY])
    _dump_prefix_mismatch("req2", [USER, CLIENT_REPLY], [USER, RAW_REPLY])

    for name in ("req1.json", "req2.json"):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o600, name


def test_debug_dump_closes_its_descriptor_when_chmod_fails(tmp_path, monkeypatch):
    # A long-running proxy must not leak a descriptor per failed debug dump.
    from headroom.proxy.handlers import anthropic as handler

    monkeypatch.setenv("HEADROOM_DEBUG_PREFIX_MISMATCH", str(tmp_path))
    opened = []
    real_fdopen = os.fdopen

    def tracking_fdopen(fd, *args, **kwargs):
        fh = real_fdopen(fd, *args, **kwargs)
        opened.append(fh)
        return fh

    def failing_fchmod(fd, mode):
        raise PermissionError("not permitted")

    monkeypatch.setattr(handler.os, "fdopen", tracking_fdopen)
    monkeypatch.setattr(handler.os, "fchmod", failing_fchmod, raising=False)

    handler._dump_prefix_mismatch("req1", [USER, CLIENT_REPLY], [USER, RAW_REPLY])

    assert len(opened) == 1
    assert opened[0].closed
