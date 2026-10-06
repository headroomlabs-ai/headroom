"""Cache mode must survive a client re-serializing the model's last reply.

Claude Code sends the model's Bash call back with the leading ``cd <cwd> &&``
stripped and without ``caller``. Before the fix the whole history counted as
diverged, cache mode fell back to raw forwarding, and every earlier compressed
message was replaced by its original: a full cache re-write (observed 66.8K
tokens per occurrence in benchmarks/tool_search_vs_native).
"""

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
