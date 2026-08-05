"""fix-3: the cache-mode delta path must ignore moved cache_control markers.

Cache mode replays the exact previously-forwarded bytes for history and
compresses ONLY the newly appended delta (compress-once-then-freeze). The gate
is ``AnthropicHandler._extract_cache_stable_delta``: it only engages when the
prior original request is a message-prefix of the current one.

Real clients (litellm, Claude Code) move the ephemeral cache_control breakpoint
to the newest message every turn, so a historical message carries the marker on
one turn and not the next. The original raw-dict prefix compare therefore failed
every turn, dropping cache mode to RAW (uncompressed) forwarding -- byte-stable
(0 busts) but 0% compression. Observed directly on the mini-swe-agent cache-mode
run: avg_compression_pct=0.0 on every instance, orig==opt on every turn.

These tests pin the scenario against the real handler method and prove the
cache_control-agnostic compare lets the delta engage while the replayed prefix
stays byte-identical (so the provider prefix still hits).
"""

import copy

from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

delta = AnthropicHandlerMixin._extract_cache_stable_delta


def B(role, text, cc=False):
    """Anthropic block-style message; cache_control lives on a content block."""
    blk = {"type": "text", "text": text}
    if cc:
        blk["cache_control"] = {"type": "ephemeral"}
    return {"role": role, "content": [blk]}


# Turn t: client marked the (then-newest) msg2. We forwarded it verbatim.
PREV_ORIG = [B("user", "sys+task"), B("assistant", "ok"), B("user", "obs-1", cc=True)]
PREV_FWD = copy.deepcopy(PREV_ORIG)
# Turn t+1: appended act-2 + obs-2 and MOVED the marker off msg2 onto the newest.
CUR = [
    B("user", "sys+task"),
    B("assistant", "ok"),
    B("user", "obs-1"),  # marker gone
    B("assistant", "act-2"),
    B("user", "obs-2", cc=True),  # marker moved here
]


def test_moved_marker_engages_delta_not_raw_fallback():
    out = delta(CUR, PREV_ORIG, PREV_FWD)
    assert out is not None, "moved marker must NOT force raw fallback"
    stable_prefix, appended = out
    # The replayed prefix is byte-identical to what we forwarded (and the
    # provider cached) last turn -> the prefix hits instead of busting.
    assert stable_prefix == PREV_FWD
    # Only the two newly appended messages are handed to compression.
    assert len(appended) == 2
    assert appended[0]["content"][0]["text"] == "act-2"
    assert appended[1]["content"][0]["text"] == "obs-2"


def test_control_marker_not_moved_also_engages():
    # Same append, marker left on the historical msg2: engages either way.
    cur = [
        B("user", "sys+task"),
        B("assistant", "ok"),
        B("user", "obs-1", cc=True),
        B("assistant", "act-2"),
        B("user", "obs-2"),
    ]
    assert delta(cur, PREV_ORIG, PREV_FWD) is not None


def test_real_content_divergence_still_falls_back():
    # Safety preserved: a genuinely different historical message (not just a
    # moved marker) must still bail to raw -- we never replay stale content.
    cur = [
        B("user", "sys+task"),
        B("assistant", "DIFFERENT"),  # content actually changed
        B("user", "obs-1"),
        B("assistant", "act-2"),
    ]
    assert delta(cur, PREV_ORIG, PREV_FWD) is None


def test_cold_start_returns_none():
    assert delta(CUR, None, None) is None
    assert delta(CUR, [], []) is None


def test_shorter_current_returns_none():
    assert delta([B("user", "sys+task")], PREV_ORIG, PREV_FWD) is None


# ── fix-4: tool_result content shape (string <-> [{type:text}]) ────────────────
def _tr(tool_use_id, text, as_string):
    content = text if as_string else [{"type": "text", "text": text}]
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}],
    }


def test_tool_result_string_vs_block_engages():
    # Stored previous_original holds the block-list form; client resends the SAME
    # tool_result as a bare string. These are Anthropic-equivalent -> must engage.
    prev_orig = [
        B("user", "task"),
        _tr("t1", "<returncode>0</returncode>\n<output>\n</output>", as_string=False),
    ]
    prev_fwd = copy.deepcopy(prev_orig)
    cur = [
        B("user", "task"),
        _tr("t1", "<returncode>0</returncode>\n<output>\n</output>", as_string=True),  # string form
        B("assistant", "next-action"),
        _tr("t2", "<output>done</output>", as_string=True),
    ]
    out = delta(cur, prev_orig, prev_fwd)
    assert out is not None, "tool_result string-vs-block must NOT force raw fallback"
    stable_prefix, appended = out
    assert stable_prefix == prev_fwd  # replay the byte-identical cached prefix
    assert len(appended) == 2  # only the new assistant + tool_result are the delta


def test_tool_result_different_text_still_falls_back():
    # Safety: same shape-normalization must NOT hide a genuine content change.
    prev_orig = [B("user", "task"), _tr("t1", "OUTPUT-A", as_string=False)]
    prev_fwd = copy.deepcopy(prev_orig)
    cur = [B("user", "task"), _tr("t1", "OUTPUT-B", as_string=True), B("assistant", "x")]
    assert delta(cur, prev_orig, prev_fwd) is None


def test_tool_use_caller_annotation_ignored():
    # mini-swe-agent/litellm adds a non-semantic `caller` tag to tool_use blocks
    # on the stored copy but not on the re-sent wire message -> must still engage.
    def _asst(with_caller):
        tu = {"type": "tool_use", "id": "tu1", "name": "bash", "input": {"command": "ls"}}
        if with_caller:
            tu["caller"] = {"type": "direct"}
        return {"role": "assistant", "content": [tu]}

    prev_orig = [B("user", "task"), _asst(with_caller=True)]  # stored: has caller
    prev_fwd = copy.deepcopy(prev_orig)
    cur = [
        B("user", "task"),
        _asst(with_caller=False),
        _tr("t1", "out", as_string=True),
    ]  # wire: no caller
    out = delta(cur, prev_orig, prev_fwd)
    assert out is not None, "a client-only `caller` annotation must not force raw fallback"
    assert out[0] == prev_fwd
    assert len(out[1]) == 1


def test_tool_use_different_input_still_falls_back():
    # Safety: a real change to the tool command must still bail.
    def _asst(cmd):
        return {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu1", "name": "bash", "input": {"command": cmd}}
            ],
        }

    prev_orig = [B("user", "task"), _asst("ls")]
    prev_fwd = copy.deepcopy(prev_orig)
    cur = [B("user", "task"), _asst("rm -rf /"), _tr("t1", "out", as_string=True)]
    assert delta(cur, prev_orig, prev_fwd) is None


def test_combined_marker_move_and_tool_result_shape_engages():
    # The real mini-swe-agent situation: marker moved AND tool_result shape differs.
    prev_orig = [
        B("user", "task", cc=True),
        _tr("t1", "obs-1", as_string=False),
    ]
    prev_fwd = copy.deepcopy(prev_orig)
    cur = [
        B("user", "task"),  # marker gone
        _tr("t1", "obs-1", as_string=True),  # shape flipped to string
        B("assistant", "act", cc=True),  # marker moved to newest
    ]
    out = delta(cur, prev_orig, prev_fwd)
    assert out is not None
    assert out[0] == prev_fwd
    assert len(out[1]) == 1
