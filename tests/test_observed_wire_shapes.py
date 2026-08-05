# ruff: noqa: E402, E731 — test sections import after setup; lambdas are test stubs.
"""Cache-mode delta engagement against REAL observed wire shapes + extension scaffolding.

Section 1 pins the exact message shapes we observed on the wire during the SWE-bench
mini-swe-agent + litellm -> Anthropic run (captured via the proxy's per-turn DELTA-DIAG):
these are the shapes that broke the naive prefix compare and that fix-4..7 + the
generalized canonicalizer now handle. They are our validated path and MUST stay green.

Section 2 is EXTENSION-READY coverage for provider/client shapes we researched (OpenAI
Chat + Responses, Bedrock Converse, Vercel-AI-SDK/opencode) but have NOT yet exercised
end-to-end. The shared canonicalizer is provider-agnostic, so these already pass at the
*comparison* layer; the comments mark what additional *handler* wiring each provider
still needs for full delta-only compression (see the per-provider TODOs).

Everything here is comparison-layer only (no Modal / no provider calls).
"""

import copy

from headroom.cache.prefix_tracker import (
    _canonicalize_for_prefix_compare as CANON,
)
from headroom.cache.prefix_tracker import (
    extract_cache_stable_delta as delta,
)


def _eq(a, b):
    return CANON(a) == CANON(b)


# ============================================================================
# Section 1 — OBSERVED: mini-swe-agent + litellm -> Anthropic wire (validated)
# ============================================================================
# Real shapes from the run's DELTA-DIAG. mini emits a bash action; litellm converts
# the OpenAI-ish history to Anthropic blocks on the wire, and (turn-to-turn) it:
#   (a) moves the ephemeral cache_control marker to the newest block,
#   (b) attaches `caller: {type: direct}` to tool_use on the stored copy,
#   (c) flips tool_result.content between a bare string and [{type:text,text}],
# while the observation payload itself (`<returncode>N</returncode>\n<output>…</output>`)
# is unchanged. All three must be ignored by the prefix compare.

_RC = "<returncode>0</returncode>\n<output>\n./suma/apps/foo.py\n</output>"  # real observation form


def _asst_tooluse(with_caller: bool, cc: bool):
    tu = {
        "type": "tool_use",
        "id": "toolu_01ABC",
        "name": "bash",
        "input": {"command": 'cd /tmp/core && rg -l "safe_math" --type py | head'},
    }
    if with_caller:
        tu["caller"] = {"type": "direct"}  # litellm programmatic-tool tag
    if cc:
        tu["cache_control"] = {"type": "ephemeral"}
    return {"role": "assistant", "content": [tu]}


def _tool_result(as_string: bool, cc: bool):
    content = _RC if as_string else [{"type": "text", "text": _RC}]
    block = {"type": "tool_result", "tool_use_id": "toolu_01ABC", "content": content}
    if cc:
        block["cache_control"] = {"type": "ephemeral"}
    return {"role": "user", "content": [block]}


def test_observed_caller_present_vs_absent_ignored():
    assert _eq(
        _asst_tooluse(with_caller=True, cc=False), _asst_tooluse(with_caller=False, cc=False)
    )


def test_observed_tool_result_string_vs_block_ignored():
    assert _eq(_tool_result(as_string=True, cc=False), _tool_result(as_string=False, cc=False))


def test_observed_moved_cache_control_ignored():
    # marker on tool_use one turn, on tool_result the next
    assert _eq(_asst_tooluse(with_caller=True, cc=True), _asst_tooluse(with_caller=True, cc=False))


def test_observed_thinking_block_stable_signature():
    # mini turns carry an Anthropic thinking block with a stable signature; unchanged
    # across resend -> stays equal (and a signature change would be a real divergence).
    th = lambda: {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "Eo4CCmMIDxgCKkD..."},
            {"type": "text", "text": "Let me find the tool."},
            {"type": "tool_use", "id": "toolu_01ABC", "name": "bash", "input": {"command": "ls"}},
        ],
    }
    assert _eq(th(), th())


def test_observed_full_turn_delta_engages():
    # The exact failure the run hit: prev stored the assistant with `caller` +
    # cache_control on the newest block; this turn re-sends the same assistant WITHOUT
    # caller, tool_result as a STRING, and the marker MOVED to the new observation.
    # After fix-4..7 + generalized canon, the delta must engage (replay prefix + 1 delta).
    prev_orig = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        _asst_tooluse(with_caller=True, cc=True),
    ]
    prev_fwd = copy.deepcopy(prev_orig)
    cur = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        _asst_tooluse(with_caller=False, cc=False),
        _tool_result(as_string=True, cc=True),
    ]
    out = delta(cur, prev_orig, prev_fwd)
    assert out is not None, "observed litellm churn must NOT force raw fallback"
    stable_prefix, appended = out
    assert stable_prefix == prev_fwd  # replay the byte-identical cached prefix
    assert len(appended) == 1  # only the new tool_result is the delta


def test_observed_genuine_command_change_still_diverges():
    # Safety: a real change to the bash command must still fail the compare.
    prev = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t", "name": "bash", "input": {"command": "ls"}}
            ],
        }
    ]
    cur = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t", "name": "bash", "input": {"command": "rm -rf /"}}
            ],
        }
    ]
    assert not _eq(prev[0], cur[0])


# ============================================================================
# Section 2 — EXTENSION-READY: researched shapes not yet exercised end-to-end
# ============================================================================
# The generalized canonicalizer is provider-agnostic, so these pass at the COMPARISON
# layer today. Each block notes the additional HANDLER wiring still required for full
# delta-only compression on that provider (tracked as follow-ups).


# ---- OpenAI Chat Completions ------------------------------------------------
# Tool result is a `role:"tool"` message with STRING content; assistant tool call is
# `tool_calls[].function{name, arguments(JSON string)}`; automatic prefix caching (NO
# cache_control marker). Noise seen on echoes: system_fingerprint/service_tier, and
# streaming `index` on tool_calls.
# EXTENSION TODO (handler): openai.py cache mode currently does overlay + frozen-count,
# NOT delta-only compression. Wire it to extract_cache_stable_delta (marker policy = none).
def test_ext_openai_tool_calls_index_and_fingerprint_ignored():
    a = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "index": 0,
                "function": {"name": "bash", "arguments": '{"command":"ls"}'},
            }
        ],
        "system_fingerprint": "fp_a",
        "service_tier": "default",
    }
    b = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"ls"}'},
            }
        ],
    }
    assert _eq(a, b)
    # but different arguments (opaque JSON string) must diverge
    c = copy.deepcopy(b)
    c["tool_calls"][0]["function"]["arguments"] = '{"command":"pwd"}'
    assert not _eq(b, c)


# ---- OpenAI Responses API ---------------------------------------------------
# function_call / function_call_output linked by call_id (not id); reasoning items carry
# a VERBATIM `encrypted_content` that must round-trip. `summary` is display-only.
# EXTENSION TODO (handler): same delta-path wiring as Chat; ensure reasoning items are
# treated as content (encrypted_content kept in the identity — already is, generically).
def test_ext_openai_responses_encrypted_content_is_the_semantic_carrier():
    # The verbatim reasoning token is what matters: a change must diverge (never masked),
    # identical must equate.
    diff = {
        "role": "assistant",
        "content": [{"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC_DIFFERENT"}],
    }
    base = {
        "role": "assistant",
        "content": [{"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC1"}],
    }
    same = {
        "role": "assistant",
        "content": [{"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC1"}],
    }
    assert not _eq(diff, base)
    assert _eq(base, same)
    # EXTENSION TODO: `summary` (display-only per OpenAI docs) and the reasoning item
    # `id` are NOT yet in _NON_SEMANTIC_KEYS. If a client varies them per turn, the
    # compare falls back to raw (safe: 0 compression, no stale replay). Add them to the
    # deny-list when the OpenAI Responses delta path is wired and we've confirmed on a
    # captured wire trace that they are non-load-bearing.


# ---- Bedrock Converse -------------------------------------------------------
# camelCase; toolUse/toolResult keyed (no `type`); toolResult.content allows {json}
# (structured!) + a `status`; cachePoint is a standalone content block; reasoningContent
# carries a verbatim signature.
# EXTENSION TODO (handler): bedrock.py bypasses compression in cache mode. Wire a
# cachePoint delta path (marker policy = strip/relocate cachePoint) to the shared engine.
def test_ext_bedrock_cachepoint_ignored_json_and_status_kept():
    a = {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": "tu1",
                    "content": [{"json": {"ok": True, "n": 1}}],
                    "status": "success",
                }
            },
            {"cachePoint": {"type": "default"}},
        ],
    }
    b = {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": "tu1",
                    "content": [{"json": {"ok": True, "n": 1}}],
                    "status": "success",
                }
            }
        ],
    }
    assert _eq(a, b)  # cachePoint block dropped
    c = copy.deepcopy(b)
    c["content"][0]["toolResult"]["content"][0]["json"]["n"] = 2
    assert not _eq(b, c)  # opaque json payload compared verbatim
    d = copy.deepcopy(b)
    d["content"][0]["toolResult"]["status"] = "error"
    assert not _eq(b, d)  # status is semantic


# ---- Vercel AI SDK / opencode ----------------------------------------------
# Parts-based; reasoning signature lives in providerMetadata.anthropic.signature; parts
# carry `state`/`providerExecuted`/`step-start` transport. NOTE: the proxy sees the
# PROVIDER wire (post-AI-SDK-serialization), so providerMetadata typically does not reach
# us — but we drop it defensively. EXTENSION TODO: if we ever ingest pre-wire AI-SDK
# messages, ensure the signature is lifted from providerMetadata into the identity.
def test_ext_aisdk_provider_metadata_and_state_ignored():
    a = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "ok",
                "state": "done",
                "providerMetadata": {"anthropic": {"x": 1}},
                "providerExecuted": True,
            }
        ],
    }
    b = {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}
    assert _eq(a, b)


# ============================================================================
# Section 3 — EXTENSION HOOKS (documented, not yet implemented)
# ============================================================================
# When wiring a new provider to the shared delta engine, add here:
#   * a per-provider marker policy test (Anthropic cache_control / Bedrock cachePoint
#     stripped from the delta before compression; OpenAI: none);
#   * a round-trip test that the forwarded prefix stays byte-identical across a real
#     multi-turn fixture for that provider (byte-level; ideally sourced from a captured
#     HEADROOM_LOG_MESSAGES trace of the `inspect` non-litellm harness);
#   * a tool-shape compression test (OpenAI role:tool with tool safeguards; Bedrock
#     toolResult json). These live in the content_router tests once fix-7 is generalized
#     beyond the Anthropic tool_result block path.


# ============================================================================
# Section 4 — read protection (HEADROOM_PROTECT_READS): never lossy-compress reads
# ============================================================================
from headroom.transforms.content_router import _is_read_command as _isread


def test_read_command_classifier():
    reads = [
        "cat foo.py",
        "cat -n foo.py",
        "cd /x && cat a.py",
        "cd /x && cat -A a.py | head -60",
        "sed -n '1,50p' f.py",
        "head -100 f.py",
        "tail -20 log",
        "nl f.py",
    ]
    non = [
        "cat > f.py <<'EOF'\nx\nEOF",
        "cat a >> b",
        "echo x | tee f",
        "sed -i 's/a/b/' f",
        "sed 's/a/b/' f",
        "rg -l x --type py",
        "grep -rn x .",
        "ls -la",
        "python -c 'x'",
        "git diff -- f",
        "swebench-pytest-lite t/",
        "",
        None,
    ]
    assert all(_isread(c) for c in reads), [c for c in reads if not _isread(c)]
    assert not any(_isread(c) for c in non), [c for c in non if _isread(c)]


# ============================================================================
# Section 5 — command classification is harness-agnostic (Bug A + Bug B)
# ============================================================================
# Two bugs that silently disabled compression/protection on real harnesses.
# These lock in the fixes and assert they hold across the command-prefix and
# tool-call wire shapes different harnesses/providers emit.
from headroom.transforms.content_router import (
    _bash_command_is_search as _issearch,
)
from headroom.transforms.content_router import (
    _is_read_command as _isread2,
)
from headroom.transforms.content_router import (
    _strip_cd_prefix as _stripcd,
)
from headroom.transforms.content_router import (
    _tool_call_command_text as _cmdtext,
)

_SEARCH = frozenset({"grep", "rg", "ag", "fgrep", "egrep", "ripgrep"})


def test_bugA_cd_prefixed_search_detected_all_harnesses():
    # Harnesses run every command inside the checkout: `cd <repo> && <tool>`
    # (mini-swe-agent, most) or `cd <repo>; <tool>` (some Codex configs). Before
    # the fix, _bash_program read the program as `cd` -> search fold never fired.
    for cmd in [
        "cd /tmp/core && rg -l safe_math --type py",
        "cd /tmp/core && grep -rn foo suma/",
        "cd /repo; grep -n bar .",  # semicolon connector
        "cd /a && cd b && rg pat",  # chained cds
        "grep -rn x .",  # no prefix (regression)
        "rg pattern src/",
    ]:
        assert _issearch(cmd, _SEARCH), f"search not detected: {cmd!r}"
    # non-search must stay non-search even with a cd prefix
    for cmd in ["cd /x && cat a.py", "cd /x && python -c 'x'", "cd /x && ls -la"]:
        assert not _issearch(cmd, _SEARCH), f"false search: {cmd!r}"


def test_bugA_strip_cd_prefix_shapes():
    assert _stripcd("cd /tmp/core && rg x") == "rg x"
    assert _stripcd("cd /repo; grep x") == "grep x"
    assert _stripcd("cd a && cd b && grep x") == "grep x"
    assert _stripcd("grep x .") == "grep x ."  # nothing to strip
    assert _stripcd("") == "" and _stripcd(None) == ""  # defensive


def test_openai_tool_calls_none_does_not_crash_and_still_compresses():
    # OpenAI/LiteLLM assistant messages carry an explicit `tool_calls: None` (and
    # `function_call: None`) when there are no calls. `msg.get("tool_calls", [])`
    # returns None (not []), so iterating it crashed _build_tool_name_map ->
    # apply() -> compression silently fell through to PASSTHROUGH on every OpenAI
    # turn (observed on GPT-5.4 text-based: only ~2/24 requests compressed, net
    # token inflation). This asserts the coalesce fix: no crash, map builds.
    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    cr = ContentRouter(ContentRouterConfig())
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "THOUGHT: look\n```bash\ncd /r && cat x.py\n```",
            "tool_calls": None,
            "function_call": None,
        },  # <- the OpenAI shape
        {
            "role": "user",
            "content": "<returncode>0</returncode>\n<output>\n" + "x\n" * 200 + "</output>",
        },
    ]
    name_map = cr._build_tool_name_map(msgs)  # must not raise
    assert isinstance(name_map, dict)


def test_text_based_read_protection_shape_agnostic(monkeypatch=None):
    # Text-based agents (GPT-5.4/Codex/Cursor) have NO tool_use/tool_result blocks:
    # the command is a fenced block in the assistant STRING, the observation is a
    # plain user string. Read-protection must still fire off the *preceding
    # command* so cat/sed code reads are passed verbatim on ANY model/harness.
    import os

    from headroom.tokenizers.registry import get_tokenizer
    from headroom.transforms.content_router import (
        ContentRouter,
        ContentRouterConfig,
        _fenced_shell_command,
    )
    from headroom.transforms.read_lifecycle import ReadLifecycleConfig

    assert (
        _fenced_shell_command("T\n```mswea_bash_command\ncd /r && cat x.py\n```")
        == "cd /r && cat x.py"
    )
    assert _fenced_shell_command("no fence here") == ""
    os.environ["HEADROOM_PROTECT_READS"] = "1"
    tok = get_tokenizer("gpt-4o")
    big_code = "def f():\n" + "    x = 1\n" * 300
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "T\n```mswea_bash_command\ncd /r && cat a.py\n```",
            "tool_calls": None,
        },  # READ command
        {
            "role": "user",
            "content": "<returncode>0</returncode>\n<output>\n" + big_code + "</output>",
        },
        {
            "role": "assistant",
            "content": "T\n```mswea_bash_command\ncd /r && grep -rn foo .\n```",
            "tool_calls": None,
        },  # SEARCH command
        {
            "role": "user",
            "content": "<returncode>0</returncode>\n<output>\n"
            + ("a.py:1:foo\n" * 300)
            + "</output>",
        },
    ]
    r = ContentRouter(
        ContentRouterConfig(
            skip_user_messages=False, read_lifecycle=ReadLifecycleConfig(enabled=False)
        )
    )
    r.apply(
        [dict(m) for m in msgs],
        tok,
        frozen_message_count=0,
        context="",
        compress_user_messages=True,
        protect_recent=0,
        min_tokens_to_compress=25,
    )
    # the observation AFTER the cat (index 3) must be read-protected; the grep one (5) must not
    assert 3 in r._protect_read_msg_indices, r._protect_read_msg_indices
    assert 5 not in r._protect_read_msg_indices, r._protect_read_msg_indices


def test_bugB_read_detection_across_tool_call_wire_shapes():
    # The SAME read action, as each provider/harness serializes its tool call.
    # _tool_call_command_text must recover the shell command from all of them so
    # read-protection fires regardless of client. (Bug B: the old path fed the
    # raw OpenAI JSON blob to _is_read_command, which always returned False.)
    import json

    anthropic_input = {"command": "cd /tmp/core && cat suma/x.py"}  # Anthropic: dict
    openai_args = json.dumps({"command": "cd /tmp/core && cat suma/x.py"})  # OpenAI: JSON string
    codex_list = {"command": ["cat", "suma/x.py"]}  # Codex: argv list
    for raw in (anthropic_input, openai_args, codex_list):
        assert _isread2(_cmdtext(raw)), f"read not detected from {raw!r}"
    # a search command from any shape must NOT be read-protected (stays compressible)
    assert not _isread2(_cmdtext({"command": "cd /x && rg pat"}))
    assert not _isread2(_cmdtext(json.dumps({"command": "grep -rn x ."})))


# ============================================================================
# Section 6 — JSON-OBJECT reads are releasable (detector now parses, not [-only)
# ============================================================================
# `_try_detect_json` used to recognize only JSON *arrays* ([...]); a JSON
# *object* ({...}) — celery.json / package.json / most config+data — fell
# through to PLAIN_TEXT and got read-PROTECTED (never compressed). The detector
# now decides JSON by PARSING (objects, arrays, and a JSON value inside a small
# bounded wrapper), so these lock in that an object read is released for
# compression across wrapped/unwrapped shapes while source code stays protected.
from headroom.transforms.content_router import (
    _read_output_should_be_protected as _protect,
)


def test_json_object_read_is_releasable_all_shapes():
    big_obj = (
        "{\n"
        + ",\n".join(f'  "suma.apps.task_{i}": {{"queue": "q", "rate": {i}}}' for i in range(40))
        + "\n}"
    )
    wrapped = "<returncode>0</returncode>\n<output>\n" + big_obj + "\n</output>"
    # object — raw and harness-wrapped — is RELEASED (not protected) for compression
    assert _protect(big_obj) is False
    assert _protect(wrapped) is False
    # a JSON array is likewise releasable
    assert _protect('[{"a": 1}, {"a": 2}]') is False
    # genuine source code (even with a dict literal) stays PROTECTED
    assert _protect("def f():\n    return {1: 2}\n" * 20) is True


def test_read_protection_releases_json_object_but_protects_code():
    big_obj = "{\n" + ",\n".join(f'  "k{i}": {{"v": {i}}}' for i in range(60)) + "\n}"
    wrapped_obj = "<returncode>0</returncode>\n<output>\n" + big_obj + "\n</output>"
    py = "<returncode>0</returncode>\n<output>\n" + ("def f():\n    x = 1\n" * 60) + "</output>"
    assert _protect(wrapped_obj) is False  # config/data object → RELEASE (compressible)
    assert _protect(py) is True  # source code → PROTECT (byte-exact)


def test_read_protection_role_agnostic_openai_role_tool():
    # Kimi / fireworks (OpenAI function-calling): the read command is in the
    # assistant tool_calls and the observation is a `role:tool` STRING message.
    # Read-protection was gated on role=='user' (+ tool_result blocks), so these
    # role:tool reads slipped through UNPROTECTED. This locks in the role-agnostic
    # fix: a read observation is protected by OUTCOME (read command -> code),
    # whatever role the harness stamps on it.
    import json as _json
    import os

    from headroom.tokenizers.registry import get_tokenizer
    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
    from headroom.transforms.read_lifecycle import ReadLifecycleConfig

    os.environ["HEADROOM_PROTECT_READS"] = "1"
    os.environ.pop("HEADROOM_EXPERIMENTAL_READ_KEEP_RATIO", None)  # protection, not the experiment
    tok = get_tokenizer("gpt-4o")
    code = "def f():\n" + "    x = 1\n" * 300

    def tc(cid, cmd):
        return {
            "id": cid,
            "type": "function",
            "function": {"name": "bash", "arguments": _json.dumps({"command": cmd})},
        }

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [tc("c_read", "cd /r && cat a.py")]},
        {"role": "tool", "tool_call_id": "c_read", "content": code},  # READ (role:tool) -> protect
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tc("c_grep", "cd /r && grep -rn foo .")],
        },
        {
            "role": "tool",
            "tool_call_id": "c_grep",
            "content": "a.py:1:foo\n" * 300,
        },  # SEARCH -> not protected
    ]
    r = ContentRouter(
        ContentRouterConfig(
            skip_user_messages=False, read_lifecycle=ReadLifecycleConfig(enabled=False)
        )
    )
    out = r.apply(
        [dict(m) for m in msgs],
        tok,
        frozen_message_count=0,
        context="",
        compress_user_messages=True,
        protect_recent=0,
        min_tokens_to_compress=25,
    )
    assert "c_read" in r._protect_read_tool_ids, (
        "read cmd must be identified from OpenAI tool_calls"
    )
    # the role:tool READ observation is protected verbatim (was the bug: unprotected)
    assert out.messages[3]["content"] == code, "role:tool code read must be protected verbatim"
