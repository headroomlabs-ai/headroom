# ruff: noqa: E402 — test sections import after helper/setup code by design.
"""Bash-search lossless fold.

`bash` is not an excluded tool, so its output normally takes the lossy strategy
path. But a read-only search run through it (grep/rg/git grep) produces byte-
losslessly foldable output — the router detects the *command* and folds it with
the same ripgrep --heading transform excluded Grep gets, instead of lossy
compression. Non-search bash commands (cat/build/mutate) are untouched.
"""

from __future__ import annotations

import json

import pytest

from headroom.providers import OpenAIProvider
from headroom.tokenizer import Tokenizer
from headroom.transforms.content_router import (
    ContentRouter,
    ContentRouterConfig,
    _bash_command_is_search,
    _bash_program,
)
from headroom.transforms.lossless_compaction import search_unheading

SEARCH = frozenset({"grep", "egrep", "fgrep", "rg", "ripgrep", "ag", "ack"})
GREP = "".join(
    f"src/module_{f}.py:{ln * 3}:matched occurrence with some real content here\n"
    for f in range(6)
    for ln in range(15)
)
CODE = "def foo(x):\n    return x + 1\n\nclass Bar:\n    value = 42\n" * 30


@pytest.fixture
def tokenizer():
    provider = OpenAIProvider()
    return Tokenizer(provider.get_token_counter("gpt-4o"), "gpt-4o")


# --- command parsing: peel wrappers, detect search programs ---


@pytest.mark.parametrize(
    "command",
    [
        "grep -rn foo .",
        "rtk grep def headroom/transforms",  # the user's token-proxy wrapper
        "rg --heading pattern src/",
        "git grep -n TODO",
        "sudo grep root /etc/passwd",
        "timeout 30 rg foo",  # wrapper takes a numeric arg
        "FOO=1 BAR=2 grep foo",  # env assignments
        "/usr/bin/grep -rn foo .",  # absolute path
        'bash -lc "grep -rn foo ."',  # Codex-style shell -c
        "nice -n 5 ack pattern",  # wrapper with option arg
    ],
)
def test_detects_search_commands(command):
    assert _bash_command_is_search(command, SEARCH) is True


@pytest.mark.parametrize(
    "command",
    [
        "cat headroom/server.py",
        "cargo test",
        "pytest tests/ -x",
        "git diff HEAD~1",  # diff, NOT search
        "echo grep",  # echo, not a real grep
        "python script.py",
        "rm -rf build",
        "ls -la",
    ],
)
def test_ignores_non_search_commands(command):
    assert _bash_command_is_search(command, SEARCH) is False


def test_bash_program_peels_wrappers():
    assert _bash_program("rtk grep foo")[0] == "grep"
    assert _bash_program("timeout 30 rg x")[0] == "rg"
    assert _bash_program("FOO=1 /usr/bin/grep y")[0] == "grep"
    assert _bash_program("")[0] == ""


# --- end-to-end through the router (both wire formats) ---


def _openai(command: str, content: str, tokenizer):
    router = ContentRouter(ContentRouterConfig())
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "bash", "arguments": json.dumps({"command": command})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": content},
    ]
    result = router.apply(messages, tokenizer, compress_user_messages=True)
    return result.messages[1]["content"], result.transforms_applied


def _anthropic(command: str, content: str, tokenizer):
    router = ContentRouter(ContentRouterConfig())
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": command}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": content}],
        },
    ]
    result = router.apply(messages, tokenizer, compress_user_messages=True)
    return result.messages[1]["content"][0]["content"], result.transforms_applied


def test_openai_bash_grep_folds_and_recovers(tokenizer):
    out, transforms = _openai("grep -rn foo .", GREP, tokenizer)
    assert "router:bash:lossless_search" in transforms
    assert len(out) < len(GREP)
    assert search_unheading(out) == GREP  # byte-exact


def test_anthropic_bash_rtk_grep_folds_and_recovers(tokenizer):
    out, transforms = _anthropic("rtk grep foo headroom/", GREP, tokenizer)
    assert "router:bash:lossless_search" in transforms
    assert search_unheading(out) == GREP


def test_non_search_bash_command_not_folded(tokenizer):
    # `cat` is not a search — must NOT take the bash-search fold.
    _out, transforms = _openai("cat headroom/server.py", GREP, tokenizer)
    assert "router:bash:lossless_search" not in transforms


def test_source_output_from_search_command_untouched(tokenizer):
    # Command is a search, but the output isn't path:line:content — the
    # reversibility guard makes compact_lossless return it unchanged.
    out, transforms = _openai("grep -l foo", CODE, tokenizer)
    assert "router:bash:lossless_search" not in transforms
    assert out == CODE


# ---- path-listing fold (find/ls -1/rg -l): fold repeated parent dirs ----
from headroom.transforms.lossless_compaction import (
    compact_lossless as _cl,
)
from headroom.transforms.lossless_compaction import (
    path_heading as _ph,
)
from headroom.transforms.lossless_compaction import (
    path_unheading as _puh,
)


def test_path_fold_roundtrip_and_shrinks_pure_list():
    c = "./suma/apps/ext/core.py\n./suma/apps/ext/dao.py\n./suma/apps/other/x.py"
    folded = _cl(c, "paths")
    assert _puh(_ph(c)) == c  # exact inverse
    assert len(folded) < len(c)  # shrinks
    assert folded != c


def test_path_fold_safe_passthrough_on_non_path_shapes():
    # grep path:line:content is the search fold's job, not paths -> unchanged
    assert _cl("a/b.py:12:def f\na/b.py:15:x", "paths") == "a/b.py:12:def f\na/b.py:15:x"
    # trailing-slash dir entries and single paths -> unchanged
    assert _cl("./a/b/\n./a/c/", "paths") == "./a/b/\n./a/c/"
    assert _cl("./only/one.py", "paths") == "./only/one.py"


def test_path_fold_mixed_content_roundtrips_or_passes_through():
    # a non-path no-slash line among paths must never corrupt: compact_lossless
    # verifies and returns original if the fold isn't exactly reversible.
    c = "./a/b/f.py\n./a/b/g.py\nsome log line\n./a/b/h.py"
    out = _cl(c, "paths")
    assert _puh(_ph(out)) == out or out == c  # never corrupts
    # simplest invariant: decoding whatever we emit reconstructs the input
    assert _puh(_ph(c)) == c or _cl(c, "paths") == c


# ---- EXPERIMENT: HEADROOM_EXPERIMENTAL_READ_KEEP_RATIO (light Kompress on reads) ----
def test_experimental_read_keep_ratio_flag_and_gating(monkeypatch):
    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    # OFF by default -> verbatim (helper returns None, no compression attempted)
    monkeypatch.delenv("HEADROOM_EXPERIMENTAL_READ_KEEP_RATIO", raising=False)
    r_off = ContentRouter(ContentRouterConfig())
    assert r_off._exp_read_keep_ratio == 0.0
    assert r_off._experimental_compress_read("x" * 500) is None

    # ON -> calls Kompress at the ratio; keeps result only if it actually shrank
    monkeypatch.setenv("HEADROOM_EXPERIMENTAL_READ_KEEP_RATIO", "0.9")
    r_on = ContentRouter(ContentRouterConfig())
    assert r_on._exp_read_keep_ratio == 0.9
    seen = {}

    def fake_ml(content, context, question=None, target_ratio=None):
        seen["ratio"] = target_ratio
        return content[: len(content) // 2], 10  # pretend it shrank

    monkeypatch.setattr(r_on, "_try_ml_compressor", fake_ml)
    out = r_on._experimental_compress_read("y" * 500, "ctx")
    assert out is not None and len(out) < 500  # adopted (shrank)
    assert seen["ratio"] == 0.9  # ratio threaded through

    # no-shrink -> None (fall back to verbatim protection)
    monkeypatch.setattr(
        r_on, "_try_ml_compressor", lambda c, ctx, question=None, target_ratio=None: (c, 1)
    )
    assert r_on._experimental_compress_read("z" * 500) is None
    # sub-floor content never attempted
    assert r_on._experimental_compress_read("short") is None


# --- directory-prefix fold: grep -rn across many distinct files ---
from headroom.transforms.lossless_compaction import (  # noqa: E402
    compact_lossless,
    search_dir_heading,
    search_dir_unheading,
)


def test_search_dir_fold_factors_directory_across_distinct_files() -> None:
    # Sorted grep -rn output: same-dir files are consecutive, one match each, so
    # the file-heading fold saves nothing but the shared directory repeats on
    # every row. The dir fold factors it out — byte-losslessly.
    grep = (
        "\n".join(f"headroom/proxy/mod_{i:02d}.py:{i + 1}:    x = compress(p)" for i in range(12))
        + "\n"
    )
    folded = compact_lossless(grep, "search")
    assert len(folded) < len(grep)  # actually shrank (0% before this fold)
    assert "headroom/proxy/" in folded  # directory factored to a header line
    assert search_dir_unheading(folded) == grep  # exact byte round-trip
    assert search_dir_unheading(search_dir_heading(grep)) == grep


def test_search_dir_fold_roundtrips_mixed_and_passthrough() -> None:
    mixed = (
        "src/a/x.py:1:hit one\nsrc/a/y.py:2:hit two\n"
        "== a plain banner ==\n"
        "src/b/z.py:3:content with a colon: value\nnoslash.py:4:pathless row\n"
    )
    out = compact_lossless(mixed, "search")
    assert search_dir_unheading(out) == mixed or search_unheading(out) == mixed or out == mixed


def test_search_file_fold_still_wins_for_many_matches_one_file() -> None:
    # Many matches in ONE file: the file fold is smaller, and compact_lossless
    # keeps whichever candidate round-trips and is smallest.
    grep = "\n".join(f"headroom/proxy/server.py:{i}:    line {i}" for i in range(1, 40)) + "\n"
    out = compact_lossless(grep, "search")
    assert len(out) < len(grep)
    assert search_unheading(out) == grep


# --- rg --heading directory fold: factor a shared dir across FILE headers ---
from headroom.transforms.lossless_compaction import (  # noqa: E402
    search_file_heading_dir,
    search_file_heading_undir,
)


def test_heading_dir_fold_factors_directory_across_rg_headings() -> None:
    # rg --heading output: each file is a bare path header followed by
    # ``line:content`` rows. Files under one directory repeat the dir on every
    # header; the fold factors it to a single ``dir/`` line, losslessly.
    rg = (
        "src/auth/login.py\n12:def login()\n15:    return token\n"
        "src/auth/session.py\n8:def session()\n"
        "src/auth/logout.py\n3:def logout()\n"
    )
    folded = search_file_heading_dir(rg)
    assert len(folded) < len(rg)
    assert folded.startswith("src/auth/\n")  # dir factored to a header line
    assert "src/auth/login.py" not in folded  # basenames left bare beneath it
    assert search_file_heading_undir(folded) == rg  # exact byte round-trip
    # And through the self-verifying dispatcher:
    assert compact_lossless(rg, "search") == folded


def test_heading_dir_fold_survives_blank_group_separators() -> None:
    # ripgrep separates file groups with a blank line; the dir must persist
    # across the blank so all files under one directory still fold.
    rg = "a/b/one.rs\n1:x\n\na/b/two.rs\n2:y\n\na/b/three.rs\n3:z\n"
    folded = search_file_heading_dir(rg)
    assert len(folded) < len(rg)
    assert search_file_heading_undir(folded) == rg


def test_heading_dir_fold_is_lossless_or_passes_through_messy_input() -> None:
    # A mix of dir'd and top-level (no-dir) headers is ambiguous to reverse, so
    # compact_lossless must fall back to a candidate that round-trips exactly
    # (never corrupt). Losslessness is the invariant, not that it always folds.
    messy = "src/a/x.py\n1:hit\ntop_level.py\n2:hit\nsrc/a/y.py\n3:hit\n"
    out = compact_lossless(messy, "search")
    assert len(out) <= len(messy)
    assert (
        search_file_heading_undir(out) == messy
        or search_dir_unheading(out) == messy
        or search_unheading(out) == messy
        or out == messy
    )


def test_grep_row_input_is_untouched_by_heading_dir_fold() -> None:
    # grep path:line:content rows are NOT rg --heading form, so the heading-dir
    # fold leaves them for search_heading / search_dir_heading (no interference).
    grep = "src/a/f1.py:1:x\nsrc/a/f2.py:2:y\n"
    assert search_file_heading_dir(grep) == grep
