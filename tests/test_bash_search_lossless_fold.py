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
from headroom.transforms.lossless_compaction import expand_runs, search_unheading

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


def _openai(command: str, content: str, tokenizer, config=None):
    router = ContentRouter(config or ContentRouterConfig())
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


def _anthropic(command: str, content: str, tokenizer, config=None):
    router = ContentRouter(config or ContentRouterConfig())
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


GO_LOG = "".join(
    f"2026/09/02 14:{30 + i // 60:02d}:{i % 60:02d} ERROR handler: upstream timeout req={i}\n"
    for i in range(40)
)


@pytest.mark.parametrize("command", ["grep ERROR /var/log/api.log", "cat /var/log/api.log"])
def test_go_log_keeps_its_timestamps(command, tokenizer):
    # Split as path:line:content, `2026/09/02 14:30:00 ...` would fold into a
    # `2026/09/02 14` header over `30:00 ...` rows. `cat` reaches the search
    # fold through the router's lossless-first stage, `grep` through the
    # bash-search fold.
    out, transforms = _anthropic(command, GO_LOG, tokenizer)
    assert "router:bash:lossless_search" not in transforms
    assert "router:tool_result:lossless_search" not in transforms
    assert "2026/09/02 14" not in out.split("\n")


TOMCAT_LOG = "\n".join(
    f"02-Sep-2026 14:{i // 60:02d}:{i % 60:02d}.123 INFO [main] "
    f"org.apache.catalina.core.ContainerBase response resource_{i:04d} handled successfully"
    for i in range(240)
)


# `grep -A1`: 12 groups of 19 matches plus one context row, with an unmatched
# row skipped between groups and grep's `--` separator printed there.
TOMCAT_CONTEXT = "\n--\n".join(
    "\n".join(TOMCAT_LOG.split("\n")[g * 20 : g * 20 + 20]) for g in range(12)
)

JAVA_FRAME = (
    "\tat org.apache.catalina.core.StandardWrapperValve.invoke(StandardWrapperValve.java:199)"
)

# Same `grep -A1` output, but one context row is a stack frame, not a log row.
_context_rows = TOMCAT_CONTEXT.split("\n")
_context_rows[19] = JAVA_FRAME
TOMCAT_FRAME_CONTEXT = "\n".join(_context_rows)


def _openai_text_blocks(command: str, content: str, tokenizer, config=None):
    # Some OpenAI-compatible clients send tool output as a list of text blocks.
    out, transforms = _openai(command, [{"type": "text", "text": content}], tokenizer, config)
    return out[0]["text"], transforms


def _openai_split_text_blocks(command: str, content: str, tokenizer, config=None):
    # Two text blocks, the second starting halfway through the second row.
    cut = content.index("\n") + 1 + 20
    blocks = [{"type": "text", "text": content[:cut]}, {"type": "text", "text": content[cut:]}]
    out, transforms = _openai(command, blocks, tokenizer, config)
    return "".join(block["text"] for block in out), transforms


@pytest.mark.parametrize(
    "send", [_openai, _anthropic, _openai_text_blocks, _openai_split_text_blocks]
)
@pytest.mark.parametrize(
    "command, output",
    [
        pytest.param("grep INFO logs/catalina.out", TOMCAT_LOG, id="matches"),
        pytest.param("grep -A1 matched logs/catalina.out", TOMCAT_CONTEXT, id="context"),
        pytest.param(
            "grep -A1 matched logs/catalina.out", TOMCAT_FRAME_CONTEXT, id="context-stack-frame"
        ),
        pytest.param(
            "grep -A1 matched logs/catalina.out",
            TOMCAT_CONTEXT.replace("\n", "\r\n"),
            id="context-crlf",
        ),
    ],
)
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(lambda: ContentRouterConfig(ccr_inject_marker=False), id="no-ccr"),
        pytest.param(
            lambda: ContentRouterConfig(ccr_enabled=False, ccr_inject_marker=False),
            id="ccr-disabled",
        ),
    ],
)
def test_timestamped_search_stays_whole_without_ccr(send, command, output, config, tokenizer):
    # Timestamp rows are not path:line:content, so the search fold leaves them
    # to the log compressor. Without a retrieval marker the dropped rows could
    # not be fetched back, so the matches must reach the model as they are.
    out, transforms = send(command, output, tokenizer, config())
    assert "router:bash:search_verbatim" in transforms
    assert out == output


@pytest.mark.parametrize("ccr_enabled", [True, False], ids=["no-ccr", "ccr-disabled"])
@pytest.mark.parametrize(
    "tail",
    # An unindented throwable header fails the whole-result check, so the
    # second block must still qualify on its own.
    [JAVA_FRAME, "java.lang.IllegalStateException: unexpected state"],
    ids=["stack-frame", "throwable-header"],
)
def test_text_blocks_with_stack_frame_stay_whole_without_ccr(tail, ccr_enabled, tokenizer):
    # First block: one log row and its stack frame. Second block: the other
    # 240 matches. Neither block may be shrunk on its own.
    first = TOMCAT_LOG.split("\n")[0] + "\n" + tail + "\n"
    blocks = [{"type": "text", "text": first}, {"type": "text", "text": TOMCAT_LOG}]
    config = ContentRouterConfig(ccr_enabled=ccr_enabled, ccr_inject_marker=False)
    out, transforms = _openai("grep INFO logs/catalina.out", blocks, tokenizer, config)
    assert "router:bash:search_verbatim" in transforms
    assert [block["text"] for block in out] == [first, TOMCAT_LOG]


def test_lossless_mode_still_folds_repeated_timestamped_matches(tokenizer):
    # Lossless mode has no markers either, but it never compresses lossily,
    # so its reversible run fold must still get the chance to shrink repeats.
    repeated = "\n".join(
        "02-Sep-2026 14:00:00.123 INFO [main] org.apache.catalina.core.ContainerBase handled"
        for _ in range(100)
    )
    out, transforms = _openai(
        "grep INFO logs/catalina.out", repeated, tokenizer, ContentRouterConfig(lossless=True)
    )
    assert "router:bash:search_verbatim" not in transforms
    assert len(out) < len(repeated)
    assert expand_runs(out) == repeated


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
    c = "src/app/main.rs\nsrc/app/lib.rs\nsrc/app/util.rs\ntests/a.rs\n"
    assert _cl(c, "paths") == "src/app/\nmain.rs\nlib.rs\nutil.rs\ntests/\na.rs\n"
    c = "app/[slug]/page.tsx\napp/[slug]/layout.tsx\n"
    assert _cl(c, "paths") == "app/[slug]/\npage.tsx\nlayout.tsx\n"


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


def test_path_fold_leaves_html_closing_tags_alone():
    # `</body>` is markup, not dir `</` + base `body>`. The fold round-trips,
    # but the model only sees the folded side and reads broken HTML.
    tail = "  </script>\n</body>\n</html>\n"
    assert _cl(tail, "paths") == tail
    assert _cl("</body>\n</html>", "paths") == "</body>\n</html>"


@pytest.mark.parametrize(
    "text",
    [
        "{{#each people}}\n{{#if active}}\n<p>{{name}}</p>\n{{/if}}\n{{/each}}\n",
        "{#each items as item}\n  <li>{item}</li>\n{/each}\n{/if}\n",
        "<pre>\n&lt;/body&gt;\n&lt;/html&gt;\n</pre>\n",
        "Results for 3 hosts:\nN/A\nN/A\nN/A\n",
        # sections made only of closers
        "{{/if}}\n{{/each}}\n",
        "{/if}\n{/each}\n",
        "[/quote]\n[/b]\n",
        "&lt;/body&gt;\n&lt;/html&gt;\n",
    ],
)
def test_path_fold_skips_text_that_only_contains_path_shaped_lines(text):
    # Closers and slashed values among other lines are not a path listing.
    assert _cl(text, "paths") == text


def test_router_keeps_html_closing_tags_in_mixed_content(monkeypatch):
    # A page tail (no DOCTYPE/<html> root, so no HTML override) with a JSON-LD
    # block routes MIXED; every section gets the lossless folds tried on it.
    monkeypatch.delenv("HEADROOM_LOSSLESS_COMPACTION", raising=False)
    ld = json.dumps(
        {
            "@context": "https://schema.org",
            "@type": "Product",
            "name": "Example Widget",
            "offers": {"@type": "Offer", "price": "19.00", "priceCurrency": "USD"},
        },
        indent=2,
    )
    body = "".join(
        f"    <p>Paragraph {i} of the example page with some ordinary text.</p>\n"
        for i in range(20)
    )
    tail = (
        "  <main>\n" + body + "  </main>\n"
        '  <script type="application/ld+json">\n' + ld + "\n  </script>\n</body>\n</html>\n"
    )
    result = ContentRouter(ContentRouterConfig()).compress(tail)
    # Fails loudly if routing changes, so the test cannot silently stop
    # covering the mixed path.
    assert result.strategy_used.value == "mixed"
    assert result.compressed.endswith("  </script>\n</body>\n</html>\n")
    assert "</\n" not in result.compressed


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
