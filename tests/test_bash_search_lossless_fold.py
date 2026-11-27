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
import tracemalloc

import pytest

from headroom.providers import OpenAIProvider
from headroom.tokenizer import Tokenizer
from headroom.transforms.content_router import (
    ContentRouter,
    ContentRouterConfig,
    _bash_command_is_search,
    _bash_command_lists_files,
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
    _ExpansionLimitExceeded,
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


# --- Windows separators: ripgrep and Claude Code's Grep print "\" in paths ---
_DEEP_ROWS = "".join(
    f"crates/headroom-core/src/{folder}/{name}.rs:{line}:use std::collections::HashMap;\n"
    for folder, names in (
        ("transforms", ("anchor", "content", "diff", "log", "search", "smart")),
        ("ccr", ("store", "marker", "resolve", "batch")),
        ("signals", ("keyword", "tokens", "budget", "error")),
    )
    for line, name in enumerate(names, start=3)
)


def _windows(rows: str) -> str:
    """The same grep rows as a Windows tool prints them: backslashes in each path."""
    out = []
    for row in rows.splitlines(keepends=True):
        path, rest = row.split(":", 1)
        out.append(path.replace("/", chr(92)) + ":" + rest)
    return "".join(out)


def test_search_dir_fold_handles_windows_backslash_paths() -> None:
    windows = _windows(_DEEP_ROWS)
    folded = compact_lossless(windows, "search")
    assert len(folded) == len(compact_lossless(_DEEP_ROWS, "search")) < len(windows)
    assert folded.split("\n")[0] == chr(92).join(
        ["crates", "headroom-core", "src", "transforms", ""]
    )
    assert search_dir_unheading(folded, backslash=True) == windows


def test_windows_rows_whose_content_ends_in_a_backslash_round_trip() -> None:
    # A shell line continuation in the matched text must not read as a header.
    posix = "".join(
        f"docs/guides/page_{i}.md:{i + 1}:headroom proxy --port 8787 {chr(92)}\n" for i in range(12)
    )
    windows = _windows(posix)
    folded = compact_lossless(windows, "search")
    assert len(folded) < len(windows)
    assert search_dir_unheading(folded, backslash=True) == windows


def test_slash_fold_keeps_a_line_ending_in_a_backslash_as_text() -> None:
    # Without backslash=True a "\" never ends a header, so a text line ending in
    # one stays text even when a grep row with no folder follows it.
    text = f"src/app/a.py:1:x\nsrc/app/b.py:2:y\nrun it with {chr(92)}\nREADME.md:3:z\n"
    assert search_dir_unheading(search_dir_heading(text)) == text


def test_mixed_separators_keep_the_slash_fold_when_it_is_smaller() -> None:
    # `rg pat headroom/cli headroom/proxy` on Windows prints headroom/cli\x.py.
    # With the two folders interleaved, factoring headroom/ beats a header per
    # folder, so the "/" fold must stay available alongside the backslash one.
    mixed = "".join(
        f"headroom/{folder}{chr(92)}mod_{i}.py:{i + 1}:import json\n"
        for i in range(12)
        for folder in ("cli", "proxy")
    )
    assert compact_lossless(mixed, "search") == search_dir_heading(mixed)


@pytest.mark.parametrize(
    ("name", "tool_input", "label"),
    [
        ("Grep", {"pattern": "HashMap", "path": "crates"}, "router:excluded:lossless_search"),
        ("bash", {"command": "rg -n HashMap crates"}, "router:bash:lossless_search"),
    ],
)
def test_windows_search_output_takes_the_lossless_dir_fold(name, tool_input, label, tokenizer):
    windows = _windows(_DEEP_ROWS)
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": name, "input": tool_input}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": windows}],
        },
    ]
    result = ContentRouter(ContentRouterConfig()).apply(
        messages, tokenizer, compress_user_messages=True
    )
    out = result.messages[1]["content"][0]["content"]
    assert label in result.transforms_applied
    assert search_dir_unheading(out, backslash=True) == windows


# --- Windows separators in path listings: rg --files, rg -l, PowerShell ---
_DEEP_FILES = "".join(
    f"crates/headroom-core/src/{folder}/{name}.rs\n"
    for folder, names in (
        ("transforms", ("anchor", "content", "diff", "log", "search", "smart")),
        ("ccr", ("store", "marker", "resolve", "batch")),
        ("signals", ("keyword", "tokens", "budget", "error")),
    )
    for name in names
)


@pytest.mark.parametrize("root", ["", "./"])  # rg --files / PowerShell Resolve-Path -Relative
def test_path_fold_handles_windows_backslash_paths(root) -> None:
    posix = "".join(root + row for row in _DEEP_FILES.splitlines(keepends=True))
    windows = posix.replace("/", chr(92))
    folded = compact_lossless(windows, "paths", listing=True)
    assert len(folded) == len(compact_lossless(posix, "paths")) < len(windows)
    first_header = f"{root}crates/headroom-core/src/transforms/".replace("/", chr(92))
    assert folded.split("\n")[0] == first_header
    assert _puh(folded, backslash=True) == windows


def test_mixed_separator_listing_takes_the_smaller_fold() -> None:
    # `rg --files headroom/cli headroom/proxy` on Windows prints headroom/cli\x.py.
    grouped = "".join(
        f"headroom/{folder}{chr(92)}mod_{i}.py\n" for folder in ("cli", "proxy") for i in range(12)
    )
    assert compact_lossless(grouped, "paths", listing=True) == _ph(grouped, backslash=True)
    assert compact_lossless(grouped, "paths") == _ph(grouped)
    # Interleaved folders: factoring headroom/ beats a header per row, so the
    # "/" fold has to stay a candidate, and its bases may hold a "\".
    interleaved = "".join(
        f"headroom/{folder}{chr(92)}mod_{i}.py\n" for i in range(12) for folder in ("cli", "proxy")
    )
    assert compact_lossless(interleaved, "paths", listing=True) == _ph(interleaved)


_REGEX_LISTS = {
    "anchored": ["^invoice@A@d$", "^invoice@A@w$"],
    "star": ["^invoice@A@d*$", "^invoice@A@w*$"],
    "quantified": ["invoice@A@d{2}@A@tvalue", "invoice@A@d{2}@A@tstatus"],
    "bracket": ["invoice@A@d[0-9].pdf", "invoice@A@d[0-9].csv"],
    "tab-and-extension": ["foo@A@tday.txt", "foo@A@tnight.txt"],
    "escaped-dots": ["ads@A@.example@A@.com", "ads@A@.example@A@.net"],
    "looks-like-a-listing": ["src@A@services@A@service_1.py", "src@A@services@A@service_2.py"],
}


@pytest.mark.parametrize("rows", _REGEX_LISTS.values(), ids=_REGEX_LISTS)
def test_backslash_rows_stay_whole_unless_known_to_be_a_listing(rows) -> None:
    text = "".join(row.replace("@A@", chr(92)) + "\n" for row in rows)
    assert compact_lossless(text, "paths") == text


@pytest.mark.parametrize(
    "text",
    [
        # A lone leading "\" is not a path root.
        f"{chr(92)}home{chr(92)}a.py\n{chr(92)}home{chr(92)}b.py\n",
        # The "/" fold's markup rule holds with "\" as a separator too.
        f"{{{{/if}}}}{chr(92)}x.py\n{{{{/if}}}}{chr(92)}y.py\n",
        f"app{chr(92)}a.py\nnot a path\napp{chr(92)}b.py\n",
    ],
    ids=["lone-root", "template-closers", "not-pure"],
)
def test_backslash_path_fold_skips_rows_that_are_not_paths(text) -> None:
    assert _ph(text, backslash=True) == text


@pytest.mark.parametrize("sep", ["/", chr(92)], ids=["slash", "backslash"])
def test_paths_fold_does_not_decode_text_that_is_not_a_listing(sep) -> None:
    # A long line ending in a separator is not a header here. Decoding the text
    # anyway would copy it onto every following row: 5000 * 500 characters.
    text = "h" * 5000 + sep + "\n" + "".join(f"f{i}.py\n" for i in range(500))
    tracemalloc.start()
    try:
        assert compact_lossless(text, "paths", listing=True) == text
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 20 * len(text)


@pytest.mark.parametrize("sep", ["/", chr(92)], ids=["slash", "backslash"])
@pytest.mark.parametrize("end", ["", "\n"], ids=["no-newline", "newline"])
def test_path_unheading_stops_at_its_limit(sep, end) -> None:
    # The blank line ends the listing, so "note" is passed through.
    text = "\n".join(["src" + sep, "a.py", "b.py", "c.py", "", "note"]) + end
    decoded = "\n".join([f"src{sep}{name}.py" for name in "abc"] + ["", "note"]) + end
    assert _puh(text, len(decoded), backslash=True) == decoded
    with pytest.raises(_ExpansionLimitExceeded):
        _puh(text, len(decoded) - 1, backslash=True)


def test_paths_fold_bounds_its_verification(monkeypatch) -> None:
    # A candidate smaller than the content that would decode to 2000 * 1000 characters.
    content = "x" * 8000
    bomb = "h" * 2000 + "/\n" + "f\n" * 1000
    monkeypatch.setattr(
        "headroom.transforms.lossless_compaction.path_heading", lambda text, **_: bomb
    )
    tracemalloc.start()
    try:
        assert compact_lossless(content, "paths") == content
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 20 * len(content)


_LISTING_COMMANDS = [
    "rg --files",
    "rg --files src tests",
    "rg -l HashMap crates",
    "rg -il 'import re' headroom",
    "rg -g '*.py' -l x",
    "rg -tpy -l x",
    "rg -e pat -l src",
    "rg --files-with-matches x",
    "rg --files-without-match x",
    "rg.exe --files",
    ".\\rg.exe --files tests",
    "C:\\tools\\rg.exe -l x",
    "rg --regexp pat -l src",
    "rg --glob=*.py -l x",
    "rg 'a # b' -l src",  # a quoted # starts no comment
    "rg C# -l src",  # nor does one inside a word
    "rg --files  # every file",
    "rg --files .\\",  # PowerShell: the trailing backslash is part of the path
    "rg --files .\\headroom\\ | sort",
    "rg --files 'C:\\src\\'",  # single quotes: "\" is a character in both shells
    "grep -rl 'foo\\|bar' .",
    'grep -rl "foo\\|bar" .',
    "grep -rl foo .",
    "grep -rL foo .",
    "grep -r --files-with-matches foo .",
    "fd",
    "fd -e py",
    "fdfind -H rs crates",
    "cd repo && rg --files",
    "sudo rg --files",
    "rg --files 2>/dev/null",
    "rg --files | sort",
    "rg -l 'a|b' src | sort | uniq | head -n 40",
    "rg --files | tail -20",
    'bash -lc "rg --files | head -20"',
    "bash -c 'rg --files' | sort",
    "pwsh -Command 'rg --files'",
    "powershell.exe -c 'Get-ChildItem -Recurse -Name'",
    "Get-ChildItem -Recurse -File -Name headroom",
    "gci -r -name",
    "Get-ChildItem -Recurse -File headroom | Resolve-Path -Relative",
    "Get-ChildItem -Recurse -Name | Sort-Object | Select-Object -First 50",
    "Resolve-Path -Relative .\\headroom\\proxy\\*.py",
    "Resolve-Path -Relative .\\src\\*.rs | Sort-Object",
]
_OTHER_COMMANDS = [
    "",
    "cat rules.txt",
    "type rules.txt",
    "rg -n foo src",
    "rg -INtlua foo",  # -t takes "lua": the l is not -l
    "rg -e -l src",  # -e takes "-l": the pattern
    "rg --no-filename --regexp -legacy rules.txt",  # so does --regexp
    "rg --replace -l foo rules.txt",
    "rg --context-separator -l -C1 foo rules.txt",
    "grep --regexp -l rules.txt",
    "grep --label -l foo",
    "rg --no-filename legacy rules.txt # -l",  # a comment is not an argument
    "rg -n foo src  # --files",
    "rg 'a -l b' rules.txt",  # a quoted pattern is one argument
    "rg -l 'unclosed",
    "rg -A l foo",
    "rg -- --files rules.txt",
    "rg -L foo src",  # follow symlinks, not a listing
    "rg --no-filename foo rules",
    "grep -h foo rules.txt",
    "grep -e -l rules.txt",
    "grep -f l rules.txt",
    "fd -x cat",
    "fd --exec-batch cat",
    "fd -HX cat",
    "rg --files | xargs cat",
    "rg --files | xargs -n1 head -5",
    "rg --files | head rules.txt",
    "rg --files | sed 's/a/b/'",
    "rg --files | wc -l",
    "rg --files | cut -c1-20",
    "bash -c 'rg --files' | xargs cat",
    "rg --files | grep -o x",
    "rg --files; cat rules.txt",
    "rg --files && cat rules.txt",
    "rg --files || cat rules.txt",
    "cat rules.txt | sort",
    "cat files.txt | xargs rg -l x",
    'bash -lc "cat rules.txt"',
    "bash list_files.sh",
    "Get-ChildItem -Recurse",
    "Get-ChildItem | Resolve-Path",
    "Get-Content rules.txt | Resolve-Path -Relative",
    "Resolve-Path .\\src\\*.rs",  # absolute paths in a table
    "Resolve-Path -Relative x | Get-Content",
    "git show HEAD:rules.txt",
    "rg --files\ncat rules.txt",
    # Bash reads "\" as an escape, PowerShell as a character; where that moves a
    # pipe or line break the shell decides, and it is not known.
    "rg --no-filename --regexp \\\n  -legacy rules.txt",  # Bash: -legacy is the pattern
    "rg \\\n  --files crates",  # PowerShell: two commands
    "rg --files .\\\nGet-Content rules.txt",
    'rg --files ".\\crates\\" | ForEach-Object { Get-Content -LiteralPath $_ }',
    "rg --files .\\| xargs cat",
    "rg --files .\\; Get-Content rules.txt",
    "rg --files .\\&& Get-Content rules.txt",
    # The same inside bash -c: the script keeps its line breaks.
    "bash -c 'rg --no-filename --regexp \\\n  -legacy rules.txt'",
    "bash -c 'rg --files\ncat rules.txt'",
    'pwsh -Command "rg --files\nGet-Content rules.txt"',
]


@pytest.mark.parametrize("command", _LISTING_COMMANDS)
def test_detects_commands_that_list_files(command) -> None:
    assert _bash_command_lists_files(command)


@pytest.mark.parametrize("command", _OTHER_COMMANDS)
def test_ignores_commands_that_do_not_list_files(command) -> None:
    assert not _bash_command_lists_files(command)


@pytest.mark.parametrize("send", [_openai, _anthropic], ids=["openai", "anthropic"])
@pytest.mark.parametrize(
    "command",
    [
        "rg --files crates",
        "rg -l HashMap crates",
        "fd -e rs | sort",
        ".\\rg.exe --files crates",
        "Resolve-Path -Relative .\\crates\\*\\src\\*\\*.rs",
        "rg --files .\\crates\\",
    ],
)
def test_windows_listing_takes_the_lossless_paths_fold(command, send, tokenizer) -> None:
    posix = "".join(
        f"crates/headroom-core/src/{folder}/module_{i:02}.rs\n"
        for folder in ("transforms", "ccr", "signals")
        for i in range(20)
    )
    windows = posix.replace("/", chr(92))
    out, transforms = send(command, windows, tokenizer)
    assert "router:bash:lossless_paths" in transforms
    assert out == _ph(windows, backslash=True)
    assert _puh(out, backslash=True) == windows


@pytest.mark.parametrize("send", [_openai, _anthropic], ids=["openai", "anthropic"])
@pytest.mark.parametrize(
    "command",
    [
        "cat rules.txt",
        "rg --no-filename foo rules",
        "rg --files | xargs cat",
        "rg --no-filename --regexp -legacy rules",
        "rg --no-filename foo rules # -l",
        "rg --no-filename --regexp \\\n  -legacy rules",
        'rg --files ".\\rules\\" | ForEach-Object { Get-Content -LiteralPath $_ }',
        "bash -c 'rg --no-filename --regexp \\\n  -legacy rules'",
        "bash -c 'rg --files\ncat rules.txt'",
    ],
)
def test_router_keeps_regex_rules_whole(command, send, tokenizer) -> None:
    # Every rule is also a valid Windows path; only the command tells them apart.
    rules = "".join(f"foo{chr(92)}tday_{i:03}.txt\n" for i in range(120))
    out, transforms = send(command, rules, tokenizer)
    assert "router:bash:lossless_paths" not in transforms
    assert "router:tool_result:lossless_paths" not in transforms
    assert "foo" + chr(92) not in out.split("\n")


def test_windows_listing_that_does_not_fold_takes_the_general_path(tokenizer) -> None:
    # One folder per file: a header per row would only add bytes.
    listing = "".join(f"crate_{i:03}{chr(92)}lib.rs\n" for i in range(120))
    out, transforms = _anthropic("rg --files", listing, tokenizer)
    assert "router:bash:lossless_paths" not in transforms


def test_posix_listing_is_left_to_the_general_path(tokenizer) -> None:
    posix = "".join(f"crates/core/src/module_{i:03}.rs\n" for i in range(120))
    out, transforms = _anthropic("rg --files crates", posix, tokenizer)
    assert "router:bash:lossless_paths" not in transforms
    assert "router:tool_result:lossless_paths" in transforms
    assert _puh(out) == posix
