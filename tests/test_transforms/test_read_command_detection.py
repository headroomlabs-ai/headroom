"""Tests for `_is_read_command` — which shell commands count as file reads.

Read detection drives read protection: a command classified as a file read has
its output passed verbatim instead of lossy-compressed, because the agent needs
exact bytes to build a patch from. Chained commands used to fall through: only
the first program was inspected, so `wc -l a.py && sed -n '1,60p' a.py` was
classified as a `wc` run and the file content was word-dropped by Kompress.
"""

import json

import pytest

from headroom.tokenizers.estimator import EstimatingTokenCounter
from headroom.transforms.content_router import (
    ContentRouter,
    ContentRouterConfig,
    _is_read_command,
)


@pytest.mark.parametrize(
    "command",
    [
        "cat foo.py",
        "head -80 foo.py",
        "tail -n 50 foo.py",
        "sed -n '1,60p' foo.py",
        "sed -ne '1,60p' foo.py",
        "sed -En '1,60p' foo.py",
        "sed --quiet '1,60p' foo.py",
        "sed --silent '1,60p' foo.py",
        "sed --quiet --expression='1,60p' foo.py",
        "timeout 30 sed -ne '1,60p' foo.py",
        "sed -n --follow-symlinks '1,60p' foo.py",
        "sed -a -n '1,60p' foo.py",  # BSD sed.
        "sed -n -e '1,60p' foo.py -l",  # BSD line buffering (no argument).
        "sed -l -n '1,60p' foo.py",
        "sed -ln '1,60p' foo.py",
        "sed -l 80 -n '1,60p' foo.py",  # GNU line length takes an argument.
        "sed -nl80 '1,60p' foo.py",
        "cd /repo && cat foo.py",
        "sudo cat foo.py",
        "rtk cat foo.py",
        'bash -lc "cat foo.py"',
        # The `-c` argument is not the first token after the shell.
        'bash --norc -lc "cat foo.py"',
    ],
)
def test_simple_reads_are_detected(command: str) -> None:
    assert _is_read_command(command) is True


@pytest.mark.parametrize(
    "command",
    [
        # The reported case: a read batched behind an unrelated first program.
        "wc -l a.py b.py && sed -n '1,60p' c.py",
        "grep -n X foo.py | head -40; head -80 bar.py",
        "ls -la || cat foo.py",
        "echo start; cat foo.py",
        # Bare `sed` is a stream edit, but the `cat` segment is still a read.
        "sed 's/a/b/' foo.py && cat -n bar.py",
        # A read in the first segment survives a non-read tail.
        "cat foo.py && pytest -q",
        # Trailing and doubled separators produce empty segments, which are skipped.
        "cat foo.py;",
        "ls -la; ; cat foo.py",
        # A write in a SIBLING segment does not unprotect the read segment: the
        # output still carries the exact bytes of foo.py.
        "cat foo.py && echo done > marker",
        "echo done > marker && cat foo.py",
        "sed -n '1,60p' foo.py; touch out >> log",
    ],
)
def test_chained_reads_are_detected(command: str) -> None:
    assert _is_read_command(command) is True


@pytest.mark.parametrize(
    "command",
    [
        # Derived output — stays compressible.
        "grep -n x foo.py",
        "pytest -q",
        "ls -la",
        # A pipeline stage consumes the previous stage's output, not a file:
        # `head` here truncates grep output, it does not read bar.py.
        "grep -n x foo.py | head -40",
        # Writes are never reads.
        "cat > foo.py",
        # The write lives in the read segment itself — a pipeline is one segment,
        # so `tee` is still visible after the split.
        "cat foo.py | tee bar.py",
        "cat foo.py > copy.py && echo done",
        "sed -n '1,60p' foo.py >> out.txt",
        "sed 's/a/b/' foo.py",
        "sed -n1,60p foo.py",  # Invalid: a script cannot be attached to -n.
        "sed -ni '1,60p' foo.py",
        "sed -n -i '1,60p' foo.py",
        "sed -n -I .bak '1,60p' foo.py",  # BSD in-place editing.
        "sed -n -I '' '1,60p' foo.py",
        "sed -n -l -i '' '1,60p' foo.py",  # BSD -l must not hide in-place flags.
        "sed -n -l -I .bak '1,60p' foo.py",
        "sed -in '1,60p' foo.py",
        "sed --quiet --in-place '1,60p' foo.py",
        "sed --quiet --in-place=.bak '1,60p' foo.py",
        "sed -e 's/a/ -n /' foo.py",  # Script text is not a quiet option.
        "sed -e ' -n ' foo.py",
        "sed -- -n foo.py",  # Option parsing ends at --.
        # Lockfiles are regenerated, never byte-patched.
        "cat uv.lock",
        # `-n` from a sibling segment must not qualify a bare `sed`.
        "sed 's/a/b/' foo.py && ls -n",
        # A segment that is only an env assignment resolves to no program.
        "FOO=1",
        # A shell invocation with no `-c` argument runs a script, not a read.
        "bash --norc",
        "bash script.sh",
    ],
)
def test_non_reads_are_not_detected(command: str) -> None:
    assert _is_read_command(command) is False


def test_heredoc_body_is_not_split_into_a_bogus_read() -> None:
    """A `;` inside a heredoc body must not produce a segment that looks like a
    read — the command as a whole writes a file."""
    command = "cat > foo.py <<EOF\nprint(1); head -5 bar.py\nEOF"
    assert _is_read_command(command) is False


@pytest.mark.parametrize("flag", ["-n", "-ne", "-En", "--quiet", "--silent"])
@pytest.mark.parametrize("comment", ["", " # avoid -i here", " # --unknown", " # author's note"])
def test_quiet_sed_output_reaches_read_protection(flag: str, comment: str, monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_PROTECT_READS", "1")
    monkeypatch.setenv("HEADROOM_SHELL_RESULT_MIN_CHARS", "2000")
    # Sixty source lines exceed the small-shell shortcut and reach read protection.
    content = "".join(
        f"def compute_{i}(value):\n"
        "    return value * 2  # Preserve this source exactly for a later patch.\n"
        for i in range(30)
    )
    assert len(content) >= 2000
    messages = [
        {"role": "user", "content": "Read the module."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "read-1",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": json.dumps({"command": f"sed {flag} '1,60p' foo.py{comment}"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "read-1", "content": content},
    ]
    router = ContentRouter(
        ContentRouterConfig(
            enable_code_aware=False,
            enable_kompress=False,
            protect_analysis_context=False,
            protect_recent_reads_fraction=0.0,
        )
    )
    result = router.apply(messages, EstimatingTokenCounter(), protect_recent=0)
    assert "router:shell_small_result_verbatim" not in result.transforms_applied
    assert "router:read_protected" in result.transforms_applied
    assert result.messages[2]["content"] == content


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("sed -n '1,60p' foo.py # avoid -i here", True),
        ("sed -n '1,60p' foo.py # --unknown", True),
        ("sed -n '1,60p' foo.py # author's note", True),
        ("sed -n '1,60p' foo.py # unclosed \"quote", True),
        ("sed -n '1,60p' 'foo#bar.py'", True),
        ('sed -n "1,60p" "foo#bar.py"', True),
        ("sed -n '1,60p' foo#bar.py", True),
        (r"sed -n '1,60p' foo\#bar.py", True),
        ("sed -n '/# marker/p' foo.py", True),
        # Literal hashes must not hide real in-place options that follow them.
        ("sed -n '1,60p' foo#bar.py -i", False),
        ("sed -n '1,60p' 'foo#bar.py' -i", False),
        (r"sed -n '1,60p' foo\#bar.py -i", False),
        (r"sed -n '1,60p' foo\ #bar.py -i", False),
        ("sed -n '/# marker/p' foo.py -i", False),
        ("sed -n '1,60p' foo.py -i # --quiet", False),
        ("sed -n 'unterminated # quote", False),
    ],
)
def test_sed_shell_comments_and_literal_hashes(command: str, expected: bool) -> None:
    assert _is_read_command(command) is expected
