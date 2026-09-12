"""Tests for `_is_read_command` — which shell commands count as file reads.

Read detection drives read protection: a command classified as a file read has
its output passed verbatim instead of lossy-compressed, because the agent needs
exact bytes to build a patch from. Chained commands used to fall through: only
the first program was inspected, so `wc -l a.py && sed -n '1,60p' a.py` was
classified as a `wc` run and the file content was word-dropped by Kompress.
"""

import pytest

from headroom.transforms.content_router import _is_read_command


@pytest.mark.parametrize(
    "command",
    [
        "cat foo.py",
        "head -80 foo.py",
        "tail -n 50 foo.py",
        "sed -n '1,60p' foo.py",
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
