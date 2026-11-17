"""Marker-delimited blocks that Headroom manages inside instruction files.

``headroom learn`` and the memory exporters rewrite a section of files the
model reads as instructions on every turn — ``CLAUDE.md``/``CLAUDE.local.md``,
``AGENTS.md``, ``GEMINI.md``, ``GROK.md``, ``.cursor/rules/*.mdc``. The section
is delimited by two HTML-comment markers and rebuilt from content derived from
session transcripts (tool output, error text, user messages, extracted
memories), which an attacker can influence.

Content containing our end marker used to terminate the block early: whatever
followed it landed outside, where the next run's non-greedy match never looked
again, leaving a persistent instruction in a file that is often committed and
team-shared. Any other HTML comment would hide text from the human reading the
file while the model still reads it.

:func:`sanitize_block_text` turns ``<!--`` / ``-->`` into ``&lt;!--`` /
``--&gt;``, so content can neither close our markers nor hide itself. It is
visible by design: a reviewer diffing the file sees that something tried to
open a comment.

:func:`block_pattern` matches from a start marker to the **nearest** end
marker. Because written content can no longer contain our end marker, that is
the whole block, and the match never runs past it into hand-written text that
happens to quote the marker later in the file. A file an earlier version
already split keeps its escaped tail outside the block: the tail no longer
grows, but it has to be removed by hand.
"""

from __future__ import annotations

import re


def sanitize_block_text(text: str) -> str:
    """Neutralise HTML comment delimiters so *text* cannot close our markers."""
    return text.replace("<!--", "&lt;!--").replace("-->", "--&gt;")


def block_pattern(start: str, end: str) -> re.Pattern[str]:
    """Pattern for the managed block: *start* marker to the nearest *end* marker."""
    return re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)


__all__ = ["block_pattern", "sanitize_block_text"]
