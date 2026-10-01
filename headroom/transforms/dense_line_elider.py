"""Elide dense machine-generated lines (minified JS/CSS, base64, RSC payloads).

Tool outputs that dump a fetched web page, a bundled asset or an encoded blob
arrive as a few very long lines with almost no whitespace. None of the
structural compressors can do anything with them: they are not code an AST
walker can prune, not a log, not a table, and the HTML extractor returns the
whole thing (or nothing) once the page is mostly ``<script>``. The result is a
multi-thousand-token block that rides through the router at ratio 1.0.

This module keeps a head and a tail of each such line and replaces the middle
with a one-line marker. It is deliberately dumb: a line is "dense" when it is
long AND has a tiny fraction of spaces. Prose, code, logs, JSON with any
indentation, CSV and markdown all have far more than 6% spaces, so they are
never touched, a JSON-shaped line is left to SmartCrusher, and a lone dense
value (JWT, signed URL, PATH) is below the per-block minimum. Measured on a replayed Codex web-research session against the
installed 0.37.0 wheel: novel savings went from 7.2% to 25.1% of new input, on
top of (not instead of) what the router already removed.
"""

from __future__ import annotations

MIN_LINE_CHARS = 300
MAX_SPACE_RATIO = 0.06
# A single dense line (a JWT, a signed URL, a PATH, an RSA modulus) is a value
# the agent asked for, not a dump: a block is only elided when its dense lines
# add up to at least this many chars. Real bundle dumps are tens of KB.
MIN_DENSE_TOTAL_CHARS = 2000
HEAD_CHARS = 160
TAIL_CHARS = 80


def is_dense_line(line: str) -> bool:
    """True when ``line`` is long, nearly whitespace-free, and not JSON-shaped.

    A compact JSON value (or a run of them) on one line is also nearly
    whitespace-free, but that is SmartCrusher's job and it does it losslessly;
    the elider must never pre-empt it.
    """
    n = len(line)
    # Tabs: TSV / ``psql -A`` rows pass the space ratio but are data the agent
    # asked for; minified assets and encoded blobs never carry tabs.
    if n < MIN_LINE_CHARS or "\t" in line or (line.count(" ") / n) >= MAX_SPACE_RATIO:
        return False
    stripped = line.strip()
    return not (stripped[:1] in "{[" and stripped[-1:] in "}]")


def elide_dense_lines(text: str) -> tuple[str, int]:
    """Return ``(text_with_dense_lines_elided, lines_elided)``.

    Byte-identical to the input (and ``0``) when no line is dense, so callers
    can use ``elided != text`` as the "did anything" signal. Line endings are
    preserved: the split is on ``"\\n"`` only, so ``"\\r"`` stays on its line.
    """
    if len(text) < MIN_LINE_CHARS:
        return text, 0
    lines = text.split("\n")
    if sum(len(line) for line in lines if is_dense_line(line)) < MIN_DENSE_TOTAL_CHARS:
        return text, 0
    out: list[str] = []
    n_elided = 0
    for line in lines:
        if is_dense_line(line):
            omitted = len(line) - HEAD_CHARS - TAIL_CHARS
            out.append(
                f"{line[:HEAD_CHARS]} ...[{omitted} chars of dense "
                f"machine-generated content elided]... {line[-TAIL_CHARS:]}"
            )
            n_elided += 1
        else:
            out.append(line)
    if n_elided == 0:
        return text, 0
    return "\n".join(out), n_elided
