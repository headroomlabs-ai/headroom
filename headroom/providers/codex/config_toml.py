"""Parse Codex ``config.toml`` with the syntax Codex itself accepts."""

from __future__ import annotations

import sys
from typing import Any, cast

import tomlkit
from tomlkit.exceptions import TOMLKitError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


def loads_codex_config(content: str) -> dict[str, Any]:
    """Parse Codex config text, falling back to ``tomlkit`` for TOML 1.1 syntax.

    Codex accepts TOML 1.1, where an inline table may span several lines and
    end with a trailing comma (#4378). ``tomllib`` before Python 3.15 only
    implements TOML 1.0 and rejects that, so retry with ``tomlkit`` (0.15+),
    which parses it. Text neither parser accepts raises the original
    ``tomllib.TOMLDecodeError``, so callers keep their existing error handling.
    """
    try:
        return cast("dict[str, Any]", tomllib.loads(content))
    except tomllib.TOMLDecodeError as exc:
        try:
            return cast("dict[str, Any]", tomlkit.parse(content).unwrap())
        except TOMLKitError:
            raise exc from None


def split_toml_lines(content: str, keepends: bool = False) -> list[str]:
    """Split ``content`` like ``str.splitlines``, but only at TOML newlines.

    TOML ends a line at LF or CRLF only. ``str.splitlines`` also breaks at
    characters such as U+2028 and U+0085, which TOML allows inside a
    single-line string or comment, so its pieces are not always TOML lines.
    """
    lines = [line + "\n" for line in content.split("\n")]
    lines[-1] = lines[-1][:-1]
    if not lines[-1]:
        lines.pop()
    if keepends:
        return lines
    return [line.removesuffix("\n").removesuffix("\r") for line in lines]


def root_assignment_lines(content: str) -> set[int]:
    """Return indexes of the lines that start a key/value pair at the document root.

    Indexes refer to ``split_toml_lines(content)``. Only lines before
    the first table header count, and a line inside a multi-line inline table,
    array, or string is skipped, so a key nested in a multi-line inline table
    (TOML 1.1, e.g. ``profiles = {\\n  work = { model_provider = ... } }``) is
    never mistaken for a root key.
    """
    roots: set[int] = set()
    depth = 0
    open_string: str | None = None  # delimiter of an open multi-line string
    for index, line in enumerate(split_toml_lines(content, keepends=True)):
        if depth == 0 and open_string is None:
            stripped = line.lstrip(" \t")
            if stripped.startswith("["):
                break  # first table header ends the root
            if stripped and stripped[0] not in "#\r\n":
                roots.add(index)
        pos = 0
        while pos < len(line):
            if open_string is not None:
                if line.startswith(open_string, pos):
                    pos += 3
                    # Up to two quotes may directly precede the closing delimiter.
                    for _ in range(2):
                        if pos < len(line) and line[pos] == open_string[0]:
                            pos += 1
                    open_string = None
                elif open_string == '"""' and line[pos] == "\\":
                    pos += 2
                else:
                    pos += 1
                continue
            char = line[pos]
            if char == "#":
                break
            if line.startswith('"""', pos) or line.startswith("'''", pos):
                open_string = line[pos : pos + 3]
                pos += 3
            elif char in "\"'":
                pos += 1
                while pos < len(line) and line[pos] not in (char, "\n"):
                    pos += 2 if char == '"' and line[pos] == "\\" else 1
                pos += 1
            else:
                if char in "[{":
                    depth += 1
                elif char in "]}" and depth:
                    depth -= 1
                pos += 1
    return roots
