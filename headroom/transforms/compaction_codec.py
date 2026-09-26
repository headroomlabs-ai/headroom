"""Inverse decoder for SmartCrusher's ``csv-schema`` densification format.

SmartCrusher (the Rust ``compaction`` module) renders a JSON array of objects
into a compact ``[N]{schema}`` header followed by CSV rows. That direction is
*encode-only* in the engine — the original is recoverable only by CCR retrieval
of the stashed blob by hash. For a **live agent** that never wants removal/CCR,
the densified text must instead be programmatically reversible so callers can:

  1. **Verify losslessness** — round-trip the densified text and compare it to
     the original (the guarantee behind ``compress(mode="agent")``).
  2. **Re-expand on demand** — a consumer expecting the original ``[{...}]``
     shape can call :func:`expand_compacted` instead of parsing the dense form.

This module is pure-Python (stdlib only) and deliberately mirrors the wire
format emitted by ``crates/headroom-core/.../compaction/formatter.rs`` so the two
stay in lockstep. The cell-encoding contract (post null/empty fix) is:

==================  ==========================  ============================
Original value      Rendered cell               Decoded back to
==================  ==========================  ============================
missing key         ``\\N`` (bare sentinel)      key omitted from the dict
JSON ``null``       empty cell                  ``None``
``""`` (empty str)  ``""`` (quoted empty)       ``""``
literal ``"\\N"``    ``"\\N"`` (quoted)           ``"\\N"``
string w/ ``,"``\\n  CSV-quoted                  the string
other string        bare                        the string
number / bool       bare                        ``int``/``float``/``bool``
nested / array      CSV-quoted compact JSON      the nested object/array
opaque blob         ``<<ccr:hash,kind,size>>``   NOT reversible from text
==================  ==========================  ============================

Schemas declare one column per field as ``name:type`` (or ``name:type?`` when the
column contains nulls/missing). ``name`` may be dotted (``meta.owner``) when the
engine flattened a uniform nested object one level; :func:`expand_compacted`
un-flattens those back into nested dicts.
"""

from __future__ import annotations

import json
from typing import Any

# Sentinel the formatter emits for an absent key (distinct from null's empty
# cell). Kept in one place so the encoder (Rust) and this decoder agree.
MISSING_SENTINEL = "\\N"

# Scalar type tags the schema header can declare. ``json`` covers nested
# objects/arrays and mixed-type columns.
_SCALAR_TYPES = frozenset({"int", "float", "bool", "string", "json", "null"})


def _peel(text: str) -> str:
    """Strip a JSON-string wrapper around a densified block.

    When ContentRouter densifies a tool result whose content was a JSON array,
    it re-serializes the compacted block as a JSON string, so the stored content
    is ``"[N]{...}\\n..."`` (leading quote, escaped newlines) rather than the
    bare block. Peel that wrapper so the same codec handles both forms.
    """
    if text[:1] == '"':
        try:
            inner = json.loads(text)
        except (ValueError, TypeError):
            return text
        if isinstance(inner, str):
            return inner
    return text


def is_compacted(text: str) -> bool:
    """Return True if ``text`` looks like a ``[N]{schema}`` densified block.

    Structural check (not a heuristic): the first non-empty line must open with
    ``[<digits>]{`` and the brace section must close on that line. Accepts a
    JSON-string-wrapped block. Cheap enough to gate ``expand_compacted``.
    """
    if not text:
        return False
    first = _peel(text).lstrip().split("\n", 1)[0]
    if not first.startswith("["):
        return False
    rbracket = first.find("]")
    if rbracket < 2 or not first[1:rbracket].isdigit():
        return False
    return first[rbracket + 1 : rbracket + 2] == "{" and "}" in first[rbracket:]


def contains_removal_marker(text: str) -> bool:
    """True if the densified text removed content (CCR marker or dropped rows).

    Either signal means the text alone cannot reconstruct the original — the
    compression was lossy/removal-based, not pure densification.
    """
    return "<<ccr:" in text or "__dropped:" in text


class _Column:
    __slots__ = ("name", "type", "nullable", "path")

    def __init__(self, decl: str) -> None:
        # Split on the final ':' so dotted names (meta.owner) survive.
        name, _, type_tag = decl.rpartition(":")
        if not name:
            name, type_tag = type_tag, "string"
        self.nullable = type_tag.endswith("?")
        self.type = type_tag[:-1] if self.nullable else type_tag
        if self.type not in _SCALAR_TYPES:
            # Unknown tag → treat as opaque string; keeps the decoder total.
            self.type = "string"
        self.name = name
        self.path = name.split(".")


def _parse_header(line: str) -> tuple[list[_Column], int, int] | None:
    """Parse the header line.

    Form: ``[N]{c:t,c:t?}`` optionally followed by ``__dict:K`` (K value-factor
    legend lines follow the header) and/or ``__dropped:k``. Returns
    ``(cols, declared_rows, dict_count)``.
    """
    if not line.startswith("["):
        return None
    rb = line.find("]")
    open_brace = line.find("{", rb)
    close_brace = line.find("}", open_brace)
    if rb < 0 or open_brace < 0 or close_brace < open_brace:
        return None
    try:
        declared_rows = int(line[1:rb])
    except ValueError:
        return None
    body = line[open_brace + 1 : close_brace]
    cols = [_Column(d) for d in body.split(",")] if body else []
    dict_count = 0
    for token in line[close_brace + 1 :].split():
        if token.startswith("__dict:"):
            try:
                dict_count = int(token[len("__dict:") :])
            except ValueError:
                dict_count = 0
    return cols, declared_rows, dict_count


def _iter_rows(blob: str) -> list[list[tuple[bool, str]]]:
    """Split CSV ``blob`` into rows of ``(was_quoted, value)`` cells.

    Hand-rolled rather than the ``csv`` module because we must preserve the
    quoted-vs-bare distinction for *empty* cells (``""`` is an empty string;
    a bare empty cell is null) — ``csv.reader`` collapses both to ``''``.
    Quotes inside quoted fields are escaped by doubling, matching the Rust
    ``csv_quote``; newlines inside quotes are honored.
    """
    rows: list[list[tuple[bool, str]]] = []
    row: list[tuple[bool, str]] = []
    i, n = 0, len(blob)
    while i < n:
        if blob[i] == '"':  # quoted cell
            i += 1
            buf: list[str] = []
            while i < n:
                if blob[i] == '"':
                    if i + 1 < n and blob[i + 1] == '"':
                        buf.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                buf.append(blob[i])
                i += 1
            row.append((True, "".join(buf)))
            if i < n and blob[i] == "\r":  # tolerate CRLF row endings
                i += 1
            if i < n and blob[i] == ",":
                i += 1
            elif i < n and blob[i] == "\n":
                rows.append(row)
                row = []
                i += 1
        else:  # bare cell up to ',' or newline
            j = i
            while j < n and blob[j] not in ",\n":
                j += 1
            cell = blob[i:j]
            # A bare cell never legitimately holds '\r' (the formatter quotes
            # any value containing it), so a trailing '\r' is a CRLF artifact.
            if cell.endswith("\r"):
                cell = cell[:-1]
            row.append((False, cell))
            if j < n and blob[j] == "\n":
                rows.append(row)
                row = []
            i = j + 1
    if row:
        rows.append(row)
    return rows


def _split_data_rows(blob: str, declared: int) -> list[list[tuple[bool, str]]]:
    """Return the CSV body's data rows, dropping only a trailing-newline artifact.

    The formatter newline-terminates every row, so a single-column ``null`` row
    renders as a bare empty line — a legitimate ``[(False, "")]`` row, NOT an
    artifact. (Unconditionally skipping such rows silently dropped a trailing
    single-column null; see regression test.) Only a lone empty cell that
    EXCEEDS the declared row count is the trailing-newline artifact and is
    dropped; rows with no cells at all are always dropped.
    """
    rows = [r for r in _iter_rows(blob) if r]
    if declared >= 0 and len(rows) == declared + 1 and rows[-1] == [(False, "")]:
        rows.pop()
    return rows


def _decode_cell(was_quoted: bool, value: str, col: _Column) -> tuple[bool, Any]:
    """Return ``(present, decoded)``. ``present=False`` means the key was absent."""
    if was_quoted:
        if col.type == "json":
            try:
                return True, json.loads(value)
            except (ValueError, TypeError):
                return True, value
        return True, value  # quoted "" -> "", quoted "\N" -> "\N", etc.
    if value == MISSING_SENTINEL:
        return False, None
    if value == "":
        return True, None  # null
    if col.type == "int":
        try:
            return True, int(value)
        except ValueError:
            return True, value
    if col.type == "float":
        try:
            return True, float(value)
        except ValueError:
            return True, value
    if col.type == "bool":
        return True, value == "true"
    if col.type == "json":
        try:
            return True, json.loads(value)
        except (ValueError, TypeError):
            return True, value
    return True, value  # string / null / unknown


def _render_raw(was_quoted: bool, value: str) -> str:
    """Reconstruct a cell's exact on-wire text from a parsed ``(quoted, value)``."""
    if was_quoted:
        return '"' + value.replace('"', '""') + '"'
    return value


def factor_values(text: str) -> str:
    """Dictionary-encode low-cardinality string columns. Lossless and reversible.

    SmartCrusher's CSV elides repeated *keys* (the schema header) but still
    repeats low-cardinality *values* on every row — e.g. a ``search_files``
    result repeats the full file path once per match. This pass replaces such a
    column's cells with integer indices into a ``@col=[...]`` legend, hoisting
    each distinct value out of the row body. A column is encoded only when it
    *strictly* saves bytes (legend + indices < the repeated cells) — no
    cardinality threshold, the byte math decides. :func:`expand_compacted`
    reverses it. Returns ``text`` unchanged when nothing benefits or the input
    is already factored / not densified.

    Preserves a JSON-string wrapper if present (agent tool results are wrapped).
    """
    wrapped = text[:1] == '"'
    inner = _peel(text)
    if not is_compacted(inner) or contains_removal_marker(inner):
        return text
    head, _, rest = inner.partition("\n")
    parsed = _parse_header(head.strip())
    if parsed is None:
        return text
    cols, declared, existing_dict = parsed
    if existing_dict:  # already value-factored — idempotent
        return text

    rows = _split_data_rows(rest, declared)
    if not rows:
        return text

    legends: dict[int, list[Any]] = {}  # col index -> distinct values (ordered)
    for j, col in enumerate(cols):
        if col.type != "string":
            continue
        order: list[Any] = []
        seen: dict[Any, int] = {}
        raw_bytes = 0
        index_bytes = 0
        for cells in rows:
            if j >= len(cells):
                continue
            was_quoted, value = cells[j]
            if not was_quoted and value in ("", MISSING_SENTINEL):
                continue  # null / missing stay inline
            # _iter_rows already unescaped quoted cells, so the logical string
            # value is `value` whether or not it was quoted on the wire.
            decoded = value
            raw_bytes += len(_render_raw(was_quoted, value))
            if decoded not in seen:
                seen[decoded] = len(order)
                order.append(decoded)
            index_bytes += len(str(seen[decoded]))
        if not order:
            continue
        legend_json = json.dumps(order, ensure_ascii=False, separators=(",", ":"))
        legend_bytes = len("@") + len(col.name) + len("=") + len(legend_json) + 1
        if legend_bytes + index_bytes < raw_bytes:
            legends[j] = order

    if not legends:
        return text

    # Rebuild: header + __dict:K, then K legend lines, then re-rendered rows.
    close = head.find("}", head.find("{"))
    prefix, trailing = head[: close + 1], head[close + 1 :]
    new_head = f"{prefix} __dict:{len(legends)}{trailing}"

    legend_lines = []
    index_maps: dict[int, dict[Any, int]] = {}
    for j in sorted(legends):
        order = legends[j]
        index_maps[j] = {v: i for i, v in enumerate(order)}
        legend_lines.append(
            f"@{cols[j].name}=" + json.dumps(order, ensure_ascii=False, separators=(",", ":"))
        )

    out_rows = []
    for cells in rows:
        rendered = []
        for j in range(len(cols)):
            if j >= len(cells):
                break
            was_quoted, value = cells[j]
            if j in index_maps and not (not was_quoted and value in ("", MISSING_SENTINEL)):
                rendered.append(str(index_maps[j][value]))
            else:
                rendered.append(_render_raw(was_quoted, value))
        out_rows.append(",".join(rendered))

    factored = "\n".join([new_head, *legend_lines, *out_rows]) + "\n"
    return json.dumps(factored, ensure_ascii=False) if wrapped else factored


def _assign(obj: dict[str, Any], path: list[str], val: Any) -> None:
    """Set ``val`` at a possibly-dotted ``path``, building nested dicts."""
    cur = obj
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = val


def expand_compacted(text: str) -> list[dict[str, Any]] | None:
    """Reconstruct the original array of objects from densified ``text``.

    Returns the list of dicts, or ``None`` if ``text`` is not a parseable
    densified block or carries an irreversible removal marker (CCR / dropped
    rows) — callers treat ``None`` as "cannot losslessly reverse".
    """
    if not is_compacted(text) or contains_removal_marker(text):
        return None
    text = _peel(text)
    head, _, rest = text.partition("\n")
    parsed = _parse_header(head.strip())
    if parsed is None:
        return None
    cols, declared, dict_count = parsed

    # Value-factor legends: ``__dict:K`` declares K ``@col=[...]`` lines that
    # map a low-cardinality column's cells (integer indices) back to values.
    legends: dict[str, list[Any]] = {}
    for _ in range(dict_count):
        legend_line, _, rest = rest.partition("\n")
        name, eq, payload = legend_line.partition("=")
        if not eq or not name.startswith("@"):
            return None
        try:
            legends[name[1:]] = json.loads(payload)
        except (ValueError, TypeError):
            return None

    out: list[dict[str, Any]] = []
    for cells in _split_data_rows(rest, declared):
        if not cells:
            continue
        record: dict[str, Any] = {}
        for idx, col in enumerate(cols):
            if idx >= len(cells):
                break
            was_quoted, value = cells[idx]
            if col.name in legends and not was_quoted:
                # Dict-encoded cell: empty -> null, \N -> missing, else index.
                if value == MISSING_SENTINEL:
                    continue
                if value == "":
                    _assign(record, col.path, None)
                    continue
                try:
                    _assign(record, col.path, legends[col.name][int(value)])
                except (ValueError, IndexError):
                    return None
                continue
            present, decoded = _decode_cell(was_quoted, value, col)
            if present:
                _assign(record, col.path, decoded)
        out.append(record)
    return out
