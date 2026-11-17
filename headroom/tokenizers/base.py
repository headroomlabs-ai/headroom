"""Base classes for tokenizer implementations.

Defines the TokenCounter protocol and BaseTokenizer class that all
tokenizer backends must implement.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import zlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

#: Cap on the serialized form of a non-string tool-call field. A malformed
#: upstream can put an arbitrarily large object where a JSON string belongs;
#: serializing it unbounded would turn a token *estimate* into a multi-megabyte
#: encode. Truncating keeps the estimate finite and the request alive.
_MAX_COERCED_FIELD_CHARS = 200_000

#: Admission policy for :class:`TokenCountCache`. These bound memory only — they
#: never change a returned count, so they are not a behavioural threshold.
#: Strings below the floor encode in microseconds, so caching them would only
#: evict the large entries that the cache exists for.
_COUNT_CACHE_MIN_CHARS = 256
_COUNT_CACHE_MAX_ENTRIES = 4096
_COUNT_CACHE_MAX_CHARS = 8_000_000

#: Document pricing (see ``BaseTokenizer._estimate_document_tokens``). Anthropic
#: documents ~1,500-3,000 text tokens per PDF page; the low end is used. Pages
#: come from the PDF's active page tree, read the way a PDF reader reads it: the
#: cross-reference data the last ``startxref`` chains to (tables and xref streams,
#: through ``/Prev``) gives ``/Root`` and where each object lives, and the catalog
#: and its ``/Pages`` node are read there (an ``N G obj`` at that offset, or entry
#: i of an object stream), each verified, never wherever their text happens to
#: appear in the file. Comments and string literals are cut from an object's
#: syntax, and each key comes from its own dictionary's top level, never from a
#: dict, array or hex string nested in it. A missing, damaged or unverifiable xref,
#: an absurd ``/Count``, or a spent budget gives ~3KB of file per page instead.
_DOCUMENT_TOKENS_PER_PAGE = 1500
_DOCUMENT_BYTES_PER_PAGE = 3000
_PDF_MAX_PAGES = 10_000  # a larger /Count is treated as malformed, not priced
# One budget per file of bytes scanned or inflated, and one of xref rows read
# (repeats included), so no chain of sections or streams can multiply the work.
_PDF_OBJECT_STREAM_MAX_BYTES = 4_000_000
_PDF_MAX_OBJECTS = 100_000
_PDF_MAX_XREF_SECTIONS = 1_000  # a longer /Prev chain is treated as malformed
# Each pattern opens with a lookahead on its first byte so the scan skips binary fast.
# A stream's data starts after its keyword, any spaces and an EOL if present (a
# lone CR too): the spec requires CRLF or LF, but readers accept the others.
_PDF_SYNTAX_STOP = re.compile(
    rb"(?=[%(se])(?:%[^\r\n]*|\(|(?<![A-Za-z/])(?:stream *(?:\r\n?|\n)?|endobj|startxref))"
)
_PDF_STRING_TOKEN = re.compile(rb"\\.|[()]", re.S)
_PDF_STREAM_LENGTH = re.compile(rb"/Length\s+(\d{1,10})(?!\d|\s+\d+\s+R)")
_PDF_STREAM_END = re.compile(rb"\s*endstream")
_PDF_ROOT = re.compile(rb"/Root\s+(\d{1,10})\s+(\d{1,5})\s+R")
_PDF_PAGES = re.compile(rb"/Pages\s+(\d{1,10})\s+(\d{1,5})\s+R")
# A longer digit run never reaches int(): it is malformed, and the size estimate stands in.
_PDF_COUNT = re.compile(rb"/Count\s+(\d{1,5})(?!\d)")
_PDF_FIRST = re.compile(rb"/First\s+(\d{1,10})")
_PDF_NESTING = re.compile(rb"<<|>>|<[^<>]*>|(\[[\d\s]*\])|\[|\]")
_PDF_LENGTH_REF = re.compile(rb"/Length\s+(\d{1,10})\s+(\d{1,5})\s+R")
_PDF_LENGTH_OBJECT = re.compile(rb"\s*(\d{1,10})\s+(\d{1,5})\s+obj\s*(\d{1,10})\s*endobj")
_PDF_OBJECT_HEADER = re.compile(rb"\s*(\d{1,10})\s+(\d{1,5})\s+obj\b")
_PDF_STARTXREF = re.compile(rb"startxref\s+(\d{1,10})")
_PDF_XREF_TABLE = re.compile(rb"\s*xref\b")
_PDF_XREF_SUBSECTION = re.compile(rb"\s*(\d{1,10})\s+(\d{1,5})\s+")
_PDF_XREF_ROW = re.compile(rb"(\d{10}) (\d{5}) ([nf])")
_PDF_XREF_TRAILER = re.compile(rb"\s*trailer\b")
_PDF_XREF_WIDTHS = re.compile(rb"/W\s*\[\s*(\d)\s+(\d)\s+(\d)\s*\]")


def _pdf_syntax(data: bytes, pos: int = 0, end: int | None = None) -> tuple[bytes, int, bool]:
    """PDF syntax from ``pos`` to the next ``stream``/``endobj``/``startxref`` (or ``end``),
    minus comments and string literals; also where it stopped and whether a stream
    starts there."""
    end = len(data) if end is None else min(end, len(data))
    parts = []
    while match := _PDF_SYNTAX_STOP.search(data, pos, end):
        parts.append(data[pos : match.start()])
        pos = match.end()
        if match.group() == b"(":
            depth = 0  # a string literal may nest balanced parentheses
            pos = end
            for token in _PDF_STRING_TOKEN.finditer(data, match.start(), end):
                depth += {b"(": 1, b")": -1}.get(token.group(), 0)
                if not depth:
                    pos = token.end()
                    break
        elif not match.group().startswith(b"%"):
            return b"".join(parts), pos, match.group().startswith(b"stream")
    return b"".join(parts) + data[pos:end], end, False


def _pdf_own_keys(syntax: bytes, nested: list[bytes] | None = None) -> bytes:
    """The top level of the dictionary ``syntax`` opens with, each nested dict, array
    and hex string cut to ``_<i>``: a key read from it is that dictionary's own. An
    array of plain integers holds no key, so it stays (an xref stream's ``/W`` and
    ``/Index``). ``nested`` collects what was cut, ``_<i>`` being entry i."""
    body = syntax.lstrip(b"\0\t\n\x0c\r ")
    if not body.startswith(b"<<"):
        return b""
    cut: list[bytes] = [] if nested is None else nested
    parts: list[bytes] = []
    depth = pos = opened = 0
    for token in _PDF_NESTING.finditer(body):
        step = {b"<<": 1, b"[": 1, b">>": -1, b"]": -1}.get(token.group(), 0)
        if depth == 1:
            parts.append(body[pos : token.start()])
            if step < 0:
                break  # the dictionary's own end
            if token.group(1):
                parts.append(token.group(1))
            else:
                parts.append(b" _%d " % len(cut))
                opened = token.start()
                if not step:
                    cut.append(token.group())  # a hex string
        depth += step
        if step < 0 and depth == 1:
            cut.append(body[opened : token.end()])
        if not depth:
            break
        pos = token.end()
    return b"".join(parts)


def _pdf_png_rows(data: bytes, columns: int) -> bytes:
    """``data`` decoded from PNG predictor rows: each a filter byte, then ``columns``
    bytes, one per sample (an xref stream's /Colors 1 /BitsPerComponent 8)."""
    out, prior = bytearray(), bytes(columns)
    for at in range(0, len(data), columns + 1):
        kind, row = data[at], bytearray(data[at + 1 : at + 1 + columns])
        if kind == 2:
            row = bytearray((a + b) & 255 for a, b in zip(row, prior))
        elif kind:
            for i in range(len(row)):
                left, up, corner = row[i - 1] if i else 0, prior[i], prior[i - 1] if i else 0
                if kind == 1:
                    row[i] = (row[i] + left) & 255
                elif kind == 3:
                    row[i] = (row[i] + (left + up) // 2) & 255
                elif kind == 4:
                    guess = left + up - corner  # Paeth: the neighbour nearest the guess
                    pa, pb, pc = abs(guess - left), abs(guess - up), abs(guess - corner)
                    row[i] = (
                        row[i] + (left if pa <= pb and pa <= pc else up if pb <= pc else corner)
                    ) & 255
                else:
                    raise _PdfUnreadable
        out += row
        prior = bytes(row)
    return bytes(out)


class _PdfUnreadable(Exception):
    """The PDF fails verification or a budget runs out: its size estimate stands in."""


class _PdfReader:
    """Reads objects at the locations the PDF's cross-reference data gives for them."""

    def __init__(self, doc: bytes) -> None:
        self.doc = doc
        self.budget = _PDF_OBJECT_STREAM_MAX_BYTES
        self.rows = _PDF_MAX_OBJECTS
        self.entries: dict[int, tuple[int, int, int]] = {}  # number: (type, field 2, field 3)
        self.packed: dict[int, tuple[list[bytes], int, bytes]] = {}  # header, /First, data
        self.root: re.Match[bytes] | None = None
        eof = doc.rfind(b"%%EOF")  # bytes after the last revision hold no objects
        last = doc.rfind(b"startxref", 0, len(doc) if eof < 0 else eof)
        start = _PDF_STARTXREF.match(doc, last) if last >= 0 else None
        todo = [int(start.group(1))] if start else []
        seen: set[int] = set()
        while todo:
            at = todo.pop(0)
            if at in seen:
                continue
            seen.add(at)
            if len(seen) > _PDF_MAX_XREF_SECTIONS:
                raise _PdfUnreadable
            self.spend(rows=1)
            table = _PDF_XREF_TABLE.match(doc, at)
            keys = self.read_table(table.end()) if table else self.read_xref_stream(at)
            self.root = self.root or _PDF_ROOT.search(keys)  # the newest trailer's
            for key in (rb"/XRefStm", rb"/Prev"):  # a table's stream entries, then older sections
                if found := re.search(key + rb"\s+(\d{1,10})(?!\d)", keys):
                    todo.append(int(found.group(1)))
        if not self.root:
            raise _PdfUnreadable

    def spend(self, scanned: int = 0, rows: int = 0) -> None:
        self.budget -= scanned
        self.rows -= rows
        if self.budget < 0 or self.rows < 0:
            raise _PdfUnreadable

    def keys_at(self, pos: int, nested: list[bytes] | None = None) -> tuple[bytes, int, bool]:
        """Own keys of the dictionary at ``pos``, where its syntax stops, and whether a
        stream starts there; the scan is charged to the budget."""
        syntax, stop, is_stream = _pdf_syntax(self.doc, pos, pos + self.budget + 1)
        self.spend(stop - pos)  # a scan past the budget stops one byte beyond it
        return _pdf_own_keys(syntax, nested), stop, is_stream

    def read_table(self, pos: int) -> bytes:
        while sub := _PDF_XREF_SUBSECTION.match(self.doc, pos):
            self.spend(rows=int(sub.group(2)) + 1)  # also caps the distinct counts compiled
            rows = re.compile(rb"(?:\d{10} \d{5} [nf]\s{1,2}){%s}" % sub.group(2))
            if not (block := rows.match(self.doc, sub.end())):
                raise _PdfUnreadable
            for number, (offset, gen, kind) in enumerate(
                _PDF_XREF_ROW.findall(block.group()), int(sub.group(1))
            ):
                if kind == b"n":  # the newest section is read first, so its entry wins
                    self.entries.setdefault(number, (1, int(offset), int(gen)))
            pos = block.end()
        if not (trailer := _PDF_XREF_TRAILER.match(self.doc, pos)):
            raise _PdfUnreadable
        return self.keys_at(trailer.end())[0]

    def read_xref_stream(self, at: int) -> bytes:
        header = _PDF_OBJECT_HEADER.match(self.doc, at)
        nested: list[bytes] = []
        keys, data = self.stream(header.end(), nested) if header else (b"", b"")
        widths = _PDF_XREF_WIDTHS.search(keys)
        size = re.search(rb"/Size\s+(\d{1,10})", keys)
        if not (widths and size and re.search(rb"/Type\s*/XRef\b", keys)):
            raise _PdfUnreadable
        kind, field, gen = map(int, widths.groups())
        width = kind + field + gen
        predicted = 0
        if parms := re.search(rb"/DecodeParms\s+_(\d{1,4})(?!\d)", keys):
            raw = nested[int(parms.group(1))] if int(parms.group(1)) < len(nested) else b""
            if not raw.startswith(b"<<"):
                raise _PdfUnreadable  # an array, or "_<i>" typed into the dict itself
            values = _pdf_own_keys(raw)
            predictor = re.search(rb"/Predictor\s+(\d{1,2})(?!\d)", values)
            columns = re.search(rb"/Columns\s+(\d{1,2})(?!\d)", values)
            if re.search(rb"/Colors\s+(?!1(?!\d))|/BitsPerComponent\s+(?!8(?!\d))", values):
                raise _PdfUnreadable
            if predictor and int(predictor.group(1)) >= 10:
                predicted = 1
                if int(columns.group(1) if columns else 1) != width:
                    raise _PdfUnreadable
            elif predictor and int(predictor.group(1)) != 1:
                raise _PdfUnreadable  # TIFF prediction: not written for xref streams
        if not width:
            raise _PdfUnreadable
        index = re.search(rb"/Index\s*\[([\d\s]*)\]", keys)
        bounds = (index.group(1) if index else b"0 " + size.group(1)).split()
        if any(len(n) > 10 for n in bounds):
            raise _PdfUnreadable
        starts, lengths = [int(n) for n in bounds[::2]], [int(n) for n in bounds[1::2]]
        count = min(sum(lengths), len(data) // (width + predicted))
        self.spend(rows=count)
        data = data[: count * (width + predicted)]
        if predicted:
            data = _pdf_png_rows(data, width)
        numbers = (n for a, length in zip(starts, lengths) for n in range(a, a + length))
        for number, at_row in zip(numbers, range(0, count * width, width)):
            entry = data[at_row : at_row + width]
            entry_type = int.from_bytes(entry[:kind] or b"\x01", "big")
            if entry_type in (1, 2):
                second = int.from_bytes(entry[kind : kind + field], "big")
                self.entries.setdefault(
                    number, (entry_type, second, int.from_bytes(entry[kind + field :], "big"))
                )
        return keys

    def stream(self, pos: int, nested: list[bytes]) -> tuple[bytes, bytes]:
        """Own keys and inflated data of the stream object whose dict starts at ``pos``.
        Its end must be verified: ``endstream`` right after ``/Length`` bytes."""
        keys, start, is_stream = self.keys_at(pos, nested)
        length = _PDF_STREAM_LENGTH.search(keys)
        size = int(length.group(1)) if length else self.integer(_PDF_LENGTH_REF.search(keys))
        if not (is_stream and _PDF_STREAM_END.match(self.doc, start + size)):
            raise _PdfUnreadable
        data = self.doc[start : start + size]
        if b"/Filter" not in keys:
            if b"/DecodeParms" in keys:
                raise _PdfUnreadable
            return keys, data
        if not re.search(rb"/Filter\s*/FlateDecode\b", keys) or self.budget <= 0:
            raise _PdfUnreadable  # (max_length 0 would mean no limit)
        inflate = zlib.decompressobj()
        try:
            data = inflate.decompress(data, self.budget)
        except zlib.error:
            raise _PdfUnreadable from None
        if inflate.unconsumed_tail:
            raise _PdfUnreadable
        self.spend(len(data))
        return keys, data

    def integer(self, ref: re.Match[bytes] | None) -> int:
        """The integer an indirect ``/Length`` names, at its xref offset."""
        entry = self.entries.get(int(ref.group(1))) if ref else None
        value = _PDF_LENGTH_OBJECT.match(self.doc, entry[1]) if entry and entry[0] == 1 else None
        if not (ref and entry and value and value.group(1, 2) == ref.group(1, 2)):
            raise _PdfUnreadable
        if int(ref.group(2)) != entry[2]:  # the generation the xref holds
            raise _PdfUnreadable
        return int(value.group(3))

    def get(self, ref: re.Match[bytes] | None) -> bytes:
        """Own keys of the object ``ref`` names, read where the xref says it lives."""
        number, gen = (int(ref.group(1)), int(ref.group(2))) if ref else (-1, -1)
        kind, where, extra = self.entries.get(number, (0, 0, 0))
        if kind == 1 and extra == gen:
            header = _PDF_OBJECT_HEADER.match(self.doc, where)
            if header and (int(header.group(1)), int(header.group(2))) == (number, gen):
                return self.keys_at(header.end())[0]
        elif kind == 2 and gen == 0:
            head, first, data = self.packed.get(where) or self.unpack(where)
            i = 2 * extra  # entry i of the stream's header of (number, offset) pairs
            if i + 1 < len(head) and int(head[i]) == number:
                stop = first + int(head[i + 3]) if i + 3 < len(head) else len(data)
                return _pdf_own_keys(_pdf_syntax(data[first + int(head[i + 1]) : stop])[0])
        raise _PdfUnreadable

    def unpack(self, number: int) -> tuple[list[bytes], int, bytes]:
        """An object stream's header numbers, its /First and its inflated data."""
        kind, where, gen = self.entries.get(number, (0, 0, 0))
        header = _PDF_OBJECT_HEADER.match(self.doc, where) if kind == 1 else None
        if not (header and (int(header.group(1)), int(header.group(2))) == (number, gen)):
            raise _PdfUnreadable
        keys, data = self.stream(header.end(), [])
        first = _PDF_FIRST.search(keys)
        if not (first and re.search(rb"/Type\s*/ObjStm\b", keys)) or b"/DecodeParms" in keys:
            raise _PdfUnreadable
        head = re.findall(rb"\d{1,10}", data[: int(first.group(1))])
        self.packed[number] = head, int(first.group(1)), data
        return self.packed[number]


def _pdf_page_count(doc: bytes) -> int:
    """Pages in the PDF's active page tree, else 0."""
    try:
        reader = _PdfReader(doc)
        count = _PDF_COUNT.search(reader.get(_PDF_PAGES.search(reader.get(reader.root))))
    except _PdfUnreadable:
        return 0
    return int(count.group(1)) if count and 0 < int(count.group(1)) <= _PDF_MAX_PAGES else 0


class TokenCountCache:
    """Exact memo for ``count_text``.

    ``count_text`` is a pure function of its text, so replaying a stored count is
    value-identical. That is the whole safety argument: the result is the same
    integer, which is why this is safe even at the call sites whose count feeds a
    routing decision (``context_pressure`` -> ``min_ratio``) rather than a log line.

    Keyed on the text itself rather than a hash. A hash collision here would hand
    back the wrong count for real content and silently change what gets
    compressed; the counts are too load-bearing to trade correctness for a
    smaller key. The price is holding the strings, so entries *and* total
    characters are capped.

    No lock: ``dict`` get/set/clear are atomic under the GIL, and the pipeline
    runs on a thread pool. A true LRU would need ``move_to_end`` on every hit,
    which is not atomic — hence no per-key eviction. A full cache still clears,
    but only on the one put that admits a new key past the cap: repeated
    re-counts of *known* texts (an agent loop re-counting a stable prefix on
    every request) are pure dict hits and never trip the eviction, so the
    classic storm — one fresh tool_result per request evicting the whole
    cache and forcing a full-prefix re-encode — cannot happen. A cleared
    cache costs one re-encode, never a wrong answer.
    """

    __slots__ = ("_chars", "_counts", "_max_chars", "_max_entries", "_min_chars")

    def __init__(
        self,
        *,
        min_chars: int = _COUNT_CACHE_MIN_CHARS,
        max_entries: int = _COUNT_CACHE_MAX_ENTRIES,
        max_chars: int = _COUNT_CACHE_MAX_CHARS,
    ) -> None:
        self._counts: dict[str, int] = {}
        self._chars = 0
        self._min_chars = min_chars
        self._max_entries = max_entries
        self._max_chars = max_chars

    def get(self, text: str) -> int | None:
        """Return the stored count for *text*, or None."""
        return self._counts.get(text)

    def put(self, text: str, count: int) -> None:
        """Store *count* for *text* if it is worth caching.

        A re-count of an entry already in the cache is a hit, not an
        admission: updating the stored value cannot grow the cache, so it
        must not run the eviction check (a dict hit would clear the whole
        cache and force a full re-encode of every cached message on the
        next miss — the storm this exists to prevent).
        Oversized texts are never admitted, so warm entries survive.
        """
        if text in self._counts:
            self._counts[text] = count
            return
        if len(text) < self._min_chars:
            return
        if len(text) > self._max_chars:
            return
        if len(self._counts) >= self._max_entries or self._chars >= self._max_chars:
            self._counts.clear()
            self._chars = 0
        self._counts[text] = count
        self._chars += len(text)

    def clear(self) -> None:
        self._counts.clear()
        self._chars = 0


def coerce_countable_text(value: Any) -> str:
    """Return *value* as text safe to pass to ``count_text``.

    Tool-call fields (``function.name``, ``function.arguments``, ``id``) are
    strings per the OpenAI spec, but OpenAI-compatible upstreams do emit
    ``None`` or a raw object there. Passing those straight to
    ``tiktoken.encode()`` raises ``TypeError: expected string or buffer``, which
    failed the whole compression with a 503 — and kept failing on every later
    request, because the malformed message stays in conversation history.

    ``None`` counts as nothing; dicts/lists are JSON-serialized so an object
    ``arguments`` is priced roughly like the JSON string it should have been;
    anything else falls back to ``str()``.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, dict | list | tuple):
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    else:
        text = str(value)
    return text[:_MAX_COERCED_FIELD_CHARS]


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for token counting implementations.

    Any class implementing this protocol can be used with Headroom
    for token counting. This allows integration with various
    tokenizer backends (tiktoken, HuggingFace, custom, etc.).
    """

    def count_text(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens in the text.
        """
        ...

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in a list of chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            Total token count including message overhead.
        """
        ...


class BaseTokenizer(ABC):
    """Abstract base class for tokenizer implementations.

    Provides common functionality for counting messages while
    requiring subclasses to implement text tokenization.
    """

    # Token overhead per message (role, formatting, etc.)
    # Override in subclasses for model-specific overhead
    MESSAGE_OVERHEAD = 4
    REPLY_OVERHEAD = 3  # Assistant reply start tokens

    # Oversized-blob token estimation (see _count_serialized). Serializing a blob
    # is cheap; running count_text over the whole multi-megabyte string is what
    # blocks the proxy event loop, so a large blob is counted from an even-spread
    # sample of the serialized string, scaled by length.
    LARGE_BLOB_CHARS = 50_000  # above this serialized size, sample instead of full count
    SAMPLE_CHARS = 20_000  # total characters fed to count_text for an oversized blob
    SAMPLE_CHUNK = 2_000  # size of each evenly-spaced chunk in that sample

    @abstractmethod
    def count_text(self, text: str) -> int:
        """Count tokens in a text string. Must be implemented by subclasses."""
        pass

    def count_message(self, message: dict[str, Any]) -> int:
        """Count tokens in a single message.

        Args:
            message: A message dict with 'role' and 'content'.

        Returns:
            Token count for this message.
        """
        return self.count_messages([message]) - self.REPLY_OVERHEAD

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in a list of chat messages.

        Uses OpenAI-style message counting as the baseline, which
        works well for most models.

        Args:
            messages: List of message dicts.

        Returns:
            Total token count.
        """
        total = 0

        for message in messages:
            # Base message overhead
            total += self.MESSAGE_OVERHEAD

            # Count role
            role = message.get("role", "")
            total += self.count_text(coerce_countable_text(role))

            # Count content
            content = message.get("content")
            if isinstance(content, dict):
                # A single block sent without its list wrapper is priced like
                # the wrapped block, so images keep their fixed estimate (#4273).
                content = [content]
            if content is not None:
                if isinstance(content, str):
                    total += self.count_text(content)
                elif isinstance(content, list):
                    # Multi-part content (images, tool results, etc.)
                    total += self._count_content_parts(content)
                else:
                    total += self.count_text(coerce_countable_text(content))

            # Count tool calls
            tool_calls = message.get("tool_calls")
            if tool_calls:
                total += self._count_tool_calls(tool_calls)

            # Count function call (legacy)
            function_call = message.get("function_call")
            if function_call:
                total += self._count_function_call(function_call)

            # Count name field
            name = message.get("name")
            if name:
                total += self.count_text(coerce_countable_text(name))
                total += 1  # Name field overhead

        # Reply start overhead
        total += self.REPLY_OVERHEAD

        return total

    def _count_content_parts(self, parts: list[Any]) -> int:
        """Count tokens in multi-part content.

        Handles both Anthropic format ({"type": "text", "text": "..."})
        and Strands SDK format ({"text": "..."} without "type" field).
        """
        total = 0
        for part in parts:
            if isinstance(part, dict):
                part_type = part.get("type", "")

                if part_type == "text":
                    total += self.count_text(coerce_countable_text(part.get("text", "")))
                elif part_type in ("image_url", "image", "input_image"):
                    # Images are NOT tokenized as text — they have a pixel-based cost.
                    # Anthropic: tokens = (width * height) / 750, max ~1600 after resize.
                    # OpenAI: similar tile-based calculation, ~765 tokens for high-detail.
                    # We use 1600 as a conservative estimate (max after auto-resize).
                    # This prevents the base64 blob from being json.dumps'd and counted
                    # as text tokens (1MB image = ~330K fake tokens without this).
                    total += 1600
                elif part_type == "document":
                    # Anthropic document block. Its base64 PDF used to reach the
                    # JSON catch-all below and be priced as prose: a 1 MB PDF
                    # counted ~907K tokens, against the ~1,600 an image of the
                    # same bytes gets.
                    total += self._count_document_block(part)
                elif part_type in ("input_audio", "audio"):
                    # Audio has fixed token cost, not tokenized as text
                    total += 200
                elif part_type == "tool_result":
                    content = part.get("content", "")
                    if isinstance(content, str):
                        total += self.count_text(content)
                    elif isinstance(content, list):
                        # Recurse into nested blocks (matching the Strands
                        # toolResult branch below). A tool that returns an image
                        # nests a base64 block here; serializing it would price
                        # the ~KB/MB base64 string as text — a 50-200x overcount
                        # (a screenshot reads as tens of thousands of tokens).
                        total += self._count_content_parts(content)
                    else:
                        total += self._count_serialized(content)
                elif part_type == "tool_use":
                    total += self.count_text(coerce_countable_text(part.get("name", "")))
                    total += self._count_serialized(part.get("input", {}))
                elif not part_type and "text" in part:
                    # Strands SDK format: {"text": "..."} without "type" field
                    total += self.count_text(coerce_countable_text(part["text"]))
                elif not part_type and "toolUse" in part:
                    # Strands SDK tool_use: {"toolUse": {"name": ..., "input": ...}}
                    tool_use = part["toolUse"]
                    total += self.count_text(coerce_countable_text(tool_use.get("name", "")))
                    total += self._count_serialized(tool_use.get("input", {}))
                elif not part_type and "toolResult" in part:
                    # Strands SDK tool_result: {"toolResult": {"content": [...]}}
                    tool_result = part["toolResult"]
                    tr_content = tool_result.get("content", [])
                    if isinstance(tr_content, str):
                        total += self.count_text(tr_content)
                    elif isinstance(tr_content, list):
                        # Recurse into nested content blocks
                        total += self._count_content_parts(tr_content)
                    else:
                        total += self._count_serialized(tr_content)
                elif not part_type and "reasoningContent" in part:
                    # Strands SDK reasoning: {"reasoningContent": {"reasoningText": {"text": "..."}}}
                    # This is actual text — count it precisely.
                    reasoning = part["reasoningContent"]
                    reasoning_text = reasoning.get("reasoningText", {})
                    if isinstance(reasoning_text, dict):
                        total += self.count_text(
                            coerce_countable_text(reasoning_text.get("text", ""))
                        )
                    elif isinstance(reasoning_text, str):
                        total += self.count_text(reasoning_text)
                elif not part_type and "document" in part:
                    # Strands SDK document: {"document": {"source": {"bytes": ...}}}
                    # Provider internally extracts text from PDF/DOCX then tokenizes.
                    # Accurate counting would require a PDF parser — instead we use
                    # the Anthropic documented estimate of ~1500 tokens per page,
                    # with ~3KB of PDF per page as a rough heuristic.
                    doc = part["document"]
                    source = doc.get("source", {})
                    doc_bytes = source.get("bytes", b"")
                    if isinstance(doc_bytes, bytes | bytearray):
                        total += self._estimate_document_tokens(bytes(doc_bytes))
                    else:
                        total += self.count_text(str(doc_bytes))
                elif not part_type and "image" in part:
                    # Strands SDK image: {"image": {"source": {"bytes": ...}}}
                    # Anthropic formula: tokens = (width * height) / 750.
                    # Decode with Pillow for exact count; fall back to estimate.
                    total += self._estimate_image_tokens(part["image"])
                elif not part_type and "video" in part:
                    # Strands SDK video: provider samples ~1 fps, each frame costs
                    # image tokens. We can't decode frames without heavy deps, so
                    # estimate from byte size assuming ~30KB per frame, ~1000 tokens
                    # per frame (average image).
                    vid = part["video"]
                    source = vid.get("source", {})
                    vid_bytes = source.get("bytes", b"")
                    if isinstance(vid_bytes, bytes | bytearray):
                        frames = max(1, len(vid_bytes) // 30000)
                        total += frames * 1000
                    else:
                        total += 3200
                elif part_type == "thinking":
                    # Anthropic extended thinking replayed by the client. The
                    # ``signature`` is the encrypted full reasoning: on the
                    # keep-all-turns models (Opus 4.5+, Sonnet 4.6+, the 5.x
                    # line) the server decrypts it into the prompt and bills
                    # it as input, and under ``display: "omitted"`` (the 5.x
                    # default) the ``thinking`` text is empty, so text alone
                    # is not the input. Nor is the base64: the JSON catch-all
                    # below priced it as prose at ~3 chars/token, and on a
                    # 369-message Claude Code session 99 signatures counted
                    # 256K of 414K tokens against ~313K provider-reported.
                    # Price the text plus the decoded signature bytes at
                    # ~4 bytes/token (len * 3/4 / 4), which lands that session
                    # near the provider figure and never prices an omitted
                    # block at zero.
                    total += self.count_text(coerce_countable_text(part.get("thinking", "") or ""))
                    total += len(part.get("signature") or "") * 3 // 16
                else:
                    # Unknown type - estimate from JSON
                    total += self._count_serialized(part)
            elif isinstance(part, str):
                total += self.count_text(part)

        return total

    def _count_serialized(self, obj: Any) -> int:
        """Count tokens for a non-string content blob.

        Small blobs are counted exactly. For an oversized one, run ``count_text``
        over an even-spread sample of the serialized string and scale by length.
        Serializing is cheap; ``count_text`` over the whole multi-megabyte string
        is what blocks the request path, so its input is bounded here. The even
        spread keeps the sample representative (a single slice would skew the scale
        high), and bounding the count biases the estimate slightly low — the safe
        direction. Fails open. Mirrors the image and document guards above.
        """
        try:
            s = json.dumps(obj)
        except Exception:
            # fail-open: nominal estimate when obj isn't JSON-serializable
            return self.LARGE_BLOB_CHARS // 4
        if len(s) <= self.LARGE_BLOB_CHARS:
            return self.count_text(s)
        chunks = max(1, self.SAMPLE_CHARS // self.SAMPLE_CHUNK)
        step = len(s) / chunks
        sample = "".join(
            s[int(i * step) : int(i * step) + self.SAMPLE_CHUNK] for i in range(chunks)
        )
        return int(self.count_text(sample) * len(s) / len(sample))

    def _count_document_block(self, part: dict[str, Any]) -> int:
        """Price an Anthropic ``document`` block by its source type."""
        source = part.get("source")
        source = source if isinstance(source, dict) else {}
        source_type = source.get("type")
        if source_type == "base64":
            try:
                doc_bytes = base64.b64decode(source.get("data") or b"")
            except (binascii.Error, TypeError, ValueError):
                doc_bytes = b""
            total = self._estimate_document_tokens(doc_bytes)
        elif source_type == "content":
            content = source.get("content")
            if isinstance(content, list):
                total = self._count_content_parts(content)
            else:
                total = self.count_text(str(content or ""))
        elif source_type in ("url", "file"):
            # The bytes live with the provider; one page as a nominal estimate.
            total = _DOCUMENT_TOKENS_PER_PAGE
        else:
            # A plain-text source, or an untyped one carrying text.
            total = self.count_text(str(source.get("data") or ""))
        for key in ("title", "context"):
            value = part.get(key)
            if isinstance(value, str):
                total += self.count_text(value)
        return total

    @staticmethod
    def _estimate_document_tokens(doc_bytes: bytes) -> int:
        """Estimate a document file's tokens from its page count (see the constants)."""
        if b"%PDF-" not in doc_bytes[:1024]:
            # Only a PDF has a page tree: other bytes (a Strands txt/csv/docx
            # document) keep the size estimate even when their text quotes PDF syntax.
            return max(1, len(doc_bytes) // _DOCUMENT_BYTES_PER_PAGE) * _DOCUMENT_TOKENS_PER_PAGE
        pages = _pdf_page_count(doc_bytes) or len(doc_bytes) // _DOCUMENT_BYTES_PER_PAGE
        return max(1, pages) * _DOCUMENT_TOKENS_PER_PAGE

    @staticmethod
    def _estimate_image_tokens(image_data: dict[str, Any]) -> int:
        """Estimate tokens for an image using Anthropic's formula: (w*h)/750.

        Tries to decode dimensions with Pillow. Falls back to a conservative
        estimate based on byte size.
        """
        source = image_data.get("source", {})
        img_bytes = source.get("bytes", b"")

        from headroom.image.decoder_guard import pillow_is_safe

        if isinstance(img_bytes, bytes | bytearray) and len(img_bytes) > 0 and pillow_is_safe():
            try:
                import io

                from PIL import Image

                img = Image.open(io.BytesIO(img_bytes))
                w, h = img.size
                # Anthropic resizes to fit 1568x1568 max
                max_dim = 1568
                if w > max_dim or h > max_dim:
                    scale = max_dim / max(w, h)
                    w, h = int(w * scale), int(h * scale)
                return max(100, (w * h) // 750)
            except Exception:
                pass

        # Fallback: estimate from byte size.
        # Typical screenshot: ~200KB ≈ 1200x800 ≈ 1280 tokens
        if isinstance(img_bytes, bytes | bytearray):
            size_kb = len(img_bytes) / 1024
            if size_kb < 50:
                return 400  # Small icon/thumbnail
            if size_kb < 500:
                return 1200  # Typical screenshot
            return 1600  # Large/high-res image

        return 1200  # Default estimate

    def _count_tool_calls(self, tool_calls: list[dict[str, Any]]) -> int:
        """Count tokens in tool calls."""
        total = 0
        for call in tool_calls:
            total += 4  # Tool call overhead

            if "function" in call:
                func = call["function"] or {}
                total += self.count_text(coerce_countable_text(func.get("name")))
                total += self.count_text(coerce_countable_text(func.get("arguments")))

            if "id" in call:
                total += self.count_text(coerce_countable_text(call["id"]))

        return total

    def _count_function_call(self, function_call: dict[str, Any]) -> int:
        """Count tokens in legacy function call."""
        total = 4  # Function call overhead
        total += self.count_text(coerce_countable_text(function_call.get("name")))
        total += self.count_text(coerce_countable_text(function_call.get("arguments")))
        return total

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Optional method - not all backends support encoding.
        Default implementation raises NotImplementedError.

        Args:
            text: Text to encode.

        Returns:
            List of token IDs.

        Raises:
            NotImplementedError: If encoding is not supported.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support encoding")

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Optional method - not all backends support decoding.
        Default implementation raises NotImplementedError.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text.

        Raises:
            NotImplementedError: If decoding is not supported.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support decoding")


class _DelegatingBlockCounter(BaseTokenizer):
    """Adapter exposing :meth:`BaseTokenizer._count_content_parts` to non-subclasses.

    The provider token counters in ``headroom/providers/`` are not
    ``BaseTokenizer`` subclasses, and each grew its own shortened content-block
    walker that handled only the shapes its provider was expected to send. The
    result was that every one of them priced most modern blocks at ~0: measured
    on a 6,800-char block, ``OpenAITokenCounter`` returned 8 tokens for
    ``tool_result``/``thinking``/``document``/``mcp_tool_result`` and — its own
    Responses shapes — ``output_text``/``refusal``; ``AnthropicTokenCounter``
    returned 7 for ``thinking``/``document``, which are Anthropic's own.

    Rather than add a fifth partial walker, this lets them borrow the audited one.
    It is image-safe (base64 blobs get a pixel-based estimate instead of being
    serialized and priced as text) and bounds oversized blobs, which a naive
    ``count_text(str(block))`` catch-all does not.
    """

    __slots__ = ("_count_text_fn",)

    def __init__(self, count_text_fn: Callable[[str], int]) -> None:
        self._count_text_fn = count_text_fn

    def count_text(self, text: str) -> int:
        return self._count_text_fn(text)


def count_content_blocks(parts: list[Any], count_text_fn: Callable[[str], int]) -> int:
    """Count a multi-part content list using *count_text_fn* for text.

    Shared entry point for provider counters. See
    :class:`_DelegatingBlockCounter` for why this exists.
    """
    return _DelegatingBlockCounter(count_text_fn)._count_content_parts(parts)
