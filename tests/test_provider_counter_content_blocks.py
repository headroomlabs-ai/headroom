"""Provider token counters must price every content block, not just ``text``.

Each counter in ``headroom/providers/`` had grown its own shortened content-block
walker handling only the shapes its provider was expected to send. Everything else
fell through and contributed nothing. Measured on one 6,800-char block
(``count_messages`` of a single-block message, so 7-8 is message overhead alone):

    block type          OpenAI ctr   Anthropic ctr
    text (control)            3409            3748
    tool_result                  8            3748
    thinking                     8               7
    document                     8               7
    mcp_tool_result              8               7
    output_text                  8               7
    refusal                      8               7

Note each counter zeroed blocks from its OWN provider — ``output_text`` and
``refusal`` are OpenAI Responses shapes, ``thinking`` and ``document`` are
Anthropic's. And these counters are what the LIVE proxy pipelines use
(``proxy/server.py`` builds them with ``AnthropicProvider`` / ``OpenAIProvider``),
so this was the main request path, not an edge case.

The counters now delegate to ``count_content_blocks``, which reuses
``BaseTokenizer._count_content_parts``. That matters beyond coverage: a naive
``count_text(str(block))`` catch-all would serialize a base64 image and price it
as text — a 1MB screenshot reads as ~330K phantom tokens. The shared walker gives
media a pixel/byte-based estimate instead.
"""

from __future__ import annotations

import base64
import zlib

import pytest

from headroom.providers.anthropic import AnthropicProvider
from headroom.providers.openai import OpenAIProvider
from headroom.tokenizers import base
from headroom.tokenizers.base import count_content_blocks

_BIG = "x " * 3400  # ~6,800 chars


def _counters():
    return {
        "openai": OpenAIProvider().get_token_counter("gpt-4o"),
        "anthropic": AnthropicProvider(warn=False).get_token_counter("claude-sonnet-4-6"),
    }


@pytest.mark.parametrize(
    "block",
    [
        {"type": "tool_result", "tool_use_id": "t", "content": _BIG},
        {"type": "tool_use", "id": "t", "name": "grep", "input": {"pattern": _BIG}},
        {"type": "thinking", "thinking": _BIG},
        {"type": "document", "source": {"data": _BIG}},
        {"type": "mcp_tool_result", "content": _BIG},
        {"type": "output_text", "text": _BIG},
        {"type": "refusal", "refusal": _BIG},
        {"type": "search_result", "content": _BIG},
    ],
    ids=lambda b: str(b.get("type")),
)
def test_no_provider_counter_prices_a_large_block_at_zero(block: dict) -> None:
    """Every one of these returned 7-8 tokens — message overhead only."""
    message = {"role": "user", "content": [block]}
    for name, counter in _counters().items():
        got = counter.count_messages([message])
        assert got > 1_000, f"{name} priced a ~6,800-char {block['type']} block at {got}"


def test_base64_media_is_not_priced_as_text() -> None:
    """The reason a str(block) catch-all would have been the wrong fix."""
    image = {"type": "image", "source": {"type": "base64", "data": "A" * 200_000}}
    message = {"role": "user", "content": [image]}

    for name, counter in _counters().items():
        got = counter.count_messages([message])
        # 200KB of base64 as text would be ~50,000 tokens; the pixel estimate is 1600.
        assert got < 5_000, f"{name} priced a 200KB base64 image as text: {got}"
        assert got > 1_000, f"{name} ignored a declared image entirely: {got}"


def test_plain_text_blocks_are_unchanged() -> None:
    """The control: the shape both counters already handled must not move."""
    message = {"role": "user", "content": [{"type": "text", "text": _BIG}]}
    for name, counter in _counters().items():
        text_form = {"role": "user", "content": _BIG}
        block_form = counter.count_messages([message])
        # A text block and the equivalent string should agree closely.
        assert abs(block_form - counter.count_messages([text_form])) <= 5, name


def test_shared_walker_ignores_non_block_parts() -> None:
    """A bare int is not a block and must contribute nothing."""
    assert count_content_blocks([123], len) == 0
    assert count_content_blocks([], len) == 0


def test_shared_walker_counts_nested_tool_result_blocks() -> None:
    """A tool that returns blocks nests them; they must be walked, not serialized."""
    nested = {
        "type": "tool_result",
        "tool_use_id": "t",
        "content": [{"type": "text", "text": _BIG}],
    }
    flat = {"type": "tool_result", "tool_use_id": "t", "content": _BIG}
    assert abs(count_content_blocks([nested], len) - count_content_blocks([flat], len)) < 100


def _stream(data: bytes, entries: bytes = b"") -> bytes:
    return b"<< /Length %d%s >>\nstream\n%s\nendstream" % (len(data), entries, data)


def _png_encoded(data: bytes, width: int) -> bytes:
    """``data`` as PNG predictor rows of ``width`` bytes, cycling the five filter types."""
    out, prior = b"", bytes(width)
    for number, at in enumerate(range(0, len(data), width)):
        row, coded = data[at : at + width], bytearray()
        for i, value in enumerate(row):
            left, up, corner = row[i - 1] if i else 0, prior[i], prior[i - 1] if i else 0
            guess = left + up - corner
            pa, pb, pc = abs(guess - left), abs(guess - up), abs(guess - corner)
            paeth = left if pa <= pb and pa <= pc else up if pb <= pc else corner
            coded.append((value - (0, left, up, (left + up) // 2, paeth)[number % 5]) & 255)
        out += bytes([number % 5]) + coded
        prior = row
    return out


def _pdf_file(
    objects: dict[int, bytes | dict[int, bytes]],
    prior: bytes = b"",
    trailer: bytes = b"",
    predictor: bool = False,
) -> bytes:
    """A PDF with catalog 1, indexed by an xref table and trailer, or by an xref stream
    when a dict value packs those objects into an object stream or ``predictor`` asks
    for a Flate xref stream with PNG predictor rows. With ``prior``, an incremental
    update of that file; ``trailer`` opens the trailer's dict."""
    out = prior or b"%PDF-1.7\n"
    rows = {} if prior else {0: (0, 0, 65535)}
    for number, body in objects.items():
        if isinstance(body, dict):
            head, data = b"", b""
            for index, (packed, obj) in enumerate(body.items()):
                head, data = head + b"%d %d " % (packed, len(data)), data + obj + b"\n"
                rows[packed] = (2, number, index)
            entries = b" /Type /ObjStm /N %d /First %d /Filter /FlateDecode" % (
                len(body),
                len(head),
            )
            body = _stream(zlib.compress(head + data), entries)
        rows[number] = (1, len(out), 0)
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    prev = b" /Prev %s" % prior.split(b"startxref")[-1].split()[0] if prior else b""
    xref_at = len(out)
    if predictor or any(kind == 2 for kind, _, _ in rows.values()):
        rows[max(rows) + 1] = (1, xref_at, 0)
        index = b" ".join(b"%d 1" % number for number in sorted(rows))
        table = b"".join(
            bytes([kind]) + a.to_bytes(4, "big") + b.to_bytes(2, "big")
            for _, (kind, a, b) in sorted(rows.items())
        )
        entries = b" %s/Type /XRef /Size %d /W [1 4 2] /Index [%s] /Root 1 0 R%s" % (
            trailer,
            max(rows) + 1,
            index,
            prev,
        )
        if predictor:
            table = zlib.compress(_png_encoded(table, 7))
            entries += b" /Filter /FlateDecode /DecodeParms << /Columns 7 /Predictor 12 >>"
        out += b"%d 0 obj\n%s\nendobj\n" % (max(rows), _stream(table, entries))
    else:
        out += b"xref\n" + b"".join(
            b"%d 1\n%010d %05d %s \n" % (number, a, b, b"n" if kind else b"f")
            for number, (kind, a, b) in sorted(rows.items())
        )
        out += b"trailer\n<< %s/Size %d /Root 1 0 R%s >>\n" % (trailer, max(rows) + 1, prev)
    return out + b"startxref\n%d\n%%%%EOF\n" % xref_at


def _page_tree(page_count: int) -> dict[int, bytes]:
    """Catalog 1, page tree 2 and ``page_count`` blank pages from object 3 on."""
    kids = range(3, 3 + page_count)
    return {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [%s] /Count %d /MediaBox [0 0 612 792] >>"
        % (b" ".join(b"%d 0 R" % kid for kid in kids), page_count),
        **dict.fromkeys(kids, b"<< /Type /Page /Parent 2 0 R >>"),
    }


def _pdf(page_count: int, *, trailer: bool = True, pad: int = 1_000_000) -> bytes:
    """Blank pages plus ``pad`` bytes of image data; ``trailer=False`` cuts the file
    before its xref and trailer, as a truncated download would."""
    image = _stream(bytes(range(256)) * (pad // 256))
    pdf = _pdf_file({**_page_tree(page_count), 3 + page_count: image})
    return pdf if trailer else pdf[: pdf.rindex(b"xref")]


def _base64_document(raw: bytes) -> dict:
    data = base64.b64encode(raw).decode()
    return {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": data},
    }


def test_base64_pdf_document_is_priced_by_pages_not_as_text() -> None:
    """A 1 MB base64 PDF read as prose was ~907K tokens; 12 pages are ~18K."""
    document = _base64_document(_pdf(12))
    for placement in (
        [document],
        [{"type": "tool_result", "tool_use_id": "t", "content": [document]}],
    ):
        assert count_content_blocks(placement, len) == 12 * 1500
        for name, counter in _counters().items():
            got = counter.count_messages([{"role": "user", "content": placement}])
            assert 12 * 1500 <= got < 12 * 1500 + 100, f"{name} priced a 12-page PDF at {got}"


def test_pdf_page_count_falls_back_from_the_tree_to_the_size_estimate() -> None:
    # A truncated file has no xref to find its page tree through: ~3KB of file per page.
    truncated = _pdf(5, trailer=False)
    assert count_content_blocks([_base64_document(truncated)], len) == len(truncated) // 3000 * 1500
    # PDF 1.5+ keeps the page tree inside a compressed object stream.
    hidden = _pdf_file({10: _page_tree(7), 11: _stream(bytes(range(256)) * 4096)})
    assert count_content_blocks([_base64_document(hidden)], len) == 7 * 1500
    # Page tree hidden (compressed object streams): ~3KB of file per page.
    opaque = bytes(range(256)) * 4096
    assert count_content_blocks([_base64_document(opaque)], len) == len(opaque) // 3000 * 1500
    # Text that quotes PDF syntax is not a PDF (a Strands txt document): size estimate.
    quoted = b"Notes on /Type /Pages /Count 1 and /Type /Page objects.\n" + b"x" * 300_000
    strands = {"document": {"format": "txt", "name": "notes", "source": {"bytes": quoted}}}
    assert count_content_blocks([strands], len) == len(quoted) // 3000 * 1500
    # Undecodable base64 is still a document: one page, no exception.
    broken = {"type": "document", "source": {"type": "base64", "data": "not base64!"}}
    assert count_content_blocks([broken], len) == 1500


def test_only_the_page_trees_object_stream_is_inflated_within_the_budget(monkeypatch) -> None:
    fonts = dict.fromkeys(range(10, 30), b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pdf = _pdf_file({30: fonts, 31: _page_tree(7), 32: _stream(bytes(range(256)) * 4096)})
    inflates = []
    real = zlib.decompressobj
    monkeypatch.setattr(base.zlib, "decompressobj", lambda *a: inflates.append(1) or real(*a))
    assert count_content_blocks([_base64_document(pdf)], len) == 7 * 1500
    assert len(inflates) == 1  # the fonts' stream is never read
    # A budget the page tree's stream does not fit in gives the size estimate.
    monkeypatch.setattr(base, "_PDF_OBJECT_STREAM_MAX_BYTES", 600)
    assert count_content_blocks([_base64_document(pdf)], len) == len(pdf) // 3000 * 1500


def test_a_pdf_with_more_xref_rows_than_the_budget_gets_the_size_estimate(monkeypatch) -> None:
    pdf = _pdf(5)
    monkeypatch.setattr(base, "_PDF_MAX_OBJECTS", 4)
    assert count_content_blocks([_base64_document(pdf)], len) == len(pdf) // 3000 * 1500


def _one_page_objects(
    text: bytes, count: bytes = b"/Count 1", length: bytes | None = None
) -> dict[int, bytes]:
    """A minimal valid one-page PDF showing ``text``; ``count`` ends its page tree's dict.
    With ``length``, that is the content stream's whole dict and object 6 holds the
    stream's real length, for ``/Length 6 0 R``."""
    data = b"BT /F1 12 Tf 72 720 Td (%s) Tj ET" % text
    return {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /MediaBox [0 0 612 792] %s >>" % count,
        3: b"<< /Type /Page /Parent 2 0 R /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        4: _stream(data) if length is None else b"<< %s >>\nstream\n%s\nendstream" % (length, data),
        5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        **({} if length is None else {6: b"%d" % len(data)}),
    }


def _one_page_pdf(text: bytes, count: bytes) -> bytes:
    return _pdf_file(_one_page_objects(text, count))


@pytest.mark.parametrize(
    ("text", "count"),
    [
        (b"hello", b"/Count 1"),
        # Page text quoting page-tree syntax is content, not structure.
        (b"/Type /Pages /Count 999999", b"/Count 1"),
        (b"/Type /Pages /Count " + b"9" * 5000, b"/Count 1"),
        (b"2 0 obj << /Type /Pages /Count 999 >> endobj", b"/Count 1"),
        # Neither is a comment or a string inside the page tree's own dict.
        (b"hello", b"% /Count 999\n/Title (a (nested) /Count 999) /Count 1"),
        # A malformed or absurd count gives the size estimate (1 page at this size).
        (b"hello", b"/Count 1000000"),
        (b"hello", b"/Count " + b"9" * 5000),
        # A count in a dict, array or hex string nested in the tree's dict is not its own.
        (b"hello", b"/PieceInfo << /Example << /Private << /Count 9999 >> >> >> /Count 1"),
        (b"hello", b"/Private [<00ff> << /Count 9999 >>] /Count 1"),
    ],
    ids=[
        "plain",
        "quoted-tree",
        "long-number",
        "fake-object",
        "comment-string",
        "absurd",
        "long",
        "nested-dict",
        "nested-array",
    ],
)
def test_page_count_comes_only_from_the_page_tree_dict(text: bytes, count: bytes) -> None:
    assert base.BaseTokenizer._estimate_document_tokens(_one_page_pdf(text, count)) == 1500


_FAKE_TREE = b"endstream 2 0 obj << /Type /Pages /Count 9999 >> endobj"


@pytest.mark.parametrize(
    ("objects", "trailer"),
    [
        # The catalog nests a private /Pages ahead of its own.
        (
            {1: b"<< /PieceInfo << /Private << /Pages 7 0 R >> >> /Type /Catalog /Pages 2 0 R >>"},
            b"",
        ),
        # The trailer nests a private /Root ahead of its own.
        ({}, b"/Private << /Root 6 0 R >> "),
        # Laid out like an object stream redefining page tree 2, but /Type /ObjStm is
        # private data, not a key of the stream's own dict.
        (
            {
                8: _stream(
                    zlib.compress(b"2 0 << /Type /Pages /Count 9999 >>"),
                    b" /Filter /FlateDecode /Private << /Type /ObjStm /N 1 /First 4 >>",
                )
            },
            b"",
        ),
        # A private /Length 0 ahead of the stream's own would end the skip at the
        # stream's first byte, and its data would be walked as objects.
        (
            {
                8: b"<< /DecodeParms << /Length 0 >> /Length %d >>\nstream\n%s\nendstream"
                % (len(_FAKE_TREE), _FAKE_TREE)
            },
            b"",
        ),
    ],
    ids=["catalog-pages", "trailer-root", "object-stream", "stream-length"],
)
def test_root_pages_and_stream_keys_come_only_from_their_own_dicts(
    objects: dict[int, bytes], trailer: bytes
) -> None:
    decoys = {
        6: b"<< /Type /Catalog /Pages 7 0 R >>",
        7: b"<< /Type /Pages /Kids [] /Count 9999 >>",
    }
    pdf = _pdf_file({**_page_tree(1), **decoys, **objects}, trailer=trailer)
    assert base.BaseTokenizer._estimate_document_tokens(pdf) == 1500


def _packed(objects: dict[int, bytes], number: int) -> dict[int, bytes | dict[int, bytes]]:
    """``objects`` with ``number`` moved into object stream 7, indexed by an xref stream."""
    return {**{n: o for n, o in objects.items() if n != number}, 7: {number: objects[number]}}


_INDIRECT = _one_page_objects(_FAKE_TREE, length=b"/Length 6 0 R")
_LONE_CR = b"BT /F1 12 Tf 72 720 Td (hi) Tj ET endobj 2 0 obj << /Type /Pages /Count 9999 >> endobj"


@pytest.mark.parametrize(
    ("pdf", "pages"),
    [
        # JerrettDavis's case: object 6, read at its xref offset, ends the stream.
        (_pdf_file(_INDIRECT), 1),
        # Page text that also fakes object 6 with the length (24) of the quoted endstream.
        (
            _pdf_file(
                _one_page_objects(_FAKE_TREE + b" 6 0 obj 24 endobj", length=b"/Length 6 0 R")
            ),
            1,
        ),
        # Through an xref stream, and through /Prev to the section that lists object 6.
        (_pdf_file(_packed(_INDIRECT, 5)), 1),
        (_pdf_file({5: _INDIRECT[5]}, prior=_pdf_file(_INDIRECT)), 1),
        # A lone CR after the stream keyword still starts a stream, so bare tokens after
        # the text operators stay data.
        (
            _pdf_file(
                {
                    **_INDIRECT,
                    4: b"<< /Length 6 0 R >>\nstream\r%s\nendstream" % _LONE_CR,
                    6: b"%d" % len(_LONE_CR),
                }
            ),
            1,
        ),
        # The page count never reads a content stream, so neither does its length: one
        # packed in an object stream, a wrong one, a reference to no object, or none.
        (_pdf_file(_packed(_INDIRECT, 6)), 1),
        (_pdf_file({**_INDIRECT, 6: b"40"}), 1),
        (_pdf_file(_one_page_objects(_FAKE_TREE, length=b"/Length 9 0 R")), 1),
        (_pdf_file(_one_page_objects(_FAKE_TREE, length=b"")), 1),
    ],
    ids=[
        "indirect",
        "fake-length-object",
        "xref-stream",
        "prev-section",
        "lone-cr",
        "packed-length",
        "wrong-length",
        "dangling-length",
        "no-length",
    ],
)
def test_stream_data_is_never_walked_as_objects(pdf: bytes, pages: int) -> None:
    assert base._pdf_page_count(pdf) == pages


def _xref_chain(*sections: bytes) -> bytes:
    """A file of xref streams chained by /Prev, each of ``sections`` the end of one's dict
    and its stream, ready as an update's ``prior``."""
    out, prev = b"%PDF-1.7\n", b""
    for number, body in enumerate(sections, 100):
        at = len(out)
        out += b"%d 0 obj\n<< /Type /XRef /Size 999999 /W [1 4 2]%s%s\nendobj\n" % (
            number,
            prev,
            body,
        )
        prev = b" /Prev %d" % at
    return out + b"startxref\n%d\n%%%%EOF\n" % at


def _flate_rows(index: bytes, data: bytes) -> bytes:
    packed = zlib.compress(data, 9)
    return b" /Index [%s] /Filter /FlateDecode /Length %d >>\nstream\n%s\nendstream" % (
        index,
        len(packed),
        packed,
    )


def _priced_after(prior: bytes) -> int:
    return base._pdf_page_count(_pdf_file(_INDIRECT, prior=prior))


def test_xref_rows_share_one_budget_across_sections() -> None:
    # Two compressed sections of 100,000 entries each: 200,000 rows read is past the
    # 100,000 the whole traversal may read, so the size estimate stands in.
    rows = bytes(7 * 100_000)
    chain = _xref_chain(_flate_rows(b"0 100000", rows), _flate_rows(b"100000 100000", rows))
    assert _priced_after(chain) == 0
    # A normal update chain through a compressed section is well inside the budgets.
    assert _priced_after(_xref_chain(_flate_rows(b"0 1", bytes(7)))) == 1


def test_xref_streams_share_the_files_one_inflate_budget(monkeypatch) -> None:
    # 100 sections each inflating to 4 MB of rows: the first runs out the file's one
    # 4 MB budget, so the size estimate stands in and no other section is inflated.
    inflates = []
    real = zlib.decompressobj
    monkeypatch.setattr(base.zlib, "decompressobj", lambda *a: inflates.append(1) or real(*a))
    assert _priced_after(_xref_chain(*[_flate_rows(b"0 571428", bytes(4_000_000))] * 100)) == 0
    assert len(inflates) == 1


def test_xref_dicts_share_one_scan_budget_across_sections() -> None:
    # 300 table sections whose trailer dicts stay open, so each holds every later one:
    # reading the chain would rescan most of the file 300 times.
    section = b"xref\n0 0\ntrailer\n<< /Prev %010d " + b"/A " * 200
    width = len(section % 0)
    chain = b"%PDF-1.7\n" + b"".join(section % (9 + (i + 1) * width) for i in range(300))
    assert _priced_after(chain + b">> " * 300 + b"\nstartxref\n9\n%%EOF\n") == 0


def _unreferenced(pdf: bytes, obj: bytes) -> bytes:
    """``pdf`` (an xref table file) with ``obj`` added before its table, which does not list it."""
    at = pdf.rindex(b"\nxref\n") + 1
    xref_at = int(pdf.rsplit(b"startxref\n", 1)[1].split()[0])
    tail = pdf[at:].replace(b"startxref\n%d" % xref_at, b"startxref\n%d" % (xref_at + len(obj)))
    return pdf[:at] + obj + tail


_DECOY_TREE = b"<< /Type /Pages /Kids [3 0 R] /Count 9999 >>"
_DECOY_STREAM = _stream(
    zlib.compress(b"2 0 " + _DECOY_TREE), b" /Type /ObjStm /N 1 /First 4 /Filter /FlateDecode"
)


@pytest.mark.parametrize(
    "pdf",
    [
        # A later "2 0 obj" outside any stream that the xref does not list.
        _unreferenced(_one_page_pdf(b"hello", b"/Count 1"), b"2 0 obj\n%s\nendobj\n" % _DECOY_TREE),
        # A later object stream holding an object 2, unlisted or listed: either way the
        # xref puts object 2 at its own offset.
        _unreferenced(
            _one_page_pdf(b"hello", b"/Count 1"), b"9 0 obj\n%s\nendobj\n" % _DECOY_STREAM
        ),
        _pdf_file({**_one_page_objects(b"hello"), 9: _DECOY_STREAM}),
    ],
    ids=["object", "unlisted-object-stream", "listed-object-stream"],
)
def test_only_the_definition_the_xref_points_at_counts(pdf: bytes) -> None:
    assert base.BaseTokenizer._estimate_document_tokens(pdf) == 1500


def test_a_predictor_12_xref_stream_is_decoded_row_by_row() -> None:
    # Its rows cycle the five PNG filter types; the tree is top level, or packed.
    for pdf in (
        _pdf_file(_page_tree(12), predictor=True),
        _pdf_file({20: _page_tree(12)}, predictor=True),
    ):
        assert base.BaseTokenizer._estimate_document_tokens(pdf) == 12 * 1500


def test_incremental_update_prices_the_active_page_tree_not_an_obsolete_one() -> None:
    original = _pdf_file(_page_tree(5))
    # An edit keeps two pages under a new tree; the old tree (/Count 5) stays in the file.
    edited = _pdf_file(
        {
            1: b"<< /Type /Catalog /Pages 8 0 R >>",
            3: b"<< /Type /Page /Parent 8 0 R >>",
            4: b"<< /Type /Page /Parent 8 0 R >>",
            8: b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 /MediaBox [0 0 612 792] >>",
        },
        prior=original,
    )
    assert base.BaseTokenizer._estimate_document_tokens(original) == 5 * 1500
    assert base.BaseTokenizer._estimate_document_tokens(edited) == 2 * 1500


def test_text_and_content_document_sources_count_their_text() -> None:
    text = {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": _BIG},
    }
    assert count_content_blocks([text], len) == len(_BIG)
    content = {
        "type": "document",
        "source": {"type": "content", "content": [{"type": "text", "text": _BIG}]},
    }
    assert count_content_blocks([content], len) == len(_BIG)
    plain = {"type": "document", "source": {"type": "content", "content": _BIG}}
    assert count_content_blocks([plain], len) == len(_BIG)
    titled = {**text, "title": "Q3 report", "context": "board deck"}
    assert count_content_blocks([titled], len) == len(_BIG) + len("Q3 report") + len("board deck")


def test_url_and_file_document_sources_are_one_nominal_page() -> None:
    for source in (
        {"type": "url", "url": "https://example.com/a.pdf"},
        {"type": "file", "file_id": "file_1"},
    ):
        assert count_content_blocks([{"type": "document", "source": source}], len) == 1500
