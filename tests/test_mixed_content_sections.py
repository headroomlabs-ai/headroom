import pytest

from headroom.transforms.content_detector import ContentType
from headroom.transforms.mixed_content import (
    _extract_json_block,
    is_mixed_content,
    split_into_sections,
)


def test_mixed_content_detection_requires_multiple_signals():
    prose = "\n".join(
        [
            "First sentence has enough words to count.",
            "Second sentence has enough words to count.",
            "Third sentence has enough words to count.",
            "Fourth sentence has enough words to count.",
            "Fifth sentence has enough words to count.",
            "Sixth sentence has enough words to count.",
        ]
    )

    assert is_mixed_content(prose) is False
    assert is_mixed_content(f"{prose}\n```python\nprint('x')\n```") is True


def test_split_into_sections_preserves_typed_boundaries():
    content = "\n".join(
        [
            "Intro text",
            "```python",
            "print('x')",
            "```",
            '[{"id": 1}]',
            "src/app.py:10:print('x')",
        ]
    )

    sections = split_into_sections(content)

    assert [section.content_type for section in sections] == [
        ContentType.PLAIN_TEXT,
        ContentType.SOURCE_CODE,
        ContentType.JSON_ARRAY,
        ContentType.SEARCH_RESULTS,
    ]
    assert sections[1].language == "python"
    assert sections[1].content == "print('x')"
    assert sections[1].is_code_fence is True
    assert sections[2].content == '[{"id": 1}]'
    assert sections[3].start_line == 5


@pytest.mark.parametrize(
    ("info_string", "language"),
    [
        ("c++", "c++"),
        ("c#", "c#"),
        ("objective-c", "objective-c"),
        ("js {1,3}", "js"),
        ('python title="app.py"', "python"),
        (" c++", "c++"),
        ("python\r", "python"),
    ],
)
def test_split_accepts_fence_info_strings_beyond_word_characters(info_string, language):
    content = "\n".join(
        [
            "Here is the header file we wrote today.",
            f"```{info_string}",
            "int add(int a, int b);",
            "```",
            "Please review this carefully and tell me what you think.",
            '{"name": "demo", "version": 1}',
        ]
    )

    sections = split_into_sections(content)

    assert [section.content_type for section in sections] == [
        ContentType.PLAIN_TEXT,
        ContentType.SOURCE_CODE,
        ContentType.PLAIN_TEXT,
        ContentType.JSON_ARRAY,
    ]
    assert sections[1].language == language
    assert sections[1].content == "int add(int a, int b);"
    assert sections[1].is_code_fence is True
    assert sections[2].content == "Please review this carefully and tell me what you think."


def test_fence_pattern_stays_linear_on_a_long_unclosed_fence_line():
    import time

    from headroom.transforms.mixed_content import _CODE_FENCE_PATTERN

    line = "```" + "a" * 50_000 + "`"
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        assert _CODE_FENCE_PATTERN.match(line) is None
        best = min(best, time.perf_counter() - start)
    # Linear: well under 1 ms here. A quadratic pattern takes seconds, so a
    # generous budget still catches it on a loaded runner.
    assert best < 0.5, f"{best * 1000:.1f}ms on a 50,000-character fence line"


def test_split_carves_grep_context_lines_into_search_sections():
    """Regression (#3580): prose + ``grep -A`` output keeps code out of text.

    Context lines (``path-NN-content``) must join the SEARCH_RESULTS section
    instead of stranding source code in a PLAIN_TEXT section, where the
    word-dropping prose compressor would corrupt it. ``--`` group separators
    ride along as ordinary text lines.
    """
    content = "\n".join(
        [
            "Some intro prose explaining the search.",
            "src/main.py-40-    context before",
            "src/main.py:42:def process_data(items):",
            "src/main.py-43-    context after",
            "--",
            "src/other.py:12:    return True",
            "Trailing prose after the results.",
        ]
    )

    sections = split_into_sections(content)

    # The ``--`` group separator breaks the run, so the two grep groups keep
    # their own SEARCH_RESULTS sections instead of merging through it.
    assert [section.content_type for section in sections] == [
        ContentType.PLAIN_TEXT,
        ContentType.SEARCH_RESULTS,
        ContentType.PLAIN_TEXT,
        ContentType.SEARCH_RESULTS,
        ContentType.PLAIN_TEXT,
    ]
    assert "src/main.py-40-    context before" in sections[1].content
    assert "src/main.py:42:def process_data(items):" in sections[1].content
    assert "src/main.py-43-    context after" in sections[1].content
    assert "src/other.py:12:    return True" in sections[3].content


def test_extract_json_block_ignores_delimiters_inside_strings():
    lines = [
        "[",
        '  {"path": "a]b", "message": "keep {literal} braces"},',
        '  {"path": "c"}',
        "]",
    ]

    block, end_line = _extract_json_block(lines, 0)

    assert end_line == 3
    assert block == "\n".join(lines)
