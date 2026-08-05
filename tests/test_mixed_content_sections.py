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
