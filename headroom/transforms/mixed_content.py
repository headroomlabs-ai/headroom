"""Pure mixed-content parsing helpers for the content router."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .content_detector import ContentType


@dataclass
class ContentSection:
    """A typed section of content."""

    content: str
    content_type: ContentType
    language: str | None = None
    start_line: int = 0
    end_line: int = 0
    is_code_fence: bool = False


_CODE_FENCE_PATTERN = re.compile(r"^```(\w*)\s*$", re.MULTILINE)
_JSON_BLOCK_START = re.compile(r"^\s*[\[{]", re.MULTILINE)
_SEARCH_RESULT_PATTERN = re.compile(r"^\S+:\d+:", re.MULTILINE)
_PROSE_PATTERN = re.compile(r"[A-Z][a-z]+\s+\w+\s+\w+")


def is_mixed_content(content: str) -> bool:
    """Detect if content contains multiple distinct content types."""
    return sum(mixed_content_indicators(content).values()) >= 2


def mixed_content_indicators(content: str) -> dict[str, bool]:
    """Return the individual signals used to classify mixed content."""
    return {
        "has_code_fences": bool(_CODE_FENCE_PATTERN.search(content)),
        "has_json_blocks": bool(_JSON_BLOCK_START.search(content)),
        "has_embedded_json_with_text": _has_valid_json_block_with_text(content),
        "has_prose": len(_PROSE_PATTERN.findall(content)) > 5,
        "has_search_results": bool(_SEARCH_RESULT_PATTERN.search(content)),
    }


def _has_valid_json_block_with_text(content: str) -> bool:
    """Return true when prose or log text wraps a valid JSON block."""
    lines = content.split("\n")

    for index, line in enumerate(lines):
        if not line.strip().startswith(("[", "{")):
            continue

        json_content, end_index = _extract_json_block(lines, index)
        if json_content is None:
            continue

        try:
            json.loads(json_content)
        except (TypeError, ValueError):
            continue

        leading_text = "\n".join(lines[:index]).strip()
        trailing_text = "\n".join(lines[end_index + 1 :]).strip()
        if leading_text or trailing_text:
            return True

    return False


def split_into_sections(content: str) -> list[ContentSection]:
    """Parse mixed content into typed sections."""
    sections: list[ContentSection] = []
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]

        if match := _CODE_FENCE_PATTERN.match(line):
            language = match.group(1) or "unknown"
            code_lines = []
            start_line = i
            i += 1

            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1

            sections.append(
                ContentSection(
                    content="\n".join(code_lines),
                    content_type=ContentType.SOURCE_CODE,
                    language=language,
                    start_line=start_line,
                    end_line=i,
                    is_code_fence=True,
                )
            )
            i += 1
            continue

        if line.strip().startswith(("[", "{")):
            json_content, end_i = _extract_json_block(lines, i)
            if json_content:
                sections.append(
                    ContentSection(
                        content=json_content,
                        content_type=ContentType.JSON_ARRAY,
                        start_line=i,
                        end_line=end_i,
                    )
                )
                i = end_i + 1
                continue

        if _SEARCH_RESULT_PATTERN.match(line):
            search_lines = []
            start_line = i
            while i < len(lines) and _SEARCH_RESULT_PATTERN.match(lines[i]):
                search_lines.append(lines[i])
                i += 1
            sections.append(
                ContentSection(
                    content="\n".join(search_lines),
                    content_type=ContentType.SEARCH_RESULTS,
                    start_line=start_line,
                    end_line=i - 1,
                )
            )
            continue

        text_lines = [line]
        start_line = i
        i += 1

        while i < len(lines):
            next_line = lines[i]
            if (
                _CODE_FENCE_PATTERN.match(next_line)
                or next_line.strip().startswith(("[", "{"))
                or _SEARCH_RESULT_PATTERN.match(next_line)
            ):
                break
            text_lines.append(next_line)
            i += 1

        text_content = "\n".join(text_lines)
        if text_content.strip():
            sections.append(
                ContentSection(
                    content=text_content,
                    content_type=ContentType.PLAIN_TEXT,
                    start_line=start_line,
                    end_line=i - 1,
                )
            )

    return sections


def _extract_json_block(lines: list[str], start: int) -> tuple[str | None, int]:
    """Extract a complete JSON object or array block from line-oriented content."""
    bracket_count = 0
    brace_count = 0
    json_lines = []
    in_string = False
    escaped = False

    for i in range(start, len(lines)):
        line = lines[i]
        json_lines.append(line)

        for ch in line:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                if in_string:
                    escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                bracket_count += 1
            elif ch == "]":
                bracket_count -= 1
            elif ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1

        if bracket_count <= 0 and brace_count <= 0 and json_lines:
            return "\n".join(json_lines), i

    return None, start
