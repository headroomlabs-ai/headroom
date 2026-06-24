"""Markdown memory file parsers for various foundational model formats.

Parses markdown files into structured sections that can be imported into
Headroom's semantic memory system. Supports:
- Claude Code MEMORY.md format (## headers + bullet facts)
- ChatGPT flat fact list (one fact per line)
- Generic structured markdown (any # headers + content)

All parsers are pure functions with no external dependencies.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedSection:
    """A section parsed from a markdown memory file.

    Represents one logical chunk: a heading and its bullet content.
    """

    heading: str  # The heading text (empty for top-level/no heading)
    heading_level: int  # 1-6, or 0 for no heading
    content: str  # The full text content of this section
    facts: list[str]  # Individual facts (one per bullet)
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash and self.content:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()


@dataclass
class ParsedFile:
    """Complete parsed representation of a memory file."""

    path: str
    format: str  # "claude_code", "chatgpt", "generic"
    sections: list[ParsedSection]
    file_hash: str
    raw_content: str


# ---------------------------------------------------------------------------
# Entity extraction (lightweight, no LLM)
# ---------------------------------------------------------------------------

# Common words to skip when extracting capitalized phrases
_STOP_WORDS = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "When",
        "Where",
        "What",
        "Which",
        "Who",
        "How",
        "Why",
        "And",
        "But",
        "For",
        "Not",
        "With",
        "From",
        "Into",
        "Over",
        "Under",
        "After",
        "Before",
        "Between",
        "Through",
        "During",
        "About",
        "Against",
        "Each",
        "Every",
        "Some",
        "Any",
        "All",
        "Most",
        "Other",
        "Such",
        "Only",
        "Also",
        "Very",
        "Just",
        "Then",
        "Than",
        "Both",
        "Even",
        "Still",
        "Here",
        "There",
        "Once",
        "Note",
        "See",
        "Use",
        "New",
        "Key",
        "Important",
        "Direct",
        "Native",
        "Managed",
        "Full",
        "Main",
        "Relevant",
    }
)


def extract_entities_from_text(text: str) -> list[str]:
    """Extract likely entity names from text using heuristics.

    Looks for:
    - Bold text (**entity**)
    - Capitalized multi-word sequences (Proper Nouns)
    - Text before colons in "Key: Value" patterns

    Returns deduplicated list of entity names.
    """
    entities: set[str] = set()

    # 1. Bold text: **entity**
    for match in re.finditer(r"\*\*([^*]+)\*\*", text):
        entity = match.group(1).strip()
        if entity and len(entity) < 100:
            entities.add(entity)

    # 2. Capitalized multi-word sequences (2+ words, both capitalized)
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
        phrase = match.group(1).strip()
        words = phrase.split()
        # Skip if first word is a stop word
        if words[0] not in _STOP_WORDS and len(phrase) < 80:
            entities.add(phrase)

    # 3. Single capitalized words that look like proper nouns / tech names
    # Must be at least 2 chars, not at start of sentence
    for match in re.finditer(r"(?<=[a-z.,;:]\s)([A-Z][a-zA-Z0-9]+)\b", text):
        word = match.group(1)
        if word not in _STOP_WORDS and len(word) >= 2:
            entities.add(word)

    # 4. CamelCase or ALL_CAPS identifiers (tech/code names)
    for match in re.finditer(r"\b([A-Z][a-z]+[A-Z][a-zA-Z]*)\b", text):
        entities.add(match.group(1))
    for match in re.finditer(r"\b([A-Z][A-Z_]{2,})\b", text):
        word = match.group(1)
        if word not in {"THE", "AND", "BUT", "FOR", "NOT", "WITH", "FROM"}:
            entities.add(word)

    return sorted(entities)


def extract_relationships_from_section(
    section: ParsedSection,
) -> list[dict[str, str]]:
    """Extract relationships from a parsed section using pattern matching.

    Looks for patterns like:
    - "X: Y" -> (X, "is", Y)
    - "X uses Y" -> (X, "uses", Y)
    - "X at Y" -> (X, "located_at", Y)

    Returns list of {"source": ..., "relationship": ..., "destination": ...}
    """
    relationships: list[dict[str, str]] = []

    # Pattern: **Key**: Value (common in Claude Code MEMORY.md)
    for match in re.finditer(r"\*\*([^*]+)\*\*:\s*(.+?)(?:\n|$)", section.content):
        key = match.group(1).strip()
        value = match.group(2).strip()
        if key and value and len(key) < 80 and len(value) < 200:
            relationships.append({"source": key, "relationship": "is", "destination": value})

    # Pattern: "X uses Y", "X with Y", "X via Y"
    verb_patterns = [
        (r"(\w+(?:\s+\w+)?)\s+uses\s+(\w+(?:\s+\w+)?)", "uses"),
        (r"(\w+(?:\s+\w+)?)\s+requires\s+(\w+(?:\s+\w+)?)", "requires"),
        (r"(\w+(?:\s+\w+)?)\s+integrates\s+with\s+(\w+(?:\s+\w+)?)", "integrates_with"),
        (r"(\w+(?:\s+\w+)?)\s+depends\s+on\s+(\w+(?:\s+\w+)?)", "depends_on"),
    ]
    for pattern, rel_type in verb_patterns:
        for match in re.finditer(pattern, section.content, re.IGNORECASE):
            source = match.group(1).strip()
            dest = match.group(2).strip()
            if source and dest:
                relationships.append(
                    {"source": source, "relationship": rel_type, "destination": dest}
                )

    return relationships


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_format(content: str) -> str:
    """Auto-detect the markdown memory format.

    Returns:
        "claude_code", "chatgpt", or "generic"
    """
    lines = content.strip().splitlines()
    if not lines:
        return "generic"

    has_h2_headers = False
    has_bullets = False
    short_line_count = 0
    total_lines = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        total_lines += 1

        if stripped.startswith("## "):
            has_h2_headers = True
        if stripped.startswith("- "):
            has_bullets = True
        if len(stripped) < 120:
            short_line_count += 1

    # Claude Code: ## headers + bullets, often with **bold** keys
    if has_h2_headers and has_bullets:
        return "claude_code"

    # ChatGPT: mostly short lines, no headers, looks like flat facts
    if total_lines > 0 and not has_h2_headers:
        short_ratio = short_line_count / total_lines
        if short_ratio > 0.8 and total_lines >= 2:
            return "chatgpt"

    return "generic"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _compute_file_hash(content: str) -> str:
    """Compute SHA-256 hash of file content."""
    return hashlib.sha256(content.encode()).hexdigest()


def parse_claude_code_memory(content: str, file_path: str = "") -> ParsedFile:
    """Parse Claude Code MEMORY.md format.

    Structure: markdown with # and ## headers, bullet lists with optional **bold** keys.

    Example:
        # Project Memory

        ## Project Overview
        - **Headroom**: Context optimization layer
        - **Repos**: OSS at ~/claude-projects/headroom

        ## Key Architecture
        - 186 Python files, 34 packages
    """
    file_hash = _compute_file_hash(content)
    sections: list[ParsedSection] = []

    current_heading = ""
    current_level = 0
    current_lines: list[str] = []

    def flush_section() -> None:
        if not current_lines and not current_heading:
            return
        section_content = "\n".join(current_lines).strip()
        if not section_content and not current_heading:
            return

        facts = []
        for line in current_lines:
            stripped = line.strip()
            if stripped.startswith("- "):
                fact = stripped[2:].strip()
                if fact:
                    facts.append(fact)

        entities = extract_entities_from_text(section_content) if section_content else []

        sections.append(
            ParsedSection(
                heading=current_heading,
                heading_level=current_level,
                content=section_content,
                facts=facts,
                entities=entities,
                metadata={"source_format": "claude_code"},
            )
        )

    for line in content.splitlines():
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if header_match:
            flush_section()
            current_level = len(header_match.group(1))
            current_heading = header_match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    flush_section()

    return ParsedFile(
        path=file_path,
        format="claude_code",
        sections=sections,
        file_hash=file_hash,
        raw_content=content,
    )


def parse_chatgpt_facts(content: str, file_path: str = "") -> ParsedFile:
    """Parse ChatGPT-style flat fact list.

    Structure: one fact per line, optionally prefixed with -.

    Example:
        User prefers Python
        User works at Netflix
        - User likes dark mode
    """
    file_hash = _compute_file_hash(content)
    facts: list[str] = []

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Strip optional bullet prefix
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        if stripped:
            facts.append(stripped)

    all_content = "\n".join(facts)
    entities = extract_entities_from_text(all_content) if facts else []

    sections = (
        [
            ParsedSection(
                heading="",
                heading_level=0,
                content=all_content,
                facts=facts,
                entities=entities,
                metadata={"source_format": "chatgpt"},
            )
        ]
        if facts
        else []
    )

    return ParsedFile(
        path=file_path,
        format="chatgpt",
        sections=sections,
        file_hash=file_hash,
        raw_content=content,
    )


def parse_generic_markdown(content: str, file_path: str = "") -> ParsedFile:
    """Parse generic structured markdown.

    Handles any markdown with #-###### headers and content below.
    Groups content by nearest heading.
    """
    file_hash = _compute_file_hash(content)
    sections: list[ParsedSection] = []

    current_heading = ""
    current_level = 0
    current_lines: list[str] = []

    def flush_section() -> None:
        if not current_lines and not current_heading:
            return
        section_content = "\n".join(current_lines).strip()
        if not section_content and not current_heading:
            return

        facts = []
        for line in current_lines:
            stripped = line.strip()
            if stripped.startswith("- "):
                fact = stripped[2:].strip()
                if fact:
                    facts.append(fact)
            elif stripped and not stripped.startswith("#"):
                # Non-bullet, non-empty lines are also facts in generic mode
                facts.append(stripped)

        entities = extract_entities_from_text(section_content) if section_content else []

        sections.append(
            ParsedSection(
                heading=current_heading,
                heading_level=current_level,
                content=section_content,
                facts=facts,
                entities=entities,
                metadata={"source_format": "generic"},
            )
        )

    for line in content.splitlines():
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if header_match:
            flush_section()
            current_level = len(header_match.group(1))
            current_heading = header_match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    flush_section()

    return ParsedFile(
        path=file_path,
        format="generic",
        sections=sections,
        file_hash=file_hash,
        raw_content=content,
    )


def parse_markdown(
    content: str,
    file_path: str = "",
    format: str | None = None,
) -> ParsedFile:
    """Parse a markdown memory file, auto-detecting format if not specified.

    Args:
        content: The markdown file content.
        file_path: Path to the source file (for metadata).
        format: Force a specific format ("claude_code", "chatgpt", "generic").
                If None, auto-detects.

    Returns:
        ParsedFile with structured sections.
    """
    if format is None or format == "auto":
        format = detect_format(content)

    if format == "claude_code":
        return parse_claude_code_memory(content, file_path)
    elif format == "chatgpt":
        return parse_chatgpt_facts(content, file_path)
    else:
        return parse_generic_markdown(content, file_path)
