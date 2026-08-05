"""JSON structure handler.

Extracts structural elements from JSON content:
- Keys (navigational - tells LLM what fields exist)
- Brackets and colons (structural syntax)
- Short values like booleans, nulls, small numbers

Values (strings, long numbers, nested content) are marked as compressible.

This enables the LLM to see the full schema while values are compressed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from headroom.compression.handlers.base import BaseStructureHandler, HandlerResult
from headroom.compression.masks import EntropyScore, StructureMask


class JSONTokenType(Enum):
    """Types of JSON tokens for structure detection."""

    KEY = "key"  # Object key (always structural)
    STRING_VALUE = "string_value"  # String value (compressible)
    NUMBER = "number"  # Numeric value (preserve if short)
    BOOLEAN = "boolean"  # true/false (always structural)
    NULL = "null"  # null (always structural)
    BRACKET = "bracket"  # {, }, [, ] (always structural)
    COLON = "colon"  # : (always structural)
    COMMA = "comma"  # , (always structural)
    WHITESPACE = "whitespace"  # spaces, newlines (compressible)


@dataclass
class JSONToken:
    """A token in JSON content with its type and position."""

    text: str
    token_type: JSONTokenType
    start: int
    end: int

    @property
    def is_structural(self) -> bool:
        """Whether this token should be preserved."""
        return self.token_type in (
            JSONTokenType.KEY,
            JSONTokenType.BOOLEAN,
            JSONTokenType.NULL,
            JSONTokenType.BRACKET,
            JSONTokenType.COLON,
            JSONTokenType.COMMA,
        )


class JSONStructureHandler(BaseStructureHandler):
    """Handler for JSON content.

    Preserves:
    - All keys (navigational - LLM sees what fields exist)
    - Structural syntax ({, }, [, ], :, ,)
    - Booleans and nulls (small, semantically important)
    - High-entropy strings (UUIDs, hashes - identifiers)
    - Short numbers (often IDs or important values)

    Compresses:
    - Long string values (descriptions, content)
    - Whitespace
    - Redundant array elements (after first few)

    Example:
        >>> handler = JSONStructureHandler()
        >>> result = handler.get_mask('{"name": "Alice", "id": "usr_123"}')
        >>> # Keys "name" and "id" preserved, values may be compressed
    """

    def __init__(
        self,
        preserve_short_values: bool = True,
        short_value_threshold: int = 20,
        preserve_high_entropy: bool = True,
        entropy_threshold: float = 0.85,
        max_array_items_full: int = 3,  # Keep first N items fully
        max_number_digits: int = 10,  # Preserve numbers up to N digits
    ):
        """Initialize the JSON handler.

        Args:
            preserve_short_values: Preserve short string values.
            short_value_threshold: Max length for "short" values.
            preserve_high_entropy: Preserve high-entropy strings (UUIDs, etc.).
            entropy_threshold: Entropy threshold for preservation.
            max_array_items_full: Number of array items to keep in full.
            max_number_digits: Max digits for numbers to preserve (often IDs).
        """
        super().__init__(name="json")
        self.preserve_short_values = preserve_short_values
        self.short_value_threshold = short_value_threshold
        self.preserve_high_entropy = preserve_high_entropy
        self.entropy_threshold = entropy_threshold
        self.max_array_items_full = max_array_items_full
        self.max_number_digits = max_number_digits

    def can_handle(self, content: str) -> bool:
        """Check if content is valid JSON."""
        stripped = content.strip()
        if not stripped.startswith(("{", "[")):
            return False
        try:
            json.loads(stripped)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def _extract_mask(
        self,
        content: str,
        tokens: list[str],
        **kwargs: Any,
    ) -> HandlerResult:
        """Extract structure mask from JSON content.

        Args:
            content: JSON content.
            tokens: Character-level tokens.
            **kwargs: Additional options.

        Returns:
            HandlerResult with mask marking structural elements.
        """
        # Tokenize JSON to identify structure
        json_tokens = self._tokenize_json(content)

        # Build character-level mask
        mask = [False] * len(content)

        # Track containers so commas inside objects are not counted as
        # array item separators — only commas whose immediate enclosing
        # container is an array advance that array's item index. Values
        # nested in an object that is itself an array item inherit the
        # innermost enclosing array's index via array_item_stack[-1].
        container_stack: list[str] = []  # "[" or "{" per open container
        array_item_stack: list[int] = []  # item count per open array

        for token in json_tokens:
            # Track containers
            if token.token_type == JSONTokenType.BRACKET:
                if token.text in "{[":
                    container_stack.append(token.text)
                    if token.text == "[":
                        array_item_stack.append(0)
                elif token.text == "}":
                    if container_stack and container_stack[-1] == "{":
                        container_stack.pop()
                elif token.text == "]":
                    if container_stack and container_stack[-1] == "[":
                        container_stack.pop()
                        if array_item_stack:
                            array_item_stack.pop()

            # Count array items only at the array's own commas
            if (
                token.token_type == JSONTokenType.COMMA
                and container_stack
                and container_stack[-1] == "["
                and array_item_stack
            ):
                array_item_stack[-1] += 1

            # Determine if this token should be preserved
            preserve = self._should_preserve_token(
                token,
                len(array_item_stack),
                array_item_stack[-1] if array_item_stack else 0,
            )

            # Mark in mask
            if preserve:
                for i in range(token.start, min(token.end, len(mask))):
                    mask[i] = True

        return HandlerResult(
            mask=StructureMask(tokens=tokens, mask=mask),
            handler_name=self.name,
            confidence=1.0,
            metadata={
                "token_count": len(json_tokens),
                "key_count": sum(1 for t in json_tokens if t.token_type == JSONTokenType.KEY),
            },
        )

    def _should_preserve_token(
        self,
        token: JSONToken,
        array_depth: int,
        array_item_index: int,
    ) -> bool:
        """Determine if a token should be preserved.

        Args:
            token: The JSON token.
            array_depth: Current array nesting depth.
            array_item_index: Index of current item in array.

        Returns:
            True if token should be preserved.
        """
        # Always preserve structural tokens
        if token.is_structural:
            return True

        # Whitespace is never preserved
        if token.token_type == JSONTokenType.WHITESPACE:
            return False

        # Numbers: preserve short ones (often IDs)
        if token.token_type == JSONTokenType.NUMBER:
            return len(token.text) <= self.max_number_digits

        # String values: selective preservation
        if token.token_type == JSONTokenType.STRING_VALUE:
            # Check if we're past the max array items threshold
            if array_depth > 0 and array_item_index >= self.max_array_items_full:
                # In deep array, be more aggressive
                return False

            # Strip quotes once: thresholds apply to the payload, not
            # the token. Counting the quote characters made a "20-char
            # threshold" effectively 18 chars of value.
            value = token.text.strip('"')

            # Preserve short values
            if self.preserve_short_values and len(value) <= self.short_value_threshold:
                return True

            # Preserve high-entropy values (UUIDs, hashes)
            if self.preserve_high_entropy:
                # Entropy targets identifiers (UUIDs, hashes, API keys).
                # Self-normalized entropy also scores English prose >0.85,
                # so gate on the cheapest identifier signal: no spaces.
                if " " not in value:
                    score = EntropyScore.compute(value, self.entropy_threshold)
                    if score.should_preserve:
                        return True

            return False

        return False

    def _tokenize_json(self, content: str) -> list[JSONToken]:
        """Tokenize JSON content into typed tokens.

        This is a simple tokenizer that identifies JSON structure.
        It's not a full parser - just enough to identify keys vs values.

        Args:
            content: JSON content.

        Returns:
            List of JSONToken objects.
        """
        tokens: list[JSONToken] = []
        i = 0
        n = len(content)

        # Track if we're expecting a key (after { or ,)
        expect_key = False
        brace_stack: list[str] = []

        while i < n:
            char = content[i]

            # Whitespace
            if char in " \t\n\r":
                start = i
                while i < n and content[i] in " \t\n\r":
                    i += 1
                tokens.append(JSONToken(content[start:i], JSONTokenType.WHITESPACE, start, i))
                continue

            # Brackets
            if char in "{}[]":
                tokens.append(JSONToken(char, JSONTokenType.BRACKET, i, i + 1))
                if char == "{":
                    brace_stack.append("{")
                    expect_key = True
                elif char == "}":
                    if brace_stack and brace_stack[-1] == "{":
                        brace_stack.pop()
                    expect_key = False
                elif char == "[":
                    brace_stack.append("[")
                    expect_key = False
                elif char == "]":
                    if brace_stack and brace_stack[-1] == "[":
                        brace_stack.pop()
                i += 1
                continue

            # Colon
            if char == ":":
                tokens.append(JSONToken(char, JSONTokenType.COLON, i, i + 1))
                expect_key = False
                i += 1
                continue

            # Comma
            if char == ",":
                tokens.append(JSONToken(char, JSONTokenType.COMMA, i, i + 1))
                # After comma in object, expect key
                if brace_stack and brace_stack[-1] == "{":
                    expect_key = True
                i += 1
                continue

            # String (key or value)
            if char == '"':
                start = i
                i += 1
                while i < n and content[i] != '"':
                    if content[i] == "\\":
                        # Clamp: a trailing backslash at EOF must not
                        # step past the buffer.
                        i = min(i + 2, n)
                    else:
                        i += 1
                i += 1  # Include closing quote

                text = content[start:i]

                # Determine if this is a key or value
                # Look ahead for colon (skipping whitespace)
                j = i
                while j < n and content[j] in " \t\n\r":
                    j += 1

                is_key = j < n and content[j] == ":" and expect_key

                if is_key:
                    tokens.append(JSONToken(text, JSONTokenType.KEY, start, i))
                    expect_key = False
                else:
                    tokens.append(JSONToken(text, JSONTokenType.STRING_VALUE, start, i))

                continue

            # Number
            if char in "-0123456789":
                start = i
                # Match JSON number pattern
                if char == "-":
                    i += 1
                while i < n and content[i] in "0123456789":
                    i += 1
                if i < n and content[i] == ".":
                    i += 1
                    while i < n and content[i] in "0123456789":
                        i += 1
                if i < n and content[i] in "eE":
                    i += 1
                    if i < n and content[i] in "+-":
                        i += 1
                    while i < n and content[i] in "0123456789":
                        i += 1

                tokens.append(JSONToken(content[start:i], JSONTokenType.NUMBER, start, i))
                continue

            # Boolean or null
            if content[i : i + 4] == "true":
                tokens.append(JSONToken("true", JSONTokenType.BOOLEAN, i, i + 4))
                i += 4
                continue
            if content[i : i + 5] == "false":
                tokens.append(JSONToken("false", JSONTokenType.BOOLEAN, i, i + 5))
                i += 5
                continue
            if content[i : i + 4] == "null":
                tokens.append(JSONToken("null", JSONTokenType.NULL, i, i + 4))
                i += 4
                continue

            # Unknown character - skip
            i += 1

        return tokens


def extract_json_schema(content: str) -> dict[str, Any] | list[Any]:
    """Extract the schema (keys only) from JSON content.

    Useful for understanding the structure without the values.

    Args:
        content: JSON content.

    Returns:
        Schema dictionary with keys and types (no values).

    Example:
        >>> extract_json_schema('{"name": "Alice", "age": 30}')
        {'name': 'string', 'age': 'number'}
    """

    def _extract(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _extract(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if obj:
                return [_extract(obj[0])]  # Schema of first item
            return []
        elif isinstance(obj, str):
            return "string"
        elif isinstance(obj, bool):
            return "boolean"
        elif isinstance(obj, int):
            return "integer"
        elif isinstance(obj, float):
            return "number"
        elif obj is None:
            return "null"
        else:
            return "unknown"

    try:
        parsed = json.loads(content)
        result = _extract(parsed)
        if isinstance(result, dict):
            return cast(dict[str, Any], result)
        elif isinstance(result, list):
            return cast(list[Any], result)
        return {}
    except (json.JSONDecodeError, ValueError):
        return {}
