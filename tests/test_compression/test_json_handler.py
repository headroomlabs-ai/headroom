"""Tests for JSON structure handler."""

import json

import pytest

from headroom.compression.handlers.json_handler import (
    JSONStructureHandler,
    JSONTokenType,
    extract_json_schema,
)


class TestJSONStructureHandler:
    """Tests for JSONStructureHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        return JSONStructureHandler()

    def test_can_handle_json_object(self, handler):
        """Test detection of JSON objects."""
        assert handler.can_handle('{"key": "value"}') is True

    def test_can_handle_json_array(self, handler):
        """Test detection of JSON arrays."""
        assert handler.can_handle('[{"id": 1}, {"id": 2}]') is True

    def test_cannot_handle_invalid_json(self, handler):
        """Test rejection of invalid JSON."""
        assert handler.can_handle("not json") is False
        assert handler.can_handle('{"unclosed": ') is False

    def test_cannot_handle_plain_text(self, handler):
        """Test rejection of plain text."""
        assert handler.can_handle("Hello, world!") is False

    def test_preserves_keys(self, handler):
        """Test that JSON keys are marked as structural."""
        content = '{"name": "Alice", "age": 30}'
        result = handler.get_mask(content)

        # Find the key positions in the mask
        # "name" should be preserved (with quotes)
        name_start = content.index('"name"')
        name_end = name_start + len('"name"')

        for i in range(name_start, name_end):
            assert result.mask.mask[i] is True, f"Key char at {i} should be preserved"

    def test_preserves_brackets(self, handler):
        """Test that brackets are preserved."""
        content = '{"items": [1, 2, 3]}'
        result = handler.get_mask(content)

        # Find bracket positions
        for i, char in enumerate(content):
            if char in "{}[]":
                assert result.mask.mask[i] is True, f"Bracket at {i} should be preserved"

    def test_preserves_booleans(self, handler):
        """Test that boolean values are preserved."""
        content = '{"active": true, "deleted": false}'
        result = handler.get_mask(content)

        # Find boolean positions
        true_start = content.index("true")
        false_start = content.index("false")

        for i in range(true_start, true_start + 4):
            assert result.mask.mask[i] is True
        for i in range(false_start, false_start + 5):
            assert result.mask.mask[i] is True

    def test_preserves_null(self, handler):
        """Test that null values are preserved."""
        content = '{"value": null}'
        result = handler.get_mask(content)

        null_start = content.index("null")
        for i in range(null_start, null_start + 4):
            assert result.mask.mask[i] is True

    def test_preserves_short_strings(self, handler):
        """Test that short string values are preserved."""
        handler = JSONStructureHandler(
            preserve_short_values=True,
            short_value_threshold=10,
        )
        content = '{"status": "ok"}'
        result = handler.get_mask(content)

        # "ok" is short, should be preserved
        ok_start = content.index('"ok"')
        for i in range(ok_start, ok_start + 4):
            assert result.mask.mask[i] is True

    def test_compresses_long_strings(self, handler):
        """Test that long string values are marked compressible."""
        handler = JSONStructureHandler(
            preserve_short_values=True,
            short_value_threshold=5,
        )
        # Use a much longer string to ensure compression kicks in
        long_value = "x" * 200  # Repetitive content with low entropy
        content = f'{{"description": "{long_value}"}}'
        result = handler.get_mask(content)

        # The long repetitive value should have many compressible characters
        desc_start = content.index('"x')
        # Not all characters in the long string should be preserved
        preserved = sum(result.mask.mask[desc_start:])
        # At minimum, some characters should be compressible (not preserved)
        total_after_desc = len(content) - desc_start
        assert preserved < total_after_desc, (
            "Long repetitive strings should be partially compressible"
        )

    def test_preserves_high_entropy_values(self):
        """Test that high-entropy values (UUIDs) are preserved."""
        handler = JSONStructureHandler(
            preserve_high_entropy=True,
            entropy_threshold=0.8,
        )
        content = '{"id": "8f14e45f-ceea-4123-8f14-e45fceea4123"}'
        result = handler.get_mask(content)

        # UUID should be mostly preserved due to high entropy
        uuid_start = content.index('"8f14e45f')
        preserved = sum(result.mask.mask[uuid_start:])
        # Should preserve a significant portion
        assert preserved > 10

    def test_nested_object(self, handler):
        """Test handling of nested objects."""
        content = '{"user": {"name": "Alice", "email": "alice@example.com"}}'
        result = handler.get_mask(content)

        # Both "user" and "name" keys should be preserved
        assert result.mask.mask[content.index('"user"')] is True
        assert result.mask.mask[content.index('"name"')] is True

    def test_array_of_objects(self, handler):
        """Test handling of array of objects."""
        content = '[{"id": 1}, {"id": 2}]'
        result = handler.get_mask(content)

        # Both "id" keys should be preserved
        first_id = content.index('"id"')
        second_id = content.index('"id"', first_id + 1)

        assert result.mask.mask[first_id] is True
        assert result.mask.mask[second_id] is True

    def test_object_commas_do_not_advance_array_item_index(self):
        """Commas between keys inside an object must not count as array
        item separators.

        Regression: depth-keyed comma counting treated every comma under
        array_depth > 0 as an item boundary, so an array's FIRST object
        exhausted max_array_items_full within its own keys and later
        values were dropped despite being in item 0.
        """
        handler = JSONStructureHandler(
            preserve_short_values=True,
            short_value_threshold=20,
            max_array_items_full=3,
        )
        # Single array item (index 0) with 5 short values — all must
        # stay eligible for preservation.
        content = '[{"a": "valuea", "b": "valueb", "c": "valuec", "d": "valued", "e": "valuee"}]'
        result = handler.get_mask(content)

        for marker in ('"valuea"', '"valueb"', '"valuec"', '"valued"', '"valuee"'):
            start = content.index(marker)
            for i in range(start, start + len(marker)):
                assert result.mask.mask[i] is True, f"{marker} in array item 0 should be preserved"

    def test_array_items_past_threshold_compressed(self):
        """Items at index >= max_array_items_full are still compressed."""
        handler = JSONStructureHandler(
            preserve_short_values=True,
            short_value_threshold=20,
            max_array_items_full=2,
        )
        content = '[{"v": "itemzero"}, {"v": "itemone"}, {"v": "itemtwo"}]'
        result = handler.get_mask(content)

        # Items 0 and 1 preserved
        for marker in ('"itemzero"', '"itemone"'):
            start = content.index(marker)
            assert result.mask.mask[start + 1] is True, f"{marker} should be preserved"

        # Item 2 is past the threshold — value chars compressed
        start = content.index('"itemtwo"')
        inner = range(start + 1, start + len('"itemtwo"') - 1)
        assert not any(result.mask.mask[i] for i in inner), (
            "values in array items past max_array_items_full should be compressible"
        )

    def test_prose_not_preserved_by_entropy(self):
        """Long natural-language strings must not pass the entropy gate.

        Regression: self-normalized Shannon entropy scores English prose
        above the 0.85 threshold, so every description survived
        compression. Identifiers (no spaces) are still preserved.
        """
        handler = JSONStructureHandler(short_value_threshold=20)
        prose = "Context optimization layer for LLM applications with caching"
        uuid = "8f14e45f-ceea-4123-8f14-e45fceea4123"
        content = f'{{"description": "{prose}", "id": "{uuid}"}}'
        result = handler.get_mask(content)

        prose_start = content.index(prose)
        assert not any(result.mask.mask[i] for i in range(prose_start, prose_start + len(prose))), (
            "long prose value should be compressible"
        )

        uuid_start = content.index(uuid)
        assert all(result.mask.mask[i] for i in range(uuid_start, uuid_start + len(uuid))), (
            "UUID value should be preserved via entropy"
        )

    def test_short_value_threshold_excludes_quotes(self):
        """The short-value threshold measures the payload, not the token.

        Regression: len(token.text) included both quote characters, so a
        value of exactly threshold length was rejected (off-by-2).
        """
        handler = JSONStructureHandler(
            preserve_short_values=True,
            short_value_threshold=20,
            preserve_high_entropy=False,
        )
        exact = "x" * 20  # exactly at threshold — must be preserved
        content = f'{{"key": "{exact}"}}'
        result = handler.get_mask(content)

        start = content.index(exact)
        assert all(result.mask.mask[i] for i in range(start, start + len(exact))), (
            "value of exactly threshold length should be preserved"
        )

    def test_metadata_contains_key_count(self, handler):
        """Test that metadata includes key count."""
        content = '{"a": 1, "b": 2, "c": 3}'
        result = handler.get_mask(content)

        assert "key_count" in result.metadata
        assert result.metadata["key_count"] == 3

    def test_empty_json_object(self, handler):
        """Test handling of empty object."""
        content = "{}"
        result = handler.get_mask(content)

        # Should preserve the brackets
        assert result.mask.mask[0] is True  # {
        assert result.mask.mask[1] is True  # }

    def test_empty_json_array(self, handler):
        """Test handling of empty array."""
        content = "[]"
        result = handler.get_mask(content)

        assert result.mask.mask[0] is True  # [
        assert result.mask.mask[1] is True  # ]

    def test_whitespace_not_preserved(self, handler):
        """Test that whitespace is not preserved."""
        content = '{\n  "key": "value"\n}'
        result = handler.get_mask(content)

        # Newlines and spaces should not be preserved
        for i, char in enumerate(content):
            if char in " \n\t":
                assert result.mask.mask[i] is False

    def test_handler_name(self, handler):
        """Test handler name is correct."""
        assert handler.name == "json"


class TestJSONTokenization:
    """Tests for JSON tokenization."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        return JSONStructureHandler()

    def test_tokenizes_simple_object(self, handler):
        """Test tokenization of simple object."""
        content = '{"key": "value"}'
        tokens = handler._tokenize_json(content)

        # Should have: {, "key", :, "value", }
        types = [t.token_type for t in tokens]
        assert JSONTokenType.BRACKET in types
        assert JSONTokenType.KEY in types
        assert JSONTokenType.COLON in types
        assert JSONTokenType.STRING_VALUE in types

    def test_identifies_keys_vs_values(self, handler):
        """Test that keys and values are correctly identified."""
        content = '{"name": "Alice"}'
        tokens = handler._tokenize_json(content)

        key_tokens = [t for t in tokens if t.token_type == JSONTokenType.KEY]
        value_tokens = [t for t in tokens if t.token_type == JSONTokenType.STRING_VALUE]

        assert len(key_tokens) == 1
        assert key_tokens[0].text == '"name"'
        assert len(value_tokens) == 1
        assert value_tokens[0].text == '"Alice"'

    def test_tokenizes_numbers(self, handler):
        """Test tokenization of numbers."""
        content = '{"int": 42, "float": 3.14, "negative": -10, "exp": 1e5}'
        tokens = handler._tokenize_json(content)

        number_tokens = [t for t in tokens if t.token_type == JSONTokenType.NUMBER]
        numbers = [t.text for t in number_tokens]

        assert "42" in numbers
        assert "3.14" in numbers
        assert "-10" in numbers
        assert "1e5" in numbers

    def test_tokenizes_booleans_and_null(self, handler):
        """Test tokenization of booleans and null."""
        content = '{"a": true, "b": false, "c": null}'
        tokens = handler._tokenize_json(content)

        bool_tokens = [t for t in tokens if t.token_type == JSONTokenType.BOOLEAN]
        null_tokens = [t for t in tokens if t.token_type == JSONTokenType.NULL]

        assert len(bool_tokens) == 2
        assert len(null_tokens) == 1


class TestExtractJSONSchema:
    """Tests for extract_json_schema function."""

    def test_simple_object_schema(self):
        """Test schema extraction from simple object."""
        content = '{"name": "Alice", "age": 30}'
        schema = extract_json_schema(content)

        assert schema == {"name": "string", "age": "integer"}

    def test_nested_object_schema(self):
        """Test schema extraction from nested object."""
        content = '{"user": {"name": "Alice", "active": true}}'
        schema = extract_json_schema(content)

        assert schema == {"user": {"name": "string", "active": "boolean"}}

    def test_array_schema(self):
        """Test schema extraction from array."""
        content = '[{"id": 1}, {"id": 2}]'
        schema = extract_json_schema(content)

        assert schema == [{"id": "integer"}]

    def test_invalid_json_returns_empty(self):
        """Test that invalid JSON returns empty schema."""
        content = "not json"
        schema = extract_json_schema(content)

        assert schema == {}

    def test_mixed_types(self):
        """Test schema with mixed types."""
        content = '{"str": "hello", "num": 1.5, "bool": true, "null": null}'
        schema = extract_json_schema(content)

        assert schema == {
            "str": "string",
            "num": "number",
            "bool": "boolean",
            "null": "null",
        }


class TestJSONStructurePreservation:
    """Integration tests for JSON structure preservation."""

    def test_all_keys_visible_after_compression(self):
        """Test that all keys remain visible in compressed output."""
        handler = JSONStructureHandler()
        content = json.dumps(
            {
                "users": [
                    {"id": 1, "name": "Alice", "email": "alice@example.com"},
                    {"id": 2, "name": "Bob", "email": "bob@example.com"},
                ],
                "total": 2,
                "page": 1,
            }
        )

        result = handler.get_mask(content)

        # All keys should be preserved
        for key in ["users", "id", "name", "email", "total", "page"]:
            key_str = f'"{key}"'
            key_start = content.find(key_str)
            if key_start != -1:
                # At least the first character of the key should be preserved
                assert result.mask.mask[key_start] is True, f"Key {key} should be preserved"

    def test_large_array_handling(self):
        """Test handling of large arrays."""
        handler = JSONStructureHandler(max_array_items_full=3)

        # Create array with 100 items
        items = [{"id": i, "value": f"item_{i}_" + "x" * 50} for i in range(100)]
        content = json.dumps(items)

        result = handler.get_mask(content)

        # Should have reasonable preservation ratio
        # Not everything should be preserved for large arrays
        assert 0.1 < result.mask.preservation_ratio < 0.9
