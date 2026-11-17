"""Tests for the memory extraction system prompt."""

from __future__ import annotations

from headroom.memory.extraction import EXTRACTION_SYSTEM_PROMPT


class TestExtractionSystemPrompt:
    """Tests for EXTRACTION_SYSTEM_PROMPT constant."""

    def test_prompt_is_non_empty_string(self):
        """Prompt should be a non-empty string."""
        assert isinstance(EXTRACTION_SYSTEM_PROMPT, str)
        assert len(EXTRACTION_SYSTEM_PROMPT) > 0

    def test_prompt_covers_all_extraction_types(self):
        """Prompt should cover facts, entities, and relationships."""
        prompt = EXTRACTION_SYSTEM_PROMPT

        assert "Facts" in prompt or "facts" in prompt
        assert "Entities" in prompt or "entities" in prompt
        assert "Relationships" in prompt or "relationships" in prompt

    def test_prompt_references_memory_save(self):
        """Prompt should mention memory_save tool."""
        prompt = EXTRACTION_SYSTEM_PROMPT

        assert "memory_save" in prompt

    def test_prompt_describes_extraction_purpose(self):
        """Prompt should explain why extraction is useful."""
        prompt = EXTRACTION_SYSTEM_PROMPT

        assert "memory" in prompt.lower()
        assert (
            "storage" in prompt.lower()
            or "saving" in prompt.lower()
            or "remember" in prompt.lower()
        )
