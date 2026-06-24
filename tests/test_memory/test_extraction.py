"""Tests for memory extraction prompts and utilities.

Tests the extraction prompts, prompt generators, and tool schemas
used for inline fact/entity/relationship extraction.
"""

from __future__ import annotations

from headroom.memory.extraction import (
    CONVERSATION_EXTRACTION_PROMPT_BASIC,
    ENTITY_EXTRACTION_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    FACT_EXTRACTION_PROMPT,
    MEMORY_SAVE_TOOL_WITH_EXTRACTION,
    RELATIONSHIP_EXTRACTION_PROMPT,
    get_conversation_extraction_prompt,
    get_extraction_tools,
    get_memory_answer_prompt,
)

# =============================================================================
# Test Prompt Constants
# =============================================================================


class TestFactExtractionPrompt:
    """Tests for FACT_EXTRACTION_PROMPT constant."""

    def test_prompt_is_non_empty_string(self):
        """Prompt should be a non-empty string."""
        assert isinstance(FACT_EXTRACTION_PROMPT, str)
        assert len(FACT_EXTRACTION_PROMPT) > 0

    def test_prompt_contains_key_instructions(self):
        """Prompt should contain key extraction instructions."""
        prompt = FACT_EXTRACTION_PROMPT

        # Core principles
        assert "Comprehensiveness" in prompt
        assert "Attribution" in prompt
        assert "Specificity" in prompt
        assert "Self-contained" in prompt
        assert "Temporal grounding" in prompt

    def test_prompt_contains_extraction_categories(self):
        """Prompt should list what to extract."""
        prompt = FACT_EXTRACTION_PROMPT

        assert "Personal details" in prompt
        assert "Preferences" in prompt
        assert "Activities" in prompt or "hobbies" in prompt.lower()
        assert "Professional" in prompt
        assert "Events" in prompt
        assert "Plans" in prompt

    def test_prompt_contains_filtering_guidance(self):
        """Prompt should explain what NOT to extract."""
        prompt = FACT_EXTRACTION_PROMPT

        assert "WHAT NOT TO EXTRACT" in prompt
        assert "Greetings" in prompt
        assert "Transient" in prompt
        assert "Sensitive data" in prompt

    def test_prompt_has_good_bad_examples(self):
        """Prompt should include good/bad examples for clarity."""
        prompt = FACT_EXTRACTION_PROMPT

        assert "Good:" in prompt
        assert "Bad:" in prompt


class TestEntityExtractionPrompt:
    """Tests for ENTITY_EXTRACTION_PROMPT constant."""

    def test_prompt_is_non_empty_string(self):
        """Prompt should be a non-empty string."""
        assert isinstance(ENTITY_EXTRACTION_PROMPT, str)
        assert len(ENTITY_EXTRACTION_PROMPT) > 0

    def test_prompt_contains_entity_types(self):
        """Prompt should define common entity types."""
        prompt = ENTITY_EXTRACTION_PROMPT

        assert "person" in prompt
        assert "organization" in prompt
        assert "technology" in prompt
        assert "location" in prompt
        assert "project" in prompt

    def test_prompt_handles_self_references(self):
        """Prompt should explain how to handle self-references."""
        prompt = ENTITY_EXTRACTION_PROMPT

        # Should mention I/me/my handling
        assert "self-references" in prompt.lower() or "'I'" in prompt or "'me'" in prompt
        assert "user_id" in prompt

    def test_prompt_contains_example(self):
        """Prompt should include usage example."""
        prompt = ENTITY_EXTRACTION_PROMPT

        assert "Example:" in prompt
        assert "Input:" in prompt
        assert "Entities:" in prompt


class TestRelationshipExtractionPrompt:
    """Tests for RELATIONSHIP_EXTRACTION_PROMPT constant."""

    def test_prompt_is_non_empty_string(self):
        """Prompt should be a non-empty string."""
        assert isinstance(RELATIONSHIP_EXTRACTION_PROMPT, str)
        assert len(RELATIONSHIP_EXTRACTION_PROMPT) > 0

    def test_prompt_contains_guidelines(self):
        """Prompt should contain extraction guidelines."""
        prompt = RELATIONSHIP_EXTRACTION_PROMPT

        assert "Guidelines" in prompt
        assert "explicitly stated" in prompt.lower()

    def test_prompt_defines_relationship_format(self):
        """Prompt should define relationship format."""
        prompt = RELATIONSHIP_EXTRACTION_PROMPT

        assert "Relationship Format" in prompt
        assert "source" in prompt
        assert "relationship" in prompt
        assert "destination" in prompt

    def test_prompt_lists_common_relationship_types(self):
        """Prompt should list common relationship types."""
        prompt = RELATIONSHIP_EXTRACTION_PROMPT

        assert "works_at" in prompt
        assert "uses" in prompt
        assert "knows" in prompt
        assert "collaborates_with" in prompt or "reports_to" in prompt

    def test_prompt_prefers_timeless_relationships(self):
        """Prompt should prefer timeless relationship types."""
        prompt = RELATIONSHIP_EXTRACTION_PROMPT

        # Should prefer "works_at" over "started_working_at"
        assert "timeless" in prompt.lower()
        assert "works_at" in prompt and "started_working_at" in prompt

    def test_prompt_contains_example(self):
        """Prompt should include usage example."""
        prompt = RELATIONSHIP_EXTRACTION_PROMPT

        assert "Example:" in prompt
        assert "Relationships:" in prompt


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


class TestConversationExtractionPromptBasic:
    """Tests for CONVERSATION_EXTRACTION_PROMPT_BASIC preset."""

    def test_prompt_is_non_empty_string(self):
        """Preset prompt should be a non-empty string."""
        assert isinstance(CONVERSATION_EXTRACTION_PROMPT_BASIC, str)
        assert len(CONVERSATION_EXTRACTION_PROMPT_BASIC) > 0

    def test_prompt_is_generated_without_arguments(self):
        """Preset should match calling generator with no args."""
        expected = get_conversation_extraction_prompt()
        assert CONVERSATION_EXTRACTION_PROMPT_BASIC == expected


# =============================================================================
# Test get_conversation_extraction_prompt()
# =============================================================================


class TestGetConversationExtractionPrompt:
    """Tests for get_conversation_extraction_prompt() function."""

    def test_returns_string(self):
        """Function should return a string."""
        result = get_conversation_extraction_prompt()
        assert isinstance(result, str)

    def test_returns_non_empty_prompt(self):
        """Function should return non-empty prompt."""
        result = get_conversation_extraction_prompt()
        assert len(result) > 100  # Should be substantial

    def test_no_args_excludes_speaker_section(self):
        """Without speaker_names, should not include SPEAKERS section."""
        result = get_conversation_extraction_prompt()
        assert "SPEAKERS:" not in result

    def test_no_args_excludes_temporal_section(self):
        """Without context_date, should not include TEMPORAL CONTEXT."""
        result = get_conversation_extraction_prompt()
        assert "TEMPORAL CONTEXT:" not in result

    def test_single_speaker_included(self):
        """Single speaker name should appear in prompt."""
        result = get_conversation_extraction_prompt(speaker_names=["Alice"])

        assert "SPEAKERS: Alice" in result
        assert "Alice" in result  # Should appear in examples too

    def test_multiple_speakers_included(self):
        """Multiple speaker names should be comma-separated."""
        result = get_conversation_extraction_prompt(speaker_names=["Alice", "Bob", "Charlie"])

        assert "SPEAKERS: Alice, Bob, Charlie" in result

    def test_first_speaker_used_in_examples(self):
        """First speaker should be used in example snippets."""
        result = get_conversation_extraction_prompt(speaker_names=["Tanay", "Bob"])

        # First speaker should replace default "Alice" in examples
        assert "Tanay" in result
        # Check specific example patterns
        assert "Tanay" in result

    def test_context_date_creates_temporal_section(self):
        """Context date should create TEMPORAL CONTEXT section."""
        result = get_conversation_extraction_prompt(context_date="May 7, 2023")

        assert "TEMPORAL CONTEXT:" in result
        assert "May 7, 2023" in result

    def test_temporal_section_explains_conversions(self):
        """Temporal section should explain date conversions."""
        result = get_conversation_extraction_prompt(context_date="January 15, 2024")

        assert "last year" in result.lower()
        assert "yesterday" in result.lower()
        assert "last week" in result.lower()
        assert "next month" in result.lower()

    def test_both_speaker_and_date_included(self):
        """Both speaker names and date should work together."""
        result = get_conversation_extraction_prompt(
            speaker_names=["Eve", "Frank"], context_date="December 1, 2023"
        )

        assert "SPEAKERS: Eve, Frank" in result
        assert "TEMPORAL CONTEXT:" in result
        assert "December 1, 2023" in result
        assert "Eve" in result  # Used in examples

    def test_contains_extraction_categories(self):
        """Prompt should list what to extract."""
        result = get_conversation_extraction_prompt()

        assert "IDENTITY" in result or "CHARACTERISTICS" in result
        assert "PREFERENCES" in result
        assert "ACTIVITIES" in result
        assert "RELATIONSHIPS" in result
        assert "EVENTS" in result
        assert "PLANS" in result or "GOALS" in result

    def test_contains_importance_scoring_guidance(self):
        """Prompt should explain importance scoring."""
        result = get_conversation_extraction_prompt()

        assert "importance" in result.lower()
        assert "0.3" in result or "0.4" in result  # Background
        assert "0.5" in result or "0.6" in result  # Useful
        assert "0.7" in result or "0.8" in result  # Important
        assert "0.9" in result or "1.0" in result  # Critical

    def test_contains_atomic_fact_format(self):
        """Prompt should explain atomic fact format."""
        result = get_conversation_extraction_prompt()

        assert "ATOMIC FACT" in result or "atomic fact" in result.lower()
        assert "GOOD:" in result or "✓ GOOD" in result
        assert "BAD:" in result or "✗ BAD" in result

    def test_contains_few_shot_examples(self):
        """Prompt should contain few-shot examples."""
        result = get_conversation_extraction_prompt()

        assert "FEW-SHOT EXAMPLES" in result or "Examples:" in result
        assert "Input:" in result
        assert "Output:" in result

    def test_contains_filtering_guidance(self):
        """Prompt should explain what NOT to extract."""
        result = get_conversation_extraction_prompt()

        assert "FILTERING" in result or "DO NOT extract" in result
        assert "greetings" in result.lower()
        assert "transient" in result.lower()
        assert "sensitive" in result.lower()

    def test_empty_speaker_list_treated_as_none(self):
        """Empty speaker list should not add SPEAKERS section."""
        result = get_conversation_extraction_prompt(speaker_names=[])
        assert "SPEAKERS:" not in result

    def test_special_characters_in_speaker_names(self):
        """Speaker names with special characters should work."""
        result = get_conversation_extraction_prompt(speaker_names=["O'Brien", "Jean-Luc"])

        assert "O'Brien" in result
        assert "Jean-Luc" in result

    def test_very_long_speaker_list(self):
        """Long speaker lists should be handled."""
        speakers = [f"Person{i}" for i in range(10)]
        result = get_conversation_extraction_prompt(speaker_names=speakers)

        assert "Person0" in result
        assert "Person9" in result
        # All speakers should be comma-separated
        assert "SPEAKERS: " in result


# =============================================================================
# Test get_memory_answer_prompt()
# =============================================================================


class TestGetMemoryAnswerPrompt:
    """Tests for get_memory_answer_prompt() function."""

    def test_returns_string(self):
        """Function should return a string."""
        result = get_memory_answer_prompt()
        assert isinstance(result, str)

    def test_returns_non_empty_prompt(self):
        """Function should return non-empty prompt."""
        result = get_memory_answer_prompt()
        assert len(result) > 50

    def test_no_args_generic_context(self):
        """Without speaker_names, context should be generic."""
        result = get_memory_answer_prompt()

        # Should mention memory system
        assert "memory" in result.lower()
        # The first line should be "You are answering questions using a memory system."
        # (no "about X" context added)
        first_line = result.split("\n")[0]
        assert "about" not in first_line

    def test_single_speaker_adds_context(self):
        """Single speaker should add context about them."""
        result = get_memory_answer_prompt(speaker_names=["Alice"])

        assert "about Alice" in result

    def test_multiple_speakers_joined_with_and(self):
        """Multiple speakers should be joined with 'and'."""
        result = get_memory_answer_prompt(speaker_names=["Alice", "Bob"])

        assert "about Alice and Bob" in result

    def test_three_speakers_joined_correctly(self):
        """Three speakers use ' and ' between all."""
        result = get_memory_answer_prompt(speaker_names=["Alice", "Bob", "Charlie"])

        # Should join with ' and ' for all names
        assert "Alice and Bob and Charlie" in result

    def test_contains_process_steps(self):
        """Prompt should explain the answer process."""
        result = get_memory_answer_prompt()

        assert "PROCESS" in result or "Process" in result
        assert "memory_search" in result

    def test_contains_answer_rules(self):
        """Prompt should contain answer rules."""
        result = get_memory_answer_prompt()

        assert "ANSWER RULES" in result or "rules" in result.lower()
        assert "CONCISE" in result or "concise" in result.lower()

    def test_handles_inference_questions(self):
        """Prompt should explain how to handle inference questions."""
        result = get_memory_answer_prompt()

        assert "INFERENCE" in result or "inference" in result.lower()
        assert "would" in result.lower() or "could" in result.lower()

    def test_handles_not_found_case(self):
        """Prompt should explain what to do when info not found."""
        result = get_memory_answer_prompt()

        assert "not found" in result.lower() or "Information not found" in result

    def test_empty_speaker_list_treated_as_none(self):
        """Empty speaker list should be treated as None."""
        result = get_memory_answer_prompt(speaker_names=[])

        # Empty list should result in no " about X" context
        # The function joins empty list which results in empty string
        # So "about " would be followed by nothing meaningful
        # Both should be similar (no specific speaker context)
        assert "about  " not in result  # No double space


# =============================================================================
# Test MEMORY_SAVE_TOOL_WITH_EXTRACTION
# =============================================================================


class TestMemorySaveToolWithExtraction:
    """Tests for MEMORY_SAVE_TOOL_WITH_EXTRACTION schema."""

    def test_is_dict(self):
        """Tool schema should be a dictionary."""
        assert isinstance(MEMORY_SAVE_TOOL_WITH_EXTRACTION, dict)

    def test_has_type_field(self):
        """Tool should have type field set to 'function'."""
        assert MEMORY_SAVE_TOOL_WITH_EXTRACTION.get("type") == "function"

    def test_has_function_field(self):
        """Tool should have function field."""
        assert "function" in MEMORY_SAVE_TOOL_WITH_EXTRACTION
        assert isinstance(MEMORY_SAVE_TOOL_WITH_EXTRACTION["function"], dict)

    def test_function_has_name(self):
        """Function should have name 'memory_save'."""
        func = MEMORY_SAVE_TOOL_WITH_EXTRACTION["function"]
        assert func.get("name") == "memory_save"

    def test_function_has_description(self):
        """Function should have non-empty description."""
        func = MEMORY_SAVE_TOOL_WITH_EXTRACTION["function"]

        assert "description" in func
        assert isinstance(func["description"], str)
        assert len(func["description"]) > 50

    def test_description_mentions_extraction(self):
        """Description should mention pre-extraction."""
        func = MEMORY_SAVE_TOOL_WITH_EXTRACTION["function"]
        desc = func["description"]

        assert "extract" in desc.lower()
        assert "facts" in desc.lower()
        assert "entities" in desc.lower()
        assert "relationships" in desc.lower()

    def test_has_parameters_field(self):
        """Function should have parameters field."""
        func = MEMORY_SAVE_TOOL_WITH_EXTRACTION["function"]

        assert "parameters" in func
        assert isinstance(func["parameters"], dict)

    def test_parameters_has_type_object(self):
        """Parameters should be type object."""
        params = MEMORY_SAVE_TOOL_WITH_EXTRACTION["function"]["parameters"]

        assert params.get("type") == "object"

    def test_parameters_has_properties(self):
        """Parameters should have properties field."""
        params = MEMORY_SAVE_TOOL_WITH_EXTRACTION["function"]["parameters"]

        assert "properties" in params
        assert isinstance(params["properties"], dict)

    def test_content_parameter_exists(self):
        """Should have content parameter."""
        props = MEMORY_SAVE_TOOL_WITH_EXTRACTION["function"]["parameters"]["properties"]

        assert "content" in props
        assert props["content"].get("type") == "string"

    def test_importance_parameter_exists(self):
        """Should have importance parameter with range."""
        props = MEMORY_SAVE_TOOL_WITH_EXTRACTION["function"]["parameters"]["properties"]

        assert "importance" in props
        importance = props["importance"]

        assert importance.get("type") == "number"
        assert importance.get("minimum") == 0.0
        assert importance.get("maximum") == 1.0

    def test_facts_parameter_exists(self):
        """Should have facts parameter as array of strings."""
        props = MEMORY_SAVE_TOOL_WITH_EXTRACTION["function"]["parameters"]["properties"]

        assert "facts" in props
        facts = props["facts"]

        assert facts.get("type") == "array"
        assert facts.get("items", {}).get("type") == "string"

    def test_extracted_entities_parameter_exists(self):
        """Should have extracted_entities parameter with proper schema."""
        props = MEMORY_SAVE_TOOL_WITH_EXTRACTION["function"]["parameters"]["properties"]

        assert "extracted_entities" in props
        entities = props["extracted_entities"]

        assert entities.get("type") == "array"

        # Check items schema
        items = entities.get("items", {})
        assert items.get("type") == "object"

        item_props = items.get("properties", {})
        assert "entity" in item_props
        assert "entity_type" in item_props

        required = items.get("required", [])
        assert "entity" in required
        assert "entity_type" in required

    def test_extracted_relationships_parameter_exists(self):
        """Should have extracted_relationships parameter with proper schema."""
        props = MEMORY_SAVE_TOOL_WITH_EXTRACTION["function"]["parameters"]["properties"]

        assert "extracted_relationships" in props
        rels = props["extracted_relationships"]

        assert rels.get("type") == "array"

        # Check items schema
        items = rels.get("items", {})
        assert items.get("type") == "object"

        item_props = items.get("properties", {})
        assert "source" in item_props
        assert "relationship" in item_props
        assert "destination" in item_props

        required = items.get("required", [])
        assert "source" in required
        assert "relationship" in required
        assert "destination" in required

    def test_required_fields(self):
        """Content and importance should be required."""
        params = MEMORY_SAVE_TOOL_WITH_EXTRACTION["function"]["parameters"]

        assert "required" in params
        required = params["required"]

        assert "content" in required
        assert "importance" in required


# =============================================================================
# Test get_extraction_tools()
# =============================================================================


class TestGetExtractionTools:
    """Tests for get_extraction_tools() function."""

    def test_returns_list(self):
        """Function should return a list."""
        result = get_extraction_tools()
        assert isinstance(result, list)

    def test_returns_three_tools(self):
        """Should return exactly 3 extraction tools."""
        result = get_extraction_tools()
        assert len(result) == 3

    def test_all_items_are_dicts(self):
        """All items should be dictionaries."""
        result = get_extraction_tools()
        for tool in result:
            assert isinstance(tool, dict)

    def test_all_tools_have_function_type(self):
        """All tools should have type 'function'."""
        result = get_extraction_tools()
        for tool in result:
            assert tool.get("type") == "function"

    def test_all_tools_have_function_field(self):
        """All tools should have function field."""
        result = get_extraction_tools()
        for tool in result:
            assert "function" in tool
            assert isinstance(tool["function"], dict)

    def test_extract_facts_tool_exists(self):
        """Should have extract_facts tool."""
        result = get_extraction_tools()
        tool_names = [t["function"]["name"] for t in result]

        assert "extract_facts" in tool_names

    def test_extract_entities_tool_exists(self):
        """Should have extract_entities tool."""
        result = get_extraction_tools()
        tool_names = [t["function"]["name"] for t in result]

        assert "extract_entities" in tool_names

    def test_extract_relationships_tool_exists(self):
        """Should have extract_relationships tool."""
        result = get_extraction_tools()
        tool_names = [t["function"]["name"] for t in result]

        assert "extract_relationships" in tool_names

    def test_extract_facts_schema(self):
        """extract_facts should have correct schema."""
        result = get_extraction_tools()
        facts_tool = next(t for t in result if t["function"]["name"] == "extract_facts")

        func = facts_tool["function"]
        assert "description" in func
        assert "parameters" in func

        params = func["parameters"]
        assert params.get("type") == "object"
        assert "facts" in params.get("properties", {})

        facts_prop = params["properties"]["facts"]
        assert facts_prop.get("type") == "array"
        assert facts_prop.get("items", {}).get("type") == "string"

        assert "facts" in params.get("required", [])

    def test_extract_entities_schema(self):
        """extract_entities should have correct schema."""
        result = get_extraction_tools()
        entities_tool = next(t for t in result if t["function"]["name"] == "extract_entities")

        func = entities_tool["function"]
        assert "description" in func
        assert "parameters" in func

        params = func["parameters"]
        assert params.get("type") == "object"
        assert "entities" in params.get("properties", {})

        entities_prop = params["properties"]["entities"]
        assert entities_prop.get("type") == "array"

        items = entities_prop.get("items", {})
        assert items.get("type") == "object"
        assert "entity" in items.get("properties", {})
        assert "entity_type" in items.get("properties", {})

        assert "entities" in params.get("required", [])

    def test_extract_relationships_schema(self):
        """extract_relationships should have correct schema."""
        result = get_extraction_tools()
        rels_tool = next(t for t in result if t["function"]["name"] == "extract_relationships")

        func = rels_tool["function"]
        assert "description" in func
        assert "parameters" in func

        params = func["parameters"]
        assert params.get("type") == "object"
        assert "relationships" in params.get("properties", {})

        rels_prop = params["properties"]["relationships"]
        assert rels_prop.get("type") == "array"

        items = rels_prop.get("items", {})
        assert items.get("type") == "object"
        item_props = items.get("properties", {})
        assert "source" in item_props
        assert "relationship" in item_props
        assert "destination" in item_props

        assert "relationships" in params.get("required", [])

    def test_returns_new_list_each_call(self):
        """Should return a new list each call (not same reference)."""
        result1 = get_extraction_tools()
        result2 = get_extraction_tools()

        assert result1 is not result2
        # But content should be equal
        assert result1 == result2


# =============================================================================
# Integration Tests
# =============================================================================


class TestExtractionIntegration:
    """Integration tests for extraction module."""

    def test_prompts_are_different(self):
        """Each prompt constant should be unique."""
        prompts = [
            FACT_EXTRACTION_PROMPT,
            ENTITY_EXTRACTION_PROMPT,
            RELATIONSHIP_EXTRACTION_PROMPT,
            EXTRACTION_SYSTEM_PROMPT,
        ]

        # All prompts should be different
        for i, p1 in enumerate(prompts):
            for j, p2 in enumerate(prompts):
                if i != j:
                    assert p1 != p2

    def test_conversation_prompt_changes_with_speaker(self):
        """Conversation prompt should change based on speaker."""
        prompt_default = get_conversation_extraction_prompt()
        prompt_alice = get_conversation_extraction_prompt(speaker_names=["Alice"])
        prompt_bob = get_conversation_extraction_prompt(speaker_names=["Bob"])

        assert prompt_default != prompt_alice
        assert prompt_alice != prompt_bob

    def test_conversation_prompt_changes_with_date(self):
        """Conversation prompt should change based on date."""
        prompt_no_date = get_conversation_extraction_prompt()
        prompt_date1 = get_conversation_extraction_prompt(context_date="January 1, 2024")
        prompt_date2 = get_conversation_extraction_prompt(context_date="December 31, 2024")

        assert prompt_no_date != prompt_date1
        assert prompt_date1 != prompt_date2

    def test_answer_prompt_changes_with_speaker(self):
        """Answer prompt should change based on speaker."""
        prompt_default = get_memory_answer_prompt()
        prompt_alice = get_memory_answer_prompt(speaker_names=["Alice"])

        assert prompt_default != prompt_alice

    def test_tool_schema_is_valid_json_serializable(self):
        """Tool schema should be JSON serializable."""
        import json

        # Should not raise
        json_str = json.dumps(MEMORY_SAVE_TOOL_WITH_EXTRACTION)
        # Should round-trip correctly
        loaded = json.loads(json_str)
        assert loaded == MEMORY_SAVE_TOOL_WITH_EXTRACTION

    def test_extraction_tools_are_json_serializable(self):
        """Extraction tools should be JSON serializable."""
        import json

        tools = get_extraction_tools()

        # Should not raise
        json_str = json.dumps(tools)
        # Should round-trip correctly
        loaded = json.loads(json_str)
        assert loaded == tools

    def test_all_tool_names_unique(self):
        """All tool names should be unique."""
        tools = get_extraction_tools()
        names = [t["function"]["name"] for t in tools]

        assert len(names) == len(set(names))

    def test_memory_save_tool_compatible_with_extraction_tools(self):
        """Memory save tool should accept outputs from extraction tools."""
        # The memory_save tool accepts:
        # - facts: array of strings (from extract_facts)
        # - extracted_entities: array of {entity, entity_type} (from extract_entities)
        # - extracted_relationships: array of {source, relationship, destination} (from extract_relationships)

        save_tool = MEMORY_SAVE_TOOL_WITH_EXTRACTION["function"]["parameters"]["properties"]
        extraction_tools = {t["function"]["name"]: t for t in get_extraction_tools()}

        # Facts compatibility
        facts_output = extraction_tools["extract_facts"]["function"]["parameters"]["properties"][
            "facts"
        ]
        save_facts_input = save_tool["facts"]
        assert facts_output["type"] == save_facts_input["type"]  # both array
        assert facts_output["items"]["type"] == save_facts_input["items"]["type"]  # both string

        # Entities compatibility
        entities_output = extraction_tools["extract_entities"]["function"]["parameters"][
            "properties"
        ]["entities"]
        save_entities_input = save_tool["extracted_entities"]
        assert entities_output["type"] == save_entities_input["type"]  # both array
        # Both have object items with entity and entity_type
        assert entities_output["items"]["type"] == save_entities_input["items"]["type"]

        # Relationships compatibility
        rels_output = extraction_tools["extract_relationships"]["function"]["parameters"][
            "properties"
        ]["relationships"]
        save_rels_input = save_tool["extracted_relationships"]
        assert rels_output["type"] == save_rels_input["type"]  # both array
        # Both have object items with source, relationship, destination
        assert rels_output["items"]["type"] == save_rels_input["items"]["type"]
