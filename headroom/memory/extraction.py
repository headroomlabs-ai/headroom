"""Memory extraction prompts and utilities.

This module contains extraction prompts derived from Mem0's internal prompts,
adapted for use with Headroom's memory system. By using these prompts in the
main LLM, we can extract facts/entities/relationships in a SINGLE pass,
avoiding the double LLM calls that occur when Mem0 does its own extraction.

Architecture:
    Traditional Mem0 flow (INEFFICIENT):
        User → Main LLM → memory_save(content) → Mem0.add() → Mem0 LLM extracts facts
                                                            → Mem0 LLM extracts entities
                                                            → Mem0 LLM extracts relationships
        Total: 3-4 LLM calls per memory save!

    Optimized Headroom flow (EFFICIENT):
        User → Main LLM (with extraction prompts) → memory_save(facts, entities, relationships)
                                                  → Direct write to Qdrant + Neo4j
        Total: 1 LLM call per memory save!

Usage:
    # Option 1: Inline extraction via tool schema
    # The main LLM extracts facts/entities when calling memory_save
    # See MEMORY_SAVE_TOOL_WITH_EXTRACTION for the enhanced tool schema

    # Option 2: System prompt injection
    # Add EXTRACTION_SYSTEM_PROMPT to your main LLM's system prompt
    # The LLM will output structured extraction in its response
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Fact Extraction Prompt (for Vector Store)
# Generic, balanced prompt - not too specific, not too broad
# =============================================================================

FACT_EXTRACTION_PROMPT = """You are a comprehensive fact extractor. Your goal is to capture ALL meaningful information from conversations as discrete, searchable facts.

CORE PRINCIPLES:

1. **Comprehensiveness**: Extract EVERY fact, not just obvious ones. If in doubt, extract it.

2. **Attribution**: Every fact MUST include WHO it's about. Use actual names, never "user" or "I".
   - Good: "Alice prefers tea over coffee"
   - Bad: "Prefers tea over coffee"

3. **Specificity**: Use exact terms from the conversation, not vague paraphrases.
   - Good: "Bob does pottery and running to destress"
   - Bad: "Bob enjoys art and exercise"

4. **Self-contained**: Each fact should be understandable on its own.
   - Good: "Carol's sister Emma works at Google"
   - Bad: "Sister works at Google"

5. **Temporal grounding**: When dates are mentioned (explicitly or relative to a known date), include the resolved date.
   - If "last year" is mentioned and the conversation is from 2023, say "in 2022"
   - If "yesterday" is mentioned and the date is May 7, say "on May 6"

WHAT TO EXTRACT:
- Personal details (identity, background, relationships, important dates)
- Preferences and opinions (likes, dislikes, choices)
- Activities and hobbies (specific activities, not categories)
- Professional information (job, company, skills, goals)
- Events and experiences (trips, meetings, milestones)
- Plans and intentions (future events, goals)
- Health and wellness (restrictions, routines, conditions)

WHAT NOT TO EXTRACT:
- Greetings and small talk
- Transient information ("I'm going to make coffee")
- Sensitive data (passwords, financial details)
- Duplicate information already captured

OUTPUT: Return a list of discrete fact strings, each complete and self-contained.
"""

# =============================================================================
# Entity Extraction Prompt (for Graph Store)
# Derived from Mem0's entity extraction system prompt
# =============================================================================

ENTITY_EXTRACTION_PROMPT = """You are a smart assistant who understands entities and their types in a given text.

Guidelines:
- Extract all named entities (people, organizations, products, technologies, locations, etc.)
- Assign appropriate entity types to each
- If the text contains self-references ('I', 'me', 'my'), use the user_id as the entity
- Do NOT answer questions - only extract entities

Entity Types to consider:
- person: Individual people (names, roles)
- organization: Companies, teams, departments
- technology: Programming languages, frameworks, tools, services
- location: Cities, countries, buildings, rooms
- project: Named projects, initiatives
- concept: Abstract concepts, topics, domains
- product: Software products, hardware, services
- event: Meetings, conferences, deadlines

Example:
- Input: "I work at Acme Corp using Python and TensorFlow"
  Entities: [
    {"entity": "user_id", "entity_type": "person"},
    {"entity": "Acme Corp", "entity_type": "organization"},
    {"entity": "Python", "entity_type": "technology"},
    {"entity": "TensorFlow", "entity_type": "technology"}
  ]
"""

# =============================================================================
# Relationship Extraction Prompt (for Graph Store)
# Derived from Mem0's EXTRACT_RELATIONS_PROMPT
# =============================================================================

RELATIONSHIP_EXTRACTION_PROMPT = """You are an advanced algorithm designed to extract structured relationships from text to construct knowledge graphs.

Guidelines:
1. Extract only explicitly stated relationships from the text
2. Use the user_id as the source entity for self-references ('I', 'me', 'my')
3. Use consistent, general, and timeless relationship types
   - Prefer "works_at" over "started_working_at"
   - Prefer "uses" over "recently_started_using"
4. Only establish relationships among entities explicitly mentioned

Relationship Format:
{
  "source": "entity name",
  "relationship": "relationship_type",
  "destination": "entity name"
}

Common Relationship Types:
- works_at, employed_by
- uses, prefers, likes, dislikes
- knows, collaborates_with, reports_to
- located_in, based_in
- part_of, belongs_to, member_of
- created, owns, maintains
- depends_on, requires, integrates_with

Example:
- Input: "I work at Netflix and use Python daily. Alice is my manager."
  Relationships: [
    {"source": "user_id", "relationship": "works_at", "destination": "Netflix"},
    {"source": "user_id", "relationship": "uses", "destination": "Python"},
    {"source": "Alice", "relationship": "manages", "destination": "user_id"}
  ]
"""

# =============================================================================
# Combined Extraction System Prompt
# For injecting into the main LLM's system prompt
# =============================================================================

EXTRACTION_SYSTEM_PROMPT = """When the user shares information worth remembering, you should extract and structure it for memory storage.

For each piece of information worth saving, extract:

1. **Facts**: Discrete, self-contained statements
   - Personal preferences, important details, plans, professional info
   - Each fact should make sense on its own
   - Format: List of strings

2. **Entities**: Named things mentioned in the text
   - People, organizations, technologies, locations, projects
   - Format: [{"entity": "name", "entity_type": "type"}]

3. **Relationships**: How entities relate to each other
   - Use consistent relationship types (works_at, uses, knows, etc.)
   - Format: [{"source": "entity1", "relationship": "rel_type", "destination": "entity2"}]

When calling memory_save, include these pre-extracted fields to optimize storage."""


# =============================================================================
# Generic Conversation Extraction Prompt
# Balanced: not too specific, not too broad
# Works with any domain, any speakers
# =============================================================================


def get_conversation_extraction_prompt(
    speaker_names: list[str] | None = None,
    context_date: str | None = None,
) -> str:
    """Generate a balanced conversation extraction prompt.

    This prompt is designed following research from Mem0, MemMachine, and ChatExtract:
    - Uses atomic fact decomposition (single-fact units)
    - Includes categorical structure (preference/fact/context)
    - Has importance scoring guidance
    - Provides few-shot examples for format clarity
    - Strong temporal grounding when date context is provided
    - Entity attribution rules

    The key balance: Generic enough for any domain, specific enough for comprehensive extraction.

    Args:
        speaker_names: Optional list of speaker names for attribution guidance
        context_date: Optional date string for temporal context (e.g., "May 7, 2023")

    Returns:
        A system prompt for conversation fact extraction
    """
    # Build speaker context
    speakers_section = ""
    example_speaker = "Alice"
    if speaker_names and len(speaker_names) >= 1:
        example_speaker = speaker_names[0]
        names = ", ".join(speaker_names)
        speakers_section = f"""
SPEAKERS: {names}
Use their actual names in every fact (never "user", "I", or "they")."""

    # Build temporal context - this is CRITICAL for accuracy
    temporal_section = ""
    if context_date:
        temporal_section = f"""
TEMPORAL CONTEXT: This conversation is from {context_date}.
Convert ALL relative dates to absolute dates:
- "last year" → the year before {context_date}
- "yesterday" → the day before {context_date}
- "last week" → approximately one week before {context_date}
- "next month" → the month after {context_date}
- "recently" → estimate based on {context_date}

IMPORTANT: Store the resolved absolute date, not the relative phrase."""

    return f"""You are a memory extraction system. Analyze the conversation and extract facts worth remembering for future retrieval.
{speakers_section}{temporal_section}

WHAT TO EXTRACT (Categories):

1. **IDENTITY & CHARACTERISTICS** - Who people ARE (most important!)
   - Gender identity, relationship status, nationality, ethnicity, age
   - "X is a transgender woman", "X is single", "X is from Sweden"
   - Personal traits, backgrounds, defining characteristics

2. **PREFERENCES** - Likes, dislikes, choices, opinions
   - "X prefers Y over Z", "X likes Y", "X dislikes Z"

3. **ACTIVITIES & HOBBIES** - Specific things people DO (be specific!)
   - "X does pottery", "X runs marathons", "X plays violin"
   - NOT vague: "X likes art" - be specific about WHAT activities

4. **RELATIONSHIPS** - How people relate to each other
   - Family: "X's sister is Y", "X is married to Y"
   - Professional: "X works at Y", "X's manager is Y"

5. **EVENTS WITH DATES** - Things that happened (always include WHEN)
   - "X attended Y on [date]", "X visited Y in [month/year]"

6. **PLANS & GOALS** - Future intentions
   - "X is planning to...", "X wants to..."

EXTRACTION FORMAT:

For each meaningful segment, call memory_save with:
- **content**: Brief summary
- **importance**: Score from 0.0 to 1.0
  - 0.3-0.4: Background info (minor details)
  - 0.5-0.6: Useful info (preferences, context)
  - 0.7-0.8: Important info (key facts, relationships)
  - 0.9-1.0: Critical info (identity, major events)
- **facts**: List of atomic facts (see format below)
- **extracted_entities**: [{{"entity": "name", "entity_type": "person|organization|location|technology|event|project"}}]
- **extracted_relationships**: [{{"source": "entity1", "relationship": "rel_type", "destination": "entity2"}}]

ATOMIC FACT FORMAT:
Each fact must be a complete, self-contained statement: [WHO] [WHAT] [WHEN/WHERE if applicable]

✓ GOOD: "{example_speaker} is a software engineer at Netflix"
✗ BAD: "Is a software engineer" (missing WHO)

✓ GOOD: "{example_speaker} visited Paris in June 2023"
✗ BAD: "{example_speaker} went somewhere recently" (too vague, relative date)

✓ GOOD: "{example_speaker} prefers Python over JavaScript for backend work"
✗ BAD: "{example_speaker} likes programming" (too vague, lost specificity)

RELATIONSHIP TYPES:
Use consistent types: works_at, lives_in, knows, manages, reports_to, married_to, sibling_of, friend_of, prefers, uses, attended, visited, member_of, created, owns, collects, studies, plays

CRITICAL - RELATIONSHIP EXTRACTION:
Relationships enable multi-hop reasoning. Extract relationships whenever:
- Two PEOPLE are connected (family, friends, colleagues, manager/report)
- A person USES/OWNS/COLLECTS something
- A person WORKS AT/STUDIES AT an organization
- A person LIVES IN/VISITED a location
- A person ATTENDS/MEMBER OF a group or event
- A person LIKES/PREFERS/INTERESTED IN a topic or thing

FEW-SHOT EXAMPLES:

Input: "{example_speaker}: I moved to Tokyo last year for my job at Sony. My colleague Kenji has been showing me around."
Output: {{
  "content": "{example_speaker} relocated to Tokyo for work",
  "importance": 0.8,
  "facts": ["{example_speaker} moved to Tokyo", "{example_speaker} works at Sony", "{example_speaker}'s colleague is Kenji", "Kenji shows {example_speaker} around Tokyo"],
  "extracted_entities": [{{"entity": "{example_speaker}", "entity_type": "person"}}, {{"entity": "Tokyo", "entity_type": "location"}}, {{"entity": "Sony", "entity_type": "organization"}}, {{"entity": "Kenji", "entity_type": "person"}}],
  "extracted_relationships": [{{"source": "{example_speaker}", "relationship": "lives_in", "destination": "Tokyo"}}, {{"source": "{example_speaker}", "relationship": "works_at", "destination": "Sony"}}, {{"source": "{example_speaker}", "relationship": "knows", "destination": "Kenji"}}]
}}

Input: "{example_speaker}: My brother Tom is a chef and he's obsessed with Italian cuisine. He studied in Rome for two years."
Output: {{
  "content": "{example_speaker}'s brother Tom's career",
  "importance": 0.7,
  "facts": ["{example_speaker}'s brother is Tom", "Tom is a chef", "Tom specializes in Italian cuisine", "Tom studied cooking in Rome for two years"],
  "extracted_entities": [{{"entity": "{example_speaker}", "entity_type": "person"}}, {{"entity": "Tom", "entity_type": "person"}}, {{"entity": "Rome", "entity_type": "location"}}],
  "extracted_relationships": [{{"source": "{example_speaker}", "relationship": "sibling_of", "destination": "Tom"}}, {{"source": "Tom", "relationship": "specializes_in", "destination": "Italian cuisine"}}, {{"source": "Tom", "relationship": "studied_in", "destination": "Rome"}}]
}}

Input: "{example_speaker}: I've been learning Spanish on Duolingo. I want to visit Mexico next summer with my family."
Output: {{
  "content": "{example_speaker}'s language learning and travel plans",
  "importance": 0.6,
  "facts": ["{example_speaker} is learning Spanish", "{example_speaker} uses Duolingo", "{example_speaker} plans to visit Mexico next summer", "{example_speaker} wants to travel with family"],
  "extracted_entities": [{{"entity": "{example_speaker}", "entity_type": "person"}}, {{"entity": "Spanish", "entity_type": "concept"}}, {{"entity": "Duolingo", "entity_type": "technology"}}, {{"entity": "Mexico", "entity_type": "location"}}],
  "extracted_relationships": [{{"source": "{example_speaker}", "relationship": "learning", "destination": "Spanish"}}, {{"source": "{example_speaker}", "relationship": "uses", "destination": "Duolingo"}}, {{"source": "{example_speaker}", "relationship": "plans_to_visit", "destination": "Mexico"}}]
}}

Input: "{example_speaker}: I prefer working from home. I find I'm more productive without the office distractions."
Output: {{
  "content": "{example_speaker}'s work preference",
  "importance": 0.6,
  "facts": ["{example_speaker} prefers working from home", "{example_speaker} finds home more productive than office", "{example_speaker} dislikes office distractions"],
  "extracted_entities": [{{"entity": "{example_speaker}", "entity_type": "person"}}],
  "extracted_relationships": [{{"source": "{example_speaker}", "relationship": "prefers", "destination": "working from home"}}]
}}

FILTERING:
- DO extract: preferences, facts, context, events, relationships
- DO NOT extract: greetings ("Hi!", "How are you?"), transient info ("I'm making coffee"), sensitive data (passwords, keys)

If nothing worth remembering, do not call memory_save."""


# Preset prompts for common use cases
CONVERSATION_EXTRACTION_PROMPT_BASIC = get_conversation_extraction_prompt()


def get_memory_answer_prompt(speaker_names: list[str] | None = None) -> str:
    """Generate a balanced answer prompt for memory-based Q&A.

    Based on research findings:
    - Use exact terms from memories (no paraphrasing to vague terms)
    - Be concise (shortest accurate answer)
    - Know when to say "Information not found"

    Args:
        speaker_names: Optional list of speaker names for context

    Returns:
        A system prompt for answering questions from memories
    """
    context = ""
    if speaker_names:
        names = " and ".join(speaker_names)
        context = f" about {names}"

    return f"""You are answering questions{context} using a memory system.

PROCESS:
1. Use memory_search to find relevant memories
2. Answer based on what the memories contain
3. For inference questions (would/could/likely), reason from memories + common knowledge

ANSWER RULES:
- Be CONCISE but include the key information
- Use SPECIFIC terms from memories (not vague paraphrases)
- For dates: give the actual date from memory
- For names: give the actual name from memory
- For yes/no: give your conclusion with brief supporting evidence

INFERENCE QUESTIONS (would, could, likely, might):
Use memories as evidence and apply common knowledge to reason:
- Memory: "enjoys mystery novels" → Q: "Would they like Agatha Christie?" → "Yes, Agatha Christie is a famous mystery author"
- Memory: "interested in space exploration" → Q: "Would they know about the Mars rover?" → "Yes, Mars missions are central to space exploration"
- Memory: "is a vegetarian" → Q: "Can they eat cheese?" → "Yes, vegetarians can eat dairy products"

If no relevant information found: "Information not found"

Answer directly and concisely."""


# =============================================================================
# Enhanced Memory Save Tool Schema
# Includes pre-extraction fields for optimized storage
# =============================================================================

MEMORY_SAVE_TOOL_WITH_EXTRACTION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_save",
        "description": """Save important information to long-term memory with optional pre-extraction.

When you extract facts, entities, and relationships yourself (recommended for efficiency),
include them in the tool call. This bypasses redundant LLM extraction in the storage backend.

DO save:
- User preferences, personal facts, project context, decisions, relationships
- Pre-extracted facts as discrete, self-contained statements
- Entities with their types (person, organization, technology, etc.)
- Relationships between entities (source, relationship, destination)

DO NOT save:
- Transient information, sensitive data (passwords, keys), redundant info""",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The original information/context to remember. Used as fallback if no facts provided.",
                },
                "importance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Importance score: 0.9-1.0 critical, 0.7-0.8 important, 0.5-0.6 useful, 0.3-0.4 background",
                },
                "facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Pre-extracted discrete facts. Each fact should be self-contained. Example: ['Uses Python for backend', 'Prefers dark mode']",
                },
                "extracted_entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "Entity name"},
                            "entity_type": {
                                "type": "string",
                                "description": "Type: person, organization, technology, location, project, concept, product, event",
                            },
                        },
                        "required": ["entity", "entity_type"],
                    },
                    "description": "Pre-extracted entities with types for graph storage.",
                },
                "extracted_relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "Source entity name"},
                            "relationship": {
                                "type": "string",
                                "description": "Relationship type (e.g., works_at, uses, knows, manages)",
                            },
                            "destination": {
                                "type": "string",
                                "description": "Destination entity name",
                            },
                        },
                        "required": ["source", "relationship", "destination"],
                    },
                    "description": "Pre-extracted relationships between entities for graph storage.",
                },
            },
            "required": ["content", "importance"],
        },
    },
}


def get_extraction_tools() -> list[dict[str, Any]]:
    """Get tool definitions for standalone extraction (if needed).

    These tools can be used to have an LLM extract facts/entities/relationships
    in a separate call, similar to how Mem0 does it internally.

    For most use cases, prefer using MEMORY_SAVE_TOOL_WITH_EXTRACTION
    which combines extraction with storage in a single tool call.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "extract_facts",
                "description": "Extract discrete facts from text for memory storage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "facts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of discrete, self-contained facts extracted from the text.",
                        }
                    },
                    "required": ["facts"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_entities",
                "description": "Extract entities and their types from text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity": {"type": "string"},
                                    "entity_type": {"type": "string"},
                                },
                                "required": ["entity", "entity_type"],
                            },
                        }
                    },
                    "required": ["entities"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_relationships",
                "description": "Extract relationships between entities from text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "relationship": {"type": "string"},
                                    "destination": {"type": "string"},
                                },
                                "required": ["source", "relationship", "destination"],
                            },
                        }
                    },
                    "required": ["relationships"],
                },
            },
        },
    ]
