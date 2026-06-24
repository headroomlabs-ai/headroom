"""Memory tool definitions for LLM function calling.

This module defines the tool specifications in OpenAI function calling format
that allow LLMs to interact with the memory system. These tools enable
autonomous memory management - saving, searching, updating, and deleting
memories as needed during conversations.

Two versions of memory_save are provided:
1. MEMORY_TOOLS - Standard version (backwards compatible)
2. MEMORY_TOOLS_OPTIMIZED - Enhanced version with pre-extraction fields

The optimized version allows the main LLM to extract facts, entities, and
relationships in a single pass, avoiding redundant LLM calls in the storage
backend (Mem0). See extraction.py for the extraction prompts.
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Memory Tool Definitions (OpenAI Function Calling Format)
# =============================================================================

MEMORY_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "memory_save",
            "description": """Save important information to long-term memory for future reference.

Use this tool when you encounter information that should be remembered across conversations, such as:
- User preferences (e.g., "prefers Python over JavaScript", "likes concise answers")
- Personal facts (e.g., "works at Acme Corp", "has a dog named Max")
- Project context (e.g., "working on a CLI tool", "using React 18")
- Decisions made (e.g., "chose PostgreSQL for the database", "decided on REST over GraphQL")
- Important relationships (e.g., "Alice is Bob's manager", "Project X depends on Service Y")
- Technical insights (e.g., "the auth module is in src/auth/", "uses custom logging format")

DO save:
- Information explicitly shared by the user that seems important for future interactions
- Corrections to previous assumptions or memories
- Key decisions and their rationale
- Recurring topics or preferences that emerge from conversation patterns

DO NOT save:
- Transient information only relevant to the current conversation
- Sensitive data like passwords, API keys, or private credentials
- Information the user explicitly asks not to remember
- Redundant information already stored (search first if unsure)

The importance score (0.0-1.0) helps prioritize memories during retrieval:
- 0.9-1.0: Critical facts that should almost always be recalled
- 0.7-0.8: Important preferences or context
- 0.5-0.6: Useful but not essential information
- 0.3-0.4: Nice-to-have background context
- 0.1-0.2: Low-priority supplementary details""",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to remember. Be specific and self-contained - this should make sense without additional context. Good: 'User prefers dark mode in all applications'. Bad: 'likes dark mode'.",
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Importance score from 0.0 (low) to 1.0 (critical). Higher importance memories are prioritized in search results and less likely to be forgotten.",
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of entity names or identifiers referenced in this memory (e.g., ['Alice', 'Project X', 'auth-service']). Used for entity-based retrieval and relationship tracking.",
                    },
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string", "description": "Source entity name"},
                                "relation": {
                                    "type": "string",
                                    "description": "Relationship type (e.g., 'works_with', 'manages', 'depends_on', 'is_part_of')",
                                },
                                "target": {"type": "string", "description": "Target entity name"},
                            },
                            "required": ["source", "relation", "target"],
                        },
                        "description": "Relationships between entities mentioned in this memory. Enables graph-based queries like 'who does Alice work with?'",
                    },
                },
                "required": ["content", "importance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": """Search stored memories to recall relevant information.

Use this tool to retrieve previously saved information before responding to questions about:
- User preferences or past decisions
- Personal or professional context
- Previously discussed topics or projects
- Relationships between people, systems, or concepts
- Historical context from past conversations

Search strategies:
1. Semantic search (default): Use natural language queries that describe what you're looking for
   - "user's programming language preferences"
   - "information about the current project"
   - "past decisions about database choices"

2. Entity-based search: Specify entities to find memories mentioning specific people/things
   - entities=["Alice", "Project X"] finds memories involving Alice or Project X

3. Related memories: Set include_related=true to also retrieve connected memories
   - Finds memories linked by shared entities or explicit relationships

Best practices:
- Search BEFORE saving to avoid duplicates
- Search when answering questions that might rely on remembered information
- Use specific queries for better precision
- Combine entity filters with semantic queries for targeted retrieval""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query describing what information you're looking for. Be specific but not too narrow.",
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter to memories mentioning any of these entities. Useful for finding information about specific people, projects, or systems.",
                    },
                    "include_related": {
                        "type": "boolean",
                        "description": "If true, also retrieve memories connected to the results via entity relationships. Helps build fuller context around a topic.",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Maximum number of memories to retrieve. Default is 10. Use higher values when you need comprehensive context.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_update",
            "description": """Update an existing memory with corrected or evolved information.

Use this tool when:
- The user provides a correction to previously stored information
  - "Actually, I prefer TypeScript now, not JavaScript"
  - "My project is called ProjectX, not Project Y"

- Information has changed over time
  - "I've switched teams from Engineering to Product"
  - "We migrated from MySQL to PostgreSQL"

- You need to add detail or clarification to an existing memory
  - Original: "Uses React" -> Updated: "Uses React 18 with TypeScript and Vite"

- Consolidating multiple related memories into one clearer entry

DO NOT use this to:
- Add completely new information (use memory_save instead)
- Delete memories (use memory_delete instead)
- Update memories with unrelated content

The update creates a new version while preserving history, allowing point-in-time queries of past states. Always provide a clear reason for the update to maintain an audit trail.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The unique ID of the memory to update. Take this from the [id] prefix shown in the auto-injected memory block, or from a memory_search / memory_list result.",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The updated content that will replace the existing memory content. Should be complete and self-contained.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Explanation for why this memory is being updated (e.g., 'user correction', 'information changed', 'adding detail'). Stored for audit trail.",
                    },
                },
                "required": ["memory_id", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": """Delete a memory that is no longer relevant or was stored in error.

Use this tool when:
- The user explicitly asks to forget something
  - "Please forget that I mentioned working at Acme"
  - "Delete what you remember about Project X"

- Information is outdated and no longer applicable (not just changed - use update for that)
  - A completed project that's no longer relevant
  - A temporary context that has expired

- A memory was saved in error
  - Duplicate information
  - Misunderstood or incorrect context

- Privacy or data hygiene reasons
  - User requests removal of personal information
  - Cleaning up test or debug memories

Before deleting:
1. Search to find the specific memory and confirm its ID
2. Verify with the user if the deletion intent is ambiguous
3. Consider if update would be more appropriate (for changed vs. obsolete info)

Deletions are soft by default - the memory history is preserved but marked as deleted.
Always provide a reason for deletion to maintain an audit trail.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The unique ID of the memory to delete. Take this from the [id] prefix shown in the auto-injected memory block, or from a memory_search / memory_list result.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Explanation for why this memory is being deleted (e.g., 'user request', 'outdated', 'stored in error'). Required for audit trail.",
                    },
                },
                "required": ["memory_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_list",
            "description": """Browse memories without a semantic query — list recent or all memories with their IDs.

Use this when:
- You want to see what's stored without a specific search term
  - "What do you remember about me / this project?"
  - "Show me everything you've saved recently"
- You need a memory ID for `memory_update` or `memory_delete` but don't have a good search query
- You're auditing the memory store (debugging, cleanup, review)

Differences from `memory_search`:
- `memory_search(query)` is SEMANTIC — finds memories similar to a query string
- `memory_list()` is CHRONOLOGICAL — returns the most recent memories first
- Use `memory_search` when you know what you're looking for; use `memory_list` when you want to browse

Returns memories in reverse chronological order (newest first). Each entry includes
the `memory_id` you'd use to update / delete it.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of memories to return (default 10, max 100). Use a smaller number for a quick overview; larger when you need to find a specific memory ID.",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": [],
            },
        },
    },
]


def get_memory_tools() -> list[dict[str, Any]]:
    """Return the list of memory tool definitions.

    Returns:
        List of tool definitions in OpenAI function calling format.
    """
    return MEMORY_TOOLS.copy()


def get_tool_names() -> list[str]:
    """Return the names of all memory tools.

    Returns:
        List of tool names.
    """
    return [tool["function"]["name"] for tool in MEMORY_TOOLS]


# =============================================================================
# Optimized Memory Tools (with pre-extraction support)
# =============================================================================
# These tools include additional fields for pre-extracted facts, entities,
# and relationships. When these fields are provided, the storage backend
# can bypass its internal LLM extraction, resulting in significant speedup.

MEMORY_SAVE_OPTIMIZED: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_save",
        "description": """Save important information to long-term memory with optional pre-extraction.

IMPORTANT: For efficiency, extract facts, entities, and relationships yourself when calling this tool.
This avoids redundant LLM calls in the storage backend.

Use this tool when you encounter information that should be remembered:
- User preferences, personal facts, project context, decisions, relationships

PRE-EXTRACTION (recommended for efficiency):
- facts: List of discrete, self-contained fact strings
  Example: ["Prefers Python over JavaScript", "Works at Acme Corp"]
- extracted_entities: List of entities with types
  Example: [{"entity": "Python", "entity_type": "technology"}]
- extracted_relationships: List of entity relationships
  Example: [{"source": "user", "relationship": "works_at", "destination": "Acme Corp"}]

ASYNC/BACKGROUND MODE (for zero latency):
- Set background=true to return immediately while saving happens in background
- Returns a task_id that can be used to check save status
- Ideal for real-time conversations where response speed is critical

The importance score (0.0-1.0) helps prioritize memories:
- 0.9-1.0: Critical facts
- 0.7-0.8: Important preferences
- 0.5-0.6: Useful information
- 0.3-0.4: Background context

DO NOT save: transient information, sensitive data (passwords, keys), redundant info""",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The original information to remember. Used as context and fallback if no facts provided.",
                },
                "importance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Importance score from 0.0 (low) to 1.0 (critical).",
                },
                "facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Pre-extracted discrete facts. Each should be self-contained and specific. Example: ['Uses PyTorch for deep learning', 'Prefers dark mode']",
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of entity names referenced (simple format for backwards compatibility).",
                },
                "extracted_entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "Entity name"},
                            "entity_type": {
                                "type": "string",
                                "description": "Type: person, organization, technology, location, project, concept",
                            },
                        },
                        "required": ["entity", "entity_type"],
                    },
                    "description": "Pre-extracted entities with types for graph storage.",
                },
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "relation": {"type": "string"},
                            "target": {"type": "string"},
                        },
                        "required": ["source", "relation", "target"],
                    },
                    "description": "Simple relationship format (backwards compatible).",
                },
                "extracted_relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "Source entity"},
                            "relationship": {
                                "type": "string",
                                "description": "Relationship type: works_at, uses, knows, manages, depends_on, etc.",
                            },
                            "destination": {"type": "string", "description": "Destination entity"},
                        },
                        "required": ["source", "relationship", "destination"],
                    },
                    "description": "Pre-extracted relationships for graph storage.",
                },
                "background": {
                    "type": "boolean",
                    "description": "If true, save in background and return immediately with task_id. "
                    "Use for zero-latency responses. The save will complete asynchronously. "
                    "Check status via memory system's get_task_status(task_id).",
                },
            },
            "required": ["content", "importance"],
        },
    },
}

# Optimized tools list - use this for better performance with DirectMem0Adapter
MEMORY_TOOLS_OPTIMIZED: list[dict[str, Any]] = [
    MEMORY_SAVE_OPTIMIZED,
    MEMORY_TOOLS[1],  # memory_search (unchanged)
    MEMORY_TOOLS[2],  # memory_update (unchanged)
    MEMORY_TOOLS[3],  # memory_delete (unchanged)
    MEMORY_TOOLS[4],  # memory_list (new — chronological browse)
]


def get_memory_tools_optimized() -> list[dict[str, Any]]:
    """Return the optimized memory tool definitions with pre-extraction support.

    Use these tools with DirectMem0Adapter for best performance.
    The main LLM should extract facts/entities/relationships when calling
    memory_save, which bypasses redundant LLM extraction in the backend.

    Returns:
        List of optimized tool definitions in OpenAI function calling format.
    """
    return MEMORY_TOOLS_OPTIMIZED.copy()
