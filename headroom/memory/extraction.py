"""Memory extraction system prompt.

``EXTRACTION_SYSTEM_PROMPT`` is injected into the main LLM's system prompt by
``with_memory_tools(..., optimized=True)`` so the model pre-extracts facts,
entities and relationships when it calls ``memory_save``. Backends that accept
those fields (``DirectMem0Adapter``) then write them directly instead of
running their own LLM extraction.
"""

from __future__ import annotations

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
