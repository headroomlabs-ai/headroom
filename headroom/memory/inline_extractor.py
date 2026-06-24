"""Inline memory extraction - zero extra latency.

Instead of making a separate LLM call to extract memories,
we modify the system prompt so the LLM outputs memories
as part of its response. This is the Letta/MemGPT approach.

Benefits:
- Zero extra latency (memory is part of response)
- Zero extra API cost (already paying for response tokens)
- Higher quality (LLM has full context)
- Intelligent filtering (LLM decides what's relevant)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# Memory extraction instruction to append to system prompt
MEMORY_INSTRUCTION = """

## Memory Instructions
After your response, if there are facts worth remembering about the user/entity for future conversations, output them in a <memory> block:

<memory>
{"memories": [{"content": "fact to remember"}]}
</memory>

What to remember:
- User preferences (likes, dislikes, preferred tools/languages/styles)
- User facts (identity, role, job, location, constraints)
- Context (current goals, ongoing tasks, recent events)

Only output memories for significant, reusable information. Skip for:
- Greetings, thanks, small talk
- One-time questions
- Information already known

If nothing worth remembering: <memory>{"memories": []}</memory>
"""

# Shorter version for token efficiency
MEMORY_INSTRUCTION_SHORT = """

After responding, output facts to remember: <memory>{"memories": [{"content": "..."}]}</memory>
Skip for greetings/small talk. If nothing: <memory>{"memories": []}</memory>"""


@dataclass
class ParsedResponse:
    """Response with extracted memories."""

    content: str  # The actual response (without memory block)
    memories: list[dict[str, Any]]  # Extracted memories
    raw: str  # Original full response


def inject_memory_instruction(
    messages: list[dict[str, Any]],
    short: bool = True,
) -> list[dict[str, Any]]:
    """Inject memory extraction instruction into system prompt.

    Args:
        messages: Original messages list
        short: Use short instruction (fewer tokens)

    Returns:
        Modified messages with memory instruction
    """
    instruction = MEMORY_INSTRUCTION_SHORT if short else MEMORY_INSTRUCTION
    messages = [m.copy() for m in messages]  # Don't modify original

    # Find or create system message
    has_system = False
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            messages[i] = {
                **msg,
                "content": msg.get("content", "") + instruction,
            }
            has_system = True
            break

    if not has_system:
        # Prepend system message
        messages.insert(
            0,
            {
                "role": "system",
                "content": "You are a helpful assistant." + instruction,
            },
        )

    return messages


def parse_response_with_memory(response_text: str) -> ParsedResponse:
    """Parse LLM response to extract memories.

    Args:
        response_text: Raw LLM response

    Returns:
        ParsedResponse with content and memories separated
    """
    memories: list[dict[str, Any]] = []
    content = response_text

    # Extract <memory> block
    memory_pattern = r"<memory>\s*(.*?)\s*</memory>"
    match = re.search(memory_pattern, response_text, re.DOTALL | re.IGNORECASE)

    if match:
        memory_json = match.group(1).strip()

        # Remove the memory block from content
        content = re.sub(memory_pattern, "", response_text, flags=re.DOTALL | re.IGNORECASE).strip()

        # Parse the JSON
        try:
            data = json.loads(memory_json)
            memories = data.get("memories", [])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse memory JSON: {e}")

    return ParsedResponse(
        content=content,
        memories=memories,
        raw=response_text,
    )


class InlineMemoryWrapper:
    """Wrapper that extracts memories from LLM responses inline.

    This is the zero-latency approach - memories are extracted
    as part of the response, not in a separate call.

    Usage:
        wrapper = InlineMemoryWrapper(openai_client)
        response, memories = wrapper.chat(
            messages=[{"role": "user", "content": "I prefer Python"}],
            model="gpt-4o-mini"
        )
        # response = "Great choice! Python is excellent..."
        # memories = [{"content": "User prefers Python"}]
    """

    def __init__(self, client: Any):
        """Initialize wrapper.

        Args:
            client: OpenAI-compatible client
        """
        self.client = client

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str = "gpt-4o-mini",
        short_instruction: bool = True,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Send chat request and extract memories inline.

        Args:
            messages: Chat messages
            model: Model to use
            short_instruction: Use shorter memory instruction
            **kwargs: Additional args for chat completion

        Returns:
            Tuple of (response_content, extracted_memories)
        """
        # Inject memory instruction
        modified_messages = inject_memory_instruction(messages, short=short_instruction)

        # Call LLM
        response = self.client.chat.completions.create(
            model=model,
            messages=modified_messages,
            **kwargs,
        )

        raw_content = response.choices[0].message.content

        # Parse response and extract memories
        parsed = parse_response_with_memory(raw_content)

        return parsed.content, parsed.memories

    def chat_with_response(
        self,
        messages: list[dict[str, Any]],
        model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> tuple[Any, str, list[dict[str, Any]]]:
        """Send chat request and return full response object.

        Args:
            messages: Chat messages
            model: Model to use
            **kwargs: Additional args for chat completion

        Returns:
            Tuple of (response_object, content, memories)
        """
        modified_messages = inject_memory_instruction(messages)

        response = self.client.chat.completions.create(
            model=model,
            messages=modified_messages,
            **kwargs,
        )

        raw_content = response.choices[0].message.content
        parsed = parse_response_with_memory(raw_content)

        # Modify response to have clean content
        response.choices[0].message.content = parsed.content

        return response, parsed.content, parsed.memories
