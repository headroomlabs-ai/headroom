"""Mistral tokenizer using the official mistral-common package.

Mistral AI released their tokenizer publicly, making accurate
token counting possible without API calls.

Requires: pip install mistral-common
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from .base import BaseTokenizer

logger = logging.getLogger(__name__)

# Check if mistral-common is available
try:
    from mistral_common.protocol.instruct.messages import (
        AssistantMessage,
        SystemMessage,
        UserMessage,
    )
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer as _MistralTokenizer

    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    _MistralTokenizer = None


def is_mistral_available() -> bool:
    """Check if mistral-common is installed."""
    return MISTRAL_AVAILABLE


# Model to tokenizer version mapping
MODEL_TO_VERSION = {
    # Mistral models use v3 tokenizer (tekken)
    "mistral-large": "v3",
    "mistral-large-latest": "v3",
    "mistral-small": "v3",
    "mistral-small-latest": "v3",
    "ministral-8b": "v3",
    "ministral-3b": "v3",
    "mistral-nemo": "v3",
    "pixtral-12b": "v3",
    "codestral": "v3",
    "codestral-latest": "v3",
    # Mixtral uses v1
    "mixtral-8x7b": "v1",
    "mixtral-8x22b": "v1",
    "open-mixtral-8x7b": "v1",
    "open-mixtral-8x22b": "v1",
    # Mistral 7B uses v1
    "mistral-7b": "v1",
    "open-mistral-7b": "v1",
    "mistral-7b-instruct": "v1",
}


@lru_cache(maxsize=4)
def _get_tokenizer(version: str):
    """Get and cache Mistral tokenizer by version."""
    if not MISTRAL_AVAILABLE:
        raise RuntimeError(
            "mistral-common is required for MistralTokenizer. "
            "Install with: pip install mistral-common"
        )

    if version == "v3":
        return _MistralTokenizer.v3(is_tekken=True)
    elif version == "v2":
        return _MistralTokenizer.v2()
    else:  # v1
        return _MistralTokenizer.v1()


def get_tokenizer_version(model: str) -> str:
    """Get tokenizer version for a model."""
    model_lower = model.lower()

    # Direct lookup
    if model_lower in MODEL_TO_VERSION:
        return MODEL_TO_VERSION[model_lower]

    # Prefix matching
    for prefix, version in [
        ("mistral-large", "v3"),
        ("mistral-small", "v3"),
        ("ministral", "v3"),
        ("codestral", "v3"),
        ("pixtral", "v3"),
        ("mistral-nemo", "v3"),
        ("mixtral", "v1"),
        ("mistral-7b", "v1"),
        ("open-mistral", "v1"),
    ]:
        if model_lower.startswith(prefix):
            return version

    # Default to v3 for newer models
    return "v3"


class MistralTokenizer(BaseTokenizer):
    """Token counter using Mistral's official tokenizer.

    Uses mistral-common package for accurate token counting.

    Requires: pip install mistral-common

    Example:
        counter = MistralTokenizer("mistral-large")
        tokens = counter.count_text("Hello, world!")
    """

    MESSAGE_OVERHEAD = 4
    REPLY_OVERHEAD = 3

    def __init__(self, model: str = "mistral-large"):
        """Initialize Mistral tokenizer.

        Args:
            model: Mistral model name.
        """
        if not MISTRAL_AVAILABLE:
            raise RuntimeError(
                "mistral-common is required for MistralTokenizer. "
                "Install with: pip install mistral-common"
            )

        self.model = model
        self.version = get_tokenizer_version(model)
        self._tokenizer = None  # Lazy load

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer (MistralTokenizer object)."""
        if self._tokenizer is None:
            self._tokenizer = _get_tokenizer(self.version)
        return self._tokenizer

    @property
    def _text_tokenizer(self):
        """Get the underlying text tokenizer for encode/decode."""
        return self.tokenizer.instruct_tokenizer.tokenizer

    def count_text(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to tokenize.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        tokens = self._text_tokenizer.encode(text, bos=False, eos=False)
        return len(tokens)

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in chat messages.

        Uses Mistral's chat template for accurate counting.

        Args:
            messages: List of chat messages.

        Returns:
            Total token count.
        """
        if not messages:
            return 0

        try:
            # Convert to Mistral message format
            mistral_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if isinstance(content, list):
                    # Multi-part content - extract text
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = "\n".join(text_parts)

                if role == "user":
                    mistral_messages.append(UserMessage(content=content))
                elif role == "assistant":
                    mistral_messages.append(AssistantMessage(content=content))
                elif role == "system":
                    mistral_messages.append(SystemMessage(content=content))
                else:
                    # Tool messages etc - treat as user
                    mistral_messages.append(UserMessage(content=content))

            # Encode with chat template
            request = ChatCompletionRequest(messages=mistral_messages)
            tokenized = self.tokenizer.encode_chat_completion(request)
            return len(tokenized.tokens)

        except Exception as e:
            logger.debug(f"Mistral chat encoding failed: {e}, falling back to text counting")
            # Fallback to base implementation
            return super().count_messages(messages)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode.

        Returns:
            List of token IDs.
        """
        return self._text_tokenizer.encode(text, bos=False, eos=False)

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text.
        """
        return self._text_tokenizer.decode(tokens)

    @classmethod
    def is_available(cls) -> bool:
        """Check if Mistral tokenizer is available."""
        return MISTRAL_AVAILABLE

    def __repr__(self) -> str:
        return f"MistralTokenizer(model={self.model!r}, version={self.version!r})"
