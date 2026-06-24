"""Pluggable tokenizer system for universal LLM support.

This module provides a registry-based tokenizer system that supports
multiple backends:

1. tiktoken - OpenAI models (GPT-3.5, GPT-4, GPT-4o)
2. HuggingFace - Open models (Llama, Mistral, Falcon, etc.)
3. Anthropic - Claude models (via SDK or estimation)
4. Estimation - Fallback for unknown models

Usage:
    from headroom.tokenizers import TokenizerRegistry, get_tokenizer

    # Auto-detect tokenizer from model name
    tokenizer = get_tokenizer("gpt-4o")
    tokens = tokenizer.count_text("Hello, world!")

    # Get tokenizer for specific backend
    tokenizer = get_tokenizer("llama-3-8b", backend="huggingface")

    # Register custom tokenizer
    TokenizerRegistry.register("my-model", my_tokenizer)
"""

from .base import BaseTokenizer, TokenCounter
from .estimator import CharacterCounter, EstimatingTokenCounter
from .registry import (
    TokenizerRegistry,
    get_tokenizer,
    list_supported_models,
    register_tokenizer,
)
from .tiktoken_counter import TiktokenCounter


# Lazy imports for optional dependencies
def get_huggingface_tokenizer():
    """Get HuggingFaceTokenizer class (requires transformers)."""
    from .huggingface import HuggingFaceTokenizer

    return HuggingFaceTokenizer


def get_mistral_tokenizer():
    """Get MistralTokenizer class (requires mistral-common)."""
    from .mistral import MistralTokenizer

    return MistralTokenizer


def is_mistral_tokenizer_available() -> bool:
    """Check if Mistral tokenizer is available."""
    from .mistral import is_mistral_available

    return is_mistral_available()


__all__ = [
    # Registry
    "TokenizerRegistry",
    "get_tokenizer",
    "register_tokenizer",
    "list_supported_models",
    # Base classes
    "TokenCounter",
    "BaseTokenizer",
    # Implementations
    "TiktokenCounter",
    "EstimatingTokenCounter",
    "CharacterCounter",
    # Lazy loaders
    "get_huggingface_tokenizer",
    "get_mistral_tokenizer",
    "is_mistral_tokenizer_available",
]
