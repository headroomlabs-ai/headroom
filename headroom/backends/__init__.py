"""Headroom Backends - API translation layers for different LLM providers.

Backends handle the translation between the proxy's canonical format
(Anthropic Messages API) and provider-specific APIs.

Supported backend libraries:
- LiteLLM: 100+ providers (bedrock, vertex_ai, azure, openrouter, etc.)
- any-llm: 38+ providers (openai, anthropic, mistral, groq, ollama, etc.)

Usage:
    # LiteLLM backend
    headroom proxy --backend litellm-bedrock --region us-west-2

    # any-llm backend
    headroom proxy --backend anyllm --anyllm-provider openai
"""

from .anyllm import AnyLLMBackend
from .base import Backend, BackendResponse, StreamEvent
from .litellm import LiteLLMBackend

__all__ = ["Backend", "BackendResponse", "StreamEvent", "LiteLLMBackend", "AnyLLMBackend"]
