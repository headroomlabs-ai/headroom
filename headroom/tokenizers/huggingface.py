"""HuggingFace tokenizer wrapper for open models.

Supports Llama, Mistral, Falcon, and other models with HuggingFace
tokenizers. Requires the `transformers` library.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from typing import Any

from .base import BaseTokenizer

logger = logging.getLogger(__name__)


# Model name to HuggingFace tokenizer mapping
# Maps common model names to their HuggingFace tokenizer identifiers
MODEL_TO_TOKENIZER: dict[str, str] = {
    # Llama 3 family
    "llama-3": "meta-llama/Meta-Llama-3-8B",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B",
    "llama-3.1-405b": "meta-llama/Llama-3.1-405B",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    # Llama 2 family
    "llama-2": "meta-llama/Llama-2-7b-hf",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "llama-2-70b": "meta-llama/Llama-2-70b-hf",
    # CodeLlama
    "codellama": "codellama/CodeLlama-7b-hf",
    "codellama-7b": "codellama/CodeLlama-7b-hf",
    "codellama-13b": "codellama/CodeLlama-13b-hf",
    "codellama-34b": "codellama/CodeLlama-34b-hf",
    # Mistral family
    "mistral": "mistralai/Mistral-7B-v0.1",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral-7b-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-nemo": "mistralai/Mistral-Nemo-Base-2407",
    "mistral-small": "mistralai/Mistral-Small-Instruct-2409",
    "mistral-large": "mistralai/Mistral-Large-Instruct-2407",
    # Mixtral
    "mixtral": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral-8x22b": "mistralai/Mixtral-8x22B-v0.1",
    # Qwen family
    "qwen": "Qwen/Qwen-7B",
    "qwen-7b": "Qwen/Qwen-7B",
    "qwen-14b": "Qwen/Qwen-14B",
    "qwen-72b": "Qwen/Qwen-72B",
    "qwen2": "Qwen/Qwen2-7B",
    "qwen2-7b": "Qwen/Qwen2-7B",
    "qwen2-72b": "Qwen/Qwen2-72B",
    "qwen2.5": "Qwen/Qwen2.5-7B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B",
    # DeepSeek V1 / Coder (legacy, 2023-2024)
    "deepseek": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-7b": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-67b": "deepseek-ai/deepseek-llm-67b-base",
    "deepseek-coder": "deepseek-ai/deepseek-coder-6.7b-base",
    "deepseek-coder-v2": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-coder-v2-lite": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    # DeepSeek V2/V3 family (2024)
    "deepseek-v2": "deepseek-ai/DeepSeek-V2",
    "deepseek-v2-lite": "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-v3-0324": "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
    # DeepSeek R1 reasoning family (2025)
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "deepseek-r1-0528": "deepseek-ai/DeepSeek-R1-0528",
    "deepseek-reasoner": "deepseek-ai/DeepSeek-R1",
    # DeepSeek V4 family (2025-2026)
    "deepseek-v4-pro": "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash",
    # DeepSeek API aliases (routed through the proxy)
    "deepseek-chat": "deepseek-ai/DeepSeek-V3",
    "deepseek-r1-distill-qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1-distill-llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # Yi family
    "yi": "01-ai/Yi-6B",
    "yi-6b": "01-ai/Yi-6B",
    "yi-34b": "01-ai/Yi-34B",
    "yi-1.5": "01-ai/Yi-1.5-6B",
    # Phi family
    "phi-2": "microsoft/phi-2",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-small": "microsoft/Phi-3-small-8k-instruct",
    "phi-3-medium": "microsoft/Phi-3-medium-4k-instruct",
    # Falcon
    "falcon": "tiiuae/falcon-7b",
    "falcon-7b": "tiiuae/falcon-7b",
    "falcon-40b": "tiiuae/falcon-40b",
    "falcon-180b": "tiiuae/falcon-180B",
    # StarCoder
    "starcoder": "bigcode/starcoder",
    "starcoder2": "bigcode/starcoder2-15b",
    "starcoder2-3b": "bigcode/starcoder2-3b",
    "starcoder2-7b": "bigcode/starcoder2-7b",
    "starcoder2-15b": "bigcode/starcoder2-15b",
    # MPT
    "mpt-7b": "mosaicml/mpt-7b",
    "mpt-30b": "mosaicml/mpt-30b",
    # Gemma
    "gemma": "google/gemma-7b",
    "gemma-2b": "google/gemma-2b",
    "gemma-7b": "google/gemma-7b",
    "gemma-2": "google/gemma-2-9b",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-27b": "google/gemma-2-27b",
}


# Bound the first (network) load of a HuggingFace tokenizer. Without a bound,
# huggingface_hub download retries can block for many minutes (GH #1701: 610s
# on a restricted Windows network). 0 disables network loads entirely.
_LOAD_TIMEOUT_ENV = "HEADROOM_HF_TOKENIZER_LOAD_TIMEOUT_SECS"
_LOAD_TIMEOUT_DEFAULT = 10.0


def _load_timeout_secs() -> float:
    try:
        return float(os.environ.get(_LOAD_TIMEOUT_ENV, _LOAD_TIMEOUT_DEFAULT))
    except (TypeError, ValueError):
        return _LOAD_TIMEOUT_DEFAULT


@lru_cache(maxsize=16)
def _load_tokenizer(tokenizer_name: str):
    """Load and cache HuggingFace tokenizer.

    The first attempt is cache-only (``local_files_only=True``) so a warm
    HF cache never touches the network. A cache miss falls through to a
    network download bounded by ``HEADROOM_HF_TOKENIZER_LOAD_TIMEOUT_SECS``
    (default 10s) on a daemon thread — the download itself cannot be
    cancelled, but the caller unblocks and falls back to estimation.
    Failures are cached by ``lru_cache`` (returns ``None``), so a slow or
    offline hub is probed at most once per process per tokenizer.

    Args:
        tokenizer_name: HuggingFace model/tokenizer name.

    Returns:
        Loaded tokenizer, or None if unavailable.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:
        pass  # Not in the local cache — try the network below, bounded.

    timeout = _load_timeout_secs()
    if timeout <= 0:
        logger.warning(
            f"Tokenizer {tokenizer_name} not in local HF cache and network "
            f"loading is disabled ({_LOAD_TIMEOUT_ENV}=0); using estimation"
        )
        return None

    result: list[Any] = []
    error: list[BaseException] = []

    def _download() -> None:
        try:
            result.append(
                AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    trust_remote_code=True,
                )
            )
        except BaseException as e:  # noqa: BLE001 — report any failure to the waiter
            error.append(e)

    thread = threading.Thread(
        target=_download,
        name=f"headroom-hf-tokenizer-load-{tokenizer_name}",
        daemon=True,
    )
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        logger.warning(
            f"Timed out loading tokenizer {tokenizer_name} after {timeout}s "
            f"(set {_LOAD_TIMEOUT_ENV} to adjust); using estimation"
        )
        return None
    if error:
        logger.warning(f"Failed to load tokenizer {tokenizer_name}: {error[0]}")
        return None
    return result[0] if result else None


def get_tokenizer_name(model: str) -> str:
    """Get HuggingFace tokenizer name for a model.

    Args:
        model: Model name.

    Returns:
        HuggingFace tokenizer identifier.
    """
    model_lower = model.lower()

    # Direct lookup
    if model_lower in MODEL_TO_TOKENIZER:
        return MODEL_TO_TOKENIZER[model_lower]

    # Try prefix matching, longest (most specific) key first. Scanning in
    # dict-insertion order is wrong: a short family key like "qwen" precedes
    # "qwen2"/"qwen2.5", so "qwen2-7b-instruct" would match "qwen" first and
    # resolve to the Qwen1 tokenizer (a different vocabulary -> wrong counts).
    # The sibling tiktoken resolver (get_encoding_for_model) documents and
    # guards this exact order-dependent pitfall.
    for key in sorted(MODEL_TO_TOKENIZER, key=len, reverse=True):
        if model_lower.startswith(key):
            return MODEL_TO_TOKENIZER[key]

    # Assume model name is the tokenizer name
    return model


class HuggingFaceTokenizer(BaseTokenizer):
    """Token counter using HuggingFace tokenizers.

    Supports any model with a HuggingFace tokenizer, including:
    - Llama family (Llama 2, Llama 3, CodeLlama)
    - Mistral family (Mistral, Mixtral)
    - Qwen family
    - DeepSeek family
    - Phi family
    - Falcon, StarCoder, MPT, Gemma, etc.

    Requires the `transformers` library:
        pip install transformers

    Some models may require authentication:
        huggingface-cli login

    Example:
        counter = HuggingFaceTokenizer("llama-3-8b")
        tokens = counter.count_text("Hello, world!")
    """

    # Overhead per message (varies by model, this is a reasonable default)
    MESSAGE_OVERHEAD = 4
    REPLY_OVERHEAD = 3

    def __init__(self, model: str):
        """Initialize HuggingFace tokenizer.

        Args:
            model: Model name (e.g., 'llama-3-8b', 'mistral-7b').
        """
        self.model = model
        self.tokenizer_name = get_tokenizer_name(model)
        self._tokenizer = None  # Lazy load

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer."""
        if self._tokenizer is None:
            loaded = _load_tokenizer(self.tokenizer_name)
            if loaded is not None:
                self._tokenizer = loaded
            else:
                # Mark as unavailable
                self._tokenizer = False
        return self._tokenizer if self._tokenizer is not False else None

    def _use_fallback(self) -> bool:
        """Check if we need to use fallback estimation."""
        return self.tokenizer is None

    def count_text(self, text: str) -> int:
        """Count tokens in text.

        Falls back to estimation if tokenizer unavailable.

        Args:
            text: Text to tokenize.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        if self._use_fallback():
            # Fall back to ~4 chars per token estimation
            return max(1, int(len(text) / 4 + 0.5))
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in chat messages.

        Uses the model's chat template if available, otherwise
        falls back to base class implementation.

        Args:
            messages: List of chat messages.

        Returns:
            Total token count.
        """
        if self._use_fallback():
            # Use base class implementation with estimation
            return super().count_messages(messages)

        # Try to use chat template for accurate counting
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                # ``return_dict=False`` is load-bearing. transformers >= 5 defaults
                # ``apply_chat_template(tokenize=True)`` to ``return_dict=True``,
                # which hands back a BatchEncoding — so ``len(formatted)`` counted
                # DICT KEYS (2: input_ids, attention_mask) instead of tokens.
                # Measured on Qwen2.5-72B, a 6,000-char message: count_messages
                # returned 2 and count_message returned -1 (base subtracts a
                # 3-token reply overhead), against a true 1,003 tokens. That is a
                # ~99.8% undercount on every HF-routed family whose resolved
                # tokenizer carries a chat template — llama, qwen, deepseek, phi,
                # yi, falcon, starcoder. pyproject pins transformers>=5.5.0,<6.0,
                # so the affected version is the only installable one.
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
                # Recent transformers return a BatchEncoding (dict-like with
                # input_ids/attention_mask) here; len() of it is the key count
                # (2), not the token count.
                if hasattr(formatted, "keys") and "input_ids" in formatted:
                    return len(formatted["input_ids"])
                return len(formatted)
            except Exception:
                # Fall back to base implementation
                pass

        return super().count_messages(messages)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode.

        Returns:
            List of token IDs.

        Raises:
            NotImplementedError: If tokenizer not available.
        """
        if self._use_fallback():
            raise NotImplementedError(
                f"Encoding not available for {self.model} - "
                f"tokenizer {self.tokenizer_name} could not be loaded"
            )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text.

        Raises:
            NotImplementedError: If tokenizer not available.
        """
        if self._use_fallback():
            raise NotImplementedError(
                f"Decoding not available for {self.model} - "
                f"tokenizer {self.tokenizer_name} could not be loaded"
            )
        return self.tokenizer.decode(tokens)

    @classmethod
    def is_available(cls) -> bool:
        """Check if HuggingFace tokenizers are available.

        Returns:
            True if transformers is installed.
        """
        try:
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def list_supported_models(cls) -> list[str]:
        """List models with known tokenizer mappings.

        Returns:
            List of supported model names.
        """
        return list(MODEL_TO_TOKENIZER.keys())

    def __repr__(self) -> str:
        return f"HuggingFaceTokenizer(model={self.model!r}, tokenizer={self.tokenizer_name!r})"
