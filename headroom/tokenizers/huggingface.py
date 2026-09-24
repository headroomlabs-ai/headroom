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


# Every repository Headroom is willing to hand to the HuggingFace loader.
#
# The loader resolves a name against the Hub and, historically, ran repository
# code while doing it (``trust_remote_code``). The model id reaching this module
# can come straight off a client request body — the proxy routes ``model`` to a
# tokenizer — so an unconstrained identifier is remote code execution in the
# proxy process by way of "publish a repo, then ask for it by name". The
# allowlist is derived from the mappings we actually ship, so adding a family to
# MODEL_TO_TOKENIZER allows it and nothing else drifts in.
ALLOWED_TOKENIZER_REPOS: frozenset[str] = frozenset(MODEL_TO_TOKENIZER.values())

# The tokenizer an unrecognised model resolves to. Llama 3 is the canonical open
# model vocabulary and is what the registry already routes bare ``llama*`` ids
# to. The repository is gated on the Hub, so a deployment without HF credentials
# simply fails the load and falls back to character estimation — which is what
# an unknown model got before this allowlist existed.
DEFAULT_TOKENIZER: str = MODEL_TO_TOKENIZER["llama-3"]

# Operator escape hatch for self-hosted or private tokenizer repositories:
# a comma-separated list of repo ids. This is deliberately environment-only —
# server-side configuration an operator sets, never anything a client can
# influence through a request body. Repositories added here are still loaded
# with ``trust_remote_code=False``.
_ALLOWLIST_ENV = "HEADROOM_HF_TOKENIZER_ALLOWLIST"


@lru_cache(maxsize=8)
def _allowlist_index(extra: str) -> dict[str, str]:
    """Lowercased repo id -> canonical repo id, for the shipped list plus ``extra``.

    Keyed on the raw environment value so the (tiny) index is built once per
    distinct configuration rather than on every model resolution.
    """
    repos = set(ALLOWED_TOKENIZER_REPOS)
    repos.update(part.strip() for part in extra.split(",") if part.strip())
    return {repo.lower(): repo for repo in repos}


def _resolve_allowed_repo(name: str) -> str | None:
    """Return the canonical allowlisted repo id for ``name``, or None.

    Matching is case-insensitive because Hub ids are commonly retyped with
    different capitalisation, but the value returned is always *our* spelling
    from the allowlist — the caller's string is never propagated to the loader.
    """
    if not name:
        return None
    return _allowlist_index(os.environ.get(_ALLOWLIST_ENV, "")).get(name.strip().lower())


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

    Only repositories on the allowlist reach ``from_pretrained`` at all. This is
    the chokepoint, not a second opinion: ``get_tokenizer_name`` already resolves
    unrecognised models to DEFAULT_TOKENIZER, but this function is importable and
    callable with an arbitrary string, and a name that is not on the allowlist
    must never turn into a Hub lookup. Refusing here fails closed to estimation
    rather than substituting a different vocabulary behind the caller's back.

    Args:
        tokenizer_name: HuggingFace model/tokenizer name.

    Returns:
        Loaded tokenizer, or None if unavailable.
    """
    repo = _resolve_allowed_repo(tokenizer_name)
    if repo is None:
        logger.warning(
            f"Refusing to load unallowlisted tokenizer {tokenizer_name!r}; using "
            f"estimation (add it to {_ALLOWLIST_ENV} if this repository is trusted)"
        )
        return None
    # Load the allowlist's own spelling, never the argument: matching is
    # case-insensitive, so the string that reaches the Hub must be the one we
    # vetted, not a variant the caller chose.
    tokenizer_name = repo

    from transformers import AutoTokenizer

    try:
        # trust_remote_code=False, always. A tokenizer repository can ship its own
        # Python, and executing it is equivalent to running whatever the repo owner
        # publishes inside the proxy. Every tokenizer we map is a plain vocabulary
        # that loads fine without it; a repo that genuinely needs custom code is a
        # repo we are not willing to execute.
        return AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=False,
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
                    trust_remote_code=False,  # see the cache-only attempt above
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

    Always returns an allowlisted repository. ``model`` is attacker-reachable —
    it is the ``model`` field of a proxied request body — so this resolver must
    never return a caller-controlled string for the loader to look up on the Hub.

    Args:
        model: Model name.

    Returns:
        HuggingFace tokenizer identifier, from ALLOWED_TOKENIZER_REPOS.
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

    # A caller may legitimately name a shipped repository outright
    # ("meta-llama/Llama-3.1-8B") instead of using the short alias, and that has
    # to keep counting exactly. Accepting it via the allowlist returns our own
    # canonical spelling, so the caller's string still never reaches the loader.
    allowed = _resolve_allowed_repo(model)
    if allowed is not None:
        return allowed

    # Fail closed. This branch used to be "assume the model name is the
    # tokenizer name", which handed an arbitrary client-supplied string to
    # AutoTokenizer.from_pretrained -> a Hub fetch of whatever repository the
    # caller named. An unrecognised model now resolves to the default vocabulary;
    # the count is an approximation, which is what an unknown model already got
    # when the made-up repo id failed to resolve.
    logger.debug(
        "No tokenizer mapping for model %r; using default tokenizer %s",
        model,
        DEFAULT_TOKENIZER,
    )
    return DEFAULT_TOKENIZER


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
