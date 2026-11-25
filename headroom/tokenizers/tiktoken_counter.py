"""Tiktoken-based token counter for OpenAI models.

Tiktoken is OpenAI's fast BPE tokenizer used by GPT models.
It supports multiple encodings:
- cl100k_base: GPT-4, GPT-3.5-turbo, text-embedding-ada-002
- o200k_base: GPT-4o, GPT-4o-mini
- p50k_base: Codex models, text-davinci-002/003
- r50k_base: GPT-3 models (davinci, curie, etc.)
- gpt2: GPT-2 (requires both vocab.bpe and encoder.json)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue as queue_module
from functools import lru_cache
from typing import Any

import tiktoken

from headroom.offline import guard_egress, is_offline

from ._tiktoken_worker import (
    _load_encoding_worker,
)
from .base import BaseTokenizer, TokenCountCache, coerce_countable_text

logger = logging.getLogger(__name__)


class TiktokenLoadError(RuntimeError):
    """Raised when a tiktoken encoding can't be loaded in time.

    tiktoken downloads its BPE vocab on first use via ``requests.get`` with no
    timeout, so a stalled/firewalled connection can block indefinitely. We bound
    that load and raise this instead, so callers fall back to estimation rather
    than hanging the request (see GH #956).
    """


# Global sets for tracking failed encodings
# _download_failed: download failures (network issues, etc.) - blocks retries
# _disk_load_failed: disk load timeout/miss - does NOT block retries, retryable after vocab available
# _disk_load_permanently_failed: disk load corruption/errors - blocks retries
_download_failed: set[str] = set()
_disk_load_failed: set[str] = set()
_disk_load_permanently_failed: set[str] = set()


def _load_timeout_seconds() -> float:
    try:
        return float(os.environ.get("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "10"))
    except (TypeError, ValueError):
        return 10.0


# Model to encoding mapping
MODEL_TO_ENCODING = {
    # GPT-4o family (o200k_base)
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4o-2024-05-13": "o200k_base",
    "gpt-4o-2024-08-06": "o200k_base",
    "gpt-4o-2024-11-20": "o200k_base",
    "gpt-4o-mini-2024-07-18": "o200k_base",
    # o1 reasoning models (o200k_base)
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    "o3-mini": "o200k_base",
    # GPT-4 family (cl100k_base)
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-4-0314": "cl100k_base",
    "gpt-4-0613": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    "gpt-4-32k-0314": "cl100k_base",
    "gpt-4-32k-0613": "cl100k_base",
    "gpt-4-1106-preview": "cl100k_base",
    "gpt-4-0125-preview": "cl100k_base",
    "gpt-4-turbo-2024-04-09": "cl100k_base",
    # GPT-3.5 family (cl100k_base)
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-0301": "cl100k_base",
    "gpt-3.5-turbo-0613": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-3.5-turbo-16k-0613": "cl100k_base",
    "gpt-3.5-turbo-instruct": "cl100k_base",
    # Embeddings (cl100k_base)
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    # Codex (p50k_base)
    "code-davinci-002": "p50k_base",
    "code-davinci-001": "p50k_base",
    "code-cushman-002": "p50k_base",
    "code-cushman-001": "p50k_base",
    # Legacy GPT-3 (p50k_base for davinci-002/003, r50k_base for older)
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-davinci-001": "r50k_base",
    "text-curie-001": "r50k_base",
    "text-babbage-001": "r50k_base",
    "text-ada-001": "r50k_base",
    "davinci": "r50k_base",
    "curie": "r50k_base",
    "babbage": "r50k_base",
    "ada": "r50k_base",
}

# Default encoding for unknown models
DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=8)
def _get_encoding(encoding_name: str):
    """Get a tiktoken encoding, cached for performance.

    Bounded by ``HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS`` (default 10s): tiktoken's
    vocab download has no network timeout, so we run the load in a worker process
    and raise :class:`TiktokenLoadError` if it doesn't finish in time, letting
    callers fall back to estimation rather than hang the request (GH #956).

    If the encoding is already cached in memory, it is loaded
    without calling ``guard_egress``, allowing fast offline operation with pre-populated
    vocabularies. If only a disk cache entry exists, the load is still bounded by the
    configured timeout to prevent a stalled constructor from blocking indefinitely.
    """
    # Fast path: encoding already in memory cache
    try:
        if encoding_name in tiktoken.registry.ENCODINGS:
            return tiktoken.get_encoding(encoding_name)
    except Exception:
        pass

    # Check if this encoding previously failed to DOWNLOAD (permanent failure - network issues)
    if encoding_name in _download_failed:
        raise TiktokenLoadError(
            f"tiktoken encoding {encoding_name!r} previously failed to download"
        )

    # NOTE: We do NOT short-circuit on disk load timeouts.
    # A transient timeout should be retryable once the vocabulary is available.
    # _disk_load_failed tracks transient disk load failures (timeout/crash).
    # These are NOT short-circuited - they remain retryable once vocabulary is available.
    # _disk_load_permanently_failed tracks corruption/invalid cache - only blocks in offline mode.

    # Try to load from disk cache with timeout using multiprocessing (can actually terminate)
    # This moves the cache check AND the load inside the timeout boundary
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_load_encoding_worker, args=(encoding_name, queue), daemon=False)
    proc.start()

    # Consume queue WHILE process is alive to ensure result is transferred (fixes warm-disk path)
    try:
        result = queue.get(timeout=_load_timeout_seconds())
    except queue_module.Empty:
        # Timeout - process is still running or dead without putting result
        if proc.is_alive():
            proc.terminate()
        proc.join(1.0)
        logger.warning(
            "tiktoken encoding %r did not load from disk cache within %.1fs (stalled constructor); "
            "falling back to token estimation. Pre-populate TIKTOKEN_CACHE_DIR "
            "or tune HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS.",
            encoding_name,
            _load_timeout_seconds(),
        )
        # Do NOT add to _download_failed - disk load timeout is retryable
        # Do NOT add to _disk_load_permanently_failed - timeout is not corruption
        # Track in _disk_load_failed for observability only (NOT a short-circuit)
        _disk_load_failed.add(encoding_name)
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} disk load timed out")

    # Got result from queue, now wait for process to finish cleanly
    proc.join(1.0)

    status = result[0]
    if status == "ok":
        # Worker built and returned the encoding - return it regardless of exit code
        # (worker may have crashed after putting result but before clean exit)
        enc = result[1]
        # Register in tiktoken's in-memory cache for future fast loads
        tiktoken.registry.ENCODINGS[encoding_name] = enc
        # Clear any previous failure tracking on success (allows recovery from transient failures)
        _disk_load_failed.discard(encoding_name)
        return enc

    # Only check exit code for non-ok results
    if proc.exitcode != 0:
        logger.warning(
            "tiktoken encoding %r disk load process exited with code %d",
            encoding_name,
            proc.exitcode,
        )
        # Process crash could be transient - add to _disk_load_failed for tracking only
        _disk_load_failed.add(encoding_name)
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} disk load failed with exit code {proc.exitcode}")
    elif status == "invalid_cache":
        # Corrupted cache - permanent failure for offline, but online can repair by downloading
        if is_offline():
            _disk_load_permanently_failed.add(encoding_name)
            raise TiktokenLoadError(
                f"tiktoken encoding {encoding_name!r} has invalid cache entry and HEADROOM_OFFLINE=1"
            )
        # Online: fall through to download attempt (do NOT mark as permanently failed)
        logger.info(
            "tiktoken encoding %r has invalid cache entry; attempting online repair",
            encoding_name,
        )
        pass
    elif status == "not_cached":
        # Not cached locally - will try download if online
        # Cache miss is NOT a failure - it's expected on first use or cold cache
        # Do NOT add to any failure set; retry will work once cache is populated
        pass
    elif status == "error":
        _disk_load_permanently_failed.add(encoding_name)
        raise TiktokenLoadError(
            f"tiktoken encoding {encoding_name!r} failed to load from local cache: {result[1]}"
        )
    else:
        _disk_load_permanently_failed.add(encoding_name)
        raise TiktokenLoadError(
            f"tiktoken encoding {encoding_name!r} disk load returned unknown status: {status}"
        )

    # If we got here, encoding is not cached locally
    # If offline, fail fast
    if is_offline():
        raise TiktokenLoadError(
            f"tiktoken encoding {encoding_name!r} not in local cache and HEADROOM_OFFLINE=1"
        )

    # Online mode, not cached: need to download. Use bounded load with guard_egress.
    return _get_encoding_with_timeout(encoding_name, ctx)


def _get_encoding_with_timeout(encoding_name: str, ctx) -> Any:
    """Bounded tiktoken.get_encoding call for online mode with cold cache."""
    queue = ctx.Queue()
    proc = ctx.Process(target=_download_encoding_worker, args=(encoding_name, queue), daemon=False)
    proc.start()

    # Consume queue WHILE process is alive to ensure result is transferred
    try:
        result = queue.get(timeout=_load_timeout_seconds())
    except queue_module.Empty:
        # Timeout - process is still running or dead without putting result
        if proc.is_alive():
            proc.terminate()
        proc.join(1.0)
        logger.warning(
            "tiktoken encoding %r download timed out after %.1fs; falling back to estimation",
            encoding_name,
            _load_timeout_seconds(),
        )
        # Do NOT add to _download_failed on timeout - transient blip shouldn't ban forever
        # Only mark as failed on actual download error (see below)
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} download timed out")

    # Got result from queue, now wait for process to finish cleanly
    proc.join(1.0)

    if result[0] == "ok":
        # Download succeeded - use encoding returned by worker (avoids second load)
        enc = result[1]
        tiktoken.registry.ENCODINGS[encoding_name] = enc
        # Clear failure tracking on success (allows retry after transient failures)
        _disk_load_failed.discard(encoding_name)
        return enc

    # Only check exit code for non-ok results
    if proc.exitcode != 0:
        logger.warning(
            "tiktoken encoding %r download process exited with code %d",
            encoding_name,
            proc.exitcode,
        )
        _download_failed.add(encoding_name)
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} download failed with exit code {proc.exitcode}")

    _download_failed.add(encoding_name)
    raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} download failed: {result[1]}")


def _download_encoding_worker(encoding_name: str, queue: Any) -> None:
    """Worker to download encoding via tiktoken.get_encoding (online mode).

    Returns the encoding object directly to avoid second load in parent.
    """
    try:
        guard_egress("tiktoken encoding download")
        enc = tiktoken.get_encoding(encoding_name)
        # Verify it's in tiktoken's registry (download succeeded)
        if encoding_name in tiktoken.registry.ENCODINGS:
            queue.put(("ok", enc))
        else:
            queue.put(("error", "download succeeded but encoding not registered"))
    except BaseException as exc:  # noqa: BLE001
        queue.put(("error", str(exc)))


def load_encoding(encoding_name: str) -> Any:
    """Public, bounded tiktoken-encoding loader.

    Returns the tiktoken encoding, or raises :class:`TiktokenLoadError` if the
    vocab can't be loaded within the timeout (see :func:`_get_encoding`, GH #956).
    """
    return _get_encoding(encoding_name)


def get_encoding_for_model(model: str) -> str:
    """Get the tiktoken encoding name for a model.

    Args:
        model: Model name (e.g., 'gpt-4o', 'gpt-3.5-turbo').

    Returns:
        Encoding name (e.g., 'o200k_base', 'cl100k_base').
    """
    # Case-insensitive: TokenizerRegistry lowercases only its *cache key*, then
    # constructs the counter from the caller's original string. An uppercase
    # deployment name ("GPT-4o", routine on Azure) therefore arrived here
    # verbatim, matched no prefix, and silently took DEFAULT_ENCODING -- so the
    # encoding a model got depended on the casing of whichever request warmed
    # the cache first, and flipped across restarts.
    # Gateway-wrapped ids ("azure/gpt-4o", "openrouter/openai/gpt-5") are picked
    # for tiktoken by the registry's unwrapping but arrive here as-is, so match on
    # the bare id or they all take DEFAULT_ENCODING.
    model = model.lower().rsplit("/", 1)[-1]

    # Direct lookup
    if model in MODEL_TO_ENCODING:
        return MODEL_TO_ENCODING[model]

    # Try prefix matching for versioned models. Ordered most-specific first
    # so that, e.g., "gpt-4o-*" resolves before "gpt-4-*".
    for prefix, encoding in (
        ("gpt-4o", "o200k_base"),
        # chatgpt-4o-latest and dated chatgpt-4o-* aliases are GPT-4o.
        ("chatgpt-4o-", "o200k_base"),
        # gpt-4.1 / gpt-4.5 use o200k_base and MUST precede the "gpt-4" prefix,
        # which they would otherwise match and be mis-encoded as cl100k_base.
        ("gpt-4.1", "o200k_base"),
        ("gpt-4.5", "o200k_base"),
        # gpt-5 uses o200k_base. Without this it fell through to the
        # cl100k_base default, over-counting CJK by ~33%.
        ("gpt-5", "o200k_base"),
        ("gpt-4-turbo", "cl100k_base"),
        ("gpt-4", "cl100k_base"),
        ("gpt-3.5", "cl100k_base"),
        ("o1", "o200k_base"),
        ("o3", "o200k_base"),
        # o4 reasoning models use o200k_base; without this they fell through to
        # the cl100k_base default.
        ("o4", "o200k_base"),
    ):
        if model.startswith(prefix):
            return encoding

    return DEFAULT_ENCODING


class TiktokenCounter(BaseTokenizer):
    """Token counter using tiktoken (OpenAI's tokenizer).

    This is the most accurate tokenizer for OpenAI models and provides
    a good approximation for many other models that use similar BPE
    tokenization.

    Example:
        counter = TiktokenCounter("gpt-4o")
        tokens = counter.count_text("Hello, world!")
        print(f"Token count: {tokens}")
    """

    # OpenAI-specific message overhead
    MESSAGE_OVERHEAD = 3
    REPLY_OVERHEAD = 3

    def __init__(self, model: str = "gpt-4o", encoding: str | None = None):
        """Initialize tiktoken counter.

        Args:
            model: Model name to determine encoding.
                   Defaults to 'gpt-4o' (o200k_base encoding).
            encoding: Explicit tiktoken encoding name (e.g. 'o200k_base') that
                   overrides model-based resolution. Used to price
                   private-tokenizer models (Claude) against a real BPE proxy
                   instead of a character estimate.
        """
        self.model = model
        self.encoding_name = encoding or get_encoding_for_model(model)
        self._encoding = None  # Lazy load
        self._count_cache = TokenCountCache()

    @property
    def encoding(self):
        """Lazy-load the encoding."""
        if self._encoding is None:
            self._encoding = _get_encoding(self.encoding_name)
        return self._encoding

    def count_text(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to tokenize.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        cached = self._count_cache.get(text)
        if cached is not None:
            return cached
        count = self._count_text_uncached(text)
        self._count_cache.put(text, count)
        return count

    def _count_text_uncached(self, text: str) -> int:
        try:
            return len(self.encoding.encode(text))
        except ValueError:
            # Passthrough content can legitimately contain strings that look
            # like tiktoken special tokens (e.g. "<|endoftext|>" or FIM markers
            # in code/tool output). Treat them as ordinary text instead of
            # raising, which would otherwise abort token counting for the whole
            # request. Matches AnthropicTokenCounter.count_text.
            return len(self.encoding.encode(text, disallowed_special=()))

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in messages using OpenAI's exact formula.

        This matches OpenAI's token counting for chat completions.

        Args:
            messages: List of chat messages.

        Returns:
            Total token count.
        """
        total = 0

        for message in messages:
            # Every message has overhead for role and formatting
            total += self.MESSAGE_OVERHEAD

            for key, value in message.items():
                if value is None:
                    continue

                if key == "content":
                    if isinstance(value, dict):
                        # Unwrapped single block: price it like the list form (#4273).
                        value = [value]
                    if isinstance(value, str):
                        total += self.count_text(value)
                    elif isinstance(value, list):
                        # Multi-part content
                        for part in value:
                            if isinstance(part, dict):
                                if part.get("type") == "text":
                                    total += self.count_text(
                                        coerce_countable_text(part.get("text", ""))
                                    )
                                elif part.get("type") == "image_url":
                                    # Image tokens vary by detail level
                                    detail = part.get("image_url", {}).get("detail", "auto")
                                    if detail == "low":
                                        total += 85
                                    else:
                                        total += 170  # Base for high detail
                                else:
                                    # Any other block shape (Anthropic
                                    # image/tool_result/tool_use, Strands blocks)
                                    # is priced by the base handler, which uses a
                                    # bounded per-image/document estimate. Stringifying
                                    # it here would json-serialize a base64 blob and
                                    # count it as text — a 1MB image becomes ~330K
                                    # phantom tokens (the exact overcount base.py
                                    # _count_content_parts exists to prevent).
                                    total += self._count_content_parts([part])
                            elif isinstance(part, str):
                                total += self.count_text(part)
                    else:
                        total += self.count_text(coerce_countable_text(value))
                elif key == "role":
                    total += self.count_text(coerce_countable_text(value))
                elif key == "name":
                    total += self.count_text(coerce_countable_text(value))
                    total += 1  # Name adds 1 token
                elif key == "tool_calls":
                    for tool_call in value or []:
                        total += 3  # Tool call overhead
                        if "function" in tool_call:
                            func = tool_call["function"] or {}
                            total += self.count_text(coerce_countable_text(func.get("name")))
                            total += self.count_text(coerce_countable_text(func.get("arguments")))
                        if "id" in tool_call:
                            total += self.count_text(coerce_countable_text(tool_call["id"]))
                elif key == "tool_call_id":
                    total += self.count_text(coerce_countable_text(value))
                elif key == "function_call":
                    value = value or {}
                    total += self.count_text(coerce_countable_text(value.get("name")))
                    total += self.count_text(coerce_countable_text(value.get("arguments")))

        # Every reply is primed with assistant
        total += self.REPLY_OVERHEAD

        return total

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode.

        Returns:
            List of token IDs.
        """
        try:
            return self.encoding.encode(text)
        except ValueError:
            # See count_text: literal special-token strings in passthrough
            # content must be encoded as ordinary text, not rejected. The
            # round-trip through decode() is unaffected.
            return self.encoding.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text.
        """
        return self.encoding.decode(tokens)

    def __repr__(self) -> str:
        return f"TiktokenCounter(model={self.model!r}, encoding={self.encoding_name!r})"
