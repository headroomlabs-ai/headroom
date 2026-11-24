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

import base64
import hashlib
import json
import logging
import multiprocessing as mp
import os
import tempfile
from functools import lru_cache
from typing import Any

import tiktoken

from headroom.offline import guard_egress, is_offline

from .base import BaseTokenizer, TokenCountCache, coerce_countable_text

logger = logging.getLogger(__name__)


class TiktokenLoadError(RuntimeError):
    """Raised when a tiktoken encoding can't be loaded in time.

    tiktoken downloads its BPE vocab on first use via ``requests.get`` with no
    timeout, so a stalled/firewalled connection can block indefinitely. We bound
    that load and raise this instead, so callers fall back to estimation rather
    than hanging the request (see GH #956).
    """


# Encoding names whose bounded *download* already timed out — don't block on them again.
# We do NOT add disk-load misses here; they remain retryable once the vocabulary is added.
# Track failed DOWNLOADS separately from disk load misses
# _download_failed: download failures (network issues, etc.) - blocks retries
# _disk_load_failed: disk load timeout/miss - does NOT block retries, retryable after vocab available
_download_failed: set[str] = set()
_disk_load_failed: set[str] = set()


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


# Encoding constructor metadata from tiktoken_ext.openai_public
_ENCODING_URLS = {
    "o200k_base": (
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    ),
    "o200k_harmony": (
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    ),
    "cl100k_base": (
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    ),
    "p50k_base": (
        "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
    ),
    "r50k_base": (
        "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
    ),
    "gpt2": (
        "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
        "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
    ),
}

_ENCODING_HASHES = {
    "o200k_base": "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    "o200k_harmony": "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    "cl100k_base": "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    "p50k_base": "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
    "r50k_base": "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
    "gpt2_vocab": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "gpt2_encoder": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
}

_ENCODING_SPECIAL_TOKENS = {
    "o200k_base": {"<|endoftext|>": 199999, "<|endofprompt|>": 200018},
    "o200k_harmony": {
        "<|endoftext|>": 199999,
        "<|endofprompt|>": 200018,
        "<|startoftext|>": 199998,
        "<|reserved_200000|>": 200000,
        "<|reserved_200001|>": 200001,
        "<|return|>": 200002,
        "<|constrain|>": 200003,
        "<|reserved_200004|>": 200004,
        "<|channel|>": 200005,
        "<|start|>": 200006,
        "<|end|>": 200007,
        "<|message|>": 200008,
        "<|reserved_200009|>": 200009,
        "<|reserved_200010|>": 200010,
        "<|reserved_200011|>": 200011,
        "<|call|>": 200012,
    }
    | {f"<|reserved_{i}|>": i for i in range(200013, 201088)},
    "cl100k_base": {
        "<|endoftext|>": 100257,
        "<|fim_prefix|>": 100258,
        "<|fim_middle|>": 100259,
        "<|fim_suffix|>": 100260,
        "<|endofprompt|>": 100276,
    },
    "p50k_base": {"<|endoftext|>": 50256},
    "r50k_base": {"<|endoftext|>": 50256},
    "p50k_edit": {
        "<|endoftext|>": 50256,
        "<|fim_prefix|>": 50281,
        "<|fim_middle|>": 50282,
        "<|fim_suffix|>": 50283,
    },
    "gpt2": {"<|endoftext|>": 50256},
}

_ENCODING_PAT_STR = {
    "o200k_base": "|".join(
        [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    ),
    "o200k_harmony": "|".join(
        [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    ),
    "cl100k_base": r"""(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""",
    "p50k_base": r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""",
    "r50k_base": r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""",
    "p50k_edit": r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""",
    "gpt2": r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""",
}

_ENCODING_EXPLICIT_N_VOCAB = {
    "o200k_base": 200019,
    "o200k_harmony": 201088,
    "cl100k_base": 100277,
    "p50k_base": 50281,
    "r50k_base": 50257,
    "p50k_edit": 50284,
    "gpt2": 50257,
}


def _tiktoken_cache_dir() -> str:
    """Get the tiktoken cache directory, matching tiktoken's logic."""
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        return os.environ["TIKTOKEN_CACHE_DIR"]
    if "DATA_GYM_CACHE_DIR" in os.environ:
        return os.environ["DATA_GYM_CACHE_DIR"]
    return os.path.join(tempfile.gettempdir(), "data-gym-cache")


def _cache_key(url: str) -> str:
    """Compute tiktoken's cache key for a URL (SHA1 of the URL)."""
    return hashlib.sha1(url.encode()).hexdigest()


def _check_hash(data: bytes, expected_hash: str) -> bool:
    """Verify SHA256 hash matches expected."""
    return hashlib.sha256(data).hexdigest() == expected_hash


def _read_cached_file(path: str, expected_hash: str | None = None) -> bytes | None:
    """Read a cached file if it exists and hash matches.

    Returns None if file doesn't exist or hash mismatches.
    Does NOT trigger any download.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb", buffering=0) as f:
            data = f.read()
    except OSError:
        return None
    if expected_hash and not _check_hash(data, expected_hash):
        return None
    return data


def _load_mergeable_ranks_from_tiktoken_file(path: str) -> dict[bytes, int]:
    """Load mergeable ranks from a .tiktoken file (base64-encoded tokens with ranks)."""
    ret: dict[bytes, int] = {}
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                token, rank = line.split()
                ret[base64.b64decode(token)] = int(rank)
            except Exception as e:
                raise ValueError(f"Error parsing line {line!r} in {path}") from e
    return ret


def _load_mergeable_ranks_from_data_gym(
    vocab_path: str, encoder_path: str
) -> dict[bytes, int]:
    """Load mergeable ranks from GPT-2 style vocab.bpe + encoder.json (data-gym format)."""
    # From tiktoken/load.py data_gym_to_mergeable_bpe_ranks
    rank_to_intbyte = [b for b in range(2**8) if chr(b).isprintable() and chr(b) != " "]
    data_gym_byte_to_byte = {chr(b): b for b in rank_to_intbyte}
    n = 0
    for b in range(2**8):
        if b not in rank_to_intbyte:
            rank_to_intbyte.append(b)
            data_gym_byte_to_byte[chr(2**8 + n)] = b
            n += 1
    assert len(rank_to_intbyte) == 2**8

    with open(vocab_path, encoding="utf-8") as f:
        vocab_bpe_contents = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in vocab_bpe_contents.split("\n")[1:-1]]

    def decode_data_gym(value: str) -> bytes:
        return bytes(data_gym_byte_to_byte[b] for b in value)

    bpe_ranks: dict[bytes, int] = {bytes([b]): i for i, b in enumerate(rank_to_intbyte)}

    n = len(bpe_ranks)
    for first, second in bpe_merges:
        bpe_ranks[decode_data_gym(first) + decode_data_gym(second)] = n
        n += 1

    with open(encoder_path, encoding="utf-8") as f:
        encoder_json = json.load(f)
    encoder_json_loaded = {decode_data_gym(k): v for k, v in encoder_json.items()}
    encoder_json_loaded.pop(b"<|endoftext|>", None)
    encoder_json_loaded.pop(b"<|startoftext|>", None)

    assert bpe_ranks == encoder_json_loaded, "vocab.bpe and encoder.json mismatch"
    return bpe_ranks


def _encoding_cached_locally(encoding_name: str) -> tuple[bool, bool]:
    """Check if a tiktoken encoding is available in local cache (memory or disk).

    Returns (is_cached, has_invalid_cache_entry).
    - is_cached: True if the encoding can be loaded without network access.
    - has_invalid_cache_entry: True if a cache file exists but is invalid (hash mismatch)
      OR if a partial cache exists (e.g., only vocab.bpe for GPT-2 without encoder.json).
    """
    # Check in-memory cache first
    try:
        import tiktoken

        if encoding_name in tiktoken.registry.ENCODINGS:
            return True, False
    except Exception:
        pass

    # Check on-disk cache
    urls = _ENCODING_URLS.get(encoding_name)
    if not urls:
        return False, False

    cache_dir = _tiktoken_cache_dir()
    if cache_dir == "":
        return False, False

    has_invalid = False

    # GPT-2 is the only encoding with multiple files (vocab.bpe + encoder.json)
    if encoding_name == "gpt2" and isinstance(urls, tuple):
        all_exist = True
        for i, url in enumerate(urls):
            cache_path = os.path.join(cache_dir, _cache_key(url))
            if i == 0:  # vocab.bpe
                expected_hash = _ENCODING_HASHES.get("gpt2_vocab")
            else:  # encoder.json
                expected_hash = _ENCODING_HASHES.get("gpt2_encoder")
            if not os.path.exists(cache_path):
                all_exist = False
                # Partial cache exists - treat as invalid (cache-only, no download)
                has_invalid = True
            elif expected_hash:
                data = _read_cached_file(cache_path, expected_hash)
                if data is None:
                    has_invalid = True
        if not all_exist:
            # Partial cache: has some files but not all - cache-only, no download
            return True, True
        return True, has_invalid

    # Single file encodings (.tiktoken) - urls may be a single-element tuple or string
    url = urls[0] if isinstance(urls, tuple) else urls
    cache_path = os.path.join(cache_dir, _cache_key(url))
    if not os.path.exists(cache_path):
        return False, False

    expected_hash = _ENCODING_HASHES.get(encoding_name)
    if expected_hash:
        data = _read_cached_file(cache_path, expected_hash)
        if data is None:
            return False, True

    return True, False


def _build_encoding_from_cache(encoding_name: str):
    """Construct a tiktoken Encoding directly from local cache files.

    This bypasses tiktoken's download logic entirely. Raises if cache files
    are missing or invalid.
    """
    from tiktoken.core import Encoding

    urls = _ENCODING_URLS.get(encoding_name)
    if not urls:
        raise ValueError(f"Unknown encoding {encoding_name}")

    cache_dir = _tiktoken_cache_dir()

    # GPT-2 is the only encoding with multiple files (vocab.bpe + encoder.json)
    if encoding_name == "gpt2" and isinstance(urls, tuple):
        # GPT-2: two files
        vocab_path = os.path.join(cache_dir, _cache_key(urls[0]))
        encoder_path = os.path.join(cache_dir, _cache_key(urls[1]))
        mergeable_ranks = _load_mergeable_ranks_from_data_gym(vocab_path, encoder_path)
    else:
        # Single file encodings (.tiktoken) - urls may be a single-element tuple or string
        url = urls[0] if isinstance(urls, tuple) else urls
        cache_path = os.path.join(cache_dir, _cache_key(url))
        mergeable_ranks = _load_mergeable_ranks_from_tiktoken_file(cache_path)

    special_tokens = _ENCODING_SPECIAL_TOKENS.get(encoding_name, {})
    pat_str = _ENCODING_PAT_STR.get(encoding_name, _ENCODING_PAT_STR["cl100k_base"])
    explicit_n_vocab = _ENCODING_EXPLICIT_N_VOCAB.get(encoding_name)

    return Encoding(
        name=encoding_name,
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
        explicit_n_vocab=explicit_n_vocab,
    )


def _load_encoding_worker(encoding_name: str, queue: mp.Queue) -> None:
    """Worker function to load encoding in a separate process.

    Uses multiprocessing so we can actually terminate on timeout.
    Does cache check AND load inside the timeout boundary.
    """
    try:
        import tiktoken

        # Fast path: already in memory
        if encoding_name in tiktoken.registry.ENCODINGS:
            queue.put(("ok", encoding_name))
            return

        # Check local cache FIRST (inside the timeout boundary)
        # This includes filesystem I/O: file open, read, hash verification
        is_cached, has_invalid = _encoding_cached_locally(encoding_name)
        if is_cached:
            # Load directly from cache files - NO network possible
            enc = _build_encoding_from_cache(encoding_name)
            # Register in tiktoken's in-memory cache for future fast loads
            tiktoken.registry.ENCODINGS[encoding_name] = enc
            queue.put(("ok", encoding_name))
            return

        if has_invalid:
            queue.put(("invalid_cache", encoding_name))
            return

        # Not cached, would need download - but we don't do that here
        queue.put(("not_cached", encoding_name))

    except BaseException as exc:  # noqa: BLE001
        queue.put(("error", str(exc)))


def _download_worker(enc_name: str, q: mp.Queue) -> None:
    """Download a tiktoken encoding in a subprocess."""
    try:
        guard_egress("load tiktoken encoding", destination=f"tiktoken encoding {enc_name}")
        import tiktoken as _tiktoken

        _tiktoken.get_encoding(enc_name)
        q.put(("ok", enc_name))
    except BaseException as exc:  # noqa: BLE001
        q.put(("error", str(exc)))


def _get_encoding_with_timeout(encoding_name: str, ctx: Any) -> Any:
    """Load a tiktoken encoding with a bounded timeout.

    Uses a subprocess to prevent a stalled vocab download from blocking
    the caller indefinitely (GH #956).
    """
    if is_offline() and encoding_name not in _download_failed:
        raise TiktokenLoadError(
            f"tiktoken encoding {encoding_name!r} not in local cache and HEADROOM_OFFLINE=1"
        )

    # Online mode, not cached: need to download. Use bounded load with guard_egress.
    queue = ctx.Queue()

    proc = ctx.Process(target=_download_worker, args=(encoding_name, queue), daemon=True)
    proc.start()
    proc.join(_load_timeout_seconds())

    if proc.is_alive():
        proc.terminate()
        proc.join(1.0)
        _download_failed.add(encoding_name)
        logger.warning(
            "tiktoken encoding %r did not load within %.1fs (likely a stalled vocab "
            "download); falling back to token estimation. Pre-populate TIKTOKEN_CACHE_DIR "
            "or tune HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS.",
            encoding_name,
            _load_timeout_seconds(),
        )
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} load timed out")

    if proc.exitcode != 0:
        _download_failed.add(encoding_name)
        logger.warning(
            "tiktoken encoding %r download process exited with code %d",
            encoding_name,
            proc.exitcode,
        )
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} download failed")

    try:
        result = queue.get_nowait()
    except Exception as e:
        _download_failed.add(encoding_name)
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} download returned no result") from e

    status = result[0]
    if status == "ok":
        return tiktoken.get_encoding(encoding_name)
    elif status == "error":
        _download_failed.add(encoding_name)
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} failed to download: {result[1]}")
    else:
        _download_failed.add(encoding_name)
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} download returned unknown status: {status}")


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

    # Check if this encoding previously failed to DOWNLOAD
    if encoding_name in _download_failed:
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} previously failed to download")

    # Try to load from disk cache with timeout using multiprocessing (can actually terminate)
    # This moves the cache check AND the load inside the timeout boundary
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_load_encoding_worker, args=(encoding_name, queue), daemon=True)
    proc.start()
    proc.join(_load_timeout_seconds())

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
        # Add to _disk_load_failed for tracking
        _disk_load_failed.add(encoding_name)
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} disk load timed out")

    if proc.exitcode != 0:
        logger.warning(
            "tiktoken encoding %r disk load process exited with code %d",
            encoding_name,
            proc.exitcode,
        )
        _disk_load_failed.add(encoding_name)
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} disk load failed")

    try:
        result = queue.get_nowait()
    except Exception as e:
        _disk_load_failed.add(encoding_name)
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} disk load returned no result") from e

    status = result[0]
    if status == "ok":
        return tiktoken.get_encoding(encoding_name)
    elif status == "invalid_cache":
        # If online, try to download fresh; if offline, fail fast (cache-only)
        if is_offline():
            raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} has invalid cache entry and HEADROOM_OFFLINE=1")
        # Online: fall through to download attempt
        pass
    elif status == "not_cached":
        # Not cached locally - will try download if online
        pass
    elif status == "error":
        _disk_load_failed.add(encoding_name)
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} failed to load from local cache: {result[1]}")
    else:
        _disk_load_failed.add(encoding_name)
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} disk load returned unknown status: {status}")

    # If we got here, encoding is not cached locally
    # If offline, fail fast
    if is_offline():
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} not in local cache and HEADROOM_OFFLINE=1")

    # Online mode, not cached: need to download. Use bounded load with guard_egress.
    return _get_encoding_with_timeout(encoding_name, ctx)


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
