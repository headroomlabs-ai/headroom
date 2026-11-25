"""Worker functions for bounded tiktoken loading.

These run in a separate process via multiprocessing spawn context so they
can be terminated on timeout. The module is imported in the child process
so must be import-safe (no module-level side effects).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import tempfile
from typing import Any

import tiktoken

from headroom.offline import OfflineEgressBlocked

logger = logging.getLogger(__name__)


# Encoding metadata - must match tiktoken_ext.openai_public
_ENCODING_URLS = {
    "o200k_base": ("https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",),
    "o200k_harmony": ("https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",),
    "cl100k_base": ("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",),
    "p50k_base": ("https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",),
    "r50k_base": ("https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",),
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

# Only encodings where explicit_n_vocab == len(mergeable_ranks) + len(special_tokens)
# For cl100k_base, o200k_base, o200k_harmony, p50k_edit this does NOT hold (they have reserved indices)
# tiktoken's own constructors OMIT explicit_n_vocab for these encodings to skip validation.
# We do the same: only pass explicit_n_vocab for encodings where it matches.
_ENCODING_EXPLICIT_N_VOCAB = {
    "gpt2": 50257,
    "r50k_base": 50257,
    "p50k_base": 50281,
    # cl100k_base, o200k_base, o200k_harmony, p50k_edit: omitted (have reserved tokens)
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


def _load_mergeable_ranks_from_data_gym(vocab_path: str, encoder_path: str) -> dict[bytes, int]:
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


def _build_encoding_from_cache(encoding_name: str) -> Any:
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


def _load_encoding_worker(encoding_name: str, queue: Any) -> None:
    """Worker function to load encoding in a separate process.

    Uses multiprocessing so we can actually terminate on timeout.
    Does cache check AND load inside the timeout boundary.
    Returns the encoding object via queue on success, or status string on failure.
    """
    try:
        # Fast path: already in memory (won't happen in child but check anyway)
        if encoding_name in tiktoken.registry.ENCODINGS:
            enc = tiktoken.get_encoding(encoding_name)
            queue.put(("ok", enc))
            return

        # Check local cache FIRST (inside the timeout boundary)
        # This includes filesystem I/O: file open, read, hash verification
        is_cached, has_invalid = _encoding_cached_locally(encoding_name)
        if has_invalid:
            # Partial or corrupted cache - don't attempt to build
            queue.put(("invalid_cache", encoding_name))
            return

        if is_cached:
            # Cache exists and is valid - build encoding in worker (bounded)
            enc = _build_encoding_from_cache(encoding_name)
            queue.put(("ok", enc))
            return

        # Not cached, would need download - but we don't do that here
        queue.put(("not_cached", encoding_name))

    except OfflineEgressBlocked:
        raise
    except BaseException as exc:  # noqa: BLE001
        queue.put(("error", str(exc)))
