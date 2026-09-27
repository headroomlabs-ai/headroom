"""Env-driven config for the edge layer.

Every edge is opt-in and independent so the replay harness can score
"one flag at a time, then all together" per the test protocol's Phase 2. Nothing
here reads config more than once per request: the pipeline extension is
re-constructed per-process (not per-request), so config is read at
``EdgeLayerConfig.from_env()`` call time — call it once, at extension
construction, and pass the same instance to every edge for a given process
lifetime. Changing an env var mid-session and expecting mid-session behavior
to change is explicitly unsupported: config must be stable for a session,
or message k stops being byte-identical across turns (cache-safety rule #2).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_TRUE = {"1", "true", "yes", "on"}


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class EdgeLayerConfig:
    """Frozen so a config identity can be folded into `explicit_hash` safely."""

    # Per-edge on/off flags. Each is independently scoreable in the replay harness.
    enable_intent_query: bool = True
    enable_grep: bool = True
    enable_file_read: bool = True
    enable_rerank: bool = True

    # Cache-safety rule #2: config version folded into every stored hash so a
    # config change never silently reuses a stale entry under the same key.
    config_version: str = "v1"

    # Rerank edge (logs/web pages).
    rerank_target_ratio: float = 0.30
    rerank_min_tokens_kept: int = 1500
    rerank_trigger_tokens: int = 2000
    rerank_chunk_max_lines: int = 30
    rerank_keep_head_lines: int = 20
    rerank_keep_tail_lines: int = 20
    rerank_context_lines: int = 3
    # HTTP reranker service. Empty => use the deterministic token-overlap
    # fallback scorer (a stand-in for wiring only, see rerank_edge.py).
    reranker_url: str = ""
    reranker_timeout_seconds: float = 0.15

    # Grep edge.
    grep_per_file_cap: int = 20

    # File-read edge.
    file_read_diff_max_ratio: float = 0.30
    # CCR entries live 1800s by default (DEFAULT_CCR_TTL_SECONDS); refresh well
    # under that so a marker replayed every turn never goes stale mid-session.
    file_read_marker_ttl_seconds: int = 1800

    # CPU-side input caps (see cache_safety.fail_open_cpu): a timeout can't
    # preempt synchronous Python, so bound the work instead of the wall clock.
    max_input_chars_per_edge: int = 2_000_000

    # Tool names treated as "reads a file" for the file-read edge. Client
    # tool names drift across agent/CLI versions (Read, view, read_file), so
    # this is a configurable set rather than a hardcoded literal.
    read_tool_names: frozenset[str] = frozenset(
        {"Read", "read", "read_file", "view", "cat", "fs_read"}
    )

    @classmethod
    def from_env(cls) -> EdgeLayerConfig:
        return cls(
            enable_intent_query=_flag("HEADROOM_EDGE_INTENT_QUERY", True),
            enable_grep=_flag("HEADROOM_EDGE_GREP", True),
            enable_file_read=_flag("HEADROOM_EDGE_FILE_READ", True),
            enable_rerank=_flag("HEADROOM_EDGE_RERANK", True),
            config_version=os.environ.get("HEADROOM_EDGE_CONFIG_VERSION", "v1"),
            rerank_target_ratio=_float("HEADROOM_EDGE_RERANK_TARGET_RATIO", 0.30),
            rerank_min_tokens_kept=_int("HEADROOM_EDGE_RERANK_MIN_TOKENS", 1500),
            rerank_trigger_tokens=_int("HEADROOM_EDGE_RERANK_TRIGGER_TOKENS", 2000),
            reranker_url=os.environ.get("HEADROOM_EDGE_RERANKER_URL", ""),
            reranker_timeout_seconds=_float("HEADROOM_EDGE_RERANKER_TIMEOUT_S", 0.15),
            grep_per_file_cap=_int("HEADROOM_EDGE_GREP_PER_FILE_CAP", 20),
            file_read_diff_max_ratio=_float("HEADROOM_EDGE_FILE_READ_DIFF_MAX_RATIO", 0.30),
            file_read_marker_ttl_seconds=_int("HEADROOM_EDGE_FILE_READ_TTL_S", 1800),
            max_input_chars_per_edge=_int("HEADROOM_EDGE_MAX_INPUT_CHARS", 2_000_000),
            read_tool_names=_read_tool_names(),
        )


def _read_tool_names() -> frozenset[str]:
    raw = os.environ.get("HEADROOM_EDGE_READ_TOOL_NAMES")
    if not raw or not raw.strip():
        return frozenset({"Read", "read", "read_file", "view", "cat", "fs_read"})
    return frozenset(name.strip() for name in raw.split(",") if name.strip())
