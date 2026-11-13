"""Deterministic compression-ratio measurement for every text compressor.

Pipes fixed corpus content in and out of each compressor and reports
byte/token compression percentages. No LLM calls, no randomness, and no
network beyond an explicit Kompress ONNX download — the same corpus through
the same code always yields the same ratios, so a changeset that silently
shifts compression shows up as a diff against the committed baseline.

Consumers:

  * ``tests/test_compression_ratio_gate.py`` — the blocking per-PR gate
    (runs in the standard ``[dev]`` shard; optional-dependency compressors
    skip when their backend is absent).
  * ``tests/fixtures/compression_ratio_golden/_generate.py`` — regenerates
    ``cases.json`` + ``baseline.json`` when a ratio change is intended.
  * ``.github/workflows/eval.yml`` — the ``[all]``-extras PR lane, where
    tree-sitter / trafilatura / Kompress are present and get measured too.

Coverage notes:

  * ``image`` is deliberately out of scope: ``ImageCompressor`` transforms
    image *message blocks*, not piped text, so there is no str -> str ratio
    to measure here. Its fidelity is covered by the image test suite.
  * ``kompress`` is measured when its ONNX model can be loaded (local cache
    or a successful download); otherwise its cases skip rather than fail —
    the release-lane benchmarks are the authoritative ML signal.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any

#: Every compressor the gate measures, in reporting order. The pytest gate
#: asserts the committed corpus covers each of these, so a new built-in
#: compressor added to ``_BUILTIN_COMPRESSOR_DESCRIPTORS`` without corpus
#: cases fails loudly instead of drifting unmeasured.
COMPRESSOR_NAMES: tuple[str, ...] = (
    "smart_crusher",
    "text_crusher",
    "log",
    "search",
    "diff",
    "tabular",
    "config",
    "code_aware",
    "html",
    "kompress",
)

#: Compressors whose Python class delegates to ``headroom._core`` (PyO3) —
#: either directly or by bridging into one that does (tabular -> SmartCrusher).
#: They have no pure-Python fallback by design. (``config`` stays off this
#: list: its tiers are ``compact_lossless`` + the Python CompressionStore,
#: so it runs without the extension.)
_RUST_BACKED: frozenset[str] = frozenset(
    {
        "smart_crusher",
        "text_crusher",
        "log",
        "search",
        "diff",
        "tabular",
    }
)

#: Per-compressor absolute tolerance on measured ratios. Deterministic
#: compressors get a tight band (2 percentage points) so real drift trips the
#: gate while float/backend noise does not. Optional-dependency compressors
#: get a wider band: tree-sitter grammars, trafilatura, and the Kompress ONNX
#: backend can legitimately shift output a few points across versions.
RATIO_TOLERANCE: dict[str, float] = {
    "smart_crusher": 0.02,
    "text_crusher": 0.02,
    "log": 0.02,
    "search": 0.02,
    "diff": 0.02,
    "tabular": 0.02,
    "config": 0.02,
    "code_aware": 0.05,
    "html": 0.10,
    "kompress": 0.10,
}
DEFAULT_TOLERANCE: float = 0.02


class CompressorUnavailable(RuntimeError):
    """Raised by :func:`measure` when a compressor cannot run in this env.

    The pytest gate converts this into a skip, so a missing optional
    backend never fails a PR — but a backend that IS present must produce
    numbers inside the baseline band.
    """


@dataclass
class CaseMeasurement:
    """Measured result of piping one corpus case through one compressor."""

    bytes_before: int
    bytes_after: int
    tokens_before: int
    tokens_after: int
    byte_ratio: float  # fraction of bytes removed (1.0 = total removal)
    token_ratio: float  # fraction of estimated tokens removed
    modified: bool  # compressor changed the content at all

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_tokens(text: str) -> int:
    """Token estimate via the shared estimator the content router uses.

    Deterministic (a fixed heuristic, no model), so the same text always
    estimates identically across runs and platforms.
    """
    from headroom.tokenizers.estimator import EstimatingTokenCounter

    return EstimatingTokenCounter().count_text(text)


# ── Availability probes ────────────────────────────────────────────────────


def _rust_core_available() -> tuple[bool, str]:
    try:
        import headroom._core  # noqa: F401
    except ImportError:
        return False, "headroom._core (Rust extension) is not built/installed"
    return True, ""


def _code_aware_available() -> tuple[bool, str]:
    try:
        from headroom.transforms.code_compressor import _check_tree_sitter_available
    except ImportError as e:
        return False, f"code compressor import failed: {e}"
    if not _check_tree_sitter_available():
        return False, "tree-sitter unavailable (pip install headroom-ai[code])"
    return True, ""


def _html_available() -> tuple[bool, str]:
    try:
        import trafilatura  # noqa: F401
    except ImportError:
        return False, "trafilatura unavailable (pip install headroom-ai[html])"
    return True, ""


def _kompress_available() -> tuple[bool, str]:
    try:
        from headroom.transforms.kompress_compressor import is_kompress_available
    except ImportError as e:
        return False, f"kompress import failed: {e}"
    if not is_kompress_available():
        return False, "kompress deps unavailable (pip install headroom-ai[all])"
    return True, ""


_AVAILABILITY_PROBES = {
    "code_aware": _code_aware_available,
    "html": _html_available,
    "kompress": _kompress_available,
}


def compressor_available(name: str) -> tuple[bool, str]:
    """Whether ``name`` can run here; the reason string when it cannot."""
    if name in _RUST_BACKED:
        return _rust_core_available()
    probe = _AVAILABILITY_PROBES.get(name)
    if probe is not None:
        return probe()
    if name in COMPRESSOR_NAMES:
        return True, ""  # pure-Python path (currently: config)
    return False, f"unknown compressor {name!r}"


# ── Compressor invocation (cached instances; lazy construction) ────────────

_compressors: dict[str, Any] = {}
_compressors_lock = threading.Lock()


def _get_compressor(name: str) -> Any:
    """Lazily construct and cache the compressor instance for ``name``."""
    with _compressors_lock:
        if name in _compressors:
            return _compressors[name]

        obj: Any
        if name == "smart_crusher":
            from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

            obj = SmartCrusher(SmartCrusherConfig())
        elif name == "text_crusher":
            from headroom.transforms.text_crusher import TextCrusher

            obj = TextCrusher()
        elif name == "log":
            from headroom.transforms.log_compressor import LogCompressor

            obj = LogCompressor()
        elif name == "search":
            from headroom.transforms.search_compressor import SearchCompressor

            obj = SearchCompressor()
        elif name == "diff":
            from headroom.transforms.diff_compressor import DiffCompressor

            obj = DiffCompressor()
        elif name == "tabular":
            from headroom.transforms.tabular_ingest import TabularCompressor

            obj = TabularCompressor()
        elif name == "config":
            from headroom.transforms.config_compressor import ConfigCompressor

            obj = ConfigCompressor()
        elif name == "code_aware":
            from headroom.transforms.code_compressor import (
                CodeAwareCompressor,
                CodeCompressorConfig,
            )

            # No Kompress fallback: unknown/unparseable languages must pass
            # through deterministically instead of hitting an ML model.
            obj = CodeAwareCompressor(CodeCompressorConfig(fallback_to_kompress=False))
        elif name == "html":
            from headroom.transforms.html_extractor import HTMLExtractor

            obj = HTMLExtractor()
        elif name == "kompress":
            from headroom.transforms.kompress_compressor import (
                KompressCompressor,
                KompressConfig,
            )

            obj = KompressCompressor(KompressConfig(enable_ccr=False))
        else:
            raise CompressorUnavailable(f"unknown compressor {name!r}")

        _compressors[name] = obj
        return obj


def compress_content(name: str, content: str, params: dict[str, Any] | None = None) -> str:
    """Run compressor ``name`` over ``content`` and return the output text.

    ``params`` carries the optional knobs a case may set: ``query`` (relevance
    hint, empty by default to keep scoring deterministic without embedding
    deps), ``bias``, and ``language`` (code_aware only).
    """
    params = params or {}
    query = str(params.get("query") or "")
    bias = float(params.get("bias") or 1.0)
    compressor = _get_compressor(name)

    if name == "smart_crusher":
        return str(compressor.crush(content, query=query, bias=bias).compressed)
    if name == "text_crusher":
        return str(compressor.compress(content, context=query).compressed)
    if name == "log":
        return str(compressor.compress(content, context=query, bias=bias).compressed)
    if name == "search":
        return str(compressor.compress(content, context=query, bias=bias).compressed)
    if name == "diff":
        return str(compressor.compress(content, context=query).compressed)
    if name == "tabular":
        return str(compressor.compress(content, context=query, bias=bias).compressed)
    if name == "config":
        return str(compressor.compress(content, context=query, bias=bias).compressed)
    if name == "code_aware":
        language = params.get("language")
        return str(compressor.compress(content, language=language, context=query).compressed)
    if name == "html":
        extracted = compressor.extract(content).extracted
        # trafilatura can return an empty string for content-free pages; the
        # router treats a None/empty extraction as passthrough, so the gate
        # measures the same thing the router ships.
        return extracted if extracted else content
    if name == "kompress":
        try:
            return str(compressor.compress(content, context=query).compressed)
        except Exception as e:
            raise CompressorUnavailable(f"kompress model unavailable: {e}") from e
    raise CompressorUnavailable(f"unknown compressor {name!r}")


def measure(name: str, content: str, params: dict[str, Any] | None = None) -> CaseMeasurement:
    """Pipe ``content`` through compressor ``name`` and measure the ratios."""
    ok, reason = compressor_available(name)
    if not ok:
        raise CompressorUnavailable(reason)

    compressed = compress_content(name, content, params)

    bytes_before = len(content.encode("utf-8"))
    bytes_after = len(compressed.encode("utf-8"))
    tokens_before = estimate_tokens(content)
    tokens_after = estimate_tokens(compressed)

    return CaseMeasurement(
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        byte_ratio=(1 - bytes_after / bytes_before) if bytes_before else 0.0,
        token_ratio=(1 - tokens_after / tokens_before) if tokens_before else 0.0,
        modified=compressed != content,
    )


def measure_case(case: dict[str, Any]) -> CaseMeasurement:
    """Measure one corpus case (``compressor`` + ``content`` + ``params``)."""
    return measure(case["compressor"], case["content"], case.get("params"))


def run_cases(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Measure every case; returns ``{"compressor/case_id": measurement}``.

    Cases whose compressor is unavailable in this environment are recorded
    with ``{"unavailable": reason}`` so the generated baseline documents the
    gap instead of silently dropping the case.
    """
    results: dict[str, dict[str, Any]] = {}
    for case in cases:
        key = f"{case['compressor']}/{case['id']}"
        try:
            results[key] = measure_case(case).to_dict()
        except CompressorUnavailable as e:
            results[key] = {"unavailable": str(e)}
    return results
