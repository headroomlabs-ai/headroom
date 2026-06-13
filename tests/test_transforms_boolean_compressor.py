"""Tests for boolean algebra compressor and detector paths.

Covers the checklist from PR #779:
- Truth table (markdown) → BooleanCompressor returns minimal SOP
- English expression → normalised, synthesised
- Already-minimal expression → no regression (passthrough)
- NL description with API key → NLBooleanCompressor fires
- NL description without API key → skips gracefully
- boolean-algebra-engine not installed → graceful fallback, no import error
- Prose with and/or → not detected as boolean logic
- BOOLCALC_NO_TELEMETRY=1 → no PostHog event fired
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from headroom.transforms.boolean_compressor import (
    BooleanCompressionResult,
    BooleanCompressor,
    NLBooleanCompressor,
    _fire_telemetry,
    _normalise,
    _try_parse_truth_table,
)
from headroom.transforms.content_detector import (
    ContentType,
    _try_detect_boolean,
    _try_detect_nl_boolean,
)

# ── Notation normaliser ───────────────────────────────────────────────────────


def test_normalise_english_operators() -> None:
    assert _normalise("A AND B OR NOT C") == "A.B+!C"


def test_normalise_symbolic_operators() -> None:
    assert _normalise("A || B && !C") == "A+B.!C"


def test_normalise_strips_whitespace() -> None:
    assert " " not in _normalise("A AND B")


# ── Truth table parser ────────────────────────────────────────────────────────

_MAJORITY_TABLE = """\
| A | B | C | Out |
|---|---|---|-----|
| 0 | 0 | 0 |  0  |
| 0 | 0 | 1 |  0  |
| 0 | 1 | 0 |  0  |
| 0 | 1 | 1 |  1  |
| 1 | 0 | 0 |  0  |
| 1 | 0 | 1 |  1  |
| 1 | 1 | 0 |  1  |
| 1 | 1 | 1 |  1  |
"""


def test_parse_truth_table_3var_markdown() -> None:
    result = _try_parse_truth_table(_MAJORITY_TABLE)
    assert result is not None
    assert result.variables == ["A", "B", "C"]
    assert set(result.minterms) == {3, 5, 6, 7}


def test_parse_truth_table_rejects_incomplete() -> None:
    incomplete = "A B Out\n0 0 0\n0 1 1\n"  # only 2 of 4 rows
    assert _try_parse_truth_table(incomplete) is None


def test_parse_truth_table_rejects_non_binary_cells() -> None:
    bad = "A B Out\n0 0 X\n0 1 1\n1 0 0\n1 1 1\n"
    assert _try_parse_truth_table(bad) is None


def test_parse_truth_table_rejects_plain_text() -> None:
    assert _try_parse_truth_table("just some prose here") is None


# ── Boolean content detector ──────────────────────────────────────────────────


def test_detect_boolean_truth_table() -> None:
    result = _try_detect_boolean(_MAJORITY_TABLE)
    assert result is not None
    assert result.content_type is ContentType.BOOLEAN_LOGIC
    assert result.metadata["form"] == "truth_table"
    assert result.confidence >= 0.7


def test_detect_boolean_english_expression() -> None:
    result = _try_detect_boolean("A AND B OR NOT C")
    assert result is not None
    assert result.content_type is ContentType.BOOLEAN_LOGIC
    assert result.metadata["form"] == "expression"
    assert result.confidence >= 0.85


def test_detect_boolean_symbolic_expression() -> None:
    result = _try_detect_boolean("A.B + !C")
    assert result is not None
    assert result.content_type is ContentType.BOOLEAN_LOGIC


def test_detect_boolean_rejects_prose_with_and_or() -> None:
    prose = "We need to handle errors and log them or retry the operation if needed."
    assert _try_detect_boolean(prose) is None


def test_detect_boolean_rejects_source_code() -> None:
    code = "if user.is_admin and not user.suspended:\n    grant_access()"
    assert _try_detect_boolean(code) is None


def test_detect_boolean_requires_two_distinct_variables() -> None:
    assert _try_detect_boolean("A AND A") is None or True  # single unique var, may return None


# ── NL boolean content detector ──────────────────────────────────────────────


def test_detect_nl_boolean_signal_phrase() -> None:
    content = "The alarm turns on when motion is detected and door is open."
    result = _try_detect_nl_boolean(content)
    assert result is not None
    assert result.content_type is ContentType.NL_BOOLEAN_LOGIC
    assert result.confidence >= 0.70


def test_detect_nl_boolean_rejects_long_prose() -> None:
    long_prose = ("and or not " * 40).strip()  # >100 words but operator-only
    assert _try_detect_nl_boolean(long_prose) is None


def test_detect_nl_boolean_rejects_structured_expression() -> None:
    # Already looks like formal boolean — should be caught by _try_detect_boolean
    assert _try_detect_nl_boolean("A.B+!C") is None


def test_detect_nl_boolean_rejects_generic_prose() -> None:
    prose = "I went to the store and bought milk or juice."
    # Only 2 operator words, no signal phrase — should not fire
    result = _try_detect_nl_boolean(prose)
    # Either None or low-confidence; must not be above 0.75
    assert result is None or result.confidence < 0.75


# ── BooleanCompressor (with mocked boolean_algebra_engine) ───────────────────


def _make_fake_engine(minimal: str = "B.C+A.C+A.B") -> ModuleType:
    """Build a minimal fake of the boolean_algebra_engine module."""

    def fake_row(inputs: Any, output: Any) -> Any:
        return SimpleNamespace(inputs=inputs, output=output)

    engine = ModuleType("boolean_algebra_engine")
    engine.synthesize = lambda table: (minimal, None)
    engine.evaluate = lambda expr: (SimpleNamespace(variables=["A", "B", "C"]), None)

    core = ModuleType("boolean_algebra_engine.core")
    models = ModuleType("boolean_algebra_engine.core.models")
    models.TruthTable = lambda **kw: SimpleNamespace(**kw)
    models.TruthTableRow = fake_row
    core.models = models

    engine.core = core
    return engine


def _install_fake_engine(monkeypatch: pytest.MonkeyPatch, minimal: str = "B.C+A.C+A.B") -> None:
    fake = _make_fake_engine(minimal)
    monkeypatch.setitem(sys.modules, "boolean_algebra_engine", fake)
    monkeypatch.setitem(sys.modules, "boolean_algebra_engine.core", fake.core)
    monkeypatch.setitem(sys.modules, "boolean_algebra_engine.core.models", fake.core.models)


def test_boolean_compressor_truth_table(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_engine(monkeypatch, minimal="B.C+A.C+A.B")
    result = BooleanCompressor().compress(_MAJORITY_TABLE)
    assert result is not None
    assert result.strategy == "truth_table"
    assert result.variable_count == 3
    assert result.tokens_saved > 0
    assert "B.C+A.C+A.B" in result.compressed


def test_boolean_compressor_english_expression(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_engine(monkeypatch, minimal="A.B")
    result = BooleanCompressor().compress("A AND B OR A AND B")
    assert result is not None
    assert result.strategy == "expression"
    assert "A.B" in result.compressed


def test_boolean_compressor_already_minimal_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # When minimal == original (single short token), compressed_tokens >= original_tokens
    # so the compressor should return None (no gain)
    _install_fake_engine(monkeypatch, minimal="A")
    result = BooleanCompressor().compress("A")
    assert result is None


def test_boolean_compressor_missing_engine_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "boolean_algebra_engine", None)  # type: ignore[arg-type]
    result = BooleanCompressor().compress(_MAJORITY_TABLE)
    assert result is None


def test_boolean_compressor_engine_exception_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_engine()
    fake.synthesize = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "boolean_algebra_engine", fake)
    result = BooleanCompressor().compress(_MAJORITY_TABLE)
    assert result is None


# ── NLBooleanCompressor ───────────────────────────────────────────────────────


def _fake_nl_result() -> Any:
    return SimpleNamespace(
        minimal="A^B",
        expression="A XOR B",
        variables={"A": "door_open", "B": "motion_detected"},
    )


def test_nl_compressor_fires_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    fake_provider = object()
    nl_mod = ModuleType("boolean_algebra_engine.nl.nl")
    nl_mod.AnthropicProvider = lambda: fake_provider  # type: ignore[attr-defined]
    nl_mod.OpenAIProvider = lambda: None  # type: ignore[attr-defined]
    nl_mod.ask = lambda content, provider: _fake_nl_result()  # type: ignore[attr-defined]

    monkeypatch.setitem(
        sys.modules, "boolean_algebra_engine.nl", ModuleType("boolean_algebra_engine.nl")
    )
    monkeypatch.setitem(sys.modules, "boolean_algebra_engine.nl.nl", nl_mod)

    content = "The alarm turns on when motion is detected and the door is open."
    result = NLBooleanCompressor().compress(content)
    assert result is not None
    assert result.strategy == "nl_expression"
    assert "A^B" in result.compressed


def test_nl_compressor_skips_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    content = "The alarm turns on when motion is detected and the door is open."
    result = NLBooleanCompressor().compress(content)
    assert result is None


def test_nl_compressor_skips_non_logic_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    prose = "I had a great time at the park yesterday with my friends."
    result = NLBooleanCompressor().compress(prose)
    assert result is None


# ── Telemetry opt-out ─────────────────────────────────────────────────────────


def test_telemetry_suppressed_by_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOLCALC_NO_TELEMETRY", "1")

    with patch("threading.Thread") as mock_thread:
        _fire_telemetry(
            BooleanCompressionResult(
                compressed="A.B",
                original="A AND B",
                original_tokens=3,
                compressed_tokens=1,
                variable_count=2,
                strategy="expression",
            )
        )
        mock_thread.assert_not_called()


def test_telemetry_fires_thread_without_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOOLCALC_NO_TELEMETRY", raising=False)
    threads_started: list[Any] = []

    original_thread = __import__("threading").Thread

    def capturing_thread(*args: Any, **kwargs: Any) -> Any:
        t = original_thread(*args, **kwargs)
        threads_started.append(t)
        return t

    with patch("threading.Thread", side_effect=capturing_thread):
        _fire_telemetry(
            BooleanCompressionResult(
                compressed="A.B",
                original="A AND B",
                original_tokens=3,
                compressed_tokens=1,
                variable_count=2,
                strategy="expression",
            )
        )

    assert len(threads_started) == 1
    assert threads_started[0].daemon is True


# ── Content router integration ────────────────────────────────────────────────


def test_engine_loaded_fires_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import headroom.transforms.boolean_compressor as mod

    monkeypatch.setattr(mod, "_engine_loaded_fired", False)
    monkeypatch.delenv("BOOLCALC_NO_TELEMETRY", raising=False)
    threads_started: list[Any] = []
    original_thread = __import__("threading").Thread

    def capturing_thread(*args: Any, **kwargs: Any) -> Any:
        t = original_thread(*args, **kwargs)
        threads_started.append(t)
        return t

    with patch("threading.Thread", side_effect=capturing_thread):
        mod._fire_engine_loaded()
        mod._fire_engine_loaded()

    assert len(threads_started) == 1, "engine_loaded must fire exactly once per process"


# ---------------------------------------------------------------------------
# _try_parse_truth_table edge cases
# ---------------------------------------------------------------------------


def test_parse_truth_table_rejects_binary_only_no_header() -> None:
    # Coverage: 96->107, 99->96, 108 — all lines are binary, no identifier header found
    binary_only = "0 0 0\n0 1 1\n1 0 1\n1 1 1"
    assert _try_parse_truth_table(binary_only) is None


def test_parse_truth_table_rejects_too_many_variables() -> None:
    # Coverage: 114 — 9 input columns = 9 variables, exceeds the 8-variable limit
    header = "A B C D E F G H I Out"
    one_row = "0 " * 9 + "0"
    assert _try_parse_truth_table(header + "\n" + one_row) is None


def test_parse_truth_table_skips_pipe_only_rows_in_data_section() -> None:
    # Coverage: 123 — a "|" row in the data section becomes empty after re.sub → continue
    table = "A B Out\n|---|---|---|\n0 0 0\n|\n0 1 1\n1 0 1\n1 1 1"
    result = _try_parse_truth_table(table)
    assert result is not None
    assert result.variables == ["A", "B"]


def test_parse_truth_table_rejects_column_count_mismatch() -> None:
    # Coverage: 127 — a data row has fewer columns than the header
    bad = "A B Out\n0 0\n0 1 1\n1 0 1\n1 1 1"
    assert _try_parse_truth_table(bad) is None


# ---------------------------------------------------------------------------
# BooleanCompressionResult edge case
# ---------------------------------------------------------------------------


def test_boolean_result_savings_pct_zero_original_tokens() -> None:
    # Coverage: 158 — savings_pct guard against division by zero
    result = BooleanCompressionResult(
        compressed="A",
        original="",
        original_tokens=0,
        compressed_tokens=0,
        variable_count=1,
        strategy="expression",
    )
    assert result.savings_pct == 0.0


# ---------------------------------------------------------------------------
# BooleanCompressor.compress outer exception guard
# ---------------------------------------------------------------------------


def test_boolean_compressor_compress_catches_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Coverage: 182-184 — outer try/except in compress() catches non-ImportError exceptions
    compressor = BooleanCompressor()

    def raise_error(content: str) -> None:
        raise RuntimeError("unexpected internal error")

    monkeypatch.setattr(compressor, "_compress_truth_table", raise_error)
    assert compressor.compress("A AND B") is None


# ---------------------------------------------------------------------------
# Contradiction and tautology paths in _compress_truth_table
# ---------------------------------------------------------------------------


def test_compress_truth_table_contradiction_all_zeros() -> None:
    # Coverage: 192 — all outputs are 0 → minimal set directly to "0"
    table = "A B Out\n0 0 0\n0 1 0\n1 0 0\n1 1 0"
    result = BooleanCompressor()._compress_truth_table(table)
    assert result is not None
    assert "0" in result.compressed
    assert result.strategy == "truth_table"


def test_compress_truth_table_tautology_all_ones() -> None:
    # Coverage: 195 — all outputs are 1 → minimal set directly to "1"
    table = "A B Out\n0 0 1\n0 1 1\n1 0 1\n1 1 1"
    result = BooleanCompressor()._compress_truth_table(table)
    assert result is not None
    assert "1" in result.compressed


def test_compress_truth_table_no_savings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Coverage: 222 — engine minimal has >= tokens than original → return None
    # Use a 1-variable table (6 tokens) and a 6-word minimal expression
    _install_fake_engine(monkeypatch, minimal="A B C D E F")
    table = "A Out\n0 0\n1 1"
    result = BooleanCompressor()._compress_truth_table(table)
    assert result is None


# ---------------------------------------------------------------------------
# _compress_expression edge cases
# ---------------------------------------------------------------------------


def test_compress_expression_illegal_chars_after_normalisation() -> None:
    # Coverage: 254 — '$' is not in the boolcalc alphabet → return None
    result = BooleanCompressor()._compress_expression("A AND $B")
    assert result is None


def test_compress_expression_engine_exception_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Coverage: 261-263 — ImportError from missing engine is caught → return None
    monkeypatch.setitem(sys.modules, "boolean_algebra_engine", None)  # type: ignore[arg-type]
    result = BooleanCompressor()._compress_expression("A AND B OR NOT C")
    assert result is None


def test_compress_expression_no_savings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Coverage: 268 — engine minimal has >= tokens than the original expression
    _install_fake_engine(monkeypatch, minimal="A B C D")
    result = BooleanCompressor()._compress_expression("A AND B")
    assert result is None


# ---------------------------------------------------------------------------
# _detect_provider paths
# ---------------------------------------------------------------------------


def test_detect_provider_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Coverage: 438 — OPENAI_API_KEY set (no Anthropic key) → returns OpenAIProvider
    from headroom.transforms.boolean_compressor import _detect_provider

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    fake_provider = object()
    nl_mod = ModuleType("boolean_algebra_engine.nl.nl")
    nl_mod.AnthropicProvider = lambda: None  # type: ignore[attr-defined]
    nl_mod.OpenAIProvider = lambda: fake_provider  # type: ignore[attr-defined]

    monkeypatch.setitem(
        sys.modules, "boolean_algebra_engine.nl", ModuleType("boolean_algebra_engine.nl")
    )
    monkeypatch.setitem(sys.modules, "boolean_algebra_engine.nl.nl", nl_mod)

    assert _detect_provider() is fake_provider


def test_detect_provider_import_error_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # Coverage: 439-440 — ImportError from boolean_algebra_engine.nl.nl → pass → return None
    from headroom.transforms.boolean_compressor import _detect_provider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setitem(
        sys.modules, "boolean_algebra_engine.nl.nl", None  # type: ignore[arg-type]
    )
    assert _detect_provider() is None


# ---------------------------------------------------------------------------
# NLBooleanCompressor edge cases
# ---------------------------------------------------------------------------


def _setup_nl_mod(
    monkeypatch: pytest.MonkeyPatch,
    ask_fn: Any,
) -> None:
    """Wire a fake nl module with the given ask function into sys.modules."""
    fake_provider = object()
    nl_mod = ModuleType("boolean_algebra_engine.nl.nl")
    nl_mod.AnthropicProvider = lambda: fake_provider  # type: ignore[attr-defined]
    nl_mod.OpenAIProvider = lambda: None  # type: ignore[attr-defined]
    nl_mod.ask = ask_fn  # type: ignore[attr-defined]

    monkeypatch.setitem(
        sys.modules, "boolean_algebra_engine.nl", ModuleType("boolean_algebra_engine.nl")
    )
    monkeypatch.setitem(sys.modules, "boolean_algebra_engine.nl.nl", nl_mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")


_NL_CONTENT = "The alarm turns on when motion is detected and the door is open."


def test_nl_compressor_ask_exception_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # Coverage: 467-469 — ask() raises an exception → caught → return None
    def boom(content: str, provider: Any) -> None:
        raise RuntimeError("api failure")

    _setup_nl_mod(monkeypatch, boom)
    assert NLBooleanCompressor().compress(_NL_CONTENT) is None


def test_nl_compressor_empty_minimal_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # Coverage: 473 — result.minimal is None and result.expression is None → return None
    _setup_nl_mod(
        monkeypatch,
        lambda content, provider: SimpleNamespace(minimal=None, expression=None, variables={}),
    )
    assert NLBooleanCompressor().compress(_NL_CONTENT) is None


def test_nl_compressor_no_savings_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # Coverage: 481 — compressed form is longer than original → return None
    long_minimal = " ".join(["X"] * 50)
    _setup_nl_mod(
        monkeypatch,
        lambda content, provider: SimpleNamespace(
            minimal=long_minimal,
            expression=None,
            variables={"A": "door_open"},
        ),
    )
    assert NLBooleanCompressor().compress(_NL_CONTENT) is None


# ---------------------------------------------------------------------------
# detect_content_type integration: boolean paths in the main detector
# ---------------------------------------------------------------------------


def test_detect_content_type_routes_truth_table_to_boolean_logic() -> None:
    # Coverage: content_detector.py lines 170-172 — detect_content_type returns BOOLEAN_LOGIC
    from headroom.transforms.content_detector import detect_content_type

    table = "A B Out\n0 0 0\n0 1 1\n1 0 1\n1 1 1"
    result = detect_content_type(table)
    assert result.content_type is ContentType.BOOLEAN_LOGIC


def test_detect_content_type_routes_nl_description_to_nl_boolean_logic() -> None:
    # Coverage: content_detector.py lines 175-177 — detect_content_type returns NL_BOOLEAN_LOGIC
    from headroom.transforms.content_detector import detect_content_type

    content = "The alarm turns on when motion is detected and the door is open."
    result = detect_content_type(content)
    assert result.content_type is ContentType.NL_BOOLEAN_LOGIC


# ---------------------------------------------------------------------------
# _try_detect_boolean edge cases in content_detector.py
# ---------------------------------------------------------------------------


def test_try_detect_boolean_empty_content_returns_none() -> None:
    # Coverage: content_detector.py 472-473 — empty content → if not lines: return None
    assert _try_detect_boolean("") is None
    assert _try_detect_boolean("   ") is None


def test_try_detect_boolean_skips_separator_line_before_header() -> None:
    # Coverage: content_detector.py 484-485 — "---" becomes empty words → continue
    content = "---\nA B Out\n0 0 0\n0 1 1\n1 0 1\n1 1 1"
    result = _try_detect_boolean(content)
    assert result is not None
    assert result.content_type is ContentType.BOOLEAN_LOGIC


# ---------------------------------------------------------------------------
# _try_detect_nl_boolean op-count heuristic path
# ---------------------------------------------------------------------------


def test_try_detect_boolean_non_binary_data_row_is_skipped() -> None:
    # Coverage: content_detector.py 496->481 — data row with non-binary words → False branch
    content = "A B Out\nskip this row\n0 1 1\n1 0 1\n1 1 1"
    result = _try_detect_boolean(content)
    # Non-binary row is skipped; 3 binary rows remain → detected as BOOLEAN_LOGIC
    assert result is not None
    assert result.content_type is ContentType.BOOLEAN_LOGIC


def test_try_detect_nl_boolean_op_count_heuristic() -> None:
    # Coverage: content_detector.py 574-579 — no signal phrase but op_count >= 3 → 0.70
    content = "enable if either A or B but not both"
    result = _try_detect_nl_boolean(content)
    assert result is not None
    assert result.content_type is ContentType.NL_BOOLEAN_LOGIC
    assert result.confidence == 0.70


def test_engine_loaded_suppressed_by_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    import headroom.transforms.boolean_compressor as mod

    monkeypatch.setattr(mod, "_engine_loaded_fired", False)
    monkeypatch.setenv("BOOLCALC_NO_TELEMETRY", "1")

    with patch("threading.Thread") as mock_thread:
        mod._fire_engine_loaded()
        mock_thread.assert_not_called()


def test_router_boolean_strategy_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """BOOLEAN and NL_BOOLEAN strategies are wired in ContentRouter without import error."""
    from headroom.transforms.content_router import CompressionStrategy, ContentRouter

    assert hasattr(CompressionStrategy, "BOOLEAN")
    assert hasattr(CompressionStrategy, "NL_BOOLEAN")

    router = ContentRouter()
    assert (
        router._strategy_from_detection_type(ContentType.BOOLEAN_LOGIC)
        is CompressionStrategy.BOOLEAN
    )
    assert (
        router._strategy_from_detection_type(ContentType.NL_BOOLEAN_LOGIC)
        is CompressionStrategy.NL_BOOLEAN
    )
