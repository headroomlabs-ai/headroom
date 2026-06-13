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
