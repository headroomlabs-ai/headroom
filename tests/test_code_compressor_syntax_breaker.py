"""Per-language circuit breaker for AST compression that keeps failing.

The compressor validates its own output and discards invalid results, so a
misbehaving grammar burns the full compression latency on every request and
throws the work away (field measurement: 270 discarded TypeScript compressions
in one install's retained logs at ~1.2s each). After enough failures in the
recent window the breaker pauses that language for a cooldown; other languages
and the original content are unaffected.
"""

from __future__ import annotations

import logging

import pytest

from headroom.transforms import code_compressor as cc


@pytest.fixture(autouse=True)
def reset_breaker_state():
    cc._syntax_breaker_outcomes.clear()
    cc._syntax_breaker_open_until.clear()
    cc._syntax_breaker_trips.clear()
    yield
    cc._syntax_breaker_outcomes.clear()
    cc._syntax_breaker_open_until.clear()
    cc._syntax_breaker_trips.clear()


def test_breaker_trips_after_min_failures_in_window():
    for _ in range(cc._SYNTAX_BREAKER_MIN_FAILURES - 1):
        cc._record_syntax_outcome("typescript", False)
    assert not cc._syntax_breaker_open("typescript")
    cc._record_syntax_outcome("typescript", False)
    assert cc._syntax_breaker_open("typescript")
    # Per-language isolation: python keeps compressing.
    assert not cc._syntax_breaker_open("python")


def test_successes_keep_the_breaker_closed():
    for _ in range(cc._SYNTAX_BREAKER_WINDOW * 2):
        cc._record_syntax_outcome("typescript", True)
    for _ in range(cc._SYNTAX_BREAKER_MIN_FAILURES - 1):
        cc._record_syntax_outcome("typescript", False)
    assert not cc._syntax_breaker_open("typescript")


def test_breaker_reopens_after_cooldown(monkeypatch):
    now = [1000.0]
    # Patch the module-local indirection, not the real time module: anything
    # else running concurrently (xdist worker, background thread) reads that one.
    monkeypatch.setattr(cc, "_now", lambda: now[0])
    for _ in range(cc._SYNTAX_BREAKER_MIN_FAILURES):
        cc._record_syntax_outcome("typescript", False)
    assert cc._syntax_breaker_open("typescript")
    now[0] += cc._SYNTAX_BREAKER_COOLDOWN_S + 1
    assert not cc._syntax_breaker_open("typescript")
    # The cleared window means one more failure does not instantly re-trip.
    cc._record_syntax_outcome("typescript", False)
    assert not cc._syntax_breaker_open("typescript")


def test_env_kill_switch(monkeypatch):
    monkeypatch.setenv("HEADROOM_CODE_SYNTAX_BREAKER", "0")
    for _ in range(cc._SYNTAX_BREAKER_WINDOW):
        cc._record_syntax_outcome("typescript", False)
    assert not cc._syntax_breaker_open("typescript")


def test_compress_short_circuits_while_open(monkeypatch):
    monkeypatch.setattr(cc, "_check_tree_sitter_available", lambda: True)

    def _boom(self, *a, **kw):  # pragma: no cover - must not run
        raise AssertionError("AST compression attempted while breaker open")

    monkeypatch.setattr(cc.CodeAwareCompressor, "_compress_with_ast", _boom)
    cc._syntax_breaker_open_until["python"] = cc._now() + 60.0

    compressor = cc.CodeAwareCompressor(cc.CodeCompressorConfig(min_tokens_for_compression=1))
    code = "def f():\n    return 1\n" * 20
    result = compressor.compress(code, language="python")
    assert result.compressed == code
    assert result.compression_ratio == 1.0
    assert result.syntax_valid is True


def test_status_reports_open_languages_for_stats(monkeypatch):
    """While the breaker is open the language compresses at ratio 1.0, so /stats
    needs to say why rather than leaving an unexplained savings drop."""
    now = [1000.0]
    monkeypatch.setattr(cc, "_now", lambda: now[0])

    assert cc.syntax_breaker_status() == {}

    for _ in range(cc._SYNTAX_BREAKER_MIN_FAILURES):
        cc._record_syntax_outcome("typescript", False)
    status = cc.syntax_breaker_status()["typescript"]
    assert status["open"] is True
    assert status["trips"] == 1
    assert status["reopens_in_seconds"] == pytest.approx(cc._SYNTAX_BREAKER_COOLDOWN_S)

    now[0] += cc._SYNTAX_BREAKER_COOLDOWN_S + 1
    reopened = cc.syntax_breaker_status()["typescript"]
    assert reopened["open"] is False
    # The trip count survives the cooldown: a language that keeps tripping is
    # the field diagnosis, and it would be invisible if this reset.
    assert reopened["trips"] == 1
    assert reopened["reopens_in_seconds"] == 0.0


# --- input that was already invalid ----------------------------------------
#
# Coding agents feed the compressor fragments all the time: a Read with an
# offset, a file cut mid-function, grep context. Their syntax was broken before
# compression, so the output fails validation too. That is not the grammar
# mangling real code, which is what the warning and the breaker are for.

_CPP_BODY = "\n".join(
    f"    int v{i} = compute(x, {i});\n    if (v{i} > limit) {{ total += v{i}; }}"
    for i in range(25)
)
_VALID_CPP = (
    "#include <vector>\n#include <string>\nnamespace app {\n"
    "template <typename T>\nclass Cache {\n public:\n"
    "  explicit Cache(int cap) : cap_(cap) {}\n"
    "  int lookup(const std::string& key, int x, int limit) {\n"
    "    int total = 0;\n" + _CPP_BODY + "\n    return total;\n  }\n"
    "  void clear() { items_.clear(); }\n private:\n  int cap_;\n"
    "  std::vector<T> items_;\n};\n}  // namespace app\n"
)
# Cut mid-function, as a Read with a line limit returns it.
_CPP_FRAGMENT = _VALID_CPP[: int(len(_VALID_CPP) * 0.6)]

needs_tree_sitter = pytest.mark.skipif(
    not cc._check_tree_sitter_available(), reason="tree-sitter not installed"
)


def _compressor() -> cc.CodeAwareCompressor:
    return cc.CodeAwareCompressor(cc.CodeCompressorConfig(min_tokens_for_compression=1))


@needs_tree_sitter
def test_already_invalid_input_is_not_blamed_on_the_compressor(caplog):
    compressor = _compressor()
    assert not compressor._verify_syntax(_CPP_FRAGMENT, cc.CodeLanguage.CPP)

    with caplog.at_level(logging.WARNING, logger=cc.logger.name):
        result = compressor.compress(_CPP_FRAGMENT, language="cpp")

    assert result.compressed == _CPP_FRAGMENT
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert False not in cc._syntax_breaker_outcomes.get("cpp", [])


@needs_tree_sitter
def test_fragments_do_not_pause_compression_of_valid_files():
    compressor = _compressor()
    for _ in range(cc._SYNTAX_BREAKER_MIN_FAILURES):
        compressor.compress(_CPP_FRAGMENT, language="cpp")

    assert not cc._syntax_breaker_open("cpp")
    result = compressor.compress(_VALID_CPP, language="cpp")
    assert result.compressed != _VALID_CPP
    assert compressor._verify_syntax(result.compressed, cc.CodeLanguage.CPP)


@needs_tree_sitter
def test_valid_input_mangled_by_the_compressor_still_warns_and_counts(monkeypatch, caplog):
    """A real compressor bug keeps its WARNING and its breaker count."""

    def _mangle(self, code, language, context, tokenizer, **kwargs):
        return "class Broken {\n  int f( {\n", None, {}

    monkeypatch.setattr(cc.CodeAwareCompressor, "_compress_with_ast", _mangle)
    compressor = _compressor()

    with caplog.at_level(logging.WARNING, logger=cc.logger.name):
        result = compressor.compress(_VALID_CPP, language="cpp")

    assert result.compressed == _VALID_CPP
    assert any("produced invalid syntax for cpp" in r.getMessage() for r in caplog.records)
    assert list(cc._syntax_breaker_outcomes["cpp"]) == [False]
