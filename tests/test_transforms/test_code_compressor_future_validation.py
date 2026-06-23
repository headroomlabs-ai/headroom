"""Compressed Python must be compile()-valid, not just tree-sitter-valid (#1233).

tree-sitter validates grammar, not semantics, so a reordered ``from __future__``
parses clean but fails at import. The compile() gate catches it.
"""

from __future__ import annotations

import pytest

from headroom.transforms.code_compressor import (
    CodeAwareCompressor,
    CodeCompressorConfig,
    CodeLanguage,
)

try:
    import tree_sitter_language_pack  # noqa: F401

    TREE_SITTER_INSTALLED = True
except ImportError:
    TREE_SITTER_INSTALLED = False

pytestmark = pytest.mark.skipif(not TREE_SITTER_INSTALLED, reason="tree-sitter required")

# A module that the AST compressor reorders such that __future__ no longer leads.
_FUTURE_MODULE = '''from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class Config:
    """Config holder."""

    name: str
    value: int = 0

    def describe(self) -> str:
        parts = []
        for i in range(self.value):
            parts.append(f"{self.name}-{i}")
            if i % 2 == 0:
                parts.append("even")
        return ", ".join(parts)


def classify(x: Any) -> str:
    """Classify via match."""
    match x:
        case int() if x > 10:
            return "big int"
        case str(s) if (n := len(s)) > 5:
            return f"long {n}"
        case _:
            return "other"
'''


@pytest.fixture
def compressor():
    return CodeAwareCompressor(
        CodeCompressorConfig(min_tokens_for_compression=10, enable_ccr=False)
    )


def test_verify_syntax_rejects_misplaced_future(compressor):
    # tree-sitter accepts this (valid import grammar); compile() rejects it.
    bad = "import os\nfrom __future__ import annotations\n"
    assert compressor._verify_syntax(bad, CodeLanguage.PYTHON) is False


def test_verify_syntax_accepts_valid_future(compressor):
    good = "from __future__ import annotations\nimport os\n"
    assert compressor._verify_syntax(good, CodeLanguage.PYTHON) is True


def test_compressed_output_compiles(compressor):
    # Never serve broken Python: compress() must return compile()-valid output
    # (either validly compressed, or the original when it can't).
    result = compressor.compress(_FUTURE_MODULE)
    compile(result.compressed, "<test>", "exec")  # raises SyntaxError if broken


def test_corpus_compressed_outputs_compile(compressor):
    modules = [
        _FUTURE_MODULE,
        "from __future__ import annotations\n\n"
        + "".join(f"def f{i}(a, b):\n    return a + b + {i}\n\n\n" for i in range(8)),
        "from __future__ import annotations\n\nimport sys\n\n\n"
        "class A:\n    def m(self):\n        x = 0\n        for i in range(50):\n"
        "            x += i\n        return x\n",
    ]
    for src in modules:
        compile(compressor.compress(src).compressed, "<test>", "exec")


def test_future_module_compresses_instead_of_rejecting(compressor):
    # Coverage half of #1233: a __future__ module should compress (not bail to
    # the original) with __future__ kept leading, so the output stays valid.
    result = compressor.compress(_FUTURE_MODULE)
    assert result.compression_ratio < 1.0
    first_line = next(ln for ln in result.compressed.splitlines() if ln.strip())
    assert first_line == "from __future__ import annotations"
    compile(result.compressed, "<test>", "exec")
