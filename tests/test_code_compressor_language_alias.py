"""Regression tests for language-hint / fence-tag coercion in code_compressor.

`CodeAwareCompressor.compress(code, language=...)` used to build the language
with `CodeLanguage(language.lower())`, which raises `ValueError` for anything
that is not an exact enum value. Common markdown fence tags and hints — `js`,
`ts`, `py` — are not enum values, so:

* direct callers (`compress(code, language="js")`) crashed, and
* inside the router the ValueError was swallowed, so ` ```js ` / ` ```ts ` /
  ` ```py ` fenced blocks silently skipped code-aware compression.

`coerce_language` maps aliases to the canonical language and returns UNKNOWN
(never raises) for unrecognized tags, letting the caller fall back to
content-based detection.
"""

import pytest

from headroom.transforms.code_compressor import CodeLanguage, coerce_language


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("js", CodeLanguage.JAVASCRIPT),
        ("jsx", CodeLanguage.JAVASCRIPT),
        ("node", CodeLanguage.JAVASCRIPT),
        ("ts", CodeLanguage.TYPESCRIPT),
        ("tsx", CodeLanguage.TYPESCRIPT),
        ("py", CodeLanguage.PYTHON),
        ("python3", CodeLanguage.PYTHON),
        ("golang", CodeLanguage.GO),
        ("rs", CodeLanguage.RUST),
        ("c++", CodeLanguage.CPP),
        ("phtml", CodeLanguage.PHP),
    ],
)
def test_coerce_language_maps_common_aliases(alias, expected):
    assert coerce_language(alias) == expected


@pytest.mark.parametrize(
    "canonical",
    ["python", "javascript", "typescript", "go", "rust", "java", "c", "cpp", "perl", "php"],
)
def test_coerce_language_accepts_canonical_values(canonical):
    assert coerce_language(canonical) == CodeLanguage(canonical)


def test_coerce_language_is_case_insensitive_and_trims():
    assert coerce_language("  JS  ") == CodeLanguage.JAVASCRIPT
    assert coerce_language("Python") == CodeLanguage.PYTHON


@pytest.mark.parametrize("value", ["", "   ", "not-a-language", "brainfuck", "yaml"])
def test_coerce_language_unknown_returns_unknown_not_valueerror(value):
    # The whole point: never raise, so an unrecognized fence tag can fall back
    # to content detection instead of crashing / being swallowed.
    assert coerce_language(value) == CodeLanguage.UNKNOWN


def test_compress_with_alias_language_does_not_raise():
    """The direct API path must not raise on a common alias."""
    from headroom.transforms.code_compressor import CodeAwareCompressor

    code = "function add(a, b) {\n  return a + b;\n}\n"
    compressor = CodeAwareCompressor()
    # Before the fix this raised ValueError: 'js' is not a valid CodeLanguage.
    result = compressor.compress(code, language="js")
    assert result is not None
    assert result.compressed is not None
