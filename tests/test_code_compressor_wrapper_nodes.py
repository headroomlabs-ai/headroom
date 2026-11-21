"""Declarations nested in wrapper nodes must not be emitted twice.

The extractor captures known declarations (functions, classes, types) and then
re-emits every top-level node it did not capture, verbatim. A wrapper the
language config does not know (a C++ ``namespace`` or ``template``, a Rust
``mod``, a Python ``if TYPE_CHECKING:`` block) was recursed into, so its nested
declarations were captured and compressed on their own, stripped of the
wrapper, and then the whole wrapper was re-emitted untouched. The output held
the code twice. Usually that made it larger than the input and callers threw
it away, but in a mixed file (a Rust module plus ``#[cfg(test)] mod tests``)
the rest compressed enough for the duplicate to reach the model.
"""

from __future__ import annotations

import pytest

from headroom.transforms import code_compressor as cc

pytestmark = pytest.mark.skipif(
    not cc._check_tree_sitter_available(), reason="tree-sitter not installed"
)


@pytest.fixture(autouse=True)
def _reset_breaker():
    cc._syntax_breaker_outcomes.clear()
    cc._syntax_breaker_open_until.clear()
    yield
    cc._syntax_breaker_outcomes.clear()
    cc._syntax_breaker_open_until.clear()


def _body(template: str, n: int = 30) -> str:
    return "\n".join(template.format(i=i) for i in range(n))


def _compress(code: str, language: str) -> cc.CodeCompressionResult:
    compressor = cc.CodeAwareCompressor(
        cc.CodeCompressorConfig(min_tokens_for_compression=1, enable_ccr=False)
    )
    return compressor.compress(code, language=language)


def _valid(code: str, language: str) -> bool:
    return cc.CodeAwareCompressor()._verify_syntax(code, cc.CodeLanguage(language))


CPP_CLASS_IN_NAMESPACE = (
    "#include <vector>\nnamespace app {\nclass Cache {\n public:\n  int lookup(int x) {\n"
    "    int t = 0;\n" + _body("    t += step(x, {i});") + "\n    return t;\n  }\n};\n}\n"
)
CPP_FUNCTION_IN_NAMESPACE = (
    "#include <vector>\nnamespace app {\nint lookup(int x) {\n  int t = 0;\n"
    + _body("  t += step(x, {i});")
    + "\n  return t;\n}\n}\n"
)
CPP_TEMPLATE_CLASS = (
    "#include <vector>\ntemplate <typename T>\nclass Cache {\n public:\n  int lookup(int x) {\n"
    "    int t = 0;\n" + _body("    t += step(x, {i});") + "\n    return t;\n  }\n};\n"
)
CPP_EXTERN_C = (
    '#include <vector>\nextern "C" {\nint lookup(int x) {\n  int t = 0;\n'
    + _body("  t += step(x, {i});")
    + "\n  return t;\n}\n}\n"
)
RUST_IMPL_IN_MOD = (
    "use std::fmt;\nmod store {\n    pub struct Cache;\n    impl Cache {\n"
    "        pub fn lookup(&self, x: i32) -> i32 {\n            let mut t = 0;\n"
    + _body("            t += step(x, {i});")
    + "\n            t\n        }\n    }\n}\n"
)
RUST_FN_AND_TEST_MOD = (
    "use std::fmt;\npub fn lookup(x: i32) -> i32 {\n    let mut t = 0;\n"
    + _body("    t += step(x, {i});")
    + "\n    t\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n    #[test]\n"
    "    fn lookup_is_stable() {\n"
    + _body("        assert_eq!(lookup({i}), lookup({i}));", 15)
    + "\n    }\n}\n"
)

# (name, language, code, a signature line both copies keep when duplicated)
WRAPPED = [
    ("cpp-class-in-namespace", "cpp", CPP_CLASS_IN_NAMESPACE, "int lookup(int x) {"),
    ("cpp-function-in-namespace", "cpp", CPP_FUNCTION_IN_NAMESPACE, "int lookup(int x) {"),
    ("cpp-template-class", "cpp", CPP_TEMPLATE_CLASS, "int lookup(int x) {"),
    ("cpp-extern-c", "cpp", CPP_EXTERN_C, "int lookup(int x) {"),
    ("rust-impl-in-mod", "rust", RUST_IMPL_IN_MOD, "pub fn lookup(&self, x: i32) -> i32 {"),
    ("rust-fn-and-test-mod", "rust", RUST_FN_AND_TEST_MOD, "fn lookup_is_stable() {"),
]


@pytest.mark.parametrize(
    ("language", "code", "kept_line"),
    [pytest.param(lang, code, line, id=name) for name, lang, code, line in WRAPPED],
)
def test_wrapped_declarations_are_not_emitted_twice(language, code, kept_line):
    result = _compress(code, language)

    assert result.compressed.count(kept_line) == 1, result.compressed
    assert _valid(result.compressed, language)


def test_rust_test_module_items_stay_inside_the_module():
    """`use super::*;` at crate root is not what the source says."""
    result = _compress(RUST_FN_AND_TEST_MOD, "rust")

    root_lines = [line for line in result.compressed.splitlines() if not line.startswith(" ")]
    assert "use super::*;" not in root_lines, result.compressed


_C_FUNCTION = (
    "int lookup(int x) {\n  int t = 0;\n" + _body("  t += step(x, {i});") + "\n  return t;\n}\n"
)

# Wrappers no language config lists. They stay verbatim (no compression
# inside), which is the price of never emitting their content twice.
UNLISTED = [
    (
        "c-include-guard",
        "c",
        "#ifndef CACHE_H\n#define CACHE_H\n#include <stdio.h>\n" + _C_FUNCTION + "#endif\n",
        "int lookup(int x) {",
    ),
    (
        "c-ifdef-block",
        "c",
        "#include <stdio.h>\n#ifdef DEBUG\n" + _C_FUNCTION + "#endif\n",
        "int lookup(int x) {",
    ),
    (
        "typescript-namespace",
        "typescript",
        "import { x } from 'y';\nnamespace App {\n  export function lookup(x: number): number {\n"
        "    let t = 0;\n" + _body("    t += step(x, {i});") + "\n    return t;\n  }\n}\n",
        "function lookup(x: number): number {",
    ),
    (
        "python-conditional-def",
        "python",
        "import os\nif os.environ.get('X'):\n    def lookup(x):\n        t = 0\n"
        + _body("        t += step(x, {i})")
        + "\n        return t\n",
        "def lookup(x):",
    ),
]


@pytest.mark.parametrize(
    ("language", "code", "kept_line"),
    [pytest.param(lang, code, line, id=name) for name, lang, code, line in UNLISTED],
)
def test_unlisted_wrappers_are_not_duplicated(language, code, kept_line):
    result = _compress(code, language)

    assert result.compressed.count(kept_line) == 1, result.compressed
    assert _valid(result.compressed, language)


# --- wrappers the config knows: compress inside, keep the wrapper ----------

CPP_TEMPLATE_CLASS_IN_NAMESPACE = (
    "#include <vector>\nnamespace app {\ntemplate <typename T>\nclass Cache {\n public:\n"
    "  int lookup(int x) {\n    int t = 0;\n"
    + _body("    t += step(x, {i});")
    + "\n    return t;\n  }\n  std::vector<T> items_;\n};\n}  // namespace app\n"
)

# (name, language, code, text that must survive exactly, in order). Each case
# must also drop a line from deep inside the wrapped body: the proof that the
# wrapper was compressed inside rather than emitted verbatim.
CONTAINERS = [
    ("cpp-class-in-namespace", "cpp", CPP_CLASS_IN_NAMESPACE, "namespace app {\nclass Cache {"),
    (
        "cpp-function-in-namespace",
        "cpp",
        CPP_FUNCTION_IN_NAMESPACE,
        "namespace app {\nint lookup(int x) {",
    ),
    ("cpp-template-class", "cpp", CPP_TEMPLATE_CLASS, "template <typename T>\nclass Cache {"),
    (
        "cpp-template-class-in-namespace",
        "cpp",
        CPP_TEMPLATE_CLASS_IN_NAMESPACE,
        "namespace app {\ntemplate <typename T>\nclass Cache {",
    ),
    ("cpp-extern-c", "cpp", CPP_EXTERN_C, 'extern "C" {\nint lookup(int x) {'),
    (
        "rust-impl-in-mod",
        "rust",
        RUST_IMPL_IN_MOD,
        "mod store {\n    pub struct Cache;\n    impl Cache {",
    ),
    (
        "rust-fn-and-test-mod",
        "rust",
        RUST_FN_AND_TEST_MOD,
        "#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn lookup_is_stable() {",
    ),
]


@pytest.mark.parametrize(
    ("language", "code", "wrapper"),
    [pytest.param(lang, code, wrapper, id=name) for name, lang, code, wrapper in CONTAINERS],
)
def test_known_wrappers_compress_inside_and_keep_their_shape(language, code, wrapper):
    result = _compress(code, language)
    deep_line = (
        "assert_eq!(lookup(14), lookup(14));" if "mod tests" in code else "t += step(x, 29);"
    )

    assert deep_line in code
    assert deep_line not in result.compressed, result.compressed
    assert result.compressed_tokens < result.original_tokens, result.compressed
    assert "omitted" in result.compressed, result.compressed
    assert wrapper in result.compressed, result.compressed
    assert _valid(result.compressed, language)


@pytest.mark.parametrize(
    ("language", "code"),
    [
        pytest.param(
            "cpp",
            'extern "C" int lookup(int x);\n' + CPP_FUNCTION_IN_NAMESPACE,
            id="cpp-extern-c-declaration",
        ),
        pytest.param("rust", "mod store;\n" + RUST_IMPL_IN_MOD, id="rust-mod-declaration"),
    ],
)
def test_bodiless_wrappers_are_kept_once(language, code):
    result = _compress(code, language)
    first_line = code.splitlines()[0]

    assert result.compressed.count(first_line) == 1, result.compressed
    assert _valid(result.compressed, language)
