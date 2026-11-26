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


# Mixed files: a wrapper or declaration next to a separate function long
# enough to compress. Every compressed output here also passed syntax
# validation before the fix, so each test checks the text itself.
_RUST_WORK = "fn work() {\n" + _body("    step({i});", 60) + "\n}\n"
_CPP_WORK = "void work() {\n" + _body("    step({i});", 60) + "\n}\n"

# (name, language, code, declaration that must appear exactly once)
SAME_LINE_DECLARATIONS = [
    ("rust-mod-decls", "rust", "mod a; mod b;\n\n" + _RUST_WORK, "mod b;"),
    (
        "cpp-extern-c-decls",
        "cpp",
        'extern "C" int f(); extern "C" int g();\n\n' + _CPP_WORK,
        'extern "C" int g();',
    ),
    (
        "rust-one-line-mod",
        "rust",
        '#[cfg(feature = "ffi")]\nmod ffi { pub struct Handle; }\n\n' + _RUST_WORK,
        "mod ffi { pub struct Handle; }",
    ),
]


@pytest.mark.parametrize(
    ("language", "code", "declaration"),
    [pytest.param(lang, code, decl, id=name) for name, lang, code, decl in SAME_LINE_DECLARATIONS],
)
def test_mixed_file_declarations_appear_once(language, code, declaration):
    result = _compress(code, language)

    assert "// [" in result.compressed, "the function should compress"
    assert result.compressed.count(declaration) == 1, result.compressed
    assert _valid(result.compressed, language)


def test_rust_attribute_stays_on_its_one_line_module():
    code = '#[cfg(feature = "ffi")]\nmod ffi { pub struct Handle; }\n\n' + _RUST_WORK

    lines = _compress(code, "rust").compressed.splitlines()

    module = lines.index("mod ffi { pub struct Handle; }")
    assert lines[module - 1] == '#[cfg(feature = "ffi")]', lines


# (name, code, definition, its use): C and C++ declare before use, so the
# definition must stay above the line that uses it.
DECLARE_BEFORE_USE = [
    (
        "cpp-macro-before-namespace",
        "#include <cstdint>\n#define API_TYPE int\nnamespace api {\nAPI_TYPE query();\n}\n\n"
        + _CPP_WORK,
        "#define API_TYPE int",
        "API_TYPE query();",
    ),
    (
        "cpp-constexpr-before-namespace",
        "#include <cstdint>\nconstexpr int Seed = 7;\nnamespace ns {\n"
        "constexpr int value = Seed;\n}\n\n" + _CPP_WORK,
        "constexpr int Seed = 7;",
        "constexpr int value = Seed;",
    ),
    (
        "cpp-global-before-function",
        "static int counter = 0;\n\nvoid work() {\n" + _body("    counter += {i};", 60) + "\n}\n",
        "static int counter = 0;",
        "void work() {",
    ),
]


@pytest.mark.parametrize(
    ("code", "definition", "use"),
    [pytest.param(code, d, u, id=name) for name, code, d, u in DECLARE_BEFORE_USE],
)
def test_cpp_definitions_stay_above_their_use(code, definition, use):
    result = _compress(code, "cpp")
    out = result.compressed

    assert "// [" in out, "the function should compress"
    assert out.count(definition) == 1, out
    assert out.index(definition) < out.index(use), out


def test_c_global_stays_above_the_function_that_uses_it():
    code = (
        "#include <stdio.h>\n\nstatic int counter = 0;\n\nint work(void) {\n"
        + _body("    counter += {i};", 60)
        + "\n    return counter;\n}\n"
    )

    out = _compress(code, "c").compressed

    assert "// [" in out, "the function should compress"
    assert out.index("static int counter = 0;") < out.index("int work(void) {"), out


def test_cpp_compressed_output_keeps_the_source_layout():
    code = "#include <cstdint>\n#define API_TYPE int\nnamespace api {\nAPI_TYPE query();\n}\n\n" + (
        _CPP_WORK
    )

    out = _compress(code, "cpp").compressed

    assert out.startswith(
        "#include <cstdint>\n#define API_TYPE int\nnamespace api {\nAPI_TYPE query();\n}\n\n"
        "void work() {\n"
    ), out


def test_template_signature_keeps_non_ascii_spelling():
    """Tree-sitter columns are UTF-8 byte offsets, not str indexes."""
    code = (
        "template <typename Größe> bool valid(Größe x) {\n"
        + _body("    step({i});", 60)
        + "\n    return true;\n}\n"
    )

    out = _compress(code, "cpp").compressed

    assert "// [" in out, "the function should compress"
    assert out.splitlines()[0] == "template <typename Größe> bool valid(Größe x) {", out


def test_class_member_on_the_brace_line_is_not_repeated():
    code = "namespace app { int first = 1;\nint second = 2;\n}\n\n" + _CPP_WORK

    out = _compress(code, "cpp").compressed

    assert out.count("int first = 1;") == 1, out


_JAVA_WORK = "    void work() {\n        int n = 0;\n" + _body("        n += 1;", 100) + "\n    }\n"

# Members that share a physical row with a sibling, the container's header or
# its closing brace. Rows are emitted whole, so compressing one such member on
# its own used to drop the sibling from the shared row (syntax still valid) or
# repeat the member before the closing brace. Each declaration must appear
# exactly once while the separate long function still shrinks.
# (name, language, code, declarations that must each appear exactly once)
SHARED_ROW_MEMBERS = [
    (
        "rust-fn-and-const-in-mod",
        "rust",
        "mod n {\n    fn f() {} pub const X: i32 = 2;\n}\n\n" + _RUST_WORK,
        ["fn f() {}", "pub const X: i32 = 2;"],
    ),
    (
        "rust-fn-on-mod-closing-row",
        "rust",
        "mod n {\n    fn f() {} }\n\n" + _RUST_WORK,
        ["fn f() {}"],
    ),
    (
        "rust-two-methods-in-impl",
        "rust",
        "struct S;\nimpl S {\n    fn a(&self) {} fn b(&self) {}\n}\n\n" + _RUST_WORK,
        ["fn a(&self) {}", "fn b(&self) {}"],
    ),
    (
        "java-method-and-field",
        "java",
        "class C {\n    void ping() {} int version = 1;\n" + _JAVA_WORK + "}\n",
        ["void ping() {}", "int version = 1;"],
    ),
    (
        "java-two-methods",
        "java",
        "class C {\n    void a() {} void b() {}\n" + _JAVA_WORK + "}\n",
        ["void a() {}", "void b() {}"],
    ),
    (
        "cpp-function-on-namespace-closing-row",
        "cpp",
        "namespace n {\nint f() { return 1; } }\n\n" + _CPP_WORK,
        ["int f() { return 1; }"],
    ),
    (
        "cpp-comment-before-namespace-closing-brace",
        "cpp",
        "namespace n { // c\nint f() { return 1; } /* x */ }\n\n" + _CPP_WORK,
        ["int f() { return 1; }", "/* x */"],
    ),
    (
        "cpp-one-line-class",
        "cpp",
        "class S {};\n\n" + _CPP_WORK,
        ["class S {};"],
    ),
    (
        "cpp-two-one-line-classes",
        "cpp",
        "class A {}; class B {};\n\n" + _CPP_WORK,
        ["class A {};", "class B {};"],
    ),
    (
        "cpp-one-line-class-in-namespace",
        "cpp",
        "namespace n {\nclass S {};\n}\n\n" + _CPP_WORK,
        ["class S {};"],
    ),
]


@pytest.mark.parametrize(
    ("language", "code", "declarations"),
    [pytest.param(lang, code, decls, id=name) for name, lang, code, decls in SHARED_ROW_MEMBERS],
)
def test_members_sharing_a_row_appear_once(language, code, declarations):
    result = _compress(code, language)
    out = result.compressed

    assert "// [" in out, "the separate long function should compress"
    assert result.compression_ratio < 0.5, out
    for declaration in declarations:
        assert out.count(declaration) == 1, (declaration, out)
    assert _valid(out, language)


# `macro_rules!` is textually scoped: an item that invokes a macro must stay
# below its definition. Grouped assembly moved functions and modules above
# top-level code, so the output used the macro before defining it (tree-sitter
# accepts that; rustc does not).
RUST_MACRO_BEFORE_USE = [
    (
        "rust-macro-before-fn",
        "use std::fmt;\nmacro_rules! make { () => { 1 } }\n\nfn v() -> i32 { make!() }\n\n"
        + _RUST_WORK,
    ),
    (
        "rust-macro-before-mod",
        "fn a() {}\nmacro_rules! make { () => { 1 } }\nmod m {\n    pub fn v() -> i32 { make!() }\n}\n\n"
        + _RUST_WORK,
    ),
]


@pytest.mark.parametrize(
    "code", [pytest.param(code, id=name) for name, code in RUST_MACRO_BEFORE_USE]
)
def test_rust_macro_stays_above_its_use(code):
    out = _compress(code, "rust").compressed

    assert "// [" in out, "the function should compress"
    assert out.count("macro_rules! make") == 1, out
    assert out.index("macro_rules! make") < out.index("make!()"), out


# A block comment that starts after a member and ends on a later row
# (`int limit = 42; /* Maximum` / `request size. */`). Rendering the member
# alone kept the `/*` and dropped the `*/`, so a later comment's terminator
# closed it and swallowed the declarations in between, with valid syntax.
# (name, language, code, declaration hidden before the fix)
COMMENT_TAIL_MEMBERS = [
    (
        "cpp-namespace",
        "cpp",
        "namespace n {\nint limit = 42; /* Maximum\nrequest size. */\nint timeout = 30;\n"
        "/* Defaults for retries. */\n}\n\n" + _CPP_WORK,
        "timeout",
    ),
    (
        "java-class",
        "java",
        "class C {\n    int limit = 42; /* Maximum\n       request size. */\n    int timeout = 30;\n"
        + _JAVA_WORK
        + "}\n",
        "timeout",
    ),
    (
        "rust-mod",
        "rust",
        "mod n {\n    pub const LIMIT: i32 = 42; /* Maximum\n       request size. */\n"
        "    pub const TIMEOUT: i32 = 30;\n}\n\n" + _RUST_WORK,
        "TIMEOUT",
    ),
]


def _identifiers(code: str, language: str) -> set[str]:
    """Identifier texts in the parsed output (comments contribute none)."""
    tree = cc._get_parser(language).parse(code.encode("utf-8"))
    found: set[str] = set()
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in ("identifier", "field_identifier"):
            found.add(node.text.decode("utf-8"))
        stack.extend(node.children)
    return found


@pytest.mark.parametrize(
    ("language", "code", "name"),
    [pytest.param(lang, code, name, id=case) for case, lang, code, name in COMMENT_TAIL_MEMBERS],
)
def test_comment_after_a_member_keeps_its_terminator(language, code, name):
    result = _compress(code, language)
    out = result.compressed

    assert "// [" in out, "the separate long function should compress"
    assert out.count("request size. */") == 1, out
    assert name in _identifiers(out, language), out
    assert _valid(out, language)
