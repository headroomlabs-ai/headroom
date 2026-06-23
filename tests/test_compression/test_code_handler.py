"""Tests for code structure handler."""

import pytest

from headroom.compression.handlers.code_handler import (
    CodeStructureHandler,
    is_tree_sitter_available,
)

requires_tree_sitter = pytest.mark.skipif(
    not is_tree_sitter_available(),
    reason="tree-sitter-language-pack not installed",
)


class TestCanHandle:
    @pytest.fixture
    def handler(self):
        return CodeStructureHandler()

    def test_detects_python(self, handler):
        assert handler.can_handle("def foo():\n    pass\n") is True

    def test_detects_javascript(self, handler):
        assert handler.can_handle("function foo() { return 1; }") is True

    def test_rejects_prose(self, handler):
        assert handler.can_handle("This is a plain sentence.") is False


class TestRegexFallback:
    """Regex path runs regardless of tree-sitter availability."""

    @pytest.fixture
    def handler(self):
        return CodeStructureHandler(use_tree_sitter=False)

    def test_python_signature_preserved_body_compressible(self, handler):
        code = "def hello(name: str) -> str:\n    message = name\n    return message\n"
        result = handler.get_mask(code, language="python")

        assert result.metadata["parser"] == "regex"
        sig = "def hello(name: str) -> str:"
        start = code.index(sig)
        assert all(result.mask.mask[i] for i in range(start, start + len(sig)))

        body_char = code.index("message = name")
        assert result.mask.mask[body_char] is False

    def test_python_import_preserved(self, handler):
        code = "import os\n\nx = 1\n"
        result = handler.get_mask(code, language="python")
        assert all(result.mask.mask[i] for i in range(len("import os")))


class TestLanguageDetection:
    @pytest.fixture
    def handler(self):
        return CodeStructureHandler()

    def test_detects_python(self, handler):
        code = "import os\n\nclass Foo:\n    def method(self):\n        pass\n"
        assert handler._detect_language(code) == "python"

    def test_detects_go(self, handler):
        code = 'package main\n\nimport (\n\t"fmt"\n)\n\nfunc main() {\n}\n'
        assert handler._detect_language(code) == "go"

    def test_detects_rust(self, handler):
        code = "use std::io;\n\npub fn main() {\n    let mut x = 1;\n}\n"
        assert handler._detect_language(code) == "rust"

    def test_detects_perl(self, handler):
        code = "use strict;\npackage Foo;\n\nsub greet {\n    my $name = shift;\n    return $name;\n}\n"
        assert handler._detect_language(code) == "perl"

    def test_falls_back_to_default(self):
        handler = CodeStructureHandler(default_language="javascript")
        assert handler._detect_language("plain words only here") == "javascript"


class TestRegexFallbackLanguages:
    """Signature/import preservation on the regex path across languages."""

    @pytest.fixture
    def handler(self):
        return CodeStructureHandler(use_tree_sitter=False)

    def test_go_func_signature_preserved(self, handler):
        code = "func Add(a int, b int) int {\n\treturn a + b\n}\n"
        result = handler.get_mask(code, language="go")
        sig = "func Add(a int, b int)"
        start = code.index(sig)
        assert all(result.mask.mask[i] for i in range(start, start + len(sig)))

    def test_rust_fn_signature_preserved(self, handler):
        code = "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n"
        result = handler.get_mask(code, language="rust")
        sig = "pub fn add(a: i32, b: i32)"
        start = code.index(sig)
        assert all(result.mask.mask[i] for i in range(start, start + len(sig)))

    def test_typescript_interface_preserved(self, handler):
        code = "interface Shape {\n  area(): number;\n}\n\nconst x = 1;\n"
        result = handler.get_mask(code, language="typescript")
        sig = "interface Shape"
        start = code.index(sig)
        assert all(result.mask.mask[i] for i in range(start, start + len(sig)))

    def test_javascript_arrow_function_preserved(self, handler):
        code = "const add = (a, b) => {\n  return a + b;\n};\n"
        result = handler.get_mask(code, language="javascript")
        sig = "const add = (a, b) =>"
        start = code.index(sig)
        assert all(result.mask.mask[i] for i in range(start, start + len(sig)))

    def test_perl_sub_signature_preserved(self, handler):
        code = "sub add {\n    my ($a, $b) = @_;\n    return $a + $b;\n}\n"
        result = handler.get_mask(code, language="perl")
        sig = "sub add"
        start = code.index(sig)
        assert all(result.mask.mask[i] for i in range(start, start + len(sig)))

    def test_perl_use_import_preserved(self, handler):
        code = "use strict;\nuse warnings;\n\nmy $x = 1;\n"
        result = handler.get_mask(code, language="perl")
        assert all(result.mask.mask[i] for i in range(len("use strict")))

    def test_regex_confidence_lower_than_tree_sitter(self, handler):
        result = handler.get_mask("def f():\n    pass\n", language="python")
        assert result.confidence == 0.7


class TestEdgeCases:
    @pytest.fixture
    def handler(self):
        return CodeStructureHandler()

    def test_empty_content(self, handler):
        result = handler.get_mask("")
        assert result.confidence == 0.0
        assert result.metadata.get("empty") is True

    def test_whitespace_only_content(self, handler):
        result = handler.get_mask("   \n\n  ")
        assert result.metadata.get("empty") is True

    def test_unknown_language_regex_no_patterns(self):
        """A language with no regex patterns yields an all-compressible
        mask rather than raising."""
        handler = CodeStructureHandler(use_tree_sitter=False)
        code = "BEGIN\n  WRITELN('hello')\nEND.\n"
        result = handler.get_mask(code, language="pascal")
        assert not any(result.mask.mask)

    def test_mask_length_matches_content(self, handler):
        code = "def f():\n    return 1\n"
        result = handler.get_mask(code, language="python")
        assert len(result.mask.mask) == len(code)


@requires_tree_sitter
class TestTreeSitterContainers:
    """Container bodies must stay compressible (signature-only spans).

    Regression: class_definition / decorated_definition / impl_item were
    marked structural over their FULL span, so every method body inside a
    class (i.e. most real code) was preserved and compression no-opped at
    confidence 0.95.
    """

    @pytest.fixture
    def handler(self):
        return CodeStructureHandler()

    def test_class_method_bodies_compressible(self, handler):
        code = (
            "class Foo:\n"
            "    def method_a(self):\n"
            "        body_line_a = 1\n"
            "        return body_line_a\n"
            "\n"
            "    def method_b(self):\n"
            "        body_line_b = 2\n"
            "        return body_line_b\n"
        )
        result = handler.get_mask(code, language="python")
        assert result.metadata["parser"] == "tree-sitter"

        # Class signature and method signatures preserved
        assert all(result.mask.mask[i] for i in range(len("class Foo:")))
        sig = "def method_a(self):"
        start = code.index(sig)
        assert all(result.mask.mask[i] for i in range(start, start + len(sig)))

        # Method bodies compressible
        for body in ("body_line_a = 1", "body_line_b = 2"):
            start = code.index(body)
            assert not any(result.mask.mask[i] for i in range(start, start + len(body))), (
                f"method body {body!r} must be compressible"
            )

    def test_decorated_function_body_compressible(self, handler):
        code = "@decorator\ndef decorated():\n    body_line = 4\n    return body_line\n"
        result = handler.get_mask(code, language="python")

        # Decorator and signature preserved
        assert all(result.mask.mask[i] for i in range(len("@decorator")))
        sig = "def decorated():"
        start = code.index(sig)
        assert all(result.mask.mask[i] for i in range(start, start + len(sig)))

        # Body compressible
        start = code.index("body_line = 4")
        assert not any(result.mask.mask[i] for i in range(start, start + len("body_line = 4"))), (
            "decorated function body must be compressible"
        )

    def test_module_function_body_compressible(self, handler):
        code = "def standalone():\n    body_line = 3\n    return body_line\n"
        result = handler.get_mask(code, language="python")

        start = code.index("body_line = 3")
        assert not any(result.mask.mask[i] for i in range(start, start + len("body_line = 3")))

    def test_rust_impl_method_bodies_compressible(self, handler):
        code = (
            "struct Foo { x: i32 }\n"
            "impl Foo {\n"
            "    fn method(&self) -> i32 {\n"
            "        let body_line = 5;\n"
            "        body_line\n"
            "    }\n"
            "}\n"
        )
        result = handler.get_mask(code, language="rust")

        # impl signature preserved
        start = code.index("impl Foo")
        assert all(result.mask.mask[i] for i in range(start, start + len("impl Foo")))

        # method body compressible
        start = code.index("let body_line = 5;")
        assert not any(
            result.mask.mask[i] for i in range(start, start + len("let body_line = 5;"))
        ), "impl method body must be compressible"

    def test_concurrent_parsing_uses_tree_sitter(self, handler):
        """Parsers must be thread-local.

        Regression: parsers were cached in a process-global dict and
        shared across threads. tree-sitter Parser objects are pyo3
        unsendable — touching one from a non-creator thread panics (or
        raises, dropping the handler to the regex fallback). Parsing
        from a thread pool must succeed on the tree-sitter path in
        every thread.
        """
        from concurrent.futures import ThreadPoolExecutor

        code = "class Foo:\n    def m(self):\n        x = 1\n        return x\n"

        def work(_: int) -> str:
            result = handler.get_mask(code, language="python")
            return str(result.metadata["parser"])

        with ThreadPoolExecutor(max_workers=4) as pool:
            parsers = list(pool.map(work, range(16)))

        assert parsers == ["tree-sitter"] * 16, (
            f"all threads must parse via tree-sitter, got: {set(parsers)}"
        )

    def test_non_ascii_content_mask_alignment(self, handler):
        """Byte offsets must be converted to char offsets.

        Regression: tree-sitter reports byte offsets into the UTF-8
        encoding, but the mask is char-indexed. Multi-byte characters
        (here: accents + an emoji, 9 extra bytes) shifted every later
        span, preserving the wrong characters.
        """
        code = (
            "# café münü 🎉 comment\n"
            "def target(x: int) -> int:\n"
            "    body_value = 9\n"
            "    return body_value\n"
        )
        result = handler.get_mask(code, language="python")

        sig = "def target(x: int) -> int:"
        start = code.index(sig)
        assert all(result.mask.mask[i] for i in range(start, start + len(sig))), (
            "signature after non-ASCII content must be exactly preserved"
        )

        bstart = code.index("body_value = 9")
        assert not any(
            result.mask.mask[i] for i in range(bstart, bstart + len("body_value = 9"))
        ), "body after non-ASCII content must stay compressible"

    def test_preservation_ratio_sane_for_class_code(self, handler):
        """A class with substantial method bodies should NOT preserve
        everything — the whole point of the handler."""
        body = "\n".join(f"        line_{i} = {i}" for i in range(20))
        code = f"class Big:\n    def method(self):\n{body}\n        return 0\n"
        result = handler.get_mask(code, language="python")
        assert result.preservation_ratio < 0.5, (
            f"class code preserved {result.preservation_ratio:.0%} — "
            "container bodies are leaking into the structural mask"
        )
