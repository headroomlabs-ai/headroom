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
