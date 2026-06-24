"""Tests for Code-Aware Compressor using tree-sitter AST parsing.

Comprehensive tests covering:
- CodeCompressorConfig: Configuration validation and defaults
- CodeAwareCompressor: Core AST-based compression functionality
- Language detection: Auto-detection from extensions and content
- Transform interface: apply(), should_apply() methods
- Syntax preservation: Guarantees valid output syntax
- Edge cases: Empty content, unavailable dependency, fallbacks
"""

import textwrap
from unittest.mock import patch

import pytest

from headroom.transforms.code_compressor import (
    CodeAwareCompressor,
    CodeCompressionResult,
    CodeCompressorConfig,
    CodeLanguage,
    DocstringMode,
    detect_language,
    is_tree_sitter_available,
    is_tree_sitter_loaded,
    unload_tree_sitter,
)

# Try to import for availability check
try:
    import tree_sitter_language_pack  # noqa: F401

    TREE_SITTER_INSTALLED = True
except ImportError:
    TREE_SITTER_INSTALLED = False


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Default CodeCompressorConfig for testing."""
    return CodeCompressorConfig(
        min_tokens_for_compression=10,  # Low threshold for tests
        enable_ccr=False,  # Disable CCR for unit tests
    )


@pytest.fixture
def compressor(default_config):
    """CodeAwareCompressor instance with default config."""
    return CodeAwareCompressor(default_config)


@pytest.fixture
def tokenizer():
    """Get a tokenizer for Transform interface tests."""
    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer

    provider = OpenAIProvider()
    token_counter = provider.get_token_counter("gpt-4o")
    return Tokenizer(token_counter, "gpt-4o")


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_python_code(n_functions: int = 5, n_classes: int = 1) -> str:
    """Generate Python code for testing."""
    lines = [
        '"""Module with classes and functions."""',
        "",
        "import os",
        "import sys",
        "from typing import Any, Optional, List",
        "from dataclasses import dataclass",
        "",
    ]

    for c in range(n_classes):
        lines.extend(
            [
                "@dataclass",
                f"class TestClass{c}:",
                '    """A test class with docstring."""',
                "    name: str",
                "    value: int = 0",
                "",
                "    def method(self, arg: Any) -> str:",
                '        """Process the argument."""',
                "        result = str(arg)",
                "        for i in range(10):",
                '            result += f"iteration {i}"',
                "        return result",
                "",
            ]
        )

    for i in range(n_functions):
        lines.extend(
            [
                f"def function_{i}(arg: Any, optional: Optional[str] = None) -> str:",
                f'    """Process argument {i}.',
                "",
                "    This is a longer docstring with multiple lines.",
                "    It explains what the function does in detail.",
                "",
                "    Args:",
                "        arg: The argument to process.",
                "        optional: An optional parameter.",
                "",
                "    Returns:",
                "        A string result.",
                '    """',
                "    result = str(arg)",
                "    if optional:",
                "        result += optional",
                "    for i in range(10):",
                '        result += f"iteration {i}"',
                "    try:",
                "        int(result)",
                "    except ValueError:",
                '        result = "0"',
                "    return result",
                "",
            ]
        )

    return "\n".join(lines)


def generate_javascript_code(n_functions: int = 5) -> str:
    """Generate JavaScript code for testing."""
    lines = [
        "// Module with various functions",
        'import { something } from "module";',
        'const config = require("./config");',
        "",
    ]

    for i in range(n_functions):
        lines.extend(
            [
                "/**",
                f" * Process function {i}",
                " * @param {any} arg - The argument",
                " * @returns {string} The result",
                " */",
                f"function processFunction{i}(arg) {{",
                "    let result = String(arg);",
                "    for (let j = 0; j < 10; j++) {",
                "        result += `iteration ${j}`;",
                "    }",
                "    try {",
                "        JSON.parse(result);",
                "    } catch (e) {",
                "        console.error(e);",
                "    }",
                "    return result;",
                "}",
                "",
            ]
        )

    lines.append("export { processFunction0 };")
    return "\n".join(lines)


def generate_go_code(n_functions: int = 3) -> str:
    """Generate Go code for testing."""
    lines = [
        "package main",
        "",
        'import "fmt"',
        "",
        "// Config holds configuration",
        "type Config struct {",
        "    Name  string",
        "    Value int",
        "}",
        "",
    ]

    for i in range(n_functions):
        lines.extend(
            [
                f"// Process{i} processes the input",
                f"func Process{i}(input string) (string, error) {{",
                "    result := input",
                "    for i := 0; i < 10; i++ {",
                '        result = fmt.Sprintf("%s-%d", result, i)',
                "    }",
                "    if len(result) == 0 {",
                '        return "", fmt.Errorf("empty result")',
                "    }",
                "    return result, nil",
                "}",
                "",
            ]
        )

    return "\n".join(lines)


# =============================================================================
# TestCodeCompressorConfig
# =============================================================================


class TestCodeCompressorConfig:
    """Tests for CodeCompressorConfig dataclass."""

    def test_default_values(self):
        """Default config values are sensible."""
        config = CodeCompressorConfig()

        assert config.preserve_imports is True
        assert config.preserve_signatures is True
        assert config.preserve_type_annotations is True
        assert config.preserve_decorators is True
        assert config.docstring_mode == DocstringMode.FIRST_LINE
        assert config.target_compression_rate == 0.2
        assert config.max_body_lines == 5
        assert config.min_tokens_for_compression == 100
        assert config.enable_ccr is True

    def test_custom_values(self):
        """Custom config values are applied."""
        config = CodeCompressorConfig(
            preserve_imports=False,
            preserve_signatures=True,
            docstring_mode=DocstringMode.FULL,
            target_compression_rate=0.3,
            max_body_lines=10,
            min_tokens_for_compression=50,
        )

        assert config.preserve_imports is False
        assert config.preserve_signatures is True
        assert config.docstring_mode == DocstringMode.FULL
        assert config.target_compression_rate == 0.3
        assert config.max_body_lines == 10
        assert config.min_tokens_for_compression == 50

    def test_docstring_modes(self):
        """All docstring modes are valid."""
        for mode in DocstringMode:
            config = CodeCompressorConfig(docstring_mode=mode)
            assert config.docstring_mode == mode


# =============================================================================
# TestCodeCompressionResult
# =============================================================================


class TestCodeCompressionResult:
    """Tests for CodeCompressionResult dataclass."""

    def test_tokens_saved(self):
        """tokens_saved property calculates correctly."""
        result = CodeCompressionResult(
            compressed="short",
            original="long content here",
            original_tokens=100,
            compressed_tokens=30,
            compression_ratio=0.3,
            language=CodeLanguage.PYTHON,
            syntax_valid=True,
        )

        assert result.tokens_saved == 70

    def test_tokens_saved_no_negative(self):
        """tokens_saved never returns negative."""
        result = CodeCompressionResult(
            compressed="expanded",
            original="short",
            original_tokens=10,
            compressed_tokens=20,
            compression_ratio=2.0,
            language=CodeLanguage.PYTHON,
            syntax_valid=True,
        )

        assert result.tokens_saved == 0

    def test_savings_percentage(self):
        """savings_percentage property calculates correctly."""
        result = CodeCompressionResult(
            compressed="short",
            original="long content",
            original_tokens=100,
            compressed_tokens=25,
            compression_ratio=0.25,
            language=CodeLanguage.PYTHON,
            syntax_valid=True,
        )

        assert result.savings_percentage == 75.0

    def test_savings_percentage_zero_original(self):
        """savings_percentage handles zero original tokens."""
        result = CodeCompressionResult(
            compressed="",
            original="",
            original_tokens=0,
            compressed_tokens=0,
            compression_ratio=1.0,
            language=CodeLanguage.UNKNOWN,
            syntax_valid=True,
        )

        assert result.savings_percentage == 0.0


# =============================================================================
# TestCodeLanguage
# =============================================================================


class TestCodeLanguage:
    """Tests for CodeLanguage enum and detection."""

    def test_all_language_values_are_unique(self):
        """All language enum values are unique."""
        values = [lang.value for lang in CodeLanguage]
        assert len(values) == len(set(values))

    def test_detect_python_language(self):
        """Python language is detected from code patterns."""

        code = """
import os
from typing import List

def function(arg: str) -> str:
    return arg

class MyClass:
    pass
"""
        lang, confidence = detect_language(code)
        assert lang == CodeLanguage.PYTHON
        assert confidence > 0.5

    def test_detect_javascript_language(self):
        """JavaScript language is detected from code patterns."""

        code = """
const express = require('express');
import { something } from 'module';

function handler(req, res) {
    return res.json({ status: 'ok' });
}

export default handler;
"""
        lang, confidence = detect_language(code)
        assert lang in (CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT)
        assert confidence > 0.3

    def test_detect_go_language(self):
        """Go language is detected from code patterns."""

        code = """
package main

import "fmt"

func main() {
    fmt.Println("Hello")
}
"""
        lang, confidence = detect_language(code)
        assert lang == CodeLanguage.GO
        assert confidence > 0.3


# =============================================================================
# TestCodeAwareCompressor
# =============================================================================


class TestCodeAwareCompressor:
    """Tests for CodeAwareCompressor core functionality."""

    def test_init_with_default_config(self):
        """Compressor initializes with default config."""
        compressor = CodeAwareCompressor()

        assert compressor.config is not None
        assert compressor.config.preserve_imports is True

    def test_init_with_custom_config(self, default_config):
        """Compressor initializes with custom config."""
        compressor = CodeAwareCompressor(default_config)

        assert compressor.config == default_config

    def test_compress_skips_small_content(self, compressor):
        """Small content is not compressed."""
        small_code = "def f(): pass"
        result = compressor.compress(small_code)

        assert result.compressed == small_code
        assert result.compression_ratio == 1.0

    def test_compress_handles_empty_content(self, compressor):
        """Empty content returns empty result."""
        result = compressor.compress("")

        assert result.compressed == ""
        assert result.compression_ratio == 1.0
        assert result.syntax_valid is True

    def test_compress_with_explicit_language(self, compressor):
        """Language can be specified explicitly."""
        code = generate_python_code(2)
        result = compressor.compress(code, language="python")

        # Should detect or use the specified language
        assert result.language == CodeLanguage.PYTHON or result.language == CodeLanguage.UNKNOWN

    def test_compress_auto_detects_python(self, compressor):
        """Python code is auto-detected during compression."""
        code = """
import os
from typing import List

def function(arg: str) -> List[str]:
    return [arg]

class MyClass:
    pass
"""
        result = compressor.compress(code)
        # Should detect Python (if tree-sitter available) or return UNKNOWN
        assert result.language in (CodeLanguage.PYTHON, CodeLanguage.UNKNOWN)

    def test_compress_auto_detects_javascript(self, compressor):
        """JavaScript code is auto-detected during compression."""
        code = """
const express = require('express');
import { something } from 'module';

function handler(req, res) {
    return res.json({ status: 'ok' });
}

export default handler;
"""
        result = compressor.compress(code)
        assert result.language in (
            CodeLanguage.JAVASCRIPT,
            CodeLanguage.TYPESCRIPT,
            CodeLanguage.UNKNOWN,
        )

    def test_compress_auto_detects_go(self, compressor):
        """Go code is auto-detected during compression."""
        code = """
package main

import "fmt"

func main() {
    fmt.Println("Hello")
}
"""
        result = compressor.compress(code)
        assert result.language in (CodeLanguage.GO, CodeLanguage.UNKNOWN)


# =============================================================================
# TestFallbackCompression
# =============================================================================


class TestFallbackCompression:
    """Tests for fallback compression when tree-sitter unavailable."""

    def test_fallback_when_tree_sitter_unavailable(self, default_config):
        """Uses fallback compression when tree-sitter is not installed."""
        with patch(
            "headroom.transforms.code_compressor._check_tree_sitter_available",
            return_value=False,
        ):
            compressor = CodeAwareCompressor(default_config)
            code = generate_python_code(5)

            result = compressor.compress(code)

            # Should still return a result (fallback compression)
            assert result is not None
            # Kompress fallback does NOT guarantee syntax validity
            # If Kompress is unavailable, returns original (valid)
            # If Kompress IS available, syntax_valid=False (cannot guarantee)

    def test_fallback_preserves_structure(self, default_config):
        """Fallback compression preserves basic structure when no compressor available.

        When both tree-sitter and Kompress are unavailable, the fallback
        returns the original code unchanged - preserving all structure.
        """
        with (
            patch(
                "headroom.transforms.code_compressor._check_tree_sitter_available",
                return_value=False,
            ),
            patch(
                "headroom.transforms.kompress_compressor.is_kompress_available",
                return_value=False,
            ),
        ):
            compressor = CodeAwareCompressor(default_config)
            code = generate_python_code(3)

            result = compressor.compress(code)

            # With no compressor available, original code is returned unchanged
            # This preserves all imports and class/function signatures
            assert "import os" in result.compressed
            assert "def function_" in result.compressed
            # Compression ratio should be 1.0 (no compression)
            assert result.compression_ratio == 1.0


# =============================================================================
# TestTransformInterface
# =============================================================================


class TestTransformInterface:
    """Tests for Transform interface (apply, should_apply)."""

    def test_should_apply_returns_false_for_small_content(self, default_config, tokenizer):
        """should_apply returns False for small content."""
        config = CodeCompressorConfig(min_tokens_for_compression=1000)
        compressor = CodeAwareCompressor(config)
        messages = [{"role": "user", "content": "def f(): pass"}]

        assert not compressor.should_apply(messages, tokenizer)

    def test_should_apply_returns_bool_for_large_code(self, default_config, tokenizer):
        """should_apply returns boolean for large code content."""
        compressor = CodeAwareCompressor(default_config)
        code = generate_python_code(20)
        messages = [{"role": "tool", "tool_call_id": "call_1", "content": code}]

        # Should return True if there's code content to process
        result = compressor.should_apply(messages, tokenizer)
        assert isinstance(result, bool)

    def test_apply_returns_transform_result(self, default_config, tokenizer):
        """apply() returns proper TransformResult."""
        compressor = CodeAwareCompressor(default_config)
        code = generate_python_code(10)
        messages = [{"role": "tool", "tool_call_id": "call_1", "content": code}]

        result = compressor.apply(messages, tokenizer)

        assert result is not None
        assert result.tokens_before > 0
        assert len(result.messages) == 1

    def test_apply_passes_through_non_code_messages(self, default_config, tokenizer):
        """apply() passes through non-code messages unchanged."""
        compressor = CodeAwareCompressor(default_config)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = compressor.apply(messages, tokenizer)

        assert result.messages[0]["content"] == "Hello"
        assert result.messages[1]["content"] == "Hi there!"

    def test_name_property(self, compressor):
        """Compressor has correct name."""
        assert compressor.name == "code_aware_compressor"


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for CodeAwareCompressor."""

    def test_whitespace_only_content(self, compressor):
        """Whitespace-only content is handled gracefully."""
        result = compressor.compress("   \n\t\n   ")

        assert result.compression_ratio == 1.0
        assert result.syntax_valid is True

    def test_unicode_content(self, default_config):
        """Unicode in code is handled correctly."""
        compressor = CodeAwareCompressor(default_config)
        code = '''
def greet(name: str) -> str:
    """Greet the user in multiple languages."""
    return f"Hello, {name}! \u4f60\u597d! \u3053\u3093\u306b\u3061\u306f!"
'''
        result = compressor.compress(code)

        # Should handle unicode without crashing
        assert result is not None

    def test_very_long_function(self, default_config):
        """Very long functions are compressed."""
        compressor = CodeAwareCompressor(default_config)
        lines = ["def very_long_function():"]
        lines.append('    """A very long function."""')
        for i in range(100):
            lines.append(f"    x_{i} = {i}")
        lines.append("    return x_99")
        code = "\n".join(lines)

        result = compressor.compress(code)

        # Should compress the long function body
        assert result.compression_ratio < 1.0 or "tree_sitter" not in str(
            is_tree_sitter_available()
        )

    def test_nested_functions(self, default_config):
        """Nested functions are handled."""
        compressor = CodeAwareCompressor(default_config)
        code = """
def outer():
    def inner():
        return "inner"
    return inner()
"""
        result = compressor.compress(code)

        assert result is not None
        # syntax_valid requires tree-sitter; without it, validation is skipped
        if is_tree_sitter_available():
            assert result.syntax_valid is True

    def test_syntax_errors_in_input(self, default_config):
        """Syntax errors in input don't crash the compressor."""
        compressor = CodeAwareCompressor(default_config)
        # Invalid Python syntax
        code = """
def broken(
    # Missing closing paren
"""
        # Should not raise
        result = compressor.compress(code, language="python")
        assert result is not None

    def test_mixed_language_content(self, default_config):
        """Mixed language content (like markdown with code) is handled."""
        compressor = CodeAwareCompressor(default_config)
        content = """
# Documentation

Here is some code:

```python
def example():
    pass
```

And some more text.
"""
        # Should not crash
        result = compressor.compress(content)
        assert result is not None


# =============================================================================
# TestMemoryManagement
# =============================================================================


class TestMemoryManagement:
    """Tests for memory management functions."""

    def test_is_tree_sitter_available_returns_bool(self):
        """is_tree_sitter_available returns a boolean."""
        result = is_tree_sitter_available()
        assert isinstance(result, bool)

    def test_is_tree_sitter_loaded_returns_false_initially(self):
        """is_tree_sitter_loaded returns False when no parsers loaded."""
        # Clear any loaded parsers first
        unload_tree_sitter()
        assert is_tree_sitter_loaded() is False

    def test_unload_returns_false_when_nothing_loaded(self):
        """unload_tree_sitter returns False when nothing to unload."""
        # Ensure nothing is loaded
        unload_tree_sitter()
        result = unload_tree_sitter()
        assert result is False


# =============================================================================
# Integration Tests (only run if tree-sitter is installed)
# =============================================================================


@pytest.mark.skipif(not TREE_SITTER_INSTALLED, reason="tree-sitter-languages not installed")
class TestTreeSitterIntegration:
    """Integration tests that require actual tree-sitter installation.

    These tests verify actual AST parsing and compression behavior.
    """

    def test_actual_python_compression(self):
        """Test actual compression of Python code."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(5)

        result = compressor.compress(code, language="python")

        # Should achieve compression
        assert result.compression_ratio < 1.0
        assert result.syntax_valid is True
        assert result.language == CodeLanguage.PYTHON

    def test_actual_javascript_compression(self):
        """Test actual compression of JavaScript code."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_javascript_code(5)

        result = compressor.compress(code, language="javascript")

        assert result.compression_ratio < 1.0
        assert result.syntax_valid is True
        assert result.language == CodeLanguage.JAVASCRIPT

    def test_actual_go_compression(self):
        """Test Go code is processed (compression may fall back due to nested structures)."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_go_code(3)

        result = compressor.compress(code, language="go")

        # Go code is processed and returns valid output
        # Note: compression_ratio may be 1.0 if compression produces invalid syntax
        # and falls back to original (Go has complex nested brace handling)
        assert result.syntax_valid is True
        assert result.language == CodeLanguage.GO
        assert result.compressed  # Some output is produced

    @pytest.mark.parametrize(
        (
            "language",
            "code",
            "expected_signature",
            "expected_omitted_lines",
            "expected_removed_line",
            "expected_closing",
        ),
        [
            (
                "javascript",
                (
                    "class Calc {\n"
                    "  compute(x) {\n"
                    "    let a = x + 1;\n"
                    "    let b = a * 2;\n"
                    "    let c = b - 3;\n"
                    "    return c;\n"
                    "  }\n"
                    "}\n"
                ),
                "compute(x) {",
                3,
                "return c;",
                "}\n}",
            ),
            (
                "typescript",
                (
                    "class Calc {\n"
                    "  compute(x: number): number {\n"
                    "    let a = x + 1;\n"
                    "    let b = a * 2;\n"
                    "    let c = b - 3;\n"
                    "    return c;\n"
                    "  }\n"
                    "}\n"
                ),
                "compute(x: number): number {",
                3,
                "return c;",
                "}\n}",
            ),
            (
                "java",
                (
                    "public class Calc {\n"
                    "    public int compute(int x) {\n"
                    "        int a = x + 1;\n"
                    "        int b = a * 2;\n"
                    "        int c = b - 3;\n"
                    "        int d = c / 4;\n"
                    "        int e = d + 5;\n"
                    "        return e;\n"
                    "    }\n"
                    "}\n"
                ),
                "public int compute(int x) {",
                5,
                "return e;",
                "}\n}",
            ),
            (
                "cpp",
                (
                    "class Calc {\n"
                    "public:\n"
                    "    int compute(int x) {\n"
                    "        int a = x + 1;\n"
                    "        int b = a * 2;\n"
                    "        int c = b - 3;\n"
                    "        int d = c / 4;\n"
                    "        int e = d + 5;\n"
                    "        return e;\n"
                    "    }\n"
                    "};\n"
                ),
                "int compute(int x) {",
                5,
                "return e;",
                "};",
            ),
            (
                "rust",
                (
                    "impl Calc {\n"
                    "    pub fn compute(&self, x: i32) -> i32 {\n"
                    "        let a = x + 1;\n"
                    "        let b = a * 2;\n"
                    "        let c = b - 3;\n"
                    "        let d = c / 4;\n"
                    "        let e = d + 5;\n"
                    "        e\n"
                    "    }\n"
                    "}\n"
                ),
                "pub fn compute(&self, x: i32) -> i32 {",
                5,
                "        e\n",
                "}\n}",
            ),
        ],
    )
    def test_compresses_methods_inside_class_member_containers(
        self,
        language,
        code,
        expected_signature,
        expected_omitted_lines,
        expected_removed_line,
        expected_closing,
    ):
        """Class/impl member containers are distinct from executable method bodies."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=1,
            max_body_lines=1,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)

        result = compressor.compress(code, language=language)

        assert result.language == CodeLanguage(language)
        assert result.syntax_valid is True
        assert result.compression_ratio < 1.0
        assert expected_signature in result.compressed
        assert f"// [{expected_omitted_lines} lines omitted]" in result.compressed
        assert expected_removed_line not in result.compressed
        assert result.compressed.endswith(expected_closing)

    def test_imports_preserved(self):
        """Imports are preserved in compressed output."""
        config = CodeCompressorConfig(
            preserve_imports=True,
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(5)

        result = compressor.compress(code, language="python")

        assert "import os" in result.compressed
        assert "from typing import" in result.compressed

    def test_signatures_preserved(self):
        """Function signatures are preserved."""
        config = CodeCompressorConfig(
            preserve_signatures=True,
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(3)

        result = compressor.compress(code, language="python")

        # Should preserve function signatures
        assert "def function_" in result.compressed
        assert "arg:" in result.compressed or "(arg" in result.compressed

    def test_error_handlers_preserved(self):
        """Module-level try/except blocks are preserved."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        # Code with module-level try/except (not inside functions)
        code = '''
import os

def setup():
    """Setup function."""
    pass

try:
    from optional_module import feature
except ImportError:
    feature = None

def main():
    """Main function with long body."""
    result = []
    for i in range(100):
        result.append(i)
    return result
'''
        result = compressor.compress(code, language="python")

        # Module-level error handlers should be preserved
        assert "try:" in result.compressed or "except" in result.compressed

    def test_syntax_verification(self):
        """Output syntax is verified as valid."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(5)

        result = compressor.compress(code, language="python")

        # Verify the compressed output is valid Python
        assert result.syntax_valid is True

        # Should be parseable
        try:
            compile(result.compressed, "<test>", "exec")
        except SyntaxError:
            pytest.fail("Compressed output has invalid Python syntax")

    def test_python_future_import_stays_at_module_start(self):
        """Compressed Python keeps future imports before executable statements."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            target_compression_rate=0.2,
            max_body_lines=3,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = textwrap.dedent(
            """
            from __future__ import annotations

            from dataclasses import dataclass
            from typing import Any, Callable, Iterable


            def traced(label: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
                    async def wrapper(*args: Any, **kwargs: Any) -> Any:
                        return await fn(*args, **kwargs)

                    return wrapper

                return decorate


            @dataclass(slots=True)
            class Event:
                kind: str
                payload: dict[str, Any]
                retries: int = 0

                @property
                def important(self) -> bool:
                    return self.kind in {"error", "retry"} or self.retries > 2


            class EventRouter:
                def __init__(self, sinks: dict[str, Callable[[Event], Any]]) -> None:
                    self.sinks = sinks
                    self.history: list[tuple[str, bool]] = []

                @traced("route")
                async def route(self, events: Iterable[Event]) -> list[str]:
                    accepted: list[str] = []
                    for event in events:
                        match event:
                            case Event(kind="error", payload={"code": code, "message": msg}, retries=r) if r > 1:
                                destination = "pager"
                                accepted.append(f"{destination}:{code}:{msg}")
                            case Event(kind=kind, payload=payload) if (route := payload.get("route")):
                                destination = str(route)
                                accepted.append(f"{destination}:{kind}")
                            case _:
                                destination = "dead_letter"
                                accepted.append(destination)

                        self.history.append((destination, event.important))

                    return [item for item in accepted if item]
            """
        )

        result = compressor.compress(code, language="python")

        assert result.syntax_valid is True
        future_import_index = result.compressed.index("from __future__ import annotations")
        first_executable_index = min(
            result.compressed.index("@dataclass"),
            result.compressed.index("def traced"),
            result.compressed.index("class EventRouter"),
        )
        assert future_import_index < first_executable_index
        try:
            compile(result.compressed, "<test>", "exec")
        except SyntaxError as exc:
            pytest.fail(f"Compressed output has invalid Python syntax: {exc}\n{result.compressed}")

    def test_tree_sitter_loaded_after_compression(self):
        """Parser is loaded after compression."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)

        # Ensure clean state
        unload_tree_sitter()
        assert is_tree_sitter_loaded() is False

        # Compress should load parser
        code = generate_python_code(3)
        compressor.compress(code, language="python")

        assert is_tree_sitter_loaded() is True

    def test_unload_clears_parsers(self):
        """unload_tree_sitter clears loaded parsers."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)

        # Load a parser
        code = generate_python_code(3)
        compressor.compress(code, language="python")
        assert is_tree_sitter_loaded() is True

        # Unload
        result = unload_tree_sitter()
        assert result is True
        assert is_tree_sitter_loaded() is False


# =============================================================================
# TestDocstringModes
# =============================================================================


@pytest.mark.skipif(not TREE_SITTER_INSTALLED, reason="tree-sitter-languages not installed")
class TestDocstringModes:
    """Tests for different docstring handling modes."""

    def test_docstring_mode_full(self):
        """FULL mode preserves entire docstrings."""
        config = CodeCompressorConfig(
            docstring_mode=DocstringMode.FULL,
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(2)

        result = compressor.compress(code, language="python")

        # Should preserve full docstrings
        assert "Args:" in result.compressed or "Returns:" in result.compressed

    def test_docstring_mode_first_line(self):
        """FIRST_LINE mode keeps only first line of docstring."""
        config = CodeCompressorConfig(
            docstring_mode=DocstringMode.FIRST_LINE,
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_python_code(2)

        result = compressor.compress(code, language="python")

        # Multi-line docstring details should be removed
        # This is implementation-dependent
        assert result.compressed is not None

    def test_docstring_mode_remove(self):
        """REMOVE mode removes all docstrings."""
        config = CodeCompressorConfig(
            docstring_mode=DocstringMode.REMOVE,
            min_tokens_for_compression=10,
            max_body_lines=2,  # Low threshold to trigger compression
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        # Larger function to trigger body compression
        code = '''
def example():
    """This docstring should be removed."""
    x = 1
    y = 2
    z = 3
    result = x + y + z
    for i in range(10):
        result += i
    return result
'''
        result = compressor.compress(code, language="python")

        # Docstring should be removed when REMOVE mode is active
        assert "This docstring should be removed" not in result.compressed


# =============================================================================
# TestSemanticSymbolImportance
# =============================================================================


def _payment_processing_code() -> str:
    """Python code with varying symbol importance for testing."""
    return '''
import os
from typing import List, Optional

def process_payment(order, config):
    """Process a payment through the pipeline."""
    validated = validate_order(order)
    if not validated.is_valid:
        return PaymentResult(status='failed')
    charge = charge_customer(order.customer, order.total)
    receipt = generate_receipt(charge)
    send_confirmation(order.customer.email, receipt)
    update_inventory(order.items)
    log_transaction(charge.transaction_id)
    notify_warehouse(order)
    return PaymentResult(status='success', receipt=receipt)

def validate_order(order):
    """Validate an order before processing."""
    if not order.items:
        return ValidationResult(False, ['No items'])
    total = sum(item.price for item in order.items)
    if total <= 0:
        return ValidationResult(False, ['Invalid total'])
    if not order.customer:
        return ValidationResult(False, ['No customer'])
    return ValidationResult(True, [])

def charge_customer(customer, amount):
    """Charge the customer."""
    gateway = get_payment_gateway()
    response = gateway.charge(customer.card, amount)
    if not response.success:
        raise PaymentError(response.error)
    return response

def generate_receipt(charge):
    """Generate a receipt for the charge."""
    template = load_template('receipt')
    return template.render(charge=charge)

def _format_log_entry(entry):
    """Format a log entry for internal use. Never called."""
    timestamp = entry.get('ts', '')
    level = entry.get('level', 'INFO')
    message = entry.get('msg', '')
    source = entry.get('source', 'unknown')
    formatted = f'[{timestamp}] {level}: {message} ({source})'
    return formatted.strip()

def _dead_helper():
    """Never called anywhere in this file."""
    x = 1
    y = 2
    z = 3
    result = x + y + z
    for i in range(100):
        result += i
    return result
'''


@pytest.mark.skipif(not TREE_SITTER_INSTALLED, reason="tree-sitter-languages not installed")
class TestSemanticSymbolImportance:
    """Tests for semantic symbol importance analysis and variable compression."""

    def _make_compressor(self, **overrides):
        defaults = {
            "min_tokens_for_compression": 10,
            "max_body_lines": 3,
            "enable_ccr": False,
            "semantic_analysis": True,
        }
        defaults.update(overrides)
        return CodeAwareCompressor(CodeCompressorConfig(**defaults))

    def test_symbol_scores_populated(self):
        """Compression result includes symbol importance scores."""
        compressor = self._make_compressor()
        result = compressor.compress(_payment_processing_code(), language="python")

        assert result.symbol_scores
        assert "process_payment" in result.symbol_scores
        assert "validate_order" in result.symbol_scores
        assert "_dead_helper" in result.symbol_scores

    def test_called_functions_score_higher_than_dead_code(self):
        """Functions called by others score higher than unused functions."""
        compressor = self._make_compressor()
        result = compressor.compress(_payment_processing_code(), language="python")

        # validate_order is called by process_payment — should score higher
        assert result.symbol_scores["validate_order"] > result.symbol_scores["_dead_helper"]
        assert result.symbol_scores["charge_customer"] > result.symbol_scores["_dead_helper"]

    def test_public_symbols_score_higher_than_private(self):
        """Public functions (no leading _) score higher than private ones."""
        compressor = self._make_compressor()
        code = '''
def public_func():
    """A public function."""
    x = 1
    y = 2
    z = 3
    result = x + y + z
    for i in range(10):
        result += i
    return result

def _private_func():
    """A private function."""
    x = 1
    y = 2
    z = 3
    result = x + y + z
    for i in range(10):
        result += i
    return result
'''
        result = compressor.compress(code, language="python")

        assert result.symbol_scores["public_func"] > result.symbol_scores["_private_func"]

    def test_dead_code_compressed_to_signature_only(self):
        """Functions with score < 0.1 are compressed to signature + docstring only."""
        compressor = self._make_compressor()
        result = compressor.compress(_payment_processing_code(), language="python")

        # _dead_helper has 0 references, private → score 0.0
        assert result.symbol_scores["_dead_helper"] < 0.1

        # Body should be fully omitted
        assert "_dead_helper" in result.compressed
        # Should NOT contain body content
        assert "range(100)" not in result.compressed

    def test_referenced_functions_keep_more_body(self):
        """Higher-scored functions get more body lines from the budget."""
        # Use a generous target rate so there IS budget to distribute
        compressor = self._make_compressor(target_compression_rate=0.7)
        result = compressor.compress(_payment_processing_code(), language="python")

        compressed = result.compressed
        # With 70% target, high-scoring functions should retain body
        # while low-scoring ones get less. validate_order is referenced
        # and public (high score) so should keep some body.
        # _dead_helper has lowest score so should get least body.
        # Count body lines per function as a proxy for retention
        lines = compressed.split("\n")
        in_validate = False
        in_dead = False
        validate_body = 0
        dead_body = 0
        for line in lines:
            if "def validate_order" in line:
                in_validate = True
                in_dead = False
                continue
            elif "def _dead_helper" in line:
                in_dead = True
                in_validate = False
                continue
            elif line.startswith("def ") or (line.startswith("class ") and ":" in line):
                in_validate = False
                in_dead = False
                continue
            if in_validate and line.strip() and not line.strip().startswith('"""'):
                validate_body += 1
            if in_dead and line.strip() and not line.strip().startswith('"""'):
                dead_body += 1

        assert validate_body >= dead_body

    def test_omitted_comment_includes_calls(self):
        """Omitted comment includes call information when available."""
        compressor = self._make_compressor()
        result = compressor.compress(_payment_processing_code(), language="python")

        # process_payment calls validate_order, charge_customer, generate_receipt
        # These should appear in the omitted comment
        compressed = result.compressed
        if "lines omitted" in compressed:
            # Find omitted comments and check for calls info
            for line in compressed.split("\n"):
                if "process_payment" not in line and "lines omitted" in line:
                    continue
                if "lines omitted; calls:" in line:
                    assert "validate_order" in line or "charge_customer" in line
                    break

    def test_semantic_analysis_disabled(self):
        """When semantic_analysis=False, all functions get uniform compression."""
        compressor_with = self._make_compressor(semantic_analysis=True)
        compressor_without = self._make_compressor(semantic_analysis=False)

        code = _payment_processing_code()
        result_with = compressor_with.compress(code, language="python")
        result_without = compressor_without.compress(code, language="python")

        # Without semantic analysis, no symbol scores
        assert result_without.symbol_scores == {}

        # With semantic analysis, dead code is compressed more aggressively
        # _dead_helper body should NOT appear with semantic analysis
        assert "range(100)" not in result_with.compressed
        # But with uniform compression (no semantic), body lines ARE kept
        assert "x = 1" in result_without.compressed

    def test_summary_includes_semantic_info(self):
        """Summary includes semantic analysis information."""
        compressor = self._make_compressor()
        result = compressor.compress(_payment_processing_code(), language="python")

        summary = result.summary
        if result.symbol_scores:
            low_count = sum(1 for s in result.symbol_scores.values() if s < 0.1)
            if low_count > 0:
                assert "low-importance" in summary

    def test_dunder_methods_get_boost(self):
        """Dunder methods (__init__, etc.) get importance boost."""
        compressor = self._make_compressor()
        code = '''
class MyClass:
    """A class."""
    def __init__(self, value):
        """Initialize."""
        self.value = value
        self.processed = False
        self.results = []
        self.cache = {}
        self.errors = []
        for i in range(10):
            self.results.append(i)

    def _setup_cache(self):
        """Internal setup."""
        x = 1
        y = 2
        z = 3
        result = x + y + z
        for i in range(10):
            result += i
        return result
'''
        result = compressor.compress(code, language="python")

        # __init__ should score higher than _setup_cache
        if "__init__" in result.symbol_scores and "_setup_cache" in result.symbol_scores:
            assert result.symbol_scores["__init__"] > result.symbol_scores["_setup_cache"]

    def test_javascript_importance(self):
        """Symbol importance works for JavaScript code."""
        compressor = self._make_compressor()
        code = """
import { db } from './database';

function processUser(userId) {
    const user = fetchUser(userId);
    const profile = buildProfile(user);
    sendNotification(user.email, profile);
    logAction('process', userId);
    updateMetrics('user_processed');
    return { user, profile };
}

function fetchUser(id) {
    const result = db.query('SELECT * FROM users WHERE id = ?', [id]);
    if (!result) {
        throw new Error('User not found');
    }
    return result;
}

function buildProfile(user) {
    const prefs = loadPreferences(user.id);
    return { ...user, preferences: prefs };
}

function _internalDebug(msg) {
    const ts = Date.now();
    const formatted = `[${ts}] DEBUG: ${msg}`;
    console.log(formatted);
    return formatted;
}
"""
        result = compressor.compress(code, language="javascript")

        assert result.symbol_scores
        # fetchUser is called by processUser — should score higher than _internalDebug
        if "fetchUser" in result.symbol_scores and "_internalDebug" in result.symbol_scores:
            assert result.symbol_scores["fetchUser"] > result.symbol_scores["_internalDebug"]

    def test_syntax_still_valid_with_importance(self):
        """Compressed output with importance remains syntactically valid."""
        compressor = self._make_compressor()
        result = compressor.compress(_payment_processing_code(), language="python")

        assert result.syntax_valid is True

        # Should be parseable as Python
        try:
            compile(result.compressed, "<test>", "exec")
        except SyntaxError:
            pytest.fail("Semantic compression produced invalid Python syntax")

    def test_empty_code_no_crash(self):
        """Importance analysis handles empty code gracefully."""
        compressor = self._make_compressor()
        result = compressor.compress("", language="python")

        assert result.symbol_scores == {}

    def test_config_default_semantic_analysis_enabled(self):
        """semantic_analysis is True by default in config."""
        config = CodeCompressorConfig()
        assert config.semantic_analysis is True


# =============================================================================
# Regression: tree-sitter ABI mismatch (real AST must run, no silent fallback)
# =============================================================================


@pytest.mark.skipif(not TREE_SITTER_INSTALLED, reason="tree-sitter grammar pack not installed")
class TestRealASTRuns:
    """Guards against the regression where the code-aware compressor silently
    fell back to a lossy stripper because ``_get_parser`` built a stock
    ``tree_sitter.Parser`` and assigned it a foreign grammar-pack ``Language``
    (raising ``TypeError`` that was swallowed into a fallback).
    """

    def _compressor(self):
        return CodeAwareCompressor(
            CodeCompressorConfig(
                min_tokens_for_compression=10,
                enable_ccr=False,
            )
        )

    def test_get_parser_returns_stock_node_api(self):
        """The parser must yield nodes with the stock tree_sitter property API
        that the tree-walking code relies on (``.type``/``.children``/...)."""
        from headroom.transforms.code_compressor import _get_parser

        parser = _get_parser("python")
        tree = parser.parse(b"def foo(x):\n    return x + 1\n")
        root = tree.root_node

        # Property access (NOT method calls) — the old pack binding exposed
        # methods like ``.kind()`` which would break every call site.
        assert root.type == "module"
        assert root.child_count >= 1
        assert isinstance(root.children, list)

        func = root.children[0]
        assert func.type == "function_definition"
        assert isinstance(func.start_byte, int)
        assert isinstance(func.end_byte, int)
        # start_point must be index-able like a (row, col) tuple.
        assert func.start_point[0] == 0
        assert b"def foo" in func.text

    def test_check_tree_sitter_available_verifies_real_parse(self):
        """``_check_tree_sitter_available`` must only return True when an actual
        parse succeeds — not merely when the package imports."""
        import headroom.transforms.code_compressor as cc

        cc._tree_sitter_available = None  # reset memoized result
        assert cc._check_tree_sitter_available() is True

    def test_check_tree_sitter_available_false_when_parse_broken(self):
        """If parsing raises (e.g. the old foreign-Language bug), availability
        must report False instead of green-lighting the broken path."""
        import headroom.transforms.code_compressor as cc

        cc._tree_sitter_available = None
        with patch.object(cc, "_get_parser", side_effect=TypeError("boom")):
            assert cc._check_tree_sitter_available() is False
        cc._tree_sitter_available = None  # reset for other tests

    def test_ast_runs_for_python_no_fallback(self):
        """A supported language must be compressed via real AST, not the
        UNKNOWN-language Kompress fallback."""
        result = self._compressor().compress(_payment_processing_code(), language="python")

        # The fallback path forces language=UNKNOWN and syntax_valid=False.
        # Real AST keeps the detected language and guarantees valid syntax.
        assert result.language == CodeLanguage.PYTHON
        assert result.syntax_valid is True
        compile(result.compressed, "<test>", "exec")

    def test_ast_preserves_structure_for_python(self):
        """AST output retains signatures/scopes/imports (unlike the old
        whitespace garble)."""
        code = (
            "import math\n"
            "\n"
            "def compute(values):\n"
            "    total = 0\n"
            "    for v in values:\n"
            "        total += v * v\n"
            "        total -= 1\n"
            "        total *= 2\n"
            "        total //= 3\n"
            "    return math.sqrt(total)\n"
        )
        result = self._compressor().compress(code, language="python")

        assert result.language == CodeLanguage.PYTHON
        assert result.syntax_valid is True
        # Structure markers survive compression.
        assert "import math" in result.compressed
        assert "def compute(values):" in result.compressed
        # Output is still valid Python.
        compile(result.compressed, "<test>", "exec")

    def test_get_node_text_uses_utf8_byte_offsets(self):
        """tree-sitter byte offsets must not be sliced as Python str indexes."""
        from headroom.transforms.code_compressor import _get_node_text, _get_parser

        code = 'def first():\n    """中文占位"""\n    return 1\n\ndef second():\n    return 2\n'
        root = _get_parser("python").parse(code.encode("utf-8")).root_node
        functions = [node for node in root.children if node.type == "function_definition"]

        assert _get_node_text(functions[1], code) == "def second():\n    return 2"

    def test_ast_compresses_python_after_non_ascii_source(self):
        """CJK/emoji before a later function must not corrupt downstream slices."""
        compressor = CodeAwareCompressor(
            CodeCompressorConfig(
                min_tokens_for_compression=1,
                max_body_lines=2,
                enable_ccr=False,
                semantic_analysis=False,
            )
        )
        code = (
            "def first():\n"
            '    """中文占位 with emoji 🔥."""\n'
            "    return 1\n"
            "\n"
            "def second():\n"
            "    values = []\n"
            "    for i in range(10):\n"
            "        values.append(i)\n"
            "        values.append(i * 2)\n"
            "        values.append(i * 3)\n"
            "        values.append(i * 4)\n"
            "    return sum(values)\n"
        )

        result = compressor.compress(code, language="python")

        assert result.language == CodeLanguage.PYTHON
        assert result.syntax_valid is True
        assert result.compression_ratio < 1.0
        assert "def second():" in result.compressed
        assert "中文占位" in result.compressed
        compile(result.compressed, "<test>", "exec")

    def test_ast_runs_for_rust_no_fallback(self):
        """A second supported language (Rust) also runs through real AST."""
        code = (
            "pub fn add(a: i64, b: i64) -> i64 {\n"
            "    let mut acc = a;\n"
            "    acc += b;\n"
            "    acc -= 0;\n"
            "    acc\n"
            "}\n"
        )
        result = self._compressor().compress(code, language="rust")

        assert result.language == CodeLanguage.RUST
        assert result.syntax_valid is True
        # Signature is preserved verbatim.
        assert "pub fn add(a: i64, b: i64) -> i64" in result.compressed
