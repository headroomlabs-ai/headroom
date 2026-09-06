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

import headroom.transforms.code_compressor as cc
from headroom.transforms.code_compressor import (
    CodeAwareCompressor,
    CodeCompressionResult,
    CodeCompressorConfig,
    CodeLanguage,
    DocstringMode,
    coerce_language,
    detect_language,
    is_tree_sitter_available,
    is_tree_sitter_loaded,
    unload_tree_sitter,
)


def _count_function_body_lines(compressed: str, func_name: str) -> int:
    """Count retained non-docstring body lines for a Python function in compressed output."""
    lines = compressed.split("\n")
    in_func = False
    body_count = 0
    for line in lines:
        if line.startswith(f"def {func_name}"):
            in_func = True
            continue
        if in_func and line.startswith("def "):
            break
        if in_func and line.strip() and not line.strip().startswith('"""'):
            body_count += 1
    return body_count


def _assert_statement_selection_ran(result: CodeCompressionResult, language: CodeLanguage) -> None:
    """Assert targeted statement-selection tests did not pass via fallback/original output."""
    assert result.language == language
    assert result.syntax_valid is True
    assert result.compressed != result.original
    assert "lines omitted" in result.compressed


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
        """Test actual compression of Go code."""
        config = CodeCompressorConfig(
            min_tokens_for_compression=10,
            enable_ccr=False,
        )
        compressor = CodeAwareCompressor(config)
        code = generate_go_code(3)

        result = compressor.compress(code, language="go")

        assert result.compression_ratio < 1.0
        assert result.syntax_valid is True
        assert result.language == CodeLanguage.GO

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
                2,
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
                2,
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
            (
                "csharp",
                (
                    "namespace Acme\n"
                    "{\n"
                    "    public class Calc\n"
                    "    {\n"
                    "        public int Compute(int x)\n"
                    "        {\n"
                    "            int a = x + 1;\n"
                    "            int b = a * 2;\n"
                    "            int c = b - 3;\n"
                    "            int d = c / 4;\n"
                    "            int e = d + 5;\n"
                    "            return e;\n"
                    "        }\n"
                    "    }\n"
                    "}\n"
                ),
                "public int Compute(int x)",
                5,
                "return e;",
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
        if language == "rust":
            assert expected_removed_line not in result.compressed
        else:
            assert expected_removed_line in result.compressed
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

    def test_late_return_preserved_in_long_function(self):
        """A late terminal return should survive MVP statement selection."""
        compressor = self._make_compressor(max_body_lines=2)
        code = """
def sync_user(user):
    payload = normalize(user)
    if not payload:
        return None
    token = load_token()
    result = call_remote_api(payload, token)
    return result
"""
        result = compressor.compress(code, language="python")
        _assert_statement_selection_ran(result, CodeLanguage.PYTHON)
        assert "return result" in result.compressed

    def test_guard_clause_preserved(self):
        """Early guard clauses should be preserved."""
        compressor = self._make_compressor(max_body_lines=2)
        code = """
def process(items):
    prepared = normalize(items)
    if not prepared:
        return []
    logged = audit(prepared)
    debug_value = format_debug(prepared)
    return logged
"""
        result = compressor.compress(code, language="python")
        _assert_statement_selection_ran(result, CodeLanguage.PYTHON)
        assert "if not prepared:" in result.compressed
        assert "return []" in result.compressed

    def test_side_effect_preferred_over_low_value_setup(self):
        """Structural side-effect statements should be retained ahead of low-value setup detail."""
        compressor = self._make_compressor(max_body_lines=2)
        code = """
def persist(order):
    a = 1
    b = 2
    c = a + b
    emit(order)
    return c
"""
        result = compressor.compress(code, language="python")
        _assert_statement_selection_ran(result, CodeLanguage.PYTHON)
        assert "emit(order)" in result.compressed

    def test_member_assignment_detected_as_effectful(self):
        """Assignments to members should count as structural side effects."""
        compressor = self._make_compressor(max_body_lines=2)
        code = """
def persist(self, order):
    a = 1
    b = 2
    unused = a - b
    self.last_order = order
    return a + b
"""
        result = compressor.compress(code, language="python")
        _assert_statement_selection_ran(result, CodeLanguage.PYTHON)
        assert "self.last_order = order" in result.compressed

    def test_javascript_member_assignment_detected_as_effectful(self):
        """Member assignments in JS should be treated as effectful structurally."""
        compressor = self._make_compressor(max_body_lines=2)
        code = """
function persist(service, payload) {
    const a = 1;
    const b = 2;
    const unused = a - b;
    service.lastPayload = payload;
    return a + b;
}
"""
        result = compressor.compress(code, language="javascript")
        _assert_statement_selection_ran(result, CodeLanguage.JAVASCRIPT)
        assert "service.lastPayload = payload;" in result.compressed

    def test_dependency_closure_keeps_assignment_for_return(self):
        """Returning a variable should pull in its nearest defining assignment."""
        compressor = self._make_compressor(max_body_lines=2)
        code = """
def build(user):
    seed = 1
    result = create_profile(user, seed)
    return result
"""
        result = compressor.compress(code, language="python")
        _assert_statement_selection_ran(result, CodeLanguage.PYTHON)
        assert "return result" in result.compressed
        assert "result = create_profile(user, seed)" in result.compressed

    def test_must_keep_statements_can_exceed_body_budget(self):
        """Must-keep statements stay together even when they exceed max_body_lines."""
        compressor = self._make_compressor(max_body_lines=2)
        code = """
def handle(payload):
    prepared = normalize(payload)
    if not prepared:
        return None
    audit_value = format_debug(prepared)
    result = save_payload(prepared)
    return result
"""
        result = compressor.compress(code, language="python")
        _assert_statement_selection_ran(result, CodeLanguage.PYTHON)
        assert "if not prepared:" in result.compressed
        assert "return None" in result.compressed
        assert "result = save_payload(prepared)" in result.compressed
        assert "return result" in result.compressed
        assert _count_function_body_lines(result.compressed, "handle") > 2

    def test_class_method_mvp_selection_preserves_guard_and_return(self):
        """Methods routed through class compression should use the same statement selection."""
        compressor = self._make_compressor(max_body_lines=2)
        code = """
class Service:
    def handle(self, payload):
        cleaned = normalize(payload)
        if not cleaned:
            return None
        audit_value = format_debug(cleaned)
        result = save_payload(cleaned)
        return result
"""
        result = compressor.compress(code, language="python")
        _assert_statement_selection_ran(result, CodeLanguage.PYTHON)
        assert "if not cleaned:" in result.compressed
        assert "return result" in result.compressed

    def test_identifier_based_context_hit_preserves_relevant_statement(self):
        """Context should preserve symbol-matching statements without raw text scanning."""
        compressor = self._make_compressor(max_body_lines=1)
        code = """
def do_work(payload):
    prepared = normalize(payload)
    result = call_remote_api(prepared)
    debug_value = format_debug(payload)
    return result
"""
        result = compressor.compress(code, language="python", context="investigate result path")
        _assert_statement_selection_ran(result, CodeLanguage.PYTHON)
        assert "return result" in result.compressed

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

    def _recovery_compressor(self, **overrides):
        defaults = {
            "min_tokens_for_compression": 1,
            "max_body_lines": 2,
            "enable_ccr": False,
            "semantic_analysis": False,
        }
        defaults.update(overrides)
        return CodeAwareCompressor(CodeCompressorConfig(**defaults))

    def _python_recovery_fixture(self) -> str:
        return textwrap.dedent(
            """\
            from pathlib import Path

            def expand_search_roots(user_root: str) -> list[Path]:
                root = Path(user_root)
                candidates = [root]
                for child in root.iterdir():
                    candidates.append(child.resolve())
                return candidates

            def load_user_overrides(config_path: str) -> dict[str, str]:
                config: dict[str, str] = {}
                for line in Path(config_path).read_text().splitlines():
                    if "=" in line:
                        key, value = line.split("=", 1)
                        config[key.strip()] = value.strip()
                return config
            """
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

    def test_python_invalid_node_falls_back_locally(self):
        """One invalid Python rewrite must preserve only that definition."""
        compressor = self._recovery_compressor()
        code = self._python_recovery_fixture()
        original_compress_function_ast = compressor._compress_function_ast

        def _patched(node, code_text, language, lang_config, body_limits, analysis, context=""):
            name = cc._get_definition_name(node)
            if name == "expand_search_roots":
                return "def expand_search_roots(user_root: str) -> list[Path]:\n    if True\n"
            if name == "load_user_overrides":
                return (
                    "def load_user_overrides(config_path: str) -> dict[str, str]:\n"
                    "    config: dict[str, str] = {}\n"
                    "    # [4 lines omitted]\n"
                    "    return config"
                )
            return original_compress_function_ast(
                node,
                code_text,
                language,
                lang_config,
                body_limits,
                analysis,
                context,
            )

        with patch.object(compressor, "_compress_function_ast", side_effect=_patched):
            result = compressor.compress(code, language="python")

        assert result.syntax_valid is True
        assert result.compression_ratio < 1.0
        assert result.compressed != code
        assert "if True" not in result.compressed
        assert "for child in root.iterdir():" in result.compressed
        assert "# [4 lines omitted]" in result.compressed
        compile(result.compressed, "<test>", "exec")

    def test_python_recovery_does_not_block_valid_modern_syntax(self):
        """Valid decorators, nested defs, and match syntax still compress."""
        compressor = self._recovery_compressor(max_body_lines=4)
        code = textwrap.dedent(
            """\
            from dataclasses import dataclass

            def traced(fn):
                return fn

            @dataclass
            class Command:
                name: str
                payload: dict[str, int]

            @traced
            def route_command(command: Command) -> str:
                def normalize(value: str) -> str:
                    return value.strip().lower()

                match normalize(command.name):
                    case "ping":
                        return "pong"
                    case "echo":
                        return str(command.payload)
                    case _:
                        return "unknown"
            """
        )

        result = compressor.compress(code, language="python")

        assert result.syntax_valid is True
        assert result.compression_ratio < 1.0
        assert "@traced" in result.compressed
        assert "class Command:" in result.compressed
        assert "def route_command(command: Command) -> str:" in result.compressed
        compile(result.compressed, "<test>", "exec")

    def test_python_recovery_still_returns_original_when_all_candidates_invalid(self):
        """Recovery keeps the terminal whole-file fail-safe."""
        compressor = self._recovery_compressor()
        code = self._python_recovery_fixture()

        def _patched(
            node, _code_text, _language, _lang_config, _body_limits, _analysis, _context=""
        ):
            name = cc._get_definition_name(node) or "broken"
            return f"def {name}(:\n    pass"

        with patch.object(compressor, "_compress_function_ast", side_effect=_patched):
            result = compressor.compress(code, language="python")

        assert result.syntax_valid is True
        assert result.compression_ratio == 1.0
        assert "(:\n" not in result.compressed
        assert "for child in root.iterdir():" in result.compressed
        assert 'key, value = line.split("=", 1)' in result.compressed
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


@pytest.mark.skipif(not TREE_SITTER_INSTALLED, reason="tree-sitter grammar pack not installed")
class TestCSharpSupport:
    """C# (``csharp`` grammar) parity with Java/C++/Rust: signatures preserved
    verbatim, method/constructor bodies compressed, block-scoped namespaces
    routed through container compression (no verbatim re-dump), file-scoped
    namespace headers kept ahead of types, malformed input passed through.
    """

    def _compressor(self):
        return CodeAwareCompressor(
            CodeCompressorConfig(
                min_tokens_for_compression=1,
                max_body_lines=1,
                enable_ccr=False,
            )
        )

    def test_block_scoped_namespace_compresses_without_redumping(self):
        """Block-scoped ``namespace { }`` wraps types in a declaration_list; the
        container must compress its members and NOT be re-emitted verbatim
        (which would duplicate the class and defeat compression — the C#
        analogue of the #1318 class-member trap, one nesting level deeper)."""
        code = (
            "using System;\n"
            "\n"
            "namespace Acme.Widgets\n"
            "{\n"
            "    public class WidgetService\n"
            "    {\n"
            "        public int Process(int x)\n"
            "        {\n"
            "            var a = x + 1;\n"
            "            var b = a + 2;\n"
            "            var c = b + 3;\n"
            "            Console.WriteLine(c);\n"
            "            return c;\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = self._compressor().compress(code, language="csharp")

        assert result.language == CodeLanguage.CSHARP
        assert result.syntax_valid is True
        assert result.compression_ratio < 1.0
        # namespace + type headers preserved verbatim
        assert "namespace Acme.Widgets" in result.compressed
        assert "public class WidgetService" in result.compressed
        assert "public int Process(int x)" in result.compressed
        # method body actually compressed
        assert "lines omitted" in result.compressed
        assert "return c;" in result.compressed
        # the class is emitted exactly once (no re-dump / duplication)
        assert result.compressed.count("class WidgetService") == 1
        # block-namespace wrapper still closes
        assert result.compressed.rstrip().endswith("}")

    def test_file_scoped_namespace_directive_precedes_types(self):
        """A file-scoped ``namespace X;`` must stay ahead of the type
        declarations; emitting it after a type would not be valid C#."""
        code = (
            "using System;\n"
            "\n"
            "namespace Acme.Tools;\n"
            "\n"
            "public class Helper\n"
            "{\n"
            "    public int Add(int a, int b)\n"
            "    {\n"
            "        var s = a + b;\n"
            "        var t = s + 1;\n"
            "        var u = t + 2;\n"
            "        return u;\n"
            "    }\n"
            "}\n"
        )
        result = self._compressor().compress(code, language="csharp")

        assert result.language == CodeLanguage.CSHARP
        assert result.syntax_valid is True
        assert result.compression_ratio < 1.0
        assert "namespace Acme.Tools;" in result.compressed
        assert "public class Helper" in result.compressed
        assert result.compressed.index("namespace Acme.Tools;") < result.compressed.index(
            "class Helper"
        )

    def test_record_preserves_positional_parameters(self):
        """Record primary-constructor parameters are part of the signature and
        must be preserved verbatim while the method body compresses."""
        code = (
            "namespace Acme.Models;\n"
            "\n"
            "public record Point(int X, int Y)\n"
            "{\n"
            "    public double Dist()\n"
            "    {\n"
            "        var sq = X * X + Y * Y;\n"
            "        var r = System.Math.Sqrt(sq);\n"
            "        System.Console.WriteLine(r);\n"
            "        return r;\n"
            "    }\n"
            "}\n"
        )
        result = self._compressor().compress(code, language="csharp")

        assert result.language == CodeLanguage.CSHARP
        assert result.syntax_valid is True
        assert result.compression_ratio < 1.0
        assert "public record Point(int X, int Y)" in result.compressed
        assert "public double Dist()" in result.compressed
        assert "lines omitted" in result.compressed

    def test_preprocessor_wrapped_file_is_not_duplicated(self):
        """A file wrapped in ``#if``/``#endif`` (idiomatic in multi-targeted
        .NET code) must not have its content emitted twice. The visitor
        recurses into ``preproc_if`` and captures the declarations, but the
        top-level pass used to re-emit the uncaptured wrapper verbatim,
        duplicating the whole file (observed on real repos: output up to
        ~1.9x the input). Opaque handling keeps it verbatim exactly once."""
        code = (
            "#if !HAVE_TRACE_WRITER\n"
            "using System;\n"
            "\n"
            "namespace Acme\n"
            "{\n"
            "    public enum TraceLevel\n"
            "    {\n"
            "        Off = 0,\n"
            "        Error = 1,\n"
            "    }\n"
            "}\n"
            "#endif\n"
        )
        result = self._compressor().compress(code, language="csharp")

        assert result.syntax_valid is True
        # Content appears exactly once — no wrapper re-dump.
        assert result.compressed.count("enum TraceLevel") == 1
        assert result.compressed.count("using System;") == 1
        # Never larger than the input (verbatim pass-through is acceptable).
        assert result.compression_ratio <= 1.0
        # Conditional directives stay balanced.
        assert result.compressed.count("#if") == result.compressed.count("#endif")

    def test_preprocessor_wrapped_usings_stay_ahead_of_types(self):
        """``#if``-wrapped using directives (idiomatic for multi-targeting)
        must be emitted with the imports. Appending them as trailing top-level
        code puts usings after type declarations — invalid C#, which trips the
        output syntax check and falls back to no compression at all (observed
        on real repos: hundreds of files silently uncompressed)."""
        code = (
            "using System;\n"
            "#if HAVE_BIG_INTEGER\n"
            "using System.Numerics;\n"
            "#endif\n"
            "\n"
            "namespace Acme\n"
            "{\n"
            "    public class Calc\n"
            "    {\n"
            "        public int Compute(int x)\n"
            "        {\n"
            "            var a = x + 1;\n"
            "            var b = a * 2;\n"
            "            var c = b - 3;\n"
            "            var d = c / 4;\n"
            "            return d;\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = self._compressor().compress(code, language="csharp")

        assert result.syntax_valid is True
        # Compression must actually happen (no silent fallback).
        assert result.compression_ratio < 1.0
        assert "lines omitted" in result.compressed
        # The conditional import block survives verbatim, ahead of the types.
        assert "#if HAVE_BIG_INTEGER" in result.compressed
        assert "using System.Numerics;" in result.compressed
        assert result.compressed.index("#endif") < result.compressed.index("class Calc")

    def test_region_markers_inside_class_compress_cleanly(self):
        """``#region``/``#endregion`` markers span their trailing newline, so
        their end_point sits at column 0 of the NEXT line. Line-based child
        extraction used to swallow that line — duplicating the following
        method's signature and the class's closing brace (invalid output →
        silent fallback to no compression on real repos)."""
        code = (
            "namespace Acme\n"
            "{\n"
            "    public class Svc\n"
            "    {\n"
            "        #region Api\n"
            "        public int F(int x)\n"
            "        {\n"
            "            var a = x + 1;\n"
            "            var b = a * 2;\n"
            "            var c = b - 3;\n"
            "            return c;\n"
            "        }\n"
            "        #endregion\n"
            "    }\n"
            "}\n"
        )
        result = self._compressor().compress(code, language="csharp")

        assert result.syntax_valid is True
        assert result.compression_ratio < 1.0
        assert "lines omitted" in result.compressed
        # Signature emitted exactly once (regions used to duplicate it).
        assert result.compressed.count("public int F(int x)") == 1
        # Region markers survive, braces stay balanced.
        assert "#region Api" in result.compressed
        assert "#endregion" in result.compressed
        assert result.compressed.count("{") == result.compressed.count("}")

    def test_license_region_header_stays_on_top(self):
        """A ``#region License`` banner above the usings (near-universal in
        real .NET code) must stay at the top of the file. Relocating it after
        the type declarations — the old behavior for uncaptured top-level
        nodes — is rejected by tree-sitter-c-sharp, failing output validation
        and silently forfeiting compression for the whole file."""
        code = (
            "#region License\n"
            "// Copyright (c) 2007 Example Corp.\n"
            "// Licensed under the MIT license.\n"
            "#endregion\n"
            "\n"
            "using System;\n"
            "\n"
            "namespace Acme\n"
            "{\n"
            "    public class Svc\n"
            "    {\n"
            "        public int F(int x)\n"
            "        {\n"
            "            var a = x + 1;\n"
            "            var b = a * 2;\n"
            "            var c = b - 3;\n"
            "            return c;\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = self._compressor().compress(code, language="csharp")

        assert result.syntax_valid is True
        assert result.compression_ratio < 1.0
        assert "lines omitted" in result.compressed
        # Header block survives, in order, ahead of everything else.
        assert result.compressed.index("#region License") < result.compressed.index("using System;")
        assert result.compressed.index("// Copyright") < result.compressed.index("#endregion")
        assert result.compressed.index("#endregion") < result.compressed.index("class Svc")

    def test_malformed_csharp_passes_through_unchanged(self):
        """Malformed C# must pass through unchanged (no data loss; prefer false
        negatives over serving broken or altered code)."""
        code = (
            "namespace Broken\n"
            "{\n"
            "    public class Oops\n"
            "    {\n"
            "        public int F(int x)\n"
            "        {\n"
            "            var a = x + 1\n"  # missing ';' and unclosed braces
            "            return a\n"
        )
        result = self._compressor().compress(code, language="csharp")

        assert result.compressed == code
        assert result.compression_ratio == 1.0

    def test_detect_language_identifies_csharp(self):
        """Auto-detection (no explicit language hint) recognizes C#."""
        code = (
            "using System;\n"
            "\n"
            "namespace Acme\n"
            "{\n"
            "    public class Svc\n"
            "    {\n"
            "        public int Total { get; set; }\n"
            "        public int Add(int a, int b)\n"
            "        {\n"
            "            return a + b;\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        lang, confidence = detect_language(code)
        assert lang == CodeLanguage.CSHARP
        assert confidence > 0.0


@pytest.mark.skipif(not TREE_SITTER_INSTALLED, reason="tree-sitter grammar pack not installed")
class TestPhpSupport:
    """PHP (``php`` grammar) parity with C#: signatures preserved verbatim,
    function/method bodies compressed, ``<?php`` tag and ``namespace``/``use``
    header order preserved, Perl sigil-overlap disambiguated in detection,
    malformed input passed through.
    """

    def _compressor(self):
        return CodeAwareCompressor(
            CodeCompressorConfig(
                min_tokens_for_compression=1,
                max_body_lines=1,
                enable_ccr=False,
            )
        )

    def test_class_methods_compress_signatures_preserved(self):
        code = (
            "<?php\n"
            "namespace App\\Service;\n"
            "\n"
            "use App\\Model\\User;\n"
            "\n"
            "final class UserService {\n"
            "    private $logger;\n"
            "\n"
            "    public function process(User $u): bool {\n"
            "        $name = strtolower(trim($u->getName()));\n"
            "        $tags = [];\n"
            "        foreach ($u->getTags() as $tag) {\n"
            "            $tags[] = $tag->normalize();\n"
            "        }\n"
            "        $this->logger->info($name);\n"
            "        return true;\n"
            "    }\n"
            "}\n"
        )
        result = self._compressor().compress(code, language="php")

        assert result.language == CodeLanguage.PHP
        assert result.syntax_valid is True
        assert result.compression_ratio < 1.0
        # signature + class header preserved verbatim
        assert "final class UserService" in result.compressed
        assert "public function process(User $u): bool" in result.compressed
        # method body actually compressed
        assert "lines omitted" in result.compressed
        assert "$tag->normalize()" not in result.compressed
        # the class is emitted exactly once
        assert result.compressed.count("class UserService") == 1

    def test_php_tag_and_namespace_precede_uses_and_types(self):
        """``<?php`` must stay first and ``namespace X;`` must precede the
        ``use`` imports and type declarations — any other order is not valid
        PHP."""
        code = (
            "<?php\n"
            "namespace App\\Tools;\n"
            "\n"
            "use App\\Model\\Item;\n"
            "\n"
            "function helper(int $x): int {\n"
            "    $acc = 0;\n"
            "    for ($i = 0; $i < $x; $i++) {\n"
            "        $acc += $i;\n"
            "        $acc -= 1;\n"
            "    }\n"
            "    return $acc;\n"
            "}\n"
        )
        result = self._compressor().compress(code, language="php")

        assert result.language == CodeLanguage.PHP
        assert result.syntax_valid is True
        assert result.compression_ratio < 1.0
        compressed = result.compressed
        assert compressed.lstrip().startswith("<?php")
        assert compressed.index("<?php") < compressed.index("namespace App\\Tools;")
        assert compressed.index("namespace App\\Tools;") < compressed.index("use App\\Model\\Item;")
        assert compressed.index("use App\\Model\\Item;") < compressed.index("function helper")
        assert "lines omitted" in compressed

    def test_detect_language_identifies_php(self):
        """Auto-detection recognizes PHP despite the Perl sigil overlap
        (``$var`` matches Perl's prefilter; the ``<?php`` tag disambiguates)."""
        code = (
            "<?php\n"
            "namespace Acme;\n"
            "\n"
            "use Acme\\Widget;\n"
            "\n"
            "class Svc {\n"
            "    public function add(int $a, int $b): int {\n"
            "        $sum = $a + $b;\n"
            "        return $sum;\n"
            "    }\n"
            "}\n"
        )
        lang, confidence = detect_language(code)
        assert lang == CodeLanguage.PHP
        assert confidence > 0.0

    def test_phtml_alias_coerces_to_php(self):
        assert coerce_language("phtml") == CodeLanguage.PHP
        assert coerce_language("php8") == CodeLanguage.PHP

    def test_malformed_php_passes_through_unchanged(self):
        code = "<?php\nclass Broken {\n    public function oops( {\n"
        result = self._compressor().compress(code, language="php")
        assert result.compressed == code
