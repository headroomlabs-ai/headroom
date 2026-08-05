from unittest.mock import patch

import pytest

import headroom.transforms.code_compressor as cc
from headroom.transforms.code_compressor import (
    CodeAwareCompressor,
    CodeCompressionResult,
    CodeCompressorConfig,
    CodeLanguage,
    unload_tree_sitter,
)
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig


@pytest.fixture(autouse=True)
def _reset_tree_sitter():
    unload_tree_sitter()
    yield
    unload_tree_sitter()


def _compressor(**overrides) -> CodeAwareCompressor:
    defaults = {
        "min_tokens_for_compression": 1,
        "fallback_to_kompress": False,
        "enable_ccr": False,
    }
    defaults.update(overrides)
    return CodeAwareCompressor(CodeCompressorConfig(**defaults))


def _record_parser_calls(monkeypatch):
    calls: list[str] = []
    real_get_parser = cc._get_parser

    def spy(language: str):
        calls.append(language)
        return real_get_parser(language)

    monkeypatch.setattr(cc, "_get_parser", spy)
    return calls


def test_typescript_with_decorators_stays_typescript_without_perl(monkeypatch):
    tree_sitter_installed = cc._tree_sitter_importable()
    calls = _record_parser_calls(monkeypatch)
    code = """
import { Component, Input } from "@angular/core";

@Component({
  selector: "demo-card",
  template: "<div>{{ title }}</div>",
})
export class DemoCardComponent {
  @Input() title: string = "";

  render(items: string[]): string {
    return items.map((item) => item.trim()).join(",");
  }
}
""".strip()

    result = _compressor().compress(code)

    assert result.language == CodeLanguage.TYPESCRIPT
    assert "perl" not in calls
    if tree_sitter_installed:
        assert "typescript" in calls


def test_explicit_perl_uses_existing_fallback_without_requesting_parser(monkeypatch):
    calls = _record_parser_calls(monkeypatch)
    compressor = _compressor(fallback_to_kompress=True)
    code = "use strict;\nsub demo {\n    my $value = shift;\n    return $value;\n}\n"
    sentinel = CodeCompressionResult(
        compressed="kompress sentinel",
        original=code,
        original_tokens=10,
        compressed_tokens=3,
        compression_ratio=0.3,
        language=CodeLanguage.UNKNOWN,
        language_confidence=0.0,
        syntax_valid=False,
    )

    with patch.object(compressor, "_fallback_compress", return_value=sentinel) as fallback:
        result = compressor.compress(code, language="perl")

    assert result is sentinel
    fallback.assert_called_once()
    assert "perl" not in calls


@pytest.mark.parametrize("hint", ["perl", "pl"])
def test_explicit_perl_passthrough_keeps_original_without_parser(monkeypatch, hint):
    calls = _record_parser_calls(monkeypatch)
    code = "use strict;\nsub demo {\n    my $value = shift;\n    return $value;\n}\n"

    result = _compressor().compress(code, language=hint)

    assert result.compressed == code
    assert result.language == CodeLanguage.PERL
    assert "perl" not in calls


def test_real_perl_detects_unknown_without_requesting_parser(monkeypatch):
    calls = _record_parser_calls(monkeypatch)
    code = """
package Demo;
use strict;
use warnings;

sub greet {
    my ($name) = @_;
    return "hello $name";
}

1;
""".strip()

    result = _compressor().compress(code)

    assert result.language == CodeLanguage.UNKNOWN
    assert result.compressed == code
    assert "perl" not in calls


def test_fenced_perl_router_path_never_requests_parser(monkeypatch):
    calls = _record_parser_calls(monkeypatch)
    router = ContentRouter(ContentRouterConfig(enable_code_aware=True, min_section_tokens=1))
    content = """
```perl
use strict;
use warnings;

sub greet {
    my ($name) = @_;
    return "hello $name";
}
```
""".strip()

    result = router.compress(content)

    assert "```perl" in result.compressed
    assert "perl" not in calls


def test_go_still_uses_non_perl_parser(monkeypatch):
    tree_sitter_installed = cc._tree_sitter_importable()
    calls = _record_parser_calls(monkeypatch)
    code = """
package main

import "fmt"

func main() {
    fmt.Println("hello")
}
""".strip()

    result = _compressor().compress(code)

    assert result.language == CodeLanguage.GO
    assert "perl" not in calls
    if tree_sitter_installed:
        assert "go" in calls


def test_get_parser_refuses_quarantined_perl():
    with pytest.raises(ValueError, match="quarantined"):
        cc._get_parser("perl")
