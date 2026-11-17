"""The fixture recorder exposes only its eight pinned grammar wheels."""

import pytest

GRAMMARS = ("python", "javascript", "typescript", "go", "rust", "java", "c", "cpp")


@pytest.fixture
def recorder_parsers(monkeypatch):
    pytest.importorskip("tree_sitter")
    for language in GRAMMARS:
        pytest.importorskip(f"tree_sitter_{language}")
    import headroom.transforms.code_compressor as compressor
    from tests.parity.recorder import install_individual_grammar_parsers

    monkeypatch.setattr(compressor, "_get_parser", compressor._get_parser)
    monkeypatch.setattr(
        compressor, "_check_tree_sitter_available", compressor._check_tree_sitter_available
    )
    install_individual_grammar_parsers()
    return compressor._get_parser


@pytest.mark.parametrize("language", GRAMMARS)
def test_recorder_retains_each_pinned_grammar(recorder_parsers, language):
    assert recorder_parsers(language).parse(b"").root_node is not None


def test_recorder_unsupported_grammar_uses_detector_skip_contract(recorder_parsers):
    with pytest.raises(ValueError, match="unsupported recorder grammar: bash"):
        recorder_parsers("bash")
