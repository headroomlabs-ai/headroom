"""Query sources must survive the actual tokenizer's finite input window."""

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing

from headroom.memory.adapters.embedders import _fit_query_to_window


def make_tokenizer() -> Tokenizer:
    words = [
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[PAD]",
        "question",
        "answer",
        "tool",
        "database",
        "corruption",
        "古",
        "い",
        "質",
        "問",
        "出",
        "力",
        "答",
        "案",
        "破",
        "損",
        "故",
        "障",
    ]
    vocab = {word: index for index, word in enumerate(words)}
    tokenizer = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.normalizer = BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return tokenizer


def test_long_question_cannot_consume_the_tool_evidence_window() -> None:
    tokenizer = make_tokenizer()
    question = "question " * 300 + "answer"
    context = "tool " * 300 + "database corruption"
    rendered = _fit_query_to_window(
        question,
        context,
        14,
        lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
    )
    inference = Tokenizer.from_str(tokenizer.to_str())
    inference.enable_truncation(max_length=16)
    ids = inference.encode(rendered).ids

    assert len(tokenizer.encode(rendered).ids) <= 16
    assert tokenizer.token_to_id("answer") in ids
    assert tokenizer.token_to_id("database") in ids
    assert tokenizer.token_to_id("corruption") in ids


def test_unicode_sources_are_budgeted_by_tokens_not_python_characters() -> None:
    tokenizer = make_tokenizer()
    question = "古い質問" * 100 + "答案"
    context = "古い出力" * 100 + "破損故障"
    rendered = _fit_query_to_window(
        question,
        context,
        14,
        lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
    )
    inference = Tokenizer.from_str(tokenizer.to_str())
    inference.enable_truncation(max_length=16)
    ids = inference.encode(rendered).ids

    assert len(tokenizer.encode(rendered).ids) <= 16
    for character in "答案破損故障":
        assert tokenizer.token_to_id(character) in ids
