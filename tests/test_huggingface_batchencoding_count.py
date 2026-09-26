"""Regression test: count_messages must unwrap BatchEncoding from apply_chat_template.

Recent transformers versions return a BatchEncoding (dict-like with
``input_ids``/``attention_mask``) from ``apply_chat_template(tokenize=True)``.
``len()`` of it is the number of keys (2), not the token count, which made
every proxy request look like optimization "inflated" tokens (2 -> N) and
silently reverted all compression on HF-tokenized models (DeepSeek, Qwen, ...).
"""

from unittest.mock import patch

from headroom.tokenizers.huggingface import HuggingFaceTokenizer


class _BatchEncodingLike(dict):
    """Mimics transformers.BatchEncoding: dict of lists."""


class _FakeInnerTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return _BatchEncodingLike(
            input_ids=list(range(1234)),
            attention_mask=[1] * 1234,
        )

    def encode(self, text, add_special_tokens=False):
        return [0] * max(1, len(text) // 4)


def _make_tokenizer(inner):
    with patch.object(HuggingFaceTokenizer, "__init__", lambda self, model: None):
        tok = HuggingFaceTokenizer.__new__(HuggingFaceTokenizer)
    tok.model = "deepseek-chat"
    tok.tokenizer_name = "fake"
    tok._tokenizer = inner
    return tok


def test_count_messages_unwraps_batch_encoding():
    tok = _make_tokenizer(_FakeInnerTokenizer())
    messages = [{"role": "system", "content": "word " * 5000}]
    assert tok.count_messages(messages) == 1234


def test_count_messages_accepts_flat_token_list():
    class _LegacyInner(_FakeInnerTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            return list(range(777))

    tok = _make_tokenizer(_LegacyInner())
    messages = [{"role": "user", "content": "hello"}]
    assert tok.count_messages(messages) == 777
