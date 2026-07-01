"""multi-wiki-qa multilingual loader registration.

The eval framework's DATASET_REGISTRY only had English datasets, so the
LLM-in-the-loop runner could not be pointed at Chinese/Japanese/Korean. This
registers `alexandrainst/multi-wiki-qa` (verbatim-span answers over full
Wikipedia articles, uniform zh/ja/ko). The live HF load is exercised in the
PR's Real Behavior Proof, not here, to keep the test offline (mirrors the other
dataset loaders).
"""

from headroom.evals.datasets import DATASET_REGISTRY, load_multi_wiki_qa


def test_multi_wiki_qa_registered():
    assert "multi_wiki_qa" in DATASET_REGISTRY
    entry = DATASET_REGISTRY["multi_wiki_qa"]
    assert entry["loader"] is load_multi_wiki_qa
    assert entry["category"] == "rag_multilingual"
    # same 4-key shape as every other registry entry
    assert set(entry) == {"loader", "description", "category", "default_n"}


def test_multi_wiki_qa_default_lang_is_callable():
    # signature is (n, lang) like the other loaders; default lang is a real config
    import inspect

    sig = inspect.signature(load_multi_wiki_qa)
    assert list(sig.parameters) == ["n", "lang"]
    assert sig.parameters["lang"].default in {"ja", "ko", "zh-cn"}
