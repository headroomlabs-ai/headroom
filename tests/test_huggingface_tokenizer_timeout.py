"""HF tokenizer loading must be bounded (GH #1701): AutoTokenizer.from_pretrained
performs unbounded network downloads/retries; called lazily from the proxy's request
path it blocked the event loop for ~10 minutes and zombified the server. The fix
tries the local HF cache first (local_files_only=True), bounds the network attempt
with HEADROOM_HF_TOKENIZER_LOAD_TIMEOUT_SECS on a daemon thread, and fails open to
estimation — caching the failure so the hub is probed at most once per process.
"""

from __future__ import annotations

import sys
import time
import types
from typing import Any

import pytest

from headroom.tokenizers import huggingface as hf_mod
from headroom.tokenizers.huggingface import (
    HuggingFaceTokenizer,
    _load_tokenizer,
    get_tokenizer_name,
)


@pytest.fixture(autouse=True)
def _fresh_cache():
    _load_tokenizer.cache_clear()
    yield
    _load_tokenizer.cache_clear()


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch, from_pretrained) -> None:
    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = type(
        "AutoTokenizer", (), {"from_pretrained": staticmethod(from_pretrained)}
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)


def test_local_cache_tried_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_from_pretrained(name: str, **kwargs: Any):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("not in cache")
        return "network-tokenizer"

    _install_fake_transformers(monkeypatch, fake_from_pretrained)
    monkeypatch.setenv("HEADROOM_HF_TOKENIZER_LOAD_TIMEOUT_SECS", "5")

    assert _load_tokenizer("some/model") == "network-tokenizer"
    assert calls[0].get("local_files_only") is True, "first attempt must be cache-only"
    assert not calls[1].get("local_files_only")


def test_cache_hit_never_touches_network(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_from_pretrained(name: str, **kwargs: Any):
        calls.append(kwargs)
        return "cached-tokenizer"

    _install_fake_transformers(monkeypatch, fake_from_pretrained)

    assert _load_tokenizer("some/model") == "cached-tokenizer"
    assert len(calls) == 1
    assert calls[0].get("local_files_only") is True


def test_slow_network_load_times_out_and_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_from_pretrained(name: str, **kwargs: Any):
        if kwargs.get("local_files_only"):
            raise OSError("not in cache")
        time.sleep(60)  # simulates hung huggingface_hub download
        return "never"

    _install_fake_transformers(monkeypatch, fake_from_pretrained)
    monkeypatch.setenv("HEADROOM_HF_TOKENIZER_LOAD_TIMEOUT_SECS", "0.2")

    start = time.monotonic()
    assert _load_tokenizer("slow/model") is None
    assert time.monotonic() - start < 5, "load must unblock at the timeout, not the download"

    # Failure is cached (lru_cache) — the second call must not re-probe the hub.
    start = time.monotonic()
    assert _load_tokenizer("slow/model") is None
    assert time.monotonic() - start < 0.05


def test_timeout_zero_disables_network_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_from_pretrained(name: str, **kwargs: Any):
        if kwargs.get("local_files_only"):
            raise OSError("not in cache")
        raise AssertionError("network load attempted despite timeout=0")

    _install_fake_transformers(monkeypatch, fake_from_pretrained)
    monkeypatch.setenv("HEADROOM_HF_TOKENIZER_LOAD_TIMEOUT_SECS", "0")

    assert _load_tokenizer("offline/model") is None


def test_count_messages_fails_open_to_estimation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_from_pretrained(name: str, **kwargs: Any):
        raise OSError("unavailable")

    _install_fake_transformers(monkeypatch, fake_from_pretrained)
    monkeypatch.setenv("HEADROOM_HF_TOKENIZER_LOAD_TIMEOUT_SECS", "0.2")

    counter = HuggingFaceTokenizer("deepseek-chat")
    tokens = counter.count_messages([{"role": "user", "content": "hello world" * 50}])
    assert tokens > 0  # estimation fallback, no exception, no hang


def test_deepseek_model_aliases_resolve_to_expected_tokenizers() -> None:
    assert get_tokenizer_name("deepseek-v3.2") == "deepseek-ai/DeepSeek-V3.2"
    assert get_tokenizer_name("deepseek-v4-pro") == "deepseek-ai/DeepSeek-V4-Pro"
    assert get_tokenizer_name("deepseek-v4-flash") == "deepseek-ai/DeepSeek-V4-Flash"
    assert get_tokenizer_name("deepseek-r1") == "deepseek-ai/DeepSeek-R1"
    assert get_tokenizer_name("deepseek-r1-0528") == "deepseek-ai/DeepSeek-R1-0528"


def test_invalid_timeout_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_HF_TOKENIZER_LOAD_TIMEOUT_SECS", "not-a-number")
    assert hf_mod._load_timeout_secs() == hf_mod._LOAD_TIMEOUT_DEFAULT


def test_get_tokenizer_name_prefers_most_specific_prefix() -> None:
    """A more-specific family key must win over a shorter one.

    Prefix matching used to scan MODEL_TO_TOKENIZER in dict-insertion order, so
    the short "qwen" key preceded "qwen2"/"qwen2.5" and shadowed them —
    "qwen2-7b-instruct" resolved to the Qwen1 tokenizer (a different vocabulary,
    hence wrong counts). The resolver now picks the longest matching prefix.
    """
    # Versioned models not present as literal keys must hit the right family.
    assert get_tokenizer_name("qwen2-7b-instruct") == "Qwen/Qwen2-7B"
    assert get_tokenizer_name("qwen2.5-turbo") == "Qwen/Qwen2.5-7B"
    assert get_tokenizer_name("deepseek-v2.5") == "deepseek-ai/DeepSeek-V2"
    # Direct hits and shorter family fallbacks still resolve through their
    # longest matching tokenizer aliases.
    assert get_tokenizer_name("qwen-14b") == "Qwen/Qwen-14B"
    assert get_tokenizer_name("deepseek-chat") == "deepseek-ai/DeepSeek-V3"
