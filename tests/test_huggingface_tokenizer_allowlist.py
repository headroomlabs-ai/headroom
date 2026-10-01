"""The HF tokenizer loader must only ever fetch repositories we ship (A-1).

``AutoTokenizer.from_pretrained`` resolves a name against the HuggingFace Hub and,
with ``trust_remote_code``, executes Python that the repository owner published.
The identifier reaching this module can come straight off a proxied request body:
the proxy picks a tokenizer from the request's ``model`` field, and the resolver's
old fallthrough was "assume the model name is the tokenizer name". Publishing a
repo and then naming it in a request body was therefore code execution in the
proxy process.

The fix is two-layered and fails closed: ``get_tokenizer_name`` never returns a
caller-controlled string, and ``_load_tokenizer`` refuses anything off the
allowlist before it imports transformers at all.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from headroom.tokenizers import huggingface as hf_mod
from headroom.tokenizers.huggingface import (
    MODEL_TO_TOKENIZER,
    HuggingFaceTokenizer,
    _load_tokenizer,
    get_tokenizer_name,
)

# A repository id an attacker could plausibly stand up and then request by name.
CRAFTED = "attacker-controlled/headroom-rce-poc"

SHIPPED_REPOS = frozenset(MODEL_TO_TOKENIZER.values())


@pytest.fixture(autouse=True)
def _fresh_caches():
    """Both caches are process-wide; env-driven tests need them cleared."""
    index = getattr(hf_mod, "_allowlist_index", None)
    _load_tokenizer.cache_clear()
    if index is not None:
        index.cache_clear()
    yield
    _load_tokenizer.cache_clear()
    if index is not None:
        index.cache_clear()


class _FakeTokenizer:
    """Minimal stand-in: enough surface for count_text/encode."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch, from_pretrained) -> None:
    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = type(
        "AutoTokenizer", (), {"from_pretrained": staticmethod(from_pretrained)}
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)


# ---------------------------------------------------------------------------
# A crafted model id must not reach the loader
# ---------------------------------------------------------------------------


def test_crafted_model_never_reaches_the_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exploit path end to end: request body -> tokenizer -> Hub fetch."""
    names: list[str] = []

    def fake_from_pretrained(name: str, **kwargs: Any):
        names.append(name)
        return _FakeTokenizer()

    _install_fake_transformers(monkeypatch, fake_from_pretrained)

    counter = HuggingFaceTokenizer(CRAFTED)
    assert counter.count_text("hello world") > 0  # counting still works

    assert CRAFTED not in names, "attacker-named repository was handed to from_pretrained"
    assert set(names) <= SHIPPED_REPOS, f"unallowlisted repository fetched: {names}"


def test_crafted_model_resolves_to_a_shipped_tokenizer() -> None:
    for model in (
        CRAFTED,
        "../../../etc/passwd",
        "evil/repo",
        "https://example.invalid/evil",
        "",
        "x" * 500,
    ):
        assert get_tokenizer_name(model) in SHIPPED_REPOS, f"{model!r} escaped the allowlist"


def test_crafted_model_resolves_to_the_declared_default() -> None:
    assert get_tokenizer_name(CRAFTED) == hf_mod.DEFAULT_TOKENIZER
    assert hf_mod.DEFAULT_TOKENIZER in hf_mod.ALLOWED_TOKENIZER_REPOS


def test_prefix_match_on_a_crafted_name_still_lands_on_a_shipped_repo() -> None:
    """A crafted id may *start* with a family alias; that path is safe too."""
    assert get_tokenizer_name("llama-3-evil/rce") == MODEL_TO_TOKENIZER["llama-3"]


def test_loader_refuses_unallowlisted_repo_without_touching_the_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loader is the chokepoint, not just the resolver."""

    def fake_from_pretrained(name: str, **kwargs: Any):
        raise AssertionError(f"Hub lookup attempted for {name!r}")

    _install_fake_transformers(monkeypatch, fake_from_pretrained)
    monkeypatch.setenv("HEADROOM_HF_TOKENIZER_LOAD_TIMEOUT_SECS", "5")

    assert _load_tokenizer(CRAFTED) is None
    assert _load_tokenizer("meta-llama/Meta-Llama-3-8B-but-not-really") is None


def test_loader_fetches_the_allowlists_own_spelling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Case-insensitive matching must not let the caller pick the fetched string."""
    names: list[str] = []

    def fake_from_pretrained(name: str, **kwargs: Any):
        names.append(name)
        return _FakeTokenizer()

    _install_fake_transformers(monkeypatch, fake_from_pretrained)

    assert _load_tokenizer("  META-LLAMA/llama-3.1-8b  ") is not None
    assert names == ["meta-llama/Llama-3.1-8B"]


# ---------------------------------------------------------------------------
# trust_remote_code is off on every path
# ---------------------------------------------------------------------------

# Built by concatenation so this file contributes no literal occurrence of the
# token to the tree-wide scan in test_no_trust_remote_code.py.
_REMOTE_CODE_KWARG = "trust_remote" + "_code"


def test_remote_code_disabled_on_cache_and_network_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_from_pretrained(name: str, **kwargs: Any):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("not in cache")
        return _FakeTokenizer()

    _install_fake_transformers(monkeypatch, fake_from_pretrained)
    monkeypatch.setenv("HEADROOM_HF_TOKENIZER_LOAD_TIMEOUT_SECS", "5")

    assert _load_tokenizer("Qwen/Qwen2.5-7B") is not None
    assert len(calls) == 2, "expected the cache-only attempt then the bounded network one"
    for kwargs in calls:
        assert kwargs.get(_REMOTE_CODE_KWARG) is False


# ---------------------------------------------------------------------------
# Legitimate resolutions are unchanged
# ---------------------------------------------------------------------------


def test_every_shipped_alias_still_resolves_to_its_mapping() -> None:
    for alias, repo in MODEL_TO_TOKENIZER.items():
        assert get_tokenizer_name(alias) == repo


def test_full_repo_id_for_a_shipped_tokenizer_is_accepted_verbatim() -> None:
    """Callers pass the repo id instead of the alias; counts must not change.

    The allowlist is consulted *after* alias and prefix matching, so ids that a
    family prefix already claimed keep resolving exactly as they did before —
    ``mistralai/Mixtral-8x7B-v0.1`` still hits the "mistral" prefix, exactly as
    it did before. This fix changes no resolution that previously worked.
    """
    assert get_tokenizer_name("meta-llama/Llama-3.1-8B") == "meta-llama/Llama-3.1-8B"
    assert get_tokenizer_name("google/gemma-2-27b") == "google/gemma-2-27b"
    assert get_tokenizer_name("microsoft/Phi-3-mini-4k-instruct") == (
        "microsoft/Phi-3-mini-4k-instruct"
    )
    # Unchanged prefix-match precedence, shown explicitly so a future reorder
    # of the resolver has to acknowledge it.
    assert get_tokenizer_name("mistralai/Mixtral-8x7B-v0.1") == "mistralai/Mistral-7B-v0.1"


def test_repo_id_matching_is_case_insensitive_but_returns_our_spelling() -> None:
    assert get_tokenizer_name("META-LLAMA/llama-3.1-8b") == "meta-llama/Llama-3.1-8B"


# ---------------------------------------------------------------------------
# Operator escape hatch (server-side env only)
# ---------------------------------------------------------------------------


def test_operator_env_allowlist_admits_a_private_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    names: list[str] = []

    def fake_from_pretrained(name: str, **kwargs: Any):
        names.append(name)
        return _FakeTokenizer()

    _install_fake_transformers(monkeypatch, fake_from_pretrained)
    monkeypatch.setenv(hf_mod._ALLOWLIST_ENV, "acme/internal-tokenizer, acme/other")

    assert _load_tokenizer("acme/internal-tokenizer") is not None
    assert names == ["acme/internal-tokenizer"]

    # Still closed for everything the operator did not name.
    assert _load_tokenizer(CRAFTED) is None
    assert CRAFTED not in names


def test_empty_operator_env_admits_nothing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(hf_mod._ALLOWLIST_ENV, " , ,")
    assert get_tokenizer_name(CRAFTED) == hf_mod.DEFAULT_TOKENIZER
