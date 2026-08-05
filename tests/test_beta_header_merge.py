from __future__ import annotations

from headroom.proxy.beta_header_merge import (
    merge_anthropic_beta,
    merge_beta_tokens,
    merge_openai_beta,
    split_beta_tokens,
)


def test_split_beta_tokens_drops_empty_entries() -> None:
    assert split_beta_tokens(None) == []
    assert split_beta_tokens("") == []
    assert split_beta_tokens(" alpha, , beta ,, ") == ["alpha", "beta"]


def test_merge_beta_tokens_preserves_client_order_then_appends_required() -> None:
    assert merge_beta_tokens("client-1,client-2", ["required-1", "required-2"]) == (
        "client-1,client-2,required-1,required-2"
    )


def test_merge_beta_tokens_dedupes_case_insensitively_with_first_casing() -> None:
    assert merge_beta_tokens("Foo,foo", ["FOO", "bar"]) == "Foo,bar"


def test_merge_beta_tokens_skips_empty_required_values() -> None:
    assert merge_beta_tokens("alpha", ["", " beta ", "  "]) == "alpha,beta"


def test_provider_wrappers_share_merge_semantics() -> None:
    assert merge_anthropic_beta("a", ["b"]) == "a,b"
    assert merge_openai_beta("a", ["b"]) == "a,b"
