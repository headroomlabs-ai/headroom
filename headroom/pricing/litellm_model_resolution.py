"""Pure LiteLLM model-name resolution rules."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# Vertex AI appends @YYYYMMDD version tags to model names at runtime
# (e.g. "claude-haiku-4-5@20251001"). LiteLLM's database stores bare
# names without version suffixes, so we strip the suffix before lookup.
_VERTEX_VERSION_SUFFIX_RE = re.compile(r"@\d{8}$")


@dataclass(frozen=True, slots=True)
class LiteLLMModelPrefixRule:
    """Case-insensitive bare-model prefix mapping to a LiteLLM provider key."""

    model_prefix: str
    provider_prefix: str

    def candidate_for(self, model: str) -> str | None:
        if model.lower().startswith(self.model_prefix):
            return f"{self.provider_prefix}{model}"
        return None


# Aliases for models removed from LiteLLM's cost database (retired/renamed).
# Maps old model name -> current LiteLLM key that has equivalent pricing.
MODEL_ALIASES: dict[str, str] = {
    # Claude 3.5 Sonnet retired Feb 2026, pricing same as claude-sonnet-4-20250514
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20240620": "claude-sonnet-4-20250514",
    # Claude 3 Sonnet retired. It was a Sonnet-tier model ($3/$15 per 1M
    # in/out) — same price as claude-sonnet-4-20250514 — so alias it there.
    # The old target, claude-3-haiku-20240307 ($0.25/$1.25), is a different
    # (Haiku) tier and underpriced every cost/savings figure ~12x.
    "claude-3-sonnet-20240229": "claude-sonnet-4-20250514",
}


MODEL_PREFIX_RULES: tuple[LiteLLMModelPrefixRule, ...] = (
    LiteLLMModelPrefixRule("claude-", "anthropic/"),
    LiteLLMModelPrefixRule("gpt-", "openai/"),
    LiteLLMModelPrefixRule("o1-", "openai/"),
    LiteLLMModelPrefixRule("o3-", "openai/"),
    LiteLLMModelPrefixRule("o4-", "openai/"),
    LiteLLMModelPrefixRule("gemini-", "google/"),
    LiteLLMModelPrefixRule("minimax-", "minimax/"),
    LiteLLMModelPrefixRule("deepseek-", "deepseek/"),
)


PRICE_LOOKUP_PROVIDER_PREFIXES: tuple[str, ...] = (
    "openai/",
    "anthropic/",
    "google/",
    "mistral/",
    "deepseek/",
    "minimax/",
)


def _strip_vertex_version_suffix(model: str) -> str:
    """Strip Vertex @YYYYMMDD version suffix if present."""
    return _VERTEX_VERSION_SUFFIX_RE.sub("", model)


def resolution_candidates(model: str) -> tuple[str, ...]:
    """Return ordered LiteLLM keys to try for cost-per-token resolution."""
    candidates = [model]

    # If the model has a Vertex @YYYYMMDD version suffix, also try the bare
    # name. Vertex appends these at runtime; LiteLLM stores bare names only.
    bare = _strip_vertex_version_suffix(model)
    is_vertex_versioned = bare != model
    if is_vertex_versioned:
        candidates.append(bare)

    # Apply prefix rules to both the original and bare name so that e.g.
    # "anthropic/claude-haiku-4-5" is tried after "claude-haiku-4-5".
    for m in dict.fromkeys([model, bare]):
        candidates.extend(
            candidate
            for rule in MODEL_PREFIX_RULES
            for candidate in (rule.candidate_for(m),)
            if candidate is not None
        )

    # Only add vertex_ai/ candidates for models with @YYYYMMDD suffix —
    # these are known Vertex-routed models. Non-versioned models should not
    # get vertex_ai/ candidates to avoid matching wrong pricing tier.
    if is_vertex_versioned:
        for m in dict.fromkeys([model, bare]):
            candidates.append(f"vertex_ai/{m}")

    alias = MODEL_ALIASES.get(model) or MODEL_ALIASES.get(bare)
    if alias:
        candidates.append(alias)
    return tuple(dict.fromkeys(candidates))


def unwrapped_model_forms(model: str) -> tuple[str, ...]:
    """Progressively drop leading gateway segments: ``a/b/c`` -> ``b/c``, ``c``.

    A gateway-routed name wraps the real model id, and ``litellm.model_cost`` keys
    the *unwrapped* form -- e.g. it has ``anthropic.claude-3-5-sonnet-20241022-v2:0``
    but not ``bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0``. Deriving the
    forms by splitting on ``/`` avoids maintaining yet another list of gateway
    prefixes, and costs nothing when wrong: each candidate is an exact dict
    lookup, so a bad guess simply misses.
    """
    parts = model.split("/")
    return tuple("/".join(parts[i:]) for i in range(1, len(parts)))


def pricing_lookup_candidates(model: str) -> tuple[str, ...]:
    """Return ordered LiteLLM model_cost keys to try for pricing lookup."""
    bare = _strip_vertex_version_suffix(model)
    is_vertex_versioned = bare != model

    candidates = [model]
    if is_vertex_versioned:
        candidates.append(bare)

    # Try all provider prefixes for both the original and bare name.
    for m in dict.fromkeys([model, bare]):
        candidates.extend(f"{prefix}{m}" for prefix in PRICE_LOOKUP_PROVIDER_PREFIXES)

    # Unwrapped forms come after the prefixed ones so existing precedence is
    # unchanged for names that already resolved.
    for m in dict.fromkeys([model, bare]):
        candidates.extend(unwrapped_model_forms(m))

    # Only add vertex_ai/ candidates for models with @YYYYMMDD suffix.
    if is_vertex_versioned:
        for m in dict.fromkeys([model, bare]):
            candidates.append(f"vertex_ai/{m}")

    candidates.extend(
        alias for candidate in tuple(candidates) if (alias := MODEL_ALIASES.get(candidate))
    )
    return tuple(dict.fromkeys(candidates))


def resolve_litellm_model_name(
    model: str,
    is_known_model: Callable[[str], bool],
) -> str:
    """Resolve ``model`` to the first candidate accepted by LiteLLM."""
    for candidate in resolution_candidates(model):
        if is_known_model(candidate):
            return candidate
    return model
