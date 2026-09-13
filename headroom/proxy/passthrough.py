"""Shared passthrough routing and telemetry helpers."""

from __future__ import annotations

from urllib.parse import urlparse

OPENCODE_ZEN_HOSTS = {"opencode.ai", "www.opencode.ai"}

# OpenAI-compatible endpoints (chat/completions, GitHub Copilot, custom bases)
# carry non-OpenAI models whose provider would otherwise collapse into the
# generic ``openai`` dashboard bucket. Classify by model id so per-provider
# reporting (dashboard ``by_provider`` / Prometheus ``requests_by_provider``)
# attributes them correctly. Ordered longest-prefix-first so more specific ids
# win. Display-only: pricing stays keyed on the model, not this label.
_MODEL_PREFIX_PROVIDERS: tuple[tuple[str, str], ...] = (("deepseek", "deepseek"),)


def provider_label_from_model(model: str | None) -> str:
    """Infer a specific provider label from an OpenAI-compatible model id.

    Returns ``""`` when the model does not map to a known non-OpenAI provider,
    so callers can keep their default label (``"openai"``).
    """
    if not model:
        return ""
    name = model.strip().lower()
    # Strip a leading ``vendor/`` routing prefix (e.g. ``deepseek/deepseek-chat``).
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    for prefix, provider in _MODEL_PREFIX_PROVIDERS:
        if name.startswith(prefix):
            return provider
    return ""


def custom_base_passthrough_telemetry(method: str, path: str, base_url: str) -> tuple[str, str]:
    """Return passthrough telemetry metadata for narrow custom-base exceptions."""
    # OpenCode Zen sends provider-prefixed OpenAI-compatible traffic through
    # custom-base routing. Keep this exact to avoid labeling arbitrary
    # custom-base tool traffic as LLM provider telemetry.
    if method.upper() != "POST":
        return "", ""
    try:
        host = (urlparse(base_url.strip()).hostname or "").lower()
    except ValueError:
        return "", ""
    if host not in OPENCODE_ZEN_HOSTS:
        return "", ""
    normalized_path = path[1:] if path.startswith("/") else path
    if normalized_path == "zen/v1/chat/completions":
        return "chat/completions", "zen"
    return "", ""
