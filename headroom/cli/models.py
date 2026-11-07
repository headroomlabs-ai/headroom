"""`headroom models` — discover the models a harness can actually use, live.

Why this exists
---------------
An orchestrating agent asked to "review this with a couple of different models"
has no way to know what its account is served, so it names models from memory --
and memory is training data. Observed live in `proxy.log`: agents reached for
``claude-3.5-sonnet``, ``gemini-2.5-pro``, ``gpt-5``, ``glm-5.2`` and
``claude-sonnet-4-6``, none of which exist in the account's live catalog. Every
one is an upstream 400 and a wasted turn.

Handing the agent a list up front fixes it, but that shouldn't need a human.
This is the discoverable form: any agent with a shell tool can run it and get the
real, current set. Output stays terse and greppable for exactly that reason.

**Nothing here is hardcoded.** Every model comes from the provider's own
``/models`` endpoint at call time. When a provider cannot be enumerated (no
resolvable credential), this says so and explains the remedy -- it never
substitutes a baked-in list, because a stale list is what causes the failure
mode this command exists to remove.
"""

from __future__ import annotations

import json as jsonlib
from dataclasses import dataclass
from typing import Any

import click

from headroom.cli.main import main
from headroom.offline import OfflineEgressBlocked, guard_egress


@dataclass(frozen=True, slots=True)
class _Row:
    """One provider-agnostic catalog row for display."""

    id: str
    name: str
    vendor: str
    tier: str | None
    endpoints: tuple[str, ...]
    reasoning_efforts: tuple[str, ...]
    context_window: int | None
    max_output_tokens: int | None
    preview: bool
    provider: str


def _copilot_rows() -> tuple[list[_Row], str | None]:
    """Enumerate GitHub Copilot models. Returns (rows, unavailable_reason).

    Raises :class:`OfflineEgressBlocked` under ``HEADROOM_OFFLINE`` before a
    credential is resolved, since resolving one can exchange a token with GitHub.
    """
    from headroom.copilot_auth import copilot_api_url, resolve_subscription_bearer_token_details
    from headroom.models.copilot_catalog import parse_models_payload

    guard_egress("Copilot model listing", f"{copilot_api_url().rstrip('/')}/models")
    resolution = resolve_subscription_bearer_token_details()
    if resolution is None:
        return [], (
            "copilot: no token resolved. Run `headroom copilot-auth login`, or set "
            "GITHUB_COPILOT_TOKEN / GITHUB_COPILOT_API_TOKEN."
        )
    try:
        import httpx

        from headroom.copilot_auth import _copilot_chat_header_defaults

        response = httpx.get(
            f"{resolution.api_url.rstrip('/')}/models",
            headers={
                "Authorization": f"Bearer {resolution.token}",
                **_copilot_chat_header_defaults(),
            },
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001
        return [], f"copilot: could not reach {resolution.api_url}/models ({exc})."
    if response.status_code != 200:
        return [], (
            f"copilot: {resolution.api_url}/models returned HTTP {response.status_code}: "
            f"{response.text[:160]}"
        )

    rows = [
        _Row(
            id=c.id,
            name=c.display_name,
            vendor=c.vendor,
            tier=c.tier,
            endpoints=c.endpoints,
            reasoning_efforts=c.reasoning_efforts,
            context_window=c.context_window,
            max_output_tokens=c.max_output_tokens,
            preview=c.preview,
            provider="copilot",
        )
        for c in parse_models_payload(response.json()).values()
        if c.is_chat_model
    ]
    return rows, None


#: Upper bound on `/v1/models` pages followed (100 models each).
_ANTHROPIC_MAX_PAGES = 20


def _anthropic_rows() -> tuple[list[_Row], str | None]:
    """Enumerate Anthropic models via the official ``/v1/models`` endpoint.

    Anthropic exposes a real listing endpoint, so Claude Code's model set is
    discoverable the same way Copilot's is -- no hardcoded table. It does
    require an API credential: a Claude subscription's OAuth token lives in the
    OS keychain and is not readable here, so subscription-only users get a clear
    "not enumerable" message rather than a guessed list.

    Returns ``(rows, note)``. With no rows the note says why Anthropic is
    unavailable; with rows it says the list may be incomplete. Raises
    :class:`OfflineEgressBlocked` under ``HEADROOM_OFFLINE`` before any request.
    """
    import os

    # Before the credential check, so offline the reason given is the switch,
    # not a key that could not enable the listing anyway.
    base = (os.environ.get("ANTHROPIC_API_URL") or "https://api.anthropic.com").rstrip("/")
    guard_egress("Anthropic model listing", f"{base}/v1/models")
    key = (
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or ""
    ).strip()
    if not key:
        return [], (
            "anthropic: no credential in the environment. Set ANTHROPIC_API_KEY to enumerate "
            "Claude models. (A Claude subscription's OAuth token is held in the OS keychain and "
            "cannot be read from here; inside a wrapped session Claude Code's own /model picker "
            "already lists what the subscription allows.)"
        )
    # `/v1/models` is paginated (`has_more` + `last_id`, continued with
    # `after_id`). Reading only the first page silently truncated the list for
    # accounts with more than one page of models. The page cap only guards
    # against a misbehaving server that never stops paginating. Every early stop
    # while the server still says `has_more` comes back as a note, so a partial
    # list is never presented as the whole one.
    entries: list[Any] = []
    after_id: str | None = None
    incomplete: str | None = None
    for _page in range(_ANTHROPIC_MAX_PAGES):
        params: dict[str, Any] = {"limit": 100}
        if after_id:
            params["after_id"] = after_id
        try:
            import httpx

            response = httpx.get(
                f"{base}/v1/models",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                },
                params=params,
                timeout=15,
            )
        except Exception as exc:  # noqa: BLE001
            return [], f"anthropic: could not reach {base}/v1/models ({exc})."
        if response.status_code != 200:
            return [], (
                f"anthropic: {base}/v1/models returned HTTP {response.status_code}: "
                f"{response.text[:160]}"
            )

        payload = response.json()
        page = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(page, list):
            return [], "anthropic: /v1/models returned an unexpected body shape."
        entries.extend(page)
        if not payload.get("has_more"):
            break
        next_id = payload.get("last_id")
        if not isinstance(next_id, str) or not next_id or next_id == after_id:
            # Another request could only repeat this page, so stop, but say so.
            incomplete = (
                "anthropic: /v1/models said more models remain but gave no new cursor to "
                "continue from; the list above may be incomplete."
            )
            break
        after_id = next_id
    else:
        incomplete = (
            f"anthropic: stopped after {_ANTHROPIC_MAX_PAGES} pages of /v1/models; "
            "the list above may be incomplete."
        )

    rows: list[_Row] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        rows.append(
            _Row(
                id=model_id,
                name=display_name
                if isinstance(display_name := entry.get("display_name"), str)
                else "",
                vendor="Anthropic",
                tier=None,
                endpoints=("/v1/messages",),
                reasoning_efforts=(),
                context_window=None,
                max_output_tokens=None,
                preview=False,
                provider="anthropic",
            )
        )
    return rows, incomplete


@main.command("models")
@click.option(
    "--provider",
    type=click.Choice(["auto", "copilot", "anthropic", "all"]),
    default="auto",
    show_default=True,
    help="Which harness to enumerate. 'auto' tries every provider with a usable credential.",
)
@click.option("--vendor", default=None, help="Filter by vendor (substring, case-insensitive)")
@click.option(
    "--tier",
    type=click.Choice(["powerful", "versatile", "lightweight"]),
    default=None,
    help="Filter by capability tier as the provider classifies it (Copilot only)",
)
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output")
def models(provider: str, vendor: str | None, tier: str | None, as_json: bool) -> None:
    """List models available right now, straight from the provider.

    \b
    Examples:
        headroom models                       # every provider with a credential
        headroom models --provider copilot    # Copilot CLI's set
        headroom models --tier powerful       # the high-capability ones
        headroom models --json                # for scripts and agents

    \b
    IDs printed here are exactly what `--model` and a subagent's model field
    accept. Nothing is hardcoded: if a provider cannot be reached, it is
    reported as unavailable rather than replaced with a stale built-in list.
    Under HEADROOM_OFFLINE no provider is asked, and each is reported as
    unavailable for that reason.
    """
    wanted = ["copilot", "anthropic"] if provider in ("auto", "all") else [provider]
    rows: list[_Row] = []
    notes: list[str] = []
    # A note that comes with rows means the provider was enumerated but stopped
    # early, which is not the same as being unavailable.
    incomplete: list[str] = []
    for name in wanted:
        try:
            got, reason = _copilot_rows() if name == "copilot" else _anthropic_rows()
        except OfflineEgressBlocked as blocked:
            # A listing is optional, so the air-gap refusal becomes the reason
            # this provider is missing rather than an error for the command.
            got, reason = [], f"{name}: {blocked}"
        rows.extend(got)
        if reason:
            (incomplete if got else notes).append(reason)

    if vendor:
        needle = vendor.strip().lower()
        rows = [r for r in rows if needle in r.vendor.lower()]
    if tier:
        rows = [r for r in rows if r.tier == tier]
    rows.sort(key=lambda r: (r.provider, r.vendor.lower(), r.id))

    if as_json:
        click.echo(
            jsonlib.dumps(
                {
                    "models": [
                        {
                            "id": r.id,
                            "name": r.name,
                            "vendor": r.vendor,
                            "tier": r.tier,
                            "provider": r.provider,
                            "endpoints": list(r.endpoints),
                            "reasoning_efforts": list(r.reasoning_efforts),
                            "context_window": r.context_window,
                            "max_output_tokens": r.max_output_tokens,
                            "preview": r.preview,
                        }
                        for r in rows
                    ],
                    "unavailable": notes,
                    "incomplete": incomplete,
                },
                indent=2,
            )
        )
        return

    if rows:
        id_w = max(len(r.id) for r in rows)
        ven_w = max(max(len(r.vendor) for r in rows), 6)
        click.echo(f"{'MODEL ID':<{id_w}}  {'VENDOR':<{ven_w}}  {'TIER':<12} CONTEXT  VIA")
        click.echo("-" * (id_w + ven_w + 34))
        for r in rows:
            context = f"{r.context_window // 1000}k" if r.context_window else "-"
            preview = " (preview)" if r.preview else ""
            click.echo(
                f"{r.id:<{id_w}}  {r.vendor:<{ven_w}}  {(r.tier or '-'):<12} "
                f"{context:<8} {r.provider}{preview}"
            )
        click.echo()
        click.echo(
            f"{len(rows)} model(s). Use an ID above verbatim as --model or a subagent model."
        )
    else:
        click.echo("No models could be enumerated.")

    for note in incomplete:
        click.echo(f"\n  incomplete — {note}")
    for note in notes:
        click.echo(f"\n  not enumerated — {note}")
