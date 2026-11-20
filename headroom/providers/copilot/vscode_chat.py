"""Route VS Code Copilot **Chat** through Headroom via the Custom Endpoint provider.

Why this exists alongside ``vscode.py``
---------------------------------------
``vscode.py`` writes ``github.copilot.advanced.debug.overrideProxyUrl``. That knob
targets Copilot's **completions** endpoint; chat talks to the CAPI endpoint, whose
knob is ``overrideCapiUrl`` -- so the existing wrapper never actually redirected
chat traffic. Both are undocumented debug settings, and the Chat extension has
been observed ignoring them outright
(microsoft/vscode-copilot-release#7802, closed unfixed, repo archived).

VS Code ships a supported alternative: the **Custom Endpoint** BYOK provider
(``vendor: "customendpoint"``), configured through ``chatLanguageModels.json``.
It speaks ``chat-completions``, ``responses`` and ``messages`` -- all three of
which the Headroom proxy already serves -- so the proxy can simply *be* the
endpoint, with no protocol work.

What this buys, and what it does not
------------------------------------
Chat and agent traffic flows through Headroom and is compressed, and **every**
model the account is entitled to appears in the picker, so a user can switch
models mid-session exactly as they do natively.

It does **not** cover inline (ghost-text) completions, semantic search, or
embeddings: VS Code routes those through GitHub regardless of BYOK. Callers must
say so rather than implying full coverage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from headroom import fsutil
from headroom.providers.copilot.vscode import (
    _read_settings,
    _validate_settings,
    vscode_user_dir,
)
from headroom.providers.copilot.wrap import VSCODE_MODEL_ID_PREFIX
from headroom.proxy.project_context import with_project_prefix

#: Provider display name. Also how a human recognises the block in the file.
HEADROOM_PROVIDER_NAME = "Headroom (GitHub Copilot)"
#: VS Code's provider id for a user-supplied HTTP endpoint.
CUSTOM_ENDPOINT_VENDOR = "customendpoint"
#: 1.132+ hides Custom Endpoint models unless this is on. Defaults to false, is
#: experiment-controlled, and is not exposed in the Settings UI, so a user who
#: upgrades silently loses every configured model (microsoft/vscode#329545).
BYOK_ENABLED_SETTING = "chat.agentHost.byokModels.enabled"

#: Endpoint suffix per API shape. VS Code resolves a model's URL from its
#: ``apiType`` when the path is omitted; we always write it explicitly so the
#: mapping is visible in the file rather than inferred.
#:
#: Only the two wires ``_api_type_for`` can actually choose are listed. Carrying
#: a ``messages`` entry as well would advertise support for a path this module
#: never emits, so a reader could not tell which mappings are live.
_API_TYPE_PATHS = {
    "chat-completions": "/v1/chat/completions",
    "responses": "/v1/responses",
}


def chat_models_path(
    *, platform: str | None = None, environ: Mapping[str, str] | None = None
) -> Path:
    """Location of VS Code's ``chatLanguageModels.json`` for the default profile."""
    return vscode_user_dir(platform=platform, environ=environ) / "chatLanguageModels.json"


def byok_entitlement_enabled(token: str) -> bool | None:
    """Whether this seat may use BYOK at all, from the Copilot token's claims.

    VS Code gates the entire Custom Endpoint feature on the ``client_byok``
    entitlement, and a Business/Enterprise admin can disable it org-wide. When it
    is off, configuration written here would simply never appear in the picker --
    so it is worth failing loudly at launch instead of leaving the user hunting
    for models that cannot exist.

    Returns ``None`` when the claim is absent (older token shapes): unknown, so
    the caller should proceed with a note rather than refuse.
    """
    body = token.split(":", 1)[0]
    claims = dict(pair.split("=", 1) for pair in body.split(";") if "=" in pair)
    raw = claims.get("client_byok")
    if raw is None:
        return None
    return raw.strip() == "1"


def _api_type_for(endpoints: list[str] | tuple[str, ...]) -> str:
    """Pick the wire VS Code should use for a model.

    Prefers ``chat-completions``: it is the shape verified working end-to-end
    through the proxy with the placeholder key VS Code sends, and Copilot serves
    most families on it. ``responses`` is used for models served only there
    (``gpt-5.6-*``, ``mai-code-1-flash-picker``).

    ``messages`` is deliberately never chosen, but not because it cannot work:
    review found that the streaming ``/v1/messages`` path VS Code actually uses
    returns 200 (only the non-streaming path 400s, and independently of the key).
    It is avoided because every Claude model is also served on
    ``chat-completions``, which is the wire verified end-to-end here -- one wire
    is one less thing to keep working, and nothing is lost by preferring it.
    """
    if not endpoints:
        return "chat-completions"
    if "/chat/completions" in endpoints:
        return "chat-completions"
    if "/responses" in endpoints:
        return "responses"
    return "chat-completions"


def build_model_entries(payload: Any, base_url: str) -> list[dict[str, Any]]:
    """Translate a Copilot ``/models`` payload into VS Code model entries.

    Capabilities are read from the **raw payload**, not from
    ``headroom.models.copilot_catalog.ModelCard`` -- that dataclass carries no
    tool-calling or vision fields, and it belongs to a separate change that must
    not be extended from here.

    ``toolCalling`` matters more than it looks: VS Code hides a model from agent
    mode entirely when it is false, so hardcoding ``true`` would surface models
    that then fail, and hardcoding ``false`` would silently remove good ones.
    """
    entries: list[dict[str, Any]] = []
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return entries

    for model in data:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        capabilities = model.get("capabilities")
        capabilities = capabilities if isinstance(capabilities, dict) else {}
        if capabilities.get("type") != "chat":
            continue  # embeddings and similar must never reach the chat picker
        policy = model.get("policy")
        policy_ok = not isinstance(policy, dict) or policy.get("state") in (None, "enabled")
        if not (model.get("model_picker_enabled") and policy_ok):
            continue

        supports = capabilities.get("supports")
        supports = supports if isinstance(supports, dict) else {}
        limits = capabilities.get("limits")
        limits = limits if isinstance(limits, dict) else {}

        api_type = _api_type_for(model.get("supported_endpoints") or [])
        display = model.get("name") if isinstance(model.get("name"), str) else model_id
        entry: dict[str, Any] = {
            # Prefixed so VS Code's picker keeps this distinct from Copilot's
            # native entry for the same model; the proxy strips it back off
            # before the request leaves for Copilot. Without the prefix, every
            # recently-used or GitHub-featured model silently lost its Headroom
            # twin -- exactly the models a user reaches for most.
            "id": f"{VSCODE_MODEL_ID_PREFIX}{model_id}",
            "name": f"{display} (Headroom)",
            "url": f"{base_url.rstrip('/')}{_API_TYPE_PATHS[api_type]}",
            "apiType": api_type,
            "toolCalling": bool(supports.get("tool_calls")),
            "vision": bool(supports.get("vision")),
        }
        if api_type == "responses":
            # VS Code's BYOK client sets `store: !zeroDataRetentionEnabled` on
            # every /responses request, and Copilot's API rejects the field with
            # `400 store is not supported` -- so without this flag every
            # responses-served model fails on first use. Declaring ZDR also
            # suppresses `previous_response_id`, which Copilot would not honour
            # either. The proxy forces `store: false` as well, so a stale config
            # written before this fix still works.
            entry["zeroDataRetentionEnabled"] = True
        # Required by the provider schema, so always emit them: a model missing
        # either field yields NaN token limits rather than a visible error.
        max_input = limits.get("max_prompt_tokens") or limits.get("max_context_window_tokens")
        entry["maxInputTokens"] = (
            max_input if isinstance(max_input, int) and max_input > 0 else 128000
        )
        max_output = limits.get("max_output_tokens")
        entry["maxOutputTokens"] = (
            max_output if isinstance(max_output, int) and max_output > 0 else 4096
        )
        context_window = limits.get("max_context_window_tokens")
        if isinstance(context_window, int) and context_window > 0:
            # Written explicitly rather than left to VS Code's
            # `maxInputTokens + maxOutputTokens` derivation, which is wrong
            # whenever the provider publishes a real window (e.g. gpt-5-mini).
            entry["contextWindow"] = context_window
        entries.append(entry)

    entries.sort(key=lambda e: e["name"].lower())
    return entries


#: Placeholder API key. The proxy substitutes the real Copilot credential itself,
#: so whatever VS Code sends here is never read upstream.
#:
#: A plain literal is used rather than ``${input:...}`` because that syntax is not
#: a user prompt -- it is VS Code's pointer into its own secret storage, which
#: only its *Manage Language Models* UI ever writes. An unresolved pointer yields
#: an empty key, so the value on the wire is the same either way; the literal is
#: simply honest about being inert instead of implying a secret exists.
PLACEHOLDER_API_KEY = "headroom-local-unused"


def build_provider_block(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """The single provider object Headroom owns in ``chatLanguageModels.json``.

    Deliberately without ``apiKey``: the inert placeholder is attached only to
    the copy that is serialized to disk (``_on_disk``). That keeps the block the
    ownership digest is computed from free of any credential-named value -- the
    placeholder carries no identity, and a value named like a key has no
    business flowing into a hash.
    """
    return {
        "name": HEADROOM_PROVIDER_NAME,
        "vendor": CUSTOM_ENDPOINT_VENDOR,
        "models": entries,
    }


def _on_disk(block: Mapping[str, Any]) -> dict[str, Any]:
    """The block as written to ``chatLanguageModels.json``, placeholder key included."""
    return {**block, "apiKey": PLACEHOLDER_API_KEY}


def _provenance_path(path: Path) -> Path:
    from headroom import paths

    digest = hashlib.sha256(str(path.resolve()).encode("utf-8", "replace")).hexdigest()[:16]
    return paths.workspace_dir() / "vscode_chat_models" / f"{digest}.json"


#: Ownership record layout. v1 (``{"path", "sha256"}``) held a single digest of
#: the whole block; v2 holds the committed digest plus a pending one, so a
#: refresh can be interrupted at any point without losing proof (see
#: ``configure_chat_models``).
_PROVENANCE_VERSION = 2


@dataclass(frozen=True)
class _Provenance:
    """What the ownership record proves about the Headroom entry on disk.

    ``committed`` is the digest of the block believed to be in the file;
    ``pending`` is the digest of a block whose write may or may not have landed.
    Either one proves ownership: a crash between the record write and the config
    write leaves the old block (``committed``), a crash after the config write
    leaves the new one (``pending``). ``legacy`` is a v1 whole-block digest,
    honoured so records written by earlier builds are not stranded, and replaced
    by a v2 record on the next write.
    """

    committed: str | None
    pending: str | None
    legacy: str | None = None

    def proves(self, entry: Mapping[str, Any]) -> bool:
        digest = _block_digest(entry)
        if digest in {self.committed, self.pending} - {None}:
            return True
        return self.legacy is not None and _legacy_block_digest(entry) == self.legacy


def _write_provenance(path: Path, *, committed: str | None, pending: str | None) -> None:
    """Write the ownership record atomically, or refuse with a clear error.

    Ownership is deliberately **not** inferred from the file's contents (a user
    may legitimately hand-edit or copy our provider entry, and a name match alone
    would then license overwriting their edits), so the recorded digest is the
    only proof a later run ever accepts. A record that cannot be written must
    therefore stop the config write before it happens: otherwise the file would
    hold a Headroom block no later run can prove it owns, stranding both refresh
    and removal until a hand edit.
    """
    try:
        record = _provenance_path(path)
        record.parent.mkdir(parents=True, exist_ok=True)
        fsutil.write_text(
            record,
            json.dumps(
                {
                    "version": _PROVENANCE_VERSION,
                    "path": str(path),
                    "committed": committed,
                    "pending": pending,
                }
            ),
        )
    except OSError as exc:
        raise click.ClickException(
            f"Could not create an ownership record for the Headroom provider entry "
            f"bound for {path} ({exc}). The entry was not written."
        ) from exc


def _read_provenance(path: Path) -> _Provenance | None:
    try:
        rec = json.loads(fsutil.read_text(_provenance_path(path)))
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict):
        return None

    def _digest(key: str) -> str | None:
        value = rec.get(key)
        return value if isinstance(value, str) and value else None

    if rec.get("version") == _PROVENANCE_VERSION:
        prov = _Provenance(committed=_digest("committed"), pending=_digest("pending"))
    else:
        prov = _Provenance(committed=None, pending=None, legacy=_digest("sha256"))
    if prov.committed is None and prov.pending is None and prov.legacy is None:
        return None
    return prov


def _clear_provenance(path: Path) -> None:
    try:
        _provenance_path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _digest_view(block: Mapping[str, Any]) -> dict[str, Any]:
    # ``apiKey`` is always the inert ``PLACEHOLDER_API_KEY`` and carries no
    # identity, so it is not part of what makes a block Headroom's. Entries read
    # back from disk carry it; blocks built in memory never do (see
    # ``build_provider_block``), so both reduce to the same view.
    return {
        "name": block.get("name"),
        "vendor": block.get("vendor"),
        "models": block.get("models"),
    }


def _block_digest(block: Mapping[str, Any]) -> str:
    """Content-equality digest identifying a Headroom provider block."""
    return hashlib.sha256(
        json.dumps(_digest_view(block), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _legacy_block_digest(entry: Mapping[str, Any]) -> str:
    """The v1 whole-block digest, only ever computed over entries read from disk."""
    return hashlib.sha256(
        json.dumps(entry, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _load_providers(path: Path) -> list[Any]:
    """Parse the existing provider list, or return ``[]`` for a fresh file."""
    if not path.exists():
        return []
    try:
        raw = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise click.ClickException(
            f"Could not read {path} as UTF-8 ({exc}); refusing to rewrite it."
        ) from exc
    if not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"Could not parse {path}: {exc}. Headroom did not overwrite it."
        ) from exc
    if not isinstance(parsed, list):
        raise click.ClickException(
            f"{path} must contain a JSON array of providers; refusing to overwrite it."
        )
    return parsed


def configure_chat_models(path: Path, block: dict[str, Any]) -> str:
    """Write/refresh Headroom's provider entry, preserving every other provider.

    Returns ``"added"`` or ``"updated"``. Raises rather than clobbering a
    same-named entry Headroom cannot prove it wrote.

    The update is a three-step transaction, each step individually atomic:

    1. record ``{committed: <block on disk>, pending: <new block>}``;
    2. write the config;
    3. finalize the record to ``{committed: <new block>, pending: None}``.

    Refresh and removal accept either digest, so an interruption between any two
    steps (an exception, a kill, a power loss) leaves the entry that is actually
    on disk provable. An interrupted *initial* add leaves a pending digest that
    matches nothing, which is harmless: the next run adds normally.
    """
    providers = _load_providers(path)
    prov = _read_provenance(path)

    owned_indexes = [
        index
        for index, entry in enumerate(providers)
        if isinstance(entry, dict) and prov is not None and prov.proves(entry)
    ]
    if len(owned_indexes) > 1:
        raise click.ClickException(
            f"{path} contains more than one provider entry matching Headroom's "
            "ownership record (for example a copy of Headroom's entry); remove the "
            "extra one so Headroom can tell which to update."
        )

    committed_before: str | None
    written = _on_disk(block)
    if owned_indexes:
        # Recomputed from disk rather than copied from the record, so a record
        # left pending (or in the v1 format) is normalized by this write.
        committed_before = _block_digest(providers[owned_indexes[0]])
        providers[owned_indexes[0]] = written
        action = "updated"
    else:
        conflicting = [
            entry
            for entry in providers
            if isinstance(entry, dict) and entry.get("name") == HEADROOM_PROVIDER_NAME
        ]
        if conflicting:
            raise click.ClickException(
                f"{path} already has a provider named {HEADROOM_PROVIDER_NAME!r} that "
                "Headroom did not write; refusing to replace it. Rename or remove it, "
                "or pass --no-configure."
            )
        providers.append(written)
        committed_before = None
        action = "added"

    proposed = _block_digest(block)

    # Step 1. Raises before the user's file is touched if no record can be made.
    _write_provenance(path, committed=committed_before, pending=proposed)

    # Step 2.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fsutil.write_text(path, json.dumps(providers, indent=2, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Best effort: drop the pending digest. If this fails too, the pending
        # record left behind still proves the (unchanged) entry on disk.
        if committed_before is None:
            _clear_provenance(path)
        else:
            try:
                _write_provenance(path, committed=committed_before, pending=None)
            except click.ClickException:
                pass
        raise click.ClickException(f"Could not write {path} ({exc}). Nothing was changed.") from exc

    # Step 3. Not fatal: the config is committed and the pending digest already
    # proves it; the next write finalizes the record.
    try:
        _write_provenance(path, committed=proposed, pending=None)
    except click.ClickException:
        pass

    return action


def remove_chat_models(path: Path) -> bool:
    """Drop Headroom's provider entry, leaving every other provider untouched."""
    if not path.exists():
        return False
    try:
        providers = _load_providers(path)
    except click.ClickException:
        return False
    prov = _read_provenance(path)
    if prov is None:
        return False
    remaining = [
        entry for entry in providers if not (isinstance(entry, dict) and prov.proves(entry))
    ]
    if len(remaining) == len(providers):
        # A record whose digests match nothing on disk proves nothing (e.g. an
        # interrupted initial add); dropping it keeps the next run clean.
        _clear_provenance(path)
        return False
    fsutil.write_text(path, json.dumps(remaining, indent=2, ensure_ascii=False) + "\n")
    _clear_provenance(path)
    return True


_BYOK_MARKER_START = "// --- Headroom VS Code chat models ---"
_BYOK_MARKER_END = "// --- end Headroom VS Code chat models ---"

#: Redirects the Copilot Chat extension's whole CAPI surface at a chosen base
#: URL. Every endpoint the extension uses is derived from it --
#: ``{base}/chat/completions``, ``/responses``, ``/v1/messages``, ``/models``,
#: ``/models/session``, ``/embeddings`` -- which is the same surface
#: ``COPILOT_API_URL`` redirects for the CLI.
#:
#: This is what makes VS Code behave like the CLI's ``--native`` mode: Copilot's
#: own models flow through Headroom, so a model chosen by the *agent* -- a
#: subagent, or auto model selection -- is compressed too. BYOK could never do
#: that: it adds a parallel provider the agent does not pick from, so a subagent
#: silently ran on Copilot's uncompressed endpoint.
#:
#: Coverage is broad but **not** total. Two paths resolve a different base and
#: are unaffected, so callers must not claim everything is captured:
#:
#: * The *execution* subagent, when the ``ExecutionSubagentUseAgenticProxy``
#:   experiment is on and the terminal is not PowerShell, uses
#:   ``proxyBaseURL`` (``copilot-proxy.githubusercontent.com``), which only
#:   ``advanced.debug.overrideProxyUrl`` moves -- the *completions* knob, which
#:   this module deliberately does not write. On PowerShell the same code takes
#:   the ordinary chat endpoint and is routed.
#: * The agent host (``chat.agentHost.enabled``) spawns the Copilot CLI binary
#:   and reads ``COPILOT_API_URL``; nothing here sets it for that process.
#:
#: It is an undocumented debug setting and user-scope only: written into a
#: workspace ``settings.json`` it is ignored. Verified end to end against the
#: shipped Copilot Chat extension on VS Code 1.132 -- ``/models``,
#: ``/models/session`` and ``/chat/completions`` all arrived at the proxy and
#: compressed. A newer ``internal.capiUrl`` key exists in the bundle but is
#: never read today; if chat adopts the completions-side resolver ordering, the
#: new key would win and this setting would stop taking effect silently.
CAPI_OVERRIDE_SETTING = "github.copilot.advanced.debug.overrideCapiUrl"

_CAPI_MARKER_START = "// --- Headroom Copilot Chat routing ---"
_CAPI_MARKER_END = "// --- end Headroom Copilot Chat routing ---"


def _mask_quoted_regions(raw: str) -> str:
    """Same-length copy with JSON string bodies and block comments blanked out.

    Marker text only counts where a marker can actually live -- a line comment.
    A user who stores our marker *inside a setting's value* (``"myext.header":
    "// --- Headroom ... ---"``) has written data, not a block, and treating it
    as one deletes every setting between the two quotes. Offsets are preserved
    so spans found here index straight back into the original text.
    """
    out: list[str] = []
    i, n = 0, len(raw)
    blank = lambda text: "".join("\n" if ch == "\n" else " " for ch in text)  # noqa: E731
    while i < n:
        ch = raw[i]
        if ch == '"':
            out.append('"')
            i += 1
            while i < n:
                if raw[i] == "\\" and i + 1 < n:
                    out.append("  ")
                    i += 2
                    continue
                if raw[i] == '"':
                    out.append('"')
                    i += 1
                    break
                out.append("\n" if raw[i] == "\n" else " ")
                i += 1
            continue
        if raw.startswith("/*", i):
            close = raw.find("*/", i + 2)
            close = n if close < 0 else close + 2
            out.append(blank(raw[i:close]))
            i = close
            continue
        if raw.startswith("//", i):
            # Line comments are kept verbatim: this is where our markers live.
            nl = raw.find("\n", i)
            nl = n if nl < 0 else nl
            out.append(raw[i:nl])
            i = nl
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _settings_block_span(raw: str, start: str, end: str) -> tuple[int, int] | None:
    """Line span of one intact marker pair, or ``None``.

    ``None`` covers every shape we must not touch: no markers, a lone start or
    end (a half-applied Settings Sync merge leaves exactly that), markers in the
    wrong order, more than one pair, or marker text that is really string data.
    """
    masked = _mask_quoted_regions(raw)
    if masked.count(start) != 1 or masked.count(end) != 1:
        return None
    start_at = masked.find(start)
    end_at = masked.find(end)
    if end_at < start_at:
        return None
    line_start = raw.rfind("\n", 0, start_at) + 1
    line_end = raw.find("\n", end_at)
    return line_start, (len(raw) if line_end < 0 else line_end + 1)


def _settings_block_text(raw: str, start: str, end: str) -> str | None:
    span = _settings_block_span(raw, start, end)
    return None if span is None else raw[span[0] : span[1]]


def _settings_provenance_path(path: Path, start: str) -> Path:
    from headroom import paths

    key = hashlib.sha256(f"{path.resolve()}|{start}".encode("utf-8", "replace")).hexdigest()[:16]
    return paths.workspace_dir() / "vscode_settings_blocks" / f"{key}.json"


def _record_settings_block(path: Path, start: str, block_text: str | None) -> None:
    """Remember the exact bytes we wrote, so a later run can prove ownership.

    Ownership is deliberately not inferred from the marker text. A user can
    legitimately copy a marker -- Settings Sync merges do it, and so does anyone
    pasting a config snippet -- and a text match alone would then license
    rewriting or deleting their settings.

    Best-effort: this is for a block already on disk, either clearing its record
    (``block_text`` is ``None``, run after the block itself is gone) or
    resyncing a *sibling* block's digest after an edit shifted its bytes. Either
    way the file mutation this call is bookkeeping for has already happened and
    cannot be undone here, so losing the record only downgrades that block to
    "not ours" on the next run -- worse than keeping the record, but not a new
    failure to surface. A *new* block's record is written by
    ``_require_settings_provenance`` instead, before the block is written, so a
    failure there can still refuse the write.
    """
    try:
        record = _settings_provenance_path(path, start)
        if block_text is None:
            record.unlink(missing_ok=True)
            return
        record.parent.mkdir(parents=True, exist_ok=True)
        fsutil.write_text(
            record,
            json.dumps(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(block_text.encode("utf-8")).hexdigest(),
                }
            ),
        )
    except (OSError, RuntimeError, ValueError):
        pass


def _require_settings_provenance(path: Path, start: str, block_text: str) -> None:
    """Record a *new* block's bytes, or refuse to write it at all.

    Called before the block is written to ``settings.json``, not after: with no
    record, ``_owns_settings_block`` can never adopt a pair later (adoption was
    the bug -- see its docstring), so writing a block whose record we failed to
    create would produce exactly the stuck state that removal exists to avoid --
    a Headroom-authored block nothing can ever prove is Headroom-authored. This
    can raise when ``HOME``/``USERPROFILE`` are both unset or the workspace
    directory is not writable; either way, refusing here means nothing has
    touched the user's file yet.
    """
    try:
        record = _settings_provenance_path(path, start)
        record.parent.mkdir(parents=True, exist_ok=True)
        fsutil.write_text(
            record,
            json.dumps(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(block_text.encode("utf-8")).hexdigest(),
                }
            ),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(
            f"Could not create an ownership record for the Headroom block bound for "
            f"{path} ({exc}). The new block was not written; a Headroom-owned block this "
            "call was replacing, if any, has already been removed. An unrecorded block "
            "could never be proven ours later and would need hand-editing to remove."
        ) from exc


def _owns_settings_block(path: Path, raw: str, start: str, end: str) -> bool:
    """True only when this exact block is one we wrote and nobody has edited.

    The recorded digest is the *only* proof accepted, and it must exist: with no
    record, a structurally intact pair is indistinguishable from one a user wrote
    or copied by hand -- a pair holding only ``github.copilot.advanced.debug.
    overrideCapiUrl`` is exactly the shape our own writer produces, so a body
    check cannot tell them apart. Adopting it anyway is how ``enable`` came to
    rewrite a user's own gateway and ``disable`` came to delete their setting.

    This also means a block written before provenance tracking existed (no
    record on disk) can never be claimed retroactively -- there is no way to
    tell that case apart from a user's own pair either. Migrating such a block
    would need an explicit user-confirmed adoption step or independent evidence
    that we wrote it; absent that, the safe answer is to leave the bytes alone
    and report "not ours".
    """
    block = _settings_block_text(raw, start, end)
    if block is None:
        return False
    digest = hashlib.sha256(block.encode("utf-8")).hexdigest()

    try:
        record = json.loads(fsutil.read_text(_settings_provenance_path(path, start)))
    except (OSError, ValueError):
        return False
    recorded = record.get("sha256") if isinstance(record, dict) else None
    if not isinstance(recorded, str):
        return False
    return recorded == digest


def _append_settings_block(path: Path, body_lines: list[str], start: str, end: str) -> str:
    """Append a marker-delimited block of settings, preserving everything else.

    Returns ``"added"``, or ``"already set"`` when either marker is already
    present -- the caller is responsible for having proved ownership first.
    Raises rather than editing a file this parser cannot validate, or one whose
    ownership record cannot be created (see ``_require_settings_provenance``);
    in both cases nothing is written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _read_settings(path) if path.exists() else "{}\n"
    _validate_settings(raw, path)
    if start in raw or end in raw:
        return "already set"

    close = raw.rfind("}")
    if close < 0:
        raise click.ClickException(f"Could not locate the root object in {path}.")
    before = raw[:close].rstrip()
    after = raw[close:]
    from headroom.providers.copilot.vscode import _strip_jsonc_comments

    inner = _strip_jsonc_comments(before).rstrip()
    separator = "" if inner.endswith("{") or inner.endswith(",") else ","
    line_sep = "\r\n" if "\r\n" in raw else "\n"
    # The separator goes on the first *setting* line, not after `before`.
    # `before` may end with a `//` comment -- our own end-marker does, once a
    # second block is appended -- and a comment runs to end of line, so a comma
    # placed there is swallowed and the file becomes invalid JSON.
    body = list(body_lines)
    if separator and body:
        body[0] = f"{separator}{body[0]}"
    block = line_sep.join([f"\t{start}", *(f"\t{line}" for line in body), f"\t{end}"])
    updated = before + line_sep + block + line_sep + after
    _validate_settings(updated, path)
    new_block = _settings_block_text(updated, start, end)
    assert new_block is not None  # the pair was just constructed above
    _require_settings_provenance(path, start, new_block)
    fsutil.write_text(path, updated)
    return "added"


def enable_capi_override(path: Path, base_url: str) -> str:
    """Point Copilot Chat's CAPI at the proxy, so every model is compressed.

    Returns ``"added"``, ``"updated"``, ``"already set"``, or
    ``"already set by user"`` when the user has their own value -- which is never
    overwritten, because someone pointing Copilot Chat at their own gateway means
    it deliberately.

    Idempotent on the **value**, not merely on the marker. The block outlives the
    session that wrote it (nothing withdraws it on exit), while the URL it
    carries can change between runs: the proxy may land on a different port, and
    the ``/p/<project>`` prefix follows the launch directory. Matching on the
    marker alone left a stale URL in place while the caller reported the new one
    -- so chat pointed at a dead port, or attributed every saving to the wrong
    project, and the terminal said otherwise.

    Ownership is proved out of band, never from the marker text. ``"not ours"``
    means marker text is present that we cannot prove we wrote -- a lone marker
    from a half-applied Settings Sync merge, a user-authored pair, or a block
    someone has since edited. Nothing is written in that case, and the caller
    must not report success: the previous value stays live.
    """
    raw = _read_settings(path) if path.exists() else ""
    has_marker = _CAPI_MARKER_START in raw or _CAPI_MARKER_END in raw
    if not has_marker:
        if CAPI_OVERRIDE_SETTING in raw:
            return "already set by user"
        return _append_settings_block(
            path,
            [f"{json.dumps(CAPI_OVERRIDE_SETTING)}: {json.dumps(base_url)}"],
            _CAPI_MARKER_START,
            _CAPI_MARKER_END,
        )

    if not _owns_settings_block(path, raw, _CAPI_MARKER_START, _CAPI_MARKER_END):
        return "not ours"

    body = [f"{json.dumps(CAPI_OVERRIDE_SETTING)}: {json.dumps(base_url)}"]
    block = _settings_block_text(raw, _CAPI_MARKER_START, _CAPI_MARKER_END) or ""
    if any(line.strip().lstrip(",") == body[0] for line in block.splitlines()):
        return "already set"

    # Owned and stale: replace it. Both halves are checked, so a refusal cannot
    # be mistaken for a rewrite.
    if not _remove_settings_block(path, _CAPI_MARKER_START, _CAPI_MARKER_END):
        return "not ours"
    if _append_settings_block(path, body, _CAPI_MARKER_START, _CAPI_MARKER_END) != "added":
        return "not ours"
    return "updated"


def disable_capi_override(path: Path) -> bool:
    """Remove only the CAPI routing block this module added."""
    return _remove_settings_block(path, _CAPI_MARKER_START, _CAPI_MARKER_END)


def enable_byok_setting(path: Path) -> str:
    """Turn on ``chat.agentHost.byokModels.enabled`` in ``settings.json``.

    VS Code 1.132 hides Custom Endpoint models without it, and the setting is not
    surfaced in the Settings UI -- so a user who upgrades sees their models vanish
    with no explanation. Writing it is necessary but not sufficient: the agent
    host process must restart before the models reappear.

    Returns ``"not ours"`` when marker text is present that we cannot prove we
    wrote; nothing is written, and the caller must not report success.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _read_settings(path) if path.exists() else "{}\n"
    _validate_settings(raw, path)

    if _BYOK_MARKER_START in raw or _BYOK_MARKER_END in raw:
        if not _owns_settings_block(path, raw, _BYOK_MARKER_START, _BYOK_MARKER_END):
            return "not ours"
        return "already set"
    if BYOK_ENABLED_SETTING in raw:
        # Never overwrite the user's own value -- but `false` is the one value
        # that silently produces zero visible models, and it is also the default,
        # so passing it off as "already configured" would send someone hunting for
        # models that cannot appear. Distinguish it so the caller can warn.
        from headroom.providers.copilot.vscode import _strip_jsonc_comments

        try:
            import re as _re

            parsed = json.loads(_re.sub(r",\s*([}\]])", r"\1", _strip_jsonc_comments(raw)))
            if isinstance(parsed, dict) and parsed.get(BYOK_ENABLED_SETTING) is False:
                return "set to false by user"
        except (ValueError, TypeError):
            pass
        return "already set by user"

    # Shares the writer with the routing block so both get the same
    # separator handling; they are routinely written into the same file.
    return _append_settings_block(
        path,
        [f"{json.dumps(BYOK_ENABLED_SETTING)}: true"],
        _BYOK_MARKER_START,
        _BYOK_MARKER_END,
    )


def disable_byok_setting(path: Path) -> bool:
    """Remove only the marker block this module added to ``settings.json``."""
    return _remove_settings_block(path, _BYOK_MARKER_START, _BYOK_MARKER_END)


def _drop_orphan_leading_comma(text: str) -> str:
    """Delete a separator comma left stranded as the object's first token.

    Removing the *first* of two appended blocks promotes the second block's
    leading comma to the front of the root object, which is invalid JSON. The
    scan skips whitespace and JSONC comments so the comma is found even though
    our own end-marker comment sits between it and the brace.
    """
    brace = text.find("{")
    if brace < 0:
        return text
    i = brace + 1
    while i < len(text):
        if text[i].isspace():
            i += 1
        elif text.startswith("//", i):
            nl = text.find("\n", i)
            i = len(text) if nl < 0 else nl + 1
        elif text.startswith("/*", i):
            close = text.find("*/", i)
            i = len(text) if close < 0 else close + 2
        elif text[i] == ",":
            return text[:i] + text[i + 1 :]
        else:
            return text
    return text


def _remove_settings_block(path: Path, start_marker: str, end_marker: str) -> bool:
    """Cut out one marker block **we can prove we wrote**, and nothing else.

    Returns False when there is nothing of ours to remove. Raises when marker
    text is present that we cannot claim -- deleting between a marker pair we
    did not write removes whatever the user happened to put there, and the text
    alone is no proof: Settings Sync merges copy markers, and so does anyone
    pasting a config snippet.
    """
    if not path.exists():
        return False
    raw = _read_settings(path)
    if start_marker not in raw and end_marker not in raw:
        return False
    if raw.count(start_marker) > 1 or raw.count(end_marker) > 1:
        # Settings Sync merges and profile copies can duplicate a block. Removing
        # only the first would report success while leaving the setting live, so
        # say so instead of half-doing it.
        raise click.ClickException(
            f"{path} contains {max(raw.count(start_marker), raw.count(end_marker))} copies of "
            f"the Headroom block {start_marker!r}. Remove the duplicates by hand, then re-run."
        )
    if not _owns_settings_block(path, raw, start_marker, end_marker):
        raise click.ClickException(
            f"{path} contains {start_marker!r} text that Headroom cannot prove it wrote "
            "(an incomplete marker pair, or a block that has been edited since). "
            "Refusing to touch it -- remove those lines by hand if they are stale."
        )
    span = _settings_block_span(raw, start_marker, end_marker)
    assert span is not None  # guaranteed by _owns_settings_block
    line_start, line_end = span
    prefix = raw[:line_start]
    suffix = raw[line_end:]
    trimmed = prefix.rstrip()
    if trimmed.endswith(","):
        prefix = trimmed[:-1] + prefix[len(trimmed) :]
    # Which of our other blocks did we own before this edit? Removing one block
    # can rewrite a sibling's text -- dropping an orphaned separator comma moves
    # the comma off the next block's first line -- and a stale digest would then
    # disown a block we do own, leaving `unwrap` unable to finish.
    others = [
        (start, end)
        for start, end in (
            (_CAPI_MARKER_START, _CAPI_MARKER_END),
            (_BYOK_MARKER_START, _BYOK_MARKER_END),
        )
        if start != start_marker and _owns_settings_block(path, raw, start, end)
    ]

    updated = _drop_orphan_leading_comma(prefix + suffix)
    _validate_settings(updated, path)
    fsutil.write_text(path, updated)
    _record_settings_block(path, start_marker, None)
    for start, end in others:
        _record_settings_block(path, start, _settings_block_text(updated, start, end))
    return True


def proxy_base_url(port: int, project: str | None = None) -> str:
    """Base URL VS Code should call, carrying the per-project savings prefix."""
    return str(with_project_prefix(f"http://127.0.0.1:{port}", project))
