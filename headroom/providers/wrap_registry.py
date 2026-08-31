"""Declarative registry of env-var wrap targets.

Extension seam for ``headroom wrap``: a tool whose integration is fully
described by data — the binary to launch, which environment variables point at
the local proxy, and the upstream defaults — is registered here as a
:class:`WrapTarget` instead of a hand-written command in ``cli/wrap.py``.
``cli/wrap.py`` generates one click command per entry, so adding a new
env-var tool is a registry entry, not a new command body. Mirrors the
declarative-route seam (:mod:`headroom.providers.route_specs`): the core
defines the contract and the launch flow; the entry supplies the data.

Tools that need imperative setup (settings files, token exchange, MCP
registrars, config rendering) do not fit this contract and keep their
bespoke commands.

User configuration
------------------

``~/.headroom/config/wrap_targets.json`` (never auto-created; see
:func:`headroom.paths.wrap_targets_config_path`) overlays these code
defaults per field, following the ``models.json`` read-if-exists pattern::

    {
      "version": 1,
      "targets": {
        "bob": {"default_mode": "cache"},
        "mytool": {
          "binaries": ["mytool"],
          "install_hint": "pip install mytool",
          "env_vars": [{"key": "OPENAI_BASE_URL", "style": "openai_v1"}]
        }
      }
    }

Precedence is unchanged from the rest of Headroom: explicit CLI flag >
environment variable > this file > code default. The file only replaces
*code defaults* — every existing env-wins check downstream (e.g.
``HEADROOM_MODE`` over ``default_mode``) is untouched by construction.

The ``version`` key is required; an unknown version rejects the whole file
loudly rather than silently misapplying it. Within a valid file each target
resolves atomically: any invalid field skips that target's entire overlay
(the code default stays in force) so a half-applied "chimera" target can
never exist. Config-defined *new* targets may use data fields only;
behavior-crossing fields (``extra_chat_routes``,
``origin_passthrough_prefixes``, ``origin_passthrough_strip_json_keys``)
are accepted only as overrides of built-in targets. Validate with
``headroom wrap targets`` or ``headroom doctor``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Literal

from headroom import paths as _paths
from headroom.providers.claude.runtime import proxy_base_url as _anthropic_proxy_base_url
from headroom.providers.codex.runtime import proxy_base_url as _openai_proxy_base_url
from headroom.proxy.project_policy import with_project_prefix
from headroom.proxy.proxy_mode_policy import normalize_proxy_mode_decision

logger = logging.getLogger(__name__)

# How an EnvVar's URL is shaped. ``openai_v1`` ends in ``/v1`` (OpenAI-style
# clients append ``/chat/completions``); ``anthropic`` is a bare origin
# (Anthropic clients append ``/v1/messages``); ``bare_origin`` is for tools
# that append their own full path prefix to the URL they are handed (e.g. IBM
# Bob appends ``/inference/v1/chat/completions``) — handing those a ``/v1``
# base would produce a doubled prefix.
UrlStyle = Literal["openai_v1", "anthropic", "bare_origin"]

_STYLE_BUILDERS = {
    "openai_v1": _openai_proxy_base_url,
    "anthropic": _anthropic_proxy_base_url,
    # Same shape as the anthropic base today (http://127.0.0.1:<port>), but a
    # distinct style: it documents *why* the URL has no /v1, and keeps targets
    # honest if the anthropic base ever grows a suffix.
    "bare_origin": _anthropic_proxy_base_url,
}


@dataclass(frozen=True, slots=True)
class EnvVar:
    """One environment variable a wrap target needs pointed at the proxy.

    ``display`` controls whether the assignment is echoed in the wrap banner;
    hidden aliases (e.g. ``OPENAI_API_BASE`` next to ``OPENAI_BASE_URL``) are
    set but not shown, matching the hand-written commands they replaced.
    """

    key: str
    style: UrlStyle
    display: bool = True


@dataclass(frozen=True, slots=True)
class WrapTarget:
    """Everything needed to generate a ``headroom wrap <name>`` command."""

    name: str
    binaries: tuple[str, ...]
    install_hint: str
    env_vars: tuple[EnvVar, ...]
    help_text: str
    # Encode the launch directory as a /p/<name> base-URL prefix so the proxy
    # can attribute savings per project (for tools that can't send headers).
    project_prefix: bool = True
    # Upstream defaults handed to the proxy at startup.
    openai_api_url: str | None = None
    anthropic_api_url: str | None = None
    # Whether the generated command exposes --backend/--anyllm-provider/--region.
    backend_options: bool = True
    # Nonstandard chat-completions paths the tool posts to. Each is registered
    # as a proxy route delegating to ``handle_openai_chat`` (see
    # ``route_specs.OPENAI_HANDLER_ROUTES``) so the traffic is compressed
    # instead of falling through to the uncompressed catch-all passthrough.
    extra_chat_routes: tuple[str, ...] = ()
    # Non-chat gateway paths the tool sends under its own prefixes (it appends
    # full paths to the bare origin it is handed). The proxy's catch-all
    # passthrough must forward these to the upstream ORIGIN with the inbound
    # path verbatim; appending them to ``openai_api_url``'s path produces a
    # doubled or misrooted URL (e.g. ``.../inference/inference/v1/model/info``,
    # ``.../inference/admin/v1/profile`` — both 403 at IBM's edge).
    origin_passthrough_prefixes: tuple[str, ...] = ()
    # (path, key) pairs to strip from JSON responses on this target's origin
    # passthrough. Bob 2.0.1's ``resolveBaseUrl`` rewrites its gateway
    # hostname from ``region_domain`` in the ``/admin/v1/profile`` response
    # while keeping the proxied port, so every request after the first profile
    # fetch targets ``api.<region>:<proxy-port>`` — unreachable. With the key
    # absent, bob falls back to its configured gateway URL (the proxy).
    origin_passthrough_strip_json_keys: tuple[tuple[str, str], ...] = ()
    # Preferred proxy mode when the user has not chosen one. The generated
    # wrap command exports it as HEADROOM_MODE before proxy startup; an
    # explicit HEADROOM_MODE in the environment always wins. Only affects a
    # proxy this wrap starts — an already-running proxy is reused with the
    # mode it booted with (wrap warns on the mismatch via /health's mode).
    default_mode: str | None = None
    agent_type: str = ""
    tool_label: str = ""

    def __post_init__(self) -> None:
        if not self.agent_type:
            object.__setattr__(self, "agent_type", self.name)
        if not self.tool_label:
            object.__setattr__(self, "tool_label", self.name.upper())


def build_launch_env(
    target: WrapTarget,
    port: int,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Build the launch environment for ``target`` routed through the proxy."""
    env = dict(environ if environ is not None else os.environ)
    display: list[str] = []
    for var in target.env_vars:
        url = _STYLE_BUILDERS[var.style](port)
        if target.project_prefix:
            url = with_project_prefix(url, project)
        env[var.key] = url
        if var.display:
            display.append(f"{var.key}={url}")
    return env, display


WRAP_TARGETS: dict[str, WrapTarget] = {
    target.name: target
    for target in (
        WrapTarget(
            name="goose",
            binaries=("goose",),
            install_hint="Install Goose: https://block.github.io/goose/",
            env_vars=(
                EnvVar("OPENAI_BASE_URL", "openai_v1"),
                EnvVar("OPENAI_API_BASE", "openai_v1", display=False),
                EnvVar("ANTHROPIC_BASE_URL", "anthropic"),
            ),
            project_prefix=False,
            help_text=(
                "Launch Goose (Block) CLI through Headroom proxy.\n"
                "\n"
                "\b\n"
                "Sets OPENAI_BASE_URL and ANTHROPIC_BASE_URL to route Goose's API calls\n"
                "through Headroom.\n"
                "\n"
                "\b\n"
                "Uninstall: there is no ``headroom unwrap goose`` subcommand — nothing is\n"
                "written to the project.\n"
                "\n"
                "\b\n"
                "Examples:\n"
                "    headroom wrap goose                          # Start proxy + goose\n"
                "    headroom wrap goose -- session               # Start a Goose session\n"
                "    headroom wrap goose -- --provider anthropic  # Pass args to goose"
            ),
        ),
        WrapTarget(
            name="openhands",
            binaries=("openhands",),
            install_hint="Install OpenHands: https://docs.all-hands.dev/",
            env_vars=(
                EnvVar("OPENAI_BASE_URL", "openai_v1"),
                EnvVar("OPENAI_API_BASE", "openai_v1", display=False),
                EnvVar("ANTHROPIC_BASE_URL", "anthropic"),
                # OpenHands' generic LLM provider config reads LLM_BASE_URL.
                EnvVar("LLM_BASE_URL", "openai_v1"),
            ),
            project_prefix=False,
            help_text=(
                "Launch OpenHands CLI through Headroom proxy.\n"
                "\n"
                "\b\n"
                "Sets OPENAI_BASE_URL / ANTHROPIC_BASE_URL to route OpenHands' API calls\n"
                "through Headroom. Nothing is written to disk, so there is nothing to undo.\n"
                "\n"
                "\b\n"
                "Examples:\n"
                "    headroom wrap openhands                # Start proxy + openhands\n"
                "    headroom wrap openhands -- --task ...  # Pass args to openhands"
            ),
        ),
        WrapTarget(
            name="bob",
            binaries=("bob",),
            install_hint="Install IBM Bob CLI: npm install -g bobshell",
            env_vars=(
                # Bob resolves its gateway as config.gatewayUrl ?? BOB_GATEWAY_URL
                # ?? default, so this env var reroutes inference without touching
                # ~/.bob/settings. Bob appends /inference/v1/... itself.
                EnvVar("BOB_GATEWAY_URL", "bare_origin"),
            ),
            # DEFAULT_API_URL carries the /inference/v1 suffix so the proxy's
            # _normalize_api_url (strips /v1) and handle_openai_chat (re-appends
            # /v1/chat/completions) compose back into the path IBM serves.
            openai_api_url="https://api.us-east.bob.ibm.com/inference/v1",
            extra_chat_routes=("/inference/v1/chat/completions",),
            origin_passthrough_prefixes=("/inference/", "/admin/"),
            origin_passthrough_strip_json_keys=(("/admin/v1/profile", "region_domain"),),
            # Bob bills flat per token (no prompt-cache discount to protect),
            # so token mode converts compression 1:1 into dollars.
            default_mode="token",
            help_text=(
                "Launch IBM Bob CLI through Headroom proxy.\n"
                "\n"
                "\b\n"
                "Sets ``BOB_GATEWAY_URL`` so Bob routes inference traffic through\n"
                "Headroom while keeping its own ``Authorization: apikey ...``\n"
                "credential and its ~/.bob/settings files untouched.\n"
                "\n"
                "\b\n"
                "Mode matters more for Bob than for most agents: its traffic is ~46%\n"
                "system prompt and ~44% tool output, and Bob bills flat per token, so\n"
                "token mode converts compression 1:1 into dollars. Token mode is the\n"
                "default for bob; set HEADROOM_MODE to override:\n"
                "    HEADROOM_MODE=cache headroom wrap bob\n"
                "\n"
                "\b\n"
                "Examples:\n"
                "    headroom wrap bob                          # Start proxy + bob\n"
                '    headroom wrap bob -- run "fix the bug"     # Pass args to bob\n'
                "    headroom wrap bob --port 9999              # Custom proxy port"
            ),
        ),
        WrapTarget(
            name="openclaude",
            binaries=("openclaude",),
            install_hint="Install OpenClaude before running `headroom wrap openclaude`.",
            env_vars=(
                EnvVar("OPENAI_API_BASE", "openai_v1"),
                EnvVar("ANTHROPIC_BASE_URL", "anthropic"),
            ),
            help_text=(
                "Launch OpenClaude through Headroom proxy.\n"
                "\n"
                "\b\n"
                "Sets OPENAI_API_BASE and ANTHROPIC_BASE_URL to route API calls\n"
                "through Headroom.\n"
                "\n"
                "\b\n"
                "Examples:\n"
                "    headroom wrap openclaude               # Start proxy + openclaude\n"
                "    headroom wrap openclaude -- --help     # Pass args to openclaude"
            ),
        ),
    )
}


def get_wrap_target(name: str) -> WrapTarget:
    """Return the effective target for ``name``; raises ``KeyError`` if unknown.

    "Effective" means code defaults with any ``wrap_targets.json`` overlay
    applied — callers see the same target the generated command launches.
    """
    return resolved_wrap_targets()[name]


# ---------------------------------------------------------------------------
# wrap_targets.json overlay
# ---------------------------------------------------------------------------

WRAP_TARGETS_CONFIG_VERSION = 1

# Effect classes label what changing a field actually does, so validation
# output (``headroom wrap targets``, doctor) can say more than "value set":
#   data         — display/launch metadata, no cross-cutting behavior
#   launch_env   — changes the environment handed to the launched tool
#   upstream     — changes where the proxy sends traffic
#   mode         — changes the proxy mode a fresh proxy boots with
#   proxy_route  — adds compressed-chat routes to the proxy's route table
#   proxy_rewrite— changes passthrough forwarding / response rewriting
_BEHAVIOR_CROSSING_EFFECTS = frozenset({"proxy_route", "proxy_rewrite"})


def _coerce_str(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected a non-empty string")
    return value


def _coerce_opt_str(value: object) -> str | None:
    if value is None:
        return None
    return _coerce_str(value)


def _coerce_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("expected true or false")
    return value


def _coerce_str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError("expected a list of non-empty strings")
    return tuple(value)


def _coerce_route_tuple(value: object) -> tuple[str, ...]:
    routes = _coerce_str_tuple(value)
    if not all(r.startswith("/") for r in routes):
        raise ValueError("every path must start with '/'")
    return routes


def _coerce_env_vars(value: object) -> tuple[EnvVar, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("expected a non-empty list of {key, style, display?} objects")
    out: list[EnvVar] = []
    for item in value:
        if not isinstance(item, dict) or not set(item) <= {"key", "style", "display"}:
            raise ValueError("each env var must be an object with keys: key, style, display?")
        key = _coerce_str(item.get("key"))
        style = item.get("style")
        if style not in _STYLE_BUILDERS:
            raise ValueError(f"style must be one of {sorted(_STYLE_BUILDERS)}, got {style!r}")
        display = item.get("display", True)
        out.append(EnvVar(key, style, _coerce_bool(display)))
    return tuple(out)


def _coerce_strip_keys(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ValueError("expected a list of [path, key] pairs")
    out: list[tuple[str, str]] = []
    for item in value:
        if not (isinstance(item, list) and len(item) == 2):
            raise ValueError("each entry must be a [path, key] pair")
        path, key = _coerce_str(item[0]), _coerce_str(item[1])
        if not path.startswith("/"):
            raise ValueError("strip path must start with '/'")
        out.append((path, key))
    return tuple(out)


def _coerce_mode(value: object) -> str:
    mode = _coerce_str(value)
    decision = normalize_proxy_mode_decision(mode, default="token")
    if decision.unknown:
        raise ValueError(f"unknown proxy mode {mode!r} (use 'token' or 'cache')")
    return decision.normalized


@dataclass(frozen=True, slots=True)
class TargetField:
    """Descriptor for one configurable :class:`WrapTarget` field."""

    name: str
    effect: str
    coerce: Callable[[object], object]


# One descriptor per WrapTarget field except ``name`` (the JSON key). A test
# pins parity with the dataclass so a field added there cannot silently
# become unconfigurable — or configurable without validation.
_TARGET_FIELDS: dict[str, TargetField] = {
    f.name: f
    for f in (
        TargetField("binaries", "data", _coerce_str_tuple),
        TargetField("install_hint", "data", _coerce_str),
        TargetField("env_vars", "launch_env", _coerce_env_vars),
        TargetField("help_text", "data", _coerce_str),
        TargetField("project_prefix", "launch_env", _coerce_bool),
        TargetField("openai_api_url", "upstream", _coerce_opt_str),
        TargetField("anthropic_api_url", "upstream", _coerce_opt_str),
        TargetField("backend_options", "data", _coerce_bool),
        TargetField("extra_chat_routes", "proxy_route", _coerce_route_tuple),
        TargetField("origin_passthrough_prefixes", "proxy_rewrite", _coerce_route_tuple),
        TargetField(
            "origin_passthrough_strip_json_keys", "proxy_rewrite", _coerce_strip_keys
        ),
        TargetField("default_mode", "mode", _coerce_mode),
        TargetField("agent_type", "data", _coerce_str),
        TargetField("tool_label", "data", _coerce_str),
    )
}

_NEW_TARGET_REQUIRED = ("binaries", "install_hint", "env_vars")


@dataclass(frozen=True, slots=True)
class TargetOutcome:
    """Resolution outcome for one config-file target entry."""

    name: str
    action: str  # "overridden" | "added" | "skipped"
    fields: tuple[str, ...] = ()  # fields applied (with effect class in report)
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OverlayStatus:
    """Full resolution report for ``headroom wrap targets`` / doctor."""

    path: str
    exists: bool
    fingerprint: str | None
    warnings: tuple[str, ...]
    outcomes: tuple[TargetOutcome, ...]

    @property
    def ok(self) -> bool:
        return not self.warnings and not any(o.action == "skipped" for o in self.outcomes)


def _validate_target_section(
    name: str, base: WrapTarget | None, section: object
) -> tuple[WrapTarget | None, TargetOutcome]:
    """Resolve one target entry atomically: all fields valid or none applied."""
    if not isinstance(section, dict):
        return base, TargetOutcome(name, "skipped", errors=("entry must be a JSON object",))

    errors: list[str] = []
    coerced: dict[str, object] = {}
    for key, raw in section.items():
        descriptor = _TARGET_FIELDS.get(key)
        if descriptor is None:
            errors.append(f"unknown field {key!r}")
            continue
        if base is None and descriptor.effect in _BEHAVIOR_CROSSING_EFFECTS:
            errors.append(
                f"{key!r} affects proxy routing and is only allowed when "
                "overriding a built-in target, not on new targets"
            )
            continue
        try:
            coerced[key] = descriptor.coerce(raw)
        except ValueError as exc:
            errors.append(f"{key}: {exc}")

    if base is None:
        missing = [k for k in _NEW_TARGET_REQUIRED if k not in coerced]
        if missing and not errors:
            errors.append(f"new target missing required fields: {', '.join(missing)}")

    if errors:
        return base, TargetOutcome(name, "skipped", errors=tuple(errors))
    if base is not None:
        applied = tuple(f"{k} ({_TARGET_FIELDS[k].effect})" for k in coerced)
        return dataclass_replace(base, **coerced), TargetOutcome(name, "overridden", applied)
    coerced.setdefault(
        "help_text", f"Launch {name} through Headroom proxy.\n(defined in wrap_targets.json)"
    )
    target = WrapTarget(name=name, **coerced)  # type: ignore[arg-type]
    applied = tuple(f"{k} ({_TARGET_FIELDS[k].effect})" for k in coerced)
    return target, TargetOutcome(name, "added", applied)


@dataclass(frozen=True, slots=True)
class _Resolution:
    targets: dict[str, WrapTarget]
    status: OverlayStatus


_RESOLUTION: _Resolution | None = None


def _resolve() -> _Resolution:
    """Merge code defaults with wrap_targets.json (read-if-exists, fail-open)."""
    path = _paths.wrap_targets_config_path()
    # Stat guard: the no-config common case does no parsing and no copying.
    if not path.exists():
        return _Resolution(
            WRAP_TARGETS, OverlayStatus(str(path), False, None, (), ())
        )

    warnings: list[str] = []
    outcomes: list[TargetOutcome] = []
    fingerprint: str | None = None
    section: dict[str, object] = {}
    try:
        raw_bytes = path.read_bytes()
        fingerprint = hashlib.sha256(raw_bytes).hexdigest()[:16]
        raw = json.loads(raw_bytes)
        if not isinstance(raw, dict):
            raise ValueError("top level must be a JSON object")
        version = raw.get("version")
        if version != WRAP_TARGETS_CONFIG_VERSION:
            raise ValueError(
                f"requires \"version\": {WRAP_TARGETS_CONFIG_VERSION}, got {version!r} "
                "(a newer Headroom may be required)"
            )
        targets_raw = raw.get("targets", {})
        if not isinstance(targets_raw, dict):
            raise ValueError('"targets" must be a JSON object')
        section = targets_raw
    except (OSError, ValueError) as exc:
        # Fail-open: the whole file is rejected loudly; built-ins stay intact.
        warnings.append(f"{path}: {exc}")
        logger.warning("wrap_targets config ignored: %s: %s", path, exc)
        return _Resolution(
            WRAP_TARGETS,
            OverlayStatus(str(path), True, fingerprint, tuple(warnings), ()),
        )

    resolved = dict(WRAP_TARGETS)
    for name, entry in section.items():
        base = WRAP_TARGETS.get(name)
        target, outcome = _validate_target_section(name, base, entry)
        outcomes.append(outcome)
        if outcome.action == "skipped":
            logger.warning(
                "wrap_targets config: target %r skipped: %s", name, "; ".join(outcome.errors)
            )
        elif target is not None:
            resolved[name] = target
    return _Resolution(
        resolved, OverlayStatus(str(path), True, fingerprint, tuple(warnings), tuple(outcomes))
    )


def resolved_wrap_targets() -> dict[str, WrapTarget]:
    """Effective registry: code defaults + wrap_targets.json overlay, cached."""
    global _RESOLUTION
    if _RESOLUTION is None:
        _RESOLUTION = _resolve()
    return _RESOLUTION.targets


def wrap_targets_overlay_status() -> OverlayStatus:
    """Resolution report for validation front doors (wrap targets, doctor)."""
    resolved_wrap_targets()
    assert _RESOLUTION is not None
    return _RESOLUTION.status


def wrap_targets_config_fingerprint() -> str | None:
    """Short sha256 of the overlay file as loaded, or None without a file.

    The proxy reports this from /health so ``headroom wrap`` can warn when a
    reused proxy was started under a different (or since-edited) config.
    """
    return wrap_targets_overlay_status().fingerprint


def current_wrap_targets_file_fingerprint() -> str | None:
    """Short sha256 of the overlay file as it exists on disk right now."""
    path = _paths.wrap_targets_config_path()
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def _reset_wrap_targets_cache() -> None:
    """Testing hook: drop the cached resolution (and derived origin index)."""
    global _RESOLUTION, _ORIGIN_INDEX
    _RESOLUTION = None
    _ORIGIN_INDEX = None


# ---------------------------------------------------------------------------
# Origin passthrough (precomputed host index over the resolved registry)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _OriginRules:
    prefixes: tuple[str, ...]
    strip_keys: tuple[tuple[str, str], ...]


_ORIGIN_INDEX: dict[tuple[str, str], _OriginRules] | None = None


def _origin_index() -> dict[tuple[str, str], _OriginRules]:
    """(scheme, netloc) -> passthrough rules, built once per process."""
    global _ORIGIN_INDEX
    if _ORIGIN_INDEX is None:
        from urllib.parse import urlsplit

        index: dict[tuple[str, str], _OriginRules] = {}
        for target in resolved_wrap_targets().values():
            if not target.openai_api_url:
                continue
            if not (target.origin_passthrough_prefixes or target.origin_passthrough_strip_json_keys):
                continue
            declared = urlsplit(target.openai_api_url)
            host = (declared.scheme, declared.netloc)
            existing = index.get(host)
            prefixes = target.origin_passthrough_prefixes
            strip_keys = target.origin_passthrough_strip_json_keys
            if existing is not None:
                prefixes = existing.prefixes + prefixes
                strip_keys = existing.strip_keys + strip_keys
            index[host] = _OriginRules(prefixes, strip_keys)
        _ORIGIN_INDEX = index
    return _ORIGIN_INDEX


def resolve_origin_passthrough_url(base_url: str | None, path: str) -> str | None:
    """Origin-rooted upstream URL for a declared origin-passthrough path.

    When ``base_url`` points at a registered target's gateway host and ``path``
    starts with one of that target's ``origin_passthrough_prefixes``, the tool
    built the path itself against a bare origin — forward it verbatim to that
    origin. Returns None when no target matches (caller keeps its normal
    base+path join).
    """
    if not base_url:
        return None
    from urllib.parse import urlsplit

    base = urlsplit(base_url)
    rules = _origin_index().get((base.scheme, base.netloc))
    if rules is not None and any(path.startswith(prefix) for prefix in rules.prefixes):
        return f"{base.scheme}://{base.netloc}{path}"
    return None


_MISSING = object()


def _strip_json_key(obj: object, key: str) -> bool:
    """Remove ``key`` from every dict in ``obj`` in place; True when removed."""
    removed = False
    if isinstance(obj, dict):
        if obj.pop(key, _MISSING) is not _MISSING:
            removed = True
        for value in obj.values():
            removed = _strip_json_key(value, key) or removed
    elif isinstance(obj, list):
        for value in obj:
            removed = _strip_json_key(value, key) or removed
    return removed


def strip_origin_passthrough_response_keys(
    base_url: str | None, path: str, body: bytes
) -> bytes | None:
    """Filtered JSON body for an origin-passthrough response, or None.

    Applies the ``origin_passthrough_strip_json_keys`` declarations of the
    target whose gateway host matches ``base_url``. Returns None when nothing
    is declared for this path, the body is not JSON, or no declared key was
    present.
    """
    if not base_url:
        return None
    from urllib.parse import urlsplit

    base = urlsplit(base_url)
    rules = _origin_index().get((base.scheme, base.netloc))
    if rules is None:
        return None
    keys = [
        key
        for declared_path, key in rules.strip_keys
        if path == declared_path or path.startswith(declared_path.rstrip("/") + "/")
    ]
    if not keys:
        return None
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    changed = False
    for key in keys:
        changed = _strip_json_key(payload, key) or changed
    if not changed:
        return None
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
