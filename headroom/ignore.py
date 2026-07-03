"""Centralized ignore-path policy for Headroom.

Some repositories generate agent-instruction files (``CLAUDE.md``,
``AGENTS.md``, ``.github/copilot-instructions.md``, ``.cursorrules``,
``ANTIGRAVITY.md``, ...) from a canonical source of truth managed elsewhere
(for example cARL-managed governance under ``.github/carl/``). If Headroom
compresses, learns from, indexes, or mutates those generated files directly,
it can create drift that gets silently overwritten by the owning tool the
next time it regenerates them.

This module gives every code path that reads/writes/learns-from/indexes
paths a single place to ask "is this path ignored for what I'm about to do?"
instead of scattering ad hoc glob checks around the codebase.

Two sources of rules, both optional and both off by default (so existing
behavior is unchanged when neither is present):

1. A ``.headroomignore`` file at the repository/workspace root, using a
   **limited gitignore-like** glob syntax — it is NOT a full gitignore or
   ``pathspec`` implementation (no ``.gitignore``-file discovery/precedence,
   no negation). Rules:
     - Blank lines and lines starting with ``#`` are ignored.
     - Lines starting with ``!`` (gitignore negation/re-inclusion) are
       **not supported**: they are skipped and reported in
       :attr:`IgnorePolicy.warnings` rather than silently matched as a
       literal path.
     - A trailing ``/`` marks a directory rule: it matches the directory
       itself and everything below it. A bare directory name (no other
       ``/``, e.g. ``node_modules/``) matches at *any* depth, same as a
       plain gitignore entry; a directory rule containing another ``/``
       (e.g. ``.github/carl/``) or a leading ``/`` is rooted instead.
     - A leading ``/`` roots the pattern at the repository root (e.g.
       ``/CLAUDE.md`` matches only the top-level ``CLAUDE.md``, never
       ``nested/CLAUDE.md``) — same as gitignore.
     - A pattern containing ``/`` (after stripping a leading ``/``, if any)
       is matched relative to the root (rooted match) using ``fnmatch`` glob
       semantics (``*``/``?``/``[...]``, and ``**`` for "any number of path
       segments"). Known deviation from real gitignore: a single ``*``
       here can still cross ``/`` boundaries (``fnmatch`` has no
       path-segment concept), so e.g. ``.github/*`` also matches
       ``.github/nested/file``, not just direct children.
     - A pattern with no ``/`` matches the basename at *any* depth, mirroring
       plain gitignore entries (e.g. ``CLAUDE.md`` matches both
       ``CLAUDE.md`` and ``nested/CLAUDE.md``).
   Every rule loaded from ``.headroomignore`` applies to *all* behaviors.

2. Config-driven rules on :class:`headroom.config.IgnoreConfig` (part of
   :class:`headroom.config.HeadroomConfig`), which support the same glob
   syntax and can optionally be scoped to a single behavior:
   ``paths`` (all behaviors), ``compress``, ``learn``, ``mutate``, ``memory``.

Behaviors are intentionally coarse and reusable:
    - ``compress``: don't compress this path's content.
    - ``learn``:    don't treat this path's content as learned/canonical input.
    - ``mutate``:   don't write/modify this path.
    - ``memory``:   don't index/persist this path into repository memory.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from . import fsutil

logger = logging.getLogger(__name__)

IgnoreBehavior = Literal["compress", "learn", "mutate", "memory"]

#: All known behaviors. Used for rules that apply globally (no scoping).
ALL_BEHAVIORS: tuple[IgnoreBehavior, ...] = ("compress", "learn", "mutate", "memory")

#: Name of the ignore file looked up at the repository/workspace root.
IGNORE_FILE_NAME = ".headroomignore"


@dataclass(frozen=True)
class IgnoreRule:
    """A single ignore rule, resolved from either a file or config."""

    pattern: str
    behaviors: frozenset[str]
    source: str  # e.g. ".headroomignore" or "config:ignore.mutate"

    @property
    def is_global(self) -> bool:
        return self.behaviors == frozenset(ALL_BEHAVIORS)

    def applies_to(self, behavior: str) -> bool:
        return behavior in self.behaviors

    def matches(self, rel_posix: str) -> bool:
        """Return True if this rule's pattern matches ``rel_posix``.

        ``rel_posix`` is a root-relative, forward-slash path with no leading
        slash (e.g. ``"a/b/c.py"``).

        This is a *limited* gitignore-like syntax, not a full gitignore/
        pathspec implementation — see the module docstring for the precise
        (and intentionally narrower) rules this implements.
        """
        pattern = self.pattern
        is_dir_rule = pattern.endswith("/")
        pat = pattern[:-1] if is_dir_rule else pattern
        if not pat:
            return False

        # A leading "/" explicitly roots the pattern at the repository root
        # (gitignore semantics), same as a pattern that contains "/"
        # elsewhere — it is never treated as a bare "match at any depth"
        # basename pattern.
        rooted = pat.startswith("/")
        if rooted:
            pat = pat[1:]
        if not pat:
            return False

        if is_dir_rule:
            if rooted or "/" in pat:
                return rel_posix == pat or rel_posix.startswith(pat + "/")
            # Bare directory name (e.g. "node_modules/"): matches that
            # directory at *any* depth, mirroring a plain gitignore entry.
            segments = rel_posix.split("/")
            return (
                rel_posix == pat
                or rel_posix.startswith(pat + "/")
                or pat in segments[:-1]
            )

        if rooted or "/" in pat:
            # Rooted pattern: match against the full relative path. Also treat
            # a pattern ending in "/**" as matching the directory itself, not
            # just its contents (gitignore lets "dir/**" cover "dir" too).
            #
            # Known limitation: unlike real gitignore, a single "*" here can
            # still cross "/" boundaries (fnmatch has no path-segment concept),
            # so e.g. ".github/*" also matches ".github/nested/file" — not
            # just direct children. Use a rooted, no-wildcard pattern or
            # ``.github/**`` if that distinction matters to you.
            if fnmatch.fnmatchcase(rel_posix, pat):
                return True
            if pat.endswith("/**"):
                base = pat[: -len("/**")]
                if rel_posix == base or rel_posix.startswith(base + "/"):
                    return True
            return False

        # Bare pattern (no slash): matches the basename at any depth, like a
        # plain gitignore entry (e.g. "CLAUDE.md" or "node_modules").
        name = rel_posix.rsplit("/", 1)[-1]
        return fnmatch.fnmatchcase(name, pat) or fnmatch.fnmatchcase(rel_posix, pat)


def _parse_lines(text: str, source: str) -> tuple[list[IgnoreRule], list[str]]:
    rules: list[IgnoreRule] = []
    warnings: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("!"):
            # Negation patterns ("!foo" = re-include a previously ignored
            # path) are valid gitignore syntax but are NOT supported here —
            # rules are a flat allow-list of ignores with no re-inclusion
            # pass. Skip (rather than silently matching the literal path
            # "!foo") and surface a warning so this is visible in
            # `headroom doctor`.
            msg = f"ignoring unsupported negation pattern {line!r} in {source} (negation is not supported)"
            logger.warning(msg)
            warnings.append(msg)
            continue
        rules.append(IgnoreRule(pattern=line, behaviors=frozenset(ALL_BEHAVIORS), source=source))
    return rules, warnings


def _rules_from_config(config: object | None) -> tuple[list[IgnoreRule], list[str]]:
    """Build rules from a ``headroom.config.IgnoreConfig``-shaped object.

    Duck-typed (rather than importing ``IgnoreConfig`` directly) to avoid a
    circular import between ``config.py`` and this module; ``config.py``
    imports ``IgnorePolicy``/``load_ignore_policy`` for diagnostics helpers.

    Returns ``(rules, warnings)``. Malformed entries are both logged *and*
    returned as warnings so ``IgnorePolicy.warnings`` (and therefore
    ``headroom doctor``) can surface them — logging alone is invisible to
    anyone not tailing application logs.
    """
    if config is None:
        return [], []

    rules: list[IgnoreRule] = []
    warnings: list[str] = []
    for behavior in ("paths", *ALL_BEHAVIORS):
        raw_patterns = getattr(config, behavior, None)
        if not raw_patterns:
            continue
        behaviors = frozenset(ALL_BEHAVIORS) if behavior == "paths" else frozenset({behavior})
        source = f"config:ignore.{behavior}"
        for pattern in raw_patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                msg = (
                    f"ignoring malformed ignore.{behavior} entry {pattern!r} "
                    "(expected non-empty string)"
                )
                logger.warning(msg)
                warnings.append(msg)
                continue
            stripped = pattern.strip()
            if stripped.startswith("!"):
                # See _parse_lines: negation patterns are valid gitignore
                # syntax but unsupported here.
                msg = (
                    f"ignoring unsupported negation pattern {stripped!r} in "
                    f"ignore.{behavior} (negation is not supported)"
                )
                logger.warning(msg)
                warnings.append(msg)
                continue
            rules.append(IgnoreRule(pattern=stripped, behaviors=behaviors, source=source))
    return rules, warnings


@dataclass
class IgnorePolicy:
    """Resolved set of ignore rules for a given repository/workspace root."""

    root: Path
    rules: list[IgnoreRule] = field(default_factory=list)
    #: Non-fatal problems encountered while loading rules (e.g. unreadable
    #: ``.headroomignore``, malformed config entries already logged above).
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, root: str | Path, config: object | None = None) -> IgnorePolicy:
        """Load ignore rules for ``root`` from ``.headroomignore`` and ``config``.

        ``config``, if given, should be a ``headroom.config.IgnoreConfig`` (or
        any object exposing ``paths``/``compress``/``learn``/``mutate``/``memory``
        list-of-str attributes). Missing file / missing config are not errors —
        the resulting policy simply has no rules, preserving prior behavior.
        """
        root_path = Path(root).resolve()
        rules: list[IgnoreRule] = []
        warnings: list[str] = []

        ignore_file = root_path / IGNORE_FILE_NAME
        if ignore_file.exists():
            try:
                text = fsutil.read_text(ignore_file)
            except OSError as exc:
                msg = f"could not read {ignore_file}: {exc}"
                logger.warning(msg)
                warnings.append(msg)
            else:
                file_rules, file_warnings = _parse_lines(text, source=IGNORE_FILE_NAME)
                rules.extend(file_rules)
                warnings.extend(file_warnings)

        try:
            config_rules, config_warnings = _rules_from_config(config)
        except (TypeError, AttributeError) as exc:  # malformed config object
            msg = f"could not load ignore rules from config: {exc}"
            logger.warning(msg)
            warnings.append(msg)
        else:
            rules.extend(config_rules)
            warnings.extend(config_warnings)

        return cls(root=root_path, rules=rules, warnings=warnings)

    def _relativize(self, path: str | Path) -> str:
        p = Path(path)
        try:
            if p.is_absolute():
                rel = p.resolve().relative_to(self.root)
            else:
                rel = p
        except ValueError:
            # Path isn't under root — fall back to the given (possibly
            # already-relative) path so bare-name rules can still match.
            rel = p
        rel_posix = rel.as_posix()
        if rel_posix.startswith("./"):
            rel_posix = rel_posix[2:]
        return rel_posix

    def is_ignored(self, path: str | Path, behavior: IgnoreBehavior) -> bool:
        """Return True if ``path`` is ignored for ``behavior``.

        ``path`` may be absolute or relative; absolute paths under ``root``
        are made root-relative before matching (requirement: ignored paths
        are matched relative to the repository/workspace root).
        """
        if not self.rules:
            return False
        rel_posix = self._relativize(path)
        return any(rule.applies_to(behavior) and rule.matches(rel_posix) for rule in self.rules)

    def matching_rule(self, path: str | Path, behavior: IgnoreBehavior) -> IgnoreRule | None:
        """Return the first rule matching ``path`` for ``behavior``, if any."""
        rel_posix = self._relativize(path)
        for rule in self.rules:
            if rule.applies_to(behavior) and rule.matches(rel_posix):
                return rule
        return None

    def active_rules(self) -> list[IgnoreRule]:
        """Return all loaded rules, for diagnostics (e.g. ``headroom doctor``)."""
        return list(self.rules)

    def describe(self) -> list[str]:
        """Human-readable one-line-per-rule summary, for diagnostics/debug output."""
        lines = []
        for rule in self.rules:
            scope = "all" if rule.is_global else ",".join(sorted(rule.behaviors))
            lines.append(f"{rule.pattern}  [{scope}]  (from {rule.source})")
        return lines


def load_ignore_policy(root: str | Path, config: object | None = None) -> IgnorePolicy:
    """Convenience wrapper around :meth:`IgnorePolicy.load`."""
    return IgnorePolicy.load(root, config)


def is_path_ignored(
    path: str | Path,
    behavior: IgnoreBehavior,
    *,
    root: str | Path | None = None,
    config: object | None = None,
) -> bool:
    """One-shot convenience check: is ``path`` ignored for ``behavior``?

    Loads a fresh :class:`IgnorePolicy` for ``root`` (defaults to ``Path.cwd()``)
    each call. Callers that check many paths against the same root/config
    (e.g. a file watcher) should build one :class:`IgnorePolicy` and reuse it
    instead of calling this repeatedly.
    """
    policy = IgnorePolicy.load(root if root is not None else Path.cwd(), config)
    return policy.is_ignored(path, behavior)
