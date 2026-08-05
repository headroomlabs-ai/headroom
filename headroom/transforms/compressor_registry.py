"""Pluggable compressor registry and ``headroom.compressor`` entry-point seam.

This module lands a *name-addressable* seam for compressors: built-in
compressors can be registered explicitly, third-party compressors can be
discovered from the ``headroom.compressor`` entry-point group, and a caller can
opt in to a specific set of them by name.

It is deliberately **additive**. Nothing here is wired into the content router,
proxy server, or config in this change — registering or discovering a compressor
has no effect on request handling until an integration point is added in a
follow-up. Constructing a ``CompressorRegistry`` has no global side effects.

Rust-portable contract
----------------------
The compressor boundary is **pure data in / data out**. No Python-only objects
(tokenizer instances, live store handles, rich config classes) cross it:

  * :class:`CompressorDescriptor` — static, declarative capability metadata.
  * :class:`CompressInput` — the content plus plain ``dict`` config/budget.
  * :class:`CompressOutput` — the compressed content plus plain counts, string
    markers, a ``hash -> original`` recovery map, and string warnings.

Every field is a ``str``, ``int``, ``bool``, ``list``, or ``dict`` of those, so
an equivalent contract can be implemented in another language (e.g. a Rust
compressor invoked over the same shapes) without carrying Python objects.

Discovery vs. selection (opt-in)
--------------------------------
Discovery (:meth:`CompressorRegistry.discover`) enumerates and *loads* every
registered entry point — it may import a module and construct the compressor
object — but it never invokes ``compress``. Selection
(:meth:`CompressorRegistry.select` / :meth:`CompressorRegistry.active`) is
opt-in: with no names (or an empty set) nothing is active; the literal ``"*"``
selects everything registered; otherwise only the named-and-registered
compressors are active. This mirrors the opt-in model used for proxy extensions
so that merely installing a third-party package cannot silently change behavior.

External packages register a compressor like this::

    [project.entry-points."headroom.compressor"]
    my_compressor = "my_pkg.compressor:MyCompressor"

The value may be a :class:`Compressor` instance or a zero-arg class implementing
the protocol; a class is instantiated during discovery.
"""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "headroom.compressor"

#: Recognized values for :attr:`CompressorDescriptor.cost_tier`.
COST_TIERS: tuple[str, ...] = ("fast", "ml", "remote")


@dataclass(frozen=True)
class CompressorDescriptor:
    """Static, declarative metadata describing a compressor's capabilities.

    Attributes:
        name: Canonical, unique compressor name used for registration/selection.
        content_types: Content types this compressor handles (e.g. ``["text/plain"]``).
        lossless: Whether compression is losslessly reversible.
        cost_tier: One of :data:`COST_TIERS` — ``"fast"`` (local/cheap),
            ``"ml"`` (local model inference), or ``"remote"`` (network call).
        recoverable: Whether the compressor can emit a ``hash -> original``
            recovery map in :attr:`CompressOutput.recoverable`.
    """

    name: str
    content_types: list[str]
    lossless: bool
    cost_tier: str
    recoverable: bool


@dataclass
class CompressInput:
    """Pure-data input to :meth:`Compressor.compress`.

    Attributes:
        content: The raw content to compress.
        content_type: The content type of ``content``.
        query: Optional task/query hint for relevance-aware compressors.
        config: Plain compressor-specific configuration.
        budget: Plain budget hints, e.g. ``target_ratio`` (float),
            ``time_ms`` (int), ``max_items`` (int).
    """

    content: str
    content_type: str
    query: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressOutput:
    """Pure-data output from :meth:`Compressor.compress`.

    Attributes:
        content: The compressed content (or the original content unchanged when
            ``compressed`` is ``False``).
        tokens_before: Token count of the input content.
        tokens_after: Token count of the compressed content.
        lossless: Whether this particular result is losslessly reversible.
        markers: Marker strings describing what was applied (e.g. for routing).
        recoverable: ``hash -> original`` map for recovering dropped content.
        warnings: Non-fatal warning strings emitted during compression.
        compressed: Whether the compressor actually compressed/extracted the
            content. ``True`` (the default) means compression/extraction was
            applied and :attr:`content` is the transformed result; ``False``
            means the compressor did not compress — a passthrough — and
            :attr:`content` is the original input unchanged. Defaults to ``True``
            so existing and external compressors that do not set it are
            unaffected (treated as having compressed). Callers can read this to
            distinguish a real (possibly no-shrink) result from a passthrough and
            run their own fallback on a passthrough.
    """

    content: str
    tokens_before: int
    tokens_after: int
    lossless: bool
    markers: list[str] = field(default_factory=list)
    recoverable: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    compressed: bool = True


@runtime_checkable
class Compressor(Protocol):
    """Name-addressable compressor contract (pure data in / data out)."""

    @property
    def descriptor(self) -> CompressorDescriptor:
        """Return this compressor's static capability metadata."""
        ...

    def compress(self, inp: CompressInput) -> CompressOutput:
        """Compress ``inp`` and return a :class:`CompressOutput`.

        A compressor that does not compress the input (a passthrough) should
        return ``CompressOutput(content=inp.content, compressed=False, ...)``;
        the ``compressed`` flag defaults to ``True`` so a compressor that always
        transforms need not set it.
        """
        ...


class CompressorRegistry:
    """Registry of compressors addressable by :attr:`CompressorDescriptor.name`.

    Starts empty. Built-in compressors are added via :meth:`register`; external
    compressors are added via :meth:`discover`. Selection is opt-in.
    """

    def __init__(self) -> None:
        self._compressors: dict[str, Compressor] = {}

    def register(self, compressor: Compressor, *, replace: bool = False) -> str:
        """Register ``compressor`` under its ``descriptor.name``.

        Args:
            compressor: The compressor to register.
            replace: If ``True``, replace an existing registration of the same
                name instead of raising.

        Returns:
            The registered name.

        Raises:
            ValueError: If the name is empty, or already registered and
                ``replace`` is ``False``.
        """
        name = compressor.descriptor.name
        if not name:
            raise ValueError("compressor descriptor.name must be non-empty")
        if name in self._compressors and not replace:
            raise ValueError(f"compressor {name!r} is already registered")
        self._compressors[name] = compressor
        return name

    def get(self, name: str) -> Compressor | None:
        """Return the registered compressor named ``name``, or ``None``."""
        return self._compressors.get(name)

    def names(self) -> list[str]:
        """Return all registered compressor names, sorted."""
        return sorted(self._compressors)

    def descriptors(self) -> list[CompressorDescriptor]:
        """Return the descriptors of all registered compressors, sorted by name."""
        return [self._compressors[n].descriptor for n in sorted(self._compressors)]

    def discover(self, *, replace: bool = False) -> list[str]:
        """Load and register compressors from the ``headroom.compressor`` group.

        Mirrors the pipeline extension discovery helper: entry points are
        enumerated fail-open, each is loaded fail-open, and a class value is
        instantiated fail-open. A broken third-party package is logged and
        skipped rather than aborting discovery. ``compress`` is never invoked
        here — discovery only loads and registers.

        Args:
            replace: Passed through to :meth:`register` for name collisions.

        Returns:
            The list of newly registered compressor names.
        """
        discovered: list[str] = []
        try:
            entries = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        except Exception as exc:  # noqa: BLE001 - importlib metadata varies by runtime
            log.debug("compressor registry: entry-point enumeration failed: %s", exc)
            return discovered

        for entry in entries:
            try:
                obj = entry.load()
            except Exception as exc:  # noqa: BLE001 - third-party load failures are isolated
                log.warning("compressor %r failed to load: %s", entry.name, exc)
                continue

            if isinstance(obj, type):
                try:
                    obj = obj()
                except Exception as exc:  # noqa: BLE001
                    log.warning("compressor %r failed to initialize: %s", entry.name, exc)
                    continue

            try:
                name = self.register(obj, replace=replace)
            except Exception as exc:  # noqa: BLE001 - bad descriptor / duplicate name
                log.warning("compressor %r failed to register: %s", entry.name, exc)
                continue

            discovered.append(name)

        return discovered

    @staticmethod
    def _resolve_selection(names: set[str] | None) -> set[str]:
        """Normalize a requested selection: strip whitespace, drop empties."""
        if not names:
            return set()
        return {stripped for n in names if (stripped := n.strip())}

    def select(self, names: set[str] | None) -> set[str]:
        """Resolve an opt-in selection to the set of *active* registered names.

        Mirrors the proxy-extension opt-in model:

          * ``None`` or an empty set selects nothing (opt-in default).
          * ``"*"`` selects every registered compressor.
          * Otherwise only names that are both requested and registered are
            active; requested-but-unregistered names are logged and skipped.

        Discovery is never triggered here and ``compress`` is never invoked;
        this only decides which already-registered compressors are active.
        """
        requested = self._resolve_selection(names)
        registered = set(self._compressors)

        if not requested:
            if registered:
                log.info(
                    "compressors registered but none selected (opt-in): %s. "
                    "Select by name or use '*' for all.",
                    ",".join(sorted(registered)),
                )
            return set()

        if "*" in requested:
            return registered

        active = requested & registered
        missing = requested - registered
        if missing:
            log.warning(
                "compressors requested but not registered: %s (available: %s)",
                ",".join(sorted(missing)),
                ",".join(sorted(registered)) or "<none>",
            )
        return active

    def active(self, selection: set[str] | None) -> list[Compressor]:
        """Return the active compressor objects for ``selection``, sorted by name.

        ``selection`` is the raw opt-in request; it is resolved via
        :meth:`select`.
        """
        return [self._compressors[name] for name in sorted(self.select(selection))]
