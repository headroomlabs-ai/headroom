"""Pluggable provider for information-preserving compaction of tool output.

Excluded ("protected") tool results are kept out of *lossy* compression for
accuracy; the content router still applies reversible/data-preserving folds to
them via :meth:`ContentRouter._lossless_compact_excluded`. This module lets an
external extension supply that compaction instead of the built-in folds — the
same open-core pattern as the ``proxy_extension`` / ``compressor`` seams.

Contract — ``provider(content: str) -> tuple[compacted: str, kind: str] | None``:

* ``compacted`` MUST be information-preserving — byte-recoverable, or
  data-lossless for structured data (same guarantee the built-in path gives).
  Return ``None`` to leave the content unchanged.
* The provider MUST be deterministic and depend only on ``content`` (no
  cross-message state), so the proxy's prefix cache stays byte-stable across
  turns.
* The provider MUST be thread-safe. It is called from the router's parallel
  compression worker pool (``HEADROOM_COMPRESS_WORKERS``), so several blocks can
  be handed to it concurrently on different threads.
* ``kind`` is a short label that ends up in ``transforms_applied`` and in
  per-strategy metric/timing keys. Only ``^[a-z0-9_]{1,32}$`` is accepted;
  anything else is replaced with a fixed fallback label so a third-party string
  cannot blow up Prometheus label cardinality.
* A malformed return (not ``None``, not a 2-tuple of ``str``) or an empty /
  whitespace-only ``compacted`` is ignored, never fatal.

Where the provider is called:

* :meth:`ContentRouter._lossless_compact_excluded` — protected/excluded tool
  output. Here the provider is *authoritative*: when one is set the router does
  not run its built-in folds; it falls back to the built-in only if the provider
  raises.
* :meth:`ContentRouter._lossless_first` — the GENERAL compression path, i.e.
  every block of every provider/harness (including ``/v1/compress`` gateway
  traffic, where tool names are the caller's own). Here the provider *competes*
  with the built-in folds and the smaller output wins, so it can only improve on
  them. Diff content is never handed to the provider — folding a diff can break
  ``git apply``.

Verification — ``verifier(original: str, compacted: str) -> bool``:

In lossless-only mode (``--lossless``) the fold IS the final answer: nothing
downstream can recover a mistake. The built-in folds are self-verifying
(``compact_lossless`` returns its input when it cannot invert), a provider is
not. Register an optional ``verifier`` alongside the provider and the router
will run it on the provider's candidate in lossless-only mode, rejecting the
candidate when it returns falsy *or* raises. With no verifier registered the
behaviour is unchanged: the provider is trusted (documented trust).

Default is ``None`` → the router uses its built-in folds, unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

#: ``content -> (compacted, kind) | None``.
LosslessProvider = Callable[[str], "tuple[str, str] | None"]
#: ``(original, compacted) -> is-it-really-lossless``.
LosslessVerifier = Callable[[str, str], bool]

_provider: LosslessProvider | None = None
_verifier: LosslessVerifier | None = None
#: Bumped on every registration. Callers that memoize a provider's output must
#: include this in their cache key, or a provider registered (or cleared) after
#: a block was already folded would be ignored for that block forever. Normal
#: deployments register once at extension install, but tests and any hot-reload
#: path re-register at will.
_generation: int = 0


def set_lossless_provider(
    provider: LosslessProvider | None,
    *,
    verifier: LosslessVerifier | None = None,
) -> None:
    """Register (or clear, with ``None``) the lossless compaction provider.

    ``verifier`` is optional and keyword-only so existing single-argument calls
    keep working. Passing ``provider=None`` clears *both* the provider and any
    previously registered verifier — a verifier without a provider is dead
    state, and leaving one behind would silently apply to the next provider.
    """
    global _provider, _verifier, _generation
    _provider = provider
    _verifier = verifier if provider is not None else None
    _generation += 1


def get_lossless_provider() -> LosslessProvider | None:
    """Return the registered provider, or ``None`` if the built-in should run."""
    return _provider


def get_lossless_generation() -> int:
    """Registration counter — include it in any key that memoizes provider output."""
    return _generation


def get_lossless_verifier() -> LosslessVerifier | None:
    """Return the registered verifier, or ``None`` when the provider is trusted."""
    return _verifier
